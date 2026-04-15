"""
CompressedKVManager: wraps a kvpress-based compression method and integrates
the shadow buffer, query buffer, CPU KV store, and idle-time scheduler into
a single coherent interface.

This is the main entry point for using IdleKV.
"""

import math
import torch
from typing import Optional, Union
from transformers import AutoModelForCausalLM
import torch.nn.functional as F

from idlekv.core.anytime_repair import (
    compute_importance_scores_gqa,
    project_queries_all_heads,
)
from idlekv.core.shadow_buffer import ShadowBuffer
from idlekv.core.query_buffer import QueryBuffer
from idlekv.core.phase2_refresh import FullKVStore
from idlekv.core.scheduler import IdleScheduler, RefinementResult
from idlekv.utils.kv_cache import get_layer_kv, set_layer_kv, num_layers
from idlekv.utils.generation import last_token_logits_kwargs


class CompressedKVManager:
    """
    Manages the full IdleKV pipeline: compression, shadow buffering,
    query buffering, CPU backup, and idle-time refinement.

    Usage:
        model = AutoModelForCausalLM.from_pretrained(...)
        manager = CompressedKVManager(model, compression_ratio=0.5, shadow_size=256)

        # Prefill
        past_kv = manager.prefill(input_ids)

        # Generate tokens
        for step in range(num_steps):
            output, past_kv = manager.generate_step(past_kv, ...)
            if is_tool_call(output):
                result = manager.idle_refine(past_kv, max_time_ms=1000)
                past_kv = result.past_key_values
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        compression_ratio: float = 0.5,
        shadow_size: int = 256,
        query_buffer_size: int = 32,
        offload_full_kv: bool = True,
        default_refinement_policy: Optional[str] = None,
        anytime_min_idle_ms: float = 20.0,
        anytime_shadow_only_max_ms: float = 80.0,
        sample_span_size: int = 16,
        sample_spans_per_layer: int = 2,
        sample_sampler_seed: int = 0,
    ):
        self.model = model
        self.compression_ratio = compression_ratio
        config = model.config

        # Fix Bug 4: Enable hidden states output
        model.config.output_hidden_states = True

        # Model architecture params
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_dim = config.hidden_size
        param = next(model.parameters())
        self.device = param.device
        self.dtype = param.dtype
        self.offload_full_kv = offload_full_kv
        self.default_refinement_policy = default_refinement_policy

        # Initialize components
        self.shadow_buffer = ShadowBuffer(
            num_layers=self.num_layers,
            max_size=shadow_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.query_buffer = QueryBuffer(
            buffer_size=query_buffer_size,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.full_kv_store = FullKVStore(offload_to_cpu=offload_full_kv)
        self.budget_per_layer = 0  # set during prefill

        # Fix Bug 3: Use list of lists to avoid O(n^2) torch.cat
        self.generated_kv_lists: list = []  # list of lists of (k, v) per layer
        self.prefill_query_seed: Optional[torch.Tensor] = None
        self.retained_positions: list[torch.Tensor] = []
        self.prefill_importance_scores: list[torch.Tensor] = []
        self.prefill_seq_len: int = 0

        # Semantic sequence length = prefill_len + tokens generated since last
        # prefill. The compressed cache's physical length differs from this,
        # but RoPE-baked keys retain their original positions, so the next
        # query must use its true absolute position (= semantic_seq_len) to
        # keep positional phase coherent.
        self.semantic_seq_len: int = 0
        self.compression_window_size: int = 32
        # Allow a small decode-time slack so we do not rebuild the physical KV
        # on every generated token once the cache is near its budget.
        self.online_eviction_slack: int = 16
        self.anytime_min_idle_ms = anytime_min_idle_ms
        self.anytime_shadow_only_max_ms = anytime_shadow_only_max_ms
        self.sample_span_size = sample_span_size
        self.sample_spans_per_layer = sample_spans_per_layer
        self.sample_sampler_seed = sample_sampler_seed

    def _run_prefill_with_query_windows(self, input_ids: torch.Tensor):
        """
        Run prefill while capturing only the tail query window per layer.

        SnapKV-style compression needs the hidden states feeding each layer for
        the last W tokens, not the full `[num_layers, seq_len, hidden_dim]`
        stack. Capturing just the tail window keeps 8K prefill viable on an
        A10G-class GPU.
        """
        captured_inputs = [None] * self.num_layers
        window_size = min(self.compression_window_size, input_ids.shape[1])
        hooks = []

        def make_hook(layer_idx):
            def hook(_, args):
                hidden_states = args[0]
                captured_inputs[layer_idx] = (
                    hidden_states[:, -window_size:, :].detach().clone()
                )
            return hook

        for layer_idx, layer in enumerate(self.model.model.layers):
            hooks.append(layer.register_forward_pre_hook(make_hook(layer_idx)))

        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    use_cache=True,
                    output_hidden_states=False,
                    **last_token_logits_kwargs(self.model),
                )
        finally:
            for hook in hooks:
                hook.remove()

        if any(window is None for window in captured_inputs) and getattr(outputs, "hidden_states", None) is not None:
            for layer_idx in range(min(self.num_layers, len(outputs.hidden_states))):
                if captured_inputs[layer_idx] is None:
                    captured_inputs[layer_idx] = (
                        outputs.hidden_states[layer_idx][:, -window_size:, :].detach().clone()
                    )

        if any(window is None for window in captured_inputs):
            missing = [idx for idx, window in enumerate(captured_inputs) if window is None]
            raise RuntimeError(f"Failed to capture prefill query windows for layers: {missing}")

        return outputs, captured_inputs

    def prefill(self, input_ids: torch.Tensor):
        """
        Run prefill: full forward pass, store full KV, then compress.

        Args:
            input_ids: [1, seq_len]

        Returns:
            Compressed past_key_values (DynamicCache or tuple format)
        """
        outputs, query_windows = self._run_prefill_with_query_windows(input_ids)

        full_kv = outputs.past_key_values
        seq_len = input_ids.shape[1]

        # SnapKV only needs the tail query window feeding each layer.
        self._prefill_hidden_states = query_windows

        # Store full KV (CPU or GPU based on config)
        self.full_kv_store.store(full_kv)

        # Reset buffers for new session
        self.shadow_buffer.clear()
        self.query_buffer.clear()
        seed_len = min(self.query_buffer.buffer_size, seq_len)
        self.prefill_query_seed = torch.stack([
            query_windows[layer_idx][0, -seed_len:, :].detach().clone()
            for layer_idx in range(self.num_layers)
        ], dim=1)
        # Initialize list of lists for generated KV (avoids O(n^2) concatenation)
        self.generated_kv_lists = [[] for _ in range(self.num_layers)]
        self.retained_positions = []
        self.prefill_importance_scores = []
        self.prefill_seq_len = seq_len

        # Compute budget
        self.budget_per_layer = int(seq_len * (1 - self.compression_ratio))
        self.semantic_seq_len = seq_len

        # Cache last-position logits so callers don't have to re-run forward
        # just to get the first post-prefill token's logits.
        self.last_prefill_logits = outputs.logits[:, -1, :].clone()

        # Compress: keep top-k tokens per layer using SnapKV-style scoring.
        # Use try/finally so a failure in _compress still releases the ~GB of
        # hidden-state memory we stashed on self.
        try:
            compressed_kv = self._compress(full_kv, input_ids, self.budget_per_layer)
        finally:
            self._prefill_hidden_states = None

        return compressed_kv

    def _compress(
        self,
        full_kv,
        input_ids: torch.Tensor,
        budget: int,
    ):
        """
        Real SnapKV compression.

        For each layer:
          1) Take the last W hidden states feeding this layer.
          2) Project through q_proj → per-Q-head query vectors.
             RoPE is NOT applied to the W queries because we don't have a
             trivial way to grab the exact cos/sin slab for their positions
             across HF model families; keys retain original RoPE, and
             dropping query RoPE here is an approximation that preserves
             ranking much better than Q-averaging or keys-as-queries.
          3) Score: softmax(QK^T / sqrt(D)) per Q-head, mean across queries,
             mean across GQA group → per-KV-head score over all S keys.
          4) 1D avg-pool (kernel=5) to preserve contiguous important regions.
          5) Always keep the last W tokens (observation window) and sink
             tokens (first 4). Top-k from the remainder. Evicted → shadow.
        """
        from idlekv.utils.kv_cache import build_cache

        compressed = []
        window_size = 32
        sink_size = 4
        pool_kernel = 5
        layers_count = num_layers(full_kv)

        num_q_heads = self.model.config.num_attention_heads
        group_size = num_q_heads // self.num_kv_heads
        hidden_states_all = getattr(self, '_prefill_hidden_states', None)

        for layer_idx in range(layers_count):
            k, v = get_layer_kv(full_kv, layer_idx)  # [1, H, S, D]
            S = k.shape[2]

            if S <= budget:
                compressed.append((k, v))
                self.retained_positions.append(torch.arange(S, dtype=torch.long))
                self.prefill_importance_scores.append(torch.ones(S, dtype=torch.float32))
                continue

            # --- (1) window hidden states feeding this layer -----------------
            if hidden_states_all is not None:
                h_layer = hidden_states_all[layer_idx]  # [1, W, hidden_dim]
                window_h = h_layer[0]  # [W, hidden_dim]
            else:
                # Fallback: no hidden states captured (shouldn't happen under
                # normal prefill). Skip scoring, keep last-budget tokens.
                keep = torch.arange(S - budget, S, device=k.device)
                compressed.append(
                    (k[:, :, keep, :], v[:, :, keep, :])
                )
                self.retained_positions.append(keep.detach().cpu())
                self.prefill_importance_scores.append(torch.ones(S, dtype=torch.float32))
                continue

            # --- (2) project to queries --------------------------------------
            q_proj = self.model.model.layers[layer_idx].self_attn.q_proj
            q = project_queries_all_heads(window_h, q_proj, self.head_dim)

            # --- (3) per-Q-head attention, then group mean ------------------
            score_kv = compute_importance_scores_gqa(
                q,
                k[0],
                self.head_dim,
                num_q_heads,
                self.num_kv_heads,
            )

            # --- (4) 1D avg-pool smoothing ----------------------------------
            # Protects clusters of adjacent important tokens from fragmentation.
            score_smoothed = F.avg_pool1d(
                score_kv.unsqueeze(0),  # [1, H_kv, S]
                kernel_size=pool_kernel,
                padding=pool_kernel // 2,
                stride=1,
            ).squeeze(0)  # [H_kv, S]
            token_importance = score_smoothed.mean(dim=0)  # [S] — mean across KV heads

            # --- (5) keep sinks + recent window + top-k from middle ---------
            force_keep = torch.zeros(S, dtype=torch.bool, device=k.device)
            force_keep[:sink_size] = True
            force_keep[S - window_size:] = True
            keep_count = int(force_keep.sum().item())
            remaining_budget = max(0, budget - keep_count)

            middle_scores = token_importance.clone()
            middle_scores[force_keep] = float('-inf')
            _, top_middle = middle_scores.topk(
                min(remaining_budget, S - keep_count)
            )
            keep_mask = force_keep.clone()
            keep_mask[top_middle] = True
            keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1).sort().values
            # Truncate if we overshot the budget (rare — happens if sinks+window > budget)
            if keep_indices.numel() > budget:
                keep_indices = keep_indices[:budget]

            compressed_k = k[:, :, keep_indices, :]
            compressed_v = v[:, :, keep_indices, :]

            evicted_mask = torch.ones(S, dtype=torch.bool, device=k.device)
            evicted_mask[keep_indices] = False
            evicted_indices = torch.nonzero(evicted_mask, as_tuple=False).squeeze(1)
            if evicted_indices.numel() > 0:
                evicted_k = k[0, :, evicted_indices, :]  # [H, num_evicted, D]
                evicted_v = v[0, :, evicted_indices, :]
                self.shadow_buffer.push(
                    layer_idx,
                    evicted_k,
                    evicted_v,
                    positions=evicted_indices,
                )

            compressed.append((compressed_k, compressed_v))
            self.retained_positions.append(keep_indices.detach().cpu())
            self.prefill_importance_scores.append(token_importance.detach().cpu().float())

        # Hidden states no longer needed after compression
        self._prefill_hidden_states = None

        return build_cache(compressed)

    def build_query_state(self, hidden_states) -> Optional[torch.Tensor]:
        """
        Extract the per-layer hidden states that feed each transformer layer.

        HF returns `hidden_states[0]` as the embedding output and
        `hidden_states[layer_idx]` as the input to transformer layer
        `layer_idx`. Phase 1 and Phase 2 must score each layer against its own
        recent query states, so we keep the first `num_layers` entries.
        """
        if hidden_states is None:
            return None

        return torch.stack([
            hidden_states[layer_idx][:, -1, :].squeeze(0).detach()
            for layer_idx in range(self.num_layers)
        ], dim=0)

    def on_token_generated(self, hidden_states_by_layer: torch.Tensor, new_kv_per_layer: list):
        """
        Call after each generated token to update query buffer and track generated KV.

        Args:
            hidden_states_by_layer: [num_layers, hidden_dim] hidden state that
                feeds each layer for the newly ingested/generated token
            new_kv_per_layer: list of (k, v) for the new token, one per layer
                              k shape: [1, H, 1, D]
        """
        self.query_buffer.append(hidden_states_by_layer)

        # Fix Bug 3: Store in list to avoid O(n^2) torch.cat every step
        for layer_idx, (k, v) in enumerate(new_kv_per_layer):
            self.generated_kv_lists[layer_idx].append((k, v))
        self.semantic_seq_len += 1

    def maybe_evict_online(
        self,
        past_key_values,
        sink_size: int = 4,
        slack_tokens: Optional[int] = None,
    ):
        """
        Streaming-LLM-style online eviction.

        If the physical cache for any layer exceeds
        `budget_per_layer + slack_tokens`, evict the oldest middle tokens
        (keep first `sink_size` + most recent `budget_per_layer - sink_size`)
        and push the evicted pairs into the shadow buffer. The small slack
        amortizes copy cost during long decode runs and reduces allocator churn
        on memory-constrained GPUs.

        This is a cheap O(1)-per-step policy; Phase 1 later uses TIR to
        promote important tokens back out of the shadow buffer.
        """
        from idlekv.utils.kv_cache import set_layer_kv
        if self.budget_per_layer <= 0:
            return past_key_values
        slack = self.online_eviction_slack if slack_tokens is None else max(slack_tokens, 0)
        for layer_idx in range(self.num_layers):
            k, v = get_layer_kv(past_key_values, layer_idx)  # [1, H, S, D]
            S = k.shape[2]
            if S <= self.budget_per_layer + slack:
                continue
            retained_positions = self.retained_positions[layer_idx]
            current_prefill_len = int(retained_positions.numel())
            if current_prefill_len <= sink_size:
                continue

            overflow = S - self.budget_per_layer
            evict_count = min(overflow, current_prefill_len - sink_size)
            if evict_count <= 0:
                continue

            evict_start = sink_size
            evict_end = sink_size + evict_count
            evicted_k = k[0, :, evict_start:evict_end, :]
            evicted_v = v[0, :, evict_start:evict_end, :]
            evicted_positions = retained_positions[evict_start:evict_end]
            self.shadow_buffer.push(
                layer_idx,
                evicted_k,
                evicted_v,
                positions=evicted_positions,
            )

            keep_prefill_k = torch.cat(
                [k[:, :, :sink_size, :], k[:, :, evict_end:current_prefill_len, :]],
                dim=2,
            )
            keep_prefill_v = torch.cat(
                [v[:, :, :sink_size, :], v[:, :, evict_end:current_prefill_len, :]],
                dim=2,
            )
            generated_k = k[:, :, current_prefill_len:, :]
            generated_v = v[:, :, current_prefill_len:, :]
            keep_k = torch.cat([keep_prefill_k, generated_k], dim=2)
            keep_v = torch.cat([keep_prefill_v, generated_v], dim=2)
            past_key_values = set_layer_kv(past_key_values, layer_idx, keep_k, keep_v)
            self.retained_positions[layer_idx] = torch.cat([
                retained_positions[:sink_size],
                retained_positions[evict_end:],
            ], dim=0)
        return past_key_values

    def next_position_ids(self, num_new_tokens: int = 1) -> torch.Tensor:
        """
        Position ids for the next `num_new_tokens` to be appended.

        Call this before each forward(..., position_ids=...) so the new
        query uses its true absolute position rather than
        `cache.get_seq_length()`, which is incorrect after compression.
        """
        start = self.semantic_seq_len
        return torch.arange(
            start, start + num_new_tokens, device=self.device
        ).unsqueeze(0)

    def seed_query_buffer_from_prefill(self) -> int:
        """
        Seed the query buffer from the tail of the prefill prompt.

        This is useful for short validation scripts that need question-
        conditioned queries before any generation has happened.

        Returns:
            Number of hidden states written into the query buffer.
        """
        self.query_buffer.clear()
        if self.prefill_query_seed is None:
            return 0

        for hidden_states_by_layer in self.prefill_query_seed:
            self.query_buffer.append(hidden_states_by_layer)
        return int(self.prefill_query_seed.shape[0])

    def _get_generated_kv(self) -> list:
        """
        Concatenate generated KV lists into tensors when needed.
        Only called during refinement, not every token.
        """
        generated_kv = []
        for layer_idx in range(self.num_layers):
            if not self.generated_kv_lists[layer_idx]:
                # No generated tokens for this layer
                generated_kv.append((
                    torch.empty(1, self.num_kv_heads, 0, self.head_dim,
                                device=self.device, dtype=self.dtype),
                    torch.empty(1, self.num_kv_heads, 0, self.head_dim,
                                device=self.device, dtype=self.dtype),
                ))
            else:
                # Concatenate all generated tokens for this layer
                k_list, v_list = zip(*self.generated_kv_lists[layer_idx])
                k_cat = torch.cat(k_list, dim=2)  # [1, H, num_generated, D]
                v_cat = torch.cat(v_list, dim=2)
                generated_kv.append((k_cat, v_cat))
        return generated_kv

    def idle_refine(
        self,
        past_key_values,
        max_time_ms: float = 1000,
        phases: Union[str, int, None] = "1+2",
        policy: Optional[str] = None,
    ) -> RefinementResult:
        """
        Run idle-time refinement (call during tool-call pauses).

        Args:
            past_key_values: current compressed cache
            max_time_ms: time budget for refinement

        Returns:
            RefinementResult with updated cache
        """
        scheduler = IdleScheduler(
            shadow_buffer=self.shadow_buffer,
            query_buffer=self.query_buffer,
            full_kv_store=self.full_kv_store,
            model=self.model,
            budget_per_layer=self.budget_per_layer,
            num_layers=self.num_layers,
            retained_positions=self.retained_positions,
            prefill_importance_scores=self.prefill_importance_scores,
            anytime_min_idle_ms=self.anytime_min_idle_ms,
            anytime_shadow_only_max_ms=self.anytime_shadow_only_max_ms,
            sample_span_size=self.sample_span_size,
            sample_spans_per_layer=self.sample_spans_per_layer,
            sample_sampler_seed=self.sample_sampler_seed,
        )
        # num_generated = number of tokens appended since last prefill.
        # Phase 1 uses this to preserve the generated tail instead of evicting
        # committed model outputs.
        num_gen = len(self.generated_kv_lists[0]) if self.generated_kv_lists else 0
        result = scheduler.run(
            past_key_values=past_key_values,
            generated_kv=self._get_generated_kv(),
            max_time_ms=max_time_ms,
            num_generated=num_gen,
            phases=phases,
            policy=policy or self.default_refinement_policy,
        )
        if result.retained_positions is not None:
            self.retained_positions = [pos.detach().cpu() for pos in result.retained_positions]
        return result

    def on_tool_result_prefill(self, full_kv, new_seq_len: int):
        """
        Update full KV store after processing tool results.
        Called after the model prefills tool-result tokens.

        Args:
            full_kv: the full (pre-compression) past_key_values after the
                tool-result prefill.
            new_seq_len: the new semantic sequence length (prefill + tool result).
        """
        self.full_kv_store.store(full_kv)
        self.shadow_buffer.clear()
        self.generated_kv_lists = [[] for _ in range(self.num_layers)]
        self.semantic_seq_len = new_seq_len
        self.prefill_seq_len = new_seq_len
        self.retained_positions = [
            torch.arange(new_seq_len, dtype=torch.long)
            for _ in range(self.num_layers)
        ]
        self.prefill_importance_scores = [
            torch.ones(new_seq_len, dtype=torch.float32)
            for _ in range(self.num_layers)
        ]
