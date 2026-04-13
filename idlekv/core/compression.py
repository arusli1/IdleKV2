"""
CompressedKVManager: wraps a kvpress-based compression method and integrates
the shadow buffer, query buffer, CPU KV store, and idle-time scheduler into
a single coherent interface.

This is the main entry point for using IdleKV.
"""

import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from idlekv.core.shadow_buffer import ShadowBuffer
from idlekv.core.query_buffer import QueryBuffer
from idlekv.core.phase2_refresh import FullKVStore
from idlekv.core.scheduler import IdleScheduler, RefinementResult
from idlekv.utils.kv_cache import get_layer_kv, set_layer_kv, num_layers


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
        offload_full_kv: bool = False,
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
            hidden_dim=self.hidden_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.full_kv_store = FullKVStore(offload_to_cpu=offload_full_kv)
        self.budget_per_layer = 0  # set during prefill

        # Fix Bug 3: Use list of lists to avoid O(n^2) torch.cat
        self.generated_kv_lists: list = []  # list of lists of (k, v) per layer

    def prefill(self, input_ids: torch.Tensor):
        """
        Run prefill: full forward pass, store full KV, then compress.

        Args:
            input_ids: [1, seq_len]

        Returns:
            Compressed past_key_values (DynamicCache or tuple format)
        """
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True, output_hidden_states=True)

        full_kv = outputs.past_key_values
        seq_len = input_ids.shape[1]

        # Store full KV (CPU or GPU based on config)
        self.full_kv_store.store(full_kv)

        # Compute budget
        self.budget_per_layer = int(seq_len * (1 - self.compression_ratio))

        # Compress: keep top-k tokens per layer using SnapKV-style scoring
        compressed_kv = self._compress(full_kv, input_ids, self.budget_per_layer)

        # Reset buffers for new session
        self.shadow_buffer.clear()
        self.query_buffer.clear()
        # Initialize list of lists for generated KV (avoids O(n^2) concatenation)
        self.generated_kv_lists = [[] for _ in range(self.num_layers)]

        return compressed_kv

    def _compress(
        self,
        full_kv,
        input_ids: torch.Tensor,
        budget: int,
    ):
        """
        SnapKV-style compression: use last-window queries to score all keys,
        keep top-k, push evicted to shadow buffer.

        This is a simplified reimplementation (~50 lines) for full control.
        kvpress is used only for baselines.
        """
        import math
        from idlekv.utils.kv_cache import build_cache

        compressed = []
        window_size = 32  # SnapKV observation window
        layers_count = num_layers(full_kv)

        for layer_idx in range(layers_count):
            k, v = get_layer_kv(full_kv, layer_idx)  # [1, H, S, D]
            S = k.shape[2]

            if S <= budget:
                compressed.append((k, v))
                continue

            # Use last `window_size` queries to score all keys
            # Get Q projection for this layer
            layer = self.model.model.layers[layer_idx]
            q_proj = layer.self_attn.q_proj

            # Get hidden states for the observation window
            # (Simplified: use the last window_size keys as proxy for queries)
            # In full implementation, would recompute hidden states
            window_k = k[:, :, -window_size:, :]  # [1, H, W, D]

            # Attention scores: window queries x all keys
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(window_k, k.transpose(-2, -1)) * scale  # [1, H, W, S]
            scores = torch.softmax(scores, dim=-1)
            importance = scores.mean(dim=2).squeeze(0)  # [H, S]

            # Mean across heads for token selection
            token_importance = importance.mean(dim=0)  # [S]

            # Always keep attention sinks (first 4 tokens) and recent window
            sink_size = 4
            keep_indices = set(range(sink_size))
            keep_indices.update(range(S - window_size, S))

            # Select top-k from remaining
            remaining_budget = budget - len(keep_indices)
            if remaining_budget > 0:
                middle_mask = torch.ones(S, dtype=torch.bool, device=k.device)
                for idx in keep_indices:
                    middle_mask[idx] = False
                middle_scores = token_importance.clone()
                middle_scores[~middle_mask] = float('-inf')
                _, top_middle = middle_scores.topk(min(remaining_budget, middle_mask.sum()))
                keep_indices.update(top_middle.tolist())

            keep_indices = sorted(keep_indices)[:budget]
            keep_tensor = torch.tensor(keep_indices, device=k.device)

            # Build compressed cache
            compressed_k = k[:, :, keep_tensor, :]
            compressed_v = v[:, :, keep_tensor, :]

            # Push evicted to shadow buffer
            all_indices = set(range(S))
            evicted_indices = sorted(all_indices - set(keep_indices))
            if evicted_indices:
                evict_tensor = torch.tensor(evicted_indices, device=k.device)
                evicted_k = k[0, :, evict_tensor, :]  # [H, num_evicted, D]
                evicted_v = v[0, :, evict_tensor, :]
                self.shadow_buffer.push(layer_idx, evicted_k, evicted_v)

            compressed.append((compressed_k, compressed_v))

        # Return cache in same format as input
        return build_cache(compressed)

    def on_token_generated(self, hidden_state: torch.Tensor, new_kv_per_layer: list):
        """
        Call after each generated token to update query buffer and track generated KV.

        Args:
            hidden_state: [1, hidden_dim] last-layer hidden state
            new_kv_per_layer: list of (k, v) for the new token, one per layer
                              k shape: [1, H, 1, D]
        """
        self.query_buffer.append(hidden_state.squeeze(0))

        # Fix Bug 3: Store in list to avoid O(n^2) torch.cat every step
        for layer_idx, (k, v) in enumerate(new_kv_per_layer):
            self.generated_kv_lists[layer_idx].append((k, v))

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
        )
        return scheduler.run(
            past_key_values=past_key_values,
            generated_kv=self._get_generated_kv(),
            max_time_ms=max_time_ms,
        )

    def on_tool_result_prefill(self, full_kv):
        """
        Update full KV store after processing tool results.
        Called after the model prefills tool-result tokens.
        """
        self.full_kv_store.store(full_kv)
        # Reset generated KV tracking
        self.generated_kv_lists = [[] for _ in range(self.num_layers)]
