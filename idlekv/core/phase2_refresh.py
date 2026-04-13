"""
Phase 2: Progressive full-attention cache refresh during idle time.

Loads the full prefill KV from CPU layer-by-layer, computes full attention
scores with recent queries, and re-selects top-k tokens for the compressed
cache. Processes shallow layers first (they propagate errors most severely).

Cost: ~15-40ms per layer on RTX 6000 / L40S. All 32 layers: ~500ms-1.3s.
Anytime: each layer is independent; interruption yields a valid partial state.
"""

import torch
import math
from typing import Optional

from idlekv.core.query_buffer import QueryBuffer


def phase2_refresh(
    past_key_values: tuple,
    cpu_full_kv: list,
    generated_kv: list,
    query_buffer: QueryBuffer,
    model,
    budget_per_layer: int,
    num_layers: int,
    interrupt_flag: Optional[callable] = None,
    max_layers: Optional[int] = None,
) -> tuple:
    """
    Progressive full-attention refresh using CPU-stored prefill KV.

    For each layer (shallow-first), loads the full prefill KV from CPU,
    concatenates with generated tokens' KV, computes importance scores
    using recent queries, and re-selects top-k for the compressed cache.

    Args:
        past_key_values: current compressed KV cache
        cpu_full_kv: list of (K, V) per layer stored on CPU from last prefill.
                     K shape: [1, H, S_prefill, D] on CPU
        generated_kv: list of (K, V) per layer for tokens generated since
                      last prefill. These are always retained (not evicted).
                      K shape: [1, H, S_gen, D] on GPU
        query_buffer: recent hidden states for scoring
        model: HF model (for Q projections)
        budget_per_layer: target compressed cache size (excluding generated tokens)
        num_layers: number of layers
        interrupt_flag: callable returning True if tool has returned
        max_layers: process at most this many layers (for partial refresh)

    Returns:
        Updated past_key_values tuple
    """
    recent_h = query_buffer.get()
    if recent_h.shape[0] == 0:
        return past_key_values

    new_kv = list(past_key_values)
    layers_to_process = min(num_layers, max_layers or num_layers)

    for layer_idx in range(layers_to_process):
        if interrupt_flag is not None and interrupt_flag():
            break

        # Load full prefill KV for this layer from CPU to GPU
        full_k_cpu, full_v_cpu = cpu_full_kv[layer_idx]
        full_k = full_k_cpu.to(recent_h.device, non_blocking=True)  # [1, H, S_prefill, D]
        full_v = full_v_cpu.to(recent_h.device, non_blocking=True)
        torch.cuda.synchronize()  # ensure transfer complete

        B, H, S_prefill, D = full_k.shape

        # Concatenate with generated tokens' KV (these are always kept)
        gen_k, gen_v = generated_kv[layer_idx] if generated_kv else (None, None)
        if gen_k is not None and gen_k.shape[2] > 0:
            S_gen = gen_k.shape[2]
            all_k = torch.cat([full_k, gen_k], dim=2)  # [1, H, S_prefill + S_gen, D]
            all_v = torch.cat([full_v, gen_v], dim=2)
        else:
            S_gen = 0
            all_k = full_k
            all_v = full_v

        S_total = all_k.shape[2]

        # Project recent hidden states to queries for this layer
        from idlekv.core.phase1_rescore import _get_q_proj, _project_queries
        q_proj = _get_q_proj(model, layer_idx)
        queries = _project_queries(recent_h, q_proj, H, D)  # [H, Q, D]

        # Compute importance scores for ALL tokens (prefill + generated)
        scale = 1.0 / math.sqrt(D)
        all_k_squeezed = all_k.squeeze(0)  # [H, S_total, D]
        all_v_squeezed = all_v.squeeze(0)
        attn = torch.bmm(queries, all_k_squeezed.transpose(1, 2)) * scale
        attn_weights = torch.softmax(attn, dim=-1)
        importance = attn_weights.mean(dim=1).mean(dim=0)  # [S_total]

        # Always keep generated tokens (they are current, not stale)
        # Select top-(budget) from prefill tokens
        prefill_importance = importance[:S_prefill]
        keep_budget = min(budget_per_layer, S_prefill)
        _, top_prefill_indices = prefill_importance.topk(keep_budget)
        top_prefill_indices = top_prefill_indices.sort().values

        # Build new cache: selected prefill tokens + all generated tokens
        selected_k = all_k_squeezed[:, top_prefill_indices, :]  # [H, keep, D]
        selected_v = all_v_squeezed[:, top_prefill_indices, :]

        if S_gen > 0:
            gen_k_squeezed = all_k_squeezed[:, S_prefill:, :]
            gen_v_squeezed = all_v_squeezed[:, S_prefill:, :]
            new_k = torch.cat([selected_k, gen_k_squeezed], dim=1).unsqueeze(0)
            new_v = torch.cat([selected_v, gen_v_squeezed], dim=1).unsqueeze(0)
        else:
            new_k = selected_k.unsqueeze(0)
            new_v = selected_v.unsqueeze(0)

        new_kv[layer_idx] = (new_k, new_v)

        # Free GPU memory from the loaded full KV
        del full_k, full_v

    return tuple(new_kv)


class CPUKVStore:
    """
    Stores full uncompressed KV cache on CPU memory.

    Created during prefill. Updated after each tool-result prefill.
    Loaded layer-by-layer to GPU during Phase 2.
    """

    def __init__(self):
        self.layers: list[tuple[torch.Tensor, torch.Tensor]] = []

    @torch.no_grad()
    def store(self, past_key_values: tuple):
        """Store full KV cache from GPU to CPU."""
        self.layers = []
        for k, v in past_key_values:
            self.layers.append((
                k.to("cpu", non_blocking=True),
                v.to("cpu", non_blocking=True),
            ))
        torch.cuda.synchronize()

    def get_layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get full KV for a single layer (still on CPU)."""
        return self.layers[layer_idx]

    def __len__(self):
        return len(self.layers)

    @property
    def memory_bytes(self) -> int:
        total = 0
        for k, v in self.layers:
            total += k.nelement() * k.element_size()
            total += v.nelement() * v.element_size()
        return total
