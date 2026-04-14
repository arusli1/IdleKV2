"""
Phase 2: Progressive full-attention cache refresh during idle time.

Loads the full prefill KV from CPU layer-by-layer, computes full attention
scores with recent queries, and re-selects top-k tokens for the compressed
cache. Processes shallow layers first (they propagate errors most severely).

Cost is hardware-dependent; expect tens of milliseconds per layer, with a
longer idle window needed to refresh all layers. Anytime: each layer is
independent; interruption yields a valid partial state.
"""

import torch
import math
from typing import Optional

from idlekv.core.query_buffer import QueryBuffer
from idlekv.utils.kv_cache import get_layer_kv, num_layers


def phase2_refresh(
    past_key_values,
    full_kv_store: 'FullKVStore',
    generated_kv: list,
    query_buffer: QueryBuffer,
    model,
    budget_per_layer: int,
    interrupt_flag: Optional[callable] = None,
    max_layers: Optional[int] = None,
):
    """
    Progressive full-attention refresh using stored prefill KV.

    For each layer (shallow-first), loads the full prefill KV,
    concatenates with generated tokens' KV, computes importance scores
    using recent queries, and re-selects top-k for the compressed cache.

    Args:
        past_key_values: current compressed KV cache
        full_kv_store: FullKVStore containing prefill KV (CPU or GPU)
        generated_kv: list of (K, V) per layer for tokens generated since
                      last prefill. These are always retained (not evicted).
                      K shape: [1, H, S_gen, D]
        query_buffer: recent hidden states for scoring
        model: HF model (for Q projections)
        budget_per_layer: target compressed cache size (excluding generated tokens)
        interrupt_flag: callable returning True if tool has returned
        max_layers: process at most this many layers (for partial refresh)

    Returns:
        Tuple of (updated past_key_values, layers_refreshed)
    """
    if query_buffer.count == 0:
        return past_key_values, 0

    from idlekv.utils.kv_cache import build_cache, clone_cache

    compressed_layers = []
    total_layers = num_layers(past_key_values)
    layers_to_process = min(total_layers, max_layers or total_layers)
    layers_refreshed = 0

    for layer_idx in range(layers_to_process):
        if interrupt_flag is not None and interrupt_flag():
            break

        # Get current compressed layer
        current_k, current_v = get_layer_kv(past_key_values, layer_idx)

        # Load full prefill KV for this layer (CPU or GPU)
        full_k, full_v = full_kv_store.get_layer(layer_idx)
        recent_h = query_buffer.get(layer_idx=layer_idx)

        # Move to device if needed (CPU->GPU transfer or no-op if already on GPU)
        if full_k.device != recent_h.device:
            full_k = full_k.to(recent_h.device, non_blocking=True)  # [1, H, S_prefill, D]
            full_v = full_v.to(recent_h.device, non_blocking=True)
            if recent_h.device.type == 'cuda':
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

        compressed_layers.append((new_k, new_v))
        layers_refreshed += 1

        # Free GPU memory from the loaded full KV if it was transferred
        if full_k.device != full_kv_store.device:
            del full_k, full_v

    # For unprocessed layers, keep the current compressed cache
    for layer_idx in range(layers_to_process, total_layers):
        current_k, current_v = get_layer_kv(past_key_values, layer_idx)
        compressed_layers.append((current_k, current_v))

    # Return in same format as input
    return build_cache(compressed_layers), layers_refreshed


class FullKVStore:
    """
    Stores full uncompressed KV cache in memory.

    On A10G-class 24GB GPUs, keep the full backup on CPU by default.
    Larger-memory GPUs can optionally keep it on-device.

    Created during prefill. Updated after each tool-result prefill.
    Used during Phase 2 for full-attention refresh.
    """

    def __init__(self, offload_to_cpu: bool = False):
        self.offload_to_cpu = offload_to_cpu
        self.layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.device = None

    @torch.no_grad()
    def store(self, past_key_values):
        """Store full KV cache, optionally moving to CPU."""
        from idlekv.utils.kv_cache import get_layer_kv, num_layers

        self.layers = []
        total_layers = num_layers(past_key_values)

        for layer_idx in range(total_layers):
            k, v = get_layer_kv(past_key_values, layer_idx)

            if self.offload_to_cpu:
                # Move to CPU to save GPU memory
                self.layers.append((
                    k.to("cpu", non_blocking=True),
                    v.to("cpu", non_blocking=True),
                ))
                self.device = "cpu"
            else:
                # Keep on the current device when GPU memory allows it.
                self.layers.append((k, v))
                self.device = k.device

        if self.offload_to_cpu and k.device.type == 'cuda':
            torch.cuda.synchronize()

    def get_layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get full KV for a single layer."""
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


# Legacy alias for backward compatibility
CPUKVStore = FullKVStore
