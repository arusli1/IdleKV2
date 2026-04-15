"""
Phase 2: progressive full-attention cache refresh during idle time.

This remains the explicit full-refresh path. It now also returns updated
retained prefill positions so the manager's metadata stays consistent.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from idlekv.core.anytime_repair import (
    _get_q_proj,
    compute_importance_scores,
    project_queries_grouped,
)
from idlekv.core.query_buffer import QueryBuffer
from idlekv.utils.kv_cache import build_cache, get_layer_kv, num_layers


def _infer_retained_positions(
    past_key_values,
    total_layers: int,
    generated_kv: list,
) -> list[torch.Tensor]:
    inferred = []
    for layer_idx in range(total_layers):
        current_k, _ = get_layer_kv(past_key_values, layer_idx)
        generated_len = 0
        if generated_kv and layer_idx < len(generated_kv):
            generated_len = generated_kv[layer_idx][0].shape[2]
        prefill_len = max(0, current_k.shape[2] - generated_len)
        inferred.append(torch.arange(prefill_len, dtype=torch.long))
    return inferred


def phase2_refresh(
    past_key_values,
    full_kv_store: "FullKVStore",
    generated_kv: list,
    query_buffer: QueryBuffer,
    model,
    budget_per_layer: int,
    interrupt_flag: Optional[callable] = None,
    max_layers: Optional[int] = None,
    retained_positions: Optional[list[torch.Tensor]] = None,
):
    """
    Progressive full-attention refresh using the stored prefill KV.

    Returns:
        Tuple of (updated past_key_values, updated_retained_positions, layers_refreshed)
    """
    total_layers = num_layers(past_key_values)
    retained_positions = retained_positions or _infer_retained_positions(
        past_key_values,
        total_layers,
        generated_kv,
    )

    if query_buffer.count == 0:
        return past_key_values, [pos.clone() for pos in retained_positions], 0

    compressed_layers = []
    updated_positions: list[torch.Tensor] = []
    layers_to_process = min(total_layers, max_layers or total_layers)
    layers_refreshed = 0

    for layer_idx in range(layers_to_process):
        if interrupt_flag is not None and interrupt_flag():
            break

        full_k, full_v = full_kv_store.get_layer(layer_idx)
        recent_h = query_buffer.get(layer_idx=layer_idx)
        if full_k.device != recent_h.device:
            full_k = full_k.to(recent_h.device, non_blocking=True)
            full_v = full_v.to(recent_h.device, non_blocking=True)
            if recent_h.device.type == "cuda":
                torch.cuda.synchronize()

        _, num_kv_heads, prefill_len, head_dim = full_k.shape
        gen_k, gen_v = generated_kv[layer_idx] if generated_kv else (None, None)
        if gen_k is not None and gen_k.shape[2] > 0:
            all_k = torch.cat([full_k, gen_k], dim=2)
            all_v = torch.cat([full_v, gen_v], dim=2)
            num_generated = gen_k.shape[2]
        else:
            all_k = full_k
            all_v = full_v
            num_generated = 0

        q_proj = _get_q_proj(model, layer_idx)
        queries = project_queries_grouped(recent_h, q_proj, num_kv_heads, head_dim)
        importance = compute_importance_scores(
            queries,
            all_k.squeeze(0),
            head_dim,
        ).mean(dim=0)

        prefill_importance = importance[:prefill_len]
        keep_budget = min(max(0, budget_per_layer - num_generated), prefill_len)
        if keep_budget > 0:
            _, selected = prefill_importance.topk(keep_budget)
            selected_positions = selected.sort().values
            selected_k = all_k.squeeze(0).index_select(1, selected_positions)
            selected_v = all_v.squeeze(0).index_select(1, selected_positions)
        else:
            selected_positions = torch.empty(0, dtype=torch.long, device=all_k.device)
            selected_k = all_k.squeeze(0)[:, :0, :]
            selected_v = all_v.squeeze(0)[:, :0, :]

        if num_generated > 0:
            gen_k_squeezed = all_k.squeeze(0)[:, prefill_len:, :]
            gen_v_squeezed = all_v.squeeze(0)[:, prefill_len:, :]
            new_k = torch.cat([selected_k, gen_k_squeezed], dim=1).unsqueeze(0)
            new_v = torch.cat([selected_v, gen_v_squeezed], dim=1).unsqueeze(0)
        else:
            new_k = selected_k.unsqueeze(0)
            new_v = selected_v.unsqueeze(0)

        compressed_layers.append((new_k, new_v))
        updated_positions.append(selected_positions.detach().cpu())
        layers_refreshed += 1

    for layer_idx in range(len(compressed_layers), total_layers):
        current_k, current_v = get_layer_kv(past_key_values, layer_idx)
        compressed_layers.append((current_k, current_v))
        updated_positions.append(retained_positions[layer_idx].clone())

    return build_cache(compressed_layers), updated_positions, layers_refreshed


class FullKVStore:
    """
    Stores full uncompressed prefill KV in memory.

    On A10G-class 24 GB GPUs, keep the full backup on CPU by default.
    """

    def __init__(self, offload_to_cpu: bool = False):
        self.offload_to_cpu = offload_to_cpu
        self.layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.device = None

    @torch.no_grad()
    def store(self, past_key_values):
        from idlekv.utils.kv_cache import get_layer_kv, num_layers as cache_num_layers

        self.layers = []
        total_layers = cache_num_layers(past_key_values)

        for layer_idx in range(total_layers):
            k, v = get_layer_kv(past_key_values, layer_idx)
            if self.offload_to_cpu:
                self.layers.append((
                    k.to("cpu", non_blocking=True),
                    v.to("cpu", non_blocking=True),
                ))
                self.device = "cpu"
            else:
                self.layers.append((k, v))
                self.device = k.device

        if self.offload_to_cpu and k.device.type == "cuda":
            torch.cuda.synchronize()

    def get_layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.layers[layer_idx]

    def get_spans(
        self,
        layer_idx: int,
        spans: list[tuple[int, int]],
        *,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Materialize a small set of contiguous prefill spans for one layer.

        Returns:
            keys: [num_heads, total_tokens, head_dim]
            values: [num_heads, total_tokens, head_dim]
            positions: [total_tokens]
            bytes_loaded: bytes transferred from the cold store tensor slices
        """
        full_k, full_v = self.get_layer(layer_idx)
        if not spans:
            head_dim = full_k.shape[-1]
            num_heads = full_k.shape[1]
            return (
                torch.empty(num_heads, 0, head_dim, device=device, dtype=full_k.dtype),
                torch.empty(num_heads, 0, head_dim, device=device, dtype=full_v.dtype),
                torch.empty(0, device=device, dtype=torch.long),
                0,
            )

        k_chunks = []
        v_chunks = []
        pos_chunks = []
        bytes_loaded = 0
        for start, end in spans:
            k_slice = full_k[0, :, start:end, :]
            v_slice = full_v[0, :, start:end, :]
            k_chunks.append(k_slice)
            v_chunks.append(v_slice)
            pos_chunks.append(torch.arange(start, end, dtype=torch.long))
            bytes_loaded += (
                k_slice.nelement() * k_slice.element_size() +
                v_slice.nelement() * v_slice.element_size()
            )

        keys = torch.cat(k_chunks, dim=1)
        values = torch.cat(v_chunks, dim=1)
        positions = torch.cat(pos_chunks, dim=0)

        if str(keys.device) != str(device):
            keys = keys.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            positions = positions.to(device)
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize()

        return keys, values, positions, bytes_loaded

    def __len__(self):
        return len(self.layers)

    @property
    def memory_bytes(self) -> int:
        total = 0
        for k, v in self.layers:
            total += k.nelement() * k.element_size()
            total += v.nelement() * v.element_size()
        return total


CPUKVStore = FullKVStore
