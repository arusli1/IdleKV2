"""
Shared utilities for interruptible cache repair.

This module centralizes the low-level mechanics needed by both the legacy
shadow-buffer pass and the new sampled cold-store repair path:

- per-layer query projection and scoring
- candidate-pool deduplication by original prefill position
- layer-local cache commit / shadow-buffer rebuild
- A10G-safe cold-span sampling from the CPU KV store
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ColdCandidateBatch:
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor
    bytes_loaded: int = 0
    span_count: int = 0


@dataclass
class LayerRepairOutcome:
    new_k: torch.Tensor
    new_v: torch.Tensor
    retained_positions: torch.Tensor
    shadow_k: torch.Tensor
    shadow_v: torch.Tensor
    shadow_positions: torch.Tensor
    candidate_count: int
    shadow_candidate_count: int
    sampled_candidate_count: int


def _get_q_proj(model, layer_idx: int):
    """Extract the Q projection from a HF transformer block."""
    layer = model.model.layers[layer_idx]
    return layer.self_attn.q_proj


def project_queries_all_heads(
    hidden_states: torch.Tensor,
    q_proj: torch.nn.Linear,
    head_dim: int,
) -> torch.Tensor:
    """
    Project hidden states through Q projection, returning all Q heads.

    No RoPE is applied to the queries. Keys in the cache retain their
    original RoPE phases; this is the same approximation used elsewhere in
    the repo and is adequate for ranking candidates.
    """
    with torch.no_grad():
        q = q_proj(hidden_states)  # [num_queries, num_q_heads * head_dim]
    num_queries = q.shape[0]
    num_q_heads = q.shape[1] // head_dim
    return q.view(num_queries, num_q_heads, head_dim).permute(1, 0, 2).contiguous()


def project_queries_grouped(
    hidden_states: torch.Tensor,
    q_proj: torch.nn.Linear,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Project queries and average Q heads inside each KV group."""
    q = project_queries_all_heads(hidden_states, q_proj, head_dim)
    num_q_heads = q.shape[0]
    if num_q_heads == num_kv_heads:
        return q
    group_size = num_q_heads // num_kv_heads
    num_queries = q.shape[1]
    return q.view(num_kv_heads, group_size, num_queries, head_dim).mean(dim=1)


def compute_importance_scores_gqa(
    q_all_heads: torch.Tensor,
    keys: torch.Tensor,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """
    Compute per-KV-head importance scores under GQA.

    This matches the repo's SnapKV-like scoring path: attention per Q head,
    mean over recent queries, then mean across the Q heads that share a KV
    head.
    """
    scale = 1.0 / math.sqrt(head_dim)
    group_size = num_q_heads // num_kv_heads
    keys_expanded = keys.repeat_interleave(group_size, dim=0)
    attn = torch.bmm(q_all_heads, keys_expanded.transpose(1, 2)) * scale
    attn_w = torch.softmax(attn, dim=-1)
    importance_q = attn_w.mean(dim=1)
    return importance_q.view(num_kv_heads, group_size, keys.shape[1]).mean(dim=1)


def compute_importance_scores(
    queries: torch.Tensor,
    keys: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Legacy grouped-query scorer used by Phase 2."""
    scale = 1.0 / math.sqrt(head_dim)
    attn_scores = torch.bmm(queries, keys.transpose(1, 2)) * scale
    attn_weights = torch.softmax(attn_scores, dim=-1)
    return attn_weights.mean(dim=1)


def _empty_candidate_tensor(reference: torch.Tensor) -> torch.Tensor:
    return reference[:, :0, :]


def _merge_candidate_sources(
    retained_k: torch.Tensor,
    retained_v: torch.Tensor,
    retained_positions: torch.Tensor,
    shadow_k: torch.Tensor,
    shadow_v: torch.Tensor,
    shadow_positions: torch.Tensor,
    cold_k: Optional[torch.Tensor] = None,
    cold_v: Optional[torch.Tensor] = None,
    cold_positions: Optional[torch.Tensor] = None,
):
    """Merge retained, shadow, and cold candidates with position-based dedup."""
    sources = [
        ("retained", retained_k, retained_v, retained_positions),
        ("shadow", shadow_k, shadow_v, shadow_positions),
    ]
    if cold_k is not None and cold_v is not None and cold_positions is not None:
        sources.append(("cold", cold_k, cold_v, cold_positions))

    priority = {"retained": 0, "shadow": 1, "cold": 2}
    merged: dict[int, tuple[str, torch.Tensor, torch.Tensor]] = {}

    for source_name, keys, values, positions in sources:
        if keys.numel() == 0 or positions.numel() == 0:
            continue
        for candidate_idx, pos in enumerate(positions.tolist()):
            if pos < 0:
                continue
            existing = merged.get(pos)
            if existing is not None and priority[existing[0]] <= priority[source_name]:
                continue
            merged[pos] = (
                source_name,
                keys[:, candidate_idx, :].clone(),
                values[:, candidate_idx, :].clone(),
            )

    if not merged:
        empty_keys = _empty_candidate_tensor(retained_k)
        empty_positions = retained_positions[:0].clone()
        return empty_keys, empty_keys.clone(), empty_positions, []

    sorted_positions = sorted(merged.keys())
    positions = torch.tensor(
        sorted_positions,
        device=retained_k.device,
        dtype=torch.long,
    )
    keys = torch.stack([merged[pos][1] for pos in sorted_positions], dim=1)
    values = torch.stack([merged[pos][2] for pos in sorted_positions], dim=1)
    source_names = [merged[pos][0] for pos in sorted_positions]
    return keys, values, positions, source_names


def _dedup_shadow_entries(
    keys: torch.Tensor,
    values: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deduplicate shadow entries while preserving FIFO order."""
    if positions.numel() == 0:
        return keys[:, :0, :], values[:, :0, :], positions[:0]

    kept_indices = []
    seen = set()
    for idx, pos in enumerate(positions.tolist()):
        if pos < 0 or pos in seen:
            continue
        seen.add(pos)
        kept_indices.append(idx)

    if not kept_indices:
        return keys[:, :0, :], values[:, :0, :], positions[:0]

    index = torch.tensor(kept_indices, device=keys.device, dtype=torch.long)
    return (
        keys.index_select(1, index),
        values.index_select(1, index),
        positions.index_select(0, index),
    )


def repair_layer_pool(
    *,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    retained_positions: torch.Tensor,
    shadow_k: torch.Tensor,
    shadow_v: torch.Tensor,
    shadow_positions: torch.Tensor,
    recent_h: torch.Tensor,
    model,
    layer_idx: int,
    budget_per_layer: int,
    cold_candidates: Optional[ColdCandidateBatch] = None,
) -> LayerRepairOutcome:
    """
    Re-score a single layer's candidate pool and commit the top-k prefill KV.

    `retained_positions` must correspond to the prefill prefix stored at the
    front of `current_k/current_v`. Any trailing tokens are treated as pinned
    generated KV and preserved unchanged.
    """
    current_prefill_len = retained_positions.numel()
    full_len = current_k.shape[2]
    if current_prefill_len > full_len:
        raise ValueError(
            f"retained_positions length {current_prefill_len} exceeds cache length {full_len}"
        )

    prefill_k = current_k[:, :, :current_prefill_len, :]
    prefill_v = current_v[:, :, :current_prefill_len, :]
    gen_k = current_k[:, :, current_prefill_len:, :]
    gen_v = current_v[:, :, current_prefill_len:, :]
    num_generated = gen_k.shape[2]

    retained_k = prefill_k.squeeze(0)
    retained_v = prefill_v.squeeze(0)
    retained_positions = retained_positions.to(device=retained_k.device, dtype=torch.long)
    shadow_positions = shadow_positions.to(device=retained_k.device, dtype=torch.long)

    cold_k = cold_v = cold_positions = None
    sampled_candidate_count = 0
    if cold_candidates is not None:
        cold_k = cold_candidates.keys.to(device=retained_k.device, dtype=retained_k.dtype)
        cold_v = cold_candidates.values.to(device=retained_k.device, dtype=retained_v.dtype)
        cold_positions = cold_candidates.positions.to(device=retained_k.device, dtype=torch.long)
        sampled_candidate_count = int(cold_positions.numel())

    all_k, all_v, all_positions, source_names = _merge_candidate_sources(
        retained_k=retained_k,
        retained_v=retained_v,
        retained_positions=retained_positions,
        shadow_k=shadow_k,
        shadow_v=shadow_v,
        shadow_positions=shadow_positions,
        cold_k=cold_k,
        cold_v=cold_v,
        cold_positions=cold_positions,
    )

    if all_positions.numel() == 0:
        return LayerRepairOutcome(
            new_k=current_k,
            new_v=current_v,
            retained_positions=retained_positions.detach().cpu(),
            shadow_k=shadow_k[:, :0, :],
            shadow_v=shadow_v[:, :0, :],
            shadow_positions=shadow_positions[:0],
            candidate_count=0,
            shadow_candidate_count=int(shadow_positions.numel()),
            sampled_candidate_count=sampled_candidate_count,
        )

    q_proj = _get_q_proj(model, layer_idx)
    q_all_heads = project_queries_all_heads(recent_h, q_proj, current_k.shape[-1])
    num_q_heads = q_all_heads.shape[0]
    scores = compute_importance_scores_gqa(
        q_all_heads,
        all_k,
        current_k.shape[-1],
        num_q_heads,
        all_k.shape[0],
    )
    mean_scores = scores.mean(dim=0)

    budget_prefill = max(0, budget_per_layer - num_generated)
    keep_count = min(budget_prefill, all_positions.numel())
    if keep_count == 0:
        selected_indices = torch.empty(0, device=all_positions.device, dtype=torch.long)
    else:
        _, selected_indices = mean_scores.topk(keep_count)
        selected_positions = all_positions.index_select(0, selected_indices)
        order = torch.argsort(selected_positions)
        selected_indices = selected_indices.index_select(0, order)

    new_positions = all_positions.index_select(0, selected_indices)
    sel_k = all_k.index_select(1, selected_indices).unsqueeze(0)
    sel_v = all_v.index_select(1, selected_indices).unsqueeze(0)
    if num_generated > 0:
        new_k = torch.cat([sel_k, gen_k], dim=2)
        new_v = torch.cat([sel_v, gen_v], dim=2)
    else:
        new_k = sel_k
        new_v = sel_v

    selected_pos_set = set(new_positions.tolist())
    surviving_shadow_indices = [
        idx
        for idx, source_name in enumerate(source_names)
        if source_name == "shadow" and all_positions[idx].item() not in selected_pos_set
    ]
    newly_evicted_retained_indices = [
        idx
        for idx, source_name in enumerate(source_names)
        if source_name == "retained" and all_positions[idx].item() not in selected_pos_set
    ]

    shadow_index = torch.tensor(
        surviving_shadow_indices,
        device=all_k.device,
        dtype=torch.long,
    ) if surviving_shadow_indices else None
    retained_index = torch.tensor(
        newly_evicted_retained_indices,
        device=all_k.device,
        dtype=torch.long,
    ) if newly_evicted_retained_indices else None

    shadow_parts = []
    if shadow_index is not None:
        shadow_parts.append((
            all_k.index_select(1, shadow_index),
            all_v.index_select(1, shadow_index),
            all_positions.index_select(0, shadow_index),
        ))
    if retained_index is not None:
        shadow_parts.append((
            all_k.index_select(1, retained_index),
            all_v.index_select(1, retained_index),
            all_positions.index_select(0, retained_index),
        ))

    if shadow_parts:
        shadow_k_out = torch.cat([part[0] for part in shadow_parts], dim=1)
        shadow_v_out = torch.cat([part[1] for part in shadow_parts], dim=1)
        shadow_positions_out = torch.cat([part[2] for part in shadow_parts], dim=0)
        shadow_k_out, shadow_v_out, shadow_positions_out = _dedup_shadow_entries(
            shadow_k_out,
            shadow_v_out,
            shadow_positions_out,
        )
    else:
        shadow_k_out = retained_k[:, :0, :]
        shadow_v_out = retained_v[:, :0, :]
        shadow_positions_out = retained_positions[:0]

    return LayerRepairOutcome(
        new_k=new_k,
        new_v=new_v,
        retained_positions=new_positions.detach().cpu(),
        shadow_k=shadow_k_out,
        shadow_v=shadow_v_out,
        shadow_positions=shadow_positions_out,
        candidate_count=int(all_positions.numel()),
        shadow_candidate_count=int(shadow_positions.numel()),
        sampled_candidate_count=sampled_candidate_count,
    )


def sample_cold_spans(
    *,
    full_kv_store,
    layer_idx: int,
    retained_positions: torch.Tensor,
    shadow_positions: torch.Tensor,
    importance_scores: torch.Tensor,
    span_size: int,
    num_spans: int,
    seed: int,
    device: torch.device | str,
) -> ColdCandidateBatch:
    """
    Sample a small number of cold prefill spans for a single layer.

    Candidates are contiguous runs of eligible positions, chunked into fixed
    span sizes. Weighted sampling favors positions that scored highly at
    prefill time while adding a small recency bonus.
    """
    if span_size <= 0 or num_spans <= 0:
        empty = torch.empty(0, device=device, dtype=torch.long)
        return ColdCandidateBatch(
            keys=torch.empty(0, 0, 0, device=device),
            values=torch.empty(0, 0, 0, device=device),
            positions=empty,
        )

    full_k, _ = full_kv_store.get_layer(layer_idx)
    prefill_len = full_k.shape[2]
    if prefill_len == 0:
        empty = torch.empty(0, device=device, dtype=torch.long)
        return ColdCandidateBatch(
            keys=torch.empty(0, 0, 0, device=device),
            values=torch.empty(0, 0, 0, device=device),
            positions=empty,
        )

    eligible = torch.ones(prefill_len, dtype=torch.bool)
    retained_cpu = retained_positions.detach().cpu().long()
    if retained_cpu.numel() > 0:
        eligible[retained_cpu] = False

    shadow_cpu = shadow_positions.detach().cpu().long()
    shadow_cpu = shadow_cpu[shadow_cpu >= 0]
    if shadow_cpu.numel() > 0:
        eligible[shadow_cpu.unique()] = False

    available = torch.nonzero(eligible, as_tuple=False).squeeze(1).tolist()
    if not available:
        empty = torch.empty(0, device=device, dtype=torch.long)
        head_dim = full_k.shape[-1]
        num_heads = full_k.shape[1]
        return ColdCandidateBatch(
            keys=torch.empty(num_heads, 0, head_dim, device=device, dtype=full_k.dtype),
            values=torch.empty(num_heads, 0, head_dim, device=device, dtype=full_k.dtype),
            positions=empty,
        )

    runs = []
    run_start = available[0]
    prev = available[0]
    for pos in available[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        runs.append((run_start, prev + 1))
        run_start = pos
        prev = pos
    runs.append((run_start, prev + 1))

    spans = []
    scores = []
    importance_scores = importance_scores.detach().cpu().float()
    denom = max(prefill_len - 1, 1)
    for run_start, run_end in runs:
        cursor = run_start
        while cursor < run_end:
            span_end = min(cursor + span_size, run_end)
            spans.append((cursor, span_end))
            span_importance = importance_scores[cursor:span_end].mean().item()
            recency_bonus = 0.05 * (((cursor + span_end - 1) / 2.0) / denom)
            scores.append(span_importance + recency_bonus)
            cursor = span_end

    if not spans:
        empty = torch.empty(0, device=device, dtype=torch.long)
        head_dim = full_k.shape[-1]
        num_heads = full_k.shape[1]
        return ColdCandidateBatch(
            keys=torch.empty(num_heads, 0, head_dim, device=device, dtype=full_k.dtype),
            values=torch.empty(num_heads, 0, head_dim, device=device, dtype=full_k.dtype),
            positions=empty,
        )

    sample_count = min(num_spans, len(spans))
    weights = torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    chosen = torch.multinomial(weights, sample_count, replacement=False, generator=generator)
    selected_spans = [spans[idx] for idx in chosen.tolist()]

    keys, values, positions, bytes_loaded = full_kv_store.get_spans(
        layer_idx,
        selected_spans,
        device=device,
    )
    return ColdCandidateBatch(
        keys=keys,
        values=values,
        positions=positions,
        bytes_loaded=bytes_loaded,
        span_count=len(selected_spans),
    )
