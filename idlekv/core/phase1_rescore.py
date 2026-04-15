"""
Phase 1: query-aware re-scoring of retained and shadowed KV tokens.

The legacy Phase 1 path is now implemented as a special case of the shared
layer-local anytime repair operator: current retained prefill tokens plus the
shadow buffer form the candidate pool, and the top-k prefill tokens are
committed while generated tokens remain pinned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from idlekv.core.anytime_repair import (
    LayerRepairOutcome,
    _get_q_proj,
    compute_importance_scores,
    compute_importance_scores_gqa,
    project_queries_all_heads,
    project_queries_grouped,
    repair_layer_pool,
)
from idlekv.core.shadow_buffer import ShadowBuffer
from idlekv.core.query_buffer import QueryBuffer
from idlekv.utils.kv_cache import build_cache, get_layer_kv, num_layers


@dataclass
class RepairPassResult:
    past_key_values: object
    retained_positions: list[torch.Tensor]
    layers_touched: int
    candidate_tokens: int
    shadow_tokens: int
    sampled_tokens: int = 0


def _infer_retained_positions(
    past_key_values,
    total_layers: int,
    num_generated: int,
) -> list[torch.Tensor]:
    inferred = []
    for layer_idx in range(total_layers):
        current_k, _ = get_layer_kv(past_key_values, layer_idx)
        full_len = current_k.shape[2]
        prefill_len = max(0, full_len - min(num_generated, full_len))
        inferred.append(torch.arange(prefill_len, dtype=torch.long))
    return inferred


def phase1_rescore(
    past_key_values,
    shadow_buffer: ShadowBuffer,
    query_buffer: QueryBuffer,
    model,
    budget_per_layer: int,
    num_layers_arg: int,
    interrupt_flag: Optional[callable] = None,
    num_generated: int = 0,
    retained_positions: Optional[list[torch.Tensor]] = None,
    cold_candidates: Optional[dict[int, object]] = None,
) -> RepairPassResult:
    """
    Re-score retained + shadow (+ optional cold) tokens and rebuild the cache.

    Args:
        past_key_values: current compressed KV cache
        shadow_buffer: per-layer shadow candidates
        query_buffer: recent per-layer hidden states used as queries
        model: HF model
        budget_per_layer: target cache budget
        num_layers_arg: kept for backward compatibility
        interrupt_flag: stop callback
        num_generated: generated tail length to pin
        retained_positions: original prefill positions of the retained prefix
        cold_candidates: optional extra per-layer candidates, keyed by layer idx
    """
    del num_layers_arg  # maintained for backward compatibility with older callers

    total_layers = num_layers(past_key_values)
    retained_positions = retained_positions or _infer_retained_positions(
        past_key_values,
        total_layers,
        num_generated=num_generated,
    )

    if query_buffer.count == 0:
        return RepairPassResult(
            past_key_values=past_key_values,
            retained_positions=[pos.clone() for pos in retained_positions],
            layers_touched=0,
            candidate_tokens=0,
            shadow_tokens=0,
            sampled_tokens=0,
        )

    compressed_layers = []
    updated_positions: list[torch.Tensor] = []
    total_candidates = 0
    total_shadow = 0
    total_sampled = 0

    for layer_idx in range(total_layers):
        if interrupt_flag is not None and interrupt_flag():
            break

        current_k, current_v = get_layer_kv(past_key_values, layer_idx)
        recent_h = query_buffer.get(layer_idx=layer_idx)
        shadow_k, shadow_v, shadow_positions = shadow_buffer.get(layer_idx)
        cold_batch = None if cold_candidates is None else cold_candidates.get(layer_idx)

        outcome: LayerRepairOutcome = repair_layer_pool(
            current_k=current_k,
            current_v=current_v,
            retained_positions=retained_positions[layer_idx],
            shadow_k=shadow_k,
            shadow_v=shadow_v,
            shadow_positions=shadow_positions,
            recent_h=recent_h,
            model=model,
            layer_idx=layer_idx,
            budget_per_layer=budget_per_layer,
            cold_candidates=cold_batch,
        )

        shadow_buffer.clear(layer_idx)
        if outcome.shadow_positions.numel() > 0:
            shadow_buffer.push(
                layer_idx,
                outcome.shadow_k,
                outcome.shadow_v,
                positions=outcome.shadow_positions,
            )

        compressed_layers.append((outcome.new_k, outcome.new_v))
        updated_positions.append(outcome.retained_positions)
        total_candidates += outcome.candidate_count
        total_shadow += outcome.shadow_candidate_count
        total_sampled += outcome.sampled_candidate_count

    for layer_idx in range(len(compressed_layers), total_layers):
        current_k, current_v = get_layer_kv(past_key_values, layer_idx)
        compressed_layers.append((current_k, current_v))
        updated_positions.append(retained_positions[layer_idx].clone())

    return RepairPassResult(
        past_key_values=build_cache(compressed_layers),
        retained_positions=updated_positions,
        layers_touched=len(compressed_layers),
        candidate_tokens=total_candidates,
        shadow_tokens=total_shadow,
        sampled_tokens=total_sampled,
    )


# Backward-compat wrappers: several modules and tests import these helpers.
def _project_queries_all_heads(hidden_states, q_proj, head_dim):
    return project_queries_all_heads(hidden_states, q_proj, head_dim)


def _project_queries(hidden_states, q_proj, num_kv_heads, head_dim):
    return project_queries_grouped(hidden_states, q_proj, num_kv_heads, head_dim)


def _compute_importance_scores_gqa(q_all_heads, keys, head_dim, num_q_heads, num_kv_heads):
    return compute_importance_scores_gqa(q_all_heads, keys, head_dim, num_q_heads, num_kv_heads)


def _compute_importance_scores(queries, keys, head_dim):
    return compute_importance_scores(queries, keys, head_dim)

