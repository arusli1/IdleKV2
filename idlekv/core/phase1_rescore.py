"""
Phase 1: TIR-informed importance re-scoring during idle time.

Given recent queries (from QueryBuffer) and candidate tokens (retained cache +
shadow buffer), re-rank all candidates and select the top-k to form the new
compressed cache. Tokens in the shadow buffer that are now more important than
the least-important retained tokens get swapped in.

Cost is hardware-dependent; the implementation is designed to fit within a
sub-100ms idle window for Llama-3.1-8B at 4K context on the target A10G setup.
"""

import torch
import math
from typing import Optional

from idlekv.core.shadow_buffer import ShadowBuffer
from idlekv.core.query_buffer import QueryBuffer
from idlekv.utils.kv_cache import get_layer_kv, build_cache, num_layers


def phase1_rescore(
    past_key_values,
    shadow_buffer: ShadowBuffer,
    query_buffer: QueryBuffer,
    model,
    budget_per_layer: int,
    num_layers_arg: int,
    interrupt_flag: Optional[callable] = None,
    num_generated: int = 0,
):
    """
    Re-score retained + shadow tokens using recent queries and rebuild cache.

    This is the cheap, fast idle-time operation (<100ms). It exploits TIR by
    giving shadow-buffered tokens a second chance to prove their importance.

    Args:
        past_key_values: current compressed KV cache (DynamicCache or tuple).
                         K shape: [batch, num_kv_heads, seq_len, head_dim]
        shadow_buffer: ShadowBuffer containing recently evicted KV pairs
        query_buffer: QueryBuffer containing recent hidden states
        model: the HF model (needed for Q/K/V projection weights)
        budget_per_layer: target number of tokens to retain per layer
        num_layers_arg: number of transformer layers
        interrupt_flag: callable returning True if tool has returned (stop early)

    Returns:
        Updated past_key_values with re-scored cache (same format as input)
    """
    recent_h = query_buffer.get()  # [num_queries, hidden_dim]
    if recent_h.shape[0] == 0:
        return past_key_values  # nothing to score with

    compressed_layers = []
    total_layers = num_layers(past_key_values)

    for layer_idx in range(total_layers):
        if interrupt_flag is not None and interrupt_flag():
            break

        # Get current retained KV using utility function
        full_k, full_v = get_layer_kv(past_key_values, layer_idx)  # [B, H, S, D]
        B, H, S_full, D = full_k.shape
        assert B == 1, "Phase 1 assumes batch_size=1"

        # Split prefill part (re-scorable) from generated tail (always preserved).
        # Generated tokens carry committed outputs — evicting them would corrupt
        # the model's self-attention over its own prior generation.
        S_gen = min(num_generated, S_full)
        S_prefill = S_full - S_gen
        prefill_k = full_k[:, :, :S_prefill, :]
        prefill_v = full_v[:, :, :S_prefill, :]
        gen_k = full_k[:, :, S_prefill:, :]
        gen_v = full_v[:, :, S_prefill:, :]

        # Get shadow KV for this layer
        shadow_k, shadow_v = shadow_buffer.get(layer_idx)  # [H, S_shadow, D]
        S_shadow = shadow_k.shape[1]

        if S_shadow == 0:
            # No shadow tokens — nothing to swap, keep current layer
            compressed_layers.append((full_k, full_v))
            continue

        # Concatenate prefill + shadow as candidates (generated is kept aside)
        all_k = torch.cat([
            prefill_k.squeeze(0),   # [H, S_prefill, D]
            shadow_k,               # [H, S_shadow, D]
        ], dim=1)  # [H, S_total, D]

        all_v = torch.cat([
            prefill_v.squeeze(0),
            shadow_v,
        ], dim=1)

        S_total = all_k.shape[1]

        # Project recent hidden states through full Q projection (keeping all
        # Q heads). We score attention per-Q-head then mean across GQA groups,
        # which matches kvpress SnapKV. Avoids the pre-attention Q-averaging
        # approximation that discards within-group head specialization.
        q_proj = _get_q_proj(model, layer_idx)
        # Returns [num_q_heads, num_queries, D] with no RoPE applied to queries.
        # Keys retain their original RoPE phases; scoring without query RoPE is
        # an approximation but preserves relative ordering far better than
        # applying RoPE at a fabricated position 0.
        q_all_heads = _project_queries_all_heads(recent_h, q_proj, D)
        num_q_heads = q_all_heads.shape[0]

        # Score all candidates per-Q-head, then group-mean to per-KV-head,
        # then head-mean for token selection. Budget is for prefill tokens only;
        # generated tokens are always kept.
        budget_prefill = max(0, budget_per_layer - S_gen)
        scores = _compute_importance_scores_gqa(
            q_all_heads, all_k, D, num_q_heads, H
        )  # [H, S_total]
        mean_scores = scores.mean(dim=0)  # [S_total]
        _, top_indices = mean_scores.topk(min(budget_prefill, S_total))
        top_indices = top_indices.sort().values

        # Rebuild cache: selected prefill/shadow tokens + generated tail
        sel_k = all_k[:, top_indices, :].unsqueeze(0)  # [1, H, budget_prefill, D]
        sel_v = all_v[:, top_indices, :].unsqueeze(0)
        if S_gen > 0:
            new_k = torch.cat([sel_k, gen_k], dim=2)
            new_v = torch.cat([sel_v, gen_v], dim=2)
        else:
            new_k = sel_k
            new_v = sel_v

        # Push the newly evicted candidates (not selected, and originally from
        # the prefill/shadow pool) to shadow buffer.
        evicted_mask = torch.ones(S_total, dtype=torch.bool, device=all_k.device)
        evicted_mask[top_indices] = False
        evicted_indices = torch.arange(S_total, device=all_k.device)[evicted_mask]
        if evicted_indices.numel() > 0:
            evicted_k = all_k[:, evicted_indices, :]
            evicted_v = all_v[:, evicted_indices, :]
            shadow_buffer.clear(layer_idx)
            shadow_buffer.push(layer_idx, evicted_k, evicted_v)

        compressed_layers.append((new_k, new_v))

    # For layers that were interrupted, keep their current state
    for layer_idx in range(len(compressed_layers), total_layers):
        current_k, current_v = get_layer_kv(past_key_values, layer_idx)
        compressed_layers.append((current_k, current_v))

    # Return in same format as input
    return build_cache(compressed_layers)


def _get_q_proj(model, layer_idx: int):
    """Extract Q projection weight from a HF model. Supports Llama and Qwen."""
    layer = model.model.layers[layer_idx]
    return layer.self_attn.q_proj


def _project_queries_all_heads(
    hidden_states: torch.Tensor,
    q_proj: torch.nn.Linear,
    head_dim: int,
) -> torch.Tensor:
    """
    Project hidden states through Q projection, returning all Q heads.

    No RoPE is applied to the queries. Keys in the cache retain their
    original RoPE phases (standard SnapKV/H2O/kvpress convention). Scoring
    without query RoPE is an approximation; the alternative (fabricated
    position 0) is worse because it introduces a per-head phase twist
    unrelated to the real attention geometry.

    Args:
        hidden_states: [num_queries, hidden_dim]
        q_proj: Q projection layer
        head_dim: dimension per head

    Returns:
        [num_q_heads, num_queries, head_dim]
    """
    with torch.no_grad():
        q = q_proj(hidden_states)  # [num_queries, num_q_heads * head_dim]
    num_queries = q.shape[0]
    num_q_heads = q.shape[1] // head_dim
    q = q.view(num_queries, num_q_heads, head_dim).permute(1, 0, 2).contiguous()
    return q  # [num_q_heads, num_queries, head_dim]


def _compute_importance_scores_gqa(
    q_all_heads: torch.Tensor,
    keys: torch.Tensor,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """
    Compute per-KV-head importance scores under GQA.

    Matches kvpress SnapKV: compute attention per Q head against its
    group's KV head, softmax, mean over queries, then mean across the
    group-size Q heads that share each KV head.

    Args:
        q_all_heads: [num_q_heads, num_queries, D]
        keys: [num_kv_heads, S_total, D] — keys for this layer (one per KV head)
        head_dim: for scaling
        num_q_heads: number of Q heads
        num_kv_heads: number of KV heads (== GQA groups)

    Returns:
        [num_kv_heads, S_total] importance per KV head
    """
    scale = 1.0 / math.sqrt(head_dim)
    group_size = num_q_heads // num_kv_heads
    num_queries = q_all_heads.shape[1]
    S_total = keys.shape[1]

    # Expand keys to match Q heads: each KV head's keys used by group_size Q heads
    # keys_expanded: [num_q_heads, S_total, D]
    keys_expanded = keys.repeat_interleave(group_size, dim=0)

    attn = torch.bmm(q_all_heads, keys_expanded.transpose(1, 2)) * scale
    attn_w = torch.softmax(attn, dim=-1)                 # [num_q_heads, Q, S]
    importance_q = attn_w.mean(dim=1)                    # [num_q_heads, S]
    # Mean across group → per-KV-head importance
    importance_kv = importance_q.view(
        num_kv_heads, group_size, S_total
    ).mean(dim=1)                                        # [num_kv_heads, S]
    return importance_kv


# Backward-compat wrappers (phase2_refresh imports these)
def _project_queries(hidden_states, q_proj, num_kv_heads, head_dim):
    """Project and collapse to num_kv_heads via group mean of Q vectors.

    Used only by phase2_refresh, which computes a coarser single-pass score.
    """
    q = _project_queries_all_heads(hidden_states, q_proj, head_dim)
    num_q_heads = q.shape[0]
    if num_q_heads == num_kv_heads:
        return q
    group_size = num_q_heads // num_kv_heads
    num_queries = q.shape[1]
    return q.view(num_kv_heads, group_size, num_queries, head_dim).mean(dim=1)


def _compute_importance_scores(queries, keys, head_dim):
    """Legacy per-head scorer used by phase2."""
    scale = 1.0 / math.sqrt(head_dim)
    attn_scores = torch.bmm(queries, keys.transpose(1, 2)) * scale
    attn_weights = torch.softmax(attn_scores, dim=-1)
    return attn_weights.mean(dim=1)
