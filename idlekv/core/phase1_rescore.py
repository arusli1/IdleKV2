"""
Phase 1: TIR-informed importance re-scoring during idle time.

Given recent queries (from QueryBuffer) and candidate tokens (retained cache +
shadow buffer), re-rank all candidates and select the top-k to form the new
compressed cache. Tokens in the shadow buffer that are now more important than
the least-important retained tokens get swapped in.

Cost: ~15-70ms on RTX 6000 / L40S for Llama-3.1-8B at 4K context.
"""

import torch
import math
from typing import Optional

from idlekv.core.shadow_buffer import ShadowBuffer
from idlekv.core.query_buffer import QueryBuffer


def phase1_rescore(
    past_key_values: tuple,
    shadow_buffer: ShadowBuffer,
    query_buffer: QueryBuffer,
    model,
    budget_per_layer: int,
    num_layers: int,
    interrupt_flag: Optional[callable] = None,
) -> tuple:
    """
    Re-score retained + shadow tokens using recent queries and rebuild cache.

    This is the cheap, fast idle-time operation (<100ms). It exploits TIR by
    giving shadow-buffered tokens a second chance to prove their importance.

    Args:
        past_key_values: current compressed KV cache (tuple of (K, V) per layer).
                         K shape: [batch, num_kv_heads, seq_len, head_dim]
        shadow_buffer: ShadowBuffer containing recently evicted KV pairs
        query_buffer: QueryBuffer containing recent hidden states
        model: the HF model (needed for Q/K/V projection weights)
        budget_per_layer: target number of tokens to retain per layer
        num_layers: number of transformer layers
        interrupt_flag: callable returning True if tool has returned (stop early)

    Returns:
        Updated past_key_values tuple with re-scored cache
    """
    recent_h = query_buffer.get()  # [num_queries, hidden_dim]
    if recent_h.shape[0] == 0:
        return past_key_values  # nothing to score with

    new_kv = list(past_key_values)

    for layer_idx in range(num_layers):
        if interrupt_flag is not None and interrupt_flag():
            break

        # Get current retained KV
        retained_k, retained_v = past_key_values[layer_idx]  # [B, H, S, D]
        B, H, S_retained, D = retained_k.shape
        assert B == 1, "Phase 1 assumes batch_size=1"

        # Get shadow KV for this layer
        shadow_k, shadow_v = shadow_buffer.get(layer_idx)  # [H, S_shadow, D]
        S_shadow = shadow_k.shape[1]

        if S_shadow == 0:
            # No shadow tokens — nothing to swap
            continue

        # Concatenate retained + shadow as candidates
        # retained_k: [1, H, S_retained, D], shadow_k: [H, S_shadow, D]
        all_k = torch.cat([
            retained_k.squeeze(0),  # [H, S_retained, D]
            shadow_k,               # [H, S_shadow, D]
        ], dim=1)  # [H, S_total, D]

        all_v = torch.cat([
            retained_v.squeeze(0),
            shadow_v,
        ], dim=1)

        S_total = all_k.shape[1]

        # Project recent hidden states to queries for this layer
        # Access Q projection weights from the model
        q_proj = _get_q_proj(model, layer_idx)
        # recent_h: [num_queries, hidden_dim]
        queries = _project_queries(recent_h, q_proj, H, D)  # [H, num_queries, D]

        # Score all candidates: mean attention across recent queries
        # scores: [H, S_total]
        scores = _compute_importance_scores(queries, all_k, D)

        # Select top-k per head (or globally — start with per-head for simplicity)
        # Use mean score across heads for selection
        mean_scores = scores.mean(dim=0)  # [S_total]
        _, top_indices = mean_scores.topk(min(budget_per_layer, S_total))
        top_indices = top_indices.sort().values

        # Rebuild cache with selected tokens
        new_k = all_k[:, top_indices, :].unsqueeze(0)  # [1, H, budget, D]
        new_v = all_v[:, top_indices, :].unsqueeze(0)

        # Push the newly evicted tokens (those NOT selected) to shadow buffer
        all_indices = torch.arange(S_total, device=all_k.device)
        evicted_mask = torch.ones(S_total, dtype=torch.bool, device=all_k.device)
        evicted_mask[top_indices] = False
        evicted_indices = all_indices[evicted_mask]

        if evicted_indices.numel() > 0:
            evicted_k = all_k[:, evicted_indices, :]
            evicted_v = all_v[:, evicted_indices, :]
            shadow_buffer.clear(layer_idx)
            shadow_buffer.push(layer_idx, evicted_k, evicted_v)

        new_kv[layer_idx] = (new_k, new_v)

    return tuple(new_kv)


def _get_q_proj(model, layer_idx: int):
    """Extract Q projection weight from a HF model. Supports Llama and Qwen."""
    layer = model.model.layers[layer_idx]
    return layer.self_attn.q_proj


def _project_queries(
    hidden_states: torch.Tensor,
    q_proj: torch.nn.Linear,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Project hidden states through Q projection and reshape for attention.

    Args:
        hidden_states: [num_queries, hidden_dim]
        q_proj: Q projection layer
        num_kv_heads: number of KV heads (GQA groups)
        head_dim: dimension per head

    Returns:
        [num_kv_heads, num_queries, head_dim] (averaged over Q heads per KV group)
    """
    with torch.no_grad():
        q = q_proj(hidden_states)  # [num_queries, num_q_heads * head_dim]

    num_queries = q.shape[0]
    num_q_heads = q.shape[1] // head_dim
    q = q.view(num_queries, num_q_heads, head_dim)  # [Q, num_q_heads, D]
    q = q.permute(1, 0, 2)  # [num_q_heads, Q, D]

    # If GQA: average Q heads within each KV group
    if num_q_heads != num_kv_heads:
        group_size = num_q_heads // num_kv_heads
        q = q.view(num_kv_heads, group_size, num_queries, head_dim).mean(dim=1)

    return q  # [num_kv_heads, num_queries, head_dim]


def _compute_importance_scores(
    queries: torch.Tensor,
    keys: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """
    Compute mean attention score for each key across all queries.

    Args:
        queries: [H, num_queries, D]
        keys: [H, S_total, D]
        head_dim: for scaling

    Returns:
        [H, S_total] importance scores
    """
    scale = 1.0 / math.sqrt(head_dim)
    # [H, num_queries, S_total]
    attn_scores = torch.bmm(queries, keys.transpose(1, 2)) * scale
    # Softmax over key dimension, then mean over queries
    attn_weights = torch.softmax(attn_scores, dim=-1)  # [H, Q, S]
    importance = attn_weights.mean(dim=1)  # [H, S]
    return importance
