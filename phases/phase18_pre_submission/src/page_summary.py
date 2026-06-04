"""Two-stage page-summary scorer (Quest/ShadowKV-inspired) for Phase 18 W1.

This module implements a *budgeted* two-stage reselection baseline that
operates at the post-compression / pre-resume lifecycle slot, like
RepairKV and Refresh-K-budgeted, but unlike them uses a cheap-then-fine
scoring strategy reminiscent of Quest's per-page criticality estimation
or ShadowKV's landmark scan.

Pipeline:

  1) Compress-time (free, paid once per example):
     For each chunk of ``chunk_size`` evicted positions, compute one
     **max-key summary vector** per layer per KV head -- elementwise max
     over the keys in that chunk along the position dimension. This is
     ~chunk_size cheaper than per-position storage, mirroring the
     "page summary" trick in Quest.

  2) Q2-time stage 1 (cheap):
     Score Q2 against the summary vectors. One small matmul of shape
     [n_layers, n_query_heads, q_len, n_chunks] -- size ``n_chunks`` not
     ``n_evicted``, hence cheap.

  3) Q2-time stage 2 (expensive, budgeted):
     Visit chunks in summary-rank order. For each chunk, fully score
     its positions with the same logic as ``score_evicted_positions``
     (concatenated active+chunk softmax). Stop when the wall-clock cap
     fires; remaining chunks get no scores.

Disclaimer: this is a two-stage scorer adapted to the lifecycle slot,
not a Quest reproduction. Quest operates per-decoding-step on a
preserved cache with min/max page envelopes; this baseline operates
once per pause boundary on an evicted host store with max-key
summaries. ArkVale and InfiniGen are the closer published prior art.
"""

from __future__ import annotations

import math
import time
from typing import Sequence

import torch

from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache


def compute_chunk_summaries(
    evicted_cache: PositionTrackedCache,
    *,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Compute per-chunk max-key summaries (compress-time cost, paid once).

    Returns
    -------
    summaries : torch.Tensor
        Shape ``[n_layers, n_kv_heads, n_chunks, head_dim]``, FP32 on CPU.
    chunk_ranges : list[(int, int)]
        ``(start, stop)`` indices into ``evicted_cache.positions`` for each
        chunk. ``len(chunk_ranges) == n_chunks``.
    """
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    n_evicted = len(evicted_cache.positions)
    if n_evicted == 0:
        return torch.empty(0), []
    n_layers = len(evicted_cache.kv)
    chunk_starts = list(range(0, n_evicted, int(chunk_size)))
    chunk_ranges = [(s, min(s + int(chunk_size), n_evicted)) for s in chunk_starts]
    n_chunks = len(chunk_ranges)
    head_dim = int(evicted_cache.kv[0][0].shape[-1])
    n_kv_heads = int(evicted_cache.kv[0][0].shape[1])
    summaries = torch.empty(
        (n_layers, n_kv_heads, n_chunks, head_dim), dtype=torch.float32
    )
    for layer_index, (key, _) in enumerate(evicted_cache.kv):
        layer_keys = key.detach().to("cpu", dtype=torch.float32)[0]  # [heads, n_evicted, head_dim]
        for chunk_idx, (start, stop) in enumerate(chunk_ranges):
            # max along the position dimension within the chunk
            chunk_keys = layer_keys[:, start:stop, :]
            summaries[layer_index, :, chunk_idx, :] = chunk_keys.amax(dim=1)
    return summaries, chunk_ranges


def _score_chunks_against_summaries(
    *,
    query_rows: torch.Tensor,
    summaries: torch.Tensor,
) -> torch.Tensor:
    """Stage 1: cheap chunk-level Q2 score using summary keys.

    Computes pooled attention logits per chunk: for each layer, head,
    query token, the inner product Q . summary, then mean over heads
    and amax over query tokens (analogous to the ``max`` pooling used
    by ``score_evicted_positions``). Returns one score per chunk.
    """
    if summaries.numel() == 0:
        return torch.empty(0)
    n_layers = int(summaries.shape[0])
    if int(query_rows.shape[0]) != n_layers:
        raise ValueError(
            f"query_rows layer count {int(query_rows.shape[0])} does not match summaries {n_layers}"
        )
    n_chunks = int(summaries.shape[2])
    chunk_scores_per_layer: list[torch.Tensor] = []
    head_dim = int(summaries.shape[-1])
    for layer_index in range(n_layers):
        query_layer = query_rows[layer_index]  # [n_query_heads, q_len, head_dim]
        n_query_heads = int(query_layer.shape[0])
        layer_summaries = summaries[layer_index]  # [n_kv_heads, n_chunks, head_dim]
        n_kv_heads = int(layer_summaries.shape[0])
        if n_query_heads != n_kv_heads:
            if n_query_heads % n_kv_heads != 0:
                raise ValueError(
                    f"query head count {n_query_heads} not compatible with kv head count {n_kv_heads}"
                )
            layer_summaries = layer_summaries.repeat_interleave(n_query_heads // n_kv_heads, dim=0)
        # [n_heads, q_len, head_dim] @ [n_heads, head_dim, n_chunks] -> [n_heads, q_len, n_chunks]
        logits = torch.matmul(query_layer, layer_summaries.transpose(-2, -1)) / math.sqrt(float(head_dim))
        # pool: amax over q_len, mean over heads -> [n_chunks]
        pooled = logits.amax(dim=1).mean(dim=0)
        chunk_scores_per_layer.append(pooled)
    # mean over layers
    chunk_scores = torch.stack(chunk_scores_per_layer, dim=0).mean(dim=0)
    return chunk_scores


def score_evicted_positions_page_summary(
    *,
    query_rows: torch.Tensor,
    evicted_cache: PositionTrackedCache,
    active_cache: PositionTrackedCache | None = None,
    pooling: str = "max",
    wallclock_deadline_s: float | None = None,
    chunk_size: int = 128,
    summaries: torch.Tensor | None = None,
    chunk_ranges: Sequence[tuple[int, int]] | None = None,
    precomputed_evicted_layer_keys: list[torch.Tensor] | None = None,
    precomputed_active_layer_keys: list[torch.Tensor | None] | None = None,
) -> tuple[dict[int, float], dict[str, int | bool | float]]:
    """Two-stage page-summary scorer with optional wall-clock cap.

    The ``summaries`` and ``chunk_ranges`` may be precomputed and passed
    in to charge the compress-time cost separately from the Q2-time
    cap. If not provided, they are computed inline (and the cap then
    covers the full pipeline, which is the conservative reading).

    Returns ``(scores, info)`` where ``info`` carries:
      - ``positions_scored``, ``positions_total``
      - ``chunks_visited``, ``chunks_total``
      - ``cap_fired``: True if cap interrupted stage 2
      - ``stage1_elapsed_s``, ``stage2_elapsed_s``, ``elapsed_s``
      - ``ranked_chunk_indices``: chunks visited in priority order
    """
    if pooling not in {"max", "mean"}:
        raise ValueError(f"pooling must be 'max' or 'mean', got {pooling!r}.")
    if len(evicted_cache) == 0:
        return ({}, {
            "positions_scored": 0,
            "positions_total": 0,
            "chunks_visited": 0,
            "chunks_total": 0,
            "cap_fired": False,
            "stage1_elapsed_s": 0.0,
            "stage2_elapsed_s": 0.0,
            "elapsed_s": 0.0,
            "ranked_chunk_indices": [],
        })
    if not isinstance(query_rows, torch.Tensor) or query_rows.ndim != 4:
        raise ValueError("query_rows must have shape [n_layers, n_query_heads, q_len, head_dim].")

    n_layers = len(evicted_cache.kv)
    if int(query_rows.shape[0]) != n_layers:
        raise ValueError(
            f"query_rows layer count {int(query_rows.shape[0])} does not match cache layers {n_layers}."
        )
    if active_cache is not None and len(active_cache.kv) != n_layers:
        raise ValueError(
            f"active_cache layer count {len(active_cache.kv)} does not match evicted layers {n_layers}."
        )

    overall_start = time.perf_counter()
    deadline = (overall_start + float(wallclock_deadline_s)) if wallclock_deadline_s is not None else None

    if summaries is None or chunk_ranges is None:
        summaries, chunk_ranges = compute_chunk_summaries(evicted_cache, chunk_size=int(chunk_size))
    chunk_ranges = list(chunk_ranges)

    # Stage 1: cheap chunk scoring
    stage1_start = time.perf_counter()
    chunk_scores = _score_chunks_against_summaries(
        query_rows=query_rows,
        summaries=summaries,
    )
    ranked_chunk_indices = chunk_scores.argsort(descending=True).tolist() if chunk_scores.numel() else []
    stage1_elapsed_s = time.perf_counter() - stage1_start

    # Hoist per-layer host-side conversions. Precomputed keys stored at
    # [n_kv_heads, T, head_dim]; head expansion deferred to chunk-time
    # to keep host memory in check.
    from phases.phase6_repair.src.selectors import precompute_host_layer_keys
    n_layers = len(evicted_cache.kv)
    if precomputed_evicted_layer_keys is not None:
        layer_evicted_keys = list(precomputed_evicted_layer_keys)
    else:
        layer_evicted_keys = precompute_host_layer_keys(evicted_cache)
    if precomputed_active_layer_keys is not None:
        layer_active_keys = list(precomputed_active_layer_keys)
    elif active_cache is not None and len(active_cache) > 0:
        layer_active_keys = list(precompute_host_layer_keys(active_cache))
    else:
        layer_active_keys = [None] * n_layers

    def _expand_heads(rows: torch.Tensor, query_heads: int) -> torch.Tensor:
        kv_heads = int(rows.shape[0])
        if query_heads == kv_heads:
            return rows
        if query_heads % kv_heads != 0:
            raise ValueError(
                f"query head count {query_heads} not divisible by kv head count {kv_heads}"
            )
        return rows.repeat_interleave(query_heads // kv_heads, dim=0)

    # Stage 2: expensive per-position scoring of high-priority chunks, with wall-clock cap
    stage2_start = time.perf_counter()
    scores: dict[int, float] = {}
    positions_scored = 0
    chunks_visited = 0
    cap_fired = False

    for chunk_idx in ranked_chunk_indices:
        if deadline is not None and time.perf_counter() >= deadline:
            cap_fired = True
            break
        start, stop = chunk_ranges[chunk_idx]
        chunk_positions = evicted_cache.positions[start:stop]
        # Full-score this chunk (same softmax-over-active+chunk structure as the
        # chunk-position Refresh-K-budgeted scorer for comparability).
        chunk_layer_scores: list[torch.Tensor] = []
        for layer_index in range(int(summaries.shape[0])):
            query_layer = query_rows[layer_index]
            query_heads = int(query_layer.shape[0])
            evicted_key_chunk = _expand_heads(
                layer_evicted_keys[layer_index][:, start:stop, :],
                query_heads,
            )
            active_key_raw = layer_active_keys[layer_index]
            if active_key_raw is not None:
                active_key = _expand_heads(active_key_raw, query_heads)
                score_keys = torch.cat((active_key, evicted_key_chunk), dim=1)
                active_len = int(active_key.shape[1])
            else:
                score_keys = evicted_key_chunk
                active_len = 0
            attn_logits = (
                torch.matmul(query_layer, score_keys.transpose(-2, -1))
                / math.sqrt(float(score_keys.shape[-1]))
            )
            attn = torch.softmax(attn_logits, dim=-1)
            if active_len:
                attn = attn[:, :, active_len:]
            if pooling == "max":
                pooled = attn.amax(dim=1).mean(dim=0)
            else:
                pooled = attn.mean(dim=(0, 1))
            chunk_layer_scores.append(pooled)
        chunk_importance = torch.stack(chunk_layer_scores, dim=0).mean(dim=0)
        for dense_index, position in enumerate(chunk_positions):
            scores[int(position)] = float(chunk_importance[dense_index].item())
            positions_scored += 1
        chunks_visited += 1

    stage2_elapsed_s = time.perf_counter() - stage2_start
    elapsed_s = time.perf_counter() - overall_start
    # For audit / disambiguation: also build a "stage-1-only" score map
    # using ONLY the cheap chunk-summary scores broadcast to constituent
    # positions. If the wall-clock cap fires before any expensive Stage 2
    # work, runners can compare "Stage-1-only ranking" vs the empty
    # `scores` dict to decide which fallback is actually being measured.
    stage1_only_scores: dict[int, float] = {}
    if chunk_scores.numel() > 0:
        for chunk_idx_int, (start_pos, stop_pos) in enumerate(chunk_ranges):
            chunk_score = float(chunk_scores[chunk_idx_int].item())
            for pos in evicted_cache.positions[start_pos:stop_pos]:
                stage1_only_scores[int(pos)] = chunk_score

    info: dict[str, int | bool | float] = {
        "positions_scored": int(positions_scored),
        "positions_total": int(len(evicted_cache.positions)),
        "chunks_visited": int(chunks_visited),
        "chunks_total": int(len(chunk_ranges)),
        "cap_fired": bool(cap_fired),
        "stage1_elapsed_s": float(stage1_elapsed_s),
        "stage2_elapsed_s": float(stage2_elapsed_s),
        "elapsed_s": float(elapsed_s),
        "ranked_chunk_indices": [int(i) for i in ranked_chunk_indices[:chunks_visited]],
        "stage1_only_scores": stage1_only_scores,
    }
    return scores, info
