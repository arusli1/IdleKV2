"""Selection and burst-packing helpers for the Phase 6 repair study."""

from __future__ import annotations

import itertools
import math
import random
import time
from typing import Sequence

import torch

from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache


def score_evicted_positions(
    *,
    query_rows: torch.Tensor,
    evicted_cache: PositionTrackedCache,
    active_cache: PositionTrackedCache | None = None,
    pooling: str = "max",
) -> dict[int, float]:
    """Score evicted positions against the active+evicted key set.

    `query_rows` may contain either proxy rows or exact extracted Q projections.
    When `active_cache` is provided, the attention normalization is performed over
    the concatenated active and evicted keys so the returned scores better match
    true decode-time competition for attention mass.
    """
    if pooling not in {"max", "mean"}:
        raise ValueError(f"pooling must be 'max' or 'mean', got {pooling!r}.")
    if len(evicted_cache) == 0:
        return {}
    if not isinstance(query_rows, torch.Tensor) or query_rows.ndim != 4:
        raise ValueError("query_rows must have shape [n_layers, n_query_heads, q_len, head_dim].")

    n_layers = len(evicted_cache.kv)
    if int(query_rows.shape[0]) != n_layers:
        raise ValueError(f"query_rows layer count {int(query_rows.shape[0])} does not match cache layers {n_layers}.")
    if active_cache is not None and len(active_cache.kv) != n_layers:
        raise ValueError(
            f"active_cache layer count {len(active_cache.kv)} does not match evicted cache layers {n_layers}."
        )

    # Optional GPU scoring path (Phase 18 round-5 attack #2 defuse).
    # When `score_on_gpu=True` (env var PHASE18_SCORE_ON_GPU=1 or
    # function attribute _score_on_gpu set), matmul + softmax stay on
    # the model's GPU so the quality numbers come from the same code
    # path as the runtime probe. Default remains CPU for backward
    # compatibility with the existing K-sweep results.
    import os as _os
    _gpu_attr = bool(getattr(score_evicted_positions, "_score_on_gpu", False))
    _gpu_env = _os.environ.get("PHASE18_SCORE_ON_GPU", "0") == "1"
    score_on_gpu = _gpu_attr or _gpu_env
    target_device = (
        evicted_cache.kv[0][0].device if score_on_gpu else torch.device("cpu")
    )

    layer_scores: list[torch.Tensor] = []
    for layer_index, (key, _) in enumerate(evicted_cache.kv):
        query_layer = query_rows[layer_index]
        query_heads = int(query_layer.shape[0])
        if score_on_gpu:
            evicted_key_float = key.detach().to(device=target_device, dtype=torch.float32)[0]
            query_layer_local = query_layer.to(device=target_device, dtype=torch.float32)
        else:
            evicted_key_float = key.detach().to("cpu", dtype=torch.float32)[0]
            query_layer_local = query_layer

        def _repeat_heads(key_rows: torch.Tensor) -> torch.Tensor:
            key_heads = int(key_rows.shape[0])
            if query_heads == key_heads:
                return key_rows
            if query_heads % key_heads != 0:
                raise ValueError(
                    f"query_rows head count {query_heads} is not compatible with cache heads {key_heads}."
                )
            return key_rows.repeat_interleave(query_heads // key_heads, dim=0)

        evicted_key_float = _repeat_heads(evicted_key_float)
        score_keys = evicted_key_float
        active_len = 0
        if active_cache is not None and len(active_cache) > 0:
            if score_on_gpu:
                active_key_float = active_cache.kv[layer_index][0].detach().to(
                    device=target_device, dtype=torch.float32
                )[0]
            else:
                active_key_float = active_cache.kv[layer_index][0].detach().to("cpu", dtype=torch.float32)[0]
            active_key_float = _repeat_heads(active_key_float)
            active_len = int(active_key_float.shape[1])
            score_keys = torch.cat((active_key_float, evicted_key_float), dim=1)

        scores = torch.matmul(query_layer_local, score_keys.transpose(-2, -1)) / math.sqrt(float(score_keys.shape[-1]))
        scores = torch.softmax(scores, dim=-1)
        if active_len:
            scores = scores[:, :, active_len:]
        if pooling == "max":
            pooled = scores.amax(dim=1).mean(dim=0)
        else:
            pooled = scores.mean(dim=(0, 1))
        # Bring back to CPU for the final dict; the matmul/softmax was on GPU.
        layer_scores.append(pooled.detach().to("cpu"))

    importance = torch.stack(layer_scores, dim=0).mean(dim=0)
    return {
        int(position): float(importance[dense_index].item())
        for dense_index, position in enumerate(evicted_cache.positions)
    }


def precompute_host_layer_keys(
    cache: PositionTrackedCache,
    *,
    query_heads_per_layer: list[int] | None = None,
) -> list[torch.Tensor]:
    """Convert a PositionTrackedCache to a list of host-FP32 key tensors.

    Returns a list of ``[n_kv_heads, T, head_dim]`` tensors, ONE per
    layer. We deliberately do NOT pre-expand grouped-attention heads
    here -- pre-expanding 4 KV heads -> 28 query heads on a 32K-position
    cache produces ~13 GB of host memory per cache (28 layers x 470 MB),
    which on a 64 GB host swaps under PageSummary + Refresh-K-budgeted
    + Refresh-K (unbudgeted) co-precomputes. Expand at score time
    instead -- the expansion is cheap relative to the matmul.

    The ``query_heads_per_layer`` argument is accepted for backward
    compatibility but ignored.
    """
    del query_heads_per_layer  # unused; expansion deferred to scorer
    layer_keys: list[torch.Tensor] = []
    for layer_index, (key, _) in enumerate(cache.kv):
        host_key = key.detach().to("cpu", dtype=torch.float32)[0]
        layer_keys.append(host_key)
    return layer_keys


def score_evicted_positions_budgeted(
    *,
    query_rows: torch.Tensor,
    evicted_cache: PositionTrackedCache,
    active_cache: PositionTrackedCache | None = None,
    pooling: str = "max",
    wallclock_deadline_s: float | None = None,
    position_chunk_size: int = 1024,
    precomputed_evicted_layer_keys: list[torch.Tensor] | None = None,
    precomputed_active_layer_keys: list[torch.Tensor | None] | None = None,
) -> tuple[dict[int, float], dict[str, int | bool]]:
    """Score evicted positions with an optional per-call wall-clock cap.

    Iterates the position dimension in chunks of ``position_chunk_size``;
    inside each chunk, scores are computed across all layers (full
    layer-aggregation per position). The wall-clock cap is checked between
    position-chunks, so a scored position is always *fully* aggregated
    across layers, and unscored positions are simply absent from the
    returned dict (cleaner semantics than partial layer aggregation).

    NOTE on softmax normalization. The unbudgeted ``score_evicted_positions``
    softmaxes over (active + ALL evicted) keys; this scorer softmaxes over
    (active + CHUNK_evicted) keys. The two are equivalent when
    ``position_chunk_size >= n_evicted`` (single chunk), and differ
    otherwise because the global softmax denominator is incomputable
    without scoring every position. Cross-chunk score magnitudes are
    therefore not directly comparable -- but rankings within each chunk
    match the unbudgeted scorer restricted to that chunk, and selecting
    top-``K`` across the union of scored chunks is still a defensible
    "score-by-attention-evidence" algorithm. Document this choice in the
    paper W4.1 novelty paragraph.

    For Phase 18 W1, this is the basis of Refresh-K-budgeted: same scorer
    as the unbudgeted Refresh-K, but stops scoring at wall-clock T_repair.
    Positions that did not get scored fall back to the runner's existing
    tiebreaker (zero-score + ascending position).

    Returns ``(scores, info)`` where ``info`` carries:
      - ``positions_scored``: how many positions got real scores
      - ``positions_total``: how many positions exist in the evicted cache
      - ``layer_chunks_completed``: completed (chunk, layer) pairs (audit)
      - ``cap_fired``: True if wall-clock cap interrupted scoring
      - ``elapsed_s``: total scoring time in seconds
    """
    if pooling not in {"max", "mean"}:
        raise ValueError(f"pooling must be 'max' or 'mean', got {pooling!r}.")
    if int(position_chunk_size) <= 0:
        raise ValueError(f"position_chunk_size must be positive, got {position_chunk_size}.")
    if len(evicted_cache) == 0:
        return ({}, {
            "positions_scored": 0,
            "positions_total": 0,
            "layer_chunks_completed": 0,
            "cap_fired": False,
            "elapsed_s": 0.0,
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
            f"active_cache layer count {len(active_cache.kv)} does not match evicted cache layers {n_layers}."
        )

    n_evicted = len(evicted_cache.positions)
    chunk_starts = list(range(0, n_evicted, int(position_chunk_size)))
    deadline = (time.perf_counter() + float(wallclock_deadline_s)) if wallclock_deadline_s is not None else None

    # Hoist per-layer host-side conversions. If the caller pre-computed
    # them (Phase 18 W1 K-loop optimization), reuse to avoid Nx redundant
    # CPU<-GPU copies per example. Precomputed keys are stored at
    # [n_kv_heads, T, head_dim] -- head expansion is done lazily inside
    # the chunk loop so we don't blow up host memory by 7x.
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

    scores: dict[int, float] = {}
    layer_chunks_completed = 0
    positions_scored = 0
    cap_fired = False
    start_time = time.perf_counter()

    for chunk_start in chunk_starts:
        # cap check fires BETWEEN chunks (and before the first chunk past
        # the deadline). Each scored chunk gets its full layer aggregation.
        if deadline is not None and time.perf_counter() >= deadline:
            cap_fired = True
            break
        chunk_stop = min(chunk_start + int(position_chunk_size), n_evicted)
        chunk_positions = evicted_cache.positions[chunk_start:chunk_stop]
        chunk_layer_scores: list[torch.Tensor] = []
        for layer_index in range(n_layers):
            query_layer = query_rows[layer_index]
            query_heads = int(query_layer.shape[0])
            evicted_key_chunk = _expand_heads(
                layer_evicted_keys[layer_index][:, chunk_start:chunk_stop, :],
                query_heads,
            )
            active_key_raw = layer_active_keys[layer_index]
            if active_key_raw is not None:
                active_key = _expand_heads(active_key_raw, query_heads)
                # IMPORTANT: softmax over active+chunk keys reflects the same
                # competition for attention mass as the unbudgeted scorer at
                # this chunk's slice. Across chunks the normalizer differs
                # (each chunk competes with active separately), but inside a
                # chunk it is consistent.
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
            layer_chunks_completed += 1

        # Average across all layers for this chunk -- full aggregation.
        chunk_importance = torch.stack(chunk_layer_scores, dim=0).mean(dim=0)
        for dense_index, position in enumerate(chunk_positions):
            scores[int(position)] = float(chunk_importance[dense_index].item())
            positions_scored += 1

    elapsed_s = time.perf_counter() - start_time
    info: dict[str, int | bool] = {
        "positions_scored": int(positions_scored),
        "positions_total": int(n_evicted),
        "layer_chunks_completed": int(layer_chunks_completed),
        "cap_fired": bool(cap_fired),
        "elapsed_s": float(elapsed_s),
    }
    return scores, info


def rank_positions(
    positions: Sequence[int],
    *,
    primary_scores: dict[int, float],
    secondary_scores: dict[int, float] | None = None,
) -> list[int]:
    """Sort positions by descending primary/secondary score and ascending position."""
    return sorted(
        (int(position) for position in positions),
        key=lambda position: (
            -float(primary_scores.get(int(position), 0.0)),
            -float((secondary_scores or {}).get(int(position), 0.0)),
            int(position),
        ),
    )


def contrastive_position_scores(
    positions: Sequence[int],
    *,
    positive_scores: dict[int, float],
    negative_scores: dict[int, float],
    alpha: float = 1.0,
) -> dict[int, float]:
    """Return standardized positive-minus-negative scores over the candidate set."""
    candidate_positions = [int(position) for position in positions]
    if not candidate_positions:
        return {}

    def _z_scores(scores: dict[int, float]) -> dict[int, float]:
        values = [float(scores.get(position, 0.0)) for position in candidate_positions]
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        if std <= 1e-12:
            return {position: 0.0 for position in candidate_positions}
        return {
            position: (float(scores.get(position, 0.0)) - mean_value) / std
            for position in candidate_positions
        }

    positive_z = _z_scores(positive_scores)
    negative_z = _z_scores(negative_scores)
    return {
        position: float(positive_z[position] - float(alpha) * negative_z[position])
        for position in candidate_positions
    }


def pack_anchor_bursts(
    *,
    anchor_positions: Sequence[int],
    available_positions: Sequence[int],
    k: int,
    left: int,
    right: int,
    backfill_positions: Sequence[int] | None = None,
) -> list[int]:
    """Expand anchors into local bursts, then backfill singles until exactly K positions are selected."""
    target_k = max(int(k), 0)
    if target_k == 0:
        return []

    available = {int(position) for position in available_positions}
    selected: list[int] = []
    selected_set: set[int] = set()

    def _try_add(position: int) -> bool:
        candidate = int(position)
        if candidate not in available or candidate in selected_set:
            return False
        selected.append(candidate)
        selected_set.add(candidate)
        return len(selected) >= target_k

    for anchor in anchor_positions:
        anchor_int = int(anchor)
        for position in range(anchor_int - int(left), anchor_int + int(right) + 1):
            if _try_add(position):
                return selected

    for position in backfill_positions or ():
        if _try_add(int(position)):
            return selected

    return selected


def select_idlekv_positions(
    *,
    evicted_positions: Sequence[int],
    q2_scores: dict[int, float],
    turn_n_scores: dict[int, float],
    k: int,
    left: int,
    right: int,
) -> list[int]:
    """Select K positions by Q2 score, then turn-N score, then burst-pack locally."""
    ranked = rank_positions(
        evicted_positions,
        primary_scores=q2_scores,
        secondary_scores=turn_n_scores,
    )
    return pack_anchor_bursts(
        anchor_positions=ranked,
        available_positions=evicted_positions,
        k=int(k),
        left=int(left),
        right=int(right),
        backfill_positions=ranked,
    )


def _score_with_secondary(
    position: int,
    *,
    primary_scores: dict[int, float],
    secondary_scores: dict[int, float],
    secondary_weight: float = 1e-6,
) -> float:
    return float(primary_scores.get(position, 0.0)) + float(secondary_weight) * float(
        secondary_scores.get(position, 0.0)
    )


def select_coverage_aware_positions(
    *,
    evicted_positions: Sequence[int],
    q2_scores: dict[int, float],
    turn_n_scores: dict[int, float],
    k: int,
    left: int,
    right: int,
) -> list[int]:
    """Greedily choose burst windows by marginal score coverage.

    The default selector ranks anchors first and then expands each anchor into a
    burst. This variant ranks each candidate burst by the score it adds beyond
    already selected rows, which avoids wasting restore slots when the highest
    scoring anchors are adjacent and their bursts mostly overlap.
    """
    target_k = max(int(k), 0)
    if target_k == 0:
        return []

    available = sorted(dict.fromkeys(int(position) for position in evicted_positions))
    if not available:
        return []
    available_set = set(available)
    selected: list[int] = []
    selected_set: set[int] = set()

    def _window(anchor: int) -> list[int]:
        return [
            position
            for position in range(int(anchor) - int(left), int(anchor) + int(right) + 1)
            if position in available_set
        ]

    ranked_backfill = rank_positions(
        available,
        primary_scores=q2_scores,
        secondary_scores=turn_n_scores,
    )
    anchors = ranked_backfill
    while len(selected) < target_k:
        best_anchor: int | None = None
        best_new_positions: list[int] = []
        best_key: tuple[float, int, float, float, int] | None = None
        for anchor in anchors:
            new_positions = [position for position in _window(anchor) if position not in selected_set]
            if not new_positions:
                continue
            marginal_score = sum(
                _score_with_secondary(
                    position,
                    primary_scores=q2_scores,
                    secondary_scores=turn_n_scores,
                )
                for position in new_positions
            )
            key = (
                float(marginal_score),
                len(new_positions),
                float(q2_scores.get(anchor, 0.0)),
                float(turn_n_scores.get(anchor, 0.0)),
                -int(anchor),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_anchor = int(anchor)
                best_new_positions = new_positions
        if best_anchor is None:
            break
        del best_anchor
        for position in best_new_positions:
            if position in selected_set:
                continue
            selected.append(position)
            selected_set.add(position)
            if len(selected) >= target_k:
                return selected

    for position in ranked_backfill:
        if position in selected_set:
            continue
        selected.append(int(position))
        selected_set.add(int(position))
        if len(selected) >= target_k:
            break
    return selected


def select_mmr_positions(
    *,
    evicted_positions: Sequence[int],
    q2_scores: dict[int, float],
    turn_n_scores: dict[int, float],
    k: int,
    left: int,
    right: int,
    diversity_weight: float = 0.25,
) -> list[int]:
    """Select burst anchors with a small diversity bonus between anchors."""
    target_k = max(int(k), 0)
    if target_k == 0:
        return []

    available = sorted(dict.fromkeys(int(position) for position in evicted_positions))
    if not available:
        return []
    base_rank = rank_positions(
        available,
        primary_scores=q2_scores,
        secondary_scores=turn_n_scores,
    )
    if len(base_rank) <= 1:
        return pack_anchor_bursts(
            anchor_positions=base_rank,
            available_positions=available,
            k=target_k,
            left=int(left),
            right=int(right),
            backfill_positions=base_rank,
        )

    score_values = [float(q2_scores.get(position, 0.0)) for position in available]
    score_min = min(score_values)
    score_range = max(score_values) - score_min
    distance_scale = max(max(available) - min(available), 1)
    selected_anchors: list[int] = []
    remaining = set(available)
    selected_positions: list[int] = []

    def _normalized_score(position: int) -> float:
        if score_range <= 1e-12:
            return 0.0
        return (float(q2_scores.get(position, 0.0)) - score_min) / score_range

    while remaining and len(selected_positions) < target_k:
        best_position: int | None = None
        best_key: tuple[float, float, float, int] | None = None
        for position in remaining:
            if selected_anchors:
                nearest = min(abs(int(position) - anchor) for anchor in selected_anchors)
                diversity = min(float(nearest) / float(distance_scale), 1.0)
            else:
                diversity = 0.0
            key = (
                _normalized_score(position) + float(diversity_weight) * diversity,
                float(q2_scores.get(position, 0.0)),
                float(turn_n_scores.get(position, 0.0)),
                -int(position),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_position = int(position)
        assert best_position is not None
        selected_anchors.append(best_position)
        remaining.remove(best_position)
        selected_positions = pack_anchor_bursts(
            anchor_positions=selected_anchors,
            available_positions=available,
            k=target_k,
            left=int(left),
            right=int(right),
            backfill_positions=(),
        )

    return pack_anchor_bursts(
        anchor_positions=selected_anchors,
        available_positions=available,
        k=target_k,
        left=int(left),
        right=int(right),
        backfill_positions=base_rank,
    )


def select_refresh_positions(
    *,
    context_positions: Sequence[int],
    mandatory_positions: Sequence[int],
    q2_scores: dict[int, float],
    turn_n_scores: dict[int, float],
    context_budget: int,
    left: int,
    right: int,
) -> list[int]:
    """Reselect the full resumed context budget using the Q2-time signal.

    This is a stronger bounded comparator than IdleKV: it may drop stale rows
    from the base compressed cache and choose any buffered context row, while
    still preserving mandatory sink/recency rows and the same total active
    context budget.
    """
    available = sorted(dict.fromkeys(int(position) for position in context_positions))
    available_set = set(available)
    target_budget = min(max(int(context_budget), 0), len(available))
    if target_budget == 0:
        return []

    mandatory = [
        int(position)
        for position in dict.fromkeys(int(position) for position in mandatory_positions)
        if int(position) in available_set
    ][:target_budget]
    mandatory_set = set(mandatory)
    remaining_slots = max(0, target_budget - len(mandatory))
    if remaining_slots == 0:
        return sorted(mandatory)

    candidates = [position for position in available if position not in mandatory_set]
    ranked = rank_positions(
        candidates,
        primary_scores=q2_scores,
        secondary_scores=turn_n_scores,
    )
    selected = pack_anchor_bursts(
        anchor_positions=ranked,
        available_positions=candidates,
        k=remaining_slots,
        left=int(left),
        right=int(right),
        backfill_positions=ranked,
    )
    return sorted(set(mandatory) | set(selected))


def select_random_positions(
    *,
    evicted_positions: Sequence[int],
    k: int,
    left: int,
    right: int,
    seed: int,
) -> list[int]:
    """Deterministic random ablation with the same burst-packing rule."""
    rng = random.Random(int(seed))
    anchors = [int(position) for position in evicted_positions]
    rng.shuffle(anchors)
    return pack_anchor_bursts(
        anchor_positions=anchors,
        available_positions=evicted_positions,
        k=int(k),
        left=int(left),
        right=int(right),
        backfill_positions=anchors,
    )


def select_oldest_positions(
    *,
    evicted_positions: Sequence[int],
    k: int,
    left: int,
    right: int,
) -> list[int]:
    """Recency-free baseline: prefer the oldest evicted positions first."""
    anchors = sorted(int(position) for position in evicted_positions)
    return pack_anchor_bursts(
        anchor_positions=anchors,
        available_positions=evicted_positions,
        k=int(k),
        left=int(left),
        right=int(right),
        backfill_positions=anchors,
    )


def select_oracle_positions(
    *,
    evicted_positions: Sequence[int],
    relevant_positions: Sequence[int],
    relevant_position_groups: Sequence[Sequence[int]] | None = None,
    q2_scores: dict[int, float],
    turn_n_scores: dict[int, float],
    k: int,
    left: int,
    right: int,
) -> list[int]:
    """Gold-span hindsight selector.

    When span groups are supplied, this performs an exact search over the small
    set of gold span groups to maximize complete recovered groups first, then
    recovered gold tokens, before score-ranked backfill.
    """
    del left, right
    available = {int(position) for position in evicted_positions}
    relevant_groups = [
        tuple(sorted(int(position) for position in group if int(position) in available))
        for group in (relevant_position_groups or ())
    ]
    relevant_groups = [group for group in relevant_groups if group]

    selected_relevant: list[int] = []
    if relevant_groups:
        best_value: tuple[int, int, float, float] = (-1, -1, float("-inf"), float("-inf"))
        best_selected: tuple[int, ...] = ()
        for subset_bits in itertools.product((0, 1), repeat=len(relevant_groups)):
            chosen_groups = [group for keep, group in zip(subset_bits, relevant_groups) if keep]
            chosen_positions = tuple(sorted({position for group in chosen_groups for position in group}))
            cost = len(chosen_positions)
            if cost > int(k):
                continue
            score_value = (
                len(chosen_groups),
                len(chosen_positions),
                float(sum(q2_scores.get(position, 0.0) for position in chosen_positions)),
                float(sum(turn_n_scores.get(position, 0.0) for position in chosen_positions)),
            )
            if score_value > best_value:
                best_value = score_value
                best_selected = chosen_positions
        selected_relevant = rank_positions(
            best_selected,
            primary_scores=q2_scores,
            secondary_scores=turn_n_scores,
        )
    else:
        selected_relevant = rank_positions(
            (int(position) for position in relevant_positions if int(position) in available),
            primary_scores=q2_scores,
            secondary_scores=turn_n_scores,
        )[: int(k)]

    selected_set = set(selected_relevant)
    remaining_relevant = rank_positions(
        (position for position in relevant_positions if int(position) in available and int(position) not in selected_set),
        primary_scores=q2_scores,
        secondary_scores=turn_n_scores,
    )
    backfill = rank_positions(
        (position for position in evicted_positions if int(position) not in selected_set),
        primary_scores=q2_scores,
        secondary_scores=turn_n_scores,
    )
    selected = list(selected_relevant)
    for position in itertools.chain(remaining_relevant, backfill):
        if len(selected) >= int(k):
            break
        if int(position) in selected_set:
            continue
        selected.append(int(position))
        selected_set.add(int(position))
    return selected
