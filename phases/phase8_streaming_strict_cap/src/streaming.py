"""Strict-cap streaming and bounded CPU-spill helpers for Phase 8."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Iterable, Literal, Sequence

import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache, inject_kv, slice_kv, to_dynamic_cache, to_tuple_cache
from phases.phase2_kv_cache.src.runtime import model_device
from phases.phase3_eviction.src.eviction import SnapKV


SpillPolicy = Literal["qnorm", "random"]


@dataclass(frozen=True)
class StreamingGeometry:
    """Pure geometry summary for the strict-cap streaming schedule."""

    total_context_length: int
    chunk_size: int
    context_cap: int
    keep_fraction: float
    final_active_tokens: int
    eviction_events: int
    peak_active_tokens: int
    active_after_evictions: tuple[int, ...]


@dataclass
class SpillBuffer:
    """CPU-side spill cache plus per-position eviction-time scores."""

    cache: PositionTrackedCache | None = None
    qnorm_scores: dict[int, float] | None = None

    def __post_init__(self) -> None:
        if self.qnorm_scores is None:
            self.qnorm_scores = {}

    @property
    def positions(self) -> tuple[int, ...]:
        if self.cache is None:
            return ()
        return tuple(int(position) for position in self.cache.positions)

    def __len__(self) -> int:
        return 0 if self.cache is None else len(self.cache)


@dataclass(frozen=True)
class StreamingPrefillResult:
    """Final active cache and bounded spill buffers after streaming context prefill."""

    active_cache: PositionTrackedCache
    qnorm_spill: SpillBuffer
    random_spill: SpillBuffer
    qnorm_by_position: dict[int, float]
    eviction_events: int
    final_active_context_tokens: int
    peak_active_context_tokens: int
    spill_selection_s: float
    stream_prefill_s: float
    event_summaries: tuple[dict[str, int | float], ...]


@dataclass(frozen=True)
class QnormSpillSweepResult:
    """Coverage-only result for top-X query-norm spill diagnostics."""

    final_active_context_tokens: int
    peak_active_context_tokens: int
    eviction_events: int
    total_evicted_tokens: int
    stream_prefill_s: float
    spill_positions_by_fraction: dict[float, tuple[int, ...]]
    qnorm_rank_by_position: dict[int, tuple[int, int]]
    event_summaries: tuple[dict[str, int | float], ...]


@dataclass(frozen=True)
class TwoTierSnapKVSpillResult:
    """Coverage-only result for two-tier SnapKV GPU/CPU partitioning."""

    active_cache: PositionTrackedCache
    cpu_spill_positions: tuple[int, ...]
    permanent_evicted_tokens: int
    final_active_context_tokens: int
    peak_active_context_tokens: int
    eviction_events: int
    total_cpu_spill_tokens: int
    stream_prefill_s: float
    event_summaries: tuple[dict[str, int | float], ...]


def two_tier_candidate_budget(
    *,
    pre_evict_tokens: int,
    gpu_keep_budget: int,
    cpu_store_fraction: float,
) -> tuple[int, int]:
    """Return candidate-pool and CPU-tier budgets for two-tier SnapKV.

    `cpu_store_fraction` is interpreted as a fraction of the tokens that would
    be evicted by a GPU-only compression from `pre_evict_tokens` to
    `gpu_keep_budget`.
    """
    if int(pre_evict_tokens) <= 0:
        raise ValueError("pre_evict_tokens must be positive.")
    if int(gpu_keep_budget) <= 0 or int(gpu_keep_budget) > int(pre_evict_tokens):
        raise ValueError("gpu_keep_budget must lie in [1, pre_evict_tokens].")
    if not 0.0 <= float(cpu_store_fraction) <= 1.0:
        raise ValueError("cpu_store_fraction must lie in [0, 1].")
    gpu_only_evicted = int(pre_evict_tokens) - int(gpu_keep_budget)
    cpu_budget = int(math.ceil(gpu_only_evicted * float(cpu_store_fraction)))
    candidate_budget = min(int(pre_evict_tokens), int(gpu_keep_budget) + cpu_budget)
    return candidate_budget, cpu_budget


def simulate_streaming_geometry(
    *,
    total_context_length: int,
    chunk_size: int,
    context_cap: int,
    keep_fraction: float,
) -> StreamingGeometry:
    """Return the expected active-cache length schedule without model execution."""
    if total_context_length <= 0:
        raise ValueError("total_context_length must be positive.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if context_cap <= 0:
        raise ValueError("context_cap must be positive.")
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must lie in (0, 1].")

    active = 0
    peak = 0
    events = 0
    after_evictions: list[int] = []
    remaining = int(total_context_length)
    while remaining > 0:
        current_chunk = min(int(chunk_size), remaining)
        if active + current_chunk > int(context_cap):
            active = max(1, int(math.ceil(active * float(keep_fraction))))
            events += 1
            after_evictions.append(active)
            if active + current_chunk > int(context_cap):
                raise ValueError(
                    "chunk_size is too large for this cap/keep_fraction schedule: "
                    f"active_after_eviction={active}, chunk={current_chunk}, cap={context_cap}."
                )
        active += current_chunk
        peak = max(peak, active)
        remaining -= current_chunk

    return StreamingGeometry(
        total_context_length=int(total_context_length),
        chunk_size=int(chunk_size),
        context_cap=int(context_cap),
        keep_fraction=float(keep_fraction),
        final_active_tokens=int(active),
        eviction_events=int(events),
        peak_active_tokens=int(peak),
        active_after_evictions=tuple(after_evictions),
    )


def assert_strict_cap(cache: PositionTrackedCache, *, cap: int, label: str) -> None:
    """Fail loudly if any active cache mutation exceeds the configured cap."""
    if len(cache) > int(cap):
        raise RuntimeError(f"{label} exceeded strict cache cap: {len(cache)} > {int(cap)}.")


def dense_indices_for_positions(cache: PositionTrackedCache, positions: Iterable[int]) -> list[int]:
    """Map absolute positions to dense cache indices, preserving cache order."""
    wanted = {int(position) for position in positions}
    return [index for index, position in enumerate(cache.positions) if int(position) in wanted]


def slice_cache_by_positions(cache: PositionTrackedCache, positions: Iterable[int]) -> PositionTrackedCache | None:
    """Slice a tracked cache by absolute positions."""
    indices = dense_indices_for_positions(cache, positions)
    if not indices:
        return None
    result = slice_kv(cache, indices)
    if not isinstance(result, PositionTrackedCache):
        raise RuntimeError("slice_kv did not preserve position tracking.")
    return result


def cache_to_cpu(cache: PositionTrackedCache) -> PositionTrackedCache:
    """Move a tracked cache to CPU without changing positions."""
    return cache.to_device("cpu")


def append_spill_fragment(
    spill: SpillBuffer,
    fragment: PositionTrackedCache | None,
    *,
    qnorm_scores: dict[int, float],
    hard_cap: int | None = None,
) -> SpillBuffer:
    """Append a CPU spill fragment and optionally trim by qnorm."""
    if fragment is None or len(fragment) == 0:
        return spill

    fragment_cpu = cache_to_cpu(fragment)
    if spill.cache is None:
        merged_cache = fragment_cpu
    else:
        merged_cache = inject_kv(spill.cache, fragment_cpu, fragment_cpu.positions)

    merged_scores = dict(spill.qnorm_scores or {})
    for position in fragment_cpu.positions:
        merged_scores[int(position)] = float(qnorm_scores.get(int(position), 0.0))

    merged = SpillBuffer(cache=merged_cache, qnorm_scores=merged_scores)
    if hard_cap is not None and len(merged) > int(hard_cap):
        return trim_spill_to_cap(merged, cap=int(hard_cap))
    return merged


def trim_spill_to_cap(spill: SpillBuffer, *, cap: int) -> SpillBuffer:
    """Keep the globally highest-qnorm spilled positions under a hard cap."""
    if spill.cache is None or len(spill) <= int(cap):
        return spill
    if cap <= 0:
        return SpillBuffer()
    ranked = sorted(
        spill.cache.positions,
        key=lambda position: (-float((spill.qnorm_scores or {}).get(int(position), 0.0)), int(position)),
    )
    keep_positions = set(int(position) for position in ranked[: int(cap)])
    trimmed_cache = slice_cache_by_positions(spill.cache, keep_positions)
    if trimmed_cache is None:
        return SpillBuffer()
    return SpillBuffer(
        cache=trimmed_cache,
        qnorm_scores={position: float((spill.qnorm_scores or {}).get(position, 0.0)) for position in trimmed_cache.positions},
    )


def select_spill_positions(
    *,
    evicted_cache: PositionTrackedCache,
    qnorm_scores: dict[int, float],
    fraction: float,
    policy: SpillPolicy,
    seed: int,
) -> list[int]:
    """Select the bounded CPU spill subset from one eviction event."""
    if len(evicted_cache) == 0:
        return []
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("spill fraction must lie in (0, 1].")
    keep_count = max(1, int(math.ceil(len(evicted_cache) * float(fraction))))
    positions = [int(position) for position in evicted_cache.positions]
    if policy == "qnorm":
        return sorted(
            positions,
            key=lambda position: (-float(qnorm_scores.get(int(position), 0.0)), int(position)),
        )[:keep_count]
    if policy == "random":
        rng = random.Random(int(seed))
        shuffled = list(positions)
        rng.shuffle(shuffled)
        return sorted(shuffled[:keep_count])
    raise ValueError(f"Unsupported spill policy: {policy!r}.")


def score_cache_positions(
    *,
    query_rows: torch.Tensor,
    target_cache: PositionTrackedCache,
    competitor_cache: PositionTrackedCache | None = None,
    pooling: str = "max",
) -> dict[int, float]:
    """Score target cache positions by exact query-row attention against optional competitors."""
    from phases.phase6_repair.src.selectors import score_evicted_positions

    return score_evicted_positions(
        query_rows=query_rows,
        evicted_cache=target_cache,
        active_cache=competitor_cache,
        pooling=pooling,
    )


def choose_lowest_score_positions(
    *,
    positions: Sequence[int],
    scores: dict[int, float],
    k: int,
) -> list[int]:
    """Choose the lowest-scored active positions to swap out."""
    return sorted(
        (int(position) for position in positions),
        key=lambda position: (float(scores.get(int(position), 0.0)), int(position)),
    )[: max(0, int(k))]


def swap_restore_positions(
    *,
    active_cache: PositionTrackedCache,
    spill_cache: PositionTrackedCache,
    restore_positions: Sequence[int],
    drop_positions: Sequence[int],
    cap: int,
) -> PositionTrackedCache:
    """Swap spilled tokens into the active cache while preserving final footprint."""
    restore = [int(position) for position in restore_positions]
    drop = {int(position) for position in drop_positions}
    if len(restore) != len(drop):
        raise ValueError(f"restore/drop counts must match, got {len(restore)} and {len(drop)}.")
    if not restore:
        return active_cache

    keep_indices = [index for index, position in enumerate(active_cache.positions) if int(position) not in drop]
    if len(keep_indices) != len(active_cache) - len(drop):
        raise ValueError("Some requested drop positions were not present in the active cache.")
    active_after_drop = slice_kv(active_cache, keep_indices)
    if not isinstance(active_after_drop, PositionTrackedCache):
        raise RuntimeError("slice_kv did not preserve position tracking.")

    fragment = slice_cache_by_positions(spill_cache, restore)
    if fragment is None or len(fragment) != len(restore):
        raise ValueError("Some requested restore positions were not present in the spill cache.")
    fragment_gpu = fragment.to_device(active_cache.device, non_blocking=True)
    repaired = inject_kv(active_after_drop, fragment_gpu, fragment_gpu.positions)
    assert_strict_cap(repaired, cap=cap, label="repaired cache")
    return repaired


def extract_chunk_qnorms(
    model,
    hidden_states,
    *,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token post-RoPE Q-vector L2 norms from model hidden states."""
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states for qnorm extraction.")

    layer_norms: list[torch.Tensor] = []
    for layer_index, layer in enumerate(model.model.layers):
        layer_input = hidden_states[layer_index]
        normalized = layer.input_layernorm(layer_input)
        attention = layer.self_attn
        query = attention.q_proj(normalized).view(*normalized.shape[:-1], -1, attention.head_dim).transpose(1, 2)
        key = attention.k_proj(normalized).view(*normalized.shape[:-1], -1, attention.head_dim).transpose(1, 2)
        cos, sin = model.model.rotary_emb(normalized, position_ids)
        query, _ = apply_rotary_pos_emb(query, key, cos, sin)
        layer_norms.append(query[0].detach().to("cpu", dtype=torch.float32).norm(dim=-1))

    # [layers, query_heads, seq] -> [seq]
    return torch.stack(layer_norms, dim=0).mean(dim=(0, 1)).contiguous()


def append_context_chunk_with_qnorm(
    *,
    model,
    active_cache: PositionTrackedCache | None,
    chunk_ids: torch.Tensor,
    logical_position_base: int,
) -> tuple[PositionTrackedCache, dict[int, float]]:
    """Append one context chunk and return exact post-RoPE Q-norms for new tokens."""
    device = model_device(model)
    chunk_ids = chunk_ids.to(device)
    if chunk_ids.ndim == 1:
        chunk_ids = chunk_ids.unsqueeze(0)
    chunk_len = int(chunk_ids.shape[1])
    if chunk_len <= 0:
        raise ValueError("chunk_ids must contain at least one token.")

    dense_base = 0 if active_cache is None else len(active_cache)
    prior_positions = [] if active_cache is None else list(active_cache.positions)
    position_ids = torch.arange(
        int(logical_position_base),
        int(logical_position_base) + chunk_len,
        device=device,
    ).unsqueeze(0)
    cache_position = torch.arange(dense_base, dense_base + chunk_len, device=device)

    kwargs = {
        "input_ids": chunk_ids,
        "position_ids": position_ids,
        "cache_position": cache_position,
        "use_cache": True,
        "output_hidden_states": True,
        "logits_to_keep": 1,
    }
    if active_cache is not None:
        kwargs["past_key_values"] = to_dynamic_cache(active_cache, config=model.config)

    with torch.no_grad():
        try:
            outputs = model(**kwargs)
        except TypeError:
            kwargs.pop("logits_to_keep", None)
            try:
                outputs = model(**kwargs)
            except TypeError:
                kwargs.pop("cache_position", None)
                outputs = model(**kwargs)

    qnorm_tensor = extract_chunk_qnorms(model, outputs.hidden_states, position_ids=position_ids)
    new_positions = list(range(int(logical_position_base), int(logical_position_base) + chunk_len))
    qnorm_scores = {
        int(position): float(qnorm_tensor[offset].item())
        for offset, position in enumerate(new_positions)
    }
    tracked = PositionTrackedCache(to_tuple_cache(outputs.past_key_values), prior_positions + new_positions)
    return tracked, qnorm_scores


def append_context_chunk(
    *,
    model,
    active_cache: PositionTrackedCache | None,
    chunk_ids: torch.Tensor,
    logical_position_base: int,
) -> PositionTrackedCache:
    """Append one context chunk without hidden-state/qnorm extraction."""
    device = model_device(model)
    chunk_ids = chunk_ids.to(device)
    if chunk_ids.ndim == 1:
        chunk_ids = chunk_ids.unsqueeze(0)
    chunk_len = int(chunk_ids.shape[1])
    if chunk_len <= 0:
        raise ValueError("chunk_ids must contain at least one token.")

    dense_base = 0 if active_cache is None else len(active_cache)
    prior_positions = [] if active_cache is None else list(active_cache.positions)
    position_ids = torch.arange(
        int(logical_position_base),
        int(logical_position_base) + chunk_len,
        device=device,
    ).unsqueeze(0)
    cache_position = torch.arange(dense_base, dense_base + chunk_len, device=device)

    kwargs = {
        "input_ids": chunk_ids,
        "position_ids": position_ids,
        "cache_position": cache_position,
        "use_cache": True,
        "logits_to_keep": 1,
    }
    if active_cache is not None:
        kwargs["past_key_values"] = to_dynamic_cache(active_cache, config=model.config)

    with torch.no_grad():
        try:
            outputs = model(**kwargs)
        except TypeError:
            kwargs.pop("logits_to_keep", None)
            try:
                outputs = model(**kwargs)
            except TypeError:
                kwargs.pop("cache_position", None)
                outputs = model(**kwargs)

    new_positions = list(range(int(logical_position_base), int(logical_position_base) + chunk_len))
    return PositionTrackedCache(to_tuple_cache(outputs.past_key_values), prior_positions + new_positions)


def stream_context_with_spill(
    *,
    model,
    context_ids: torch.Tensor,
    total_context_length: int,
    chunk_size: int,
    gpu_cache_cap: int,
    turn_headroom: int,
    keep_fraction: float,
    spill_fraction: float,
    sink_size: int,
    recency_window: int,
    obs_window_size: int,
    pooling: str,
    spill_hard_cap: int | None,
    seed: int,
) -> StreamingPrefillResult:
    """Stream context under a strict cap, spilling only bounded evicted subsets to CPU."""
    if context_ids.ndim == 1:
        context_ids = context_ids.unsqueeze(0)
    if int(context_ids.shape[1]) < int(total_context_length):
        raise ValueError(
            f"context_ids has only {int(context_ids.shape[1])} tokens, "
            f"but total_context_length={int(total_context_length)} was requested."
        )

    context_cap = int(gpu_cache_cap) - int(turn_headroom)
    if context_cap <= 0:
        raise ValueError("gpu_cache_cap must exceed turn_headroom.")

    geometry = simulate_streaming_geometry(
        total_context_length=int(total_context_length),
        chunk_size=int(chunk_size),
        context_cap=context_cap,
        keep_fraction=float(keep_fraction),
    )

    qnorm_spill = SpillBuffer()
    random_spill = SpillBuffer()
    qnorm_by_position: dict[int, float] = {}
    active_cache: PositionTrackedCache | None = None
    logical_cursor = 0
    event_summaries: list[dict[str, int | float]] = []
    spill_selection_s = 0.0
    peak_active = 0
    eviction_events = 0
    stream_start = time.perf_counter()

    while logical_cursor < int(total_context_length):
        current_chunk = min(int(chunk_size), int(total_context_length) - logical_cursor)
        if active_cache is not None and len(active_cache) + current_chunk > context_cap:
            keep_budget = max(1, int(math.ceil(len(active_cache) * float(keep_fraction))))
            policy = SnapKV(
                obs_window_size=int(obs_window_size),
                sink_size=int(sink_size),
                recency_window=min(int(recency_window), max(0, keep_budget - int(sink_size))),
                pooling=pooling,
            )
            event_start = time.perf_counter()
            eviction = policy.evict(active_cache, k_budget=keep_budget)
            qnorm_positions = select_spill_positions(
                evicted_cache=eviction.evicted,
                qnorm_scores=qnorm_by_position,
                fraction=float(spill_fraction),
                policy="qnorm",
                seed=int(seed) + eviction_events,
            )
            random_positions = select_spill_positions(
                evicted_cache=eviction.evicted,
                qnorm_scores=qnorm_by_position,
                fraction=float(spill_fraction),
                policy="random",
                seed=int(seed) + 10_000 + eviction_events,
            )
            qnorm_fragment = slice_cache_by_positions(eviction.evicted, qnorm_positions)
            random_fragment = slice_cache_by_positions(eviction.evicted, random_positions)
            qnorm_spill = append_spill_fragment(
                qnorm_spill,
                qnorm_fragment,
                qnorm_scores=qnorm_by_position,
                hard_cap=spill_hard_cap,
            )
            random_spill = append_spill_fragment(
                random_spill,
                random_fragment,
                qnorm_scores=qnorm_by_position,
                hard_cap=spill_hard_cap,
            )
            active_cache = eviction.compressed
            spill_selection_s += time.perf_counter() - event_start
            eviction_events += 1
            event_summaries.append(
                {
                    "event_index": eviction_events,
                    "logical_cursor": int(logical_cursor),
                    "pre_evict_active": int(len(eviction.compressed) + len(eviction.evicted)),
                    "post_evict_active": int(len(active_cache)),
                    "evicted_count": int(len(eviction.evicted)),
                    "qnorm_spill_size": int(len(qnorm_spill)),
                    "random_spill_size": int(len(random_spill)),
                }
            )
            assert_strict_cap(active_cache, cap=context_cap, label="post-eviction context cache")

        chunk_ids = context_ids[:, logical_cursor : logical_cursor + current_chunk]
        active_cache, new_qnorms = append_context_chunk_with_qnorm(
            model=model,
            active_cache=active_cache,
            chunk_ids=chunk_ids,
            logical_position_base=logical_cursor,
        )
        qnorm_by_position.update(new_qnorms)
        logical_cursor += current_chunk
        peak_active = max(peak_active, len(active_cache))
        assert_strict_cap(active_cache, cap=context_cap, label="post-append context cache")

    if active_cache is None:
        raise RuntimeError("Streaming prefill produced no active cache.")
    if geometry.eviction_events != eviction_events:
        raise RuntimeError(f"Geometry mismatch: simulated {geometry.eviction_events}, observed {eviction_events}.")

    return StreamingPrefillResult(
        active_cache=active_cache,
        qnorm_spill=qnorm_spill,
        random_spill=random_spill,
        qnorm_by_position=qnorm_by_position,
        eviction_events=int(eviction_events),
        final_active_context_tokens=int(len(active_cache)),
        peak_active_context_tokens=int(peak_active),
        spill_selection_s=float(spill_selection_s),
        stream_prefill_s=float(time.perf_counter() - stream_start),
        event_summaries=tuple(event_summaries),
    )


def stream_context_qnorm_spill_sweep(
    *,
    model,
    context_ids: torch.Tensor,
    total_context_length: int,
    chunk_size: int,
    gpu_cache_cap: int,
    turn_headroom: int,
    spill_fractions: Sequence[float],
    sink_size: int,
    recency_window: int,
    obs_window_size: int,
    pooling: str,
) -> QnormSpillSweepResult:
    """Stream context once and measure top-X qnorm CPU-spill coverage for many X values.

    This diagnostic uses fill-cap eviction: before each chunk that would overflow,
    it evicts only enough tokens to fit that chunk. That matches the 64K -> 32K
    calibration question and avoids the final-cache underutilization caused by
    repeated 90% eviction.
    """
    if context_ids.ndim == 1:
        context_ids = context_ids.unsqueeze(0)
    if int(context_ids.shape[1]) < int(total_context_length):
        raise ValueError(
            f"context_ids has only {int(context_ids.shape[1])} tokens, "
            f"but total_context_length={int(total_context_length)} was requested."
        )
    fractions = tuple(sorted(dict.fromkeys(float(value) for value in spill_fractions)))
    if not fractions or any(value <= 0.0 or value > 1.0 for value in fractions):
        raise ValueError("spill_fractions must contain values in (0, 1].")

    context_cap = int(gpu_cache_cap) - int(turn_headroom)
    if context_cap <= 0:
        raise ValueError("gpu_cache_cap must exceed turn_headroom.")
    if int(chunk_size) > context_cap:
        raise ValueError("chunk_size must not exceed the effective context cap.")

    active_cache: PositionTrackedCache | None = None
    qnorm_by_position: dict[int, float] = {}
    spill_sets = {fraction: set() for fraction in fractions}
    qnorm_rank_by_position: dict[int, tuple[int, int]] = {}
    logical_cursor = 0
    peak_active = 0
    eviction_events = 0
    total_evicted = 0
    event_summaries: list[dict[str, int | float]] = []
    stream_start = time.perf_counter()

    while logical_cursor < int(total_context_length):
        current_chunk = min(int(chunk_size), int(total_context_length) - logical_cursor)
        if active_cache is not None and len(active_cache) + current_chunk > context_cap:
            keep_budget = max(1, context_cap - current_chunk)
            policy = SnapKV(
                obs_window_size=int(obs_window_size),
                sink_size=int(sink_size),
                recency_window=min(int(recency_window), max(0, keep_budget - int(sink_size))),
                pooling=pooling,
            )
            eviction = policy.evict(active_cache, k_budget=keep_budget)
            evicted_positions = [int(position) for position in eviction.evicted.positions]
            total_evicted += len(evicted_positions)
            ranked = sorted(
                evicted_positions,
                key=lambda position: (-float(qnorm_by_position.get(int(position), 0.0)), int(position)),
            )
            for rank_index, position in enumerate(ranked, start=1):
                qnorm_rank_by_position[int(position)] = (int(rank_index), int(len(ranked)))
            event_summary: dict[str, int | float] = {
                "event_index": eviction_events + 1,
                "logical_cursor": int(logical_cursor),
                "pre_evict_active": int(len(active_cache)),
                "post_evict_active": int(len(eviction.compressed)),
                "evicted_count": int(len(evicted_positions)),
            }
            for fraction in fractions:
                keep_count = max(1, int(math.ceil(len(ranked) * fraction))) if ranked else 0
                selected = ranked[:keep_count]
                spill_sets[fraction].update(selected)
                event_summary[f"spill_{fraction:g}_size"] = int(len(spill_sets[fraction]))
            active_cache = eviction.compressed
            eviction_events += 1
            event_summaries.append(event_summary)
            assert_strict_cap(active_cache, cap=context_cap, label="post-eviction context cache")

        chunk_ids = context_ids[:, logical_cursor : logical_cursor + current_chunk]
        active_cache, new_qnorms = append_context_chunk_with_qnorm(
            model=model,
            active_cache=active_cache,
            chunk_ids=chunk_ids,
            logical_position_base=logical_cursor,
        )
        qnorm_by_position.update(new_qnorms)
        logical_cursor += current_chunk
        peak_active = max(peak_active, len(active_cache))
        assert_strict_cap(active_cache, cap=context_cap, label="post-append context cache")

    if active_cache is None:
        raise RuntimeError("Streaming prefill produced no active cache.")

    return QnormSpillSweepResult(
        final_active_context_tokens=int(len(active_cache)),
        peak_active_context_tokens=int(peak_active),
        eviction_events=int(eviction_events),
        total_evicted_tokens=int(total_evicted),
        stream_prefill_s=float(time.perf_counter() - stream_start),
        spill_positions_by_fraction={
            float(fraction): tuple(sorted(int(position) for position in positions))
            for fraction, positions in spill_sets.items()
        },
        qnorm_rank_by_position=qnorm_rank_by_position,
        event_summaries=tuple(event_summaries),
    )


def stream_context_two_tier_snapkv_spill(
    *,
    model,
    context_ids: torch.Tensor,
    total_context_length: int,
    chunk_size: int,
    gpu_cache_cap: int,
    turn_headroom: int,
    cpu_store_fraction: float,
    sink_size: int,
    recency_window: int,
    obs_window_size: int,
    pooling: str,
) -> TwoTierSnapKVSpillResult:
    """Stream context with a two-tier SnapKV GPU/CPU split.

    At each overflow event:
    1. SnapKV compresses the active cache to a candidate pool. Tokens outside
       this pool are permanently dropped.
    2. SnapKV compresses the candidate pool to the GPU keep budget. Tokens
       evicted by this second pass are retained as CPU spill positions.
    """
    if context_ids.ndim == 1:
        context_ids = context_ids.unsqueeze(0)
    if int(context_ids.shape[1]) < int(total_context_length):
        raise ValueError(
            f"context_ids has only {int(context_ids.shape[1])} tokens, "
            f"but total_context_length={int(total_context_length)} was requested."
        )
    context_cap = int(gpu_cache_cap) - int(turn_headroom)
    if context_cap <= 0:
        raise ValueError("gpu_cache_cap must exceed turn_headroom.")
    if int(chunk_size) > context_cap:
        raise ValueError("chunk_size must not exceed the effective context cap.")

    active_cache: PositionTrackedCache | None = None
    logical_cursor = 0
    peak_active = 0
    eviction_events = 0
    permanent_evicted = 0
    cpu_spill_positions: set[int] = set()
    event_summaries: list[dict[str, int | float]] = []
    stream_start = time.perf_counter()

    while logical_cursor < int(total_context_length):
        current_chunk = min(int(chunk_size), int(total_context_length) - logical_cursor)
        if active_cache is not None and len(active_cache) + current_chunk > context_cap:
            gpu_keep_budget = max(1, context_cap - current_chunk)
            candidate_budget, cpu_budget = two_tier_candidate_budget(
                pre_evict_tokens=len(active_cache),
                gpu_keep_budget=gpu_keep_budget,
                cpu_store_fraction=float(cpu_store_fraction),
            )
            candidate_policy = SnapKV(
                obs_window_size=int(obs_window_size),
                sink_size=int(sink_size),
                recency_window=min(int(recency_window), max(0, candidate_budget - int(sink_size))),
                pooling=pooling,
            )
            candidate_result = candidate_policy.evict(active_cache, k_budget=candidate_budget)
            permanent_evicted += len(candidate_result.evicted)

            gpu_policy = SnapKV(
                obs_window_size=int(obs_window_size),
                sink_size=int(sink_size),
                recency_window=min(int(recency_window), max(0, gpu_keep_budget - int(sink_size))),
                pooling=pooling,
            )
            gpu_result = gpu_policy.evict(candidate_result.compressed, k_budget=gpu_keep_budget)
            cpu_spill_positions.update(int(position) for position in gpu_result.evicted.positions)
            active_cache = gpu_result.compressed
            eviction_events += 1
            event_summaries.append(
                {
                    "event_index": int(eviction_events),
                    "logical_cursor": int(logical_cursor),
                    "pre_evict_active": int(len(candidate_result.compressed) + len(candidate_result.evicted)),
                    "candidate_budget": int(candidate_budget),
                    "gpu_keep_budget": int(gpu_keep_budget),
                    "cpu_budget": int(cpu_budget),
                    "permanent_evicted_this_event": int(len(candidate_result.evicted)),
                    "cpu_spill_this_event": int(len(gpu_result.evicted)),
                    "post_evict_active": int(len(active_cache)),
                    "cpu_spill_size": int(len(cpu_spill_positions)),
                }
            )
            assert_strict_cap(active_cache, cap=context_cap, label="post-two-tier-eviction context cache")

        chunk_ids = context_ids[:, logical_cursor : logical_cursor + current_chunk]
        active_cache = append_context_chunk(
            model=model,
            active_cache=active_cache,
            chunk_ids=chunk_ids,
            logical_position_base=logical_cursor,
        )
        logical_cursor += current_chunk
        peak_active = max(peak_active, len(active_cache))
        assert_strict_cap(active_cache, cap=context_cap, label="post-append context cache")

    if active_cache is None:
        raise RuntimeError("Streaming prefill produced no active cache.")

    return TwoTierSnapKVSpillResult(
        active_cache=active_cache,
        cpu_spill_positions=tuple(sorted(cpu_spill_positions)),
        permanent_evicted_tokens=int(permanent_evicted),
        final_active_context_tokens=int(len(active_cache)),
        peak_active_context_tokens=int(peak_active),
        eviction_events=int(eviction_events),
        total_cpu_spill_tokens=int(len(cpu_spill_positions)),
        stream_prefill_s=float(time.perf_counter() - stream_start),
        event_summaries=tuple(event_summaries),
    )
