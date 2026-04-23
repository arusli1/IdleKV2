"""Live fixture builders for the GPU-backed Phase 4 profiling path."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from phases.phase2_kv_cache.src.runtime import make_exact_length_input_ids, model_device
from phases.phase3_eviction.src.eviction import SnapKV
from phases.phase3_eviction.src.runtime import build_position_tracked_cache

from .eviction_buffer import EvictionBuffer, SelectionStrategy

_DEFAULT_CONTEXT_TEXT = (
    "IdleKV profiling context about tool calls, deferred work, cache repair, eviction scores, "
    "retrieval buffers, telemetry, diagnostics, git traces, patch review notes, and benchmark "
    "artifacts repeated to fill the requested token window. "
)
_DEFAULT_QUERY_TEXT = (
    "Summarize the repair-relevant diagnostics from the cached context in a short answer. "
)


@dataclass(frozen=True)
class LiveRepairFixture:
    """One live compressed cache paired with its CPU eviction buffer."""

    active_cache: object
    buffer: EvictionBuffer
    query_ids: torch.Tensor
    context_tokens: int
    k_budget: int
    evicted_tokens: int
    obs_window_size: int


def build_snapkv_live_fixture(
    model,
    tokenizer,
    *,
    context_tokens: int = 32_768,
    k_budget: int = 4_096,
    selection_strategy: SelectionStrategy = "l2_norm",
    max_buffer_tokens: int = 10_000,
    obs_window_size: int = 32,
    sink_size: int = 4,
    recency_window: int = 1_024,
    query_len: int = 20,
) -> LiveRepairFixture:
    """Build one representative SnapKV eviction result plus its Phase 4 buffer."""
    device = model_device(model)
    context_ids = make_exact_length_input_ids(
        tokenizer,
        target_tokens=context_tokens,
        device=device,
        base_text=_DEFAULT_CONTEXT_TEXT,
    )
    full_cache = build_position_tracked_cache(model, context_ids)
    policy = SnapKV(
        obs_window_size=obs_window_size,
        sink_size=sink_size,
        recency_window=min(max(0, int(recency_window)), max(0, int(k_budget) - int(sink_size))),
        pooling="max",
    )
    eviction_result = policy.evict(full_cache, k_budget=k_budget)

    buffer = EvictionBuffer(max_tokens=max_buffer_tokens, selection_strategy=selection_strategy)
    buffer.push_from_result(eviction_result)
    query_ids = make_exact_length_input_ids(
        tokenizer,
        target_tokens=query_len,
        device=device,
        base_text=_DEFAULT_QUERY_TEXT,
    ).to(model_device(model))

    return LiveRepairFixture(
        active_cache=eviction_result.compressed,
        buffer=buffer,
        query_ids=query_ids,
        context_tokens=int(context_tokens),
        k_budget=int(k_budget),
        evicted_tokens=len(eviction_result.evicted.positions),
        obs_window_size=int(obs_window_size),
    )


__all__ = [
    "LiveRepairFixture",
    "build_snapkv_live_fixture",
]
