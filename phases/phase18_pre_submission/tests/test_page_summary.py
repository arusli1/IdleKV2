"""Unit tests for Step 0c: page-summary scorer."""

from __future__ import annotations

import pytest
import torch

from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache
from phases.phase6_repair.src.selectors import (
    score_evicted_positions,
    score_evicted_positions_budgeted,
)
from phases.phase18_pre_submission.src.page_summary import (
    compute_chunk_summaries,
    score_evicted_positions_page_summary,
)


def _make_synthetic_caches(
    *,
    n_layers: int = 4,
    n_query_heads: int = 8,
    n_kv_heads: int = 4,
    q_len: int = 16,
    head_dim: int = 64,
    n_evicted: int = 128,
    n_active: int = 32,
) -> tuple[torch.Tensor, PositionTrackedCache, PositionTrackedCache]:
    torch.manual_seed(13)
    query_rows = torch.randn(n_layers, n_query_heads, q_len, head_dim, dtype=torch.float32)
    evicted_layers = []
    for _ in range(n_layers):
        key = torch.randn(1, n_kv_heads, n_evicted, head_dim, dtype=torch.float32)
        value = torch.randn(1, n_kv_heads, n_evicted, head_dim, dtype=torch.float32)
        evicted_layers.append((key, value))
    evicted = PositionTrackedCache(tuple(evicted_layers), list(range(200, 200 + n_evicted)))
    active_layers = []
    for _ in range(n_layers):
        key = torch.randn(1, n_kv_heads, n_active, head_dim, dtype=torch.float32)
        value = torch.randn(1, n_kv_heads, n_active, head_dim, dtype=torch.float32)
        active_layers.append((key, value))
    active = PositionTrackedCache(tuple(active_layers), list(range(n_active)))
    return query_rows, evicted, active


def test_compute_chunk_summaries_shape() -> None:
    """Summaries have the right shape: [n_layers, n_kv_heads, n_chunks, head_dim]."""
    _, evicted, _ = _make_synthetic_caches(n_layers=4, n_kv_heads=4, head_dim=64, n_evicted=128)
    summaries, chunk_ranges = compute_chunk_summaries(evicted, chunk_size=32)
    # 128 / 32 = 4 chunks
    assert summaries.shape == (4, 4, 4, 64)
    assert len(chunk_ranges) == 4
    assert chunk_ranges[0] == (0, 32)
    assert chunk_ranges[-1] == (96, 128)


def test_page_summary_no_cap_scores_all_positions() -> None:
    """With no cap, all positions get a score and no cap fires."""
    query_rows, evicted, active = _make_synthetic_caches()
    scores, info = score_evicted_positions_page_summary(
        query_rows=query_rows,
        evicted_cache=evicted,
        active_cache=active,
        chunk_size=32,
        wallclock_deadline_s=None,
    )
    assert info["cap_fired"] is False
    assert info["positions_scored"] == info["positions_total"]
    assert info["chunks_visited"] == info["chunks_total"]
    assert len(scores) == 128


def test_page_summary_no_cap_matches_chunked_refresh_per_chunk() -> None:
    """Without cap, page-summary's per-chunk scoring matches chunk-position
    Refresh-K-budgeted's per-chunk scoring exactly (same softmax denominator
    structure: active + chunk_evicted)."""
    query_rows, evicted, active = _make_synthetic_caches()
    page_scores, page_info = score_evicted_positions_page_summary(
        query_rows=query_rows,
        evicted_cache=evicted,
        active_cache=active,
        chunk_size=32,
        wallclock_deadline_s=None,
    )
    refresh_scores, _ = score_evicted_positions_budgeted(
        query_rows=query_rows,
        evicted_cache=evicted,
        active_cache=active,
        wallclock_deadline_s=None,
        position_chunk_size=32,
    )
    # Both score every position. Each chunk's scores are computed
    # against active + chunk_evicted, so per-position values match.
    assert page_info["cap_fired"] is False
    assert set(page_scores.keys()) == set(refresh_scores.keys())
    for pos, score in refresh_scores.items():
        assert page_scores[pos] == pytest.approx(score, abs=1e-5), (
            f"position {pos}: refresh {score} vs page {page_scores[pos]}"
        )


def test_page_summary_immediate_cap_returns_empty() -> None:
    """A zero deadline should fire before any chunk's expensive scoring."""
    query_rows, evicted, active = _make_synthetic_caches(n_evicted=256)
    scores, info = score_evicted_positions_page_summary(
        query_rows=query_rows,
        evicted_cache=evicted,
        active_cache=active,
        chunk_size=32,
        wallclock_deadline_s=0.0,
    )
    assert info["cap_fired"] is True
    assert info["positions_scored"] == 0
    assert scores == {}


def test_page_summary_partial_cap_visits_chunks_in_priority_order() -> None:
    """Partial cap visits chunks in stage-1 ranked order (not position order)."""
    query_rows, evicted, active = _make_synthetic_caches(n_evicted=256)
    # First time without cap to know the ordering
    _, info_full = score_evicted_positions_page_summary(
        query_rows=query_rows,
        evicted_cache=evicted,
        active_cache=active,
        chunk_size=32,
        wallclock_deadline_s=None,
    )
    full_elapsed = float(info_full["elapsed_s"])
    if full_elapsed < 0.005:
        pytest.skip(f"full scorer too fast ({full_elapsed*1000:.3f}ms) for partial-cap test")
    full_ranking = info_full["ranked_chunk_indices"]

    # 0.1x of full elapsed should reliably trigger the cap mid-sweep
    # even after the precompute hoist made chunks much cheaper.
    deadline_s = full_elapsed * 0.02
    _, info_partial = score_evicted_positions_page_summary(
        query_rows=query_rows,
        evicted_cache=evicted,
        active_cache=active,
        chunk_size=32,
        wallclock_deadline_s=deadline_s,
    )
    assert info_partial["cap_fired"] is True
    assert 0 < int(info_partial["chunks_visited"]) < int(info_partial["chunks_total"])
    # Visited chunks should be the prefix of the full ranking
    n_visited = int(info_partial["chunks_visited"])
    assert info_partial["ranked_chunk_indices"] == full_ranking[:n_visited]


def test_page_summary_uses_precomputed_summaries() -> None:
    """Passing precomputed summaries skips inline computation; same result."""
    query_rows, evicted, active = _make_synthetic_caches(n_evicted=64)
    summaries, chunk_ranges = compute_chunk_summaries(evicted, chunk_size=16)
    # With precomputed summaries
    scores_pre, _ = score_evicted_positions_page_summary(
        query_rows=query_rows,
        evicted_cache=evicted,
        active_cache=active,
        chunk_size=16,
        wallclock_deadline_s=None,
        summaries=summaries,
        chunk_ranges=chunk_ranges,
    )
    # Without (computed inline)
    scores_inline, _ = score_evicted_positions_page_summary(
        query_rows=query_rows,
        evicted_cache=evicted,
        active_cache=active,
        chunk_size=16,
        wallclock_deadline_s=None,
    )
    assert set(scores_pre.keys()) == set(scores_inline.keys())
    for pos, score in scores_inline.items():
        assert scores_pre[pos] == pytest.approx(score, abs=1e-5)
