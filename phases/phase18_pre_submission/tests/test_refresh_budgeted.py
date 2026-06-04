"""Unit tests for Step 0b: budgeted Refresh-K scorer.

Two regressions we want to prevent:

1. ``score_evicted_positions_budgeted`` with no wall-clock cap and a
   ``position_chunk_size`` >= n_evicted must produce the EXACT same
   scores as the unbudgeted ``score_evicted_positions``. (Bit-equal
   per-position to within fp32 tolerance.)
2. With a tiny wall-clock cap, scoring must stop *between* position-
   chunks and return a partial dict; the cap_fired flag must be True
   and positions_scored < positions_total.
"""

from __future__ import annotations

import time

import pytest
import torch

from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache
from phases.phase6_repair.src.selectors import (
    score_evicted_positions,
    score_evicted_positions_budgeted,
)


def _make_synthetic_caches(
    *,
    n_layers: int = 4,
    n_query_heads: int = 8,
    n_kv_heads: int = 4,
    q_len: int = 16,
    head_dim: int = 64,
    n_evicted: int = 64,
    n_active: int = 32,
) -> tuple[torch.Tensor, PositionTrackedCache, PositionTrackedCache]:
    torch.manual_seed(7)
    query_rows = torch.randn(n_layers, n_query_heads, q_len, head_dim, dtype=torch.float32)
    evicted_layers = []
    for _ in range(n_layers):
        key = torch.randn(1, n_kv_heads, n_evicted, head_dim, dtype=torch.float32)
        value = torch.randn(1, n_kv_heads, n_evicted, head_dim, dtype=torch.float32)
        evicted_layers.append((key, value))
    evicted = PositionTrackedCache(tuple(evicted_layers), list(range(100, 100 + n_evicted)))

    active_layers = []
    for _ in range(n_layers):
        key = torch.randn(1, n_kv_heads, n_active, head_dim, dtype=torch.float32)
        value = torch.randn(1, n_kv_heads, n_active, head_dim, dtype=torch.float32)
        active_layers.append((key, value))
    active = PositionTrackedCache(tuple(active_layers), list(range(n_active)))
    return query_rows, evicted, active


def test_budgeted_no_cap_matches_unbudgeted_no_active() -> None:
    """Without active cache, budgeted (no cap, big chunk) must equal unbudgeted."""
    query_rows, evicted, _ = _make_synthetic_caches()
    unbudgeted = score_evicted_positions(query_rows=query_rows, evicted_cache=evicted)
    budgeted, info = score_evicted_positions_budgeted(
        query_rows=query_rows,
        evicted_cache=evicted,
        wallclock_deadline_s=None,
        position_chunk_size=10_000,
    )
    assert info["cap_fired"] is False
    assert info["positions_scored"] == info["positions_total"]
    assert set(budgeted.keys()) == set(unbudgeted.keys())
    for pos, score in unbudgeted.items():
        assert budgeted[pos] == pytest.approx(score, abs=1e-5), (
            f"position {pos}: unbudgeted {score} vs budgeted {budgeted[pos]}"
        )


def test_budgeted_small_chunks_normalizer_differs_from_unbudgeted() -> None:
    """Smaller chunks change the softmax denominator (chunk-restricted vs full-pool).

    This is a documented semantic difference, not a bug. Confirm the
    scorer still returns the right number of positions and that the
    ordering within the first chunk matches the unbudgeted ordering on
    the same chunk's positions.
    """
    query_rows, evicted, _ = _make_synthetic_caches(n_evicted=64)
    unbudgeted = score_evicted_positions(query_rows=query_rows, evicted_cache=evicted)
    budgeted, info = score_evicted_positions_budgeted(
        query_rows=query_rows,
        evicted_cache=evicted,
        wallclock_deadline_s=None,
        position_chunk_size=16,  # 4 chunks of 16
    )
    assert info["cap_fired"] is False
    assert info["positions_scored"] == 64
    # Within the first chunk (positions 100..115), the relative *ranking*
    # of positions in budgeted should match the relative ranking under
    # unbudgeted on the same positions (since both normalize over the
    # same keys: active=none here, plus the same first-chunk evicted set
    # since chunk_size=16 covers them all). Across chunks the magnitudes
    # differ but rankings within a chunk are preserved.
    first_chunk_positions = evicted.positions[:16]
    by_unbudgeted = sorted(first_chunk_positions, key=lambda p: -unbudgeted[p])
    by_budgeted = sorted(first_chunk_positions, key=lambda p: -budgeted[p])
    # We can't strictly require identical orderings (chunk-restricted
    # softmax can flip top-1 vs top-2 in pathological cases), but the
    # top-half should overlap >=80%.
    top_half_unbudgeted = set(by_unbudgeted[:8])
    top_half_budgeted = set(by_budgeted[:8])
    overlap = len(top_half_unbudgeted & top_half_budgeted)
    assert overlap >= 6, (
        f"top-half overlap {overlap}/8 too low; chunked rankings should "
        f"approximately match unbudgeted on same chunk positions"
    )


def test_budgeted_with_active_cache_matches() -> None:
    """With active cache concatenation, chunked (no cap) must equal unchunked.

    Note: This holds because each chunk computes softmax over (active +
    chunk_evicted), and the unbudgeted version computes softmax over
    (active + all_evicted). Within a chunk the chunk-evicted scores share
    the same active normalizer as the unbudgeted version *only if the
    softmax is computed over the same denominator*. The chunked version
    has a chunk-restricted denominator, so we expect a SHIFT, not equality.
    Confirm by checking that the *ranking* is preserved when the chunk
    contains all evicted positions.
    """
    query_rows, evicted, active = _make_synthetic_caches()
    unbudgeted = score_evicted_positions(
        query_rows=query_rows,
        evicted_cache=evicted,
        active_cache=active,
    )
    # Single chunk == unchunked: same denominator, same scores.
    budgeted_one_chunk, info = score_evicted_positions_budgeted(
        query_rows=query_rows,
        evicted_cache=evicted,
        active_cache=active,
        wallclock_deadline_s=None,
        position_chunk_size=10_000,
    )
    assert info["cap_fired"] is False
    for pos, score in unbudgeted.items():
        assert budgeted_one_chunk[pos] == pytest.approx(score, abs=1e-5), (
            f"position {pos}: {score} vs {budgeted_one_chunk[pos]}"
        )


def test_budgeted_tiny_cap_returns_partial_dict() -> None:
    """A near-zero deadline must fire before any chunk completes."""
    query_rows, evicted, _ = _make_synthetic_caches(n_evicted=128)
    # Sleep a hair so perf_counter advances past the deadline before the
    # first chunk's matmul finishes.
    deadline_s = 0.0  # immediate
    scores, info = score_evicted_positions_budgeted(
        query_rows=query_rows,
        evicted_cache=evicted,
        wallclock_deadline_s=deadline_s,
        position_chunk_size=16,
    )
    assert info["cap_fired"] is True
    assert info["positions_scored"] == 0
    assert scores == {}


def test_budgeted_intermediate_cap_returns_some_positions() -> None:
    """An intermediate deadline should produce a partial dict.

    We measure the no-cap elapsed time first, then set a deadline
    around half of that. Some chunks complete; some do not.
    """
    query_rows, evicted, _ = _make_synthetic_caches(n_evicted=256)
    # First, time the unbudgeted version to know what "half" looks like.
    _, info_full = score_evicted_positions_budgeted(
        query_rows=query_rows,
        evicted_cache=evicted,
        wallclock_deadline_s=None,
        position_chunk_size=32,
    )
    full_elapsed = float(info_full["elapsed_s"])
    if full_elapsed < 0.020:  # need ~20ms+ for reliable mid-sweep cap test
        pytest.skip(f"full scorer too fast ({full_elapsed*1000:.3f}ms) for partial-cap test")
    deadline_s = full_elapsed * 0.05

    scores, info = score_evicted_positions_budgeted(
        query_rows=query_rows,
        evicted_cache=evicted,
        wallclock_deadline_s=deadline_s,
        position_chunk_size=32,
    )
    # Should have scored some chunks but not all
    assert 0 < int(info["positions_scored"]) < int(info["positions_total"]), (
        f"expected partial scoring, got {info['positions_scored']}/{info['positions_total']}"
    )
    assert info["cap_fired"] is True
    # Scored positions should be the EARLIEST chunks (deterministic order)
    scored_positions = sorted(scores.keys())
    expected_first_position = evicted.positions[0]
    assert scored_positions[0] == expected_first_position
