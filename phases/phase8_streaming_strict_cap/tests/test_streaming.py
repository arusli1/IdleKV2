"""Unit tests for Phase 8 strict-cap streaming helpers."""

from __future__ import annotations

import unittest

import torch

from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache
from phases.phase8_streaming_strict_cap.src.streaming import (
    SpillBuffer,
    append_spill_fragment,
    choose_lowest_score_positions,
    select_spill_positions,
    simulate_streaming_geometry,
    swap_restore_positions,
    trim_spill_to_cap,
    two_tier_candidate_budget,
)


def _make_cache(positions: list[int]) -> PositionTrackedCache:
    seq_len = len(positions)
    keys = torch.arange(seq_len, dtype=torch.float32).reshape(1, 1, seq_len, 1)
    values = keys + 100
    return PositionTrackedCache(((keys, values),), positions)


class Phase8StreamingTests(unittest.TestCase):
    def test_geometry_enforces_cap_and_counts_evictions(self) -> None:
        geometry = simulate_streaming_geometry(
            total_context_length=100,
            chunk_size=10,
            context_cap=32,
            keep_fraction=0.25,
        )

        self.assertLessEqual(geometry.peak_active_tokens, 32)
        self.assertEqual(geometry.eviction_events, 4)
        self.assertEqual(geometry.active_after_evictions, (8, 7, 7, 7))

    def test_select_spill_positions_takes_top_fraction_by_qnorm(self) -> None:
        evicted = _make_cache([10, 11, 12, 13, 14])
        selected = select_spill_positions(
            evicted_cache=evicted,
            qnorm_scores={10: 0.0, 11: 4.0, 12: 1.0, 13: 3.0, 14: 2.0},
            fraction=0.4,
            policy="qnorm",
            seed=0,
        )

        self.assertEqual(selected, [11, 13])

    def test_append_and_trim_spill_keeps_highest_qnorm_positions(self) -> None:
        spill = append_spill_fragment(
            SpillBuffer(),
            _make_cache([5, 6, 7]),
            qnorm_scores={5: 0.5, 6: 0.6, 7: 0.7},
        )
        spill = append_spill_fragment(
            spill,
            _make_cache([8, 9]),
            qnorm_scores={8: 0.8, 9: 0.1},
            hard_cap=4,
        )
        trimmed = trim_spill_to_cap(spill, cap=2)

        self.assertEqual(trimmed.positions, (7, 8))
        self.assertEqual(set(trimmed.qnorm_scores), {7, 8})

    def test_choose_lowest_score_positions_is_stable(self) -> None:
        selected = choose_lowest_score_positions(
            positions=[13, 10, 12],
            scores={10: 0.2, 12: 0.1, 13: 0.1},
            k=2,
        )

        self.assertEqual(selected, [12, 13])

    def test_swap_restore_preserves_active_size_and_order(self) -> None:
        active = _make_cache([0, 1, 4, 5])
        spill = _make_cache([2, 3])
        repaired = swap_restore_positions(
            active_cache=active,
            spill_cache=spill,
            restore_positions=[3, 2],
            drop_positions=[1, 5],
            cap=4,
        )

        self.assertEqual(repaired.positions, [0, 2, 3, 4])
        self.assertEqual(len(repaired), len(active))

    def test_two_tier_candidate_budget_interprets_fraction_of_gpu_only_evictions(self) -> None:
        candidate_budget, cpu_budget = two_tier_candidate_budget(
            pre_evict_tokens=100,
            gpu_keep_budget=80,
            cpu_store_fraction=0.25,
        )

        self.assertEqual(cpu_budget, 5)
        self.assertEqual(candidate_budget, 85)

        full_candidate_budget, full_cpu_budget = two_tier_candidate_budget(
            pre_evict_tokens=100,
            gpu_keep_budget=80,
            cpu_store_fraction=1.0,
        )
        self.assertEqual(full_cpu_budget, 20)
        self.assertEqual(full_candidate_budget, 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
