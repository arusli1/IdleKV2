"""Unit tests for the Phase 6 selector helpers."""

from __future__ import annotations

import unittest

import torch

from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache
from phases.phase6_repair.src.selectors import (
    contrastive_position_scores,
    pack_anchor_bursts,
    score_evicted_positions,
    select_coverage_aware_positions,
    select_idlekv_positions,
    select_mmr_positions,
    select_oldest_positions,
    select_oracle_positions,
    select_refresh_positions,
)


def _make_evicted_cache() -> PositionTrackedCache:
    keys = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]]],
        dtype=torch.float16,
    )
    values = torch.zeros_like(keys)
    return PositionTrackedCache(((keys, values),), [10, 11, 12])


class Phase6SelectorTests(unittest.TestCase):
    def test_score_evicted_positions_prefers_matching_key(self) -> None:
        cache = _make_evicted_cache()
        query_rows = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
        scores = score_evicted_positions(query_rows=query_rows, evicted_cache=cache, pooling="max")
        self.assertGreater(scores[10], scores[11])
        self.assertGreater(scores[10], scores[12])

    def test_score_evicted_positions_supports_query_head_replication(self) -> None:
        cache = _make_evicted_cache()
        query_rows = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=torch.float32)
        scores = score_evicted_positions(query_rows=query_rows, evicted_cache=cache, pooling="max")
        self.assertGreater(scores[10], scores[11])
        self.assertGreater(scores[10], scores[12])

    def test_score_evicted_positions_accounts_for_active_cache_competition(self) -> None:
        evicted = _make_evicted_cache()
        active_keys = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float16)
        active = PositionTrackedCache(((active_keys, torch.zeros_like(active_keys)),), [0, 1])
        query_rows = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)

        without_active = score_evicted_positions(query_rows=query_rows, evicted_cache=evicted, pooling="max")
        with_active = score_evicted_positions(
            query_rows=query_rows,
            evicted_cache=evicted,
            active_cache=active,
            pooling="max",
        )

        self.assertLess(with_active[10], without_active[10])
        self.assertGreater(with_active[10], with_active[11])

    def test_pack_anchor_bursts_hits_exact_k_then_stops(self) -> None:
        selected = pack_anchor_bursts(
            anchor_positions=[10],
            available_positions=list(range(8, 16)),
            k=5,
            left=2,
            right=2,
            backfill_positions=list(range(8, 16)),
        )
        self.assertEqual(selected, [8, 9, 10, 11, 12])

    def test_contrastive_position_scores_subtract_standardized_negative_signal(self) -> None:
        positions = [10, 11, 12]
        scores = contrastive_position_scores(
            positions,
            positive_scores={10: 10.0, 11: 6.0, 12: 1.0},
            negative_scores={10: 9.0, 11: 1.0, 12: 1.0},
        )

        ranked = sorted(positions, key=lambda position: -scores[position])
        self.assertEqual(ranked[0], 11)
        self.assertLess(scores[10], scores[11])

    def test_selectors_respect_relevant_positions(self) -> None:
        evicted = [10, 11, 12, 13, 14, 15]
        q2_scores = {10: 0.1, 11: 0.2, 12: 0.9, 13: 0.8, 14: 0.05, 15: 0.04}
        turn_scores = {position: float(position) for position in evicted}

        idlekv = select_idlekv_positions(
            evicted_positions=evicted,
            q2_scores=q2_scores,
            turn_n_scores=turn_scores,
            k=3,
            left=0,
            right=0,
        )
        oldest = select_oldest_positions(
            evicted_positions=evicted,
            k=3,
            left=0,
            right=0,
        )
        oracle = select_oracle_positions(
            evicted_positions=evicted,
            relevant_positions=[13],
            q2_scores=q2_scores,
            turn_n_scores=turn_scores,
            k=3,
            left=1,
            right=1,
        )

        self.assertEqual(idlekv, [12, 13, 11])
        self.assertEqual(oldest, [10, 11, 12])
        self.assertEqual(oracle, [13, 12, 11])

    def test_coverage_aware_selector_avoids_overlapping_burst_waste(self) -> None:
        evicted = list(range(20))
        q2_scores = {5: 10.0, 6: 9.0, 10: 8.0, 11: 7.0}
        turn_scores = {position: 0.0 for position in evicted}

        selected = select_coverage_aware_positions(
            evicted_positions=evicted,
            q2_scores=q2_scores,
            turn_n_scores=turn_scores,
            k=6,
            left=1,
            right=1,
        )

        self.assertEqual(selected, [4, 5, 6, 9, 10, 11])

    def test_coverage_aware_selector_is_deterministic_and_backfills_to_k(self) -> None:
        evicted = [0, 2, 4, 6]
        selected = select_coverage_aware_positions(
            evicted_positions=evicted,
            q2_scores={},
            turn_n_scores={},
            k=3,
            left=0,
            right=0,
        )

        self.assertEqual(selected, [0, 2, 4])

    def test_mmr_selector_adds_diverse_anchor_when_scores_are_close(self) -> None:
        evicted = list(range(100))
        q2_scores = {10: 1.0, 11: 0.99, 80: 0.98}
        turn_scores = {position: 0.0 for position in evicted}

        selected = select_mmr_positions(
            evicted_positions=evicted,
            q2_scores=q2_scores,
            turn_n_scores=turn_scores,
            k=3,
            left=0,
            right=0,
            diversity_weight=0.5,
        )

        self.assertEqual(selected, [10, 80, 11])

    def test_refresh_selector_reselects_full_context_budget(self) -> None:
        refresh = select_refresh_positions(
            context_positions=list(range(10)),
            mandatory_positions=[0, 1],
            q2_scores={6: 10.0, 5: 8.0, 7: 7.0, 9: 6.0},
            turn_n_scores={position: 0.0 for position in range(10)},
            context_budget=5,
            left=1,
            right=1,
        )

        self.assertEqual(refresh, [0, 1, 5, 6, 7])

    def test_oracle_can_select_exact_gold_span_groups(self) -> None:
        evicted = [10, 11, 12, 13, 14, 15]
        q2_scores = {10: 0.5, 11: 0.4, 12: 0.3, 13: 0.2, 14: 0.1, 15: 0.0}
        turn_scores = {position: float(position) for position in evicted}

        oracle = select_oracle_positions(
            evicted_positions=evicted,
            relevant_positions=[10, 11, 14],
            relevant_position_groups=[(10, 11), (14,)],
            q2_scores=q2_scores,
            turn_n_scores=turn_scores,
            k=3,
            left=2,
            right=20,
        )

        self.assertEqual(oracle, [10, 11, 14])


if __name__ == "__main__":
    unittest.main()
