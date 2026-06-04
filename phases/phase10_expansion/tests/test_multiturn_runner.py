from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache
from phases.phase10_expansion.scripts.run_multiturn_smoke import (
    _active_key_indices,
    _append_new_rows,
    _budget_audit_fields,
    _cache_nbytes,
    _select_repair_positions,
    _slice_by_positions,
)


def _cache(positions: list[int]) -> PositionTrackedCache:
    seq_len = len(positions)
    keys = torch.arange(seq_len, dtype=torch.float16).reshape(1, 1, seq_len, 1)
    values = keys + 100
    return PositionTrackedCache(((keys, values),), positions)


class MultiTurnRunnerHelperTests(unittest.TestCase):
    def test_append_new_rows_preserves_existing_pool_and_adds_only_new_positions(self) -> None:
        pool = _cache([0, 1, 2])
        generated = _cache([0, 1, 2, 3, 4])

        updated = _append_new_rows(pool, generated)

        self.assertEqual(updated.positions, [0, 1, 2, 3, 4])
        self.assertTrue(torch.equal(updated.kv[0][0][0, 0, :, 0], torch.tensor([0, 1, 2, 3, 4], dtype=torch.float16)))

    def test_append_new_rows_returns_pool_when_no_positions_are_new(self) -> None:
        pool = _cache([0, 1, 2])
        generated = _cache([0, 1, 2])

        self.assertIs(_append_new_rows(pool, generated), pool)

    def test_slice_by_positions_uses_absolute_position_metadata(self) -> None:
        cache = _cache([10, 20, 30, 40])

        fragment = _slice_by_positions(cache, [20, 40])

        self.assertEqual(fragment.positions, [20, 40])
        self.assertTrue(torch.equal(fragment.kv[0][0][0, 0, :, 0], torch.tensor([1, 3], dtype=torch.float16)))

    def test_cache_nbytes_counts_key_and_value_tensors(self) -> None:
        cache = _cache([10, 20, 30, 40])

        self.assertEqual(_cache_nbytes(cache), 16)

    def test_budget_audit_fields_report_target_gap_and_buffer_accounting(self) -> None:
        fields = _budget_audit_fields(
            active_context_count=104,
            target_active_context_count=108,
            base_active_context_count=96,
            evicted_buffer_count=900,
            evicted_buffer_bytes=1234,
        )

        self.assertEqual(fields["target_active_context_count"], 108)
        self.assertEqual(fields["base_active_context_count"], 96)
        self.assertEqual(fields["evicted_buffer_count"], 900)
        self.assertEqual(fields["evicted_buffer_bytes"], 1234)
        self.assertEqual(fields["active_budget_gap"], -4)
        self.assertFalse(fields["active_context_matches_target"])

    def test_active_key_indices_tracks_requested_span_presence(self) -> None:
        prepared = SimpleNamespace(
            span_token_positions={
                "needle_1": [2, 3],
                "needle_2": [10],
                "needle_3": [20, 21],
            }
        )

        active = _active_key_indices(prepared=prepared, active_context_positions=[0, 3, 21], key_count=3)

        self.assertEqual(active, [0, 2])

    def test_stale_query_selector_has_no_turn_zero_repair_without_previous_query(self) -> None:
        selected, timings = _select_repair_positions(
            condition="StaleQ-K",
            model=None,
            active_cache=_cache([0, 1]),
            evicted_cache=_cache([2, 3]),
            prepared=SimpleNamespace(question_ids=torch.tensor([[1]])),
            previous_prepared=None,
            relevant_positions=(2,),
            relevant_groups=((2,),),
            turn_n_scores={2: 1.0, 3: 0.0},
            k=1,
            query_scoring_mode="proxy",
            pooling="max",
            burst_left=0,
            burst_right=0,
            seed=0,
        )

        self.assertEqual(selected, [])
        self.assertEqual(timings, {})

    def test_stale_query_only_selector_has_no_turn_zero_repair_without_previous_query(self) -> None:
        selected, timings = _select_repair_positions(
            condition="StaleQOnly-K",
            model=None,
            active_cache=_cache([0, 1]),
            evicted_cache=_cache([2, 3]),
            prepared=SimpleNamespace(question_ids=torch.tensor([[1]])),
            previous_prepared=None,
            relevant_positions=(2,),
            relevant_groups=((2,),),
            turn_n_scores={2: 1.0, 3: 0.0},
            k=1,
            query_scoring_mode="proxy",
            pooling="max",
            burst_left=0,
            burst_right=0,
            seed=0,
        )

        self.assertEqual(selected, [])
        self.assertEqual(timings, {})

    def test_query_only_selector_ignores_turn_importance_tie_breaker(self) -> None:
        with patch(
            "phases.phase10_expansion.scripts.run_multiturn_smoke._compute_query_scores",
            return_value={2: 0.5, 3: 0.5},
        ):
            selected, timings = _select_repair_positions(
                condition="CurrentQOnly-K",
                model=None,
                active_cache=_cache([0, 1]),
                evicted_cache=_cache([2, 3]),
                prepared=SimpleNamespace(question_ids=torch.tensor([[1]])),
                previous_prepared=None,
                relevant_positions=(2,),
                relevant_groups=((2,),),
                turn_n_scores={2: 0.0, 3: 10.0},
                k=1,
                query_scoring_mode="proxy",
                pooling="max",
                burst_left=0,
                burst_right=0,
                seed=0,
            )

        self.assertEqual(selected, [2])
        self.assertIn("query_score_s", timings)

    def test_idlekv_selector_keeps_turn_importance_tie_breaker(self) -> None:
        with patch(
            "phases.phase10_expansion.scripts.run_multiturn_smoke._compute_query_scores",
            return_value={2: 0.5, 3: 0.5},
        ):
            selected, _timings = _select_repair_positions(
                condition="IdleKV",
                model=None,
                active_cache=_cache([0, 1]),
                evicted_cache=_cache([2, 3]),
                prepared=SimpleNamespace(question_ids=torch.tensor([[1]])),
                previous_prepared=None,
                relevant_positions=(2,),
                relevant_groups=((2,),),
                turn_n_scores={2: 0.0, 3: 10.0},
                k=1,
                query_scoring_mode="proxy",
                pooling="max",
                burst_left=0,
                burst_right=0,
                seed=0,
            )

        self.assertEqual(selected, [3])


if __name__ == "__main__":
    unittest.main()
