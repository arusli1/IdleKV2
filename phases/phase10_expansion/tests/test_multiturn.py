from __future__ import annotations

import unittest

from phases.phase10_expansion.src.multiturn import (
    DEFAULT_8Q_CHALLENGE_REVISIT,
    DEFAULT_8Q_HARD_REVISIT,
    DEFAULT_8Q_SHIFT_REVISIT,
    DEFAULT_8Q_SWEEP_REVISIT,
    MultiTurnSchedule,
    cache_state_grid,
    normalize_active_sets,
    normalize_turns,
    per_turn_active_churn,
    per_turn_overlap,
    per_turn_recovery,
    revisit_events,
    span_names_by_turn,
    summarize_score_trajectory,
    evaluate_multiturn_summary_rows,
    validate_schedule,
)


class MultiTurnScheduleTests(unittest.TestCase):
    def test_normalize_turns_deduplicates_within_turn_and_requires_two_turns(self) -> None:
        self.assertEqual(normalize_turns([[1, 1, 2], [3]]), ((1, 2), (3,)))
        with self.assertRaisesRegex(ValueError, "at least two"):
            normalize_turns([[1]])
        with self.assertRaisesRegex(ValueError, "turn 0"):
            normalize_turns([[], [1]])

    def test_validate_schedule_rejects_out_of_range_indices(self) -> None:
        schedule = MultiTurnSchedule(
            name="bad",
            base_task_key="mq_niah_4q",
            turns=((0, 4), (1,)),
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_schedule(schedule, key_count=4)

    def test_default_shift_revisit_has_expected_span_names_and_revisits(self) -> None:
        validate_schedule(DEFAULT_8Q_SHIFT_REVISIT, key_count=8)

        self.assertEqual(
            span_names_by_turn(DEFAULT_8Q_SHIFT_REVISIT),
            (
                ("needle_7", "needle_8"),
                ("needle_1", "needle_2"),
                ("needle_4", "needle_5"),
                ("needle_1", "needle_2"),
            ),
        )
        self.assertEqual(
            revisit_events(DEFAULT_8Q_SHIFT_REVISIT),
            (
                {"query_index": 0, "first_turn": 1, "revisit_turn": 3},
                {"query_index": 1, "first_turn": 1, "revisit_turn": 3},
            ),
        )

    def test_per_turn_overlap_tracks_immediate_shift(self) -> None:
        schedule = MultiTurnSchedule(
            name="overlap",
            base_task_key="mq_niah_4q",
            turns=((0, 1), (1, 2), (3,)),
        )

        self.assertEqual(per_turn_overlap(schedule), (0.0, 0.5, 0.0))

    def test_default_sweep_revisit_is_valid_and_revisits_tail(self) -> None:
        validate_schedule(DEFAULT_8Q_SWEEP_REVISIT, key_count=8)

        self.assertEqual(
            revisit_events(DEFAULT_8Q_SWEEP_REVISIT),
            (
                {"query_index": 6, "first_turn": 0, "revisit_turn": 3},
                {"query_index": 7, "first_turn": 0, "revisit_turn": 3},
            ),
        )

    def test_default_hard_revisit_separates_stale_query_from_current_revisit(self) -> None:
        validate_schedule(DEFAULT_8Q_HARD_REVISIT, key_count=8)

        self.assertEqual(
            per_turn_overlap(DEFAULT_8Q_HARD_REVISIT),
            (0.0, 0.0, 0.0, 0.0, 0.0),
        )
        self.assertEqual(
            revisit_events(DEFAULT_8Q_HARD_REVISIT),
            (
                {"query_index": 0, "first_turn": 1, "revisit_turn": 4},
                {"query_index": 1, "first_turn": 1, "revisit_turn": 4},
            ),
        )

    def test_default_challenge_revisit_avoids_the_easy_middle_pair(self) -> None:
        validate_schedule(DEFAULT_8Q_CHALLENGE_REVISIT, key_count=8)

        self.assertEqual(
            span_names_by_turn(DEFAULT_8Q_CHALLENGE_REVISIT),
            (
                ("needle_7", "needle_8"),
                ("needle_1", "needle_2"),
                ("needle_3", "needle_4"),
                ("needle_1", "needle_2"),
                ("needle_3", "needle_4"),
            ),
        )
        self.assertEqual(per_turn_overlap(DEFAULT_8Q_CHALLENGE_REVISIT), (0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(
            revisit_events(DEFAULT_8Q_CHALLENGE_REVISIT),
            (
                {"query_index": 0, "first_turn": 1, "revisit_turn": 3},
                {"query_index": 1, "first_turn": 1, "revisit_turn": 3},
                {"query_index": 2, "first_turn": 2, "revisit_turn": 4},
                {"query_index": 3, "first_turn": 2, "revisit_turn": 4},
            ),
        )

    def test_normalize_active_sets_requires_non_empty_turns(self) -> None:
        self.assertEqual(normalize_active_sets([[2, 1, 1], [3]]), ((1, 2), (3,)))
        with self.assertRaisesRegex(ValueError, "turn 0"):
            normalize_active_sets([[], [1]])
        with self.assertRaisesRegex(ValueError, "at least one"):
            normalize_active_sets([])

    def test_per_turn_recovery_scores_requested_keys(self) -> None:
        schedule = MultiTurnSchedule(
            name="recovery",
            base_task_key="mq_niah_4q",
            turns=((0, 1), (2, 3), (0, 1)),
        )

        self.assertEqual(
            per_turn_recovery(schedule, active_by_turn=((0, 1), (2,), (1, 3))),
            (1.0, 0.5, 0.5),
        )
        with self.assertRaisesRegex(ValueError, "length"):
            per_turn_recovery(schedule, active_by_turn=((0, 1),))

    def test_per_turn_active_churn_tracks_added_removed_and_jaccard(self) -> None:
        churn = per_turn_active_churn(((0, 1), (1, 2), (2, 3)))

        self.assertEqual(churn[0], {"turn": 0.0, "added": 2.0, "removed": 0.0, "jaccard": 1.0})
        self.assertEqual(churn[1]["added"], 1.0)
        self.assertEqual(churn[1]["removed"], 1.0)
        self.assertAlmostEqual(churn[1]["jaccard"], 1 / 3)

    def test_cache_state_grid_marks_active_and_requested_cells(self) -> None:
        schedule = MultiTurnSchedule(
            name="grid",
            base_task_key="mq_niah_4q",
            turns=((0, 1), (2,), (0,)),
        )
        rows = cache_state_grid(schedule, active_by_turn=((0, 1), (1, 2), (0, 3)), key_count=4)

        requested_active = [
            row
            for row in rows
            if row["turn"] == 1 and row["key_index"] == 2
        ][0]
        inactive_revisit = [
            row
            for row in rows
            if row["turn"] == 2 and row["key_index"] == 1
        ][0]

        self.assertEqual(len(rows), 12)
        self.assertTrue(requested_active["requested"])
        self.assertTrue(requested_active["active"])
        self.assertFalse(inactive_revisit["requested"])
        self.assertFalse(inactive_revisit["active"])

    def test_summarize_score_trajectory_reports_paired_noninitial_and_revisit_gain(self) -> None:
        rows = [
            {"example_index": 0, "turn": 0, "condition": "Matched", "score": 0.5},
            {"example_index": 0, "turn": 1, "condition": "Matched", "score": 0.25},
            {"example_index": 0, "turn": 2, "condition": "Matched", "score": 0.25},
            {"example_index": 1, "turn": 0, "condition": "Matched", "score": 0.75},
            {"example_index": 1, "turn": 1, "condition": "Matched", "score": 0.25},
            {"example_index": 1, "turn": 2, "condition": "Matched", "score": 0.50},
            {"example_index": 0, "turn": 0, "condition": "IdleKV", "score": 0.5},
            {"example_index": 0, "turn": 1, "condition": "IdleKV", "score": 0.75},
            {"example_index": 0, "turn": 2, "condition": "IdleKV", "score": 1.0},
            {"example_index": 1, "turn": 0, "condition": "IdleKV", "score": 0.75},
            {"example_index": 1, "turn": 1, "condition": "IdleKV", "score": 0.50},
            {"example_index": 1, "turn": 2, "condition": "IdleKV", "score": 0.75},
        ]

        summary = summarize_score_trajectory(
            rows,
            revisit_turns=(2,),
            condition_order=("Matched", "IdleKV"),
        )
        by_condition = {row["condition"]: row for row in summary}

        self.assertEqual(by_condition["Matched"]["mean_gain_vs_matched"], 0.0)
        self.assertEqual(by_condition["IdleKV"]["mean_gain_vs_matched"], 0.291667)
        self.assertEqual(by_condition["IdleKV"]["mean_noninitial_gain_vs_matched"], 0.4375)
        self.assertEqual(by_condition["IdleKV"]["mean_revisit_gain_vs_matched"], 0.5)
        self.assertEqual(by_condition["IdleKV"]["win_rate_vs_matched"], 0.666667)
        self.assertEqual(by_condition["IdleKV"]["n_paired_rows"], 6)

        with self.assertRaisesRegex(ValueError, "example_index"):
            summarize_score_trajectory([{"turn": 0, "condition": "Matched", "score": 1.0}])

    def test_evaluate_multiturn_summary_rows_gates_main_candidate(self) -> None:
        rows = [
            {
                "k": 96,
                "condition": "IdleKV",
                "mean_noninitial_gain_vs_matched": 0.25,
                "mean_revisit_gain_vs_matched": 0.5,
                "win_rate_vs_matched": 0.75,
            },
            {
                "k": 96,
                "condition": "Random-K",
                "mean_noninitial_gain_vs_matched": 0.05,
            },
            {
                "k": 96,
                "condition": "Oldest-K",
                "mean_noninitial_gain_vs_matched": 0.0,
            },
        ]

        recommendation = evaluate_multiturn_summary_rows(rows)[0]

        self.assertEqual(recommendation["action"], "main_candidate_if_artifact_checks_pass")
        self.assertEqual(recommendation["control_margin"], 0.2)

    def test_evaluate_multiturn_summary_rows_rejects_no_revisit_gain(self) -> None:
        rows = [
            {
                "k": 48,
                "condition": "IdleKV",
                "mean_noninitial_gain_vs_matched": 0.2,
                "mean_revisit_gain_vs_matched": 0.0,
                "win_rate_vs_matched": 0.75,
            }
        ]

        recommendation = evaluate_multiturn_summary_rows(rows)[0]

        self.assertEqual(recommendation["action"], "do_not_promote_no_revisit_gain")

    def test_evaluate_multiturn_summary_rows_rejects_when_stale_query_closes_gap(self) -> None:
        rows = [
            {
                "k": 96,
                "condition": "IdleKV",
                "mean_noninitial_gain_vs_matched": 0.3,
                "mean_revisit_gain_vs_matched": 0.5,
                "win_rate_vs_matched": 0.75,
            },
            {
                "k": 96,
                "condition": "Random-K",
                "mean_noninitial_gain_vs_matched": 0.0,
            },
            {
                "k": 96,
                "condition": "Oldest-K",
                "mean_noninitial_gain_vs_matched": 0.0,
            },
            {
                "k": 96,
                "condition": "StaleQ-K",
                "mean_noninitial_gain_vs_matched": 0.3,
            },
        ]

        recommendation = evaluate_multiturn_summary_rows(rows)[0]

        self.assertEqual(recommendation["action"], "do_not_promote_stale_query_closes_gap")
        self.assertEqual(recommendation["stale_margin"], 0.0)


if __name__ == "__main__":
    unittest.main()
