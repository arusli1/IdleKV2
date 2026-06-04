"""Unit tests for the Phase 6 runner helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache
from phases.phase6_repair.src.runner import (
    DEFAULT_WRONG_QUERY_DONOR_OFFSET,
    _backup_existing_artifact,
    _build_gold_span_oracle_candidates,
    _choose_gold_span_oracle_candidate,
    _artifact_path,
    _condition_label,
    _contiguous_run_count,
    _extract_lexical_anchor_terms,
    _select_file_gated_idlekv_positions,
    _select_lexical_anchor_positions,
    _jaccard_fraction,
    _needs_q2_candidate_scores,
    _restore_positions,
    _select_anchor_window_positions,
    _select_segment_positions,
    _wrong_query_ids_by_split,
    build_config,
    summarize_rows,
)


def _make_cache(positions: list[int]) -> PositionTrackedCache:
    seq_len = len(positions)
    keys = torch.arange(seq_len, dtype=torch.float16).reshape(1, 1, seq_len, 1)
    values = (keys + 100).clone()
    return PositionTrackedCache(((keys, values),), positions)


class _CharTokenizer:
    def __call__(self, text: str, **_kwargs):
        return {"offset_mapping": [(index, index + 1) for index, _char in enumerate(text)]}


class Phase6RunnerTests(unittest.TestCase):
    def test_backup_existing_artifact_copies_prior_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "test_artifact.json"
            artifact.write_text(json.dumps({"old": True}), encoding="utf-8")

            backup = _backup_existing_artifact(artifact)

            self.assertIsNotNone(backup)
            assert backup is not None
            self.assertTrue(backup.exists())
            self.assertEqual(json.loads(backup.read_text(encoding="utf-8")), {"old": True})

    def test_build_config_uses_stage_defaults_and_overrides(self) -> None:
        config = build_config(
            stage="smoke",
            task="clean_suite",
            num_samples=7,
            k_values=[24],
            conditions=[
                "A",
                "B",
                "B_match",
                "IdleKV",
                "FileGatedIdleKV-K",
                "IdleKV-Coverage",
                "IdleKV-MMR",
                "WrongQ-K",
                "Refresh-K",
                "Oracle-K",
            ],
            base_context_budget=640,
            recency_window=64,
        )
        self.assertEqual(config.stage, "smoke")
        self.assertEqual(config.task, "clean_suite")
        self.assertEqual(len(config.split_specs), 3)
        self.assertEqual(config.num_samples, 7)
        self.assertEqual(config.k_values, (24,))
        self.assertEqual(
            config.conditions,
            (
                "A",
                "B",
                "B_match",
                "IdleKV",
                "FileGatedIdleKV-K",
                "IdleKV-Coverage",
                "IdleKV-MMR",
                "WrongQ-K",
                "Refresh-K",
                "Oracle-K",
            ),
        )
        self.assertEqual(config.base_context_budget, 640)
        self.assertEqual(config.recency_window, 64)
        self.assertEqual(config.wrong_query_mode, "phantom_key")
        self.assertEqual(config.wrong_query_donor_offset, DEFAULT_WRONG_QUERY_DONOR_OFFSET)
        self.assertTrue(config.model_dir.endswith("models/Qwen2.5-7B-Instruct"))
        self.assertEqual(config.initial_compressor, "snapkv")

        sixq_suite = build_config(stage="smoke", task="mq_niah_6q_clean_suite")
        self.assertEqual(sixq_suite.task, "mq_niah_6q_clean_suite")
        self.assertEqual(len(sixq_suite.split_specs), 4)

        twoq_suite = build_config(stage="smoke", task="mq_niah_2q_clean_suite")
        self.assertEqual(twoq_suite.task, "mq_niah_2q_clean_suite")
        self.assertEqual(len(twoq_suite.split_specs), 1)

        threeq_suite = build_config(stage="smoke", task="mq_niah_3q_clean_suite")
        self.assertEqual(threeq_suite.task, "mq_niah_3q_clean_suite")
        self.assertEqual(len(threeq_suite.split_specs), 1)

        eightq_suite = build_config(stage="smoke", task="mq_niah_8q_clean_suite")
        self.assertEqual(eightq_suite.task, "mq_niah_8q_clean_suite")
        self.assertEqual(len(eightq_suite.split_specs), 5)

        donor_wrong_query = build_config(
            stage="smoke",
            task="clean_suite",
            wrong_query_mode="donor_q2",
            wrong_query_donor_offset=123,
        )
        self.assertEqual(donor_wrong_query.wrong_query_mode, "donor_q2")
        self.assertEqual(donor_wrong_query.wrong_query_donor_offset, 123)

        streaming = build_config(
            stage="smoke",
            task="clean_suite",
            initial_compressor="streaming_llm",
        )
        self.assertEqual(streaming.initial_compressor, "streaming_llm")

        h2o = build_config(
            stage="smoke",
            task="clean_suite",
            initial_compressor="h2o",
        )
        self.assertEqual(h2o.initial_compressor, "h2o")

        scissorhands = build_config(
            stage="smoke",
            task="clean_suite",
            initial_compressor="scissorhands",
        )
        self.assertEqual(scissorhands.initial_compressor, "scissorhands")

    def test_build_config_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            build_config(stage="smoke", task="mq_niah_4q_split_14_to_23", k_values=[0])
        with self.assertRaises(ValueError):
            build_config(stage="smoke", task="mq_niah_4q_split_14_to_23", conditions=["A", "Bad"])
        with self.assertRaises(ValueError):
            build_config(stage="smoke", task="mq_niah_4q_split_14_to_23", base_context_budget=0)
        with self.assertRaises(ValueError):
            build_config(stage="smoke", task="mq_niah_4q_split_14_to_23", recency_window=-1)
        with self.assertRaises(ValueError):
            build_config(stage="smoke", task="mq_niah_4q_split_14_to_23", query_scoring_mode="bad")
        with self.assertRaises(ValueError):
            build_config(stage="smoke", task="mq_niah_4q_split_14_to_23", oracle_mode="bad")
        with self.assertRaises(ValueError):
            build_config(stage="smoke", task="mq_niah_4q_split_14_to_23", wrong_query_mode="bad")
        with self.assertRaises(ValueError):
            build_config(stage="smoke", task="mq_niah_4q_split_14_to_23", wrong_query_donor_offset=0)
        with self.assertRaises(ValueError):
            build_config(stage="smoke", task="mq_niah_4q_split_14_to_23", model_dir="/path/that/does/not/exist")
        with self.assertRaises(ValueError):
            build_config(stage="smoke", task="mq_niah_4q_split_14_to_23", initial_compressor="bad")

    def test_artifact_path_includes_budget_and_recency(self) -> None:
        config = build_config(stage="smoke", task="clean_suite", num_samples=3, k_values=[12], base_context_budget=768, recency_window=64)
        self.assertIn("clean_suite_b768_r64_n3_k12_c", str(_artifact_path(config)))

    def test_file_gated_idlekv_selects_file_rows_then_global_backfill(self) -> None:
        selected, meta = _select_file_gated_idlekv_positions(
            evicted_positions=range(10),
            segment_token_ranges=[("file:pkg/mod.py", 4, 6)],
            segment_name="file:pkg/mod.py",
            q2_scores={4: 1.0, 5: 0.9, 8: 0.8, 9: 0.7},
            turn_n_scores={},
            k=4,
            left=0,
            right=0,
        )

        self.assertEqual(selected, [4, 5, 8, 9])
        self.assertEqual(meta["candidate_count"], 2)
        self.assertEqual(meta["selected_from_file_count"], 2)
        self.assertEqual(meta["backfill_count"], 2)
        self.assertEqual(meta["selected_from_file_fraction"], 0.5)
        self.assertTrue(meta["budget_matched"])

    def test_contiguous_run_count_deduplicates_and_counts_windows(self) -> None:
        self.assertEqual(_contiguous_run_count([]), 0)
        self.assertEqual(_contiguous_run_count([5, 4, 4, 7, 8, 10]), 3)

    def test_lexical_anchor_terms_exclude_answer_variants_and_common_words(self) -> None:
        terms = _extract_lexical_anchor_terms(
            repair_cue=(
                "Tool event: a repository check failed while executing "
                "`pkg/base_finder.py` inside `run_check`. The statement "
                "`<identifier>(payload, AppGroup)` must be recovered."
            ),
            answer="BaseFinder",
        )

        lowered = {term.lower() for term in terms}
        self.assertNotIn("tool", lowered)
        self.assertNotIn("identifier", lowered)
        self.assertNotIn("base_finder", lowered)
        self.assertNotIn("base", lowered)
        self.assertNotIn("finder", lowered)
        self.assertIn("run_check", lowered)
        self.assertIn("payload", lowered)

    def test_lexical_anchor_backfill_count_tracks_rows_after_anchor_bursts(self) -> None:
        rendered = "call foo then continue"
        selected, meta = _select_lexical_anchor_positions(
            tokenizer=_CharTokenizer(),
            prepared=SimpleNamespace(rendered_context=rendered),
            evicted_positions=range(len(rendered)),
            segment_token_ranges=[("file:pkg/mod.py", 0, len(rendered))],
            segment_name="file:pkg/mod.py",
            repair_cue="Tool event: function foo failed.",
            answer="bar",
            k=8,
            left=1,
            right=1,
        )

        self.assertEqual(len(selected), 8)
        self.assertEqual(meta["anchor_position_count"], 3)
        self.assertEqual(meta["backfill_count"], 3)
        self.assertTrue(meta["budget_matched"])

    def test_artifact_path_includes_nonzero_seed_offset(self) -> None:
        config = build_config(
            stage="smoke",
            task="clean_suite",
            num_samples=3,
            k_values=[12],
            base_context_budget=768,
            recency_window=64,
            dataset_seed_offset=1000,
        )
        self.assertIn("clean_suite_b768_r64_seed1000_n3_k12_c", str(_artifact_path(config)))

    def test_artifact_path_includes_nondefault_modes(self) -> None:
        config = build_config(
            stage="smoke",
            task="clean_suite",
            num_samples=3,
            k_values=[12],
            base_context_budget=768,
            recency_window=64,
            query_scoring_mode="exact_q",
            oracle_mode="gold_spans",
        )
        self.assertIn("clean_suite_b768_r64_qexact_q_ogold_spans_n3_k12_c", str(_artifact_path(config)))

    def test_artifact_path_includes_donor_wrong_query_mode(self) -> None:
        config = build_config(
            stage="smoke",
            task="clean_suite",
            num_samples=3,
            k_values=[12],
            base_context_budget=768,
            recency_window=64,
            wrong_query_mode="donor_q2",
            wrong_query_donor_offset=123,
        )
        self.assertIn("clean_suite_b768_r64_wqdonor_q2_wqd123_n3_k12_c", str(_artifact_path(config)))

    def test_artifact_path_includes_nondefault_model_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "Llama-3.1-8B-Instruct"
            model_dir.mkdir()
            config = build_config(
                stage="smoke",
                task="clean_suite",
                num_samples=3,
                k_values=[12],
                base_context_budget=768,
                recency_window=64,
                model_dir=model_dir,
            )

            self.assertEqual(Path(config.model_dir), model_dir)
            self.assertIn("_mllama318binstruct_n3_k12_c", str(_artifact_path(config)))

    def test_artifact_path_includes_nondefault_initial_compressor(self) -> None:
        config = build_config(
            stage="smoke",
            task="clean_suite",
            num_samples=3,
            k_values=[12],
            base_context_budget=768,
            recency_window=64,
            initial_compressor="streaming_llm",
        )
        self.assertIn("clean_suite_b768_r64_istreaming_llm_n3_k12_c", str(_artifact_path(config)))

        h2o_config = build_config(
            stage="smoke",
            task="clean_suite",
            num_samples=3,
            k_values=[12],
            base_context_budget=768,
            recency_window=64,
            initial_compressor="h2o",
        )
        self.assertIn("clean_suite_b768_r64_ih2o_n3_k12_c", str(_artifact_path(h2o_config)))

        scissorhands_config = build_config(
            stage="smoke",
            task="clean_suite",
            num_samples=3,
            k_values=[12],
            base_context_budget=768,
            recency_window=64,
            initial_compressor="scissorhands",
        )
        self.assertIn("clean_suite_b768_r64_iscissorhands_n3_k12_c", str(_artifact_path(scissorhands_config)))

    def test_condition_label_strips_symbols(self) -> None:
        label = _condition_label(["A", "B_match", "IdleKV", "IdleKV-Coverage", "WrongQ-K", "Refresh-K", "Oracle-K"])
        self.assertEqual(label, "a-bmatch-idlekv-idlekvcoverage-wrongqk-refreshk-oraclek")

    def test_needs_q2_candidate_scores_only_for_repair_scored_conditions(self) -> None:
        self.assertFalse(_needs_q2_candidate_scores(["A", "B", "B_match", "Random-K", "Oldest-K"]))
        self.assertFalse(_needs_q2_candidate_scores(["WrongQ-K", "StaleQ-K"]))
        self.assertTrue(_needs_q2_candidate_scores(["IdleKV"]))
        self.assertTrue(_needs_q2_candidate_scores(["IdleKV-Coverage"]))
        self.assertTrue(_needs_q2_candidate_scores(["IdleKV-MMR"]))
        self.assertTrue(_needs_q2_candidate_scores(["ContrastiveQ-K"]))
        self.assertTrue(_needs_q2_candidate_scores(["Refresh-K"]))
        self.assertTrue(_needs_q2_candidate_scores(["Oracle-K"]))
        self.assertTrue(_needs_q2_candidate_scores(["FileGatedIdleKV-K"]))

    @patch("phases.phase6_repair.src.runner.build_mismatched_question_ids")
    def test_wrong_query_ids_by_split_keeps_legacy_phantom_key_mode(self, mock_build_mismatched) -> None:
        config = build_config(stage="smoke", task="clean_suite", wrong_query_mode="phantom_key")
        split_views = [
            SimpleNamespace(split_spec=split_spec, base_example=object())
            for split_spec in config.split_specs[:2]
        ]
        ids_by_name = {
            split.split_spec.name: torch.tensor([slot], dtype=torch.long)
            for slot, split in enumerate(split_views)
        }

        def fake_mismatch(*, base_example, split_spec, tokenizer):
            return ids_by_name[split_spec.name]

        mock_build_mismatched.side_effect = fake_mismatch
        actual = _wrong_query_ids_by_split(split_views, tokenizer="tok", config=config, index=5)

        self.assertEqual(set(actual), set(ids_by_name))
        for name, expected_ids in ids_by_name.items():
            self.assertTrue(torch.equal(actual[name], expected_ids))
        self.assertEqual(mock_build_mismatched.call_count, 2)

    @patch("phases.phase6_repair.src.runner.build_split_prepared_from_base_example")
    @patch("phases.phase6_repair.src.runner.build_base_example")
    def test_wrong_query_ids_by_split_can_use_donor_q2_questions(
        self,
        mock_build_base,
        mock_build_split,
    ) -> None:
        config = build_config(
            stage="smoke",
            task="clean_suite",
            wrong_query_mode="donor_q2",
            wrong_query_donor_offset=17,
        )
        split_views = [
            SimpleNamespace(split_spec=split_spec, base_example=object())
            for split_spec in config.split_specs
        ]
        donor_base = object()
        mock_build_base.return_value = donor_base
        ids_by_name = {
            split_spec.name: torch.tensor([slot + 10], dtype=torch.long)
            for slot, split_spec in enumerate(config.split_specs)
        }

        def fake_build_split(*, base_example, split_spec, tokenizer):
            self.assertIs(base_example, donor_base)
            self.assertEqual(tokenizer, "tok")
            return SimpleNamespace(
                split_spec=split_spec,
                q2_prepared=SimpleNamespace(question_ids=ids_by_name[split_spec.name]),
            )

        mock_build_split.side_effect = fake_build_split
        actual = _wrong_query_ids_by_split(split_views, tokenizer="tok", config=config, index=5)

        mock_build_base.assert_called_once_with(
            split_spec=config.split_specs[0],
            index=22,
            context_length=config.context_length,
            tokenizer="tok",
            dataset_seed_offset=config.dataset_seed_offset,
        )
        self.assertEqual(set(actual), set(ids_by_name))
        for name, expected_ids in ids_by_name.items():
            self.assertTrue(torch.equal(actual[name], expected_ids))

    def test_restore_positions_injects_selected_tokens_in_sorted_order(self) -> None:
        active = _make_cache([0, 3])
        evicted = _make_cache([1, 2, 4])
        repaired, timing = _restore_positions(
            active_cache=active,
            evicted_cache=evicted,
            selected_positions=[4, 1],
        )

        self.assertEqual(repaired.positions, [0, 1, 3, 4])
        self.assertEqual(int(timing["restored_count"]), 2)
        self.assertGreaterEqual(float(timing["transfer_ms"]), 0.0)
        self.assertGreaterEqual(float(timing["inject_ms"]), 0.0)

    def test_jaccard_fraction_handles_overlap_and_empty_sets(self) -> None:
        self.assertEqual(_jaccard_fraction([], []), 1.0)
        self.assertAlmostEqual(_jaccard_fraction([1, 2, 3], [2, 3, 4]), 0.5)

    def test_select_segment_positions_backfills_to_keep_budget_matched(self) -> None:
        selected = _select_segment_positions(
            evicted_positions=[1, 2, 3, 4, 5],
            segment_token_ranges=[("file:target.py", 3, 5)],
            segment_name="file:target.py",
            k=4,
        )

        self.assertEqual(selected, [3, 4, 1, 2])

    def test_select_anchor_window_positions_prefers_nearest_evicted_rows(self) -> None:
        selected = _select_anchor_window_positions(
            evicted_positions=[1, 2, 10, 11, 12, 20],
            anchor_positions=[11],
            k=4,
        )

        self.assertEqual(selected, [11, 10, 12, 2])

    def test_summarize_rows_reports_lifts_and_overlap(self) -> None:
        rows = [
            {
                "k": 100,
                "task": "mq_niah_4q_split_14_to_23",
                "q1_score": 1.0,
                "condition_a_score": 1.0,
                "condition_b_score": 0.0,
                "b_match_score": 0.0,
                "idlekv_score": 1.0,
                "idlekv_overlap_fraction": 0.5,
                "idlekv_active_overlap_fraction": 0.75,
                "idlekv_selection_s": 0.001,
                "idlekv_transfer_ms": 2.0,
                "idlekv_inject_ms": 1.0,
                "idlekv_coverage_score": 1.0,
                "idlekv_coverage_overlap_fraction": 0.75,
                "idlekv_mmr_score": 0.5,
                "idlekv_mmr_overlap_fraction": 0.25,
                "wrong_q_k_score": 0.0,
                "refresh_k_score": 1.0,
                "refresh_k_overlap_fraction": 0.75,
                "anchor_window_k_score": 0.5,
                "oracle_k_score": 1.0,
            },
            {
                "k": 100,
                "task": "mq_niah_4q_split_14_to_23",
                "q1_score": 1.0,
                "condition_a_score": 1.0,
                "condition_b_score": 0.5,
                "b_match_score": 0.5,
                "idlekv_score": 0.5,
                "idlekv_overlap_fraction": 0.0,
                "idlekv_active_overlap_fraction": 0.25,
                "idlekv_selection_s": 0.002,
                "idlekv_transfer_ms": 3.0,
                "idlekv_inject_ms": 1.0,
                "idlekv_coverage_score": 0.75,
                "idlekv_coverage_overlap_fraction": 0.5,
                "idlekv_mmr_score": 1.0,
                "idlekv_mmr_overlap_fraction": 0.75,
                "wrong_q_k_score": 0.5,
                "refresh_k_score": 0.75,
                "refresh_k_overlap_fraction": 0.5,
                "anchor_window_k_score": 1.0,
                "oracle_k_score": 1.0,
            },
        ]

        summary = summarize_rows(rows)
        k100 = summary["k100"]
        self.assertEqual(k100["mean_condition_b"], 0.25)
        self.assertEqual(k100["mean_b_match"], 0.25)
        self.assertEqual(k100["mean_idlekv"], 0.75)
        self.assertEqual(k100["mean_selection_lift"], 0.5)
        self.assertEqual(k100["pct_idlekv_gt_b_match"], 0.5)
        self.assertEqual(k100["pct_idlekv_lt_b_match"], 0.0)
        self.assertEqual(k100["mean_idlekv_active_overlap_fraction"], 0.5)
        self.assertEqual(k100["mean_idlekv_coverage"], 0.875)
        self.assertEqual(k100["mean_idlekv_coverage_lift"], 0.625)
        self.assertEqual(k100["pct_idlekv_coverage_gt_b_match"], 1.0)
        self.assertEqual(k100["mean_idlekv_coverage_overlap_fraction"], 0.625)
        self.assertEqual(k100["mean_idlekv_mmr"], 0.75)
        self.assertEqual(k100["mean_idlekv_mmr_lift"], 0.5)
        self.assertEqual(k100["pct_idlekv_mmr_gt_b_match"], 1.0)
        self.assertEqual(k100["mean_idlekv_mmr_overlap_fraction"], 0.5)
        self.assertEqual(k100["mean_wrong_q_k"], 0.25)
        self.assertEqual(k100["mean_refresh_k"], 0.875)
        self.assertEqual(k100["mean_refresh_lift"], 0.625)
        self.assertEqual(k100["mean_refresh_overlap_fraction"], 0.625)
        self.assertEqual(k100["mean_anchor_window_k"], 0.75)
        self.assertEqual(k100["mean_anchor_window_lift"], 0.5)
        self.assertEqual(k100["mean_oracle_k"], 1.0)

    def test_summarize_rows_groups_multi_split_runs(self) -> None:
        rows = [
            {
                "k": 12,
                "task": "mq_niah_4q_split_14_to_23",
                "q1_score": 1.0,
                "condition_a_score": 1.0,
                "condition_b_score": 0.0,
                "b_match_score": 0.0,
                "idlekv_score": 0.5,
                "idlekv_overlap_fraction": 0.0,
                "idlekv_selection_s": 0.001,
                "idlekv_transfer_ms": 1.0,
                "idlekv_inject_ms": 1.0,
            },
            {
                "k": 12,
                "task": "mq_niah_4q_split_24_to_13",
                "q1_score": 1.0,
                "condition_a_score": 1.0,
                "condition_b_score": 0.0,
                "b_match_score": 0.0,
                "idlekv_score": 1.0,
                "idlekv_overlap_fraction": 1.0,
                "idlekv_selection_s": 0.001,
                "idlekv_transfer_ms": 1.0,
                "idlekv_inject_ms": 1.0,
            },
        ]

        summary = summarize_rows(rows)
        self.assertIn("overall", summary)
        self.assertIn("by_task", summary)
        self.assertEqual(summary["overall"]["k12"]["mean_idlekv"], 0.75)
        self.assertEqual(summary["by_task"]["mq_niah_4q_split_14_to_23"]["k12"]["mean_idlekv"], 0.5)

    def test_choose_gold_span_oracle_candidate_prefers_best_smaller_subset(self) -> None:
        candidates = [
            {"positions": (), "cost": 0, "score": 0.25, "q2_sum": 0.0, "turn_n_sum": 0.0},
            {"positions": (10, 11), "cost": 2, "score": 1.0, "q2_sum": 2.0, "turn_n_sum": 2.0},
            {"positions": (10, 11, 12), "cost": 3, "score": 0.5, "q2_sum": 3.0, "turn_n_sum": 3.0},
        ]

        chosen = _choose_gold_span_oracle_candidate(candidates=candidates, k=3)
        self.assertEqual(chosen["positions"], (10, 11))

    def test_choose_gold_span_oracle_candidate_is_monotone_in_budget_for_fixed_candidates(self) -> None:
        candidates = [
            {"positions": (), "cost": 0, "score": 0.0, "q2_sum": 0.0, "turn_n_sum": 0.0},
            {"positions": (10,), "cost": 1, "score": 0.25, "q2_sum": 1.0, "turn_n_sum": 1.0},
            {"positions": (10, 11), "cost": 2, "score": 0.5, "q2_sum": 2.0, "turn_n_sum": 2.0},
            {"positions": (10, 11, 12), "cost": 3, "score": 0.75, "q2_sum": 3.0, "turn_n_sum": 3.0},
        ]

        chosen_scores = [
            _choose_gold_span_oracle_candidate(candidates=candidates, k=k)["score"]
            for k in (0, 1, 2, 3)
        ]
        self.assertEqual(chosen_scores, [0.0, 0.25, 0.5, 0.75])

    @patch("phases.phase6_repair.src.runner._run_condition")
    @patch("phases.phase6_repair.src.runner._restore_positions")
    def test_build_gold_span_oracle_candidates_enumerates_group_subsets(
        self,
        mock_restore_positions,
        mock_run_condition,
    ) -> None:
        active = _make_cache([0])
        evicted = _make_cache([10, 11, 12])

        def fake_restore(*, active_cache, evicted_cache, selected_positions):
            positions = list(active_cache.positions) + list(selected_positions)
            cache = _make_cache(positions)
            return cache, {"transfer_ms": 0.0, "inject_ms": 0.0, "restored_count": float(len(selected_positions))}

        def fake_run(*, model, tokenizer, prepared, cache):
            selected = tuple(sorted(position for position in cache.positions if position >= 10))
            score_map = {
                (): 0.25,
                (10, 11): 1.0,
                (12,): 0.5,
                (10, 11, 12): 0.75,
            }
            return str(selected), score_map[selected], 0.01

        mock_restore_positions.side_effect = fake_restore
        mock_run_condition.side_effect = fake_run

        candidates = _build_gold_span_oracle_candidates(
            model=None,
            tokenizer=None,
            prepared=None,
            active_cache=active,
            evicted_cache=evicted,
            relevant_position_groups=[(10, 11), (12,)],
            q2_scores={10: 1.0, 11: 1.0, 12: 0.5},
            turn_n_scores={10: 0.1, 11: 0.1, 12: 0.1},
            base_output="base",
            base_score=0.25,
            base_generation_s=0.01,
        )

        self.assertEqual({tuple(candidate["positions"]) for candidate in candidates}, {(), (10, 11), (12,), (10, 11, 12)})
        best_k2 = _choose_gold_span_oracle_candidate(candidates=candidates, k=2)
        best_k3 = _choose_gold_span_oracle_candidate(candidates=candidates, k=3)
        self.assertEqual(best_k2["positions"], (10, 11))
        self.assertEqual(best_k3["positions"], (10, 11))


if __name__ == "__main__":
    unittest.main()
