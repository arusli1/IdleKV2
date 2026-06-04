"""Unit tests for Phase 6 reporting helpers."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from phases.phase6_repair.scripts.export_phase6_frontier import _extract_export_sections, _merge_frontier_ci
from phases.phase6_repair.src.reporting import (
    bootstrap_frontier_rows,
    overlap_rows,
    frontier_rows,
    runtime_rows,
    split_rows,
    write_csv,
)

AUDIT_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "phase7_broader_evidence"
    / "scripts"
    / "audit_phase7_artifact.py"
)
_AUDIT_SPEC = importlib.util.spec_from_file_location("phase7_audit_phase7_artifact", AUDIT_SCRIPT_PATH)
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)


class Phase6ReportingTests(unittest.TestCase):
    def test_frontier_rows_sort_k_labels_and_map_metrics(self) -> None:
        rows = frontier_rows(
            {
                "k32": {"mean_condition_a": 1.0, "mean_idlekv": 0.6, "mean_wrong_q_k": 0.4, "mean_b_match": 0.1},
                "k8": {"mean_condition_a": 0.9, "mean_idlekv": 0.3, "mean_wrong_q_k": 0.2, "mean_b_match": 0.1},
            }
        )
        self.assertEqual([row["k"] for row in rows], [8, 32])
        self.assertEqual(rows[0]["condition_a"], 0.9)
        self.assertEqual(rows[0]["idlekv"], 0.3)
        self.assertEqual(rows[0]["wrong_q_k"], 0.2)
        self.assertEqual(rows[1]["b_match"], 0.1)

    def test_split_rows_attach_task_name(self) -> None:
        rows = split_rows(
            {
                "task_a": {"k8": {"mean_idlekv": 0.25}},
                "task_b": {"k16": {"mean_idlekv": 0.5}},
            }
        )
        self.assertEqual(rows[0]["task"], "task_a")
        self.assertEqual(rows[1]["task"], "task_b")
        self.assertEqual(rows[1]["k"], 16)

    def test_write_csv_uses_first_row_field_order(self) -> None:
        rows = [
            {"k": 8, "b_match": 0.1, "idlekv": 0.3},
            {"k": 16, "b_match": 0.1, "idlekv": 0.4},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "frontier.csv"
            write_csv(rows, out_path)
            with out_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ["k", "b_match", "idlekv"])
                loaded = list(reader)
        self.assertEqual(loaded[1]["idlekv"], "0.4")

    def test_runtime_rows_aggregate_percentiles_and_components(self) -> None:
        rows = [
            {
                "task": "task_a",
                "k": 8,
                "q2_query_rows_s": 0.010,
                "q2_evicted_scoring_s": 0.020,
                "idlekv_selection_s": 0.030,
                "idlekv_transfer_ms": 4.0,
                "idlekv_inject_ms": 5.0,
                "wrong_q_query_rows_s": 0.012,
                "wrong_q_evicted_scoring_s": 0.022,
                "wrong_q_k_selection_s": 0.032,
                "wrong_q_k_transfer_ms": 8.0,
                "wrong_q_k_inject_ms": 9.0,
                "wrong_q_k_score": 0.4,
                "contrastive_q_k_selection_s": 0.040,
                "contrastive_q_k_transfer_ms": 12.0,
                "contrastive_q_k_inject_ms": 13.0,
                "contrastive_q_k_score": 0.6,
            },
            {
                "task": "task_a",
                "k": 8,
                "q2_query_rows_s": 0.011,
                "q2_evicted_scoring_s": 0.021,
                "idlekv_selection_s": 0.031,
                "idlekv_transfer_ms": 6.0,
                "idlekv_inject_ms": 7.0,
                "wrong_q_query_rows_s": 0.014,
                "wrong_q_evicted_scoring_s": 0.024,
                "wrong_q_k_selection_s": 0.034,
                "wrong_q_k_transfer_ms": 10.0,
                "wrong_q_k_inject_ms": 11.0,
                "wrong_q_k_score": 0.5,
                "contrastive_q_k_selection_s": 0.042,
                "contrastive_q_k_transfer_ms": 14.0,
                "contrastive_q_k_inject_ms": 15.0,
                "contrastive_q_k_score": 0.7,
            },
        ]
        summary = runtime_rows(rows, by_task=True)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["task"], "task_a")
        self.assertEqual(summary[0]["k"], 8)
        self.assertEqual(summary[0]["n"], 2)
        self.assertAlmostEqual(float(summary[0]["p50_query_ms"]), 10.5)
        self.assertAlmostEqual(float(summary[0]["p50_score_ms"]), 20.5)
        self.assertAlmostEqual(float(summary[0]["p50_select_ms"]), 30.5)
        self.assertAlmostEqual(float(summary[0]["p50_transfer_ms"]), 5.0)
        self.assertAlmostEqual(float(summary[0]["p50_inject_ms"]), 6.0)
        self.assertAlmostEqual(float(summary[0]["p50_wrong_q_query_ms"]), 13.0)
        self.assertAlmostEqual(float(summary[0]["p50_wrong_q_score_ms"]), 23.0)
        self.assertAlmostEqual(float(summary[0]["p50_wrong_q_select_ms"]), 33.0)
        self.assertAlmostEqual(float(summary[0]["p50_wrong_q_transfer_ms"]), 9.0)
        self.assertAlmostEqual(float(summary[0]["p50_wrong_q_inject_ms"]), 10.0)
        self.assertAlmostEqual(float(summary[0]["p50_contrastive_q_select_ms"]), 41.0)
        self.assertAlmostEqual(float(summary[0]["p50_contrastive_q_transfer_ms"]), 13.0)
        self.assertAlmostEqual(float(summary[0]["p50_contrastive_q_inject_ms"]), 14.0)

    def test_overlap_rows_mean_fields(self) -> None:
        rows = [
            {
                "task": "task_a",
                "k": 16,
                "condition_b_active_overlap_fraction": 0.05,
                "b_match_active_overlap_fraction": 0.1,
                "b_match_overlap_fraction": 0.1,
                "idlekv_active_overlap_fraction": 0.7,
                "idlekv_overlap_fraction": 0.4,
                "wrong_q_k_active_overlap_fraction": 0.35,
                "wrong_q_k_overlap_fraction": 0.3,
                "contrastive_q_k_active_overlap_fraction": 0.6,
                "contrastive_q_k_overlap_fraction": 0.5,
                "random_k_active_overlap_fraction": 0.25,
                "random_k_overlap_fraction": 0.2,
                "oldest_k_active_overlap_fraction": 0.15,
                "oldest_k_overlap_fraction": 0.1,
                "oracle_k_active_overlap_fraction": 0.85,
                "oracle_k_overlap_fraction": 0.8,
            },
            {
                "task": "task_a",
                "k": 16,
                "condition_b_active_overlap_fraction": 0.15,
                "b_match_active_overlap_fraction": 0.3,
                "b_match_overlap_fraction": 0.3,
                "idlekv_active_overlap_fraction": 0.9,
                "idlekv_overlap_fraction": 0.6,
                "wrong_q_k_active_overlap_fraction": 0.55,
                "wrong_q_k_overlap_fraction": 0.5,
                "contrastive_q_k_active_overlap_fraction": 0.8,
                "contrastive_q_k_overlap_fraction": 0.7,
                "random_k_active_overlap_fraction": 0.05,
                "random_k_overlap_fraction": 0.0,
                "oldest_k_active_overlap_fraction": 0.35,
                "oldest_k_overlap_fraction": 0.3,
                "oracle_k_active_overlap_fraction": 1.0,
                "oracle_k_overlap_fraction": 1.0,
            },
        ]
        summary = overlap_rows(rows, by_task=True)
        self.assertEqual(summary[0]["task"], "task_a")
        self.assertEqual(summary[0]["k"], 16)
        self.assertAlmostEqual(float(summary[0]["condition_b_overlap"]), 0.1)
        self.assertAlmostEqual(float(summary[0]["b_match_overlap"]), 0.2)
        self.assertAlmostEqual(float(summary[0]["idlekv_overlap"]), 0.8)
        self.assertAlmostEqual(float(summary[0]["wrong_q_k_overlap"]), 0.45)
        self.assertAlmostEqual(float(summary[0]["contrastive_q_k_overlap"]), 0.7)
        self.assertAlmostEqual(float(summary[0]["oracle_k_overlap"]), 0.925)

    def test_overlap_rows_reconstruct_active_overlap_from_positions(self) -> None:
        rows = [
            {
                "task": "task_b",
                "k": 32,
                "q2_relevant_positions": [1, 2, 10, 11],
                "b_kept_context_positions": [0, 1, 2, 3],
                "idlekv_selected_positions": [10],
                "wrong_q_k_selected_positions": [11],
                "contrastive_q_k_selected_positions": [10, 11],
                "random_k_selected_positions": [20],
                "oldest_k_selected_positions": [],
                "oracle_k_selected_positions": [10, 11],
                "b_match_overlap_fraction": 0.5,
            }
        ]
        summary = overlap_rows(rows, by_task=True)
        self.assertEqual(summary[0]["task"], "task_b")
        self.assertAlmostEqual(float(summary[0]["condition_b_overlap"]), 0.5)
        self.assertAlmostEqual(float(summary[0]["idlekv_overlap"]), 0.75)
        self.assertAlmostEqual(float(summary[0]["wrong_q_k_overlap"]), 0.75)
        self.assertAlmostEqual(float(summary[0]["contrastive_q_k_overlap"]), 1.0)
        self.assertAlmostEqual(float(summary[0]["random_k_overlap"]), 0.5)
        self.assertAlmostEqual(float(summary[0]["oldest_k_overlap"]), 0.5)
        self.assertAlmostEqual(float(summary[0]["oracle_k_overlap"]), 1.0)

    def test_overlap_rows_prefer_reconstructed_active_overlap_over_legacy_restored_only_fields(self) -> None:
        rows = [
            {
                "task": "task_c",
                "k": 64,
                "q2_relevant_positions": [1, 2, 10, 11],
                "b_kept_context_positions": [1, 2],
                "idlekv_selected_positions": [10, 11],
                "wrong_q_k_selected_positions": [20],
                "contrastive_q_k_selected_positions": [10],
                "random_k_selected_positions": [20],
                "oldest_k_selected_positions": [],
                "oracle_k_selected_positions": [10, 11],
                # Legacy restored-only fractions that should not override final-active reconstruction.
                "idlekv_overlap_fraction": 0.25,
                "wrong_q_k_overlap_fraction": 0.75,
                "contrastive_q_k_overlap_fraction": 0.0,
                "random_k_overlap_fraction": 0.0,
                "oldest_k_overlap_fraction": 0.0,
                "oracle_k_overlap_fraction": 0.5,
            }
        ]
        summary = overlap_rows(rows, by_task=True)
        self.assertAlmostEqual(float(summary[0]["condition_b_overlap"]), 0.5)
        self.assertAlmostEqual(float(summary[0]["idlekv_overlap"]), 1.0)
        self.assertAlmostEqual(float(summary[0]["wrong_q_k_overlap"]), 0.5)
        self.assertAlmostEqual(float(summary[0]["contrastive_q_k_overlap"]), 0.75)
        self.assertAlmostEqual(float(summary[0]["random_k_overlap"]), 0.5)
        self.assertAlmostEqual(float(summary[0]["oldest_k_overlap"]), 0.5)
        self.assertAlmostEqual(float(summary[0]["oracle_k_overlap"]), 1.0)

    def test_extract_export_sections_accepts_suite_style_artifacts(self) -> None:
        artifact = {
            "aggregate": {
                "overall": {"k8": {"mean_idlekv": 0.25}},
                "by_task": {"task_a": {"k8": {"mean_idlekv": 0.25}}},
            }
        }
        overall, by_task = _extract_export_sections(artifact)
        self.assertIn("k8", overall)
        self.assertIn("task_a", by_task)

    def test_extract_export_sections_wraps_single_task_artifacts(self) -> None:
        artifact = {
            "task": "task_single",
            "aggregate": {"k8": {"mean_idlekv": 0.25}},
            "rows": [{"task": "task_single", "k": 8}],
        }
        overall, by_task = _extract_export_sections(artifact)
        self.assertEqual(overall["k8"]["mean_idlekv"], 0.25)
        self.assertEqual(list(by_task.keys()), ["task_single"])
        self.assertEqual(by_task["task_single"]["k8"]["mean_idlekv"], 0.25)

    def test_bootstrap_frontier_rows_adds_confidence_bounds(self) -> None:
        rows = [
            {"task": "task_a", "k": 8, "condition_a_score": 1.0, "b_match_score": 0.0, "idlekv_score": 0.5, "wrong_q_k_score": 0.25, "contrastive_q_k_score": 0.5, "oracle_k_score": 1.0},
            {"task": "task_a", "k": 8, "condition_a_score": 0.5, "b_match_score": 0.5, "idlekv_score": 1.0, "wrong_q_k_score": 0.5, "contrastive_q_k_score": 0.75, "oracle_k_score": 1.0},
            {"task": "task_a", "k": 16, "condition_a_score": 1.0, "b_match_score": 0.0, "idlekv_score": 0.0, "wrong_q_k_score": 0.0, "contrastive_q_k_score": 0.25, "oracle_k_score": 0.5},
            {"task": "task_a", "k": 16, "condition_a_score": 1.0, "b_match_score": 0.0, "idlekv_score": 0.5, "wrong_q_k_score": 0.25, "contrastive_q_k_score": 0.5, "oracle_k_score": 1.0},
        ]
        summary = bootstrap_frontier_rows(rows, by_task=True, num_bootstrap=100, seed=7)
        self.assertEqual(summary[0]["task"], "task_a")
        self.assertEqual(summary[0]["k"], 8)
        self.assertIn("condition_a", summary[0])
        self.assertIn("condition_a_lo", summary[0])
        self.assertIn("idlekv_lo", summary[0])
        self.assertIn("idlekv_hi", summary[0])
        self.assertIn("wrong_q_k_lo", summary[0])
        self.assertAlmostEqual(float(summary[0]["idlekv"]), 0.75)
        self.assertAlmostEqual(float(summary[0]["wrong_q_k"]), 0.375)
        self.assertAlmostEqual(float(summary[0]["contrastive_q_k"]), 0.625)
        self.assertLessEqual(float(summary[0]["idlekv_lo"]), float(summary[0]["idlekv"]))
        self.assertGreaterEqual(float(summary[0]["idlekv_hi"]), float(summary[0]["idlekv"]))

    def test_merge_frontier_ci_preserves_existing_columns_and_adds_bounds(self) -> None:
        base_rows = [{"k": 8, "b_match": 0.25, "idlekv": 0.5}]
        ci_rows = [{"k": 8, "n": 2, "b_match": 0.25, "b_match_lo": 0.0, "b_match_hi": 0.5, "idlekv": 0.5, "idlekv_lo": 0.25, "idlekv_hi": 0.75}]
        merged = _merge_frontier_ci(base_rows, ci_rows, by_task=False)
        self.assertEqual(merged[0]["b_match"], 0.25)
        self.assertEqual(merged[0]["idlekv_hi"], 0.75)

    def test_phase7_audit_metric_at_k_reads_requested_point(self) -> None:
        curve = {"k8": {"mean_idlekv": 0.2}, "k96": {"mean_idlekv": 0.9}}
        self.assertEqual(_AUDIT_MODULE._metric_at_k(curve, k=96, key="mean_idlekv"), 0.9)

    def test_phase7_audit_example_diagnostics_group_by_task_and_example(self) -> None:
        rows = [
            {"task": "split_a", "example_id": 0, "k": 8, "idlekv_score": 1.0},
            {"task": "split_b", "example_id": 0, "k": 16, "idlekv_score": 0.0},
        ]
        non_monotone, total = _AUDIT_MODULE._example_level_monotonicity(rows, score_key="idlekv_score")
        self.assertEqual(total, 2)
        self.assertEqual(non_monotone, 0)

    def test_phase7_audit_sixq_acceptance_prints_gate_results(self) -> None:
        curve = {
            "k48": {
                "mean_condition_a": 0.97,
                "mean_b_match": 0.20,
                "mean_idlekv": 0.40,
                "mean_random_k": 0.20,
                "mean_oldest_k": 0.15,
                "mean_oracle_k": 0.50,
            },
            "k96": {
                "mean_condition_a": 0.98,
                "mean_b_match": 0.22,
                "mean_idlekv": 0.48,
                "mean_random_k": 0.20,
                "mean_oldest_k": 0.19,
                "mean_oracle_k": 0.60,
            },
            "k128": {
                "mean_condition_a": 0.98,
                "mean_b_match": 0.22,
                "mean_idlekv": 0.55,
                "mean_random_k": 0.21,
                "mean_oldest_k": 0.20,
                "mean_oracle_k": 0.60,
            },
        }
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _AUDIT_MODULE._print_sixq_acceptance(curve)
        output = buffer.getvalue()
        self.assertIn("[6q acceptance]", output)
        self.assertIn("PASS A @ K=128 >= 0.95", output)
        self.assertIn("PASS matched no-repair @ K=128 >= 0.10", output)

    def test_phase7_audit_sixq_ci_gate_reads_exported_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "overall.csv"
            csv_path.write_text(
                "k,b_match_hi,idlekv_lo\n"
                "48,0.30,0.40\n"
                "96,0.41,0.42\n"
                "128,0.45,0.50\n",
                encoding="utf-8",
            )
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                _AUDIT_MODULE._print_sixq_ci_gate(csv_path)
        output = buffer.getvalue()
        self.assertIn("[6q bootstrap CI gate]", output)
        self.assertIn("PASS IdleKV_lo @ K=96 > matched-no-repair_hi @ K=96", output)

    def test_phase7_audit_main_uses_artifact_task_name_for_sixq_gate(self) -> None:
        artifact = {
            "task": "mq_niah_6q_clean_suite",
            "aggregate": {
                "overall": {
                    "k48": {
                        "mean_condition_a": 0.97,
                        "mean_b_match": 0.20,
                        "mean_idlekv": 0.40,
                        "mean_random_k": 0.20,
                        "mean_oldest_k": 0.15,
                        "mean_oracle_k": 0.50,
                    },
                    "k96": {
                        "mean_condition_a": 0.98,
                        "mean_b_match": 0.22,
                        "mean_idlekv": 0.48,
                        "mean_random_k": 0.20,
                        "mean_oldest_k": 0.19,
                        "mean_oracle_k": 0.60,
                    },
                    "k128": {
                        "mean_condition_a": 0.98,
                        "mean_b_match": 0.22,
                        "mean_idlekv": 0.55,
                        "mean_random_k": 0.21,
                        "mean_oldest_k": 0.20,
                        "mean_oracle_k": 0.60,
                    },
                },
                "by_task": {
                    "split_without_6q_in_name": {
                        "k96": {
                            "mean_b_match": 0.20,
                            "mean_idlekv": 0.40,
                        },
                        "k128": {
                            "mean_b_match": 0.22,
                            "mean_idlekv": 0.55,
                            "mean_random_k": 0.20,
                            "mean_oldest_k": 0.19,
                        },
                    }
                },
            },
            "rows": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "artifact.json"
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            overall_csv = Path(tmpdir) / "overall.csv"
            overall_csv.write_text(
                "k,b_match_hi,idlekv_lo\n96,0.30,0.40\n",
                encoding="utf-8",
            )
            buffer = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "audit_phase7_artifact.py",
                    "--artifact",
                    str(artifact_path),
                    "--overall-csv",
                    str(overall_csv),
                ],
            ):
                with redirect_stdout(buffer):
                    _AUDIT_MODULE.main()
        output = buffer.getvalue()
        self.assertIn("[6q acceptance]", output)
        self.assertIn("[6q per-split gates]", output)
        self.assertIn("[6q bootstrap CI gate]", output)

    def test_phase7_audit_main_uses_config_task_name_for_phase6_artifacts(self) -> None:
        artifact = {
            "config": {"task": "mq_niah_6q_clean_suite"},
            "aggregate": {
                "overall": {
                    "k48": {
                        "mean_condition_a": 0.97,
                        "mean_b_match": 0.20,
                        "mean_idlekv": 0.40,
                        "mean_random_k": 0.20,
                        "mean_oldest_k": 0.15,
                        "mean_oracle_k": 0.50,
                    },
                    "k96": {
                        "mean_condition_a": 0.98,
                        "mean_b_match": 0.22,
                        "mean_idlekv": 0.48,
                        "mean_random_k": 0.20,
                        "mean_oldest_k": 0.19,
                        "mean_oracle_k": 0.60,
                    },
                    "k128": {
                        "mean_condition_a": 0.98,
                        "mean_b_match": 0.22,
                        "mean_idlekv": 0.55,
                        "mean_random_k": 0.21,
                        "mean_oldest_k": 0.20,
                        "mean_oracle_k": 0.60,
                    },
                },
                "by_task": {},
            },
            "rows": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "artifact.json"
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            buffer = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "audit_phase7_artifact.py",
                    "--artifact",
                    str(artifact_path),
                ],
            ):
                with redirect_stdout(buffer):
                    _AUDIT_MODULE.main()
        output = buffer.getvalue()
        self.assertIn("[6q acceptance]", output)


if __name__ == "__main__":
    unittest.main()
