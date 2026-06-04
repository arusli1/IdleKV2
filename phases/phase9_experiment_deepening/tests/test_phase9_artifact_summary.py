"""Tests for Phase 9 artifact summarization and smoke gates."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from phases.phase9_experiment_deepening.scripts.phase9_artifact_summary import (
    contrastive_gate,
    frontier_rows_for_artifact,
    future_query_gate,
    proxy_gate,
    stale_query_gate,
    write_csv,
)


class Phase9ArtifactSummaryTests(unittest.TestCase):
    def test_frontier_rows_compute_phase9_derived_columns(self) -> None:
        artifact = {
            "config": {
                "task": "mq_niah_6q_clean_suite",
                "stage": "smoke",
                "query_scoring_mode": "exact_q",
                "oracle_mode": "gold_spans",
                "wrong_query_mode": "donor_q2",
                "wrong_query_donor_offset": 100000,
                "base_context_budget": 18432,
                "recency_window": 128,
                "num_samples": 4,
                "conditions": ["A", "B", "B_match", "IdleKV", "WrongQ-K", "Oracle-K"],
            },
            "aggregate": {
                "overall": {
                    "k48": {
                        "mean_condition_a": 0.99,
                        "mean_b_match": 0.40,
                        "mean_idlekv": 0.70,
                        "mean_wrong_q_k": 0.44,
                        "mean_stale_q_k": 0.50,
                        "mean_contrastive_q_k": 0.80,
                        "mean_oracle_k": 0.95,
                    }
                }
            },
            "rows": [
                {
                    "task": "mq_niah_6q_split_456_to_123",
                    "k": 48,
                    "q2_query_rows_s": 0.01,
                    "q2_evicted_scoring_s": 0.02,
                    "idlekv_selection_s": 0.003,
                    "idlekv_transfer_ms": 4.0,
                    "idlekv_inject_ms": 5.0,
                    "wrong_q_k_score": 0.44,
                    "wrong_q_query_rows_s": 0.011,
                    "wrong_q_evicted_scoring_s": 0.021,
                    "wrong_q_k_selection_s": 0.004,
                    "wrong_q_k_transfer_ms": 4.5,
                    "wrong_q_k_inject_ms": 5.5,
                }
            ],
        }
        rows = frontier_rows_for_artifact(artifact, artifact_path="artifact.json")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["artifact"], "artifact.json")
        self.assertEqual(rows[0]["task"], "mq_niah_6q_clean_suite")
        self.assertEqual(rows[0]["base_context_budget"], 18432)
        self.assertEqual(rows[0]["wrong_query_mode"], "donor_q2")
        self.assertEqual(rows[0]["wrong_query_donor_offset"], 100000)
        self.assertAlmostEqual(float(rows[0]["idlekv_lift"]), 0.30)
        self.assertAlmostEqual(float(rows[0]["wrong_q_lift"]), 0.04)
        self.assertAlmostEqual(float(rows[0]["stale_q_lift"]), 0.10)
        self.assertAlmostEqual(float(rows[0]["true_minus_wrong_q"]), 0.26)
        self.assertAlmostEqual(float(rows[0]["true_minus_stale_q"]), 0.20)
        self.assertAlmostEqual(float(rows[0]["contrastive_q_lift"]), 0.40)
        self.assertAlmostEqual(float(rows[0]["contrastive_minus_idlekv"]), 0.10)
        self.assertAlmostEqual(float(rows[0]["gold_headroom"]), 0.25)
        self.assertAlmostEqual(float(rows[0]["gold_normalized_recovery"]), (0.70 - 0.40) / (0.95 - 0.40))
        self.assertAlmostEqual(float(rows[0]["p50_total_ms"]), 42.0)
        self.assertAlmostEqual(float(rows[0]["p50_wrong_q_total_ms"]), 46.0)

    def test_future_query_gate_requires_true_query_separation_and_wrong_query_near_baseline(self) -> None:
        artifact = {
            "aggregate": {
                "overall": {
                    "k48": {"mean_b_match": 0.40, "mean_idlekv": 0.62, "mean_wrong_q_k": 0.50},
                    "k96": {"mean_b_match": 0.41, "mean_idlekv": 0.90, "mean_wrong_q_k": 0.47},
                }
            }
        }
        checks = future_query_gate(artifact)
        self.assertTrue(all(passed for _, passed, _ in checks))

    def test_future_query_gate_fails_if_wrong_query_also_repairs(self) -> None:
        artifact = {
            "aggregate": {
                "overall": {
                    "k48": {"mean_b_match": 0.40, "mean_idlekv": 0.62, "mean_wrong_q_k": 0.60},
                    "k96": {"mean_b_match": 0.41, "mean_idlekv": 0.90, "mean_wrong_q_k": 0.75},
                }
            }
        }
        checks = future_query_gate(artifact)
        self.assertFalse(all(passed for _, passed, _ in checks))

    def test_proxy_gate_uses_lower_lift_threshold_for_6q(self) -> None:
        artifact = {
            "config": {"task": "mq_niah_6q_clean_suite"},
            "aggregate": {"overall": {"k96": {"mean_b_match": 0.40, "mean_idlekv": 0.51}}},
        }
        checks = proxy_gate(artifact)
        self.assertTrue(checks[0][1])

    def test_contrastive_gate_requires_mid_budget_gain_without_high_budget_loss(self) -> None:
        artifact = {
            "aggregate": {
                "overall": {
                    "k48": {
                        "mean_idlekv": 0.62,
                        "mean_wrong_q_k": 0.50,
                        "mean_contrastive_q_k": 0.70,
                    },
                    "k96": {
                        "mean_idlekv": 0.90,
                        "mean_contrastive_q_k": 0.89,
                    },
                }
            }
        }
        checks = contrastive_gate(artifact)
        self.assertTrue(all(passed for _, passed, _ in checks))

    def test_contrastive_gate_fails_if_only_matching_default_idlekv(self) -> None:
        artifact = {
            "aggregate": {
                "overall": {
                    "k48": {
                        "mean_idlekv": 0.62,
                        "mean_wrong_q_k": 0.50,
                        "mean_contrastive_q_k": 0.62,
                    },
                    "k96": {
                        "mean_idlekv": 0.90,
                        "mean_contrastive_q_k": 0.90,
                    },
                }
            }
        }
        checks = contrastive_gate(artifact)
        self.assertFalse(all(passed for _, passed, _ in checks))

    def test_stale_query_gate_requires_future_query_to_beat_previous_query(self) -> None:
        artifact = {
            "aggregate": {
                "overall": {
                    "k48": {
                        "mean_idlekv": 0.70,
                        "mean_stale_q_k": 0.50,
                    },
                    "k96": {
                        "mean_b_match": 0.40,
                        "mean_idlekv": 0.90,
                        "mean_stale_q_k": 0.55,
                    },
                }
            }
        }
        checks = stale_query_gate(artifact)
        self.assertTrue(all(passed for _, passed, _ in checks))

    def test_write_csv_unions_fields_across_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "summary.csv"
            write_csv([{"k": 48, "idlekv": 0.5}, {"k": 96, "wrong_q_k": 0.3}], out_path)
            with out_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
        self.assertEqual(reader.fieldnames, ["k", "idlekv", "wrong_q_k"])
        self.assertEqual(rows[1]["wrong_q_k"], "0.3")


if __name__ == "__main__":
    unittest.main()
