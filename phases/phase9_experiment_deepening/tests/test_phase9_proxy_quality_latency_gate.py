"""Tests for the Phase 9 proxy quality-latency gate."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from phases.phase9_experiment_deepening.scripts.check_proxy_quality_latency import evaluate_proxy_quality_latency


class Phase9ProxyQualityLatencyGateTests(unittest.TestCase):
    def test_gate_checks_quality_retention_and_speedups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            exact_csv = Path(tmpdir) / "exact.csv"
            proxy_csv = Path(tmpdir) / "proxy.csv"
            exact_csv.write_text(
                "k,b_match,idlekv,p50_total_ms,p50_score_ms\n"
                "48,0.40,0.70,6000,5900\n"
                "96,0.40,1.00,6000,5900\n",
                encoding="utf-8",
            )
            proxy_csv.write_text(
                "k,b_match,idlekv,p50_total_ms,p50_score_ms\n"
                "48,0.40,0.60,900,700\n"
                "96,0.40,0.92,900,700\n",
                encoding="utf-8",
            )

            checks = evaluate_proxy_quality_latency(exact_csv=exact_csv, proxy_csv=proxy_csv)

        self.assertTrue(all(check.passed for check in checks))
        self.assertIn("retention=0.867", [check.detail for check in checks][1])
        self.assertIn("speedup=6.67x", [check.detail for check in checks][3])

    def test_gate_fails_hidden_guardrail_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            exact_csv = Path(tmpdir) / "exact.csv"
            proxy_csv = Path(tmpdir) / "proxy.csv"
            exact_csv.write_text(
                "k,b_match,idlekv,p50_total_ms,p50_score_ms\n"
                "48,0.40,0.70,6000,5900\n"
                "96,0.40,1.00,6000,5900\n",
                encoding="utf-8",
            )
            proxy_csv.write_text(
                "k,b_match,idlekv,p50_total_ms,p50_score_ms\n"
                "48,0.40,0.45,900,700\n"
                "96,0.40,0.92,900,700\n",
                encoding="utf-8",
            )

            checks = evaluate_proxy_quality_latency(exact_csv=exact_csv, proxy_csv=proxy_csv)

        self.assertFalse(all(check.passed for check in checks))
        self.assertFalse(checks[-1].passed)


if __name__ == "__main__":
    unittest.main()
