"""Tests for paired exact-vs-proxy bootstrap summaries."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from phases.phase9_experiment_deepening.scripts.proxy_paired_bootstrap import paired_rows


def _artifact(rows: list[dict[str, object]]) -> str:
    return json.dumps({"rows": rows})


class Phase9ProxyPairedBootstrapTests(unittest.TestCase):
    def test_paired_rows_match_on_task_index_and_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            exact_path = Path(tmpdir) / "exact.json"
            proxy_path = Path(tmpdir) / "proxy.json"
            exact_path.write_text(
                _artifact(
                    [
                        {"task": "a", "index": 1, "k": 96, "b_match_score": 0.0, "idlekv_score": 1.0},
                        {"task": "a", "index": 2, "k": 96, "b_match_score": 0.5, "idlekv_score": 1.0},
                        {"task": "a", "index": 1, "k": 48, "b_match_score": 0.0, "idlekv_score": 0.5},
                    ]
                ),
                encoding="utf-8",
            )
            proxy_path.write_text(
                _artifact(
                    [
                        {"task": "a", "index": 1, "k": 96, "b_match_score": 0.0, "idlekv_score": 1.0},
                        {"task": "a", "index": 2, "k": 96, "b_match_score": 0.5, "idlekv_score": 0.5},
                        {"task": "a", "index": 1, "k": 48, "b_match_score": 0.0, "idlekv_score": 0.5},
                    ]
                ),
                encoding="utf-8",
            )

            rows = paired_rows(
                exact_artifact=exact_path,
                proxy_artifact=proxy_path,
                k_values=[96],
                num_bootstrap=10,
                seed=7,
            )

        self.assertEqual(rows[0]["n_pairs"], 2)
        self.assertAlmostEqual(float(rows[0]["exact_lift"]), 0.75)
        self.assertAlmostEqual(float(rows[0]["proxy_lift"]), 0.5)
        self.assertAlmostEqual(float(rows[0]["retained_lift"]), 2.0 / 3.0)
        self.assertAlmostEqual(float(rows[0]["proxy_minus_exact"]), -0.25)

    def test_missing_pair_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            exact_path = Path(tmpdir) / "exact.json"
            proxy_path = Path(tmpdir) / "proxy.json"
            exact_path.write_text(_artifact([{"task": "a", "index": 1, "k": 96, "b_match_score": 0.0, "idlekv_score": 1.0}]), encoding="utf-8")
            proxy_path.write_text(_artifact([{"task": "a", "index": 2, "k": 96, "b_match_score": 0.0, "idlekv_score": 1.0}]), encoding="utf-8")

            with self.assertRaises(ValueError):
                paired_rows(exact_artifact=exact_path, proxy_artifact=proxy_path, k_values=[96])


if __name__ == "__main__":
    unittest.main()
