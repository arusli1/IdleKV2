"""Tests for Phase 9-compatible plotting helpers."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from phases.phase7_broader_evidence.scripts.plot_frontier_svg import _load_rows, render_svg


class Phase9PlottingTests(unittest.TestCase):
    def test_frontier_plot_loader_ignores_metadata_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "phase9.csv"
            csv_path.write_text(
                "artifact,task,k,b_match,idlekv,wrong_q_k,contrastive_q_k\n"
                "artifact.json,mq_niah_6q_clean_suite,48,0.4,0.7,0.5,0.8\n",
                encoding="utf-8",
            )

            rows = _load_rows(csv_path)

        self.assertEqual(rows, [{"k": 48.0, "b_match": 0.4, "idlekv": 0.7, "wrong_q_k": 0.5, "contrastive_q_k": 0.8}])

    def test_render_svg_accepts_phase9_series_without_legend_overlap_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "plot.svg"
            render_svg(
                rows=[
                    {"k": 48.0, "b_match": 0.4, "idlekv": 0.7, "wrong_q_k": 0.5, "contrastive_q_k": 0.8, "oracle_k": 1.0},
                    {"k": 96.0, "b_match": 0.4, "idlekv": 1.0, "wrong_q_k": 0.8, "contrastive_q_k": 0.95, "oracle_k": 1.0},
                ],
                output_path=out_path,
                title="Phase 9 smoke",
                suffix="",
                series=(
                    ("b_match", "#374151", "Matched no repair"),
                    ("idlekv", "#2563eb", "IdleKV"),
                    ("wrong_q_k", "#db2777", "WrongQ-K"),
                    ("contrastive_q_k", "#059669", "ContrastiveQ-K"),
                    ("oracle_k", "#dc2626", "Gold-K"),
                ),
            )

            svg = out_path.read_text(encoding="utf-8")

        self.assertIn("ContrastiveQ-K", svg)
        self.assertIn('height="640"', svg)


if __name__ == "__main__":
    unittest.main()
