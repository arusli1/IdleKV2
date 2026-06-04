"""Tests for Phase 9 quality-latency ladder plotting."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from phases.phase9_experiment_deepening.scripts.plot_quality_latency_svg import LadderPoint, build_points, render_svg


class Phase9QualityLatencyPlotTests(unittest.TestCase):
    def test_build_points_uses_fixed_k_exact_and_proxy_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            exact_csv = Path(tmpdir) / "exact.csv"
            proxy_csv = Path(tmpdir) / "proxy.csv"
            exact_csv.write_text(
                "k,b_match,idlekv,p50_total_ms,p50_transfer_ms,p50_inject_ms\n"
                "48,0.40,0.70,6000,1,4\n"
                "96,0.42,1.00,6750,0.9,4.1\n",
                encoding="utf-8",
            )
            proxy_csv.write_text(
                "k,b_match,idlekv,p50_total_ms\n"
                "96,0.42,0.96,784\n",
                encoding="utf-8",
            )

            points = build_points(exact_csv=exact_csv, proxy_csv=proxy_csv, k=96)

        self.assertEqual(
            [point.label for point in points],
            ["Matched no-repair", "KV move only", "IdleKV proxy scorer", "IdleKV exact scorer"],
        )
        self.assertAlmostEqual(points[1].latency_ms, 5.0)
        self.assertAlmostEqual(points[2].score, 0.96)
        self.assertAlmostEqual(points[3].latency_ms, 6750.0)

    def test_render_svg_labels_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "ladder.svg"
            render_svg(
                points=[
                    LadderPoint("Matched no-repair", 0.0, 0.42, "#6b7280"),
                    LadderPoint("IdleKV proxy scorer", 784.0, 0.96, "#16a34a"),
                ],
                output_path=out_path,
                title="Latency smoke",
            )

            svg = out_path.read_text(encoding="utf-8")

        self.assertIn("Latency smoke", svg)
        self.assertIn("IdleKV proxy scorer", svg)
        self.assertIn("Mean Q2 score", svg)


if __name__ == "__main__":
    unittest.main()
