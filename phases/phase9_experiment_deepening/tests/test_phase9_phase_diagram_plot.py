"""Tests for Phase 9 operating-regime heatmap plotting."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from phases.phase9_experiment_deepening.scripts.plot_phase_diagram_svg import _load_rows, render_svg
from phases.phase9_experiment_deepening.scripts.export_phase_diagram_tikz import load_cells, render_tikz
from phases.phase9_experiment_deepening.scripts.plot_phase_diagram_png import (
    load_cells as load_png_cells,
    render_png,
)


class Phase9PhaseDiagramPlotTests(unittest.TestCase):
    def test_load_rows_keeps_numeric_phase9_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "phase9.csv"
            csv_path.write_text(
                "artifact,base_context_budget,k,idlekv_lift,gold_headroom\n"
                "a.json,12288,48,0.17,0.79\n",
                encoding="utf-8",
            )

            rows = _load_rows(csv_path)

        self.assertEqual(rows, [{"base_context_budget": 12288.0, "k": 48.0, "idlekv_lift": 0.17, "gold_headroom": 0.79}])

    def test_render_svg_marks_gold_headroom_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "heatmap.svg"
            render_svg(
                rows=[
                    {"base_context_budget": 12288.0, "k": 48.0, "idlekv_lift": 0.17, "gold_headroom": 0.79},
                    {"base_context_budget": 18432.0, "k": 48.0, "idlekv_lift": 0.25, "gold_headroom": 0.33},
                ],
                output_path=out_path,
                metric="idlekv_lift",
                title="Smoke heatmap",
            )

            svg = out_path.read_text(encoding="utf-8")

        self.assertIn("Smoke heatmap", svg)
        self.assertIn("B=12288", svg)
        self.assertIn("<circle", svg)

    def test_export_tikz_uses_budget_labels_and_no_embedded_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "phase9.csv"
            csv_path.write_text(
                "base_context_budget,k,idlekv_lift,gold_headroom\n"
                "12288,48,0.17,0.79\n"
                "18432,96,0.25,0.01\n",
                encoding="utf-8",
            )

            cells = load_cells(csv_path, panel="6Q")
            tikz = render_tikz({"6Q": cells})

        self.assertIn("12K", tikz)
        self.assertIn("18K", tikz)
        self.assertIn("score gain", tikz)
        self.assertIn("\\draw[black", tikz)
        self.assertNotIn("Repair operating regime", tikz)

    def test_render_png_writes_single_column_heatmap_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "phase9.csv"
            out_path = Path(tmpdir) / "heatmap.png"
            csv_path.write_text(
                "base_context_budget,k,idlekv_lift,gold_headroom\n"
                "12288,16,0.01,0.00\n"
                "12288,48,0.17,0.79\n"
                "18432,16,0.02,0.00\n"
                "18432,48,0.25,0.33\n",
                encoding="utf-8",
            )

            cells = load_png_cells(csv_path)
            render_png(cells_4q=cells, cells_6q=cells, output_path=out_path)

            header = out_path.read_bytes()[:8]

        self.assertEqual(header, b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
