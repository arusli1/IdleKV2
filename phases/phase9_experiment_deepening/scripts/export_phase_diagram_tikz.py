#!/usr/bin/env python3
"""Export operating-regime CSVs as a paper-ready TikZ heatmap."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HeatmapCell:
    panel: str
    budget: int
    k: int
    value: float
    gold_headroom: float


def _budget_label(budget: int) -> str:
    return f"{round(budget / 1024):d}K"


def load_cells(path: Path, *, panel: str, metric: str = "idlekv_lift") -> list[HeatmapCell]:
    cells: list[HeatmapCell] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row.get("base_context_budget") or not row.get("k"):
                continue
            cells.append(
                HeatmapCell(
                    panel=panel,
                    budget=int(float(row["base_context_budget"])),
                    k=int(float(row["k"])),
                    value=float(row.get(metric) or 0.0),
                    gold_headroom=float(row.get("gold_headroom") or 0.0),
                )
            )
    return cells


def _cell_color_percent(value: float, *, max_value: float) -> int:
    if max_value <= 0:
        return 0
    return round(max(0.0, min(value, max_value)) / max_value * 82)


def render_tikz(
    cells_by_panel: dict[str, list[HeatmapCell]],
    *,
    max_value: float = 0.8,
    headroom_threshold: float = 0.05,
) -> str:
    if not cells_by_panel:
        raise ValueError("No heatmap cells supplied")

    lines: list[str] = [
        "% Auto-generated operating-regime heatmap.",
        "\\begin{tikzpicture}[x=0.76cm,y=0.46cm,font=\\scriptsize]",
        "  \\def\\cellw{0.92}",
        "  \\def\\cellh{0.70}",
    ]
    panel_offsets = {"4Q": 0.0, "6Q": 4.55}
    all_k = sorted({cell.k for cells in cells_by_panel.values() for cell in cells})
    if not all_k:
        raise ValueError("No restore-budget values found")

    for panel in ("4Q", "6Q"):
        cells = cells_by_panel.get(panel, [])
        if not cells:
            continue
        budgets = sorted({cell.budget for cell in cells})
        by_cell = {(cell.budget, cell.k): cell for cell in cells}
        x_offset = panel_offsets.get(panel, 0.0)
        lines.append(f"  % {panel} panel")
        lines.append(
            f"  \\node[font=\\bfseries] at "
            f"({x_offset + 1.84:.2f},3.48) {{MQ-NIAH-{panel}}};"
        )
        lines.append(f"  \\node at ({x_offset + 1.84:.2f},3.14) {{$K$}};")
        for col_index, k in enumerate(all_k):
            lines.append(
                f"  \\node at "
                f"({x_offset + col_index + 0.46:.2f},2.84) {{{k}}};"
            )
        lines.append(
            f"  \\node[rotate=90] at "
            f"({x_offset - 0.76:.2f},1.40) {{$B_{{\\mathrm{{base}}}}$}};"
        )
        for row_index, budget in enumerate(budgets):
            y = 2.12 - row_index * 0.82
            lines.append(
                f"  \\node[anchor=east] at "
                f"({x_offset - 0.15:.2f},{y + 0.35:.2f}) {{{_budget_label(budget)}}};"
            )
            for col_index, k in enumerate(all_k):
                cell = by_cell.get((budget, k))
                value = cell.value if cell is not None else 0.0
                pct = _cell_color_percent(value, max_value=max_value)
                x = x_offset + col_index
                lines.append(
                    f"  \\fill[blue!{pct}!white,draw=white,line width=0.35pt] "
                    f"({x:.2f},{y:.2f}) rectangle ++(\\cellw,\\cellh);"
                )
                text_color = "white" if pct >= 62 else "black"
                lines.append(
                    f"  \\node[text={text_color}] at "
                    f"({x + 0.46:.2f},{y + 0.35:.2f}) {{+{value:.2f}}};"
                )
                if cell is not None and cell.gold_headroom > headroom_threshold:
                    lines.append(
                        f"  \\draw[black,line width=0.35pt] "
                        f"({x + 0.46:.2f},{y + 0.35:.2f}) circle[radius=0.20];"
                    )
    lines.extend(
        [
            "  \\node[anchor=west] at (0.00,-0.65) {score gain $\\Delta_{\\mathrm{repair}}$:};",
            "  \\foreach \\xx/\\pp in {1.36/0,1.80/20,2.24/41,2.68/62,3.12/82}{",
            "    \\fill[blue!\\pp!white,draw=white,line width=0.35pt] (\\xx,-0.77) rectangle ++(0.38,0.24);",
            "  }",
            "  \\node[anchor=west] at (3.70,-0.65) {0};",
            "  \\node[anchor=west] at (4.12,-0.65) {0.8};",
            "  \\draw[black,line width=0.35pt] (5.08,-0.65) circle[radius=0.16];",
            f"  \\node[anchor=west] at (5.34,-0.65) {{$\\goldk$ headroom $>{headroom_threshold:.2f}$}};",
            "\\end{tikzpicture}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv4q", type=Path, required=True)
    parser.add_argument("--csv6q", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metric", type=str, default="idlekv_lift")
    parser.add_argument("--max-value", type=float, default=0.8)
    parser.add_argument("--headroom-threshold", type=float, default=0.05)
    args = parser.parse_args()

    tikz = render_tikz(
        {
            "4Q": load_cells(args.csv4q, panel="4Q", metric=args.metric),
            "6Q": load_cells(args.csv6q, panel="6Q", metric=args.metric),
        },
        max_value=args.max_value,
        headroom_threshold=args.headroom_threshold,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tikz, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
