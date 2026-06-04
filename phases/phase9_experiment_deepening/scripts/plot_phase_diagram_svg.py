#!/usr/bin/env python3
"""Render a compact operating-regime heatmap from summary CSV rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _load_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            numeric_row: dict[str, float] = {}
            for key, value in row.items():
                if not value:
                    continue
                try:
                    numeric_row[key] = float(value)
                except ValueError:
                    continue
            if "base_context_budget" in numeric_row and "k" in numeric_row:
                rows.append(numeric_row)
    return rows


def _lerp(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * max(0.0, min(1.0, t))))


def _color(value: float, *, vmin: float, vmax: float) -> str:
    if abs(vmax - vmin) < 1e-12:
        t = 0.0
    else:
        t = (float(value) - vmin) / (vmax - vmin)
    stops = (
        (248, 250, 252),
        (191, 219, 254),
        (37, 99, 235),
    )
    if t <= 0.5:
        local = t / 0.5
        start, end = stops[0], stops[1]
    else:
        local = (t - 0.5) / 0.5
        start, end = stops[1], stops[2]
    r = _lerp(start[0], end[0], local)
    g = _lerp(start[1], end[1], local)
    b = _lerp(start[2], end[2], local)
    return f"#{r:02x}{g:02x}{b:02x}"


def render_svg(
    *,
    rows: list[dict[str, float]],
    output_path: Path,
    metric: str = "idlekv_lift",
    title: str = "Repair operating regime",
    headroom_threshold: float = 0.05,
) -> None:
    if not rows:
        raise ValueError("Cannot render an empty heatmap")
    budgets = sorted({int(row["base_context_budget"]) for row in rows})
    k_values = sorted({int(row["k"]) for row in rows})
    by_cell = {
        (int(row["base_context_budget"]), int(row["k"])): row
        for row in rows
    }
    values = [float(row.get(metric, 0.0)) for row in rows]
    vmin = min(0.0, min(values))
    vmax = max(1e-9, max(values))

    cell_w = 116
    cell_h = 62
    margin_left = 112
    margin_right = 28
    margin_top = 72
    margin_bottom = 82
    width = margin_left + margin_right + cell_w * len(k_values)
    height = margin_top + margin_bottom + cell_h * len(budgets)
    plot_top = margin_top

    metric_label = "IdleKV gain over matched no-repair" if metric == "idlekv_lift" else metric
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: "DejaVu Sans Mono", monospace; fill: #111827; }',
        '.title { font-size: 20px; font-weight: 700; }',
        '.axis { font-size: 13px; }',
        '.cell { font-size: 14px; font-weight: 700; }',
        '.note { font-size: 12px; fill: #374151; }',
        '</style>',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text class="title" x="{margin_left}" y="34">{title}</text>',
        f'<text class="note" x="{margin_left}" y="56">Cell value: {metric_label}; dot marks Gold-K headroom > {headroom_threshold:.2f}</text>',
    ]

    for col, k in enumerate(k_values):
        x = margin_left + col * cell_w + cell_w / 2
        svg.append(f'<text class="axis" x="{x:.1f}" y="{plot_top - 14}" text-anchor="middle">K={k}</text>')
    for row_index, budget in enumerate(budgets):
        y = plot_top + row_index * cell_h + cell_h / 2
        svg.append(f'<text class="axis" x="{margin_left - 14}" y="{y + 5:.1f}" text-anchor="end">B={budget}</text>')
        for col, k in enumerate(k_values):
            x0 = margin_left + col * cell_w
            y0 = plot_top + row_index * cell_h
            row = by_cell.get((budget, k), {})
            value = float(row.get(metric, 0.0))
            fill = _color(value, vmin=vmin, vmax=vmax)
            svg.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')
            svg.append(f'<text class="cell" x="{x0 + cell_w / 2:.1f}" y="{y0 + cell_h / 2 + 5:.1f}" text-anchor="middle">{value:.2f}</text>')
            if float(row.get("gold_headroom", 0.0)) > float(headroom_threshold):
                svg.append(f'<circle cx="{x0 + cell_w - 16:.1f}" cy="{y0 + 16:.1f}" r="5" fill="#111827"/>')

    svg.append(f'<text class="axis" x="{margin_left + cell_w * len(k_values) / 2:.1f}" y="{height - 28}" text-anchor="middle">Restore budget K</text>')
    svg.append(
        f'<text class="axis" x="26" y="{plot_top + cell_h * len(budgets) / 2:.1f}" transform="rotate(-90 26 {plot_top + cell_h * len(budgets) / 2:.1f})" text-anchor="middle">Base compressed budget</text>'
    )
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metric", type=str, default="idlekv_lift")
    parser.add_argument("--title", type=str, default="Repair operating regime")
    parser.add_argument("--headroom-threshold", type=float, default=0.05)
    args = parser.parse_args()
    render_svg(
        rows=_load_rows(args.csv),
        output_path=args.out,
        metric=args.metric,
        title=args.title,
        headroom_threshold=args.headroom_threshold,
    )


if __name__ == "__main__":
    main()
