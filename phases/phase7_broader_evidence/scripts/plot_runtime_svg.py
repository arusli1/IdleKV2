#!/usr/bin/env python3
"""Render a small runtime SVG from an exported Phase 6/7 runtime CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_SERIES = (
    ("p50_total_ms", "#111827", "total p50"),
    ("p50_score_ms", "#dc2626", "score p50"),
    ("p50_query_ms", "#2563eb", "query p50"),
    ("p50_select_ms", "#0f766e", "select p50"),
    ("p50_transfer_ms", "#9333ea", "transfer p50"),
    ("p50_inject_ms", "#b45309", "inject p50"),
)


def _load_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: float(value) for key, value in row.items() if value})
    return rows


def _scale(value: float, *, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if abs(src_max - src_min) < 1e-9:
        return (dst_min + dst_max) * 0.5
    alpha = (value - src_min) / (src_max - src_min)
    return dst_min + alpha * (dst_max - dst_min)


def render_svg(*, rows: list[dict[str, float]], output_path: Path, title: str) -> None:
    width = 920
    height = 560
    margin_left = 90
    margin_right = 28
    margin_top = 56
    margin_bottom = 76
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom

    k_values = [row["k"] for row in rows]
    k_min = min(k_values)
    k_max = max(k_values)
    y_values = [row[column] for row in rows for column, _, _ in DEFAULT_SERIES if column in row]
    y_min = 0.0
    y_max = max(y_values) * 1.05

    def x_pos(k: float) -> float:
        return _scale(k, src_min=k_min, src_max=k_max, dst_min=plot_left, dst_max=plot_right)

    def y_pos(v: float) -> float:
        return _scale(v, src_min=y_min, src_max=y_max, dst_min=plot_bottom, dst_max=plot_top)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: "DejaVu Sans Mono", monospace; fill: #111827; }',
        '.axis { stroke: #111827; stroke-width: 1.4; }',
        '.grid { stroke: #e5e7eb; stroke-width: 1; }',
        '.tick { font-size: 13px; }',
        '.title { font-size: 20px; font-weight: 700; }',
        '.label { font-size: 14px; }',
        '.legend { font-size: 13px; }',
        '</style>',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text class="title" x="{plot_left}" y="32">{title}</text>',
    ]

    y_ticks = [0.0, y_max * 0.25, y_max * 0.5, y_max * 0.75, y_max]
    for tick in y_ticks:
        y = y_pos(tick)
        svg.append(f'<line class="grid" x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}"/>')
        svg.append(f'<text class="tick" x="{plot_left - 12}" y="{y + 5:.2f}" text-anchor="end">{tick:.0f}</text>')

    for k in k_values:
        x = x_pos(k)
        svg.append(f'<line class="grid" x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_bottom}"/>')
        svg.append(f'<text class="tick" x="{x:.2f}" y="{plot_bottom + 22}" text-anchor="middle">{int(k)}</text>')

    svg.append(f'<line class="axis" x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}"/>')
    svg.append(f'<line class="axis" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}"/>')
    svg.append(f'<text class="label" x="{(plot_left + plot_right) / 2:.2f}" y="{height - 24}" text-anchor="middle">Restore budget K</text>')
    svg.append(
        f'<text class="label" x="24" y="{(plot_top + plot_bottom) / 2:.2f}" transform="rotate(-90 24 {(plot_top + plot_bottom) / 2:.2f})" text-anchor="middle">Latency (ms)</text>'
    )

    legend_x = plot_left
    legend_y = height - 18
    legend_step = 130
    for idx, (column, color, label) in enumerate(DEFAULT_SERIES):
        if column not in rows[0]:
            continue
        points = " ".join(f"{x_pos(row['k']):.2f},{y_pos(row[column]):.2f}" for row in rows)
        svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>')
        for row in rows:
            x = x_pos(row["k"])
            y = y_pos(row[column])
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.0" fill="{color}" stroke="white" stroke-width="1"/>')
        lx = legend_x + idx * legend_step
        svg.append(f'<line x1="{lx}" y1="{legend_y}" x2="{lx + 22}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<circle cx="{lx + 11}" cy="{legend_y}" r="4.0" fill="{color}" stroke="white" stroke-width="1"/>')
        svg.append(f'<text class="legend" x="{lx + 30}" y="{legend_y + 4}">{label}</text>')

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", type=str, required=True)
    args = parser.parse_args()

    rows = _load_rows(args.csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.csv}")
    render_svg(rows=rows, output_path=args.out, title=args.title)


if __name__ == "__main__":
    main()
