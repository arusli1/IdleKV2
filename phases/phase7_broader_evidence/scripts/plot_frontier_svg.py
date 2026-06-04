#!/usr/bin/env python3
"""Render a compact score-vs-budget SVG from an exported overall CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_SERIES = (
    ("condition_b", "#6b7280", "Base compressed"),
    ("b_match", "#374151", "Matched no repair"),
    ("idlekv", "#2563eb", "IdleKV"),
    ("stale_q_k", "#7c3aed", "StaleQ-K"),
    ("wrong_q_k", "#db2777", "WrongQ-K"),
    ("contrastive_q_k", "#059669", "ContrastiveQ-K"),
    ("random_k", "#0f766e", "Random-K"),
    ("oldest_k", "#b45309", "Oldest-K"),
    ("oracle_k", "#dc2626", "Gold-K"),
)
SERIES_BY_KEY = {key: (key, color, label) for key, color, label in DEFAULT_SERIES}


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
            rows.append(numeric_row)
    return rows


def _scale(value: float, *, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if abs(src_max - src_min) < 1e-9:
        return (dst_min + dst_max) * 0.5
    alpha = (value - src_min) / (src_max - src_min)
    return dst_min + alpha * (dst_max - dst_min)


def render_svg(
    *,
    rows: list[dict[str, float]],
    output_path: Path,
    title: str,
    suffix: str,
    series: tuple[tuple[str, str, str], ...],
) -> None:
    width = 900
    height = 640
    margin_left = 84
    margin_right = 28
    margin_top = 56
    margin_bottom = 162
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom

    k_values = [row["k"] for row in rows]
    k_min = min(k_values)
    k_max = max(k_values)
    y_min = 0.0
    y_max = 1.0

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

    # Grid + ticks
    y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    for tick in y_ticks:
        y = y_pos(tick)
        svg.append(f'<line class="grid" x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}"/>')
        svg.append(f'<text class="tick" x="{plot_left - 12}" y="{y + 5:.2f}" text-anchor="end">{tick:.2f}</text>')

    for k in k_values:
        x = x_pos(k)
        svg.append(f'<line class="grid" x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_bottom}"/>')
        svg.append(f'<text class="tick" x="{x:.2f}" y="{plot_bottom + 22}" text-anchor="middle">{int(k)}</text>')

    svg.append(f'<line class="axis" x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}"/>')
    svg.append(f'<line class="axis" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}"/>')
    svg.append(f'<text class="label" x="{(plot_left + plot_right) / 2:.2f}" y="{plot_bottom + 52}" text-anchor="middle">Restore budget K</text>')
    svg.append(
        f'<text class="label" x="22" y="{(plot_top + plot_bottom) / 2:.2f}" transform="rotate(-90 22 {(plot_top + plot_bottom) / 2:.2f})" text-anchor="middle">Mean turn-2 score</text>'
    )

    # Series
    legend_x = plot_left
    legend_y = plot_bottom + 86
    legend_col_step = 198
    legend_row_step = 22
    legend_cols = 4
    for idx, (column, color, label) in enumerate(series):
        series_key = f"{column}{suffix}" if suffix else column
        if series_key not in rows[0]:
            continue
        lo_key = f"{series_key}_lo"
        hi_key = f"{series_key}_hi"
        if lo_key in rows[0] and hi_key in rows[0]:
            upper = " ".join(f"{x_pos(row['k']):.2f},{y_pos(row[hi_key]):.2f}" for row in rows)
            lower = " ".join(f"{x_pos(row['k']):.2f},{y_pos(row[lo_key]):.2f}" for row in reversed(rows))
            svg.append(
                f'<polygon fill="{color}" fill-opacity="0.14" stroke="none" points="{upper} {lower}"/>'
            )
        points = " ".join(f"{x_pos(row['k']):.2f},{y_pos(row[series_key]):.2f}" for row in rows)
        svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>')
        for row in rows:
            x = x_pos(row["k"])
            y = y_pos(row[series_key])
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}" stroke="white" stroke-width="1"/>')
        legend_index = len([item for item in svg if item.startswith('<text class="legend"')])
        lx = legend_x + (legend_index % legend_cols) * legend_col_step
        ly = legend_y + (legend_index // legend_cols) * legend_row_step
        svg.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 22}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<circle cx="{lx + 11}" cy="{ly}" r="4.5" fill="{color}" stroke="white" stroke-width="1"/>')
        svg.append(f'<text class="legend" x="{lx + 30}" y="{ly + 4}">{label}</text>')

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument("--suffix", type=str, default="", help="Optional column suffix such as '_overlap'")
    parser.add_argument(
        "--series",
        type=str,
        default=",".join(key for key, _, _ in DEFAULT_SERIES),
        help="Comma-separated series keys to render, chosen from the default set",
    )
    args = parser.parse_args()

    rows = _load_rows(args.csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.csv}")
    series_keys = [item.strip() for item in args.series.split(",") if item.strip()]
    series: list[tuple[str, str, str]] = []
    for key in series_keys:
        if key not in SERIES_BY_KEY:
            raise SystemExit(f"Unknown series key '{key}'. Choose from: {', '.join(SERIES_BY_KEY)}")
        series.append(SERIES_BY_KEY[key])
    render_svg(rows=rows, output_path=args.out, title=args.title, suffix=args.suffix, series=tuple(series))


if __name__ == "__main__":
    main()
