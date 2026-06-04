#!/usr/bin/env python3
"""Render a compact Phase 9 quality-latency ladder from summary CSV rows."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LadderPoint:
    label: str
    latency_ms: float
    score: float
    color: str


def _load_numeric_rows(path: Path) -> list[dict[str, float]]:
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
            if numeric_row:
                rows.append(numeric_row)
    return rows


def _row_for_k(rows: list[dict[str, float]], k: int) -> dict[str, float]:
    for row in rows:
        if int(row.get("k", -1)) == int(k):
            return row
    raise ValueError(f"No row found for K={k}")


def build_points(*, exact_csv: Path, proxy_csv: Path, k: int) -> list[LadderPoint]:
    exact = _row_for_k(_load_numeric_rows(exact_csv), k)
    proxy = _row_for_k(_load_numeric_rows(proxy_csv), k)
    move_only_ms = exact.get("p50_transfer_ms", 0.0) + exact.get("p50_inject_ms", 0.0)
    return [
        LadderPoint("Matched no-repair", 0.0, exact["b_match"], "#6b7280"),
        LadderPoint("KV move only", move_only_ms, exact["b_match"], "#9ca3af"),
        LadderPoint("IdleKV proxy scorer", proxy["p50_total_ms"], proxy["idlekv"], "#16a34a"),
        LadderPoint("IdleKV exact scorer", exact["p50_total_ms"], exact["idlekv"], "#2563eb"),
    ]


def _scale(value: float, *, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:
    if abs(domain_max - domain_min) < 1e-12:
        return (range_min + range_max) / 2.0
    t = (value - domain_min) / (domain_max - domain_min)
    return range_min + t * (range_max - range_min)


def render_svg(
    *,
    points: list[LadderPoint],
    output_path: Path,
    title: str = "Quality-latency ladder",
) -> None:
    if not points:
        raise ValueError("Cannot render an empty ladder")

    width = 760
    height = 430
    margin_left = 82
    margin_right = 42
    margin_top = 72
    margin_bottom = 74
    plot_x0 = margin_left
    plot_x1 = width - margin_right
    plot_y0 = margin_top
    plot_y1 = height - margin_bottom

    max_latency = max(point.latency_ms for point in points)
    x_max = max(1.0, max_latency * 1.08)
    y_min = max(0.0, min(point.score for point in points) - 0.08)
    y_max = min(1.0, max(point.score for point in points) + 0.04)
    if y_max - y_min < 0.12:
        y_min = max(0.0, y_max - 0.12)

    def x_for(latency_ms: float) -> float:
        return _scale(latency_ms, domain_min=0.0, domain_max=x_max, range_min=plot_x0, range_max=plot_x1)

    def y_for(score: float) -> float:
        return _scale(score, domain_min=y_min, domain_max=y_max, range_min=plot_y1, range_max=plot_y0)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: "DejaVu Sans Mono", monospace; fill: #111827; }',
        '.title { font-size: 20px; font-weight: 700; }',
        '.axis { font-size: 13px; }',
        '.tick { font-size: 11px; fill: #4b5563; }',
        '.label { font-size: 12px; font-weight: 700; }',
        '.note { font-size: 12px; fill: #374151; }',
        '</style>',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text class="title" x="{plot_x0}" y="34">{title}</text>',
        f'<text class="note" x="{plot_x0}" y="56">Score at fixed restore budget K; up-left movement indicates faster repair at similar quality.</text>',
        f'<line x1="{plot_x0}" y1="{plot_y1}" x2="{plot_x1}" y2="{plot_y1}" stroke="#111827" stroke-width="1.5"/>',
        f'<line x1="{plot_x0}" y1="{plot_y0}" x2="{plot_x0}" y2="{plot_y1}" stroke="#111827" stroke-width="1.5"/>',
    ]

    for i in range(5):
        latency = x_max * i / 4.0
        x = x_for(latency)
        svg.append(f'<line x1="{x:.1f}" y1="{plot_y1}" x2="{x:.1f}" y2="{plot_y1 + 5}" stroke="#111827"/>')
        svg.append(f'<text class="tick" x="{x:.1f}" y="{plot_y1 + 21}" text-anchor="middle">{latency / 1000.0:.1f}s</text>')

    for i in range(5):
        score = y_min + (y_max - y_min) * i / 4.0
        y = y_for(score)
        svg.append(f'<line x1="{plot_x0 - 5}" y1="{y:.1f}" x2="{plot_x0}" y2="{y:.1f}" stroke="#111827"/>')
        svg.append(f'<text class="tick" x="{plot_x0 - 10}" y="{y + 4:.1f}" text-anchor="end">{score:.2f}</text>')
        if i not in (0, 4):
            svg.append(f'<line x1="{plot_x0}" y1="{y:.1f}" x2="{plot_x1}" y2="{y:.1f}" stroke="#e5e7eb"/>')

    ordered_points = sorted(points, key=lambda point: point.latency_ms)
    line_coords = " ".join(f'{x_for(point.latency_ms):.1f},{y_for(point.score):.1f}' for point in ordered_points)
    svg.append(f'<polyline points="{line_coords}" fill="none" stroke="#d1d5db" stroke-width="2"/>')

    for point in ordered_points:
        x = x_for(point.latency_ms)
        y = y_for(point.score)
        label_anchor = "start" if x < plot_x1 - 130 else "end"
        label_dx = 10 if label_anchor == "start" else -10
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{point.color}" stroke="white" stroke-width="2"/>')
        svg.append(
            f'<text class="label" x="{x + label_dx:.1f}" y="{y - 10:.1f}" text-anchor="{label_anchor}">'
            f'{point.label} ({point.score:.2f})</text>'
        )

    svg.append(f'<text class="axis" x="{(plot_x0 + plot_x1) / 2:.1f}" y="{height - 28}" text-anchor="middle">p50 repair latency</text>')
    svg.append(f'<text class="axis" x="24" y="{(plot_y0 + plot_y1) / 2:.1f}" transform="rotate(-90 24 {(plot_y0 + plot_y1) / 2:.1f})" text-anchor="middle">Mean Q2 score</text>')
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-csv", type=Path, required=True)
    parser.add_argument("--proxy-csv", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Quality-latency ladder")
    args = parser.parse_args()
    render_svg(
        points=build_points(exact_csv=args.exact_csv, proxy_csv=args.proxy_csv, k=args.k),
        output_path=args.out,
        title=args.title,
    )


if __name__ == "__main__":
    main()
