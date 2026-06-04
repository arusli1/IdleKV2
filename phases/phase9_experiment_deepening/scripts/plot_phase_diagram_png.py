#!/usr/bin/env python3
"""Render a paper-style operating-regime heatmap image."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from paper.scripts.render_paper_figures import GAIN_CMAP, PALETTE, configure_matplotlib


@dataclass(frozen=True)
class Cell:
    budget: int
    k: int
    value: float
    gold_headroom: float


def load_cells(path: Path, *, metric: str = "idlekv_lift") -> list[Cell]:
    cells: list[Cell] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("base_context_budget") or not row.get("k"):
                continue
            cells.append(
                Cell(
                    budget=int(float(row["base_context_budget"])),
                    k=int(float(row["k"])),
                    value=float(row.get(metric) or 0.0),
                    gold_headroom=float(row.get("gold_headroom") or 0.0),
                )
            )
    return cells


def _pivot(cells: list[Cell], all_k: list[int]) -> tuple[list[int], np.ndarray, np.ndarray]:
    budgets = sorted({cell.budget for cell in cells})
    by_cell = {(cell.budget, cell.k): cell for cell in cells}
    values = np.array(
        [[by_cell.get((budget, k), Cell(budget, k, 0.0, 0.0)).value for k in all_k] for budget in budgets],
        dtype=float,
    )
    headroom = np.array(
        [[by_cell.get((budget, k), Cell(budget, k, 0.0, 0.0)).gold_headroom for k in all_k] for budget in budgets],
        dtype=float,
    )
    return budgets, values, headroom


def _budget_label(budget: int) -> str:
    return f"{budget / 1000:.1f}k"


def _draw_panel(
    ax: plt.Axes,
    *,
    label: str,
    cells: list[Cell],
    all_k: list[int],
    norm: Normalize,
    headroom_threshold: float,
) -> mpl.image.AxesImage:
    budgets, values, headroom = _pivot(cells, all_k)
    im = ax.imshow(np.clip(values, 0.0, None), cmap=GAIN_CMAP, norm=norm, aspect="auto")
    ax.set_xticks(range(len(all_k)), [str(k) for k in all_k])
    ax.set_yticks(range(len(budgets)), [_budget_label(budget) for budget in budgets])
    ax.set_ylabel("$B_{base}$", labelpad=1.0)
    ax.tick_params(length=0, pad=1.2)
    ax.set_xticks(np.arange(-0.5, len(all_k), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(budgets), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#666666")
    ax.text(
        -0.43,
        -0.72,
        label,
        transform=ax.transData,
        ha="left",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color=PALETTE["text"],
    )
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            rgba = GAIN_CMAP(norm(max(value, 0.0)))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "white" if luminance < 0.52 else PALETTE["text"]
            ax.text(col, row, f"{value:+.2f}", ha="center", va="center", fontsize=5.9, color=text_color)
            if headroom[row, col] > headroom_threshold:
                ax.scatter(
                    col + 0.32,
                    row - 0.32,
                    marker="*",
                    s=20,
                    color=PALETTE["gold"],
                    edgecolor="white",
                    linewidth=0.25,
                    zorder=4,
                )
    return im


def render_png(
    *,
    cells_4q: list[Cell],
    cells_6q: list[Cell],
    output_path: Path,
    max_value: float = 0.8,
    headroom_threshold: float = 0.20,
) -> None:
    if not cells_4q and not cells_6q:
        raise ValueError("Cannot render an empty heatmap")
    configure_matplotlib()
    all_k = sorted({cell.k for cell in [*cells_4q, *cells_6q]})
    max_observed = max([cell.value for cell in [*cells_4q, *cells_6q]], default=max_value)
    norm = Normalize(vmin=0.0, vmax=max(max_value, max_observed))

    n_panels = int(bool(cells_4q)) + int(bool(cells_6q))
    fig_height = 1.42 * n_panels + 0.45
    fig, axes_obj = plt.subplots(n_panels, 1, figsize=(3.35, fig_height), constrained_layout=False)
    axes = np.atleast_1d(axes_obj)
    fig.subplots_adjust(left=0.15, right=0.86, top=0.96, bottom=0.14, hspace=0.28)

    panel_idx = 0
    im = None
    if cells_4q:
        im = _draw_panel(
            axes[panel_idx],
            label="MQ-NIAH-4Q",
            cells=cells_4q,
            all_k=all_k,
            norm=norm,
            headroom_threshold=headroom_threshold,
        )
        panel_idx += 1
    if cells_6q:
        im = _draw_panel(
            axes[panel_idx],
            label="MQ-NIAH-6Q",
            cells=cells_6q,
            all_k=all_k,
            norm=norm,
            headroom_threshold=headroom_threshold,
        )

    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("restore budget $K$", labelpad=1.0)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.05, pad=0.035)
        cbar.set_label("score gain vs. matched", labelpad=2.0)
        cbar.outline.set_linewidth(0.45)
        cbar.ax.tick_params(width=0.45, length=2.0, pad=1.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.018, facecolor="white", dpi=360)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv4q", type=Path, required=True)
    parser.add_argument("--csv6q", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metric", type=str, default="idlekv_lift")
    parser.add_argument("--max-value", type=float, default=0.8)
    parser.add_argument("--headroom-threshold", type=float, default=0.20)
    args = parser.parse_args()
    render_png(
        cells_4q=load_cells(args.csv4q, metric=args.metric),
        cells_6q=load_cells(args.csv6q, metric=args.metric),
        output_path=args.out,
        max_value=args.max_value,
        headroom_threshold=args.headroom_threshold,
    )


if __name__ == "__main__":
    main()
