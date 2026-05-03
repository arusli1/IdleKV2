#!/usr/bin/env python3
"""Render publication-quality paper figure assets.

The figures are generated as vector PDFs for LaTeX and as high-DPI PNGs
for quick visual inspection. The style intentionally matches ICML-like
two-column papers: Times-compatible serif text, thin axes, restrained
grids, colorblind-safe categorical colors, and no large in-plot titles.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PAPER_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PAPER_DIR.parent
FIGURE_DIR = PAPER_DIR / "figures"
PHASE9_DIR = REPO_DIR / "phases" / "phase9_experiment_deepening" / "results"
PHASE10_DIR = REPO_DIR / "phases" / "phase10_expansion" / "results"
PHASE11_DIR = REPO_DIR / "phases" / "phase11_main_robustness" / "results"
PHASE12_DIR = REPO_DIR / "phases" / "phase12_policy_breadth" / "results"
PHASE13_DIR = REPO_DIR / "phases" / "phase13_iteration_framework" / "results"
PHASE4_RUNTIME_DIR = REPO_DIR / "phases" / "phase4_eviction_buffer" / "results" / "runtime_capacity"
PHASE8_STREAMING_DIR = (
    REPO_DIR / "phases" / "phase8_streaming_strict_cap" / "results" / "two_tier_snapkv"
)

COLUMN_WIDTH_IN = 3.35
PALETTE = {
    "idlekv": "#0072B2",  # Okabe-Ito blue
    "gold": "#E69F00",  # Okabe-Ito orange
    "matched": "#333333",
    "random": "#8A8A8A",
    "oldest": "#009E73",  # Okabe-Ito bluish green
    "refresh": "#CC79A7",  # Okabe-Ito reddish purple
    "stale": "#6F6F6F",
    "wrong": "#A0A0A0",
    "proxy": "#D55E00",  # Okabe-Ito vermillion
    "grid": "#D9D9D9",
    "text": "#1B1B1B",
}
GAIN_CMAP = LinearSegmentedColormap.from_list(
    "idlekv_gain",
    ["#F7F7F7", "#D8EFF0", "#7FCDBB", "#2C7FB8", "#08306B"],
)
LATENCY_CMAP = LinearSegmentedColormap.from_list(
    "idlekv_latency",
    ["#F7FBFF", "#DEEBF7", "#9ECAE1", "#3182BD", "#08519C"],
)


@dataclass(frozen=True)
class OperatingRegimeSource:
    csv4q: Path
    csv6q: Path | None
    uses_final_6q: bool


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.3,
            "figure.titlesize": 8,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 360,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURE_DIR / f"{stem}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.018,
            facecolor="none",
        )
    plt.close(fig)


def load_numeric_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() == df[col].notna().sum():
            df[col] = converted
    return df


def _line_style(name: str) -> dict[str, object]:
    styles: dict[str, dict[str, object]] = {
        "IdleKV": {"color": PALETTE["idlekv"], "marker": "o", "linestyle": "-", "linewidth": 1.45},
        "Gold-K": {"color": PALETTE["gold"], "marker": "s", "linestyle": "--", "linewidth": 1.2},
        "Matched": {"color": PALETTE["matched"], "marker": "D", "linestyle": ":", "linewidth": 1.05},
        "Random-K": {"color": PALETTE["random"], "marker": "^", "linestyle": "-.", "linewidth": 1.0},
        "Oldest-K": {"color": PALETTE["oldest"], "marker": "x", "linestyle": (0, (3, 1.2)), "linewidth": 1.0},
    }
    return styles[name]


def _format_axes(ax: plt.Axes, *, x_label: str | None = None, y_label: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)
    if x_label:
        ax.set_xlabel(x_label, labelpad=1.5)
    if y_label:
        ax.set_ylabel(y_label, labelpad=1.5)
    ax.tick_params(pad=1.5)


def _plot_score_or_overlap(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    mode: str,
    panel_label: str,
    y_label: str,
) -> None:
    if mode == "score":
        series = {
            "IdleKV": "idlekv",
            "Matched": "b_match",
            "Random-K": "random_k",
            "Oldest-K": "oldest_k",
            "Gold-K": "oracle_k",
        }
    elif mode == "overlap":
        series = {
            "IdleKV": "idlekv_overlap",
            "Matched": "b_match_overlap",
            "Random-K": "random_k_overlap",
            "Oldest-K": "oldest_k_overlap",
            "Gold-K": "oracle_k_overlap",
        }
    else:
        raise ValueError(f"Unknown panel mode: {mode}")

    for label, column in series.items():
        if column not in df:
            continue
        style = _line_style(label)
        ax.plot(df["k"], df[column], label=label, **style)

    ax.set_ylim(-0.03, 1.04)
    ax.set_xlim(6, 131)
    ax.set_xticks([8, 16, 32, 48, 64, 96, 128])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    _format_axes(ax, y_label=y_label)
    ax.text(
        0.02,
        0.94,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        fontweight="bold",
        color=PALETTE["text"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.7},
    )


def render_selection_diagnostic() -> None:
    """Dense real-data diagnostic: quality and recovered-span overlap."""

    score_4q = load_numeric_csv(FIGURE_DIR / "phase7_clean_suite_b16384_exact_overall.csv")
    score_6q = load_numeric_csv(FIGURE_DIR / "phase7_mq_niah_6q_clean_suite_b18432_exact_overall.csv")
    overlap_4q = load_numeric_csv(FIGURE_DIR / "phase7_clean_suite_b16384_exact_overlap_overall.csv")
    overlap_6q = load_numeric_csv(FIGURE_DIR / "phase7_mq_niah_6q_clean_suite_b18432_exact_overlap_overall.csv")
    merged_4q = pd.merge(score_4q, overlap_4q, on="k", how="inner")
    merged_6q = pd.merge(score_6q, overlap_6q, on="k", how="inner")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(COLUMN_WIDTH_IN, 3.12),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.115, right=0.995, top=0.86, bottom=0.105, wspace=0.25, hspace=0.18)
    _plot_score_or_overlap(axes[0, 0], merged_4q, mode="score", panel_label="4Q", y_label="$Q_2$ score")
    _plot_score_or_overlap(axes[1, 0], merged_6q, mode="score", panel_label="6Q", y_label="$Q_2$ score")
    _plot_score_or_overlap(
        axes[0, 1],
        merged_4q,
        mode="overlap",
        panel_label="4Q",
        y_label="",
    )
    _plot_score_or_overlap(
        axes[1, 1],
        merged_6q,
        mode="overlap",
        panel_label="6Q",
        y_label="",
    )
    axes[0, 0].set_title("Exact answer", pad=2.0)
    axes[0, 1].set_title("Restored span coverage", pad=2.0)
    for ax in axes[1, :]:
        ax.set_xlabel("restore budget $K$", labelpad=1.0)
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=True)

    handles = [Line2D([0], [0], **_line_style(label)) for label in ["IdleKV", "Gold-K", "Matched", "Random-K", "Oldest-K"]]
    labels = ["IdleKV", "Gold-K", "Matched", "Random-K", "Oldest-K"]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.998),
        ncol=3,
        frameon=False,
        columnspacing=0.85,
        handlelength=1.65,
        handletextpad=0.35,
    )
    save_figure(fig, "repair_selection_diagnostic")


def _choose_operating_regime_source() -> OperatingRegimeSource:
    csv4q_paper = FIGURE_DIR / "phase9_phase_diagram_4q_final_n24.csv"
    csv6q_paper = FIGURE_DIR / "phase9_phase_diagram_6q_final_n24.csv"
    csv4q = csv4q_paper if csv4q_paper.exists() else PHASE9_DIR / "phase9_phase_diagram_4q_final_n24.csv"
    csv6q_final = csv6q_paper if csv6q_paper.exists() else PHASE9_DIR / "phase9_phase_diagram_6q_final_n24.csv"
    if csv6q_final.exists():
        return OperatingRegimeSource(csv4q=csv4q, csv6q=csv6q_final, uses_final_6q=True)
    return OperatingRegimeSource(csv4q=csv4q, csv6q=None, uses_final_6q=False)


def _fallback_6q_operating_regime() -> pd.DataFrame:
    # Preliminary stand-in values. They are intentionally plausible but not
    # used as evidence; the appendix caption marks the 6Q panel as preliminary
    # if the final sweep has not been inserted.
    rows = []
    for budget, values, headroom in [
        (12288, [0.03, 0.20, 0.52, 0.58], [0.45, 0.62, 0.35, 0.05]),
        (18432, [0.05, 0.24, 0.57, 0.58], [0.40, 0.34, 0.06, 0.04]),
        (24576, [0.01, 0.09, 0.25, 0.30], [0.31, 0.25, 0.12, 0.07]),
    ]:
        for k, value, oracle_gap in zip([16, 48, 96, 128], values, headroom):
            rows.append(
                {
                    "base_context_budget": budget,
                    "k": k,
                    "idlekv_lift": value,
                    "gold_headroom": oracle_gap,
                }
            )
    return pd.DataFrame(rows)


def _annotate_heatmap_cells(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    norm: Normalize,
) -> None:
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            color_value = norm(max(value, 0.0))
            rgba = GAIN_CMAP(color_value)
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "white" if luminance < 0.52 else PALETTE["text"]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=5.9, color=text_color)


def _normalized_recovery(row: object) -> float:
    if hasattr(row, "gold_normalized_recovery"):
        value = float(row.gold_normalized_recovery)
        if np.isfinite(value):
            return float(np.clip(value, 0.0, 1.0))
    gain = float(row.idlekv_lift)
    remaining = float(row.gold_headroom)
    denominator = gain + remaining
    if denominator <= 1e-12:
        return np.nan
    return float(np.clip(gain / denominator, 0.0, 1.0))


def _pivot_heatmap(df: pd.DataFrame) -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
    budgets = sorted(int(v) for v in df["base_context_budget"].dropna().unique())
    ks = sorted(int(v) for v in df["k"].dropna().unique())
    value_by_key = {
        (int(row.base_context_budget), int(row.k)): _normalized_recovery(row)
        for row in df.itertuples(index=False)
    }
    headroom_by_key = {
        (int(row.base_context_budget), int(row.k)): float(row.gold_headroom)
        for row in df.itertuples(index=False)
    }
    values = np.array([[value_by_key.get((budget, k), np.nan) for k in ks] for budget in budgets], dtype=float)
    headroom = np.array([[headroom_by_key.get((budget, k), 0.0) for k in ks] for budget in budgets], dtype=float)
    return budgets, ks, values, headroom


def _draw_heatmap_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    label: str,
    norm: Normalize,
) -> mpl.image.AxesImage:
    budgets, ks, values, _headroom = _pivot_heatmap(df)
    im = ax.imshow(np.clip(values, 0.0, 1.0), cmap=GAIN_CMAP, norm=norm, aspect="auto")
    ax.set_xticks(range(len(ks)), [str(k) for k in ks])
    ax.set_yticks(range(len(budgets)), [f"{budget / 1000:.1f}k" for budget in budgets])
    ax.set_ylabel(r"base budget $B_{\mathrm{base}}$", labelpad=1.0)
    ax.tick_params(length=0, pad=1.2)
    ax.set_xticks(np.arange(-0.5, len(ks), 1), minor=True)
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
        fontsize=6.4,
        fontweight="bold",
        color=PALETTE["text"],
    )
    _annotate_heatmap_cells(ax, values, norm=norm)
    return im


def render_operating_regime_heatmap() -> bool:
    """Render the operating-regime heatmap.

    Returns True when both panels use final n=24 data; False when the 6Q
    panel uses the explicitly captioned preliminary fallback.
    """

    source = _choose_operating_regime_source()
    df4q = load_numeric_csv(source.csv4q)
    df6q = load_numeric_csv(source.csv6q) if source.csv6q else _fallback_6q_operating_regime()

    norm = Normalize(vmin=0.0, vmax=1.0)

    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH_IN, 2.58), constrained_layout=False)
    fig.subplots_adjust(left=0.15, right=0.86, top=0.96, bottom=0.14, hspace=0.28)
    im = _draw_heatmap_panel(
        axes[0],
        df4q,
        label="MQ-NIAH-4Q",
        norm=norm,
    )
    _draw_heatmap_panel(
        axes[1],
        df6q,
        label="MQ-NIAH-6Q" if source.uses_final_6q else "MQ-NIAH-6Q prelim.",
        norm=norm,
    )
    axes[1].set_xlabel("restore budget $K$", labelpad=1.0)
    axes[0].tick_params(labelbottom=False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.05, pad=0.035)
    cbar.set_label("Gold-K headroom closed", labelpad=2.0)
    cbar.outline.set_linewidth(0.45)
    cbar.ax.tick_params(width=0.45, length=2.0, pad=1.2)
    save_figure(fig, "operating_regime_heatmap")
    return source.uses_final_6q


def _load_8q_full_frontier() -> pd.DataFrame | None:
    paper_csv = FIGURE_DIR / "phase10_mq_niah_8q_frontier_n24_overall.csv"
    if paper_csv.exists():
        rows = load_numeric_csv(paper_csv)
        rows = rows.rename(columns={"gold_k": "oracle_k"})
        needed = ["k", "b_match", "idlekv", "random_k", "oldest_k", "oracle_k"]
        if any(column not in rows for column in needed):
            return None
        return rows[needed].sort_values("k")

    full_csv = PHASE10_DIR / "mq_niah_8q_frontier_n24.csv"
    if not full_csv.exists():
        return None
    rows = load_numeric_csv(full_csv)
    rows = rows[rows["task"].astype(str) == "mq_niah_8q_clean_suite"].copy()
    if rows.empty:
        return None
    rows = rows.rename(columns={"gold_k": "oracle_k"})
    needed = ["k", "b_match", "idlekv", "random_k", "oldest_k", "oracle_k"]
    if any(column not in rows for column in needed):
        return None
    return rows[needed].sort_values("k")


def _load_2q_full_frontier() -> pd.DataFrame | None:
    paper_csv = FIGURE_DIR / "phase10_mq_niah_2q_frontier_n100_overall.csv"
    full_csv = PHASE10_DIR / "mq_niah_2q_frontier_n100.csv"
    if paper_csv.exists():
        rows = load_numeric_csv(paper_csv)
    elif full_csv.exists():
        rows = load_numeric_csv(full_csv)
        rows = rows[rows["task"].astype(str) == "mq_niah_2q_clean_suite"].copy()
        if rows.empty:
            return None
    else:
        return None
    rows = rows.rename(columns={"gold_k": "oracle_k"})
    needed = ["k", "b_match", "idlekv", "random_k", "oldest_k", "oracle_k"]
    if any(column not in rows for column in needed):
        return None
    return rows[needed].sort_values("k")


def _score_frontier(rows: pd.DataFrame, *, query_count: int) -> pd.DataFrame:
    df = rows.copy()
    df = df.rename(columns={"gold_k": "oracle_k"})
    needed = ["k", "b_match", "idlekv", "oracle_k"]
    missing = [column for column in needed if column not in df]
    if missing:
        raise ValueError(f"frontier rows are missing columns: {missing}")
    if "random_k" not in df:
        df["random_k"] = df["b_match"]
    if "oldest_k" not in df:
        df["oldest_k"] = df["b_match"]
    out = pd.DataFrame(
        {
            "query_count": int(query_count),
            "k": df["k"].astype(int),
            "b_match_score": df["b_match"].astype(float),
            "idlekv_score": df["idlekv"].astype(float),
            "random_score": df["random_k"].astype(float),
            "oldest_score": df["oldest_k"].astype(float),
            "gold_score": df["oracle_k"].astype(float),
        }
    )
    return out.sort_values("k")


def _load_frontier_datasets() -> list[tuple[int, pd.DataFrame]]:
    datasets: list[tuple[int, pd.DataFrame]] = []
    rows_2q = _load_2q_full_frontier()
    if rows_2q is not None:
        datasets.append((2, _score_frontier(rows_2q, query_count=2)))
    datasets.extend(
        [
            (4, _score_frontier(load_numeric_csv(FIGURE_DIR / "phase7_clean_suite_b16384_exact_overall.csv"), query_count=4)),
            (6, _score_frontier(load_numeric_csv(FIGURE_DIR / "phase7_mq_niah_6q_clean_suite_b18432_exact_overall.csv"), query_count=6)),
        ]
    )
    rows_8q = _load_8q_full_frontier()
    if rows_8q is not None:
        datasets.append((8, _score_frontier(rows_8q, query_count=8)))
    return datasets


def render_frontier_raw_overlay() -> bool:
    """Render a headline single-axis frontier with only the core contrast."""

    datasets = _load_frontier_datasets()
    if not datasets:
        return False

    query_palette = {
        2: "#CC79A7",
        4: PALETTE["idlekv"],
        6: PALETTE["proxy"],
        8: PALETTE["oldest"],
    }
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.42), constrained_layout=False)
    fig.subplots_adjust(left=0.15, right=0.90, top=0.84, bottom=0.18)
    label_y_positions = {
        2: 0.995,
        4: 0.890,
        6: 0.835,
        8: 0.780,
    }

    for query_count, df in datasets:
        color = query_palette.get(query_count, PALETTE["idlekv"])
        x = df["k"].to_numpy(dtype=float)
        ax.plot(
            x,
            df["b_match_score"],
            color=color,
            linestyle=(0, (1.0, 1.3)),
            linewidth=0.9,
            alpha=0.46,
            zorder=1,
        )
        ax.plot(
            x,
            df["idlekv_score"],
            color=color,
            linestyle="-",
            marker="o",
            markersize=2.95,
            markeredgecolor="white",
            markeredgewidth=0.35,
            linewidth=1.55,
            zorder=3,
        )
        endpoint = df[df["k"] == 96].iloc[0] if (df["k"] == 96).any() else df.iloc[-1]
        endpoint_delta = float(endpoint["idlekv_score"] - endpoint["b_match_score"])
        ax.text(
            131.0,
            label_y_positions.get(query_count, float(endpoint["idlekv_score"])),
            f"{query_count}Q  {endpoint_delta:+.2f}",
            ha="left",
            va="center",
            fontsize=5.65,
            color=color,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.20},
            clip_on=False,
        )

    ax.set_xlim(6, 145)
    ax.set_ylim(-0.03, 1.04)
    ax.set_xticks([8, 32, 64, 96, 128])
    ax.set_yticks([0.0, 0.5, 1.0])
    _format_axes(ax, x_label="restore budget $K$", y_label="exact $Q_2$ score")
    handles = [
        Line2D([0], [0], color=PALETTE["text"], linewidth=1.55, linestyle="-", marker="o", markersize=2.95, label="IdleKV"),
        Line2D([0], [0], color=PALETTE["text"], linewidth=0.9, linestyle=(0, (1.0, 1.3)), alpha=0.55, label="Matched"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=2,
        frameon=False,
        columnspacing=0.85,
        handlelength=1.35,
        handletextpad=0.32,
        borderaxespad=0.0,
        fontsize=5.9,
    )
    save_figure(fig, "frontier_raw_overlay")
    return True


def render_specificity_panel() -> bool:
    """Render the locked specificity contrast when data exists."""
    paper_locked = FIGURE_DIR / "specificity_locked_n24_k48.csv"
    locked_paths = [paper_locked] if paper_locked.exists() else sorted(
        PHASE10_DIR.glob("specificity_locked_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not locked_paths:
        for ext in ("pdf", "png"):
            (FIGURE_DIR / f"specificity_control_dotplot.{ext}").unlink(missing_ok=True)
        return False
    csv_path = locked_paths[0]

    df = load_numeric_csv(csv_path)
    if df.empty:
        return False

    condition_order = ["StaleQ-K", "WrongQ-K", "IdleKV", "Refresh-K", "Gold-K"]
    labels = {
        "Matched": "Matched",
        "StaleQ-K": "Stale query",
        "WrongQ-K": "Donor query",
        "Refresh-K": "Refresh-buffered",
        "IdleKV": "IdleKV",
        "Gold-K": "Gold-K",
    }
    colors = {
        "Matched": PALETTE["matched"],
        "StaleQ-K": PALETTE["stale"],
        "WrongQ-K": PALETTE["wrong"],
        "Refresh-K": PALETTE["refresh"],
        "IdleKV": PALETTE["idlekv"],
        "Gold-K": PALETTE["gold"],
    }
    markers = {
        "Matched": "D",
        "StaleQ-K": "v",
        "WrongQ-K": "^",
        "Refresh-K": "P",
        "IdleKV": "o",
        "Gold-K": "s",
    }
    ks = sorted(int(value) for value in df["k"].dropna().unique())
    k = ks[0]
    subset = df[df["k"] == k]
    if subset.empty:
        return False

    row_by_condition = {str(row.condition): row for row in subset.itertuples(index=False)}
    matched_row = row_by_condition.get("Matched")
    if matched_row is None or pd.isna(matched_row.mean_score):
        return False
    y_positions = np.arange(len(condition_order))[::-1]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 1.42), constrained_layout=False)
    fig.subplots_adjust(left=0.30, right=0.985, top=0.93, bottom=0.27)

    ax.axvspan(-0.025, 0.025, color="#EFEFEF", alpha=0.82, zorder=0)
    ax.axhline(2.5, color=PALETTE["grid"], linewidth=0.35, alpha=0.7, zorder=1)
    ax.axhline(0.5, color=PALETTE["grid"], linewidth=0.35, alpha=0.7, zorder=1)
    ax.axvline(0.0, color=PALETTE["matched"], linewidth=0.95, linestyle=(0, (1.0, 1.3)), zorder=2)

    for y, condition in zip(y_positions, condition_order, strict=True):
        row = row_by_condition.get(condition)
        if row is None or pd.isna(row.mean_score):
            continue
        gain = float(row.mean_gain_vs_matched)
        ax.hlines(
            y,
            min(0.0, gain),
            max(0.0, gain),
            color=colors[condition],
            linewidth=4.4,
            alpha=0.22 if condition in {"StaleQ-K", "WrongQ-K"} else 0.30,
            zorder=2,
        )
        if not pd.isna(row.gain_ci95_low) and not pd.isna(row.gain_ci95_high):
            lo = float(row.gain_ci95_low)
            hi = float(row.gain_ci95_high)
            ax.errorbar(
                gain,
                y,
                xerr=[
                    [max(gain - lo, 0.0)],
                    [max(hi - gain, 0.0)],
                ],
                fmt="none",
                ecolor=colors[condition],
                elinewidth=0.78,
                capsize=1.25,
                capthick=0.52,
                alpha=0.80,
                zorder=3,
            )
        facecolor = "white" if condition in {"Refresh-K", "Gold-K"} else colors[condition]
        ax.scatter(
            gain,
            y,
            s=20 if condition == "IdleKV" else 17,
            marker=markers[condition],
            facecolor=facecolor,
            edgecolor=colors[condition],
            linewidth=0.82 if condition in {"Refresh-K", "Gold-K"} else 0.32,
            zorder=4,
        )
        if condition not in {"StaleQ-K", "WrongQ-K"}:
            ax.text(
                gain + 0.025,
                y,
                f"+{gain:.2f}",
                ha="left",
                va="center",
                fontsize=5.6,
                color=colors[condition],
                fontweight="bold" if condition == "IdleKV" else "normal",
                clip_on=False,
            )

    ax.set_xlim(-0.07, 0.86)
    ax.set_xticks([0.0, 0.4, 0.8])
    ax.set_ylim(-0.55, len(condition_order) - 0.45)
    _format_axes(ax, x_label="score gain over matched no-repair", y_label=None)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.42, alpha=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_yticks(y_positions, [labels[value] for value in condition_order])
    ax.tick_params(axis="y", labelsize=6.35, length=0, pad=1.6)
    ax.tick_params(axis="x", pad=1.1)
    save_figure(fig, "specificity_control_dotplot")
    return True


def render_query_count_breadth() -> bool:
    """Render a query-count breadth panel once locked breadth data exists."""
    locked_csv = (
        FIGURE_DIR / "query_count_locked_n12.csv"
        if (FIGURE_DIR / "query_count_locked_n12.csv").exists()
        else PHASE10_DIR / "query_count_locked_n12.csv"
    )
    endpoint_csv = (
        FIGURE_DIR / "query_count_even_locked_n24.csv"
        if (FIGURE_DIR / "query_count_even_locked_n24.csv").exists()
        else PHASE10_DIR / "query_count_even_locked_n24.csv"
    )
    full_2q_csv = (
        FIGURE_DIR / "phase10_mq_niah_2q_frontier_n100_overall.csv"
        if (FIGURE_DIR / "phase10_mq_niah_2q_frontier_n100_overall.csv").exists()
        else PHASE10_DIR / "mq_niah_2q_frontier_n100.csv"
    )
    full_8q_csv = (
        FIGURE_DIR / "phase10_mq_niah_8q_frontier_n24_overall.csv"
        if (FIGURE_DIR / "phase10_mq_niah_8q_frontier_n24_overall.csv").exists()
        else PHASE10_DIR / "mq_niah_8q_frontier_n24.csv"
    )
    if not locked_csv.exists() and not endpoint_csv.exists() and not full_2q_csv.exists() and not full_8q_csv.exists():
        for ext in ("pdf", "png"):
            (FIGURE_DIR / f"query_count_breadth.{ext}").unlink(missing_ok=True)
        return False

    records_by_key: dict[tuple[int, int], dict[str, float | int]] = {}
    task_to_count = {
        "mq_niah_2q_clean_suite": 2,
        "mq_niah_3q_clean_suite": 3,
        "mq_niah_8q_clean_suite": 8,
    }

    def _add_query_count_records(path: Path, *, allowed_counts: set[int]) -> None:
        if not path.exists():
            return
        rows = load_numeric_csv(path)
        for row in rows.itertuples(index=False):
            query_count = task_to_count.get(str(row.task)) if hasattr(row, "task") else None
            if query_count is None and hasattr(row, "query_count"):
                query_count = int(row.query_count)
            k = int(row.k)
            if query_count not in allowed_counts or k not in {48, 96}:
                continue
            b_match = float(row.b_match)
            records_by_key[(query_count, k)] = {
                "query_count": query_count,
                "k": k,
                "idlekv_gain": float(row.idlekv) - b_match,
                "gold_gain": float(row.gold_k) - b_match,
            }

    _add_query_count_records(locked_csv, allowed_counts={2, 3, 8})
    _add_query_count_records(endpoint_csv, allowed_counts={2, 8})
    _add_query_count_records(full_2q_csv, allowed_counts={2})
    _add_query_count_records(full_8q_csv, allowed_counts={8})

    for query_count, csv_name in (
        (4, "mq_niah_4q_frontier.csv"),
        (6, "mq_niah_6q_frontier.csv"),
    ):
        frontier_path = FIGURE_DIR / csv_name
        if not frontier_path.exists():
            continue
        frontier = load_numeric_csv(frontier_path)
        for row in frontier[frontier["k"].isin([48, 96])].itertuples(index=False):
            k = int(row.k)
            records_by_key[(query_count, k)] = {
                "query_count": query_count,
                "k": k,
                "idlekv_gain": float(row.idlekv_gain),
                "gold_gain": float(row.goldk_gain),
            }

    records = list(records_by_key.values())
    if not records:
        return False

    df = pd.DataFrame.from_records(records)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 1.55), constrained_layout=False)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.78, bottom=0.25)

    styles = {
        48: {"color": PALETTE["idlekv"], "marker": "o", "label": "$K=48$"},
        96: {"color": PALETTE["proxy"], "marker": "s", "label": "$K=96$"},
    }
    for k, style in styles.items():
        subset = df[df["k"] == k].sort_values("query_count")
        if subset.empty:
            continue
        xs = subset["query_count"].to_numpy(dtype=float)
        gains = subset["idlekv_gain"].to_numpy(dtype=float)
        ax.plot(
            xs,
            gains,
            color=style["color"],
            linewidth=1.05,
            alpha=0.86,
            zorder=2,
        )
        ax.scatter(
            xs,
            gains,
            color=style["color"],
            marker=style["marker"],
            label=str(style["label"]),
            s=16,
            linewidth=0.35,
            edgecolor="white",
            zorder=3,
        )

    ax.set_xlim(1.55, 8.45)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks([2, 3, 4, 6, 8])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("number of future-turn questions", labelpad=1.0)
    ax.set_ylabel("score gain over matched", labelpad=1.0)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.42, alpha=0.82)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(pad=1.2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=2,
        frameon=False,
        handlelength=1.3,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
    )
    save_figure(fig, "query_count_breadth")
    return True


def render_streaming_spill_heatmap() -> bool:
    """Render strict active-cache spill coverage as an appendix heatmap."""

    paths = sorted(
        PHASE8_STREAMING_DIR.glob(
            "clean_suite_l*_cap32768_h512_chunk2048_n4_x0.5-0.625-0.75-0.875-1.json"
        )
    )
    if not paths:
        for ext in ("pdf", "png"):
            (FIGURE_DIR / f"streaming_spill_heatmap.{ext}").unlink(missing_ok=True)
        return False

    spill_fracs = [0.5, 0.625, 0.75, 0.875]
    rows: list[tuple[int, list[float]]] = []
    for path in paths:
        try:
            logical_tokens = int(path.name.split("_l", maxsplit=1)[1].split("_", maxsplit=1)[0])
        except (IndexError, ValueError):
            continue
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        aggregate = payload.get("aggregate", {})
        values = []
        for frac in spill_fracs:
            key = f"x{frac:g}"
            values.append(float(aggregate[key]["mean_accessible_coverage"]))
        rows.append((logical_tokens, values))

    if not rows:
        return False
    rows.sort(key=lambda item: item[0])
    logical_contexts = [tokens for tokens, _ in rows]
    data = np.array([values for _, values in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 1.95), constrained_layout=False)
    fig.subplots_adjust(left=0.17, right=0.90, top=0.93, bottom=0.23)
    image = ax.imshow(data, cmap=GAIN_CMAP, vmin=0.30, vmax=1.0, aspect="auto")
    ax.set_xticks(
        np.arange(len(spill_fracs)),
        [f"{frac:.3g}" for frac in spill_fracs],
    )
    ax.set_yticks(
        np.arange(len(logical_contexts)),
        [f"{tokens // 1024}K" for tokens in logical_contexts],
    )
    ax.set_xlabel("CPU spill fraction $X$", labelpad=1.0)
    ax.set_ylabel("logical context", labelpad=1.0)
    ax.tick_params(length=0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value >= 0.70 else PALETTE["text"],
                fontsize=5.8,
            )

    cax = fig.add_axes([0.925, 0.23, 0.025, 0.70])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_ticks([0.4, 0.6, 0.8, 1.0])
    colorbar.ax.tick_params(labelsize=5.9, width=0.45, length=2.0, pad=1.0)
    colorbar.outline.set_linewidth(0.45)
    colorbar.set_label("accessible coverage", fontsize=6.2, labelpad=2.0)
    save_figure(fig, "streaming_spill_heatmap")
    return True


def render_proxy_latency_tradeoff() -> bool:
    """Render exact-vs-proxy scorer latency as an appendix tradeoff plot."""

    configs = [
        (
            "4Q",
            FIGURE_DIR / "phase9_proxy_exact_4q_reference.csv"
            if (FIGURE_DIR / "phase9_proxy_exact_4q_reference.csv").exists()
            else PHASE9_DIR / "phase9_proxy_exact_4q_reference.csv",
            FIGURE_DIR / "phase9_proxy_4q_full_n100.csv"
            if (FIGURE_DIR / "phase9_proxy_4q_full_n100.csv").exists()
            else PHASE9_DIR / "phase9_proxy_4q_full_n100.csv",
            FIGURE_DIR / "phase9_proxy_4q_full_n100_paired.csv"
            if (FIGURE_DIR / "phase9_proxy_4q_full_n100_paired.csv").exists()
            else PHASE9_DIR / "phase9_proxy_4q_full_n100_paired.csv",
        ),
        (
            "6Q",
            FIGURE_DIR / "phase9_proxy_exact_6q_reference.csv"
            if (FIGURE_DIR / "phase9_proxy_exact_6q_reference.csv").exists()
            else PHASE9_DIR / "phase9_proxy_exact_6q_reference.csv",
            FIGURE_DIR / "phase9_proxy_6q_full_n100.csv"
            if (FIGURE_DIR / "phase9_proxy_6q_full_n100.csv").exists()
            else PHASE9_DIR / "phase9_proxy_6q_full_n100.csv",
            FIGURE_DIR / "phase9_proxy_6q_full_n100_paired.csv"
            if (FIGURE_DIR / "phase9_proxy_6q_full_n100_paired.csv").exists()
            else PHASE9_DIR / "phase9_proxy_6q_full_n100_paired.csv",
        ),
    ]
    if not all(path.exists() for _, exact, proxy, paired in configs for path in (exact, proxy, paired)):
        for ext in ("pdf", "png"):
            (FIGURE_DIR / f"proxy_latency_tradeoff.{ext}").unlink(missing_ok=True)
        return False

    fig, axes_raw = plt.subplots(1, 2, figsize=(COLUMN_WIDTH_IN, 1.70), constrained_layout=False)
    axes = np.atleast_1d(axes_raw)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.82, bottom=0.28, wspace=0.25)

    k_styles = {
        48: {"color": PALETTE["idlekv"], "label": "$K=48$"},
        96: {"color": PALETTE["proxy"], "label": "$K=96$"},
    }
    method_markers = {
        "Move only": {"marker": "x", "size": 28},
        "Proxy": {"marker": "s", "size": 23},
        "Exact": {"marker": "o", "size": 23},
    }

    for ax, (variant, exact_path, proxy_path, paired_path) in zip(axes, configs, strict=True):
        exact = load_numeric_csv(exact_path).set_index("k")
        proxy = load_numeric_csv(proxy_path).set_index("k")
        paired = load_numeric_csv(paired_path).set_index("k")
        for k, style in k_styles.items():
            if k not in exact.index or k not in proxy.index or k not in paired.index:
                continue
            exact_row = exact.loc[k]
            proxy_row = proxy.loc[k]
            paired_row = paired.loc[k]
            move_s = max(
                0.004,
                (float(exact_row["p50_transfer_ms"]) + float(exact_row["p50_inject_ms"])) / 1000.0,
            )
            points = [
                ("Move only", move_s, float(paired_row["exact_b_match"])),
                ("Proxy", float(proxy_row["p50_total_ms"]) / 1000.0, float(paired_row["proxy_idlekv"])),
                ("Exact", float(exact_row["p50_total_ms"]) / 1000.0, float(paired_row["exact_idlekv"])),
            ]
            for method, x_value, y_value in points:
                marker_style = method_markers[method]
                if method == "Move only":
                    ax.scatter(
                        [x_value],
                        [y_value],
                        marker=str(marker_style["marker"]),
                        s=float(marker_style["size"]),
                        color=str(style["color"]),
                        linewidth=0.9,
                        zorder=3,
                    )
                else:
                    ax.scatter(
                        [x_value],
                        [y_value],
                        marker=str(marker_style["marker"]),
                        s=float(marker_style["size"]),
                        facecolor="white",
                        edgecolor=str(style["color"]),
                        color=str(style["color"]),
                        linewidth=0.85,
                        zorder=3,
                    )
            exact_y = float(paired_row["exact_idlekv"])
            ax.text(
                float(exact_row["p50_total_ms"]) / 1000.0 * 0.94,
                exact_y + (0.035 if k == 96 else -0.045),
                f"$K={k}$",
                color=str(style["color"]),
                ha="right",
                va="center",
                fontsize=5.8,
            )

        ax.set_xscale("log")
        ax.set_xlim(0.0035, 9.5)
        ax.set_ylim(0.15, 1.04)
        ax.set_xticks([0.005, 0.05, 0.5, 5.0], ["5ms", "50ms", "0.5s", "5s"])
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.set_yticks([0.25, 0.50, 0.75, 1.00])
        ax.text(
            0.03,
            0.95,
            variant,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.5},
        )
        _format_axes(ax, x_label="p50 repair latency")

    axes[0].set_ylabel("exact $Q_2$ score", labelpad=1.0)
    handles = [
        Line2D([0], [0], marker="x", linestyle="None", color=PALETTE["matched"], label="Move only", markersize=4.2),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor=PALETTE["matched"],
            color=PALETTE["matched"],
            label="Proxy",
            markersize=4.0,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor=PALETTE["matched"],
            color=PALETTE["matched"],
            label="Exact",
            markersize=4.0,
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=3,
        frameon=False,
        handlelength=0.75,
        handletextpad=0.25,
        columnspacing=0.55,
        borderaxespad=0.0,
        fontsize=5.75,
    )
    save_figure(fig, "proxy_latency_tradeoff")
    return True


def render_runtime_capacity_curve() -> bool:
    """Render move+inject capacity from the synthetic Qwen-shaped KV profiler."""

    robust_path = (
        FIGURE_DIR / "runtime_capacity_8k_32k_100k.csv"
        if (FIGURE_DIR / "runtime_capacity_8k_32k_100k.csv").exists()
        else next(iter(sorted(PHASE4_RUNTIME_DIR.glob("runtime_capacity_robust_*.csv"))), None)
    )
    large_path = (
        FIGURE_DIR / "runtime_capacity_250k_500k_k5000.csv"
        if (FIGURE_DIR / "runtime_capacity_250k_500k_k5000.csv").exists()
        else next(iter(sorted(PHASE4_RUNTIME_DIR.glob("runtime_capacity_large_*.csv"))), None)
    )
    if robust_path is None or not robust_path.exists() or large_path is None or not large_path.exists():
        for ext in ("pdf", "png"):
            (FIGURE_DIR / f"runtime_capacity_curve.{ext}").unlink(missing_ok=True)
        return False

    robust = load_numeric_csv(robust_path)
    large = load_numeric_csv(large_path)
    combined = pd.concat([robust, large], ignore_index=True)
    k5000 = combined[combined["k"].astype(int) == 5000].copy()
    if robust.empty or k5000.empty:
        return False

    fig, axes_raw = plt.subplots(1, 2, figsize=(COLUMN_WIDTH_IN, 1.58), constrained_layout=False)
    axes = np.atleast_1d(axes_raw)
    fig.subplots_adjust(left=0.15, right=0.99, top=0.94, bottom=0.31, wspace=0.36)

    active_styles = {
        8192: {"label": "8K active", "color": "#56B4E9", "marker": "o"},
        32768: {"label": "32K active", "color": PALETTE["idlekv"], "marker": "s"},
        100000: {"label": "100K active", "color": PALETTE["proxy"], "marker": "^"},
    }
    ax = axes[0]
    for active_tokens, style in active_styles.items():
        subset = robust[robust["active_tokens"].astype(int) == active_tokens].sort_values("k")
        if subset.empty:
            continue
        ax.plot(
            subset["k"].astype(float),
            subset["p95_total_ms"].astype(float),
            color=str(style["color"]),
            marker=str(style["marker"]),
            linewidth=1.15,
            markersize=3.1,
            label=str(style["label"]),
        )
    ax.set_xscale("log")
    ax.set_xticks([96, 500, 1000, 5000], ["96", "500", "1k", "5k"])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_ylim(0, 30)
    ax.set_yticks([0, 10, 20, 30])
    ax.legend(
        loc="upper left",
        frameon=False,
        handlelength=1.0,
        borderaxespad=0.0,
        labelspacing=0.15,
        fontsize=5.7,
    )
    _format_axes(ax, x_label="restored rows $K$", y_label="p95 move+inject (ms)")

    ax = axes[1]
    k5000 = k5000.sort_values("active_tokens")
    active_k = k5000["active_tokens"].astype(float)
    p95 = k5000["p95_total_ms"].astype(float)
    ax.plot(
        active_k,
        p95,
        color=PALETTE["idlekv"],
        marker="o",
        linewidth=1.25,
        markersize=3.2,
    )
    for x_value, y_value in zip(active_k, p95, strict=True):
        label = f"{int(round(y_value))}ms"
        ax.text(x_value * 1.035, y_value, label, ha="left", va="center", fontsize=5.1, color=PALETTE["idlekv"])
    ax.axhline(100.0, color=PALETTE["matched"], linestyle=(0, (2, 1.4)), linewidth=0.8)
    ax.axhline(500.0, color=PALETTE["wrong"], linestyle=(0, (2, 1.4)), linewidth=0.65)
    ax.text(9_000, 104.0, "0.1s", ha="left", va="bottom", fontsize=5.6, color=PALETTE["matched"])
    ax.text(9_000, 504.0, "0.5s", ha="left", va="bottom", fontsize=5.6, color=PALETTE["wrong"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(7_000, 720_000)
    ax.set_ylim(5, 700)
    ax.set_xticks([10_000, 100_000, 500_000], ["10k", "100k", "500k"])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_yticks([10, 100, 500], ["10", "100", "500"])
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.text(0.98, 0.08, "$K=5000$", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8)
    _format_axes(ax, x_label="active cache rows", y_label=None)

    save_figure(fig, "runtime_capacity_curve")
    return True


def _runtime_select_with_move(
    select_df: pd.DataFrame,
    move_df: pd.DataFrame,
    *,
    active_tokens: int = 32768,
    percentile_field: str = "p95_total_ms",
) -> pd.DataFrame:
    """Combine chunked-selection latency with measured move/inject latency."""
    required_select = {"candidate_tokens", "k", percentile_field}
    required_move = {"active_tokens", "k", percentile_field}
    if not required_select.issubset(select_df.columns):
        missing = sorted(required_select.difference(select_df.columns))
        raise ValueError(f"selection CSV missing columns: {missing}")
    if not required_move.issubset(move_df.columns):
        missing = sorted(required_move.difference(move_df.columns))
        raise ValueError(f"move CSV missing columns: {missing}")

    move_subset = move_df[move_df["active_tokens"].astype(int) == int(active_tokens)]
    records: list[dict[str, float | int]] = []
    for row in select_df.itertuples(index=False):
        k_value = int(getattr(row, "k"))
        move_row = move_subset[move_subset["k"].astype(int) == k_value]
        if move_row.empty:
            continue
        select_ms = float(getattr(row, percentile_field))
        move_ms = float(move_row.iloc[0][percentile_field])
        records.append(
            {
                "candidate_tokens": int(getattr(row, "candidate_tokens")),
                "k": k_value,
                "select_ms": select_ms,
                "move_ms": move_ms,
                "repair_ms": select_ms + move_ms,
                "repair_s": (select_ms + move_ms) / 1000.0,
            }
        )
    return pd.DataFrame.from_records(records)


def _runtime_e2e_repair_frame(
    e2e_df: pd.DataFrame,
    *,
    active_tokens: int = 32768,
    query_len: int = 64,
    percentile_field: str = "p95_total_ms",
) -> pd.DataFrame:
    """Normalize integrated end-to-end repair rows for plotting."""
    required = {"active_tokens", "candidate_tokens", "k", percentile_field}
    if not required.issubset(e2e_df.columns):
        missing = sorted(required.difference(e2e_df.columns))
        raise ValueError(f"end-to-end runtime CSV missing columns: {missing}")
    subset = e2e_df[e2e_df["active_tokens"].astype(int) == int(active_tokens)].copy()
    if "query_len" in subset.columns:
        subset = subset[subset["query_len"].astype(int) == int(query_len)].copy()
    records: list[dict[str, float | int]] = []
    for row in subset.itertuples(index=False):
        repair_ms = float(getattr(row, percentile_field))
        records.append(
            {
                "candidate_tokens": int(getattr(row, "candidate_tokens")),
                "k": int(getattr(row, "k")),
                "repair_ms": repair_ms,
                "repair_s": repair_ms / 1000.0,
            }
        )
    return pd.DataFrame.from_records(records)


def _runtime_component_shares(
    select_df: pd.DataFrame,
    move_df: pd.DataFrame,
    *,
    active_tokens: int = 32768,
    query_len: int = 64,
    k_value: int = 5000,
) -> pd.DataFrame:
    """Return p95 component shares for the decomposed synthetic repair path."""
    required_select = {"candidate_tokens", "k", "query_len", "p95_scan_ms", "p95_topk_ms"}
    required_move = {"active_tokens", "k", "p95_transfer_ms", "p95_inject_ms"}
    if not required_select.issubset(select_df.columns):
        missing = sorted(required_select.difference(select_df.columns))
        raise ValueError(f"selection CSV missing columns: {missing}")
    if not required_move.issubset(move_df.columns):
        missing = sorted(required_move.difference(move_df.columns))
        raise ValueError(f"move CSV missing columns: {missing}")
    select_subset = select_df[
        (select_df["query_len"].astype(int) == int(query_len))
        & (select_df["k"].astype(int) == int(k_value))
    ].copy()
    move_subset = move_df[
        (move_df["active_tokens"].astype(int) == int(active_tokens))
        & (move_df["k"].astype(int) == int(k_value))
    ]
    if select_subset.empty or move_subset.empty:
        return pd.DataFrame()
    move_row = move_subset.iloc[0]
    copy_insert_ms = float(move_row["p95_transfer_ms"]) + float(move_row["p95_inject_ms"])
    records: list[dict[str, float | int]] = []
    for row in select_subset.sort_values("candidate_tokens").itertuples(index=False):
        scan_ms = float(getattr(row, "p95_scan_ms"))
        topk_ms = float(getattr(row, "p95_topk_ms"))
        total_ms = scan_ms + topk_ms + copy_insert_ms
        if total_ms <= 0:
            continue
        records.append(
            {
                "candidate_tokens": int(getattr(row, "candidate_tokens")),
                "scan_share": scan_ms / total_ms,
                "topk_share": topk_ms / total_ms,
                "copy_insert_share": copy_insert_ms / total_ms,
                "component_total_ms": total_ms,
            }
        )
    return pd.DataFrame.from_records(records)


def _runtime_repair_grid(
    select_df: pd.DataFrame,
    move_df: pd.DataFrame,
    *,
    active_tokens: int = 32768,
    query_len: int = 64,
    percentile_field: str = "p95_total_ms",
) -> tuple[list[int], list[int], np.ndarray]:
    """Return a K-by-candidate grid of component-summed repair latency."""
    if "query_len" not in select_df.columns:
        raise ValueError("selection CSV missing query_len column")
    subset = select_df[select_df["query_len"].astype(int) == int(query_len)].copy()
    repair = _runtime_select_with_move(
        subset,
        move_df,
        active_tokens=active_tokens,
        percentile_field=percentile_field,
    )
    if repair.empty:
        return [], [], np.empty((0, 0), dtype=float)
    ks = sorted(int(value) for value in repair["k"].dropna().unique())
    candidates = sorted(int(value) for value in repair["candidate_tokens"].dropna().unique())
    value_by_key = {
        (int(row.k), int(row.candidate_tokens)): float(row.repair_s)
        for row in repair.itertuples(index=False)
    }
    values = np.array(
        [[value_by_key.get((k_value, candidate_tokens), np.nan) for candidate_tokens in candidates] for k_value in ks],
        dtype=float,
    )
    return ks, candidates, values


def _format_runtime_cell(seconds: float) -> str:
    if not np.isfinite(seconds):
        return ""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _format_candidate_rows(value: int) -> str:
    if int(value) >= 1_048_576 and int(value) % 1_048_576 == 0:
        return f"{int(value) // 1_048_576}M"
    return f"{int(value) // 1024}K"


def _max_measured_candidates_by_idle(
    repair_df: pd.DataFrame,
    *,
    k_value: int,
    idle_windows_s: Iterable[float] = (0.1, 0.5, 1.0, 2.0, 5.0),
    budget_fraction: float = 0.90,
) -> pd.DataFrame:
    """Return largest measured candidate store fitting each idle-window budget."""
    if not 0.0 < float(budget_fraction) <= 1.0:
        raise ValueError("budget_fraction must lie in (0, 1].")
    subset = repair_df[repair_df["k"].astype(int) == int(k_value)].copy()
    records: list[dict[str, float | int]] = []
    for idle_s in idle_windows_s:
        budget_s = float(idle_s) * float(budget_fraction)
        fitting = subset[subset["repair_s"].astype(float) <= budget_s]
        records.append(
            {
                "idle_window_s": float(idle_s),
                "budget_s": float(budget_s),
                "max_candidate_tokens": int(fitting["candidate_tokens"].max()) if not fitting.empty else 0,
            }
        )
    return pd.DataFrame.from_records(records)


def render_runtime_repair_scaling() -> bool:
    """Render measured repair time as offloaded candidate context grows."""

    e2e_path = next(
        (
            path
            for path in (
                FIGURE_DIR / "runtime_e2e_frontier_32k.csv",
                FIGURE_DIR / "runtime_e2e_frontier.csv",
            )
            if path.exists()
        ),
        next(iter(sorted(PHASE4_RUNTIME_DIR.glob("runtime_e2e_frontier_*.csv"), reverse=True)), None),
    )
    select_path = next(
        (
            path
            for path in (
                FIGURE_DIR / "runtime_latency_envelope_select.csv",
                FIGURE_DIR / "runtime_chunked_select_fullpool_32k_1m.csv",
                FIGURE_DIR / "runtime_chunked_select_32k_1m.csv",
            )
            if path.exists()
        ),
        next(
            iter(
                sorted(PHASE4_RUNTIME_DIR.glob("runtime_latency_envelope_[0-9]*_select.csv"), reverse=True)
                + sorted(PHASE4_RUNTIME_DIR.glob("runtime_chunked_select_fullpool_*.csv"), reverse=True)
            ),
            None,
        ),
    )
    move_path = next(
        (
            path
            for path in (
                FIGURE_DIR / "runtime_latency_envelope_move.csv",
                FIGURE_DIR / "runtime_move_inject_32k.csv",
                FIGURE_DIR / "runtime_capacity_8k_32k_100k.csv",
            )
            if path.exists()
        ),
        next(
            iter(
                sorted(PHASE4_RUNTIME_DIR.glob("runtime_latency_envelope_[0-9]*_move.csv"), reverse=True)
                + sorted(PHASE4_RUNTIME_DIR.glob("runtime_move_inject_32k_*.csv"), reverse=True)
            ),
            None,
        ),
    )
    has_e2e = e2e_path is not None and e2e_path.exists()
    has_decomposed = (
        select_path is not None
        and select_path.exists()
        and move_path is not None
        and move_path.exists()
    )
    if not has_e2e and not has_decomposed:
        for ext in ("pdf", "png"):
            (FIGURE_DIR / f"runtime_repair_scaling.{ext}").unlink(missing_ok=True)
        return False

    repair_df = pd.DataFrame()
    component_df = pd.DataFrame()
    select_all = pd.DataFrame()
    if has_decomposed:
        select_all = load_numeric_csv(select_path)
        move_df = load_numeric_csv(move_path)
        if "host_pool_coverage" in select_all.columns and float(select_all["host_pool_coverage"].min()) < 0.999:
            return False
        select_df = select_all[select_all["query_len"].astype(int) == 64].copy() if "query_len" in select_all.columns else select_all
        move_32k = move_df[move_df["active_tokens"].astype(int) == 32768]
        if select_df.empty or move_32k.empty:
            return False
        repair_df = _runtime_select_with_move(select_df, move_df, active_tokens=32768, percentile_field="p95_total_ms")
        component_df = _runtime_component_shares(
            select_all,
            move_df,
            active_tokens=32768,
            query_len=64,
            k_value=5000,
        )
    else:
        e2e_df = load_numeric_csv(e2e_path)
        if "host_pool_coverage" in e2e_df.columns and float(e2e_df["host_pool_coverage"].min()) < 0.999:
            return False
        repair_df = _runtime_e2e_repair_frame(
            e2e_df,
            active_tokens=32768,
            query_len=64,
            percentile_field="p95_total_ms",
        )
    if repair_df.empty:
        return False

    if has_decomposed and not component_df.empty and "query_len" in select_all.columns:
        ks, candidates, latency_grid = _runtime_repair_grid(
            select_all,
            move_df,
            active_tokens=32768,
            query_len=64,
            percentile_field="p95_total_ms",
        )
        if not ks or not candidates:
            return False
        fig, axes_raw = plt.subplots(2, 1, figsize=(COLUMN_WIDTH_IN, 2.78), constrained_layout=False)
        axes = np.atleast_1d(axes_raw)
        fig.subplots_adjust(left=0.16, right=0.90, top=0.965, bottom=0.13, hspace=0.45)

        ax = axes[0]
        norm = LogNorm(vmin=0.04, vmax=5.0)
        im = ax.imshow(latency_grid, aspect="auto", cmap=LATENCY_CMAP, norm=norm)
        candidate_labels = [_format_candidate_rows(value) for value in candidates]
        ax.set_xticks(np.arange(len(candidates)), candidate_labels)
        ax.set_yticks(np.arange(len(ks)), [str(value) for value in ks])
        ax.set_ylabel("restore $K$", labelpad=1.0)
        ax.tick_params(length=0, pad=1.3)
        ax.set_xticks(np.arange(-0.5, len(candidates), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ks), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.85)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
            spine.set_color("#666666")
        cell_fontsize = 4.8 if len(candidates) >= 8 else 5.25
        for row_idx in range(latency_grid.shape[0]):
            for col_idx in range(latency_grid.shape[1]):
                value = float(latency_grid[row_idx, col_idx])
                rgba = LATENCY_CMAP(norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(
                    col_idx,
                    row_idx,
                    _format_runtime_cell(value),
                    ha="center",
                    va="center",
                    fontsize=cell_fontsize,
                    color="white" if luminance < 0.50 else PALETTE["text"],
                )
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
        cbar.set_ticks([0.05, 0.1, 0.5, 1.0, 2.0, 5.0])
        cbar.ax.set_yticklabels(["50ms", "0.1s", "0.5s", "1s", "2s", "5s"])
        cbar.outline.set_linewidth(0.45)
        cbar.ax.tick_params(width=0.45, length=2.0, pad=1.0, labelsize=5.6)
        cbar.set_label("p95 repair", labelpad=1.2, fontsize=6.1)

        ax2 = axes[1]
        component_df = component_df.sort_values("candidate_tokens")
        x_values = np.arange(len(component_df))
        labels = [_format_candidate_rows(int(value)) for value in component_df["candidate_tokens"]]
        scan = component_df["scan_share"].to_numpy(dtype=float)
        topk = component_df["topk_share"].to_numpy(dtype=float)
        copy_insert = component_df["copy_insert_share"].to_numpy(dtype=float)
        ax2.bar(x_values, scan, width=0.72, color=PALETTE["idlekv"], label="scan")
        ax2.bar(
            x_values,
            topk,
            bottom=scan,
            width=0.72,
            color=PALETTE["gold"],
            edgecolor="#6A5700",
            linewidth=0.25,
            label="top-$K$",
        )
        ax2.bar(
            x_values,
            copy_insert,
            bottom=scan + topk,
            width=0.72,
            color=PALETTE["random"],
            label="copy+insert",
        )
        ax2.set_ylim(0.0, 1.0)
        ax2.set_yticks([0.0, 0.5, 1.0], ["0", "50", "100"])
        ax2.set_xticks(x_values, labels)
        _format_axes(ax2, x_label="offloaded candidate rows", y_label="p95 share (%)")
        ax2.grid(axis="y", color=PALETTE["grid"], linewidth=0.42, alpha=0.82)
        ax2.legend(
            loc="upper center",
            bbox_to_anchor=(0.51, 1.18),
            ncol=3,
            frameon=False,
            handlelength=1.0,
            handletextpad=0.28,
            columnspacing=0.7,
            fontsize=5.7,
        )
        ax.text(0.0, 1.04, "(a)", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.9, fontweight="bold")
        ax2.text(0.0, 1.03, "(b)", transform=ax2.transAxes, ha="left", va="bottom", fontsize=5.9, fontweight="bold")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(COLUMN_WIDTH_IN, 1.84), constrained_layout=False)
        fig.subplots_adjust(left=0.13, right=0.985, top=0.93, bottom=0.275)
        styles = {
            96: {"label": "$K=96$", "color": PALETTE["idlekv"], "marker": "o"},
            5000: {"label": "$K=5000$", "color": PALETTE["proxy"], "marker": "s"},
        }
        for k_value, style in styles.items():
            selected = repair_df[repair_df["k"].astype(int) == int(k_value)].sort_values("candidate_tokens")
            if selected.empty:
                continue
            ax.plot(
                selected["candidate_tokens"].astype(float),
                selected["repair_s"].astype(float),
                label=str(style["label"]),
                color=str(style["color"]),
                marker=str(style["marker"]),
                linewidth=1.35,
                markersize=3.2,
            )
        budget_lines = [
            (0.1, "#BDBDBD", "0.1s"),
            (0.5, PALETTE["wrong"], "0.5s"),
            (1.0, PALETTE["matched"], "1s"),
            (2.0, PALETTE["grid"], "2s"),
        ]
        for y_value, color, label in budget_lines:
            ax.axhline(y_value, color=color, linestyle=(0, (2, 1.5)), linewidth=0.68)
            ax.text(
                1_170_000,
                y_value * 1.035,
                label,
                ha="right",
                va="bottom",
                fontsize=5.2,
                color=color,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(28_000, 1_250_000)
        ax.set_ylim(0.025, 2.65)
        ax.set_xticks([32_768, 65_536, 131_072, 262_144, 524_288, 1_048_576])
        ax.set_xticklabels(["32K", "64K", "128K", "256K", "512K", "1M"])
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.set_yticks([0.05, 0.1, 0.5, 1.0, 2.0], ["0.05", "0.1", "0.5", "1", "2"])
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
        _format_axes(ax, x_label="offloaded candidate rows", y_label="p95 repair (s)")
        ax.legend(
            loc="lower right",
            frameon=False,
            ncol=1,
            handlelength=1.0,
            borderaxespad=0.1,
            handletextpad=0.25,
            fontsize=5.9,
        )

    save_figure(fig, "runtime_repair_scaling")
    return True


def _split_label(task: str) -> str:
    prefix = "mq_niah_"
    if prefix not in task or "_split_" not in task:
        return task
    variant = task.split("_split_", maxsplit=1)[0].replace(prefix, "").upper()
    split = task.split("_split_", maxsplit=1)[1].replace("_to_", r"$\to$")
    return f"{variant} {split}"


def render_partition_endpoint_dotplot() -> bool:
    """Render per-partition endpoint robustness without a wide appendix table."""

    paths = [
        FIGURE_DIR / "phase7_clean_suite_b16384_exact_by_split.csv",
        FIGURE_DIR / "phase7_mq_niah_6q_clean_suite_b18432_exact_by_split.csv",
        FIGURE_DIR / "phase10_mq_niah_8q_frontier_n24_by_split.csv",
    ]
    if not all(path.exists() for path in paths):
        for ext in ("pdf", "png"):
            (FIGURE_DIR / f"partition_endpoint_dotplot.{ext}").unlink(missing_ok=True)
        return False

    records = []
    for path in paths:
        rows = load_numeric_csv(path)
        rows = rows[rows["k"] == 128].copy()
        for row in rows.itertuples(index=False):
            records.append(
                {
                    "task": str(row.task),
                    "label": _split_label(str(row.task)),
                    "full": float(row.condition_a),
                    "matched": float(row.b_match),
                    "idlekv": float(row.idlekv),
                    "gold": float(row.oracle_k),
                }
            )
    if not records:
        return False

    order = [
        "mq_niah_4q_split_14_to_23",
        "mq_niah_4q_split_24_to_13",
        "mq_niah_4q_split_34_to_12",
        "mq_niah_6q_split_156_to_234",
        "mq_niah_6q_split_256_to_134",
        "mq_niah_6q_split_356_to_124",
        "mq_niah_6q_split_456_to_123",
        "mq_niah_8q_split_5678_to_1234",
        "mq_niah_8q_split_1678_to_2345",
        "mq_niah_8q_split_2678_to_1345",
        "mq_niah_8q_split_3678_to_1245",
        "mq_niah_8q_split_4678_to_1235",
    ]
    by_task = {record["task"]: record for record in records}
    ordered = [by_task[task] for task in order if task in by_task]
    if not ordered:
        return False

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.35), constrained_layout=False)
    fig.subplots_adjust(left=0.31, right=0.955, top=0.86, bottom=0.14)
    y_positions = np.arange(len(ordered))[::-1]

    for y, record in zip(y_positions, ordered, strict=True):
        ax.hlines(
            y,
            float(record["matched"]),
            float(record["idlekv"]),
            color="#B8B8B8",
            linewidth=0.8,
            zorder=1,
        )

    series = [
        ("Matched", "matched", PALETTE["matched"], "D", 13),
        ("IdleKV", "idlekv", PALETTE["idlekv"], "o", 18),
        ("Gold-K", "gold", PALETTE["gold"], "s", 15),
    ]
    for label, key, color, marker, size in series:
        ax.scatter(
            [float(record[key]) for record in ordered],
            y_positions,
            label=label,
            color=color,
            marker=marker,
            s=size,
            linewidth=0.35,
            edgecolor="white" if marker != "D" else color,
            zorder=3,
        )

    for y, record in zip(y_positions, ordered, strict=True):
        ax.text(
            1.012,
            y,
            f"{float(record['idlekv']) - float(record['matched']):+.2f}",
            ha="left",
            va="center",
            fontsize=5.8,
            color=PALETTE["idlekv"],
        )

    ax.set_yticks(y_positions, [str(record["label"]) for record in ordered])
    ax.set_xlim(0.0, 1.13)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xlabel("exact $Q_2$ score at $K=128$", labelpad=1.0)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.42, alpha=0.82)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=5.75, pad=1.2)
    ax.tick_params(axis="x", pad=1.2)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, 1.05),
        ncol=3,
        frameon=False,
        columnspacing=0.75,
        handlelength=1.0,
        handletextpad=0.3,
        borderaxespad=0.0,
    )
    save_figure(fig, "partition_endpoint_dotplot")
    return True


def render_multiturn_hard_trajectory() -> bool:
    """Render the locked hard multi-turn trajectory when it exists."""

    paper_rows = FIGURE_DIR / "multiturn_hard_locked_rows_n24_k80.csv"
    phase13_rows = sorted(
        PHASE13_DIR.glob("multiturn_hard_locked_rows_n*_k*.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if paper_rows.exists():
        rows_path = paper_rows
        raw_path = FIGURE_DIR / "multiturn_hard_locked_metadata.json"
        locked_followup = True
    elif phase13_rows:
        rows_path = phase13_rows[-1]
        raw_name = rows_path.name.replace("multiturn_hard_locked_rows_", "multiturn_hard_locked_").replace(".csv", "_raw.json")
        raw_path = rows_path.with_name(raw_name)
        locked_followup = True
    else:
        rows_path = PHASE10_DIR / "multiturn_hard_locked_rows_n12.csv"
        raw_path = PHASE10_DIR / "multiturn_hard_locked_n12_raw.json"
        locked_followup = False
    if not rows_path.exists():
        for ext in ("pdf", "png"):
            (FIGURE_DIR / f"multiturn_hard_trajectory.{ext}").unlink(missing_ok=True)
        return False

    rows = load_numeric_csv(rows_path)
    if rows.empty:
        return False
    required = {"example_index", "turn", "k", "condition", "score"}
    if not required.issubset(rows.columns):
        return False

    matched = rows[rows["condition"].astype(str) == "Matched"][
        ["example_index", "turn", "k", "score"]
    ].rename(columns={"score": "matched_score"})
    merged = rows.merge(matched, on=["example_index", "turn", "k"], how="inner")
    merged["gain"] = merged["score"] - merged["matched_score"]

    interval_records = []
    for (k, turn, condition), group in merged.groupby(["k", "turn", "condition"], sort=True):
        gains = group["gain"].to_numpy(dtype=float)
        scores = group["score"].to_numpy(dtype=float)
        gain_mean = float(np.mean(gains)) if gains.size else 0.0
        score_mean = float(np.mean(scores)) if scores.size else 0.0
        if scores.size <= 1:
            score_lo = score_hi = score_mean
            gain_lo = gain_hi = gain_mean
        else:
            seed = int(k) * 1009 + int(turn) * 917 + sum(ord(char) for char in str(condition))
            rng = np.random.default_rng(seed)
            sample_indices = rng.integers(0, scores.size, size=(500, scores.size))
            score_boot_means = scores[sample_indices].mean(axis=1)
            gain_boot_means = gains[sample_indices].mean(axis=1)
            score_lo, score_hi = np.quantile(score_boot_means, [0.025, 0.975])
            gain_lo, gain_hi = np.quantile(gain_boot_means, [0.025, 0.975])
        interval_records.append(
            {
                "k": int(k),
                "turn": int(turn),
                "condition": str(condition),
                "score": score_mean,
                "score_lo": float(score_lo),
                "score_hi": float(score_hi),
                "gain": gain_mean,
                "gain_lo": float(gain_lo),
                "gain_hi": float(gain_hi),
            }
        )
    mean_gain = pd.DataFrame.from_records(interval_records).sort_values(["k", "turn", "condition"])
    ks = sorted(int(value) for value in mean_gain["k"].dropna().unique())
    if not ks:
        return False
    if locked_followup:
        ks = [80] if 80 in ks else ks[:1]
    else:
        ks = [k for k in (48, 96) if k in ks] or ks[:2]
    if not locked_followup and raw_path.exists():
        with raw_path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        candidate_ks = {
            int(row["k"])
            for row in raw.get("recommendations", [])
            if row.get("action") == "main_candidate_if_artifact_checks_pass"
        }
        ks = [k for k in ks if k in candidate_ks]
        if not ks:
            for ext in ("pdf", "png"):
                (FIGURE_DIR / f"multiturn_hard_trajectory.{ext}").unlink(missing_ok=True)
            return False
    turns = sorted(int(value) for value in mean_gain["turn"].dropna().unique())
    revisit_turns: set[int] = set()
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        for event in raw.get("schedule", {}).get("revisit_events", []):
            if "revisit_turn" in event:
                revisit_turns.add(int(event["revisit_turn"]) + 1)

    display_turns = [turn for turn in turns if turn > 0] or turns
    if len(ks) > 1:
        display_turns = turns

    fig_height = 1.72 if len(ks) == 1 else 2.20
    fig, axes_raw = plt.subplots(
        len(ks),
        1,
        figsize=(COLUMN_WIDTH_IN, fig_height),
        sharex=True,
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes_raw)
    fig.subplots_adjust(left=0.18, right=0.985, top=0.82, bottom=0.24, hspace=0.20)

    matched_fill = "#D7DCE2"
    gold_fill = "#F0D28D"
    control_color = "#8A8A8A"
    observed_conditions = {str(condition) for condition in mean_gain["condition"].dropna().unique()}
    for ax, k in zip(axes, ks, strict=True):
        subset = mean_gain[mean_gain["k"] == k]
        x_positions = np.arange(len(display_turns), dtype=float)
        tick_labels = []
        bar_width = 0.13
        offsets = {"Rand/old": -0.34, "Matched": -0.17, "Stale-Q": 0.0, "IdleKV": 0.17, "Gold-K": 0.34}
        for x, turn in zip(x_positions, display_turns, strict=True):
            matched_series = subset[
                (subset["turn"] == turn)
                & (subset["condition"].astype(str) == "Matched")
            ]
            idle_series = subset[
                (subset["turn"] == turn)
                & (subset["condition"].astype(str) == "IdleKV")
            ]
            gold_series = subset[
                (subset["turn"] == turn)
                & (subset["condition"].astype(str) == "Gold-K")
            ]
            if matched_series.empty or idle_series.empty:
                continue

            matched_score = float(matched_series.iloc[0]["score"])
            idle = idle_series.iloc[0]
            idle_score = float(idle["score"])
            idle_lo = float(idle["score_lo"])
            idle_hi = float(idle["score_hi"])
            gold_score = (
                float(gold_series.iloc[0]["score"])
                if not gold_series.empty
                else max(matched_score, idle_score)
            )

            control_values = subset[
                (subset["turn"] == turn)
                & (subset["condition"].astype(str).isin(["Random-K", "Oldest-K"]))
            ]["score"].to_numpy(dtype=float)
            if control_values.size:
                control_lo = float(np.min(control_values))
                control_hi = float(np.max(control_values))
                control_mid = 0.5 * (control_lo + control_hi)
                ax.bar(
                    x + offsets["Rand/old"],
                    control_mid,
                    width=bar_width,
                    color="#B6B6B6",
                    edgecolor="#7D7D7D",
                    linewidth=0.35,
                    alpha=0.88,
                    zorder=2,
                )
                ax.errorbar(
                    [x + offsets["Rand/old"]],
                    [control_mid],
                    yerr=[[max(control_mid - control_lo, 0.0)], [max(control_hi - control_mid, 0.0)]],
                    fmt="none",
                    color="#7D7D7D",
                    ecolor="#7D7D7D",
                    elinewidth=0.62,
                    capsize=1.0,
                    zorder=4,
                )
            ax.bar(x + offsets["Matched"], matched_score, width=bar_width, color=matched_fill, edgecolor="none", zorder=2)
            stale_series = subset[
                (subset["turn"] == turn)
                & (subset["condition"].astype(str) == "StaleQ-K")
            ]
            if not stale_series.empty:
                stale_score = float(stale_series.iloc[0]["score"])
                ax.bar(
                    x + offsets["Stale-Q"],
                    stale_score,
                    width=bar_width,
                    color=PALETTE["refresh"],
                    edgecolor="none",
                    alpha=0.82,
                    zorder=3,
                )
            ax.bar(
                x + offsets["IdleKV"],
                idle_score,
                width=bar_width,
                color=PALETTE["idlekv"],
                alpha=0.90,
                edgecolor="none",
                zorder=3,
            )
            ax.bar(
                x + offsets["Gold-K"],
                gold_score,
                width=bar_width,
                color=gold_fill,
                edgecolor=PALETTE["gold"],
                linewidth=0.45,
                alpha=0.78,
                zorder=2,
            )
            ax.errorbar(
                [x + offsets["IdleKV"]],
                [idle_score],
                yerr=[[max(0.0, idle_score - idle_lo)], [max(0.0, idle_hi - idle_score)]],
                fmt="none",
                color=PALETTE["idlekv"],
                ecolor=PALETTE["idlekv"],
                elinewidth=0.78,
                capsize=1.4,
                zorder=4,
            )

            displayed = int(turn) + 1
            if displayed in revisit_turns:
                label = f"T{displayed}\nrevisit"
            elif gold_score - matched_score < 0.03:
                label = f"T{displayed}\nno gap"
            else:
                label = f"T{displayed}\nshift"
            tick_labels.append(label)

        if len(ks) > 1:
            ax.text(
                0.02,
                0.92,
                f"$K={k}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.6,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.7},
            )
        ax.set_xticks(x_positions, tick_labels)
        ax.set_xlim(-0.58, len(display_turns) - 0.42)
        ax.set_ylim(0.0, 1.08)
        ax.set_yticks([0.0, 0.5, 1.0])
        _format_axes(ax, x_label=None, y_label="exact score" if ax is axes[0] else None)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.45, alpha=0.78)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=5.25, pad=1.2)

    handles = [
        Patch(facecolor="#B6B6B6", alpha=0.88, edgecolor="#7D7D7D", label="Rand/old"),
        Patch(facecolor=matched_fill, edgecolor="none", label="matched"),
        Patch(facecolor=PALETTE["refresh"], alpha=0.82, edgecolor="none", label="Stale-Q"),
        Patch(facecolor=PALETTE["idlekv"], alpha=0.90, edgecolor="none", label="IdleKV"),
        Patch(facecolor=gold_fill, alpha=0.78, edgecolor=PALETTE["gold"], label="Gold-K"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.985),
        ncol=5,
        frameon=False,
        columnspacing=0.34,
        handlelength=0.54,
        handletextpad=0.18,
        borderaxespad=0.0,
        fontsize=4.95,
    )
    save_figure(fig, "multiturn_hard_trajectory")
    return True


def _plot_control_band_frontier(ax: plt.Axes, rows: pd.DataFrame, *, label: str | None = None) -> None:
    """Plot a compact score frontier with content-agnostic controls as a band."""

    rows = rows.sort_values("k").copy()
    xs = rows["k"].to_numpy(dtype=float)
    control_cols = [col for col in ("b_match", "random_k", "oldest_k") if col in rows.columns]
    controls = rows[control_cols].to_numpy(dtype=float)
    control_min = np.nanmin(controls, axis=1)
    control_max = np.nanmax(controls, axis=1)

    ax.fill_between(xs, control_min, control_max, color="#C9C9C9", alpha=0.32, linewidth=0.0, zorder=1)
    ax.plot(
        xs,
        rows["b_match"],
        color=PALETTE["matched"],
        linestyle=(0, (1.0, 1.3)),
        linewidth=0.95,
        zorder=2,
    )
    ax.plot(
        xs,
        rows["idlekv"],
        color=PALETTE["idlekv"],
        marker="o",
        markersize=3.0,
        markeredgecolor="white",
        markeredgewidth=0.35,
        linewidth=1.45,
        zorder=4,
    )
    ax.plot(
        xs,
        rows["gold_k"],
        color=PALETTE["gold"],
        marker="s",
        markersize=2.8,
        markeredgecolor="white",
        markeredgewidth=0.35,
        linestyle="--",
        linewidth=1.10,
        zorder=3,
    )
    if label:
        ax.text(
            0.02,
            0.91,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.3,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.5},
        )


def _format_budget_label(row: pd.Series, *, include_budget: bool = False) -> str:
    k = int(row["k"])
    if include_budget and "base_context_budget" in row and not pd.isna(row["base_context_budget"]):
        return f"$B={float(row['base_context_budget']) / 1000:.1f}$k, $K={k}$"
    return f"$K={k}$"


def _render_sparse_robustness_rows(
    rows: pd.DataFrame,
    *,
    stem: str,
    include_budget: bool = False,
    height: float | None = None,
) -> bool:
    """Render sparse robustness data as score intervals instead of a fake frontier."""

    if rows.empty:
        return False
    sort_cols = ["k"]
    if include_budget and "base_context_budget" in rows:
        sort_cols = ["base_context_budget", "k"]
    rows = rows.sort_values(sort_cols).reset_index(drop=True)

    row_height = 0.28
    fig_height = height or max(1.35, 0.68 + row_height * len(rows))
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, fig_height), constrained_layout=False)
    fig.subplots_adjust(left=0.30 if include_budget else 0.18, right=0.965, top=0.84, bottom=0.22)

    y_positions = np.arange(len(rows))[::-1]
    control_cols = [col for col in ("b_match", "random_k", "oldest_k") if col in rows.columns]
    labels: list[str] = []
    for y, row in zip(y_positions, rows.itertuples(index=False), strict=True):
        series = pd.Series(row._asdict())
        controls = np.array([float(series[col]) for col in control_cols], dtype=float)
        control_min = float(np.nanmin(controls))
        control_max = float(np.nanmax(controls))
        if control_max - control_min < 0.012:
            midpoint = 0.5 * (control_min + control_max)
            control_min = max(0.0, midpoint - 0.012)
            control_max = min(1.0, midpoint + 0.012)
        matched = float(series["b_match"])
        idle = float(series["idlekv"])
        gold = float(series["gold_k"])
        ax.hlines(y, control_min, control_max, color="#BDBDBD", linewidth=3.0, alpha=0.62, zorder=1)
        ax.scatter([matched], [y], marker="D", s=15, color=PALETTE["matched"], linewidth=0.0, zorder=3)
        ax.scatter(
            [idle],
            [y],
            marker="o",
            s=22,
            color=PALETTE["idlekv"],
            edgecolor="white",
            linewidth=0.35,
            zorder=4,
        )
        ax.scatter(
            [gold],
            [y],
            marker="s",
            s=25,
            facecolor="none",
            edgecolor=PALETTE["gold"],
            linewidth=0.95,
            zorder=5,
        )
        ax.text(
            1.012,
            y,
            f"+{idle - matched:.2f}",
            color=PALETTE["idlekv"],
            ha="left",
            va="center",
            fontsize=5.8,
            clip_on=False,
        )
        labels.append(_format_budget_label(series, include_budget=include_budget))

    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_yticks(y_positions, labels)
    ax.set_xticks([0.0, 0.5, 1.0])
    _format_axes(ax, x_label="exact $Q_2$ score", y_label=None)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.45, alpha=0.78)
    ax.grid(axis="y", visible=False)
    handles = [
        Line2D([0], [0], marker="D", linestyle="None", color=PALETTE["matched"], label="Matched", markersize=3.6),
        Patch(facecolor="#BDBDBD", alpha=0.62, edgecolor="none", label="Random/oldest"),
        Line2D([0], [0], marker="o", linestyle="None", color=PALETTE["idlekv"], label="IdleKV", markersize=4.0),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor=PALETTE["gold"],
            color=PALETTE["gold"],
            label="Gold-K",
            markersize=3.8,
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.985),
        ncol=4,
        frameon=False,
        columnspacing=0.48,
        handlelength=0.7,
        handletextpad=0.25,
        borderaxespad=0.0,
        fontsize=5.65,
    )
    save_figure(fig, stem)
    return True


def _format_frontier_xaxis(ax: plt.Axes, rows: pd.DataFrame) -> None:
    xs = sorted(int(k) for k in rows["k"].dropna().unique())
    if len(xs) <= 3:
        ax.set_xlim(min(xs) - 8, max(xs) + 8)
        ax.set_xticks(xs)
    else:
        ax.set_xlim(6, max(136, max(xs) + 6))
        ticks = [tick for tick in [8, 32, 64, 96, 128] if min(xs) <= tick <= max(xs)]
        if min(xs) not in ticks:
            ticks.insert(0, min(xs))
        if max(xs) not in ticks:
            ticks.append(max(xs))
        ax.set_xticks(sorted(set(ticks)))
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks([0.0, 0.5, 1.0])
    _format_axes(ax, x_label="restore budget $K$", y_label="exact $Q_2$ score")


def _robustness_handles() -> list[object]:
    return [
        Line2D([0], [0], color=PALETTE["idlekv"], marker="o", linewidth=1.45, markersize=3.0, label="IdleKV"),
        Line2D([0], [0], color=PALETTE["gold"], marker="s", linestyle="--", linewidth=1.10, markersize=2.8, label="Gold-K"),
        Line2D([0], [0], color=PALETTE["matched"], linestyle=(0, (1.0, 1.3)), linewidth=0.95, label="Matched"),
        Patch(facecolor="#C9C9C9", alpha=0.32, edgecolor="none", label="control band"),
    ]


def render_h2o_compressor_breadth() -> bool:
    """Render the H2O-inspired accumulated-attention retention check."""

    candidates = [
        FIGURE_DIR / "h2o_4q_fullgrid_n24.csv",
        PHASE11_DIR / "h2o_4q_fullgrid_n24.csv",
        PHASE10_DIR / "h2o_compressor_locked_n12.csv",
    ]
    csv_path = next((path for path in candidates if path.exists()), None)
    if csv_path is None:
        for ext in ("pdf", "png"):
            (FIGURE_DIR / f"h2o_compressor_breadth.{ext}").unlink(missing_ok=True)
        return False

    rows = load_numeric_csv(csv_path)
    required = {"k", "b_match", "idlekv", "random_k", "oldest_k", "gold_k"}
    if rows.empty or not required.issubset(rows.columns):
        return False
    rows = rows.sort_values("k")
    if rows["k"].nunique() <= 3:
        return _render_sparse_robustness_rows(rows, stem="h2o_compressor_breadth")

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 1.58), constrained_layout=False)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.78, bottom=0.25)
    _plot_control_band_frontier(ax, rows)
    _format_frontier_xaxis(ax, rows)

    fig.legend(
        handles=_robustness_handles(),
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=4,
        frameon=False,
        columnspacing=0.58,
        handlelength=1.12,
        handletextpad=0.28,
        borderaxespad=0.0,
        fontsize=5.75,
    )
    save_figure(fig, "h2o_compressor_breadth")
    return True


def render_model_transfer_breadth() -> bool:
    """Render the strongest locked model-transfer robustness check."""

    candidates = [
        FIGURE_DIR / "llama31_8b_6q_locked_n12_b18432_k64-96-128.csv",
        PHASE13_DIR / "llama31_8b_6q_locked_n12_b18432_k64-96-128.csv",
        PHASE11_DIR / "llama31_8b_4q_fullgrid_n24.csv",
        PHASE10_DIR / "model_transfer_llama_3_1_8b_instruct__locked_n12.csv",
        PHASE10_DIR / "model_transfer_qwen2_5_3b_instruct__locked_n12.csv",
    ]
    csv_path = next((path for path in candidates if path.exists()), None)
    if csv_path is None:
        for ext in ("pdf", "png"):
            (FIGURE_DIR / f"model_transfer_breadth.{ext}").unlink(missing_ok=True)
        return False

    rows = load_numeric_csv(csv_path)
    required = {"base_context_budget", "k", "b_match", "idlekv", "random_k", "oldest_k", "gold_k"}
    if rows.empty or not required.issubset(rows.columns):
        return False

    budgets = sorted(int(budget) for budget in rows["base_context_budget"].dropna().unique())
    if not budgets:
        return False
    if rows["k"].nunique() <= 3:
        return _render_sparse_robustness_rows(
            rows,
            stem="model_transfer_breadth",
            include_budget=len(budgets) > 1,
        )

    if len(budgets) == 1:
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 1.58), constrained_layout=False)
        fig.subplots_adjust(left=0.15, right=0.985, top=0.78, bottom=0.25)
        panel_rows = rows[rows["base_context_budget"] == budgets[0]].sort_values("k")
        _plot_control_band_frontier(ax, panel_rows)
        _format_frontier_xaxis(ax, panel_rows)
    else:
        fig, axes = plt.subplots(1, len(budgets), figsize=(COLUMN_WIDTH_IN, 1.48), sharey=True, constrained_layout=False)
        axes = np.atleast_1d(axes)
        fig.subplots_adjust(left=0.15, right=0.985, top=0.76, bottom=0.28, wspace=0.18)
        for ax, budget in zip(axes, budgets, strict=True):
            panel_rows = rows[rows["base_context_budget"] == budget].sort_values("k")
            _plot_control_band_frontier(ax, panel_rows, label=f"$B={budget / 1000:.1f}$k")
            _format_frontier_xaxis(ax, panel_rows)
            if ax is not axes[0]:
                ax.set_ylabel("")
        axes[-1].set_xlabel("restore budget $K$", labelpad=1.0)

    fig.legend(
        handles=_robustness_handles(),
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=4,
        frameon=False,
        columnspacing=0.58,
        handlelength=1.12,
        handletextpad=0.28,
        borderaxespad=0.0,
        fontsize=5.75,
    )
    save_figure(fig, "model_transfer_breadth")
    return True


def _load_policy_breadth_rows(path: Path, *, name: str) -> pd.DataFrame:
    rows = load_numeric_csv(path).rename(columns={"oracle_k": "gold_k", "goldk_gain": "gold_gain"})
    if {"k", "idlekv_gain"}.issubset(rows.columns):
        out = pd.DataFrame(
            {
                "k": rows["k"],
                "idlekv_gain": rows["idlekv_gain"],
                "gold_gain": rows.get("gold_gain", rows["idlekv_gain"]),
                "random_gain": 0.0,
                "oldest_gain": 0.0,
                "policy": name,
            }
        )
        return out

    required = {"k", "b_match", "idlekv", "random_k", "oldest_k", "gold_k"}
    if not required.issubset(rows.columns):
        raise ValueError(f"{path} is missing policy-breadth columns.")
    matched = rows["b_match"].astype(float)
    return pd.DataFrame(
        {
            "k": rows["k"],
            "idlekv_gain": rows["idlekv"].astype(float) - matched,
            "gold_gain": rows["gold_k"].astype(float) - matched,
            "random_gain": rows["random_k"].astype(float) - matched,
            "oldest_gain": rows["oldest_k"].astype(float) - matched,
            "policy": name,
        }
    )


def render_policy_breadth_delta() -> bool:
    """Render a main-candidate first-stage-policy breadth figure if all data exists."""

    streaming_candidates = sorted(FIGURE_DIR.glob("streamingllm_4q_fullgrid_n24_b*.csv"))
    if not streaming_candidates:
        streaming_candidates = sorted(PHASE12_DIR.glob("streamingllm_4q_fullgrid_n24_b*.csv"))
    required_paths = {
        "SnapKV": FIGURE_DIR / "phase7_clean_suite_b16384_exact_overall.csv",
        "Accumulated attention": (
            FIGURE_DIR / "h2o_4q_fullgrid_n24.csv"
            if (FIGURE_DIR / "h2o_4q_fullgrid_n24.csv").exists()
            else PHASE11_DIR / "h2o_4q_fullgrid_n24.csv"
        ),
    }
    if not streaming_candidates or not all(path.exists() for path in required_paths.values()):
        for ext in ("pdf", "png"):
            (FIGURE_DIR / f"policy_breadth_delta.{ext}").unlink(missing_ok=True)
        return False

    frames = [_load_policy_breadth_rows(path, name=name) for name, path in required_paths.items()]
    frames.append(_load_policy_breadth_rows(streaming_candidates[-1], name="Sink+recent"))
    rows = pd.concat(frames, ignore_index=True)
    rows = rows[rows["k"].isin([8, 16, 24, 32, 48, 64, 80, 96, 128])].copy()
    if rows.empty or rows["policy"].nunique() < 3:
        return False

    policies = ["SnapKV", "Accumulated attention", "Sink+recent"]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 1.86), constrained_layout=False)
    fig.subplots_adjust(left=0.17, right=0.88, top=0.90, bottom=0.22)
    grouped = rows.groupby("k", sort=True)
    k_values = np.asarray(sorted(rows["k"].unique()), dtype=float)
    control_lo = grouped[["random_gain", "oldest_gain"]].min().min(axis=1).reindex(k_values).to_numpy(dtype=float)
    control_hi = grouped[["random_gain", "oldest_gain"]].max().max(axis=1).reindex(k_values).to_numpy(dtype=float)
    gold_lo = grouped["gold_gain"].min().reindex(k_values).to_numpy(dtype=float)
    gold_hi = grouped["gold_gain"].max().reindex(k_values).to_numpy(dtype=float)

    ax.fill_between(
        k_values,
        gold_lo,
        gold_hi,
        color=PALETTE["gold"],
        alpha=0.14,
        linewidth=0.0,
        zorder=1,
    )
    ax.plot(k_values, gold_lo, color=PALETTE["gold"], linewidth=0.38, alpha=0.35, zorder=1)
    ax.plot(k_values, gold_hi, color=PALETTE["gold"], linewidth=0.38, alpha=0.35, zorder=1)
    ax.fill_between(
        k_values,
        control_lo,
        control_hi,
        color="#A9A9A9",
        alpha=0.24,
        linewidth=0.0,
        zorder=2,
    )
    ax.plot(k_values, control_lo, color="#7D7D7D", linewidth=0.38, alpha=0.55, zorder=2)
    ax.plot(k_values, control_hi, color="#7D7D7D", linewidth=0.38, alpha=0.55, zorder=2)
    ax.axhline(0.0, color=PALETTE["matched"], linewidth=0.70, linestyle=(0, (1.0, 1.4)), zorder=3)

    styles = {
        "SnapKV": {"color": PALETTE["idlekv"], "marker": "o", "label": "SnapKV"},
        "Accumulated attention": {"color": PALETTE["oldest"], "marker": "s", "label": "Accum.-attn"},
        "Sink+recent": {"color": PALETTE["proxy"], "marker": "^", "label": "Sink+recent"},
    }
    label_offsets = {"SnapKV": 0.035, "Accumulated attention": -0.035, "Sink+recent": 0.0}
    for policy in policies:
        panel = rows[rows["policy"] == policy].sort_values("k").copy()
        if panel.empty:
            continue
        style = styles[policy]
        ax.plot(
            panel["k"],
            panel["idlekv_gain"],
            color=str(style["color"]),
            marker=str(style["marker"]),
            linewidth=1.42,
            markersize=2.9,
            label=str(style["label"]),
            zorder=4,
        )
        endpoint = panel.iloc[-1]
        ax.text(
            132.5,
            float(endpoint["idlekv_gain"]) + label_offsets[policy],
            f"{style['label']} +{float(endpoint['idlekv_gain']):.2f}",
            color=str(style["color"]),
            fontsize=6.0,
            va="center",
            ha="left",
            fontweight="bold",
            clip_on=False,
        )

    _format_axes(ax, x_label="restore budget $K$", y_label="score gain over matched")
    ax.set_xlim(6, 149)
    ax.set_ylim(-0.07, 0.90)
    ax.set_xticks([8, 32, 64, 96, 128])
    ax.set_yticks([0.0, 0.4, 0.8])
    ax.grid(axis="x", visible=False)
    band_handles = [
        Patch(facecolor=PALETTE["gold"], alpha=0.18, edgecolor=PALETTE["gold"], linewidth=0.35, label="Gold-K range"),
        Patch(facecolor="#A9A9A9", alpha=0.28, edgecolor="#7D7D7D", linewidth=0.35, label="Rand/old range"),
    ]
    ax.legend(
        handles=band_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.995),
        ncol=2,
        frameon=False,
        borderaxespad=0.0,
        handlelength=0.9,
        handletextpad=0.25,
        columnspacing=0.55,
        fontsize=5.3,
    )
    save_figure(fig, "policy_breadth_delta")
    return True


def generated_assets() -> Iterable[Path]:
    for stem in [
        "operating_regime_heatmap",
        "repair_selection_diagnostic",
        "frontier_raw_overlay",
        "specificity_control_dotplot",
        "query_count_breadth",
        "streaming_spill_heatmap",
        "proxy_latency_tradeoff",
        "runtime_capacity_curve",
        "runtime_repair_scaling",
        "partition_endpoint_dotplot",
        "multiturn_hard_trajectory",
        "h2o_compressor_breadth",
        "model_transfer_breadth",
        "policy_breadth_delta",
    ]:
        yield FIGURE_DIR / f"{stem}.pdf"
        yield FIGURE_DIR / f"{stem}.png"


def main() -> None:
    configure_matplotlib()
    render_operating_regime_heatmap()
    render_selection_diagnostic()
    render_frontier_raw_overlay()
    render_specificity_panel()
    render_query_count_breadth()
    render_streaming_spill_heatmap()
    render_proxy_latency_tradeoff()
    render_runtime_capacity_curve()
    render_runtime_repair_scaling()
    render_partition_endpoint_dotplot()
    render_multiturn_hard_trajectory()
    render_h2o_compressor_breadth()
    render_model_transfer_breadth()
    render_policy_breadth_delta()


if __name__ == "__main__":
    main()
