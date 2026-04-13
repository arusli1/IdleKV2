#!/usr/bin/env python3
"""
Generate paper figures from experiment results.

Figure 1: Quality-vs-idle-budget curve
Figure 2: Quality-vs-throughput Pareto frontier
Figure 3: Per-subtask heatmap (RULER)
Figure 4: Component ablation (Phase 1 vs Phase 2 vs both)
Figure 5: Shadow buffer size ablation

Usage:
    python scripts/plot_figures.py --results-dir results/ --output-dir figures/
"""

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "full_cache": "#2c3e50",
    "snapkv_0.3": "#3498db",
    "snapkv_0.5": "#e67e22",
    "idlekv_p1": "#27ae60",
    "idlekv_p1p2": "#e74c3c",
    "sync_refresh": "#9b59b6",
    "streaminglm": "#95a5a6",
    "h2o": "#1abc9c",
}


def load_results(results_dir: Path) -> dict:
    """Load all JSON result files from the results directory."""
    all_results = {}
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fp:
            all_results[f.stem] = json.load(fp)
    return all_results


def plot_quality_vs_budget(results: dict, output_dir: Path):
    """
    Figure 1: Quality-vs-idle-budget curve.

    X: idle compute (log scale, 0-5s)
    Y: RULER accuracy
    One curve per compression ratio, one panel per model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    budgets_ms = [0, 50, 100, 500, 1000, 2000, 5000]
    budgets_s = [b / 1000 for b in budgets_ms]

    for ax_idx, model_name in enumerate(["Llama-3.1-8B", "Qwen2.5-7B"]):
        ax = axes[ax_idx]

        # TODO: Extract actual data from results dict
        # Placeholder with expected shape
        for ratio, color, label in [
            (0.3, "#3498db", "r=0.3"),
            (0.5, "#e74c3c", "r=0.5"),
            (0.7, "#e67e22", "r=0.7"),
        ]:
            # Placeholder data — replace with actual results
            baseline = 90 - ratio * 15  # SnapKV baseline (y-intercept)
            ceiling = 96  # full cache
            recovery = np.array([0, 0.1, 0.2, 0.4, 0.6, 0.75, 0.85])
            scores = baseline + recovery * (ceiling - baseline)

            ax.semilogx(
                [max(b, 0.01) for b in budgets_s],
                scores,
                "-o", color=color, label=label, markersize=4, linewidth=1.5,
            )

        ax.axhline(y=96, color=COLORS["full_cache"], linestyle="--",
                    alpha=0.5, label="Full cache")
        ax.set_xlabel("Idle compute budget (seconds)")
        if ax_idx == 0:
            ax.set_ylabel("RULER accuracy (%)")
        ax.set_title(model_name)
        ax.legend(loc="lower right")
        ax.set_xlim(0.01, 6)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "fig1_quality_vs_budget.pdf")
    fig.savefig(output_dir / "fig1_quality_vs_budget.png")
    plt.close()
    print("Saved Figure 1: quality_vs_budget")


def plot_pareto_frontier(results: dict, output_dir: Path):
    """
    Figure 2: Quality-vs-throughput Pareto frontier.

    X: decode tokens/sec
    Y: RULER accuracy
    Points for each method.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # TODO: Replace with actual data
    methods = [
        ("Full cache",       30,  96, COLORS["full_cache"], "s"),
        ("SnapKV r=0.3",     38,  93, COLORS["snapkv_0.3"], "^"),
        ("SnapKV r=0.5",     45,  89, COLORS["snapkv_0.5"], "o"),
        ("SnapKV r=0.7",     52,  82, "#d35400", "v"),
        ("IdleKV r=0.5",     45,  92, COLORS["idlekv_p1p2"], "D"),
        ("Sync refresh r=0.5", 34, 92, COLORS["sync_refresh"], "P"),
        ("StreamingLLM",     48,  73, COLORS["streaminglm"], "X"),
    ]

    for name, tps, acc, color, marker in methods:
        ax.scatter(tps, acc, c=color, marker=marker, s=80, zorder=5,
                   edgecolors="white", linewidth=0.5)
        ax.annotate(name, (tps, acc), textcoords="offset points",
                    xytext=(8, -4), fontsize=7.5)

    # Draw Pareto frontier
    pareto_x = [30, 38, 45]  # full, snap0.3, idlekv
    pareto_y = [96, 93, 92]
    ax.plot(pareto_x, pareto_y, "--", color="#bdc3c7", alpha=0.7, zorder=1)

    ax.set_xlabel("Decode throughput (tokens/sec)")
    ax.set_ylabel("RULER 4K accuracy (%)")
    ax.set_title("Quality-Throughput Pareto Frontier")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "fig2_pareto.pdf")
    fig.savefig(output_dir / "fig2_pareto.png")
    plt.close()
    print("Saved Figure 2: pareto")


def plot_component_ablation(results: dict, output_dir: Path):
    """Figure 4: Phase 1 vs Phase 2 vs both."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # TODO: Replace with actual data
    methods = ["SnapKV\nr=0.5", "Phase 1\nonly", "Phase 2\nonly", "Phase\n1+2", "Full\ncache"]
    scores = [89, 91, 92, 93, 96]  # placeholder
    colors = [COLORS["snapkv_0.5"], COLORS["idlekv_p1"], "#2ecc71",
              COLORS["idlekv_p1p2"], COLORS["full_cache"]]

    bars = ax.bar(methods, scores, color=colors, edgecolor="white", width=0.6)
    ax.set_ylabel("RULER 4K accuracy (%)")
    ax.set_title("Component Ablation (r=0.5, 1s idle budget)")
    ax.set_ylim(85, 97)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{score}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "fig4_ablation.pdf")
    fig.savefig(output_dir / "fig4_ablation.png")
    plt.close()
    print("Saved Figure 4: component ablation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument("--output-dir", type=str, default="figures/")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir) if results_dir.exists() else {}

    plot_quality_vs_budget(results, output_dir)
    plot_pareto_frontier(results, output_dir)
    plot_component_ablation(results, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
