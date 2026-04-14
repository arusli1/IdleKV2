#!/usr/bin/env python3
"""
Generate paper figures from real experiment results.

The script reads experiment JSON files recursively from `results/` and only
plots figures that have sufficient measured data. Missing figures are skipped
with an explicit message instead of using placeholders.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

RESULT_SUBDIRS = {"baselines", "idlekv", "ablations"}
MODEL_LABELS = {
    "llama8b": "Llama-3.1-8B",
    "qwen7b": "Qwen2.5-7B",
}
PHASE_LABELS = {
    "1": "Phase 1",
    "2": "Phase 2",
    "1+2": "Phase 1+2",
}
COLORS = {
    "full_cache": "#2c3e50",
    "snapkv_0.3": "#3498db",
    "snapkv_0.5": "#e67e22",
    "snapkv_0.7": "#d35400",
    "streaminglm": "#95a5a6",
    "h2o_0.5": "#1abc9c",
    "idlekv_phase_1": "#27ae60",
    "idlekv_phase_2": "#16a085",
    "idlekv_phase_1+2": "#e74c3c",
    "buffer": "#8e44ad",
}


def iter_result_paths(results_dir: Path):
    """Yield experiment result files under known result subdirectories."""
    for path in sorted(results_dir.rglob("*.json")):
        if path.name.startswith("run_"):
            continue
        if not RESULT_SUBDIRS.intersection(path.parts):
            continue
        yield path


def load_results(results_dir: Path) -> list[dict]:
    """Load all experiment result JSONs recursively."""
    loaded = []
    for path in iter_result_paths(results_dir):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if "experiment" not in payload or "benchmark_results" not in payload:
            continue
        payload["_path"] = str(path)
        loaded.append(payload)
    return loaded


def display_model(model_short: str) -> str:
    return MODEL_LABELS.get(model_short, model_short)


def metric_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else np.nan


def extract_ruler_avg(result: dict, context_length: str = "4096") -> float | None:
    ruler = result.get("benchmark_results", {}).get("ruler", {})
    if context_length not in ruler:
        return None
    avg = ruler[context_length].get("avg_accuracy")
    if avg is None:
        return None
    return float(avg) * 100.0


def extract_ruler_subtasks(result: dict, context_length: str = "4096") -> dict[str, float]:
    ruler = result.get("benchmark_results", {}).get("ruler", {})
    bucket = ruler.get(context_length, {})
    return {
        name: float(entry["accuracy"]) * 100.0
        for name, entry in bucket.items()
        if isinstance(entry, dict) and "accuracy" in entry
    }


def extract_longbench_avg(result: dict) -> float | None:
    longbench = result.get("benchmark_results", {}).get("longbench")
    if not longbench:
        return None
    avg = longbench.get("avg_score")
    return float(avg) if avg is not None else None


def method_sort_key(method: str) -> tuple[int, str]:
    order = {
        "full_cache": 0,
        "snapkv_0.3": 1,
        "snapkv_0.5": 2,
        "snapkv_0.7": 3,
        "h2o_0.5": 4,
        "streaminglm": 5,
    }
    return (order.get(method, 999), method)


def aggregate(results: list[dict], key_fn, value_fn) -> dict:
    grouped = defaultdict(list)
    for result in results:
        value = value_fn(result)
        if value is None:
            continue
        grouped[key_fn(result)].append(value)
    return {key: metric_mean(values) for key, values in grouped.items()}


def plot_quality_vs_budget(results: list[dict], output_dir: Path):
    """Plot IdleKV quality vs idle budget using real RULER 4K results."""
    idlekv_results = [
        result for result in results
        if result["experiment"]["type"] == "idlekv" and extract_ruler_avg(result) is not None
    ]
    if not idlekv_results:
        print("Skipping Figure 1: no IdleKV RULER results found")
        return

    models = sorted({result["experiment"]["model"]["short"] for result in idlekv_results})
    ratios = sorted({float(result["experiment"]["ratio"]) for result in idlekv_results})
    fig, axes = plt.subplots(
        len(models),
        len(ratios),
        figsize=(4.8 * len(ratios), 3.6 * len(models)),
        squeeze=False,
        sharey=True,
    )

    full_cache = aggregate(
        [r for r in results if r["experiment"]["type"] == "baseline" and r.get("method") == "full_cache"],
        lambda r: r["experiment"]["model"]["short"],
        extract_ruler_avg,
    )
    snapkv = aggregate(
        [
            r for r in results
            if r["experiment"]["type"] == "baseline"
            and isinstance(r.get("ratio"), (int, float))
            and r.get("method", "").startswith("snapkv_")
        ],
        lambda r: (r["experiment"]["model"]["short"], float(r["ratio"])),
        extract_ruler_avg,
    )

    for row_idx, model_short in enumerate(models):
        for col_idx, ratio in enumerate(ratios):
            ax = axes[row_idx][col_idx]
            subset = [
                result for result in idlekv_results
                if result["experiment"]["model"]["short"] == model_short
                and float(result["experiment"]["ratio"]) == ratio
            ]
            if not subset:
                ax.set_visible(False)
                continue

            phase_series = aggregate(
                subset,
                lambda r: (r["phases"], int(r["idle_budget_ms"])),
                extract_ruler_avg,
            )
            for phase in ["1", "2", "1+2"]:
                points = sorted(
                    [
                        (budget, score)
                        for (phase_key, budget), score in phase_series.items()
                        if phase_key == phase
                    ],
                    key=lambda item: item[0],
                )
                if not points:
                    continue
                budgets = [max(budget / 1000.0, 0.01) for budget, _ in points]
                scores = [score for _, score in points]
                ax.semilogx(
                    budgets,
                    scores,
                    "-o",
                    color=COLORS[f"idlekv_phase_{phase}"],
                    label=PHASE_LABELS[phase],
                    markersize=4,
                    linewidth=1.5,
                )

            fc = full_cache.get(model_short)
            if not np.isnan(fc):
                ax.axhline(fc, color=COLORS["full_cache"], linestyle="--", alpha=0.6, label="Full cache")
            snap = snapkv.get((model_short, ratio))
            if snap is not None and not np.isnan(snap):
                ax.axhline(snap, color=COLORS.get(f"snapkv_{ratio}", "#7f8c8d"), linestyle=":", alpha=0.7, label=f"SnapKV r={ratio:g}")

            ax.set_title(f"{display_model(model_short)} | r={ratio:g}")
            ax.set_xlabel("Idle compute budget (seconds)")
            if col_idx == 0:
                ax.set_ylabel("RULER 4K accuracy (%)")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.01, 6)
            ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(output_dir / "fig1_quality_vs_budget.pdf")
    fig.savefig(output_dir / "fig1_quality_vs_budget.png")
    plt.close(fig)
    print("Saved Figure 1: quality_vs_budget")


def plot_ruler_heatmap(results: list[dict], output_dir: Path):
    """Plot per-subtask RULER heatmap for selected methods."""
    candidates = []
    for result in results:
        exp = result["experiment"]
        model_short = exp["model"]["short"]
        subtasks = extract_ruler_subtasks(result)
        if not subtasks:
            continue

        label = None
        if exp["type"] == "baseline":
            label = result["method"]
        elif exp["type"] == "idlekv":
            phase = result["phases"]
            budget = int(result["idle_budget_ms"])
            ratio = float(result["ratio"])
            if phase == "1+2" and budget == 1000:
                label = f"idlekv_r{ratio:g}_b{budget}"

        if label is not None:
            candidates.append((model_short, label, subtasks))

    if not candidates:
        print("Skipping Figure 3: no suitable RULER heatmap inputs found")
        return

    models = sorted({model_short for model_short, _, _ in candidates})
    fig, axes = plt.subplots(len(models), 1, figsize=(8, 2.8 * len(models)), squeeze=False)

    for row_idx, model_short in enumerate(models):
        ax = axes[row_idx][0]
        rows = [(label, subtasks) for m, label, subtasks in candidates if m == model_short]
        rows.sort(key=lambda item: method_sort_key(item[0]))
        if not rows:
            ax.set_visible(False)
            continue

        subtasks = sorted(rows[0][1].keys())
        matrix = np.array([[row[subtask] for subtask in subtasks] for _, row in rows])
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            cbar=row_idx == 0,
            xticklabels=subtasks,
            yticklabels=[label for label, _ in rows],
            ax=ax,
            vmin=0,
            vmax=100,
        )
        ax.set_title(f"RULER per-subtask accuracy | {display_model(model_short)}")
        ax.set_xlabel("Subtask")
        ax.set_ylabel("Method")

    plt.tight_layout()
    fig.savefig(output_dir / "fig3_ruler_heatmap.pdf")
    fig.savefig(output_dir / "fig3_ruler_heatmap.png")
    plt.close(fig)
    print("Saved Figure 3: ruler_heatmap")


def preferred_budget(results: list[dict]) -> int | None:
    budgets = sorted({int(result["idle_budget_ms"]) for result in results})
    for candidate in [1000, 500, 2000, 100]:
        if candidate in budgets:
            return candidate
    return budgets[0] if budgets else None


def plot_component_ablation(results: list[dict], output_dir: Path):
    """Plot component ablation using real data at a preferred budget."""
    idlekv_results = [
        result for result in results
        if result["experiment"]["type"] == "idlekv" and extract_ruler_avg(result) is not None
    ]
    if not idlekv_results:
        print("Skipping Figure 4: no IdleKV results found")
        return

    models = sorted({result["experiment"]["model"]["short"] for result in idlekv_results})
    ratios = sorted({float(result["ratio"]) for result in idlekv_results})
    chosen_ratio = ratios[0]
    chosen_budget = preferred_budget([r for r in idlekv_results if float(r["ratio"]) == chosen_ratio])
    if chosen_budget is None:
        print("Skipping Figure 4: no preferred budget available")
        return

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False, sharey=True)

    full_cache = aggregate(
        [r for r in results if r["experiment"]["type"] == "baseline" and r.get("method") == "full_cache"],
        lambda r: r["experiment"]["model"]["short"],
        extract_ruler_avg,
    )
    snapkv = aggregate(
        [
            r for r in results
            if r["experiment"]["type"] == "baseline"
            and r.get("method") == f"snapkv_{chosen_ratio}"
        ],
        lambda r: r["experiment"]["model"]["short"],
        extract_ruler_avg,
    )

    for idx, model_short in enumerate(models):
        ax = axes[0][idx]
        phase_scores = aggregate(
            [
                r for r in idlekv_results
                if r["experiment"]["model"]["short"] == model_short
                and float(r["ratio"]) == chosen_ratio
                and int(r["idle_budget_ms"]) == chosen_budget
            ],
            lambda r: r["phases"],
            extract_ruler_avg,
        )
        labels = [f"SnapKV\nr={chosen_ratio:g}", "Phase 1", "Phase 2", "Phase 1+2", "Full\ncache"]
        values = [
            snapkv.get(model_short, np.nan),
            phase_scores.get("1", np.nan),
            phase_scores.get("2", np.nan),
            phase_scores.get("1+2", np.nan),
            full_cache.get(model_short, np.nan),
        ]
        if all(np.isnan(value) for value in values):
            ax.set_visible(False)
            continue
        bar_colors = [
            COLORS.get(f"snapkv_{chosen_ratio}", "#e67e22"),
            COLORS["idlekv_phase_1"],
            COLORS["idlekv_phase_2"],
            COLORS["idlekv_phase_1+2"],
            COLORS["full_cache"],
        ]
        bars = ax.bar(labels, values, color=bar_colors, edgecolor="white", width=0.65)
        ax.set_title(f"{display_model(model_short)} | {chosen_budget}ms")
        ax.set_ylabel("RULER 4K accuracy (%)")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.4,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    fig.savefig(output_dir / "fig4_component_ablation.pdf")
    fig.savefig(output_dir / "fig4_component_ablation.png")
    plt.close(fig)
    print("Saved Figure 4: component_ablation")


def plot_shadow_buffer_ablation(results: list[dict], output_dir: Path):
    """Plot shadow buffer size ablation using real results."""
    ablations = [
        result for result in results
        if result["experiment"]["type"] == "ablation_buffer" and extract_ruler_avg(result) is not None
    ]
    if not ablations:
        print("Skipping Figure 5: no shadow buffer ablation results found")
        return

    models = sorted({result["experiment"]["model"]["short"] for result in ablations})
    ratios = sorted({float(result["ratio"]) for result in ablations})
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False, sharey=True)

    for idx, model_short in enumerate(models):
        ax = axes[0][idx]
        subset = [result for result in ablations if result["experiment"]["model"]["short"] == model_short]
        grouped = aggregate(
            subset,
            lambda r: (float(r["ratio"]), int(r["shadow_buffer_size"])),
            extract_ruler_avg,
        )
        for ratio in ratios:
            points = sorted(
                [
                    (buffer_size, score)
                    for (ratio_key, buffer_size), score in grouped.items()
                    if ratio_key == ratio
                ],
                key=lambda item: item[0],
            )
            if not points:
                continue
            ax.plot(
                [buffer_size for buffer_size, _ in points],
                [score for _, score in points],
                "-o",
                label=f"r={ratio:g}",
                linewidth=1.6,
                markersize=4,
            )
        ax.set_title(display_model(model_short))
        ax.set_xlabel("Shadow buffer size")
        ax.set_ylabel("RULER 4K accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(output_dir / "fig5_shadow_buffer_ablation.pdf")
    fig.savefig(output_dir / "fig5_shadow_buffer_ablation.png")
    plt.close(fig)
    print("Saved Figure 5: shadow_buffer_ablation")


def write_summary(results: list[dict], output_dir: Path):
    """Write a small summary of what data the plotting pass saw."""
    summary = {
        "num_results": len(results),
        "by_type": defaultdict(int),
        "models": sorted({result["experiment"]["model"]["short"] for result in results}),
    }
    for result in results:
        summary["by_type"][result["experiment"]["type"]] += 1
    summary["by_type"] = dict(summary["by_type"])
    with open(output_dir / "plot_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument("--output-dir", type=str, default="figures/")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir) if results_dir.exists() else []
    write_summary(results, output_dir)
    print(f"Loaded {len(results)} experiment result files")

    plot_quality_vs_budget(results, output_dir)
    print("Skipping Figure 2: no throughput measurements are stored in experiment JSONs yet")
    plot_ruler_heatmap(results, output_dir)
    plot_component_ablation(results, output_dir)
    plot_shadow_buffer_ablation(results, output_dir)

    print(f"\nFinished figure generation pass for {output_dir}/")


if __name__ == "__main__":
    main()
