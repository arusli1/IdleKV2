"""Render Phase 18 frontier figure A (and Figure B inset wall-clock bar).

Reads the K-sweep artifact analysis output and produces:
- ``frontier_4q_ksweep.pdf``: x = K (log scale), y = exact Q2 score,
  curves for B_match, RepairKV, Refresh-K (unbudgeted ceiling),
  Refresh-K-budgeted, PageSummary-Quest-inspired, RepairKV-no-burst,
  Random-K, Oldest-K, with an A=ceiling reference at the top.
- ``walltime_bar_K96.pdf``: log-scale wall-clock per condition at
  K=96 (single panel; companion to Figure A).

Usage::

    python -m phases.phase18_pre_submission.scripts.render_phase18_figures \
        --frontier-csv phases/phase18_pre_submission/results/w1/<artifact>_w1_frontier.csv \
        --artifact phases/phase6_repair/results/full/<artifact>.json \
        --out-dir phases/phase18_pre_submission/results/figures/
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITION_PLOT_ORDER = [
    ("A", "Condition A (full cache)", "k", "--", 1.0),
    ("Oracle-K", "Oracle-K (gold-span ceiling)", "tab:brown", "-.", 1.5),
    ("RepairKV", "RepairKV", "tab:blue", "-", 2.0),
    ("Refresh-K", "Refresh-K (unbudgeted ceiling)", "tab:cyan", ":", 1.5),
    ("Refresh-K-budgeted", "Refresh-K-budgeted", "tab:purple", "-", 1.5),
    ("PageSummary-Quest-inspired", "PageSummary-Quest-inspired", "tab:orange", "-", 1.5),
    ("RepairKV-no-burst", "RepairKV-no-burst", "tab:green", "-", 1.2),
    ("Random-K", "Random-K", "tab:gray", "-", 1.0),
    ("Oldest-K", "Oldest-K", "tab:olive", "-", 1.0),
    ("B_match", "Matched no-repair", "tab:red", "--", 1.5),
]


def render_frontier(frontier_csv: Path, out_pdf: Path) -> None:
    rows = list(csv.DictReader(open(frontier_csv)))
    if not rows:
        print(f"[figures] frontier CSV empty: {frontier_csv}")
        return
    ks = sorted({int(r["k"]) for r in rows})
    by_k = {int(r["k"]): r for r in rows}

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for cond, label, color, ls, lw in CONDITION_PLOT_ORDER:
        ys = []
        xs = []
        for k in ks:
            mean_field = f"{cond}_mean"
            if mean_field in by_k[k] and by_k[k][mean_field]:
                ys.append(float(by_k[k][mean_field]))
                xs.append(k)
        if not ys:
            continue
        ax.plot(xs, ys, color=color, linestyle=ls, linewidth=lw, marker="o", markersize=4, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Restore budget $K$ (log)")
    ax.set_ylabel("Exact $Q_2$ score")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("MQ-NIAH-4Q frontier on Qwen2.5-7B-Instruct, $B_{\\mathrm{base}}=16384$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=200)
    print(f"[figures] wrote {out_pdf}")
    plt.close(fig)


def render_walltime_bar(artifact: Path, out_pdf: Path, target_k: int = 96) -> None:
    payload = json.load(open(artifact))
    rows = [r for r in payload["rows"] if int(r.get("k", 0)) == target_k]
    if not rows:
        print(f"[figures] no rows at K={target_k}")
        return

    # Per-condition wall-clock (per-example, per-K share).
    field_map = {
        "RepairKV (per-K)": ("idlekv_t_repair_s", lambda r: float(r.get("idlekv_t_repair_s", 0))),
        "Refresh-K-budgeted scoring": ("refresh_k_budgeted_score_s", lambda r: float(r.get("refresh_k_budgeted_score_s", 0))),
        "PageSummary scoring": ("page_summary_score_s", lambda r: float(r.get("page_summary_score_s", 0))),
        "Refresh-K (unbudgeted)": ("refresh_context_scoring_s", lambda r: float(r.get("refresh_context_scoring_s", 0))),
    }
    labels = []
    medians = []
    p95s = []
    for label, (_field, getter) in field_map.items():
        vals = [getter(r) for r in rows if getter(r) > 0]
        if not vals:
            continue
        labels.append(label)
        medians.append(np.median(vals))
        p95s.append(np.percentile(vals, 95))

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, medians, yerr=[np.array(medians) * 0.0, np.array(p95s) - np.array(medians)],
                  color=["tab:blue", "tab:purple", "tab:orange", "tab:cyan"][:len(labels)],
                  capsize=4)
    ax.set_yscale("log")
    ax.set_ylabel("Per-example wall-clock (seconds, log)")
    ax.set_title(f"Per-example wall-clock at $K={target_k}$ (median, p95 error bars)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    for bar, val in zip(bars, medians):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.1, f"{val*1000:.0f} ms",
                ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=200)
    print(f"[figures] wrote {out_pdf}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-csv", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("phases/phase18_pre_submission/results/figures/"), type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    render_frontier(args.frontier_csv, args.out_dir / "frontier_4q_ksweep.pdf")
    render_walltime_bar(args.artifact, args.out_dir / "walltime_bar_K96.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
