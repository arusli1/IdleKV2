"""Render the tight-budget multiplier sweep figure for the paper.

Shows RepairKV / Refresh-K-budgeted / PageSummary scores vs budget
multiplier on a log-x axis. Demonstrates that Refresh-K-budgeted
is budget-responsive and that RepairKV maintains its quality
across budgets (because its scoring is outside the budget loop).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render(in_dir: Path, out_pdf: Path) -> None:
    arts = sorted(glob.glob(str(in_dir / "tight_mult*_clean_suite_*.json")))
    if not arts:
        print(f"[render] no artifacts in {in_dir}")
        return

    rows = []
    for art in arts:
        m = re.match(r"tight_mult([0-9.]+)_", Path(art).name)
        if not m:
            continue
        mult = float(m.group(1))
        d = json.load(open(art))
        ex = d["rows"]
        rows.append({
            "multiplier": mult,
            "RepairKV": np.mean([r["idlekv_score"] for r in ex]),
            "Refresh-K-budgeted": np.mean([r["refresh_k_budgeted_score"] for r in ex]),
            "PageSummary-Quest-inspired": np.mean([r["page_summary_score"] for r in ex]),
            "RepairKV-no-burst": np.mean([r["repairkv_no_burst_score"] for r in ex]),
            "Refresh-K (unbudgeted)": np.mean([r["refresh_k_score"] for r in ex]),
            "B_match": np.mean([r["b_match_score"] for r in ex]),
            "A": np.mean([r["condition_a_score"] for r in ex]),
            "budget_ms": np.mean([r.get("refresh_k_budgeted_t_repair_s", 0) * 1000 for r in ex]),
        })
    rows.sort(key=lambda r: r["multiplier"])

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    multipliers = [r["multiplier"] for r in rows]
    budgets_ms = [r["budget_ms"] for r in rows]
    style = [
        ("A", "k", "--", 1.0, "Condition A"),
        ("RepairKV", "tab:blue", "-", 2.0, "RepairKV"),
        ("Refresh-K (unbudgeted)", "tab:cyan", ":", 1.5, "Refresh-K (unbudgeted)"),
        ("Refresh-K-budgeted", "tab:purple", "-", 1.5, "Refresh-K-budgeted"),
        ("PageSummary-Quest-inspired", "tab:orange", "-", 1.5, "PageSummary-Quest"),
        ("RepairKV-no-burst", "tab:green", "-", 1.2, "RepairKV-no-burst"),
        ("B_match", "tab:red", "--", 1.5, "Matched no-repair"),
    ]
    for label_key, color, ls, lw, label in style:
        ys = [r[label_key] for r in rows]
        ax.plot(budgets_ms, ys, color=color, linestyle=ls, linewidth=lw, marker="o", markersize=5, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Per-example wall-clock budget (ms; multipliers " + ", ".join(f"{m:.2f}x" for m in multipliers) + ")")
    ax.set_ylabel("Exact $Q_2$ score")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Tight-budget sweep on MQ-NIAH-4Q at $K=96$ ($n=12 \\times 3$ partitions)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=200)
    print(f"[render] wrote {out_pdf}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("phases/phase18_pre_submission/results/w1_tight"))
    parser.add_argument("--out-pdf", type=Path, default=Path("phases/phase18_pre_submission/results/figures/tight_sweep_K96.pdf"))
    args = parser.parse_args()
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    render(args.dir, args.out_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
