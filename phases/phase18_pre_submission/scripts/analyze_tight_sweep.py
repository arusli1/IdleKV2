"""Analyze the budget-multiplier sweep at K=96.

Reads per-multiplier artifacts under
phases/phase18_pre_submission/results/w1_tight/ and computes:

- Mean/median scores per condition per multiplier
- Cap-firing rate per condition per multiplier
- Per-example wall-clock distribution per multiplier

Used to defuse the "Refresh-K-budgeted is unbudgeted in disguise"
hostile-reviewer attack by showing how scores degrade as the budget
actually binds.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path


COND_FIELDS = {
    "A": "condition_a_score",
    "B_match": "b_match_score",
    "RepairKV": "idlekv_score",
    "Refresh-K (unbudgeted)": "refresh_k_score",
    "Refresh-K-budgeted": "refresh_k_budgeted_score",
    "PageSummary-Quest-inspired": "page_summary_score",
    "RepairKV-no-burst": "repairkv_no_burst_score",
}


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def analyze_directory(d: Path) -> None:
    artifacts = sorted(d.glob("tight_mult*_clean_suite_*.json"))
    if not artifacts:
        print(f"[analyze] no per-multiplier artifacts found under {d}")
        return

    rows = []
    for art in artifacts:
        m = re.match(r"tight_mult([0-9.]+)_", art.name)
        if not m:
            continue
        mult = float(m.group(1))
        payload = json.load(open(art))
        ex_rows = payload["rows"]
        n = len(ex_rows)
        scores = {}
        for label, field in COND_FIELDS.items():
            vals = [r.get(field) for r in ex_rows if r.get(field) is not None]
            scores[label] = _mean(vals)
        # Cap firing rate
        rkb_caps = sum(1 for r in ex_rows if r.get("refresh_k_budgeted_cap_fired"))
        ps_caps = sum(1 for r in ex_rows if r.get("page_summary_cap_fired"))
        # Positions scored
        rkb_pos = _mean([r.get("refresh_k_budgeted_positions_scored", 0) for r in ex_rows])
        ps_pos = _mean([r.get("page_summary_positions_scored", 0) for r in ex_rows])
        ps_chunks = _mean([r.get("page_summary_chunks_visited", 0) for r in ex_rows])
        # Budget
        budget_ms = _mean([r.get("refresh_k_budgeted_t_repair_s", 0) * 1000 for r in ex_rows])
        # T_repair
        t_repair_ms = _mean([r.get("idlekv_t_repair_s", 0) * 1000 for r in ex_rows])
        rows.append({
            "multiplier": mult,
            "n": n,
            **scores,
            "RKB_cap_fired_pct": 100 * rkb_caps / n,
            "RKB_positions_scored": rkb_pos,
            "PS_cap_fired_pct": 100 * ps_caps / n,
            "PS_chunks_visited": ps_chunks,
            "PS_positions_scored": ps_pos,
            "budget_ms": budget_ms,
            "t_repair_full_ms": t_repair_ms,
        })

    # Pairwise Wilcoxon vs RepairKV per multiplier
    from scipy import stats
    import numpy as np
    for art in artifacts:
        m = re.match(r"tight_mult([0-9.]+)_", art.name)
        mult = float(m.group(1))
        d = json.load(open(art))
        ex = d["rows"]
        repkv = np.array([r["idlekv_score"] for r in ex])
        rkbud = np.array([r["refresh_k_budgeted_score"] for r in ex])
        psum = np.array([r["page_summary_score"] for r in ex])
        for label, opp in [("RKbud", rkbud), ("PSum", psum)]:
            diff = repkv - opp
            if np.allclose(diff, 0):
                p = 1.0
            else:
                try:
                    res = stats.wilcoxon(diff, zero_method="pratt", method="exact")
                except ValueError:
                    res = stats.wilcoxon(diff, zero_method="pratt", method="approx")
                p = float(res.pvalue)
            for r in rows:
                if r["multiplier"] == mult:
                    r[f"p_wilcoxon_{label}"] = p
                    r[f"delta_RepairKV_minus_{label}"] = float(diff.mean())

    rows.sort(key=lambda r: r["multiplier"])
    print(f"[analyze] {len(rows)} multipliers x n={rows[0]['n']} examples each")
    print()
    print(f"{'mult':>6} {'budget':>9} | "
          + " | ".join(f"{label:>13}" for label in COND_FIELDS.keys())
          + " | RKB_cap RKB_pos | PS_cap PS_chunks PS_pos")
    print("-" * 200)
    for r in rows:
        scores_str = " | ".join(f"{r[label]:>13.3f}" for label in COND_FIELDS.keys())
        print(f"{r['multiplier']:>6.2f} {r['budget_ms']:>9.0f} | "
              f"{scores_str} | "
              f"{r['RKB_cap_fired_pct']:>4.0f}%  {r['RKB_positions_scored']:>5.0f} | "
              f"{r['PS_cap_fired_pct']:>4.0f}%   {r['PS_chunks_visited']:>4.1f}  {r['PS_positions_scored']:>5.0f}")

    # Save to CSV
    out_csv = Path("phases/phase18_pre_submission/results/w1_tight/tight_sweep_summary.csv")
    if rows:
        with open(out_csv, "w") as fp:
            w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[analyze] wrote {out_csv}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("phases/phase18_pre_submission/results/w1_tight"),
    )
    args = parser.parse_args()
    analyze_directory(args.dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
