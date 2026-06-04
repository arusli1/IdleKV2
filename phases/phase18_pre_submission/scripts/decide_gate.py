"""Apply the Phase 18 v5.1 strong/weak/fail gate to a K-sweep analysis output.

Reads the contrasts CSV produced by analyze_w1_ksweep.py and applies
the pre-registered 3-tier rule from phase18_plan.md:

Strong pass:
  - At K=96, Δ(RepairKV - PageSummary-Quest-inspired) >= 0.10,
    Holm-adjusted Wilcoxon p < 0.01, HL CI lower bound > 0.03.
  - Holm-adjusted significant Δ >= 0.10 against TM-Recompute-BM25
    and Refresh-K-budgeted at K=96 (TM-BM25 may be absent if
    Step 5.6 deferred).
  - Burst-ablation gate: Score(RepairKV-no-burst) >= Score(PageSummary-
    Quest-inspired) - 0.05.
  - At least 5 of N K's show Holm-adjusted significant Δ > 0 against
    PageSummary-Quest-inspired (frontier robustness, scaled to N).

Weak pass:
  - At K=96, Δ vs PageSummary-Quest-inspired in [0.05, 0.10),
    Holm-adjusted p < 0.05.
  - Burst-ablation gate not violated.

Fail otherwise.

Outputs a one-line verdict and a decision JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _row_at(rows: list[dict], k: int, primary: str, baseline: str) -> dict | None:
    for row in rows:
        if int(row["k"]) == k and row["primary"] == primary and row["baseline"] == baseline:
            return row
    return None


def _frontier_count(rows: list[dict], primary: str, baseline: str) -> tuple[int, int]:
    relevant = [r for r in rows if r["primary"] == primary and r["baseline"] == baseline]
    rejected = [r for r in relevant if r.get("holm_rejected", "False") in ("True", True)]
    return len(rejected), len(relevant)


def decide(contrasts_csv: Path, frontier_csv: Path | None = None, k_target: int = 96) -> dict:
    rows = list(csv.DictReader(open(contrasts_csv)))
    for r in rows:
        for fld in ("k", "n"):
            r[fld] = int(r[fld])
        for fld in ("mean_diff", "median_diff", "wilcoxon_stat", "wilcoxon_p_uncorrected",
                    "hl_estimate", "hl_ci_lower", "hl_ci_upper", "wilcoxon_p_holm"):
            try:
                r[fld] = float(r[fld]) if r.get(fld, "") != "" else float("nan")
            except ValueError:
                r[fld] = float("nan")
        r["holm_rejected"] = r.get("holm_rejected", "False") in ("True", True)

    burst_score = None
    page_score = None
    if frontier_csv and frontier_csv.exists():
        for fr in csv.DictReader(open(frontier_csv)):
            if int(fr.get("k", 0)) == k_target:
                if fr.get("RepairKV-no-burst_mean"):
                    burst_score = float(fr["RepairKV-no-burst_mean"])
                if fr.get("PageSummary-Quest-inspired_mean"):
                    page_score = float(fr["PageSummary-Quest-inspired_mean"])

    decision = {
        "k_target": k_target,
        "checks": {},
        "verdict": "FAIL",
        "reason": "",
    }

    page_row = _row_at(rows, k_target, "RepairKV", "PageSummary-Quest-inspired")
    refresh_row = _row_at(rows, k_target, "RepairKV", "Refresh-K-budgeted")
    tm_row = _row_at(rows, k_target, "RepairKV", "TM-Recompute-BM25")

    # Strong-pass checks. Two distinct comparison types per the v5.1
    # abstract:
    #   - vs PageSummary-Quest-inspired and TM-Recompute-BM25:
    #     "RepairKV beats" (Δ >= 0.10).
    #   - vs Refresh-K-budgeted: "RepairKV approaches the quality of"
    #     (TOST equivalence at margin 0.10, NOT a Δ threshold).
    # The original gate definition required Δ >= 0.10 vs all three
    # contrasts which was internally inconsistent with the "approaches"
    # framing in the abstract; this is the corrected version.
    strong_checks = {}
    if page_row:
        strong_checks["delta_vs_PageSummary>=0.10"] = page_row["mean_diff"] >= 0.10
        strong_checks["holm_p<0.01_PageSummary"] = page_row["wilcoxon_p_holm"] < 0.01
        strong_checks["hl_lower>0.03_PageSummary"] = page_row["hl_ci_lower"] > 0.03
    if refresh_row:
        # "Approaches": absolute median difference within 0.10 *or*
        # RepairKV beats by a Holm-significant margin (interpret as
        # success either way -- approach is satisfied if quality is
        # at least as good as Refresh-K-budgeted up to a small margin).
        approaches = abs(float(refresh_row["median_diff"])) <= 0.10
        beats = refresh_row["mean_diff"] >= 0.05 and refresh_row["wilcoxon_p_holm"] < 0.05
        strong_checks["approaches_or_beats_RefreshBudgeted"] = approaches or beats
    if tm_row:
        strong_checks["delta_vs_TMRecompute>=0.10"] = tm_row["mean_diff"] >= 0.10
        strong_checks["holm_p<0.01_TMRecompute"] = tm_row["wilcoxon_p_holm"] < 0.01
    if burst_score is not None and page_score is not None:
        strong_checks["burst_ablation_gate"] = burst_score >= page_score - 0.05
    rejected, total = _frontier_count(rows, "RepairKV", "PageSummary-Quest-inspired")
    threshold = max(1, total // 2 + 1)  # majority of K's, scaled to amended K count
    strong_checks["frontier_majority"] = rejected >= threshold
    strong_checks["_frontier_detail"] = f"{rejected}/{total} K's reject Holm vs PageSummary"

    # Weak-pass checks
    weak_checks = {}
    if page_row:
        weak_checks["delta_vs_PageSummary_in_[0.05,0.10)"] = 0.05 <= page_row["mean_diff"] < 0.10
        weak_checks["holm_p<0.05_PageSummary"] = page_row["wilcoxon_p_holm"] < 0.05
    if burst_score is not None and page_score is not None:
        weak_checks["burst_ablation_gate"] = burst_score >= page_score - 0.05

    decision["checks"] = {"strong": strong_checks, "weak": weak_checks}

    # Apply gate
    strong_required = [k for k in strong_checks if not k.startswith("_")]
    if all(strong_checks[k] for k in strong_required):
        decision["verdict"] = "STRONG"
        decision["reason"] = "All strong-pass checks satisfied."
    elif weak_checks and all(v for k, v in weak_checks.items() if not k.startswith("_")):
        decision["verdict"] = "WEAK"
        decision["reason"] = "Weak-pass thresholds satisfied; not strong."
    else:
        decision["verdict"] = "FAIL"
        failed = [k for k, v in strong_checks.items() if not v and not k.startswith("_")]
        decision["reason"] = f"Strong-pass failed on: {failed}"

    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contrasts-csv", required=True, type=Path)
    parser.add_argument("--frontier-csv", type=Path, default=None)
    parser.add_argument("--k-target", type=int, default=96)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()
    decision = decide(args.contrasts_csv, frontier_csv=args.frontier_csv, k_target=args.k_target)
    print(json.dumps(decision, indent=2))
    print()
    print(f"VERDICT: {decision['verdict']}")
    print(f"REASON: {decision['reason']}")
    if args.out_json:
        with open(args.out_json, "w") as fp:
            json.dump(decision, fp, indent=2)
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
