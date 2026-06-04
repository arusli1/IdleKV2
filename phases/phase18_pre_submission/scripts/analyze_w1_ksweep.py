"""Phase 18 W1 K-sweep analysis: Wilcoxon + Hodges-Lehmann + signed-rank TOST.

Pre-registered (Phase 18 v5 plan, commit 601d807):
- Primary test per pairwise contrast: paired Wilcoxon signed-rank with
  ``zero_method='pratt'``, ``method='exact'``.
- Companion CI: Hodges-Lehmann at 95% (Walsh-averages inversion).
- TOST equivalence vs Condition A at margin 0.20: signed-rank TOST.
- Multi-comparison: Holm-FWER over 27 tests (9 K's x 3 binding contrasts).
- Three binding contrasts: RepairKV vs (TM-Recompute-BM25,
  PageSummary-Quest-inspired, Refresh-K-budgeted).

Usage::

    python -m phases.phase18_pre_submission.scripts.analyze_w1_ksweep \
        --artifact phases/phase6_repair/results/full/<artifact>.json \
        --out-dir phases/phase18_pre_submission/results/w1/

The artifact must contain per-(example, K) rows with score columns
keyed by condition (idlekv_score, b_match_score, refresh_k_score,
refresh_k_budgeted_score, page_summary_score, repairkv_no_burst_score,
condition_a_score, condition_b_score). TM-Recompute-BM25 is optional
and reads tm_recompute_bm25_score if present.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats


def holm_correction(p_values: list[float], alpha: float = 0.05) -> tuple[list[bool], list[float]]:
    """Holm-Bonferroni step-down. Returns (rejected, adjusted_p) preserving input order.

    Adjusted p-value at rank i = max over j<=i of (m - j + 1) * p_(j), capped at 1.0.
    """
    m = len(p_values)
    if m == 0:
        return ([], [])
    order = sorted(range(m), key=lambda i: p_values[i])
    sorted_p = [p_values[i] for i in order]
    adj_sorted = []
    running_max = 0.0
    for i, p in enumerate(sorted_p):
        adj = min(1.0, (m - i) * p)
        running_max = max(running_max, adj)
        adj_sorted.append(running_max)
    rejected_sorted = [adj < alpha for adj in adj_sorted]
    adj = [0.0] * m
    rejected = [False] * m
    for rank, original_idx in enumerate(order):
        adj[original_idx] = adj_sorted[rank]
        rejected[original_idx] = rejected_sorted[rank]
    return (rejected, adj)


# Map condition label -> field name on the artifact row
CONDITION_FIELDS = {
    "A": "condition_a_score",
    "B": "condition_b_score",
    "B_match": "b_match_score",
    "RepairKV": "idlekv_score",
    "Refresh-K": "refresh_k_score",  # unbudgeted ceiling
    "Refresh-K-budgeted": "refresh_k_budgeted_score",
    "PageSummary-Quest-inspired": "page_summary_score",
    "RepairKV-no-burst": "repairkv_no_burst_score",
    "RepairKV-chunked": "repairkv_chunked_score",
    "Oracle-K": "oracle_k_score",
    "TM-Recompute-BM25": "tm_recompute_bm25_score",
    "Random-K": "random_k_score",
    "Oldest-K": "oldest_k_score",
}

BINDING_CONTRASTS = [
    ("RepairKV", "Refresh-K-budgeted"),
    ("RepairKV", "PageSummary-Quest-inspired"),
    ("RepairKV", "TM-Recompute-BM25"),  # may be absent if Step 5.6 skipped
    ("RepairKV", "RepairKV-chunked"),  # round-3 attack 2 defuse
]


def _grouped_by_k(rows: list[dict]) -> dict[int, list[dict]]:
    by_k: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        k = int(row.get("k", row.get("idlekv_restored_count", 0)))
        if k > 0:
            by_k[k].append(row)
    for k in by_k:
        by_k[k].sort(key=lambda r: str(r.get("example_id", "")))
    return by_k


def _wilcoxon_signed_rank(
    diffs: np.ndarray,
) -> tuple[float, float, float]:
    """Two-sided Wilcoxon signed-rank with Pratt zero-method.

    Returns ``(stat, p, median_diff)``. If all diffs are zero, returns
    ``(0.0, 1.0, 0.0)``.
    """
    diffs = np.asarray(diffs, dtype=float)
    if diffs.size == 0:
        return (0.0, 1.0, 0.0)
    if np.allclose(diffs, 0.0):
        return (0.0, 1.0, 0.0)
    try:
        result = stats.wilcoxon(diffs, zero_method="pratt", method="exact")
        return (float(result.statistic), float(result.pvalue), float(np.median(diffs)))
    except ValueError:
        # exact may fail with too-large N; fall back to approximate.
        result = stats.wilcoxon(diffs, zero_method="pratt", method="approx")
        return (float(result.statistic), float(result.pvalue), float(np.median(diffs)))


def _hodges_lehmann_ci(diffs: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """Hodges-Lehmann point estimator + symmetric CI on Walsh averages.

    For paired data, the HL estimator is the median of all pairwise
    averages (d_i + d_j) / 2 for i <= j (n*(n+1)/2 averages). The
    distribution-free CI is found by inverting the signed-rank test.

    Returns ``(estimate, lower, upper)``. If n is too small for a valid
    1-alpha interval, ``(estimate, nan, nan)``.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = diffs.size
    if n == 0:
        return (0.0, float("nan"), float("nan"))
    walsh = []
    for i in range(n):
        for j in range(i, n):
            walsh.append((diffs[i] + diffs[j]) / 2.0)
    walsh = np.sort(np.asarray(walsh, dtype=float))
    estimate = float(np.median(walsh))
    # critical value index from signed-rank distribution
    M = walsh.size  # n*(n+1)/2
    # For two-sided 1-alpha interval, find the largest k such that
    # P(W <= k) <= alpha/2 under H0. Use scipy's wilcoxon distribution.
    # When n is small, exact tables apply. Approximate via normal:
    if n < 5:
        return (estimate, float("nan"), float("nan"))
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    z = stats.norm.ppf(1 - alpha / 2)
    k_offset = int(round(z * math.sqrt(var_w)))
    lower_idx = int(M / 2) - k_offset - 1
    upper_idx = int(M / 2) + k_offset
    lower_idx = max(0, lower_idx)
    upper_idx = min(M - 1, upper_idx)
    return (estimate, float(walsh[lower_idx]), float(walsh[upper_idx]))


def _signed_rank_tost(
    diffs: np.ndarray,
    margin: float,
    alpha: float = 0.05,
) -> tuple[bool, float, float]:
    """TOST equivalence at margin via two one-sided signed-rank tests.

    H0_lower: median(d) <= -margin -> wilcoxon(d - (-margin), alternative='greater')
    H0_upper: median(d) >= +margin -> wilcoxon(d - (+margin), alternative='less')
    Both must reject at alpha; conclude equivalence within +/- margin.

    Returns ``(equivalent, p_lower, p_upper)``.
    """
    diffs = np.asarray(diffs, dtype=float)
    if diffs.size == 0:
        return (False, 1.0, 1.0)
    try:
        lower = stats.wilcoxon(
            diffs - (-margin),
            zero_method="pratt",
            method="exact",
            alternative="greater",
        )
        upper = stats.wilcoxon(
            diffs - (+margin),
            zero_method="pratt",
            method="exact",
            alternative="less",
        )
        equivalent = float(lower.pvalue) < alpha and float(upper.pvalue) < alpha
        return (equivalent, float(lower.pvalue), float(upper.pvalue))
    except ValueError:
        lower = stats.wilcoxon(
            diffs - (-margin),
            zero_method="pratt",
            method="approx",
            alternative="greater",
        )
        upper = stats.wilcoxon(
            diffs - (+margin),
            zero_method="pratt",
            method="approx",
            alternative="less",
        )
        equivalent = float(lower.pvalue) < alpha and float(upper.pvalue) < alpha
        return (equivalent, float(lower.pvalue), float(upper.pvalue))


def _diff_for_contrast(group: list[dict], a_field: str, b_field: str) -> np.ndarray | None:
    """Return paired score differences ``a - b`` if both fields present."""
    diffs = []
    for row in group:
        a = row.get(a_field)
        b = row.get(b_field)
        if a is None or b is None:
            return None
        diffs.append(float(a) - float(b))
    return np.asarray(diffs, dtype=float) if diffs else None


def analyze(artifact_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.load(open(artifact_path))
    rows = payload["rows"]
    by_k = _grouped_by_k(rows)
    ks = sorted(by_k.keys())
    print(f"[analyze] artifact={artifact_path.name} K's={ks} n_per_K={[len(by_k[k]) for k in ks]}")

    # Collect per-K per-contrast results
    per_k_results: list[dict] = []
    p_pool: list[tuple[int, str, str, float]] = []  # (k, primary, baseline, p)
    for k in ks:
        group = by_k[k]
        for primary, baseline in BINDING_CONTRASTS:
            a_field = CONDITION_FIELDS[primary]
            b_field = CONDITION_FIELDS[baseline]
            diffs = _diff_for_contrast(group, a_field, b_field)
            if diffs is None:
                continue
            stat, p, median_d = _wilcoxon_signed_rank(diffs)
            est, lo, hi = _hodges_lehmann_ci(diffs)
            row = {
                "k": k,
                "primary": primary,
                "baseline": baseline,
                "n": len(diffs),
                "mean_diff": float(np.mean(diffs)),
                "median_diff": median_d,
                "wilcoxon_stat": stat,
                "wilcoxon_p_uncorrected": p,
                "hl_estimate": est,
                "hl_ci_lower": lo,
                "hl_ci_upper": hi,
            }
            per_k_results.append(row)
            p_pool.append((k, primary, baseline, p))

    # Holm correction over the 27-test family (or fewer if TM-BM25 absent)
    if p_pool:
        p_values = [entry[3] for entry in p_pool]
        rejected, p_corrected = holm_correction(p_values, alpha=0.05)
        for row, rej, pc in zip(per_k_results, rejected, p_corrected):
            row["holm_rejected"] = bool(rej)
            row["wilcoxon_p_holm"] = float(pc)
    print(f"[analyze] Holm-{len(p_pool)} family applied")

    # TOST equivalence vs Condition A at K=96 (the abstract sentence's
    # equivalence claim)
    tost_results = []
    for primary in ("RepairKV", "RepairKV-no-burst"):
        a_field = CONDITION_FIELDS[primary]
        ref_field = CONDITION_FIELDS["A"]
        for k in ks:
            diffs = _diff_for_contrast(by_k[k], a_field, ref_field)
            if diffs is None:
                continue
            equivalent, p_lo, p_hi = _signed_rank_tost(diffs, margin=0.20, alpha=0.05)
            tost_results.append({
                "k": k,
                "primary": primary,
                "reference": "A",
                "margin": 0.20,
                "n": len(diffs),
                "median_diff": float(np.median(diffs)),
                "p_lower": p_lo,
                "p_upper": p_hi,
                "equivalent": equivalent,
            })

    # Per-K mean scores per condition (for the frontier figure CSV)
    frontier_rows = []
    for k in ks:
        row = {"k": k, "n": len(by_k[k])}
        for label, field in CONDITION_FIELDS.items():
            vals = [r.get(field) for r in by_k[k] if r.get(field) is not None]
            if vals:
                row[f"{label}_mean"] = float(np.mean(vals))
                row[f"{label}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                row[f"{label}_n"] = len(vals)
        frontier_rows.append(row)

    # Persist
    artifact_name = artifact_path.stem
    contrasts_csv = out_dir / f"{artifact_name}_w1_contrasts.csv"
    tost_csv = out_dir / f"{artifact_name}_w1_tost.csv"
    frontier_csv = out_dir / f"{artifact_name}_w1_frontier.csv"
    summary_json = out_dir / f"{artifact_name}_w1_summary.json"
    with open(contrasts_csv, "w") as fp:
        if per_k_results:
            writer = csv.DictWriter(fp, fieldnames=list(per_k_results[0].keys()))
            writer.writeheader()
            writer.writerows(per_k_results)
    with open(tost_csv, "w") as fp:
        if tost_results:
            writer = csv.DictWriter(fp, fieldnames=list(tost_results[0].keys()))
            writer.writeheader()
            writer.writerows(tost_results)
    with open(frontier_csv, "w") as fp:
        if frontier_rows:
            all_keys = set()
            for r in frontier_rows:
                all_keys.update(r.keys())
            writer = csv.DictWriter(fp, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(frontier_rows)
    summary = {
        "artifact": str(artifact_path),
        "k_values": ks,
        "n_per_k": {k: len(by_k[k]) for k in ks},
        "binding_contrasts_tested": [
            (p, b) for p, b in BINDING_CONTRASTS
            if any(row["primary"] == p and row["baseline"] == b for row in per_k_results)
        ],
        "holm_family_size": len(p_pool),
        "contrasts_csv": str(contrasts_csv),
        "tost_csv": str(tost_csv),
        "frontier_csv": str(frontier_csv),
    }
    with open(summary_json, "w") as fp:
        json.dump(summary, fp, indent=2)

    # Print headline numbers at K=96
    target_k = 96
    if target_k in by_k:
        print(f"\n[analyze] K={target_k} headline:")
        for row in per_k_results:
            if row["k"] == target_k:
                print(
                    f"  {row['primary']} vs {row['baseline']:30s}: "
                    f"Δ={row['mean_diff']:+.3f} (HL median {row['median_diff']:+.3f}) "
                    f"HL CI [{row['hl_ci_lower']:+.3f},{row['hl_ci_upper']:+.3f}]  "
                    f"Wilcoxon p_uncorrected={row['wilcoxon_p_uncorrected']:.4f} "
                    f"p_holm={row.get('wilcoxon_p_holm', float('nan')):.4f} "
                    f"reject={row.get('holm_rejected', False)}"
                )
        for row in tost_results:
            if row["k"] == target_k:
                print(
                    f"  TOST {row['primary']} vs A at margin {row['margin']}: "
                    f"p_lower={row['p_lower']:.4f} p_upper={row['p_upper']:.4f} "
                    f"equivalent={row['equivalent']}"
                )

    print(f"\n[analyze] wrote {contrasts_csv}")
    print(f"[analyze] wrote {tost_csv}")
    print(f"[analyze] wrote {frontier_csv}")
    print(f"[analyze] wrote {summary_json}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument(
        "--out-dir",
        default=Path("phases/phase18_pre_submission/results/w1"),
        type=Path,
    )
    args = parser.parse_args()
    analyze(Path(args.artifact), Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
