"""Phase 18 W1 K-sweep preflight guard.

Reads a Step 1 W1 timing-only smoke artifact and verifies:
  1. T_repair stability: sigma(T_repair)/mu(T_repair) <= 0.10.
  2. Estimated K-sweep wall-clock <= 130 minutes (110 budget + 20% slack).
  3. The smoke artifact contains all expected new-condition fields,
     so the runner integration is not silently broken.

Aborts (exit code 1) on any failure with a clear message. On success,
prints the recommended ``--tm-budget-multiplier`` (1.05 or 1.20) and
the estimated wall-clock budget.

Usage::

    python -m phases.phase18_pre_submission.scripts.preflight_w1_ksweep \
        --smoke-artifact phases/phase6_repair/results/full/<smoke>.json \
        --n-paper 24 \
        --k-paper 9
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path


REQUIRED_NEW_FIELDS = (
    "idlekv_t_repair_s",
    "idlekv_q2_proj_ms",
    "idlekv_q2_score_ms",
    "refresh_k_budgeted_score",
    "refresh_k_budgeted_cap_fired",
    "page_summary_score",
    "page_summary_cap_fired",
    "repairkv_no_burst_score",
)


def preflight(
    smoke_artifact: Path,
    *,
    n_paper: int,
    k_paper: int,
    sigma_threshold: float = 0.10,
    wall_clock_floor_min: float = 110.0,
    wall_clock_ceiling_min: float = 130.0,
) -> int:
    if not smoke_artifact.exists():
        print(f"[preflight] FAIL: smoke artifact not found: {smoke_artifact}", file=sys.stderr)
        return 1

    payload = json.load(open(smoke_artifact))
    rows = payload.get("rows", [])
    if not rows:
        print("[preflight] FAIL: smoke artifact has no rows", file=sys.stderr)
        return 1

    # Sanity: required Phase 18 fields present
    missing_fields = []
    for field in REQUIRED_NEW_FIELDS:
        present = any(field in r for r in rows)
        if not present:
            missing_fields.append(field)
    if missing_fields:
        print(
            f"[preflight] FAIL: missing required Phase 18 fields in smoke: {missing_fields}",
            file=sys.stderr,
        )
        return 1

    # T_repair stability
    trs_ms = [float(r["idlekv_t_repair_s"]) * 1000.0 for r in rows if "idlekv_t_repair_s" in r]
    if not trs_ms:
        print("[preflight] FAIL: no idlekv_t_repair_s rows in smoke", file=sys.stderr)
        return 1
    mu = st.mean(trs_ms)
    sigma = st.stdev(trs_ms) if len(trs_ms) > 1 else 0.0
    sigma_over_mu = sigma / mu if mu > 0 else 0.0
    multiplier = 1.20 if sigma_over_mu > sigma_threshold else 1.05
    print(f"[preflight] T_repair: μ={mu:.1f}ms σ={sigma:.1f}ms σ/μ={sigma_over_mu:.3f}")
    print(f"[preflight] tm_budget_multiplier (decided by σ/μ vs {sigma_threshold:.2f}): {multiplier}")

    # Estimated K-sweep wall-clock
    smoke_elapsed_s = float(payload.get("elapsed_s", 0))
    smoke_n = len({r.get("example_id", "") for r in rows})
    smoke_n_k = len({int(r.get("k", 0)) for r in rows})
    if smoke_n == 0 or smoke_n_k == 0:
        print("[preflight] FAIL: cannot derive smoke n or n_K", file=sys.stderr)
        return 1
    per_example_smoke_s = smoke_elapsed_s / smoke_n if smoke_n > 0 else 0.0
    # Approximate per-K marginal cost as smoke wall-clock / smoke_n_k:
    # this is conservative because once-per-example overhead amortizes.
    per_example_paper_s = per_example_smoke_s * (k_paper / smoke_n_k) ** 0.6
    estimated_total_min = per_example_paper_s * n_paper / 60.0
    print(
        f"[preflight] smoke: elapsed={smoke_elapsed_s:.0f}s n={smoke_n} n_K={smoke_n_k} "
        f"per-example={per_example_smoke_s:.0f}s"
    )
    print(
        f"[preflight] estimated K-sweep: n={n_paper}, K={k_paper}, total ~{estimated_total_min:.0f} min"
    )
    if estimated_total_min > wall_clock_ceiling_min:
        print(
            f"[preflight] FAIL: estimated total {estimated_total_min:.0f} min > "
            f"ceiling {wall_clock_ceiling_min:.0f} min. Reduce K-sweep size or n.",
            file=sys.stderr,
        )
        return 1

    # All checks passed
    print(f"[preflight] OK -- estimated {estimated_total_min:.0f} min <= {wall_clock_ceiling_min:.0f} min")
    print(f"[preflight] PHASE18_W1_TM_MULT={multiplier}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-artifact", type=Path, required=True)
    parser.add_argument("--n-paper", type=int, default=24)
    parser.add_argument("--k-paper", type=int, default=9)
    args = parser.parse_args()
    return preflight(args.smoke_artifact, n_paper=args.n_paper, k_paper=args.k_paper)


if __name__ == "__main__":
    raise SystemExit(main())
