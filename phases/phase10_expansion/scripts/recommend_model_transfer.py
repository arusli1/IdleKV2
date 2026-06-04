#!/usr/bin/env python3
"""Recommend whether a cross-model repair run belongs in the appendix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phases.phase10_expansion.src.model_transfer import (  # noqa: E402
    evaluate_model_transfer_rows,
    load_model_transfer_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    args = parser.parse_args()

    recommendations = evaluate_model_transfer_rows(load_model_transfer_rows(args.summary_csv))
    all_candidate = bool(recommendations) and all(bool(rec["appendix_candidate"]) for rec in recommendations)
    for rec in recommendations:
        print(
            "task={task} base_context_budget={base_context_budget} points={points} "
            "best_k={best_k} best_gain={best_gain:.3f} k96_gain={k96_gain:.3f} "
            "full_gap={full_vs_matched_gap:.3f} control_lift={max_control_lift:.3f} "
            "gold_covers_idle={gold_covers_idle} appendix_candidate={appendix_candidate} "
            "action={action}".format(**rec)
        )
    print(f"all_budgets_appendix_candidate={all_candidate}")


if __name__ == "__main__":
    main()
