#!/usr/bin/env python3
"""Recommend whether a Phase 11 full-grid run deserves main-paper space."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phases.phase11_main_robustness.src import evaluate_main_candidate, load_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    args = parser.parse_args()

    for decision in evaluate_main_candidate(load_rows(args.summary_csv)):
        print(
            "task={task} base_context_budget={base_context_budget} points={points} "
            "best_k={best_k} best_gain={best_gain:.3f} "
            "best_eligible_k={best_eligible_k} best_eligible_gain={best_eligible_gain:.3f} "
            "full_gap={full_vs_matched_gap:.3f} non_saturated={non_saturated} "
            "main_candidate={main_candidate} action={action}".format(**decision)
        )


if __name__ == "__main__":
    main()

