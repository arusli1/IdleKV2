#!/usr/bin/env python3
"""Recommend whether a compressor smoke deserves a locked follow-up."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phases.phase10_expansion.src.compressor import (  # noqa: E402
    evaluate_compressor_smoke,
    load_compressor_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    args = parser.parse_args()

    for rec in evaluate_compressor_smoke(load_compressor_rows(args.summary_csv)):
        print(
            "task={task} base_context_budget={base_context_budget} points={points} "
            "best_k={best_k} best_gain={best_gain:.3f} "
            "full_gap={full_vs_matched_gap:.3f} control_lift={max_control_lift:.3f} "
            "gold_covers_idle={gold_covers_idle} lock_followup={lock_followup} "
            "action={action}".format(**rec)
        )


if __name__ == "__main__":
    main()
