#!/usr/bin/env python3
"""Recommend whether query-count breadth results should be promoted."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phases.phase10_expansion.src.query_count import (  # noqa: E402
    evaluate_query_count_breadth,
    load_query_count_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    args = parser.parse_args()

    rows = load_query_count_rows(args.summary_csv)
    for row in evaluate_query_count_breadth(rows):
        print(
            "query_count={query_count} task={task} best_k={best_k} "
            "gain={idlekv_gain:.3f} full_gap={full_vs_matched_gap:.3f} "
            "control_lift={max_control_lift:.3f} action={action}".format(**row)
        )


if __name__ == "__main__":
    main()
