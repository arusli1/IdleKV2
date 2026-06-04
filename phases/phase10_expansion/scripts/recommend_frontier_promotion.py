#!/usr/bin/env python3
"""Recommend whether a full frontier sweep should enter the main paper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phases.phase10_expansion.src.frontier import (  # noqa: E402
    evaluate_frontier_promotion,
    load_frontier_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    args = parser.parse_args()

    rec = evaluate_frontier_promotion(load_frontier_rows(args.summary_csv))
    print(
        "query_count={query_count} task={task} points={points} best_k={best_k} "
        "best_gain={best_gain:.3f} control_lift={max_control_lift:.3f} "
        "large_drops={large_drops} gold_covers_idle={gold_covers_idle} "
        "boundary_review={boundary_review} action={action}".format(**rec)
    )


if __name__ == "__main__":
    main()
