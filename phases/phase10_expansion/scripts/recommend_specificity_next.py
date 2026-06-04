#!/usr/bin/env python3
"""Recommend follow-up actions from a specificity summary CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_ROOT.parents[1]
for root in (PHASE_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from phases.phase10_expansion.src.specificity import recommend_specificity_followup  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in recommend_specificity_followup(rows):
        print(
            "k={k} action={action} "
            "Idle-Matched={idle_vs_matched:+.3f} "
            "Idle-Stale={idle_vs_stale:+.3f} "
            "Idle-Wrong={idle_vs_wrong:+.3f} "
            "Refresh-Idle={refresh_vs_idle:+.3f} "
            "Gold-Idle={gold_headroom:+.3f} "
            "IdleGainCIlo={idle_gain_ci95_low:+.3f} "
            "IdleWinRate={idle_win_rate_vs_matched:.2f} "
            "saturated={saturated}".format(**row)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
