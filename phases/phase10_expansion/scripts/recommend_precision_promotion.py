#!/usr/bin/env python3
"""Recommend whether precision-promotion rows are paper-ready."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phases.phase10_expansion.src.precision_promotion import evaluate_precision_promotion_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    args = parser.parse_args()

    with Path(args.summary_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in evaluate_precision_promotion_rows(rows):
        print(
            "nbits={nbits} k={k} lowbit_drop={lowbit_drop:.3f} "
            "idle_gain={idle_gain_vs_lowbit:.3f} "
            "idle_vs_static={idle_margin_vs_static:.3f} "
            "idle_vs_random={idle_margin_vs_random:.3f} "
            "active_bytes={active_bytes:.0f} action={action}".format(**row)
        )


if __name__ == "__main__":
    main()
