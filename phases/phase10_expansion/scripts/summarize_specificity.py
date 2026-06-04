#!/usr/bin/env python3
"""Summarize Phase 10 specificity artifacts into compact CSV rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_ROOT.parents[1]
for root in (PHASE_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from phases.phase10_expansion.src.specificity import (  # noqa: E402
    summarize_specificity_artifact,
    write_specificity_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = summarize_specificity_artifact(args.artifact)
    if not rows:
        raise SystemExit(f"No specificity rows found in {args.artifact}")
    write_specificity_csv(rows, args.out_csv)
    print(args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
