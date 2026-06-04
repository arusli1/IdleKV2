#!/usr/bin/env python3
"""Write paired bootstrap intervals for Phase 13 multi-turn rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

REPO_DIR = Path(__file__).resolve().parents[3]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from phases.phase13_iteration_framework.src import multiturn_uncertainty_rows  # noqa: E402


DEFAULT_CONDITIONS = (
    "IdleKV",
    "CurrentQOnly-K",
    "StaleQOnly-K",
    "StaleQ-K",
    "Random-K",
    "Oldest-K",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_revisit_turns(path: Path | None) -> tuple[int, ...]:
    if path is None:
        return tuple()
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("schedule", {}).get("revisit_events", [])
    turns = sorted({int(event["revisit_turn"]) for event in events if "revisit_turn" in event})
    return tuple(turns)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "k",
        "condition",
        "mean_gain",
        "gain_lo",
        "gain_hi",
        "n_paired",
        "mean_noninitial_gain",
        "noninitial_gain_lo",
        "noninitial_gain_hi",
        "n_noninitial",
        "mean_revisit_gain",
        "revisit_gain_lo",
        "revisit_gain_hi",
        "n_revisit",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-csv", type=Path, required=True, help="Per-turn Phase 13 rows CSV.")
    parser.add_argument("--raw-json", type=Path, default=None, help="Raw run JSON with schedule.revisit_events.")
    parser.add_argument(
        "--revisit-turn",
        type=int,
        action="append",
        default=None,
        help="Explicit revisit turn. Repeat for multiple revisit turns; overrides --raw-json values.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
        help="Conditions to summarize. Defaults to repair and control conditions.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    revisit_turns = tuple(sorted(set(args.revisit_turn or _read_revisit_turns(args.raw_json))))
    if not revisit_turns:
        raise SystemExit("No revisit turns found. Pass --raw-json or at least one --revisit-turn.")

    summaries = multiturn_uncertainty_rows(
        _read_rows(args.rows_csv),
        revisit_turns=revisit_turns,
        conditions=tuple(args.conditions),
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
    )
    _write_rows(args.out_csv, [dict(row) for row in summaries])
    print(args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
