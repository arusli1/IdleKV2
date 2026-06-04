#!/usr/bin/env python3
"""Evaluate Phase 14 controlled proxy smoke summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return int(float(value))


def evaluate_rows(
    rows: list[dict[str, str]],
    *,
    headline_k: int = 96,
    min_lift: float = 0.10,
    max_control_lift: float = 0.10,
) -> list[dict[str, object]]:
    by_task: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_task.setdefault(row.get("task", ""), []).append(row)

    decisions: list[dict[str, object]] = []
    for task, task_rows in sorted(by_task.items()):
        row = next((candidate for candidate in task_rows if _int(candidate, "k", -1) == headline_k), None)
        if row is None:
            decisions.append(
                {
                    "task": task,
                    "status": "missing_headline_k",
                    "headline_k": headline_k,
                    "failures": ["missing_headline_k"],
                }
            )
            continue

        b_match = _float(row, "b_match")
        idlekv = _float(row, "idlekv")
        random_k = _float(row, "random_k", b_match)
        oldest_k = _float(row, "oldest_k", b_match)
        gold_k = _float(row, "gold_k", idlekv)
        idlekv_lift = idlekv - b_match
        random_lift = random_k - b_match
        oldest_lift = oldest_k - b_match
        max_control = max(random_lift, oldest_lift)

        failures: list[str] = []
        if idlekv_lift < min_lift:
            failures.append("weak_proxy_idlekv_lift")
        if max_control > max_control_lift:
            failures.append("content_agnostic_control_lift_too_high")
        if gold_k + 1e-9 < idlekv:
            failures.append("gold_reference_below_proxy_idlekv")

        status = "controlled_proxy_smoke_pass" if not failures else "controlled_proxy_smoke_fail"
        decisions.append(
            {
                "task": task,
                "status": status,
                "headline_k": headline_k,
                "b_match": round(b_match, 6),
                "idlekv": round(idlekv, 6),
                "gold_k": round(gold_k, 6),
                "idlekv_lift": round(idlekv_lift, 6),
                "random_lift": round(random_lift, 6),
                "oldest_lift": round(oldest_lift, 6),
                "max_control_lift": round(max_control, 6),
                "failures": failures,
            }
        )
    return decisions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--headline-k", type=int, default=96)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    decisions = evaluate_rows(_read_rows(args.summary_csv), headline_k=args.headline_k)
    if args.json:
        print(json.dumps(decisions, indent=2, sort_keys=True))
    else:
        print("[phase14 controlled proxy smoke]")
        for decision in decisions:
            print(
                "{task}: {status}; lift={idlekv_lift}; max_control_lift={max_control_lift}; "
                "failures={failures}".format(**decision)
            )
    return 0 if all(decision["status"] == "controlled_proxy_smoke_pass" for decision in decisions) else 2


if __name__ == "__main__":
    raise SystemExit(main())

