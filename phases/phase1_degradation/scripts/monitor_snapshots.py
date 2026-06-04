#!/usr/bin/env python3
"""Periodic progress snapshots for a running Phase 1 rerun."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PHASE_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_ROOT))

from phase1.evaluation import build_phase1_summary, write_json
from phase1.paths import RESULTS_DIR
from phase1.task_registry import get_task_spec


def _read_rows(eviction_log_dir: Path) -> list[dict[str, Any]]:
    """Load every detailed per-example log currently present for the run."""
    rows: list[dict[str, Any]] = []
    if not eviction_log_dir.exists():
        return rows
    for path in sorted(eviction_log_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    return rows


def _expected_groups(tasks: list[str], budgets: list[int], num_samples: int) -> dict[str, dict[str, int]]:
    """Build the expected sample count table used for progress percentages."""
    return {
        task: {f"k{budget}": int(num_samples) for budget in budgets}
        for task in tasks
    }


def _normalize_task_names(tasks: list[str]) -> list[str]:
    """Accept either internal task ids or display names and normalize to display names."""
    normalized: list[str] = []
    for task in tasks:
        try:
            display_name = get_task_spec(task).display_name
        except Exception:
            display_name = task
        if display_name not in normalized:
            normalized.append(display_name)
    return normalized


def _counts_by_group(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Count how many completed samples exist per task/budget group."""
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        grouped[row["task"]][f"k{int(row['k_budget'])}"] += 1
    return {task: dict(budgets) for task, budgets in grouped.items()}


def _augment_summary(
    summary: dict[str, Any],
    *,
    counts: dict[str, dict[str, int]],
    expected: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Attach progress metadata alongside the partial summary metrics."""
    augmented: dict[str, Any] = {}
    for task, budget_expectations in expected.items():
        task_summary = dict(summary.get(task, {}))
        for budget_key, expected_count in budget_expectations.items():
            entry = dict(task_summary.get(budget_key, {}))
            completed = int(counts.get(task, {}).get(budget_key, 0))
            entry["completed_examples"] = completed
            entry["expected_examples"] = expected_count
            entry["completion_rate"] = round(completed / expected_count, 6) if expected_count else 0.0
            task_summary[budget_key] = entry
        augmented[task] = task_summary
    return augmented


def _snapshot_payload(
    *,
    run_label: str,
    rows: list[dict[str, Any]],
    tasks: list[str],
    budgets: list[int],
    num_samples: int,
    interval_seconds: int,
) -> dict[str, Any]:
    """Assemble one snapshot payload from the current partial outputs."""
    expected = _expected_groups(tasks, budgets, num_samples)
    counts = _counts_by_group(rows)
    total_expected = len(tasks) * len(budgets) * int(num_samples)
    total_completed = len(rows)
    summary = build_phase1_summary(rows) if rows else {}
    return {
        "run_label": run_label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "interval_seconds": interval_seconds,
        "total_completed": total_completed,
        "total_expected": total_expected,
        "total_remaining": max(total_expected - total_completed, 0),
        "completion_rate": round(total_completed / total_expected, 6) if total_expected else 0.0,
        "counts_by_group": counts,
        "expected_by_group": expected,
        "summary": _augment_summary(summary, counts=counts, expected=expected),
    }


def _snapshot_dir(run_label: str) -> Path:
    """Directory where timestamped snapshots for one run are written."""
    return RESULTS_DIR / "progress_snapshots" / run_label


def _write_snapshot(run_label: str, payload: dict[str, Any]) -> Path:
    """Persist both a timestamped snapshot and a rolling latest file."""
    out_dir = _snapshot_dir(run_label)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    timestamped_path = out_dir / f"snapshot_{timestamp}.json"
    latest_path = out_dir / "latest.json"
    write_json(timestamped_path, payload)
    write_json(latest_path, payload)
    return timestamped_path


def parse_args() -> argparse.Namespace:
    """Minimal CLI for snapshot monitoring."""
    parser = argparse.ArgumentParser(description="Write periodic progress snapshots for a live Phase 1 rerun.")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--budgets", nargs="+", type=int, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--interval-seconds", type=int, default=150)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Continuously snapshot a running rerun until it finishes."""
    args = parse_args()
    task_names = _normalize_task_names(args.tasks)
    eviction_log_dir = RESULTS_DIR / f"{args.run_label}_eviction_logs"
    total_expected = len(task_names) * len(args.budgets) * int(args.num_samples)

    while True:
        rows = _read_rows(eviction_log_dir)
        payload = _snapshot_payload(
            run_label=args.run_label,
            rows=rows,
            tasks=task_names,
            budgets=args.budgets,
            num_samples=args.num_samples,
            interval_seconds=args.interval_seconds,
        )
        snapshot_path = _write_snapshot(args.run_label, payload)
        print(
            f"[snapshot-monitor] wrote {snapshot_path} completed={payload['total_completed']}/{total_expected}",
            flush=True,
        )
        if args.once or payload["total_completed"] >= total_expected:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
