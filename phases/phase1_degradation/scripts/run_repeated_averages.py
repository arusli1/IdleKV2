#!/usr/bin/env python3
"""Run repeated Phase 1 sweeps with randomized dataset seeds and aggregate averages."""

from __future__ import annotations

import argparse
import json
import secrets
import statistics
import subprocess
import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_ROOT))

from phase1.evaluation import write_json
from phase1.paths import RESULTS_DIR


def _normalize_budgets(budgets: list[int]) -> list[int]:
    """Deduplicate requested budgets while preserving CLI order."""
    ordered: list[int] = []
    for budget in budgets:
        value = int(budget)
        if value <= 0:
            raise ValueError(f"Budgets must be positive; got {value}.")
        if value not in ordered:
            ordered.append(value)
    return ordered


def _summary_path(run_label: str) -> Path:
    """Return the summary JSON path for one repeat run."""
    return RESULTS_DIR / f"{run_label}_summary.json"


def _aggregate_path(run_label: str) -> Path:
    """Return the aggregate summary JSON path for the repeated sweep."""
    return RESULTS_DIR / f"{run_label}_aggregate_summary.json"


def _repeat_run_label(base_run_label: str, repeat_index: int) -> str:
    """Build a stable per-repeat label for artifacts."""
    return f"{base_run_label}_r{repeat_index:02d}"


def _draw_seed_offsets(repeats: int) -> list[int]:
    """Sample distinct non-zero seed offsets from system entropy."""
    offsets: set[int] = set()
    while len(offsets) < repeats:
        offsets.add(secrets.randbelow((2**31) - 1) + 1)
    return list(offsets)


def _scalar_metrics(payload: dict) -> dict[str, float]:
    """Keep just the scalar numeric metrics for aggregation."""
    return {
        key: float(value)
        for key, value in payload.items()
        if isinstance(value, (int, float))
    }


def _aggregate_repeat_summaries(
    repeat_summaries: list[dict],
    *,
    repeat_labels: list[str],
    seed_offsets: list[int],
) -> dict:
    """Average scalar per-budget metrics across repeated sweeps."""
    aggregate: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    for task_name in repeat_summaries[0]:
        task_payload: dict[str, dict[str, dict[str, object]]] = {}
        for budget_key in repeat_summaries[0][task_name]:
            metrics_by_name: dict[str, list[float]] = {}
            for repeat_summary in repeat_summaries:
                scalar_metrics = _scalar_metrics(repeat_summary[task_name][budget_key])
                for metric_name, value in scalar_metrics.items():
                    metrics_by_name.setdefault(metric_name, []).append(value)
            task_payload[budget_key] = {
                metric_name: {
                    "values": values,
                    "mean": round(sum(values) / len(values), 6),
                    "std": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
                    "min": round(min(values), 6),
                    "max": round(max(values), 6),
                }
                for metric_name, values in sorted(metrics_by_name.items())
            }
        aggregate[task_name] = task_payload

    return {
        "repeats": len(repeat_summaries),
        "repeat_labels": repeat_labels,
        "dataset_seed_offsets": seed_offsets,
        "aggregate": aggregate,
    }


def parse_args() -> argparse.Namespace:
    """Expose a small CLI for repeated Phase 1 sweeps."""
    parser = argparse.ArgumentParser(description="Run repeated Phase 1 sweeps and average per-budget metrics.")
    parser.add_argument("--tasks", nargs="+", default=["vt_4hop_permute"])
    parser.add_argument("--budgets", nargs="+", type=int, default=[1024, 2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--query-log-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the repeated sweeps and write an aggregate summary JSON."""
    args = parse_args()
    budgets = _normalize_budgets(args.budgets)
    if args.repeats < 1:
        raise ValueError(f"--repeats must be at least 1; got {args.repeats}.")

    script_path = Path(__file__).resolve().parent / "run_parallel_lowbudgets.py"
    repeat_labels: list[str] = []
    repeat_summaries: list[dict] = []
    seed_offsets = _draw_seed_offsets(args.repeats)

    for repeat_index, seed_offset in enumerate(seed_offsets, start=1):
        repeat_label = _repeat_run_label(args.run_label, repeat_index)
        repeat_labels.append(repeat_label)
        cmd = [
            sys.executable,
            str(script_path),
            "--tasks",
            *args.tasks,
            "--budgets",
            *[str(budget) for budget in budgets],
            "--run-label",
            repeat_label,
            "--max-parallel",
            str(args.max_parallel),
            "--query-log-tokens",
            str(args.query_log_tokens),
            "--dataset-seed-offset",
            str(seed_offset),
        ]
        if args.num_samples is not None:
            cmd.extend(["--num-samples", str(args.num_samples)])
        if args.force:
            cmd.append("--force")

        print(
            f"[repeat-sweep] repeat={repeat_index}/{args.repeats} run_label={repeat_label} dataset_seed_offset={seed_offset}",
            flush=True,
        )
        subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parent),
            check=True,
        )
        with open(_summary_path(repeat_label), "r", encoding="utf-8") as handle:
            repeat_summaries.append(json.load(handle))

    aggregate_payload = _aggregate_repeat_summaries(
        repeat_summaries,
        repeat_labels=repeat_labels,
        seed_offsets=seed_offsets,
    )
    write_json(_aggregate_path(args.run_label), aggregate_payload)
    print(f"[repeat-sweep] wrote {_aggregate_path(args.run_label)}", flush=True)


if __name__ == "__main__":
    main()
