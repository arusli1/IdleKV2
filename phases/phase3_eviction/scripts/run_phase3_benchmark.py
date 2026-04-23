#!/usr/bin/env python3
"""CLI entrypoint for the resumable Phase 3 benchmark runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_ROOT.parents[1]
for root in (PHASE_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from src.benchmark import DEFAULT_BUDGETS, DEFAULT_CONTEXT_LENGTH, DEFAULT_METHODS, DEFAULT_TASKS, run_phase3_benchmark  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="manual", help="Summary subdirectory name under results/phase3_degradation/.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of deterministic examples per task.")
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH, help="Context length to generate.")
    parser.add_argument("--dataset-seed-offset", type=int, default=0, help="Optional deterministic dataset seed offset.")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), help="Task keys to run.")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), help="Methods to run.")
    parser.add_argument(
        "--budgets",
        nargs="+",
        default=[str(budget) if budget != DEFAULT_CONTEXT_LENGTH else "full" for budget in DEFAULT_BUDGETS],
        help="Budget list. Use integers or the literal 'full'.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute example results even if cached rows already exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_phase3_benchmark(
        num_samples=args.num_samples,
        tasks=args.tasks,
        methods=args.methods,
        budgets=args.budgets,
        context_length=args.context_length,
        dataset_seed_offset=args.dataset_seed_offset,
        label=args.label,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
