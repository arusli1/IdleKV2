#!/usr/bin/env python3
"""CLI entrypoint for the Phase 5 oracle sweep."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_ROOT.parents[1]
for root in (PHASE_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from oracle.runner import (  # noqa: E402
    DEFAULT_BUDGETS,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_METHODS,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SERIALIZATION_SAMPLES,
    DEFAULT_TASKS,
    run_phase5_oracle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Number of deterministic examples per task.")
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH, help="Context length to generate.")
    parser.add_argument("--dataset-seed-offset", type=int, default=0, help="Optional deterministic dataset seed offset.")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), help="Task keys to run.")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), help="Eviction methods to run.")
    parser.add_argument("--budgets", nargs="+", default=[str(budget) for budget in DEFAULT_BUDGETS], help="Budget list.")
    parser.add_argument("--min-gap", type=float, default=0.05, help="Minimum A-B gap required for a recovery ratio.")
    parser.add_argument("--skip-serialization-diagnostic", action="store_true", help="Skip the exact full-cache serialization check.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse matching existing Phase 5 artifacts so interrupted full sweeps can resume without redoing completed slices.",
    )
    parser.add_argument(
        "--serialization-examples",
        type=int,
        default=DEFAULT_SERIALIZATION_SAMPLES,
        help="Examples per task for the exact serialization diagnostic.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_phase5_oracle(
        num_samples=args.num_samples,
        tasks=args.tasks,
        methods=args.methods,
        budgets=args.budgets,
        context_length=args.context_length,
        dataset_seed_offset=args.dataset_seed_offset,
        min_gap=args.min_gap,
        run_serialization_diagnostic=not args.skip_serialization_diagnostic,
        serialization_examples=args.serialization_examples,
        reuse_matching_artifacts=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
