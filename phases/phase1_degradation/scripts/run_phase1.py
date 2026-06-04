#!/usr/bin/env python3
"""Thin CLI entrypoint for the repo-local Phase 1 benchmark runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_ROOT))

from phase1 import DEFAULT_ALGORITHMS, DEFAULT_BUDGETS, DEFAULT_CONTEXT_LENGTHS, DEFAULT_TASKS
from phase1.runner import run_phase1


def parse_args() -> argparse.Namespace:
    """Expose the small set of knobs used by the reduced Phase 1 runner."""
    parser = argparse.ArgumentParser(description="Run the reduced Phase 1 RULER-KVR SnapKV degradation experiments.")
    # Keep argument parsing minimal; defaults come from the shared phase1 package.
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--context-lengths", nargs="+", type=int, default=DEFAULT_CONTEXT_LENGTHS)
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_ALGORITHMS)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--query-log-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    """Translate CLI flags into the package-level runner call."""
    # Step 1: capture user intent from the CLI.
    args = parse_args()
    # Step 2: hand off the parsed knobs to the runner that owns the experiment logic.
    # The CLI stays intentionally thin so the real logic lives in importable code.
    run_phase1(
        tasks=args.tasks,
        context_lengths=args.context_lengths,
        budgets=args.budgets,
        algorithms=args.algorithms,
        num_samples=args.num_samples,
        force=args.force,
        query_log_tokens=args.query_log_tokens,
    )


if __name__ == "__main__":
    main()
