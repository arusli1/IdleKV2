#!/usr/bin/env python3
"""Postprocess the latest locked Phase 13 multi-turn run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[3]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from phases.phase13_iteration_framework.scripts.summarize_multiturn_uncertainty import (  # noqa: E402
    DEFAULT_CONDITIONS,
    _read_revisit_turns,
    _read_rows,
    _write_rows,
)
from phases.phase13_iteration_framework.src import multiturn_uncertainty_rows  # noqa: E402


def latest_locked_summary(results_dir: Path) -> Path:
    candidates = sorted(
        results_dir.glob("multiturn_hard_locked_summary_n*_k*.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No locked multi-turn summary found in {results_dir}.")
    return candidates[-1]


def locked_paths(summary_path: Path) -> dict[str, Path]:
    rows_path = summary_path.with_name(summary_path.name.replace("summary", "rows", 1))
    raw_name = summary_path.name.replace("multiturn_hard_locked_summary_", "multiturn_hard_locked_").replace(
        ".csv",
        "_raw.json",
    )
    raw_path = summary_path.with_name(raw_name)
    uncertainty_path = summary_path.with_name(summary_path.name.replace("summary", "uncertainty", 1))
    return {
        "summary": summary_path,
        "rows": rows_path,
        "raw": raw_path,
        "uncertainty": uncertainty_path,
    }


def write_locked_uncertainty(
    summary_path: Path,
    *,
    conditions: tuple[str, ...] = DEFAULT_CONDITIONS,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> Path:
    paths = locked_paths(summary_path)
    rows_path = paths["rows"]
    raw_path = paths["raw"]
    if not rows_path.exists():
        raise FileNotFoundError(f"Missing locked rows CSV: {rows_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing locked raw JSON: {raw_path}")

    revisit_turns = _read_revisit_turns(raw_path)
    if not revisit_turns:
        raise ValueError(f"No revisit turns found in {raw_path}.")
    summaries = multiturn_uncertainty_rows(
        _read_rows(rows_path),
        revisit_turns=revisit_turns,
        conditions=conditions,
        n_bootstrap=bootstrap_samples,
        seed=seed,
    )
    _write_rows(paths["uncertainty"], [dict(row) for row in summaries])
    return paths["uncertainty"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_DIR / "phases" / "phase13_iteration_framework" / "results",
    )
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = args.summary_csv or latest_locked_summary(args.results_dir)
    out_path = write_locked_uncertainty(
        summary_path,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
