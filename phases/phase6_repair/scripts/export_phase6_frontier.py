#!/usr/bin/env python3
"""Export Phase 6 frontier CSVs for paper figures and tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phases.phase6_repair.src.reporting import (
    bootstrap_frontier_rows,
    frontier_rows,
    overlap_rows,
    runtime_rows,
    split_rows,
    write_csv,
)


def _extract_export_sections(artifact: dict) -> tuple[dict, dict[str, dict]]:
    """Normalize artifact aggregate payloads for exporters.

    Suite artifacts store nested `overall` / `by_task` sections.
    Single-task artifacts store one flat aggregate keyed only by `k...`.
    """
    aggregate = artifact["aggregate"]
    if "overall" in aggregate and "by_task" in aggregate:
        return aggregate["overall"], aggregate["by_task"]

    rows = artifact.get("rows", [])
    if rows:
        task_name = str(rows[0].get("task") or artifact.get("task") or "task")
    else:
        task_name = str(artifact.get("task") or artifact.get("suite_task") or "task")
    return aggregate, {task_name: aggregate}


def _merge_frontier_ci(
    base_rows: list[dict],
    ci_rows: list[dict],
    *,
    by_task: bool,
) -> list[dict]:
    ci_lookup: dict[tuple[object, int], dict] = {}
    for row in ci_rows:
        key = (row.get("task") if by_task else None, int(row["k"]))
        ci_lookup[key] = row
    merged: list[dict] = []
    for row in base_rows:
        key = (row.get("task") if by_task else None, int(row["k"]))
        extra = ci_lookup.get(key, {})
        merged_row = dict(row)
        for field, value in extra.items():
            if field in ("task", "k", "n"):
                continue
            merged_row[field] = value
        merged.append(merged_row)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True, help="Path to a Phase 6 result JSON artifact")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for generated CSV files")
    parser.add_argument("--prefix", type=str, default="phase6_frontier", help="Output filename prefix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    rows = artifact.get("rows", [])
    overall_section, by_task_section = _extract_export_sections(artifact)

    overall_path = args.outdir / f"{args.prefix}_overall.csv"
    by_split_path = args.outdir / f"{args.prefix}_by_split.csv"
    runtime_overall_path = args.outdir / f"{args.prefix}_runtime_overall.csv"
    runtime_by_split_path = args.outdir / f"{args.prefix}_runtime_by_split.csv"
    overlap_overall_path = args.outdir / f"{args.prefix}_overlap_overall.csv"
    overlap_by_split_path = args.outdir / f"{args.prefix}_overlap_by_split.csv"

    overall_frontier = frontier_rows(overall_section)
    split_frontier = split_rows(by_task_section)
    overall_frontier = _merge_frontier_ci(overall_frontier, bootstrap_frontier_rows(rows, by_task=False), by_task=False)
    split_frontier = _merge_frontier_ci(split_frontier, bootstrap_frontier_rows(rows, by_task=True), by_task=True)

    write_csv(overall_frontier, overall_path)
    write_csv(split_frontier, by_split_path)
    write_csv(runtime_rows(rows, by_task=False), runtime_overall_path)
    write_csv(runtime_rows(rows, by_task=True), runtime_by_split_path)
    write_csv(overlap_rows(rows, by_task=False), overlap_overall_path)
    write_csv(overlap_rows(rows, by_task=True), overlap_by_split_path)

    print(overall_path)
    print(by_split_path)
    print(runtime_overall_path)
    print(runtime_by_split_path)
    print(overlap_overall_path)
    print(overlap_by_split_path)


if __name__ == "__main__":
    main()
