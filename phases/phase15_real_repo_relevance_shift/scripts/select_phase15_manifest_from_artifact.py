#!/usr/bin/env python3
"""Select Phase 15 rows that pass an ability/gap discovery gate.

This script is intentionally for diagnostic follow-up only. A manifest derived
from model outcomes must not be used as confirmatory paper evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from phases.phase15_real_repo_relevance_shift.src.manifest import (
    manifest_row_from_dict,
    stable_manifest_hash,
)


def _read_manifest(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(manifest_row_from_dict(json.loads(line)))
    return rows


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True, ensure_ascii=True) + "\n")


def _artifact_rows_by_example(payload: dict[str, Any], *, k: int) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in payload.get("rows", []):
        if int(row.get("k", -1)) != int(k):
            continue
        example_id = str(row.get("example_id", ""))
        if example_id:
            rows[example_id] = row
    return rows


def validate_artifact_manifest_hash(*, source_manifest_hash: str, artifact_payload: dict[str, Any]) -> None:
    """Require selected rows to come from the exact artifact manifest."""
    artifact_hash = str(artifact_payload.get("manifest_hash", ""))
    if artifact_hash and artifact_hash != str(source_manifest_hash):
        raise ValueError(
            "Artifact manifest hash does not match source manifest: "
            f"{artifact_hash} != {source_manifest_hash}"
        )


def _rejection_reason(
    row: dict[str, Any] | None,
    *,
    max_cue_only: float,
    max_answer_overlap: float,
    require_q1: bool,
) -> str | None:
    if row is None:
        return "missing_artifact_row"
    if require_q1 and float(row.get("q1_score", 0.0)) != 1.0:
        return "q1_miss"
    if float(row.get("condition_a_score", 0.0)) != 1.0:
        return "full_cache_miss"
    if float(row.get("b_match_score", 0.0)) != 0.0:
        return "b_match_success"
    if float(row.get("cue_only_score", 0.0)) > float(max_cue_only):
        return "cue_only_success"
    if int(row.get("evicted_context_tokens", 0)) <= 0:
        return "no_eviction"
    if float(row.get("b_answer_token_overlap_fraction", 1.0)) > float(max_answer_overlap):
        return "b_answer_retained"
    if float(row.get("b_match_answer_token_overlap_fraction", 1.0)) > float(max_answer_overlap):
        return "b_match_answer_retained"
    return None


def select_rows(
    *,
    manifest_rows,
    artifact_payload: dict[str, Any],
    k: int,
    max_cue_only: float,
    max_answer_overlap: float,
    require_q1: bool = False,
    max_selected_rows: int | None = None,
    max_rows_per_repo: int | None = None,
):
    artifact_by_id = _artifact_rows_by_example(artifact_payload, k=k)
    selected = []
    diagnostics = []
    reasons: Counter[str] = Counter()
    for manifest_row in manifest_rows:
        artifact_row = artifact_by_id.get(manifest_row.example_id)
        reason = _rejection_reason(
            artifact_row,
            max_cue_only=max_cue_only,
            max_answer_overlap=max_answer_overlap,
            require_q1=require_q1,
        )
        if reason is None:
            selected.append(manifest_row)
            reason = "selected"
        reasons[reason] += 1
        diagnostics.append(
            {
                "example_id": manifest_row.example_id,
                "repo_id": manifest_row.repo.repo_id,
                "answer": manifest_row.answer,
                "edge_type": str(manifest_row.q2.get("edge_type", "")),
                "reason": reason,
                "condition_a_score": None if artifact_row is None else artifact_row.get("condition_a_score"),
                "b_match_score": None if artifact_row is None else artifact_row.get("b_match_score"),
                "cue_only_score": None if artifact_row is None else artifact_row.get("cue_only_score"),
                "q1_score": None if artifact_row is None else artifact_row.get("q1_score"),
                "b_match_answer_token_overlap_fraction": (
                    None if artifact_row is None else artifact_row.get("b_match_answer_token_overlap_fraction")
                ),
            }
        )
    selected_before_balance = list(selected)
    if max_selected_rows is not None or max_rows_per_repo is not None:
        balanced = []
        repo_counts: Counter[str] = Counter()
        selected_ids = {row.example_id for row in selected_before_balance}
        kept_ids: set[str] = set()
        for row in selected_before_balance:
            repo_id = row.repo.repo_id
            if max_rows_per_repo is not None and repo_counts[repo_id] >= int(max_rows_per_repo):
                continue
            if max_selected_rows is not None and len(balanced) >= int(max_selected_rows):
                continue
            balanced.append(row)
            kept_ids.add(row.example_id)
            repo_counts[repo_id] += 1
        selected = balanced
        for item in diagnostics:
            if item["example_id"] in selected_ids and item["example_id"] not in kept_ids:
                item["reason"] = "repo_balance_truncated"
                reasons["repo_balance_truncated"] += 1
                reasons["selected"] -= 1
        if reasons.get("selected") == 0:
            reasons.pop("selected", None)
    return selected, diagnostics, +reasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--k", type=int, default=96)
    parser.add_argument("--max-cue-only", type=float, default=0.0)
    parser.add_argument("--max-answer-overlap", type=float, default=0.0)
    parser.add_argument("--require-q1", action="store_true")
    parser.add_argument("--max-selected-rows", type=int, default=None)
    parser.add_argument("--max-rows-per-repo", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_rows = _read_manifest(Path(args.manifest))
    artifact_payload = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    source_manifest_hash = stable_manifest_hash(manifest_rows)
    validate_artifact_manifest_hash(
        source_manifest_hash=source_manifest_hash,
        artifact_payload=artifact_payload,
    )
    selected, diagnostics, reasons = select_rows(
        manifest_rows=manifest_rows,
        artifact_payload=artifact_payload,
        k=int(args.k),
        max_cue_only=float(args.max_cue_only),
        max_answer_overlap=float(args.max_answer_overlap),
        require_q1=bool(args.require_q1),
        max_selected_rows=args.max_selected_rows,
        max_rows_per_repo=args.max_rows_per_repo,
    )
    _write_jsonl(Path(args.output), selected)
    summary = {
        "source_manifest": str(args.manifest),
        "source_manifest_hash": source_manifest_hash,
        "source_artifact": str(args.artifact),
        "source_artifact_manifest_hash": artifact_payload.get("manifest_hash"),
        "source_artifact_protocol_hash": artifact_payload.get("protocol_hash"),
        "k": int(args.k),
        "max_cue_only": float(args.max_cue_only),
        "max_answer_overlap": float(args.max_answer_overlap),
        "require_q1": bool(args.require_q1),
        "max_selected_rows": args.max_selected_rows,
        "max_rows_per_repo": args.max_rows_per_repo,
        "selected_rows": len(selected),
        "selected_repos": sorted({row.repo.repo_id for row in selected}),
        "selected_repo_counts": dict(sorted(Counter(row.repo.repo_id for row in selected).items())),
        "selected_manifest_hash": stable_manifest_hash(selected),
        "reason_counts": dict(sorted(reasons.items())),
        "diagnostics": diagnostics,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("selected_rows", "selected_manifest_hash", "reason_counts")}, sort_keys=True))
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
