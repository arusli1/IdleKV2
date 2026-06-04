#!/usr/bin/env python3
"""Audit Phase 15 smoke artifacts against predeclared gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def audit_artifact(
    payload: dict[str, Any],
    *,
    primary_k: int,
    min_full_cache: float,
    max_cue_only: float,
    min_gap: float,
    max_answer_overlap: float = 0.0,
    require_individual_gap: bool = False,
    max_cue_only_hits: int | None = None,
) -> dict[str, Any]:
    rows = [row for row in payload.get("rows", []) if int(row.get("k", -1)) == int(primary_k)]
    if not rows:
        raise ValueError(f"Artifact has no rows for k={primary_k}.")
    full_cache = _mean([float(row["condition_a_score"]) for row in rows])
    b_match = _mean([float(row["b_match_score"]) for row in rows])
    cue_only_values = [float(row["cue_only_score"]) for row in rows if "cue_only_score" in row]
    cue_only = _mean(cue_only_values)
    cue_only_hits = sum(value > 0.0 for value in cue_only_values)
    evicted = [int(row.get("evicted_context_tokens", 0)) for row in rows]
    q1 = _mean([float(row.get("q1_score", 0.0)) for row in rows])
    b_answer_overlap = [float(row.get("b_answer_token_overlap_fraction", 1.0)) for row in rows]
    b_match_answer_overlap = [float(row.get("b_match_answer_token_overlap_fraction", 1.0)) for row in rows]
    individual_gap_ok = all(
        float(row["condition_a_score"]) == 1.0
        and float(row["b_match_score"]) == 0.0
        and float(row.get("cue_only_score", 0.0)) <= float(max_cue_only)
        for row in rows
    )
    gate_results = {
        "full_cache_ok": full_cache >= float(min_full_cache),
        "cue_only_ok": (not cue_only_values) or cue_only <= float(max_cue_only),
        "matched_gap_ok": (full_cache - b_match) >= float(min_gap),
        "eviction_ok": min(evicted) > 0,
        "answer_retention_ok": (
            max(b_answer_overlap) <= float(max_answer_overlap)
            and max(b_match_answer_overlap) <= float(max_answer_overlap)
        ),
    }
    if max_cue_only_hits is not None:
        gate_results["cue_only_hit_count_ok"] = cue_only_hits <= int(max_cue_only_hits)
    if require_individual_gap:
        gate_results["individual_gap_ok"] = individual_gap_ok
    return {
        "k": int(primary_k),
        "n_rows": len(rows),
        "mean_condition_a": round(full_cache, 6),
        "mean_b_match": round(b_match, 6),
        "mean_a_minus_b_match": round(full_cache - b_match, 6),
        "mean_cue_only": round(cue_only, 6),
        "cue_only_hits": int(cue_only_hits),
        "mean_q1": round(q1, 6),
        "max_b_answer_token_overlap_fraction": round(max(b_answer_overlap), 6),
        "max_b_match_answer_token_overlap_fraction": round(max(b_match_answer_overlap), 6),
        "individual_gap_ok": individual_gap_ok,
        "min_evicted_context_tokens": min(evicted),
        "max_evicted_context_tokens": max(evicted),
        "gate_results": gate_results,
        "passed": all(gate_results.values()),
        "manifest_hash": payload.get("manifest_hash"),
        "protocol_hash": payload.get("protocol_hash"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", help="Phase 15 JSON artifact.")
    parser.add_argument("--primary-k", type=int, default=96)
    parser.add_argument("--min-full-cache", type=float, default=0.80)
    parser.add_argument("--max-cue-only", type=float, default=0.20)
    parser.add_argument("--min-gap", type=float, default=0.15)
    parser.add_argument("--max-answer-overlap", type=float, default=0.0)
    parser.add_argument("--require-individual-gap", action="store_true")
    parser.add_argument("--max-cue-only-hits", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    result = audit_artifact(
        payload,
        primary_k=args.primary_k,
        min_full_cache=args.min_full_cache,
        max_cue_only=args.max_cue_only,
        min_gap=args.min_gap,
        max_answer_overlap=args.max_answer_overlap,
        require_individual_gap=bool(args.require_individual_gap),
        max_cue_only_hits=args.max_cue_only_hits,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
