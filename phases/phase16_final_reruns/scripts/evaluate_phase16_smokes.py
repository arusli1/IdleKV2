#!/usr/bin/env python3
"""Evaluate Phase 16 smoke summaries against paper-action gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


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
    min_full_score: float = 0.90,
    min_matched_gap: float = 0.20,
    min_idlekv_gain: float = 0.10,
    min_control_gap: float = 0.05,
    max_saturation_floor: float = 0.97,
) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: _int(item, "k", -1)):
        full = _float(row, "condition_a")
        matched = _float(row, "b_match")
        idle = _float(row, "idlekv")
        random_k = _float(row, "random_k", matched)
        oldest_k = _float(row, "oldest_k", matched)
        best_control = max(random_k, oldest_k)
        failures: list[str] = []
        if full < min_full_score:
            failures.append("full_context_not_reliable")
        if full - matched < min_matched_gap:
            failures.append("matched_gap_too_small")
        if idle - matched < min_idlekv_gain:
            failures.append("weak_idlekv_gain")
        if idle - best_control < min_control_gap:
            failures.append("content_agnostic_controls_too_close")
        decisions.append(
            {
                "k": _int(row, "k", -1),
                "full": round(full, 6),
                "matched": round(matched, 6),
                "idlekv": round(idle, 6),
                "best_control": round(best_control, 6),
                "matched_gap": round(full - matched, 6),
                "idlekv_gain": round(idle - matched, 6),
                "idlekv_minus_best_control": round(idle - best_control, 6),
                "failures": failures,
            }
        )

    clean = [item for item in decisions if not item["failures"]]
    saturated = bool(decisions) and min(float(item["idlekv"]) for item in decisions) >= max_saturation_floor
    if not decisions:
        status = "missing_rows"
        action = "rerun_smoke"
    elif saturated:
        status = "appendix_only_saturated"
        action = "do_not_launch_locked_without_budget_redesign"
    elif len(clean) >= 2:
        status = "smoke_pass_run_locked"
        action = "eligible_for_locked_run"
    elif any("full_context_not_reliable" in item["failures"] for item in decisions):
        status = "smoke_fail_redesign_model_or_task"
        action = "fix_ability_before_locked_run"
    else:
        status = "smoke_fail_or_appendix_only"
        action = "do_not_promote"

    return {
        "status": status,
        "action": action,
        "clean_k": len(clean),
        "num_k": len(decisions),
        "saturated": saturated,
        "decisions": decisions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_rows(_read_rows(args.summary_csv))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"[phase16 smoke] {result['status']} action={result['action']}")
        for item in result["decisions"]:
            print(json.dumps(item, sort_keys=True))
    return 0 if result["status"] == "smoke_pass_run_locked" else 2


if __name__ == "__main__":
    raise SystemExit(main())

