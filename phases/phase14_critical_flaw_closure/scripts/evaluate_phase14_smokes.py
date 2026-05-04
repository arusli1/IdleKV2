#!/usr/bin/env python3
"""Evaluate Phase 14 smoke outputs against promotion gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


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


def _by_k_condition(rows: Iterable[dict[str, str]]) -> dict[int, dict[str, dict[str, str]]]:
    grouped: dict[int, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(_int(row, "k", -1), {})[row.get("condition", "")] = row
    return {k: value for k, value in grouped.items() if k >= 0}


def evaluate_refresh_rows(
    rows: list[dict[str, str]],
    *,
    min_idle_gain: float = 0.15,
    min_stale_gap: float = 0.10,
    min_wrong_gap: float = 0.05,
    refresh_boundary_gap: float = 0.05,
) -> dict[str, object]:
    """Classify a Refresh-K frontier smoke.

    A Refresh-K win is not a failed run. It means the paper should call IdleKV
    an incremental repair operator and describe Refresh-K as full-budget
    reselection headroom.
    """
    decisions: list[dict[str, object]] = []
    for k, by_condition in sorted(_by_k_condition(rows).items()):
        matched = _float(by_condition.get("Matched", {}), "mean_score")
        stale = _float(by_condition.get("StaleQ-K", {}), "mean_score", matched)
        wrong = _float(by_condition.get("WrongQ-K", {}), "mean_score", matched)
        refresh = _float(by_condition.get("Refresh-K", {}), "mean_score", matched)
        idle = _float(by_condition.get("IdleKV", {}), "mean_score", matched)
        gold = _float(by_condition.get("Gold-K", {}), "mean_score", idle)

        idle_gain = idle - matched
        idle_vs_stale = idle - stale
        idle_vs_wrong = idle - wrong
        refresh_vs_idle = refresh - idle
        failures: list[str] = []
        warnings: list[str] = []
        if idle_gain < min_idle_gain:
            failures.append("weak_idlekv_gain")
        if idle_vs_stale < min_stale_gap:
            failures.append("weak_stale_query_separation")
        if idle_vs_wrong < min_wrong_gap:
            failures.append("weak_donor_query_separation")
        if gold + 1e-9 < idle:
            warnings.append("gold_span_reference_below_idlekv")
        if refresh_vs_idle > refresh_boundary_gap:
            failures.append("refresh_k_is_stronger")

        if "weak_idlekv_gain" in failures:
            action = "drop_or_redesign_specificity_cell"
        elif "refresh_k_is_stronger" in failures:
            action = "frame_as_incremental_repair_boundary"
        elif failures:
            action = "appendix_or_redesign_controls"
        else:
            action = "candidate_specificity_cell"

        decisions.append(
            {
                "k": k,
                "matched": round(matched, 6),
                "idlekv": round(idle, 6),
                "stale_q": round(stale, 6),
                "wrong_q": round(wrong, 6),
                "refresh_k": round(refresh, 6),
                "gold_k": round(gold, 6),
                "idlekv_gain": round(idle_gain, 6),
                "idle_vs_stale": round(idle_vs_stale, 6),
                "idle_vs_wrong": round(idle_vs_wrong, 6),
                "refresh_vs_idle": round(refresh_vs_idle, 6),
                "gold_headroom": round(gold - idle, 6),
                "action": action,
                "failures": failures,
                "warnings": warnings,
            }
        )

    refresh_dominates = sum("refresh_k_is_stronger" in item["failures"] for item in decisions)
    useful_idle = sum(item["idlekv_gain"] >= min_idle_gain for item in decisions)
    clean_specificity = sum(item["action"] == "candidate_specificity_cell" for item in decisions)
    if not decisions:
        status = "missing_refresh_rows"
    elif useful_idle == 0:
        status = "refresh_smoke_fail_redesign_task"
    elif refresh_dominates >= max(1, (len(decisions) + 1) // 2):
        status = "refresh_boundary_confirmed"
    elif clean_specificity >= 2:
        status = "idlekv_specificity_frontier_candidate"
    else:
        status = "specificity_frontier_needs_redesign"

    return {
        "status": status,
        "num_k": len(decisions),
        "refresh_dominates_k": refresh_dominates,
        "useful_idlekv_k": useful_idle,
        "clean_specificity_k": clean_specificity,
        "decisions": decisions,
    }


def evaluate_llama_rows(
    rows: list[dict[str, str]],
    *,
    min_full_score: float = 0.90,
    min_matched_gap: float = 0.20,
    min_idlekv_gain: float = 0.10,
    min_control_gap: float = 0.05,
) -> dict[str, object]:
    decisions: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: _int(item, "k", -1)):
        full = _float(row, "condition_a")
        matched = _float(row, "b_match")
        idle = _float(row, "idlekv")
        random_k = _float(row, "random_k", matched)
        oldest_k = _float(row, "oldest_k", matched)
        gold = _float(row, "gold_k", idle)
        best_control = max(random_k, oldest_k)
        failures: list[str] = []
        warnings: list[str] = []
        if full < min_full_score:
            failures.append("full_context_not_reliable")
        if full - matched < min_matched_gap:
            failures.append("matched_gap_too_small")
        if idle - matched < min_idlekv_gain:
            failures.append("weak_idlekv_gain")
        if idle - best_control < min_control_gap:
            failures.append("content_agnostic_controls_too_close")
        if gold + 1e-9 < idle:
            warnings.append("gold_span_reference_below_idlekv")
        decisions.append(
            {
                "k": _int(row, "k", -1),
                "full": round(full, 6),
                "matched": round(matched, 6),
                "idlekv": round(idle, 6),
                "best_control": round(best_control, 6),
                "gold_k": round(gold, 6),
                "matched_gap": round(full - matched, 6),
                "idlekv_gain": round(idle - matched, 6),
                "idlekv_minus_best_control": round(idle - best_control, 6),
                "failures": failures,
                "warnings": warnings,
            }
        )

    useful = sum(not item["failures"] for item in decisions)
    saturated = bool(decisions) and min(float(item["idlekv"]) for item in decisions) >= 0.95
    n = max((_int(row, "num_samples", 0) for row in rows), default=0)
    if not decisions:
        status = "missing_llama_rows"
    elif saturated:
        status = "appendix_portability_only_saturated"
    elif useful >= 2:
        status = "llama_smoke_pass_run_locked"
    else:
        status = "llama_smoke_fail_redesign_budget_or_task"
    return {
        "status": status,
        "num_samples": n,
        "useful_k": useful,
        "saturated": saturated,
        "decisions": decisions,
    }


def evaluate_selector_rows(
    rows: list[dict[str, str]],
    *,
    mid_k: int = 48,
    high_k: int = 96,
    min_mid_gain: float = 0.05,
    max_high_loss: float = 0.02,
) -> dict[str, object]:
    by_k = {_int(row, "k", -1): row for row in rows}
    variants = {
        "coverage": "idlekv_coverage",
        "mmr": "idlekv_mmr",
    }
    decisions: list[dict[str, object]] = []
    for label, column in variants.items():
        mid = by_k.get(mid_k)
        high = by_k.get(high_k)
        failures: list[str] = []
        if mid is None or high is None:
            failures.append("missing_mid_or_high_k")
            decisions.append({"variant": label, "status": "missing", "failures": failures})
            continue
        if column not in mid or column not in high:
            failures.append("missing_variant_metric")
            decisions.append({"variant": label, "status": "missing", "failures": failures})
            continue
        mid_gain = _float(mid, column) - _float(mid, "idlekv")
        high_loss = _float(high, "idlekv") - _float(high, column)
        gold_gap = max(0.0, _float(mid, "gold_k", _float(mid, "idlekv")) - _float(mid, "idlekv"))
        gap_closure = mid_gain / gold_gap if gold_gap > 1e-12 else 0.0
        if mid_gain < min_mid_gain:
            failures.append("mid_k_gain_below_gate")
        if high_loss > max_high_loss:
            failures.append("high_k_loss_too_large")
        status = "selector_variant_candidate" if not failures else "selector_variant_reject"
        decisions.append(
            {
                "variant": label,
                "status": status,
                "mid_k": mid_k,
                "high_k": high_k,
                "mid_gain_vs_idlekv": round(mid_gain, 6),
                "high_loss_vs_idlekv": round(high_loss, 6),
                "mid_gold_gap_closure": round(gap_closure, 6),
                "failures": failures,
            }
        )
    passed = [item for item in decisions if item.get("status") == "selector_variant_candidate"]
    return {
        "status": "selector_smoke_pass" if passed else "selector_smoke_reject",
        "candidates": [item["variant"] for item in passed],
        "decisions": decisions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--kind", choices=("refresh", "llama", "selector"), required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = _read_rows(args.summary_csv)
    if args.kind == "refresh":
        result = evaluate_refresh_rows(rows)
        pass_statuses = {"refresh_boundary_confirmed", "idlekv_specificity_frontier_candidate"}
    elif args.kind == "llama":
        result = evaluate_llama_rows(rows)
        pass_statuses = {"llama_smoke_pass_run_locked"}
    else:
        result = evaluate_selector_rows(rows)
        pass_statuses = {"selector_smoke_pass"}

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"[phase14 {args.kind} smoke] {result['status']}")
        for item in result.get("decisions", []):
            print(json.dumps(item, sort_keys=True))
    return 0 if result["status"] in pass_statuses else 2


if __name__ == "__main__":
    raise SystemExit(main())
