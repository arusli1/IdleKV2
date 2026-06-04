"""Promotion gates for Phase 11 main-candidate robustness runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return int(float(value))


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load a Phase 11 summary CSV."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate_main_candidate(
    rows: list[dict[str, Any]],
    *,
    min_points: int = 5,
    min_full_score: float = 0.90,
    min_full_vs_matched_gap: float = 0.20,
    min_idle_gain: float = 0.15,
    max_control_lift_at_gate: float = 0.10,
    saturation_score: float = 0.98,
) -> tuple[dict[str, Any], ...]:
    """Classify whether a full-grid robustness run can enter the main paper.

    The gate is intentionally stricter than appendix-readiness: it requires a
    real curve, not just saturated positive endpoints.
    """
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("task", "")), _int(row, "base_context_budget")), []).append(dict(row))

    decisions: list[dict[str, Any]] = []
    for (task, budget), group_rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        scored: list[dict[str, float | int]] = []
        for row in group_rows:
            matched = _float(row, "b_match")
            idlekv = _float(row, "idlekv")
            random_k = _float(row, "random_k", matched)
            oldest_k = _float(row, "oldest_k", matched)
            full = _float(row, "condition_a")
            gold = _float(row, "gold_k", _float(row, "oracle_k", idlekv))
            scored.append(
                {
                    "k": _int(row, "k"),
                    "full": full,
                    "matched": matched,
                    "idlekv": idlekv,
                    "gold": gold,
                    "idle_gain": idlekv - matched,
                    "full_gap": full - matched,
                    "control_lift": max(random_k - matched, oldest_k - matched),
                }
            )
        scored.sort(key=lambda row: int(row["k"]))
        if not scored:
            continue

        eligible = [
            row
            for row in scored
            if float(row["idle_gain"]) >= min_idle_gain
            and float(row["control_lift"]) <= max_control_lift_at_gate
            and float(row["gold"]) + 1e-9 >= float(row["idlekv"])
        ]
        best = max(scored, key=lambda row: (float(row["idle_gain"]), int(row["k"])))
        best_eligible = max(eligible, key=lambda row: (float(row["idle_gain"]), int(row["k"]))) if eligible else None
        full_ok = max(float(row["full"]) for row in scored) >= min_full_score
        gap_ok = max(float(row["full_gap"]) for row in scored) >= min_full_vs_matched_gap
        enough_points = len(scored) >= min_points
        non_saturated = any(
            float(row["idlekv"]) < saturation_score and float(row["idle_gain"]) >= min_idle_gain for row in scored
        )
        main_candidate = all([full_ok, gap_ok, enough_points, bool(best_eligible), non_saturated])

        if main_candidate:
            action = "main_candidate"
        elif not enough_points:
            action = "appendix_only_too_few_points"
        elif not full_ok:
            action = "reject_full_cache_weak"
        elif not gap_ok:
            action = "appendix_only_saturated_or_weak_gap"
        elif not best_eligible:
            action = "appendix_only_controls_or_gold_gate_failed"
        elif not non_saturated:
            action = "appendix_only_saturated_curve"
        else:
            action = "appendix_only"

        decisions.append(
            {
                "task": task,
                "base_context_budget": budget,
                "points": len(scored),
                "best_k": int(best["k"]),
                "best_gain": round(float(best["idle_gain"]), 6),
                "best_eligible_k": int(best_eligible["k"]) if best_eligible else None,
                "best_eligible_gain": round(float(best_eligible["idle_gain"]), 6) if best_eligible else 0.0,
                "full_vs_matched_gap": round(max(float(row["full_gap"]) for row in scored), 6),
                "non_saturated": non_saturated,
                "main_candidate": main_candidate,
                "action": action,
            }
        )
    return tuple(decisions)

