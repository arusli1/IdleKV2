"""Gates for first-stage retention-rule breadth smokes."""

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


def load_compressor_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load a compressor smoke summary CSV."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate_compressor_smoke(
    rows: list[dict[str, Any]],
    *,
    min_points: int = 3,
    min_full_score: float = 0.85,
    min_full_vs_matched_gap: float = 0.15,
    min_idle_gain: float = 0.10,
    max_control_lift: float = 0.08,
) -> tuple[dict[str, Any], ...]:
    """Classify each base-budget group in a compressor smoke.

    The smoke is only a gate for a larger locked run. A budget passes when
    the full-cache reference is capable, matched no-repair leaves room,
    IdleKV improves meaningfully, content-agnostic controls stay near
    matched, and the benchmark-metadata reference covers IdleKV.
    """
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("task", "")), _int(row, "base_context_budget")), []).append(dict(row))

    recommendations: list[dict[str, Any]] = []
    for (task, budget), group_rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        scored: list[dict[str, float | int]] = []
        for row in group_rows:
            matched = _float(row, "b_match")
            idlekv = _float(row, "idlekv")
            full = _float(row, "condition_a")
            gold = _float(row, "gold_k", _float(row, "oracle_k", idlekv))
            random_k = _float(row, "random_k", matched)
            oldest_k = _float(row, "oldest_k", matched)
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
        if not scored:
            continue
        scored.sort(key=lambda row: int(row["k"]))
        best = max(scored, key=lambda row: (float(row["idle_gain"]), int(row["k"])))
        full_ok = max(float(row["full"]) for row in scored) >= min_full_score
        gap_ok = max(float(row["full_gap"]) for row in scored) >= min_full_vs_matched_gap
        gain_ok = float(best["idle_gain"]) >= min_idle_gain
        controls_ok = max(float(row["control_lift"]) for row in scored) <= max_control_lift
        gold_ok = all(float(row["gold"]) + 1e-9 >= float(row["idlekv"]) for row in scored)
        enough_points = len(scored) >= min_points
        lock_followup = all([enough_points, full_ok, gap_ok, gain_ok, controls_ok, gold_ok])
        if lock_followup:
            action = "lock_followup"
        elif not gain_ok:
            action = "do_not_lock_low_gain"
        elif not controls_ok:
            action = "do_not_lock_controls_explain_gain"
        elif not gap_ok:
            action = "do_not_lock_saturated_or_weak_gap"
        else:
            action = "do_not_lock_smoke_failed"
        recommendations.append(
            {
                "task": task,
                "base_context_budget": int(budget),
                "points": len(scored),
                "best_k": int(best["k"]),
                "best_gain": round(float(best["idle_gain"]), 6),
                "full_vs_matched_gap": round(max(float(row["full_gap"]) for row in scored), 6),
                "max_control_lift": round(max(float(row["control_lift"]) for row in scored), 6),
                "gold_covers_idle": gold_ok,
                "lock_followup": lock_followup,
                "action": action,
            }
        )
    return tuple(recommendations)
