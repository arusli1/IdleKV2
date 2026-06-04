"""Gates for cross-model repair checks."""

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


def load_model_transfer_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load a model-transfer summary CSV."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate_model_transfer_rows(
    rows: list[dict[str, Any]],
    *,
    min_points: int = 2,
    min_full_score: float = 0.90,
    min_full_vs_matched_gap: float = 0.15,
    min_idle_gain: float = 0.15,
    max_control_lift: float = 0.10,
) -> tuple[dict[str, Any], ...]:
    """Classify each base-budget group in a cross-model repair run.

    A budget is appendix-ready when the new model can solve the task with
    the full cache, matched no-repair leaves headroom, IdleKV improves by a
    meaningful margin, content-agnostic restore controls do not explain the
    gain, and the benchmark-metadata reference covers IdleKV.
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
        k96_rows = [row for row in scored if int(row["k"]) == 96]
        k96_gain = float(k96_rows[0]["idle_gain"]) if k96_rows else float(best["idle_gain"])
        full_ok = max(float(row["full"]) for row in scored) >= min_full_score
        gap_ok = max(float(row["full_gap"]) for row in scored) >= min_full_vs_matched_gap
        gain_ok = float(best["idle_gain"]) >= min_idle_gain
        controls_ok = max(float(row["control_lift"]) for row in scored) <= max_control_lift
        gold_ok = all(float(row["gold"]) + 1e-9 >= float(row["idlekv"]) for row in scored)
        enough_points = len(scored) >= min_points
        appendix_candidate = all([enough_points, full_ok, gap_ok, gain_ok, controls_ok, gold_ok])

        if appendix_candidate:
            action = "appendix_candidate"
        elif not full_ok:
            action = "reject_model_cannot_solve_full_cache"
        elif not gap_ok:
            action = "reject_saturated_or_weak_gap"
        elif not gain_ok:
            action = "reject_low_idle_gain"
        elif not controls_ok:
            action = "reject_controls_explain_gain"
        else:
            action = "reject_artifact_check_failed"

        recommendations.append(
            {
                "task": task,
                "base_context_budget": int(budget),
                "points": len(scored),
                "best_k": int(best["k"]),
                "best_gain": round(float(best["idle_gain"]), 6),
                "k96_gain": round(k96_gain, 6),
                "full_vs_matched_gap": round(max(float(row["full_gap"]) for row in scored), 6),
                "max_control_lift": round(max(float(row["control_lift"]) for row in scored), 6),
                "gold_covers_idle": gold_ok,
                "appendix_candidate": appendix_candidate,
                "action": action,
            }
        )
    return tuple(recommendations)
