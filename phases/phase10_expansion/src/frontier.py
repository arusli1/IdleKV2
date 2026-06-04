"""Promotion gates for full restore-budget frontier sweeps."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .query_count import QUERY_COUNT_BY_TASK


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def load_frontier_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load a Phase 9/10 frontier summary CSV."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate_frontier_promotion(
    rows: list[dict[str, Any]],
    *,
    min_points: int = 5,
    min_gain: float = 0.25,
    max_control_lift: float = 0.06,
    max_large_drops: int = 1,
    drop_tolerance: float = 0.08,
    easy_boundary_query_count: int = 2,
) -> dict[str, Any]:
    """Evaluate whether a full K-grid frontier is main-paper safe.

    This gate is intentionally conservative but not brittle. It checks
    that the curve has enough K points, that IdleKV has a meaningful
    best gain over matched no-repair, that random/oldest restores do not
    explain the effect, and that the frontier is broadly increasing
    rather than a noisy endpoint-only result.
    """
    if not rows:
        return {
            "task": "",
            "query_count": 0,
            "points": 0,
            "best_k": 0,
            "best_gain": 0.0,
            "max_control_lift": 0.0,
            "large_drops": 0,
            "gold_covers_idle": False,
            "promote": False,
            "action": "do_not_promote_empty",
        }

    task = str(rows[0].get("task", ""))
    query_count = QUERY_COUNT_BY_TASK.get(task, 0)
    scored: list[dict[str, float | int]] = []
    for row in rows:
        k = int(float(row.get("k", 0)))
        matched = _float(row, "b_match")
        idlekv = _float(row, "idlekv")
        random_k = _float(row, "random_k", matched)
        oldest_k = _float(row, "oldest_k", matched)
        gold = _float(row, "gold_k", _float(row, "oracle_k", idlekv))
        scored.append(
            {
                "k": k,
                "matched": matched,
                "idlekv": idlekv,
                "gold": gold,
                "gain": idlekv - matched,
                "control_lift": max(random_k - matched, oldest_k - matched),
            }
        )
    scored.sort(key=lambda row: int(row["k"]))

    gains = [float(row["gain"]) for row in scored]
    best = max(scored, key=lambda row: (float(row["gain"]), int(row["k"])))
    large_drops = sum(
        1
        for previous, current in zip(gains, gains[1:], strict=False)
        if current + drop_tolerance < previous
    )
    max_control = max(float(row["control_lift"]) for row in scored)
    gold_covers_idle = all(float(row["gold"]) + 1e-9 >= float(row["idlekv"]) for row in scored)
    enough_points = len(scored) >= min_points
    gain_ok = float(best["gain"]) >= min_gain
    controls_ok = max_control <= max_control_lift
    shape_ok = large_drops <= max_large_drops
    mechanical_promote = all([enough_points, gain_ok, controls_ok, shape_ok, gold_covers_idle])
    boundary_review = mechanical_promote and 0 < query_count <= easy_boundary_query_count
    promote = mechanical_promote and not boundary_review
    if promote:
        action = "promote_main_frontier"
    elif boundary_review:
        action = "review_easy_boundary_frontier"
    else:
        action = "do_not_promote_frontier"

    return {
        "task": task,
        "query_count": query_count,
        "points": len(scored),
        "best_k": int(best["k"]),
        "best_gain": round(float(best["gain"]), 6),
        "max_control_lift": round(max_control, 6),
        "large_drops": large_drops,
        "gold_covers_idle": gold_covers_idle,
        "boundary_review": boundary_review,
        "promote": promote,
        "action": action,
    }
