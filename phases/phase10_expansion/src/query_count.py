"""Promotion gates for query-count breadth experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


QUERY_COUNT_BY_TASK = {
    "mq_niah_2q_clean_suite": 2,
    "mq_niah_3q_clean_suite": 3,
    "clean_suite": 4,
    "mq_niah_6q_clean_suite": 6,
    "mq_niah_8q_clean_suite": 8,
}


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def load_query_count_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load a query-count summary CSV."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate_query_count_breadth(
    rows: list[dict[str, Any]],
    *,
    min_full_score: float = 0.90,
    min_full_vs_matched_gap: float = 0.20,
    min_idle_gain: float = 0.15,
    max_control_lift: float = 0.05,
) -> list[dict[str, Any]]:
    """Evaluate appendix/main promotion gates for breadth rows.

    The input is the compact artifact summary emitted by
    ``phase9_artifact_summary.py``. Each output row is one task with the
    best IdleKV K selected by score gain over matched no-repair.
    """
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row.get("task", "")), []).append(dict(row))

    out: list[dict[str, Any]] = []
    for task, task_rows in sorted(by_task.items(), key=lambda item: QUERY_COUNT_BY_TASK.get(item[0], 999)):
        scored_rows: list[dict[str, Any]] = []
        for row in task_rows:
            matched = _float(row, "b_match")
            idlekv = _float(row, "idlekv")
            full = _float(row, "condition_a")
            gold = _float(row, "gold_k", idlekv)
            random_k = _float(row, "random_k", matched)
            oldest_k = _float(row, "oldest_k", matched)
            idle_gain = idlekv - matched
            row = dict(row)
            row["_full"] = full
            row["_matched"] = matched
            row["_idlekv"] = idlekv
            row["_gold"] = gold
            row["_idle_gain"] = idle_gain
            row["_full_gap"] = full - matched
            row["_max_control_lift"] = max(random_k - matched, oldest_k - matched)
            scored_rows.append(row)

        if not scored_rows:
            continue
        best = max(scored_rows, key=lambda row: (row["_idle_gain"], int(row.get("k", 0))))
        full_ok = best["_full"] >= min_full_score
        gap_ok = best["_full_gap"] >= min_full_vs_matched_gap
        gain_ok = best["_idle_gain"] >= min_idle_gain
        controls_ok = best["_max_control_lift"] <= max_control_lift
        gold_ok = best["_gold"] + 1e-9 >= best["_idlekv"]
        appendix_ok = all([full_ok, gap_ok, gain_ok, controls_ok, gold_ok])
        query_count = QUERY_COUNT_BY_TASK.get(task, 0)
        main_ok = appendix_ok and query_count in {3, 8} and best["_idle_gain"] >= 0.25
        out.append(
            {
                "task": task,
                "query_count": query_count,
                "best_k": int(best.get("k", 0)),
                "full_score": round(best["_full"], 6),
                "matched_score": round(best["_matched"], 6),
                "idlekv_score": round(best["_idlekv"], 6),
                "gold_score": round(best["_gold"], 6),
                "idlekv_gain": round(best["_idle_gain"], 6),
                "full_vs_matched_gap": round(best["_full_gap"], 6),
                "max_control_lift": round(best["_max_control_lift"], 6),
                "appendix_ok": appendix_ok,
                "main_ok": main_ok,
                "action": "promote_main_candidate"
                if main_ok
                else "appendix_breadth"
                if appendix_ok
                else "do_not_promote",
            }
        )
    return out
