"""Summaries for next-turn signal specificity experiments."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


SPECIFICITY_CONDITIONS: tuple[tuple[str, str, str | None], ...] = (
    ("Matched", "b_match_score", "b_match_active_overlap_fraction"),
    ("StaleQ-K", "stale_q_k_score", "stale_q_k_active_overlap_fraction"),
    ("WrongQ-K", "wrong_q_k_score", "wrong_q_k_active_overlap_fraction"),
    ("Refresh-K", "refresh_k_score", "refresh_k_active_overlap_fraction"),
    ("IdleKV", "idlekv_score", "idlekv_active_overlap_fraction"),
    ("Gold-K", "oracle_k_score", "oracle_k_active_overlap_fraction"),
)


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return float(fmean(values)) if values else 0.0


def _rate(values: Iterable[bool]) -> float:
    values = list(values)
    return float(sum(1 for value in values if value) / len(values)) if values else 0.0


def _bootstrap_mean_ci(values: Iterable[float], *, seed: int) -> tuple[float, float]:
    values = [float(value) for value in values]
    if len(values) <= 1:
        mean = _mean(values)
        return mean, mean

    rng = random.Random(seed)
    n = len(values)
    draws = 2000
    means = []
    for _ in range(draws):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * (draws - 1))]
    hi = means[int(0.975 * (draws - 1))]
    return float(lo), float(hi)


def summarize_specificity_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate Phase 6 specificity rows by restore budget and condition."""
    rows = list(payload.get("rows", []))
    config = dict(payload.get("config", {}))
    if not rows:
        return []

    by_k: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_k.setdefault(int(row["k"]), []).append(dict(row))

    out_rows: list[dict[str, Any]] = []
    for k, group in sorted(by_k.items()):
        matched_scores = [float(row["b_match_score"]) for row in group]
        matched_mean = _mean(matched_scores)
        for label, score_key, overlap_key in SPECIFICITY_CONDITIONS:
            if score_key not in group[0]:
                continue
            scores = [float(row[score_key]) for row in group if score_key in row]
            gains = [score - matched for score, matched in zip(scores, matched_scores, strict=False)]
            paired = list(zip(scores, matched_scores, strict=False))
            ci_seed = 1009 + 37 * int(k) + sum(ord(char) for char in label)
            score_ci_low, score_ci_high = _bootstrap_mean_ci(scores, seed=ci_seed)
            gain_ci_low, gain_ci_high = _bootstrap_mean_ci(gains, seed=ci_seed + 17)
            overlaps = (
                [float(row[overlap_key]) for row in group if overlap_key and overlap_key in row]
                if overlap_key
                else []
            )
            out_rows.append(
                {
                    "task": config.get("task", ""),
                    "base_context_budget": int(config.get("base_context_budget", 0)),
                    "k": int(k),
                    "condition": label,
                    "mean_score": round(_mean(scores), 6),
                    "score_ci95_low": round(score_ci_low, 6),
                    "score_ci95_high": round(score_ci_high, 6),
                    "mean_gain_vs_matched": round(_mean(gains), 6),
                    "gain_ci95_low": round(gain_ci_low, 6),
                    "gain_ci95_high": round(gain_ci_high, 6),
                    "win_rate_vs_matched": round(_rate(score > matched for score, matched in paired), 6),
                    "loss_rate_vs_matched": round(_rate(score < matched for score, matched in paired), 6),
                    "mean_overlap_fraction": round(_mean(overlaps), 6) if overlaps else "",
                    "n_rows": len(scores),
                    "num_samples": int(config.get("num_samples", 0)),
                    "query_scoring_mode": config.get("query_scoring_mode", ""),
                    "wrong_query_mode": config.get("wrong_query_mode", ""),
                    "refresh_scope": "buffered_active_plus_evicted" if label == "Refresh-K" else "",
                    "mean_refresh_selected_from_evicted_fraction": round(
                        _mean(row["refresh_selected_from_evicted_fraction"] for row in group if "refresh_selected_from_evicted_fraction" in row),
                        6,
                    )
                    if label == "Refresh-K"
                    else "",
                    "mean_refresh_dropped_base_fraction": round(
                        _mean(row["refresh_dropped_base_fraction"] for row in group if "refresh_dropped_base_fraction" in row),
                        6,
                    )
                    if label == "Refresh-K"
                    else "",
                    "mean_refresh_jaccard_with_b_match": round(
                        _mean(row["refresh_jaccard_with_b_match"] for row in group if "refresh_jaccard_with_b_match" in row),
                        6,
                    )
                    if label == "Refresh-K"
                    else "",
                    "mean_refresh_jaccard_with_idlekv": round(
                        _mean(row["refresh_jaccard_with_idlekv"] for row in group if "refresh_jaccard_with_idlekv" in row),
                        6,
                    )
                    if label == "Refresh-K"
                    else "",
                    "artifact_path": payload.get("artifact_path", ""),
                }
            )
    return out_rows


def summarize_specificity_artifact(path: str | Path) -> list[dict[str, Any]]:
    """Load a Phase 6 artifact and summarize specificity conditions."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return summarize_specificity_payload(payload)


def write_specificity_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write specificity summary rows to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "base_context_budget",
        "k",
        "condition",
        "mean_score",
        "score_ci95_low",
        "score_ci95_high",
        "mean_gain_vs_matched",
        "gain_ci95_low",
        "gain_ci95_high",
        "win_rate_vs_matched",
        "loss_rate_vs_matched",
        "mean_overlap_fraction",
        "n_rows",
        "num_samples",
        "query_scoring_mode",
        "wrong_query_mode",
        "refresh_scope",
        "mean_refresh_selected_from_evicted_fraction",
        "mean_refresh_dropped_base_fraction",
        "mean_refresh_jaccard_with_b_match",
        "mean_refresh_jaccard_with_idlekv",
        "artifact_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def recommend_specificity_followup(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Classify each K setting and recommend the next specificity action."""
    by_k: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_k.setdefault(int(row["k"]), {})[str(row["condition"])] = dict(row)

    recommendations: list[dict[str, Any]] = []
    for k, condition_rows in sorted(by_k.items()):
        scores = {
            condition: float(row["mean_score"])
            for condition, row in condition_rows.items()
            if row.get("mean_score") not in ("", None)
        }
        matched = scores.get("Matched", 0.0)
        stale = scores.get("StaleQ-K", matched)
        wrong = scores.get("WrongQ-K", matched)
        refresh = scores.get("Refresh-K", matched)
        idlekv = scores.get("IdleKV", matched)
        gold = scores.get("Gold-K", idlekv)

        idle_vs_matched = idlekv - matched
        idle_vs_stale = idlekv - stale
        idle_vs_wrong = idlekv - wrong
        refresh_vs_idle = refresh - idlekv
        gold_headroom = gold - idlekv
        saturated = idlekv >= 0.95 and refresh >= 0.95 and gold >= 0.95
        idle_row = condition_rows.get("IdleKV", {})
        idle_gain_ci_low = float(idle_row.get("gain_ci95_low", idle_vs_matched))
        idle_win_rate = float(idle_row.get("win_rate_vs_matched", 1.0 if idle_vs_matched > 0 else 0.0))

        if idle_vs_matched < 0.15 or idle_gain_ci_low <= 0.0:
            action = "rerun_lower_budget_or_drop_panel"
        elif idle_vs_stale < 0.10:
            action = "demote_specificity_no_stale_separation"
        elif idle_vs_wrong < 0.10:
            action = "inspect_donor_control_but_do_not_block"
        elif saturated:
            action = "rerun_lower_k_before_promotion"
        elif idle_win_rate < 0.55:
            action = "inspect_paired_instability_before_promotion"
        elif refresh_vs_idle > 0.05:
            action = "promote_only_as_low_recompute_repair_if_costs_reported"
        elif idle_vs_matched >= 0.15:
            action = "promote_locked_specificity_run"
        else:
            action = "rerun_lower_budget_or_drop_panel"

        recommendations.append(
            {
                "k": int(k),
                "action": action,
                "idle_vs_matched": round(idle_vs_matched, 6),
                "idle_vs_stale": round(idle_vs_stale, 6),
                "idle_vs_wrong": round(idle_vs_wrong, 6),
                "refresh_vs_idle": round(refresh_vs_idle, 6),
                "gold_headroom": round(gold_headroom, 6),
                "idle_gain_ci95_low": round(idle_gain_ci_low, 6),
                "idle_win_rate_vs_matched": round(idle_win_rate, 6),
                "saturated": saturated,
            }
        )
    return recommendations
