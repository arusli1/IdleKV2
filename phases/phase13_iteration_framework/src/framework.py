"""Reusable gates for iterative experiment closure.

These helpers are intentionally small and CPU-only. They do not decide paper
truth by themselves; they make the smoke/full-run promotion criteria explicit
so failed runs can be diagnosed before redesigning the next experiment.
"""

from __future__ import annotations

from collections.abc import Iterable
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


def _bool(row: dict[str, Any], key: str, default: bool = False) -> bool:
    value = row.get(key, default)
    if isinstance(value, bool):
        return value
    if value in ("", None):
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def consecutive_positive_budget_count(
    rows: Iterable[dict[str, Any]],
    *,
    gain_key: str = "idle_gain",
    min_gain: float = 0.15,
) -> int:
    """Return the longest adjacent positive run in the observed K grid."""

    ordered = sorted((_int(row, "k"), _float(row, gain_key)) for row in rows)
    best = 0
    current = 0
    previous_index: int | None = None
    for index, (_k, gain) in enumerate(ordered):
        if gain >= min_gain and (previous_index is None or index == previous_index + 1):
            current += 1
        elif gain >= min_gain:
            current = 1
        else:
            current = 0
        best = max(best, current)
        previous_index = index if gain >= min_gain else None
    return best


def classify_result_rigor(
    spec: dict[str, Any],
    *,
    min_main_n: int = 12,
    min_main_grid_points: int = 5,
    min_effect_size: float = 0.15,
) -> dict[str, Any]:
    """Classify whether a result is rigorous enough for a main-paper claim.

    This gate is deliberately orthogonal to whether the numerical result is
    positive. A strong smoke can justify a locked run, but it cannot justify a
    main figure by itself.
    """

    failures: list[str] = []
    stage = str(spec.get("stage", "full")).strip().lower()
    is_smoke = _bool(spec, "is_smoke", default=stage == "smoke")
    n = _int(spec, "n", 0)
    grid_points = _int(spec, "grid_points", _int(spec, "points", 0))
    effect_size = _float(spec, "effect_size", _float(spec, "best_gain", 0.0))
    primary_claim = _bool(spec, "primary_claim", False)

    if is_smoke or stage == "smoke":
        failures.append("smoke_only")
    if n < min_main_n:
        failures.append("too_few_examples")
    if grid_points < min_main_grid_points:
        failures.append("too_few_budget_points")
    if not _bool(spec, "paired_or_shared_examples", True):
        failures.append("unpaired_examples")
    if not _bool(spec, "matched_budget_audited", False):
        failures.append("matched_budget_not_audited")
    if not _bool(spec, "full_cache_ok", False):
        failures.append("full_cache_weak")
    if not _bool(spec, "controls_clean", False):
        failures.append("controls_not_clean")
    if not _bool(spec, "non_saturated", True):
        failures.append("saturated")
    if effect_size < min_effect_size:
        failures.append("small_effect")
    if not _bool(spec, "confound_checked", False):
        failures.append("confounds_not_checked")
    if primary_claim and not _bool(spec, "paired_uncertainty_reported", False):
        failures.append("paired_uncertainty_missing")
    if (
        primary_claim
        and _bool(spec, "paired_uncertainty_reported", False)
        and not _bool(spec, "paired_uncertainty_positive", True)
    ):
        failures.append("paired_uncertainty_not_positive")

    if not failures:
        action = "main_ready_result"
        main_ready = True
    elif "smoke_only" in failures:
        action = "run_locked_full_before_main"
        main_ready = False
    elif "matched_budget_not_audited" in failures or "unpaired_examples" in failures:
        action = "rerun_or_audit_accounting"
        main_ready = False
    elif "paired_uncertainty_missing" in failures:
        action = "add_paired_uncertainty_or_demote"
        main_ready = False
    elif "paired_uncertainty_not_positive" in failures:
        action = "demote_or_rerun_for_uncertainty"
        main_ready = False
    elif "full_cache_weak" in failures:
        action = "reject_or_redesign_task"
        main_ready = False
    elif "controls_not_clean" in failures:
        action = "redesign_controls_or_appendix"
        main_ready = False
    elif "saturated" in failures or "small_effect" in failures:
        action = "appendix_or_harder_task"
        main_ready = False
    else:
        action = "rerun_for_rigor"
        main_ready = False

    return {
        "action": action,
        "main_ready": main_ready,
        "failures": tuple(failures),
        "n": n,
        "grid_points": grid_points,
        "effect_size": round(effect_size, 6),
        "primary_claim": primary_claim,
    }


def classify_figure_quality(
    spec: dict[str, Any],
    *,
    min_data_points: int = 10,
) -> dict[str, Any]:
    """Classify whether a figure is publication-ready for the main paper."""

    failures: list[str] = []
    data_points = _int(spec, "data_points", 0)
    if not _bool(spec, "real_data", False) or not _bool(spec, "no_fake_data", True):
        failures.append("not_real_data")
    if data_points < min_data_points:
        failures.append("low_data_density")
    if not _bool(spec, "graph_type_fits_claim", False):
        failures.append("wrong_graph_type")
    if not _bool(spec, "one_column_fit", True):
        failures.append("layout_not_one_column")
    if not _bool(spec, "legend_outside_data", False):
        failures.append("legend_or_labels_overlap_data")
    if not _bool(spec, "labels_readable", False):
        failures.append("unreadable_labels")
    if not _bool(spec, "caption_scopes_claim", False):
        failures.append("caption_overclaims_or_underdefines")
    if not _bool(spec, "controls_visible", False):
        failures.append("missing_controls")
    if not _bool(spec, "not_redundant", True):
        failures.append("redundant_with_existing_figure")
    if not _bool(spec, "top_paper_style", False):
        failures.append("below_style_bar")

    if not failures:
        action = "main_ready_figure"
        main_ready = True
    elif "not_real_data" in failures:
        action = "reject_until_real_data"
        main_ready = False
    elif "wrong_graph_type" in failures or "low_data_density" in failures:
        action = "redesign_graph_type"
        main_ready = False
    elif "layout_not_one_column" in failures or "legend_or_labels_overlap_data" in failures or "unreadable_labels" in failures:
        action = "revise_layout"
        main_ready = False
    elif "caption_overclaims_or_underdefines" in failures:
        action = "revise_caption_or_terminology"
        main_ready = False
    elif "redundant_with_existing_figure" in failures:
        action = "move_to_appendix_or_drop"
        main_ready = False
    else:
        action = "revise_before_main"
        main_ready = False

    return {
        "action": action,
        "main_ready": main_ready,
        "failures": tuple(failures),
        "data_points": data_points,
    }


def classify_policy_curve(
    rows: Iterable[dict[str, Any]],
    *,
    min_points: int = 5,
    min_full_score: float = 0.90,
    min_full_vs_matched_gap: float = 0.20,
    min_idle_gain: float = 0.15,
    max_control_lift: float = 0.10,
    min_adjacent_positive: int = 2,
    saturation_score: float = 0.98,
) -> dict[str, Any]:
    """Classify a full-grid first-stage-policy robustness curve."""

    scored: list[dict[str, float | int]] = []
    for row in rows:
        matched = _float(row, "b_match")
        idlekv = _float(row, "idlekv")
        random_k = _float(row, "random_k", matched)
        oldest_k = _float(row, "oldest_k", matched)
        full = _float(row, "condition_a", _float(row, "full", 0.0))
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
        return {"action": "missing_data", "main_candidate": False, "reason": "no_rows"}

    eligible = [
        row
        for row in scored
        if float(row["idle_gain"]) >= min_idle_gain
        and float(row["control_lift"]) <= max_control_lift
        and float(row["gold"]) + 1e-9 >= float(row["idlekv"])
    ]
    full_ok = max(float(row["full"]) for row in scored) >= min_full_score
    gap_ok = max(float(row["full_gap"]) for row in scored) >= min_full_vs_matched_gap
    enough_points = len(scored) >= min_points
    adjacent = consecutive_positive_budget_count(eligible, min_gain=min_idle_gain)
    non_saturated = any(
        float(row["idlekv"]) < saturation_score and float(row["idle_gain"]) >= min_idle_gain for row in scored
    )
    main_candidate = all(
        [
            enough_points,
            full_ok,
            gap_ok,
            bool(eligible),
            adjacent >= min_adjacent_positive,
            non_saturated,
        ]
    )

    if main_candidate:
        action = "main_candidate"
    elif not enough_points:
        action = "smoke_or_appendix_too_few_points"
    elif not full_ok:
        action = "reject_full_cache_weak"
    elif not gap_ok:
        action = "reject_no_repair_gap"
    elif not eligible:
        action = "appendix_or_redesign_controls_or_gold_failed"
    elif adjacent < min_adjacent_positive:
        action = "appendix_only_endpoint_positive"
    elif not non_saturated:
        action = "appendix_only_saturated"
    else:
        action = "appendix_only"

    best = max(scored, key=lambda row: (float(row["idle_gain"]), int(row["k"])))
    return {
        "action": action,
        "main_candidate": main_candidate,
        "points": len(scored),
        "best_k": int(best["k"]),
        "best_gain": round(float(best["idle_gain"]), 6),
        "eligible_points": len(eligible),
        "adjacent_positive": adjacent,
        "full_ok": full_ok,
        "gap_ok": gap_ok,
        "non_saturated": non_saturated,
    }


def classify_multiturn_candidate(
    rows: Iterable[dict[str, Any]],
    *,
    min_idle_noninitial_gain: float = 0.35,
    min_revisit_gain: float = 0.75,
    max_control_noninitial_gain: float = 0.10,
    max_stale_fraction: float = 0.45,
    min_query_only_margin: float = 0.10,
) -> tuple[dict[str, Any], ...]:
    """Classify multi-turn summary rows by restore budget."""

    by_k: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_k.setdefault(_int(row, "k"), {})[str(row.get("condition", ""))] = dict(row)

    decisions: list[dict[str, Any]] = []
    for k, conditions in sorted(by_k.items()):
        idle = conditions.get("IdleKV")
        if idle is None:
            decisions.append({"k": k, "action": "missing_idlekv", "main_candidate": False})
            continue
        idle_noninitial = _float(idle, "mean_noninitial_gain_vs_matched")
        idle_revisit = _float(idle, "mean_revisit_gain_vs_matched")
        stale = conditions.get("StaleQ-K", {})
        stale_noninitial = _float(stale, "mean_noninitial_gain_vs_matched")
        current_q_only = conditions.get("CurrentQOnly-K", {})
        stale_q_only = conditions.get("StaleQOnly-K", {})
        current_q_only_noninitial = _float(current_q_only, "mean_noninitial_gain_vs_matched")
        stale_q_only_noninitial = _float(stale_q_only, "mean_noninitial_gain_vs_matched")
        has_query_only_controls = bool(current_q_only and stale_q_only)
        query_only_margin = current_q_only_noninitial - stale_q_only_noninitial
        random_noninitial = _float(conditions.get("Random-K", {}), "mean_noninitial_gain_vs_matched")
        oldest_noninitial = _float(conditions.get("Oldest-K", {}), "mean_noninitial_gain_vs_matched")
        best_control = max(random_noninitial, oldest_noninitial)
        stale_fraction = stale_noninitial / idle_noninitial if idle_noninitial > 0 else 1.0
        query_only_ok = (not has_query_only_controls) or query_only_margin >= min_query_only_margin
        main_candidate = all(
            [
                idle_noninitial >= min_idle_noninitial_gain,
                idle_revisit >= min_revisit_gain,
                best_control <= max_control_noninitial_gain,
                stale_fraction <= max_stale_fraction,
                query_only_ok,
            ]
        )
        if main_candidate:
            action = "main_candidate"
        elif idle_noninitial < min_idle_noninitial_gain or idle_revisit < min_revisit_gain:
            action = "redesign_or_recalibrate_budget"
        elif best_control > max_control_noninitial_gain:
            action = "reject_content_agnostic_controls_explain_gain"
        elif stale_fraction > max_stale_fraction:
            action = "appendix_only_stale_query_too_strong"
        elif not query_only_ok:
            action = "appendix_only_query_only_controls_too_close"
        else:
            action = "appendix_only"
        decision = {
            "k": k,
            "action": action,
            "main_candidate": main_candidate,
            "idle_noninitial_gain": round(idle_noninitial, 6),
            "idle_revisit_gain": round(idle_revisit, 6),
            "best_control_noninitial_gain": round(best_control, 6),
            "stale_fraction": round(stale_fraction, 6),
        }
        if has_query_only_controls:
            decision.update(
                {
                    "current_q_only_noninitial_gain": round(current_q_only_noninitial, 6),
                    "stale_q_only_noninitial_gain": round(stale_q_only_noninitial, 6),
                    "query_only_margin": round(query_only_margin, 6),
                }
            )
        decisions.append(decision)
    return tuple(decisions)
