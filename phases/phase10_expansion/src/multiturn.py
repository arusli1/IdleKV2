"""Pure helpers for Phase 10 rolling multi-turn repair schedules."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class MultiTurnSchedule:
    """A fixed sequence of relevance targets over one multi-query context."""

    name: str
    base_task_key: str
    turns: tuple[tuple[int, ...], ...]
    max_new_tokens: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "turns", normalize_turns(self.turns))
        if not self.name:
            raise ValueError("schedule name must be non-empty.")
        if not self.base_task_key:
            raise ValueError("base_task_key must be non-empty.")
        if int(self.max_new_tokens) <= 0:
            raise ValueError("max_new_tokens must be positive.")


def normalize_turns(turns: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    """Normalize and validate a sequence of turn index groups."""
    normalized: list[tuple[int, ...]] = []
    for turn_index, turn in enumerate(turns):
        indices = tuple(dict.fromkeys(int(index) for index in turn))
        if not indices:
            raise ValueError(f"turn {turn_index} must contain at least one query index.")
        normalized.append(indices)
    if len(normalized) < 2:
        raise ValueError("a multi-turn schedule must contain at least two turns.")
    return tuple(normalized)


def validate_schedule(schedule: MultiTurnSchedule, *, key_count: int) -> None:
    """Validate that every scheduled query index is in range for a base task."""
    key_count = int(key_count)
    if key_count <= 0:
        raise ValueError("key_count must be positive.")
    for turn_index, turn in enumerate(schedule.turns):
        out_of_range = [index for index in turn if index < 0 or index >= key_count]
        if out_of_range:
            raise ValueError(
                f"turn {turn_index} has query indices outside [0, {key_count}): {out_of_range}."
            )


def span_names_by_turn(schedule: MultiTurnSchedule) -> tuple[tuple[str, ...], ...]:
    """Return benchmark span names for every turn in a schedule."""
    return tuple(
        tuple(f"needle_{index + 1}" for index in turn)
        for turn in schedule.turns
    )


def revisit_events(schedule: MultiTurnSchedule) -> tuple[dict[str, int], ...]:
    """Return positions where a later turn asks for a previously requested key."""
    first_seen: dict[int, int] = {}
    events: list[dict[str, int]] = []
    for turn_index, turn in enumerate(schedule.turns):
        for query_index in turn:
            if query_index in first_seen:
                events.append(
                    {
                        "query_index": int(query_index),
                        "first_turn": int(first_seen[query_index]),
                        "revisit_turn": int(turn_index),
                    }
                )
            else:
                first_seen[query_index] = turn_index
    return tuple(events)


def per_turn_overlap(schedule: MultiTurnSchedule) -> tuple[float, ...]:
    """Measure how much each turn overlaps the immediately previous turn."""
    overlaps: list[float] = []
    previous: set[int] | None = None
    for turn in schedule.turns:
        current = {int(index) for index in turn}
        if previous is None:
            overlaps.append(0.0)
        else:
            overlaps.append(float(len(current & previous) / max(1, len(current))))
        previous = current
    return tuple(overlaps)


def normalize_active_sets(active_by_turn: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    """Normalize active key groups observed before each turn answer."""
    normalized: list[tuple[int, ...]] = []
    for turn_index, active in enumerate(active_by_turn):
        values = tuple(sorted(dict.fromkeys(int(index) for index in active)))
        if not values:
            raise ValueError(f"active set for turn {turn_index} must be non-empty.")
        normalized.append(values)
    if not normalized:
        raise ValueError("active_by_turn must contain at least one turn.")
    return tuple(normalized)


def per_turn_recovery(
    schedule: MultiTurnSchedule,
    active_by_turn: Iterable[Iterable[int]],
) -> tuple[float, ...]:
    """Fraction of each turn's requested key groups present in active state."""
    active_sets = normalize_active_sets(active_by_turn)
    if len(active_sets) != len(schedule.turns):
        raise ValueError("active_by_turn length must match schedule turns.")
    recovered: list[float] = []
    for requested, active in zip(schedule.turns, active_sets, strict=True):
        requested_set = {int(index) for index in requested}
        active_set = {int(index) for index in active}
        recovered.append(float(len(requested_set & active_set) / max(1, len(requested_set))))
    return tuple(recovered)


def per_turn_active_churn(active_by_turn: Iterable[Iterable[int]]) -> tuple[dict[str, float], ...]:
    """Compute active-key additions, removals, and Jaccard continuity per turn."""
    active_sets = tuple(set(active) for active in normalize_active_sets(active_by_turn))
    rows: list[dict[str, float]] = []
    previous: set[int] | None = None
    for turn_index, active in enumerate(active_sets):
        if previous is None:
            rows.append(
                {
                    "turn": float(turn_index),
                    "added": float(len(active)),
                    "removed": 0.0,
                    "jaccard": 1.0,
                }
            )
        else:
            union = active | previous
            rows.append(
                {
                    "turn": float(turn_index),
                    "added": float(len(active - previous)),
                    "removed": float(len(previous - active)),
                    "jaccard": float(len(active & previous) / max(1, len(union))),
                }
            )
        previous = active
    return tuple(rows)


def cache_state_grid(
    schedule: MultiTurnSchedule,
    active_by_turn: Iterable[Iterable[int]],
    *,
    key_count: int,
) -> tuple[dict[str, int | bool], ...]:
    """Return key-by-turn active/requested cells for a cache-state heatmap."""
    validate_schedule(schedule, key_count=key_count)
    active_sets = normalize_active_sets(active_by_turn)
    if len(active_sets) != len(schedule.turns):
        raise ValueError("active_by_turn length must match schedule turns.")
    rows: list[dict[str, int | bool]] = []
    for turn_index, (requested, active) in enumerate(zip(schedule.turns, active_sets, strict=True)):
        requested_set = {int(index) for index in requested}
        active_set = {int(index) for index in active}
        for key_index in range(int(key_count)):
            rows.append(
                {
                    "turn": int(turn_index),
                    "key_index": int(key_index),
                    "requested": key_index in requested_set,
                    "active": key_index in active_set,
                }
            )
    return tuple(rows)


def _mean_or_zero(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return float(fmean(values)) if values else 0.0


def summarize_score_trajectory(
    rows: Iterable[dict[str, Any]],
    *,
    matched_condition: str = "Matched",
    score_key: str = "score",
    condition_key: str = "condition",
    turn_key: str = "turn",
    example_key: str = "example_index",
    revisit_turns: Iterable[int] = (),
    condition_order: Sequence[str] | None = None,
) -> tuple[dict[str, float | int | str], ...]:
    """Summarize paired multi-turn score rows against matched no-repair.

    The future GPU runner should emit one row per example, turn, and
    condition. This helper keeps the paper gate explicit: promoted
    multi-turn results should improve non-initial turns and revisit turns,
    not only the first easy shift.
    """
    materialized = [dict(row) for row in rows]
    if not materialized:
        return tuple()

    paired_key_to_matched: dict[tuple[str, int], float] = {}
    for row_index, row in enumerate(materialized):
        if example_key not in row:
            raise ValueError(f"row {row_index} is missing {example_key!r}.")
        condition = str(row.get(condition_key, ""))
        if condition != matched_condition:
            continue
        example = str(row[example_key])
        turn = int(row[turn_key])
        paired_key_to_matched[(example, turn)] = float(row[score_key])

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row_index, row in enumerate(materialized):
        condition = str(row.get(condition_key, ""))
        if not condition:
            raise ValueError(f"row {row_index} is missing {condition_key!r}.")
        grouped.setdefault(condition, []).append(row)

    revisit_turn_set = {int(turn) for turn in revisit_turns}
    if condition_order is None:
        ordered_conditions = sorted(grouped)
    else:
        ordered_conditions = [condition for condition in condition_order if condition in grouped]
        ordered_conditions.extend(condition for condition in sorted(grouped) if condition not in ordered_conditions)

    summaries: list[dict[str, float | int | str]] = []
    for condition in ordered_conditions:
        group = grouped[condition]
        scores: list[float] = []
        gains: list[float] = []
        noninitial_gains: list[float] = []
        revisit_gains: list[float] = []
        wins = 0
        paired_count = 0

        for row_index, row in enumerate(group):
            score = float(row[score_key])
            turn = int(row[turn_key])
            example = str(row[example_key])
            scores.append(score)
            matched = paired_key_to_matched.get((example, turn))
            if matched is None:
                continue
            gain = score - matched
            gains.append(gain)
            paired_count += 1
            if score > matched:
                wins += 1
            if turn > 0:
                noninitial_gains.append(gain)
            if turn in revisit_turn_set:
                revisit_gains.append(gain)

        summaries.append(
            {
                "condition": condition,
                "mean_score": round(_mean_or_zero(scores), 6),
                "mean_gain_vs_matched": round(_mean_or_zero(gains), 6),
                "mean_noninitial_gain_vs_matched": round(_mean_or_zero(noninitial_gains), 6),
                "mean_revisit_gain_vs_matched": round(_mean_or_zero(revisit_gains), 6),
                "win_rate_vs_matched": round(float(wins / paired_count), 6) if paired_count else 0.0,
                "n_rows": int(len(group)),
                "n_paired_rows": int(paired_count),
            }
        )
    return tuple(summaries)


def evaluate_multiturn_summary_rows(
    rows: Iterable[dict[str, Any]],
    *,
    idle_condition: str = "IdleKV",
    random_condition: str = "Random-K",
    oldest_condition: str = "Oldest-K",
    stale_condition: str | None = "StaleQ-K",
    min_noninitial_gain: float = 0.10,
    min_revisit_gain: float = 0.15,
    min_win_rate: float = 0.50,
    min_control_margin: float = 0.05,
    min_stale_margin: float = 0.05,
) -> tuple[dict[str, float | int | str], ...]:
    """Gate per-K multi-turn trajectory summaries for paper promotion."""
    materialized = [dict(row) for row in rows]
    if not materialized:
        return tuple()

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in materialized:
        grouped.setdefault(int(row.get("k", 0)), []).append(row)

    recommendations: list[dict[str, float | int | str]] = []
    for k, group in sorted(grouped.items()):
        by_condition = {str(row.get("condition", "")): row for row in group}
        idle = by_condition.get(idle_condition)
        if idle is None:
            recommendations.append({"k": int(k), "action": "do_not_promote_missing_idlekv"})
            continue

        idle_noninitial = float(idle.get("mean_noninitial_gain_vs_matched", 0.0))
        idle_revisit = float(idle.get("mean_revisit_gain_vs_matched", 0.0))
        idle_win_rate = float(idle.get("win_rate_vs_matched", 0.0))
        control_gains = [
            float(by_condition[condition].get("mean_noninitial_gain_vs_matched", 0.0))
            for condition in (random_condition, oldest_condition)
            if condition in by_condition
        ]
        best_control_gain = max(control_gains) if control_gains else 0.0
        control_margin = idle_noninitial - best_control_gain
        stale_gain = None
        stale_margin = None
        if stale_condition is not None and stale_condition in by_condition:
            stale_gain = float(by_condition[stale_condition].get("mean_noninitial_gain_vs_matched", 0.0))
            stale_margin = idle_noninitial - stale_gain

        main_ok = (
            idle_noninitial >= float(min_noninitial_gain)
            and idle_revisit >= float(min_revisit_gain)
            and idle_win_rate >= float(min_win_rate)
            and control_margin >= float(min_control_margin)
            and (stale_margin is None or stale_margin >= float(min_stale_margin))
        )
        appendix_ok = idle_noninitial > 0.0 and idle_revisit > 0.0

        if main_ok:
            action = "main_candidate_if_artifact_checks_pass"
        elif idle_revisit <= 0.0:
            action = "do_not_promote_no_revisit_gain"
        elif stale_margin is not None and stale_margin < float(min_stale_margin):
            action = "do_not_promote_stale_query_closes_gap"
        elif appendix_ok:
            action = "appendix_candidate"
        elif control_margin < float(min_control_margin):
            action = "do_not_promote_controls_close_gap"
        else:
            action = "do_not_promote"

        recommendation = {
            "k": int(k),
            "idle_noninitial_gain": round(idle_noninitial, 6),
            "idle_revisit_gain": round(idle_revisit, 6),
            "idle_win_rate": round(idle_win_rate, 6),
            "best_control_noninitial_gain": round(best_control_gain, 6),
            "control_margin": round(control_margin, 6),
            "action": action,
        }
        if stale_gain is not None and stale_margin is not None:
            recommendation["stale_noninitial_gain"] = round(stale_gain, 6)
            recommendation["stale_margin"] = round(stale_margin, 6)
        recommendations.append(recommendation)
    return tuple(recommendations)


DEFAULT_8Q_SHIFT_REVISIT = MultiTurnSchedule(
    name="mq_niah_8q_shift_revisit",
    base_task_key="mq_niah_8q",
    turns=((6, 7), (0, 1), (3, 4), (0, 1)),
    max_new_tokens=64,
)

DEFAULT_8Q_SWEEP_REVISIT = MultiTurnSchedule(
    name="mq_niah_8q_sweep_revisit",
    base_task_key="mq_niah_8q",
    turns=((6, 7), (2, 3), (4, 5), (6, 7)),
    max_new_tokens=64,
)

DEFAULT_8Q_HARD_REVISIT = MultiTurnSchedule(
    name="mq_niah_8q_hard_revisit",
    base_task_key="mq_niah_8q",
    turns=((6, 7), (0, 1), (4, 5), (2, 3), (0, 1)),
    max_new_tokens=64,
)

DEFAULT_8Q_CHALLENGE_REVISIT = MultiTurnSchedule(
    name="mq_niah_8q_challenge_revisit",
    base_task_key="mq_niah_8q",
    turns=((6, 7), (0, 1), (2, 3), (0, 1), (2, 3)),
    max_new_tokens=64,
)


__all__ = [
    "DEFAULT_8Q_CHALLENGE_REVISIT",
    "DEFAULT_8Q_HARD_REVISIT",
    "DEFAULT_8Q_SHIFT_REVISIT",
    "DEFAULT_8Q_SWEEP_REVISIT",
    "MultiTurnSchedule",
    "cache_state_grid",
    "normalize_turns",
    "normalize_active_sets",
    "per_turn_active_churn",
    "per_turn_overlap",
    "per_turn_recovery",
    "revisit_events",
    "span_names_by_turn",
    "summarize_score_trajectory",
    "evaluate_multiturn_summary_rows",
    "validate_schedule",
]
