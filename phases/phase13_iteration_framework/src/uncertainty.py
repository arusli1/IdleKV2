"""Paired uncertainty helpers for Phase 13 result audits."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import random
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


def paired_gain_values(
    rows: Iterable[dict[str, Any]],
    *,
    condition: str,
    k: int,
    turns: Sequence[int] | None = None,
    matched_condition: str = "Matched",
) -> tuple[float, ...]:
    """Return condition-minus-matched paired gains for one K and turn filter."""

    materialized = [dict(row) for row in rows]
    turn_set = {int(turn) for turn in turns} if turns is not None else None
    matched_by_key: dict[tuple[int, int], float] = {}
    for row in materialized:
        if str(row.get("condition", "")) != matched_condition or _int(row, "k") != int(k):
            continue
        turn = _int(row, "turn")
        if turn_set is not None and turn not in turn_set:
            continue
        matched_by_key[(_int(row, "example_index"), turn)] = _float(row, "score")

    gains: list[float] = []
    for row in materialized:
        if str(row.get("condition", "")) != condition or _int(row, "k") != int(k):
            continue
        turn = _int(row, "turn")
        if turn_set is not None and turn not in turn_set:
            continue
        matched = matched_by_key.get((_int(row, "example_index"), turn))
        if matched is not None:
            gains.append(_float(row, "score") - matched)
    return tuple(gains)


def paired_condition_difference_values(
    rows: Iterable[dict[str, Any]],
    *,
    condition: str,
    baseline_condition: str,
    k: int,
    turns: Sequence[int] | None = None,
) -> tuple[float, ...]:
    """Return condition-minus-baseline paired differences for one K and turn filter."""

    materialized = [dict(row) for row in rows]
    turn_set = {int(turn) for turn in turns} if turns is not None else None
    baseline_by_key: dict[tuple[int, int], float] = {}
    for row in materialized:
        if str(row.get("condition", "")) != baseline_condition or _int(row, "k") != int(k):
            continue
        turn = _int(row, "turn")
        if turn_set is not None and turn not in turn_set:
            continue
        baseline_by_key[(_int(row, "example_index"), turn)] = _float(row, "score")

    differences: list[float] = []
    for row in materialized:
        if str(row.get("condition", "")) != condition or _int(row, "k") != int(k):
            continue
        turn = _int(row, "turn")
        if turn_set is not None and turn not in turn_set:
            continue
        baseline = baseline_by_key.get((_int(row, "example_index"), turn))
        if baseline is not None:
            differences.append(_float(row, "score") - baseline)
    return tuple(differences)


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int]:
    """Return deterministic percentile bootstrap interval for a mean."""

    samples = [float(value) for value in values]
    if not samples:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    mean = sum(samples) / len(samples)
    if len(samples) == 1 or int(n_bootstrap) <= 0:
        rounded = round(mean, 6)
        return {"mean": rounded, "lo": rounded, "hi": rounded, "n": len(samples)}

    rng = random.Random(int(seed))
    boot_means: list[float] = []
    for _ in range(int(n_bootstrap)):
        total = 0.0
        for _sample_index in range(len(samples)):
            total += samples[rng.randrange(len(samples))]
        boot_means.append(total / len(samples))
    boot_means.sort()
    alpha = max(0.0, min(1.0, 1.0 - float(confidence)))
    lo_index = min(len(boot_means) - 1, max(0, int((alpha / 2.0) * len(boot_means))))
    hi_index = min(len(boot_means) - 1, max(0, int((1.0 - alpha / 2.0) * len(boot_means)) - 1))
    return {
        "mean": round(mean, 6),
        "lo": round(boot_means[lo_index], 6),
        "hi": round(boot_means[hi_index], 6),
        "n": len(samples),
    }


def multiturn_uncertainty_rows(
    rows: Iterable[dict[str, Any]],
    *,
    revisit_turns: Sequence[int],
    conditions: Sequence[str] | None = None,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> tuple[dict[str, float | int | str], ...]:
    """Summarize paired all/noninitial/revisit gains with bootstrap intervals."""

    materialized = [dict(row) for row in rows]
    if not materialized:
        return tuple()
    k_values = sorted({_int(row, "k") for row in materialized})
    if conditions is None:
        condition_values = sorted({str(row.get("condition", "")) for row in materialized if row.get("condition")})
    else:
        condition_values = [str(condition) for condition in conditions]
    max_turn = max(_int(row, "turn") for row in materialized)
    noninitial_turns = tuple(turn for turn in range(1, max_turn + 1))

    summaries: list[dict[str, float | int | str]] = []
    for k in k_values:
        for condition in condition_values:
            all_interval = bootstrap_mean_interval(
                paired_gain_values(materialized, condition=condition, k=k),
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            noninitial_interval = bootstrap_mean_interval(
                paired_gain_values(materialized, condition=condition, k=k, turns=noninitial_turns),
                n_bootstrap=n_bootstrap,
                seed=seed + 1,
            )
            revisit_interval = bootstrap_mean_interval(
                paired_gain_values(materialized, condition=condition, k=k, turns=revisit_turns),
                n_bootstrap=n_bootstrap,
                seed=seed + 2,
            )
            summaries.append(
                {
                    "k": int(k),
                    "condition": condition,
                    "mean_gain": all_interval["mean"],
                    "gain_lo": all_interval["lo"],
                    "gain_hi": all_interval["hi"],
                    "n_paired": all_interval["n"],
                    "mean_noninitial_gain": noninitial_interval["mean"],
                    "noninitial_gain_lo": noninitial_interval["lo"],
                    "noninitial_gain_hi": noninitial_interval["hi"],
                    "n_noninitial": noninitial_interval["n"],
                    "mean_revisit_gain": revisit_interval["mean"],
                    "revisit_gain_lo": revisit_interval["lo"],
                    "revisit_gain_hi": revisit_interval["hi"],
                    "n_revisit": revisit_interval["n"],
                }
            )
    return tuple(summaries)
