"""Reporting helpers for Phase 6 result artifacts."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
import random
from statistics import median

DEFAULT_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("condition_a", "mean_condition_a"),
    ("condition_b", "mean_condition_b"),
    ("b_match", "mean_b_match"),
    ("idlekv", "mean_idlekv"),
    ("idlekv_coverage", "mean_idlekv_coverage"),
    ("idlekv_mmr", "mean_idlekv_mmr"),
    ("wrong_q_k", "mean_wrong_q_k"),
    ("stale_q_k", "mean_stale_q_k"),
    ("contrastive_q_k", "mean_contrastive_q_k"),
    ("random_k", "mean_random_k"),
    ("oldest_k", "mean_oldest_k"),
    ("oracle_k", "mean_oracle_k"),
    ("selection_lift", "mean_selection_lift"),
    ("idlekv_coverage_lift", "mean_idlekv_coverage_lift"),
    ("idlekv_mmr_lift", "mean_idlekv_mmr_lift"),
    ("wrong_q_lift", "mean_wrong_q_lift"),
    ("stale_q_lift", "mean_stale_q_lift"),
    ("contrastive_q_lift", "mean_contrastive_q_lift"),
    ("oracle_lift", "mean_oracle_lift"),
    ("pct_idlekv_gt_b_match", "pct_idlekv_gt_b_match"),
    ("pct_idlekv_lt_b_match", "pct_idlekv_lt_b_match"),
    ("pct_idlekv_coverage_gt_b_match", "pct_idlekv_coverage_gt_b_match"),
    ("pct_idlekv_mmr_gt_b_match", "pct_idlekv_mmr_gt_b_match"),
    ("pct_wrong_q_gt_b_match", "pct_wrong_q_gt_b_match"),
    ("pct_stale_q_gt_b_match", "pct_stale_q_gt_b_match"),
    ("pct_contrastive_q_gt_b_match", "pct_contrastive_q_gt_b_match"),
    ("select_transfer_inject_ms", "mean_idlekv_repair_ms"),
)

BOOTSTRAP_SCORE_FIELDS: tuple[tuple[str, str], ...] = (
    ("condition_a", "condition_a_score"),
    ("condition_b", "condition_b_score"),
    ("b_match", "b_match_score"),
    ("idlekv", "idlekv_score"),
    ("idlekv_coverage", "idlekv_coverage_score"),
    ("idlekv_mmr", "idlekv_mmr_score"),
    ("wrong_q_k", "wrong_q_k_score"),
    ("stale_q_k", "stale_q_k_score"),
    ("contrastive_q_k", "contrastive_q_k_score"),
    ("random_k", "random_k_score"),
    ("oldest_k", "oldest_k_score"),
    ("oracle_k", "oracle_k_score"),
)


def _k_value(k_label: str) -> int:
    if not k_label.startswith("k"):
        raise ValueError(f"Expected k-prefixed label, got: {k_label}")
    return int(k_label[1:])


def frontier_rows(
    aggregate_section: Mapping[str, Mapping[str, float]],
    field_map: Iterable[tuple[str, str]] = DEFAULT_FIELD_MAP,
) -> list[dict[str, float | int]]:
    """Return numerically sorted frontier rows for one aggregate section."""
    rows: list[dict[str, float | int]] = []
    for k_label in sorted(aggregate_section, key=_k_value):
        metrics = aggregate_section[k_label]
        row: dict[str, float | int] = {"k": _k_value(k_label)}
        for out_key, metric_key in field_map:
            if metric_key in metrics:
                row[out_key] = metrics[metric_key]
        rows.append(row)
    return rows


def split_rows(
    by_task_section: Mapping[str, Mapping[str, Mapping[str, float]]],
    field_map: Iterable[tuple[str, str]] = DEFAULT_FIELD_MAP,
) -> list[dict[str, float | int | str]]:
    """Flatten per-task aggregates into task-tagged frontier rows."""
    rows: list[dict[str, float | int | str]] = []
    for task_name in sorted(by_task_section):
        for row in frontier_rows(by_task_section[task_name], field_map=field_map):
            rows.append({"task": task_name, **row})
    return rows


def write_csv(rows: list[Mapping[str, object]], path: Path) -> None:
    """Write rows to CSV with the field order defined by the first row."""
    if not rows:
        raise ValueError("Cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 1:
        return float(max(values))
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float((1.0 - weight) * ordered[lower] + weight * ordered[upper])


def _group_rows(rows: Iterable[Mapping[str, object]], *, by_task: bool) -> dict[tuple[str | None, int], list[Mapping[str, object]]]:
    grouped: dict[tuple[str | None, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        k = int(row["k"])
        task = str(row["task"]) if by_task else None
        grouped[(task, k)].append(row)
    return grouped


def _bootstrap_bounds(values: list[float], *, num_bootstrap: int, seed: str) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        value = float(values[0])
        return value, value
    rng = random.Random(seed)
    count = len(values)
    means: list[float] = []
    for _ in range(num_bootstrap):
        sample_sum = 0.0
        for _ in range(count):
            sample_sum += float(values[rng.randrange(count)])
        means.append(sample_sum / count)
    means.sort()
    return _percentile(means, 0.025), _percentile(means, 0.975)


def bootstrap_frontier_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    by_task: bool = False,
    num_bootstrap: int = 1000,
    seed: int = 0,
    fields: Iterable[tuple[str, str]] = BOOTSTRAP_SCORE_FIELDS,
) -> list[dict[str, float | int | str]]:
    """Aggregate mean score columns with deterministic bootstrap confidence bounds."""
    output: list[dict[str, float | int | str]] = []
    for (task, k), grouped_rows in sorted(_group_rows(rows, by_task=by_task).items(), key=lambda item: ((item[0][0] or ""), item[0][1])):
        out_row: dict[str, float | int | str] = {"k": k, "n": len(grouped_rows)}
        for out_key, row_key in fields:
            values = [float(row.get(row_key, 0.0)) for row in grouped_rows]
            mean = float(sum(values) / len(values)) if values else 0.0
            lo, hi = _bootstrap_bounds(values, num_bootstrap=num_bootstrap, seed=f"{seed}:{task or 'overall'}:{k}:{out_key}")
            out_row[out_key] = mean
            out_row[f"{out_key}_lo"] = lo
            out_row[f"{out_key}_hi"] = hi
        if task is not None:
            out_row["task"] = task
            out_row = {"task": task, **{key: value for key, value in out_row.items() if key != "task"}}
        output.append(out_row)
    return output


def runtime_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    by_task: bool = False,
) -> list[dict[str, float | int | str]]:
    """Aggregate exact repair runtime fields from artifact rows."""
    output: list[dict[str, float | int | str]] = []
    for (task, k), grouped_rows in sorted(_group_rows(rows, by_task=by_task).items(), key=lambda item: ((item[0][0] or ""), item[0][1])):
        totals = [
            1000.0 * float(row.get("q2_query_rows_s", 0.0))
            + 1000.0 * float(row.get("q2_evicted_scoring_s", 0.0))
            + 1000.0 * float(row.get("idlekv_selection_s", 0.0))
            + float(row.get("idlekv_transfer_ms", 0.0))
            + float(row.get("idlekv_inject_ms", 0.0))
            for row in grouped_rows
        ]
        query_ms = [1000.0 * float(row.get("q2_query_rows_s", 0.0)) for row in grouped_rows]
        score_ms = [1000.0 * float(row.get("q2_evicted_scoring_s", 0.0)) for row in grouped_rows]
        select_ms = [1000.0 * float(row.get("idlekv_selection_s", 0.0)) for row in grouped_rows]
        transfer_ms = [float(row.get("idlekv_transfer_ms", 0.0)) for row in grouped_rows]
        inject_ms = [float(row.get("idlekv_inject_ms", 0.0)) for row in grouped_rows]
        wrong_q_totals = [
            1000.0 * float(row.get("wrong_q_query_rows_s", 0.0))
            + 1000.0 * float(row.get("wrong_q_evicted_scoring_s", 0.0))
            + 1000.0 * float(row.get("wrong_q_k_selection_s", 0.0))
            + float(row.get("wrong_q_k_transfer_ms", 0.0))
            + float(row.get("wrong_q_k_inject_ms", 0.0))
            for row in grouped_rows
            if "wrong_q_k_score" in row
        ]
        wrong_q_query_ms = [
            1000.0 * float(row.get("wrong_q_query_rows_s", 0.0))
            for row in grouped_rows
            if "wrong_q_k_score" in row
        ]
        wrong_q_score_ms = [
            1000.0 * float(row.get("wrong_q_evicted_scoring_s", 0.0))
            for row in grouped_rows
            if "wrong_q_k_score" in row
        ]
        wrong_q_select_ms = [
            1000.0 * float(row.get("wrong_q_k_selection_s", 0.0))
            for row in grouped_rows
            if "wrong_q_k_score" in row
        ]
        wrong_q_transfer_ms = [
            float(row.get("wrong_q_k_transfer_ms", 0.0))
            for row in grouped_rows
            if "wrong_q_k_score" in row
        ]
        wrong_q_inject_ms = [
            float(row.get("wrong_q_k_inject_ms", 0.0))
            for row in grouped_rows
            if "wrong_q_k_score" in row
        ]
        stale_q_totals = [
            1000.0 * float(row.get("stale_q_query_rows_s", 0.0))
            + 1000.0 * float(row.get("stale_q_evicted_scoring_s", 0.0))
            + 1000.0 * float(row.get("stale_q_k_selection_s", 0.0))
            + float(row.get("stale_q_k_transfer_ms", 0.0))
            + float(row.get("stale_q_k_inject_ms", 0.0))
            for row in grouped_rows
            if "stale_q_k_score" in row
        ]
        stale_q_query_ms = [
            1000.0 * float(row.get("stale_q_query_rows_s", 0.0))
            for row in grouped_rows
            if "stale_q_k_score" in row
        ]
        stale_q_score_ms = [
            1000.0 * float(row.get("stale_q_evicted_scoring_s", 0.0))
            for row in grouped_rows
            if "stale_q_k_score" in row
        ]
        stale_q_select_ms = [
            1000.0 * float(row.get("stale_q_k_selection_s", 0.0))
            for row in grouped_rows
            if "stale_q_k_score" in row
        ]
        stale_q_transfer_ms = [
            float(row.get("stale_q_k_transfer_ms", 0.0))
            for row in grouped_rows
            if "stale_q_k_score" in row
        ]
        stale_q_inject_ms = [
            float(row.get("stale_q_k_inject_ms", 0.0))
            for row in grouped_rows
            if "stale_q_k_score" in row
        ]
        contrastive_q_totals = [
            1000.0 * float(row.get("q2_query_rows_s", 0.0))
            + 1000.0 * float(row.get("wrong_q_query_rows_s", 0.0))
            + 1000.0 * float(row.get("q2_evicted_scoring_s", 0.0))
            + 1000.0 * float(row.get("wrong_q_evicted_scoring_s", 0.0))
            + 1000.0 * float(row.get("contrastive_q_k_selection_s", 0.0))
            + float(row.get("contrastive_q_k_transfer_ms", 0.0))
            + float(row.get("contrastive_q_k_inject_ms", 0.0))
            for row in grouped_rows
            if "contrastive_q_k_score" in row
        ]
        contrastive_q_select_ms = [
            1000.0 * float(row.get("contrastive_q_k_selection_s", 0.0))
            for row in grouped_rows
            if "contrastive_q_k_score" in row
        ]
        contrastive_q_transfer_ms = [
            float(row.get("contrastive_q_k_transfer_ms", 0.0))
            for row in grouped_rows
            if "contrastive_q_k_score" in row
        ]
        contrastive_q_inject_ms = [
            float(row.get("contrastive_q_k_inject_ms", 0.0))
            for row in grouped_rows
            if "contrastive_q_k_score" in row
        ]
        out_row: dict[str, float | int | str] = {
            "k": k,
            "n": len(grouped_rows),
            "p50_total_ms": _percentile(totals, 0.50),
            "p95_total_ms": _percentile(totals, 0.95),
            "p99_total_ms": _percentile(totals, 0.99),
            "p50_query_ms": median(query_ms),
            "p50_score_ms": median(score_ms),
            "p50_select_ms": median(select_ms),
            "p50_transfer_ms": median(transfer_ms),
            "p50_inject_ms": median(inject_ms),
        }
        if wrong_q_totals:
            out_row.update(
                {
                    "p50_wrong_q_total_ms": _percentile(wrong_q_totals, 0.50),
                    "p95_wrong_q_total_ms": _percentile(wrong_q_totals, 0.95),
                    "p99_wrong_q_total_ms": _percentile(wrong_q_totals, 0.99),
                    "p50_wrong_q_query_ms": median(wrong_q_query_ms),
                    "p50_wrong_q_score_ms": median(wrong_q_score_ms),
                    "p50_wrong_q_select_ms": median(wrong_q_select_ms),
                    "p50_wrong_q_transfer_ms": median(wrong_q_transfer_ms),
                    "p50_wrong_q_inject_ms": median(wrong_q_inject_ms),
                }
            )
        if stale_q_totals:
            out_row.update(
                {
                    "p50_stale_q_total_ms": _percentile(stale_q_totals, 0.50),
                    "p95_stale_q_total_ms": _percentile(stale_q_totals, 0.95),
                    "p99_stale_q_total_ms": _percentile(stale_q_totals, 0.99),
                    "p50_stale_q_query_ms": median(stale_q_query_ms),
                    "p50_stale_q_score_ms": median(stale_q_score_ms),
                    "p50_stale_q_select_ms": median(stale_q_select_ms),
                    "p50_stale_q_transfer_ms": median(stale_q_transfer_ms),
                    "p50_stale_q_inject_ms": median(stale_q_inject_ms),
                }
            )
        if contrastive_q_totals:
            out_row.update(
                {
                    "p50_contrastive_q_total_ms": _percentile(contrastive_q_totals, 0.50),
                    "p95_contrastive_q_total_ms": _percentile(contrastive_q_totals, 0.95),
                    "p99_contrastive_q_total_ms": _percentile(contrastive_q_totals, 0.99),
                    "p50_contrastive_q_select_ms": median(contrastive_q_select_ms),
                    "p50_contrastive_q_transfer_ms": median(contrastive_q_transfer_ms),
                    "p50_contrastive_q_inject_ms": median(contrastive_q_inject_ms),
                }
            )
        if task is not None:
            out_row["task"] = task
            out_row = {"task": task, **{key: value for key, value in out_row.items() if key != "task"}}
        output.append(out_row)
    return output


def overlap_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    by_task: bool = False,
) -> list[dict[str, float | int | str]]:
    """Aggregate gold-overlap fractions from artifact rows."""
    output: list[dict[str, float | int | str]] = []
    for (task, k), grouped_rows in sorted(_group_rows(rows, by_task=by_task).items(), key=lambda item: ((item[0][0] or ""), item[0][1])):
        count = len(grouped_rows)

        def _row_overlap(row: Mapping[str, object], *fields: str) -> float:
            for explicit_field in (
                "condition_b_active_overlap_fraction",
                "condition_b_overlap_fraction",
                "b_match_active_overlap_fraction",
                "b_match_overlap_fraction",
                "idlekv_active_overlap_fraction",
                "stale_q_k_active_overlap_fraction",
                "contrastive_q_k_active_overlap_fraction",
                "wrong_q_k_active_overlap_fraction",
                "random_k_active_overlap_fraction",
                "oldest_k_active_overlap_fraction",
                "oracle_k_active_overlap_fraction",
            ):
                if explicit_field in fields and explicit_field in row:
                    return float(row.get(explicit_field, 0.0))
            relevant = {int(position) for position in row.get("q2_relevant_positions", [])}
            if not relevant:
                return 0.0
            base_kept = {int(position) for position in row.get("b_kept_context_positions", [])}
            if "condition_b_active_overlap_fraction" in fields:
                active = base_kept
            elif "condition_b_overlap_fraction" in fields:
                active = base_kept
            elif "idlekv_active_overlap_fraction" in fields:
                active = base_kept | {int(position) for position in row.get("idlekv_selected_positions", [])}
            elif "idlekv_overlap_fraction" in fields:
                active = base_kept | {int(position) for position in row.get("idlekv_selected_positions", [])}
                if not active and "idlekv_overlap_fraction" in row:
                    return float(row.get("idlekv_overlap_fraction", 0.0))
            elif "wrong_q_k_active_overlap_fraction" in fields:
                active = base_kept | {int(position) for position in row.get("wrong_q_k_selected_positions", [])}
            elif "wrong_q_k_overlap_fraction" in fields:
                active = base_kept | {int(position) for position in row.get("wrong_q_k_selected_positions", [])}
                if not active and "wrong_q_k_overlap_fraction" in row:
                    return float(row.get("wrong_q_k_overlap_fraction", 0.0))
            elif "stale_q_k_active_overlap_fraction" in fields:
                active = base_kept | {int(position) for position in row.get("stale_q_k_selected_positions", [])}
            elif "stale_q_k_overlap_fraction" in fields:
                active = base_kept | {int(position) for position in row.get("stale_q_k_selected_positions", [])}
                if not active and "stale_q_k_overlap_fraction" in row:
                    return float(row.get("stale_q_k_overlap_fraction", 0.0))
            elif "contrastive_q_k_active_overlap_fraction" in fields:
                active = base_kept | {int(position) for position in row.get("contrastive_q_k_selected_positions", [])}
            elif "contrastive_q_k_overlap_fraction" in fields:
                active = base_kept | {int(position) for position in row.get("contrastive_q_k_selected_positions", [])}
                if not active and "contrastive_q_k_overlap_fraction" in row:
                    return float(row.get("contrastive_q_k_overlap_fraction", 0.0))
            elif "random_k_active_overlap_fraction" in fields:
                active = base_kept | {int(position) for position in row.get("random_k_selected_positions", [])}
            elif "random_k_overlap_fraction" in fields:
                active = base_kept | {int(position) for position in row.get("random_k_selected_positions", [])}
                if not active and "random_k_overlap_fraction" in row:
                    return float(row.get("random_k_overlap_fraction", 0.0))
            elif "oldest_k_active_overlap_fraction" in fields:
                active = base_kept | {int(position) for position in row.get("oldest_k_selected_positions", [])}
            elif "oldest_k_overlap_fraction" in fields:
                active = base_kept | {int(position) for position in row.get("oldest_k_selected_positions", [])}
                if not active and "oldest_k_overlap_fraction" in row:
                    return float(row.get("oldest_k_overlap_fraction", 0.0))
            elif "oracle_k_active_overlap_fraction" in fields:
                active = base_kept | {int(position) for position in row.get("oracle_k_selected_positions", [])}
            elif "oracle_k_overlap_fraction" in fields:
                active = base_kept | {int(position) for position in row.get("oracle_k_selected_positions", [])}
                if not active and "oracle_k_overlap_fraction" in row:
                    return float(row.get("oracle_k_overlap_fraction", 0.0))
            else:
                active = set()
            return float(len(active & relevant) / len(relevant))

        def _mean(*fields: str) -> float:
            if not count:
                return 0.0
            values = [_row_overlap(row, *fields) for row in grouped_rows]
            return float(sum(values) / count)

        out_row: dict[str, float | int | str] = {
            "k": k,
            "n": count,
            "condition_b_overlap": _mean("condition_b_active_overlap_fraction", "condition_b_overlap_fraction"),
            "b_match_overlap": _mean("b_match_active_overlap_fraction", "b_match_overlap_fraction"),
            "idlekv_overlap": _mean("idlekv_active_overlap_fraction", "idlekv_overlap_fraction"),
            "wrong_q_k_overlap": _mean("wrong_q_k_active_overlap_fraction", "wrong_q_k_overlap_fraction"),
            "stale_q_k_overlap": _mean("stale_q_k_active_overlap_fraction", "stale_q_k_overlap_fraction"),
            "contrastive_q_k_overlap": _mean("contrastive_q_k_active_overlap_fraction", "contrastive_q_k_overlap_fraction"),
            "random_k_overlap": _mean("random_k_active_overlap_fraction", "random_k_overlap_fraction"),
            "oldest_k_overlap": _mean("oldest_k_active_overlap_fraction", "oldest_k_overlap_fraction"),
            "oracle_k_overlap": _mean("oracle_k_active_overlap_fraction", "oracle_k_overlap_fraction"),
        }
        if task is not None:
            out_row["task"] = task
            out_row = {"task": task, **{key: value for key, value in out_row.items() if key != "task"}}
        output.append(out_row)
    return output
