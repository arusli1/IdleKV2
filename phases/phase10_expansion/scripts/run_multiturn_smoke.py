#!/usr/bin/env python3
"""Run a rolling multi-turn IdleKV repair smoke.

This is the first GPU implementation for the Phase 10 multi-turn branch.
It runs one condition trajectory at a time to avoid keeping many long-context
KV states resident simultaneously.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch

PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_ROOT.parents[1]
for root in (PHASE_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from phases.phase1_degradation.phase1.evaluation import sample_score  # noqa: E402
from phases.phase1_degradation.phase1.inference import prepare_example_for_model  # noqa: E402
from phases.phase1_degradation.phase1.task_registry import build_task_example  # noqa: E402
from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache, inject_kv, slice_kv  # noqa: E402
from phases.phase2_kv_cache.src.runtime import MODEL_DIR as DEFAULT_MODEL_DIR  # noqa: E402
from phases.phase2_kv_cache.src.runtime import load_model, load_tokenizer  # noqa: E402
from phases.phase3_eviction.src.runtime import build_position_tracked_cache  # noqa: E402
from phases.phase6_repair.src.protocol import (  # noqa: E402
    build_turn_n_keep_plan,
    compute_q2_exact_query_rows,
    compute_q2_query_rows,
    generate_turn,
    materialize_context_partition,
    relevant_position_groups_for_spans,
    relevant_positions_for_spans,
    split_example_for_turn,
)
from phases.phase6_repair.src.runner import _overlap_fraction, _restore_positions  # noqa: E402
from phases.phase6_repair.src.selectors import (  # noqa: E402
    score_evicted_positions,
    select_idlekv_positions,
    select_oldest_positions,
    select_oracle_positions,
    select_random_positions,
)
from phases.phase10_expansion.src.multiturn import (  # noqa: E402
    DEFAULT_8Q_CHALLENGE_REVISIT,
    DEFAULT_8Q_HARD_REVISIT,
    DEFAULT_8Q_SHIFT_REVISIT,
    DEFAULT_8Q_SWEEP_REVISIT,
    MultiTurnSchedule,
    evaluate_multiturn_summary_rows,
    revisit_events,
    span_names_by_turn,
    summarize_score_trajectory,
    validate_schedule,
)

CONDITIONS = (
    "Full",
    "Matched",
    "IdleKV",
    "CurrentQOnly-K",
    "Random-K",
    "Oldest-K",
    "StaleQ-K",
    "StaleQOnly-K",
    "Gold-K",
)
SCHEDULES: dict[str, MultiTurnSchedule] = {
    DEFAULT_8Q_CHALLENGE_REVISIT.name: DEFAULT_8Q_CHALLENGE_REVISIT,
    DEFAULT_8Q_HARD_REVISIT.name: DEFAULT_8Q_HARD_REVISIT,
    DEFAULT_8Q_SHIFT_REVISIT.name: DEFAULT_8Q_SHIFT_REVISIT,
    DEFAULT_8Q_SWEEP_REVISIT.name: DEFAULT_8Q_SWEEP_REVISIT,
}


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _score_text(text: str, outputs: Iterable[str]) -> float:
    return float(sample_score(text, list(outputs)))


def _active_key_indices(
    *,
    prepared,
    active_context_positions: Iterable[int],
    key_count: int,
) -> list[int]:
    active = {int(position) for position in active_context_positions}
    active_keys: list[int] = []
    for key_index in range(int(key_count)):
        span_positions = {
            int(position)
            for position in prepared.span_token_positions.get(f"needle_{key_index + 1}", [])
        }
        if span_positions & active:
            active_keys.append(int(key_index))
    return active_keys


def _slice_by_positions(cache: PositionTrackedCache, positions: Iterable[int]) -> PositionTrackedCache:
    position_to_dense = {int(position): dense_index for dense_index, position in enumerate(cache.positions)}
    dense_indices = [position_to_dense[int(position)] for position in positions]
    fragment = slice_kv(cache, dense_indices)
    if not isinstance(fragment, PositionTrackedCache):
        raise RuntimeError("slice_kv did not preserve position metadata.")
    return fragment


def _cache_nbytes(cache: PositionTrackedCache) -> int:
    """Return key/value tensor bytes for a position-tracked cache."""
    total = 0
    for key, value in cache.kv:
        total += int(key.numel() * key.element_size())
        total += int(value.numel() * value.element_size())
    return total


def _budget_audit_fields(
    *,
    active_context_count: int,
    target_active_context_count: int,
    base_active_context_count: int,
    evicted_buffer_count: int,
    evicted_buffer_bytes: int,
) -> dict[str, int | bool]:
    return {
        "target_active_context_count": int(target_active_context_count),
        "base_active_context_count": int(base_active_context_count),
        "evicted_buffer_count": int(evicted_buffer_count),
        "evicted_buffer_bytes": int(evicted_buffer_bytes),
        "active_budget_gap": int(active_context_count) - int(target_active_context_count),
        "active_context_matches_target": int(active_context_count) == int(target_active_context_count),
    }


def _append_new_rows(pool_cache: PositionTrackedCache, generated_cache: PositionTrackedCache) -> PositionTrackedCache:
    """Append newly generated turn rows to the full condition-local KV pool."""
    existing = {int(position) for position in pool_cache.positions}
    new_positions = [int(position) for position in generated_cache.positions if int(position) not in existing]
    if not new_positions:
        return pool_cache
    new_fragment = _slice_by_positions(generated_cache, new_positions)
    return inject_kv(pool_cache, new_fragment, new_fragment.positions)


def _compute_query_scores(
    *,
    model,
    active_cache,
    evicted_cache,
    question_ids: torch.Tensor,
    query_scoring_mode: str,
    pooling: str,
) -> dict[int, float]:
    if query_scoring_mode == "exact_q":
        query_rows = compute_q2_exact_query_rows(
            model,
            active_cache=active_cache,
            question_ids=question_ids,
        )
    else:
        query_rows = compute_q2_query_rows(
            model,
            active_cache=active_cache,
            question_ids=question_ids,
        )
    return score_evicted_positions(
        query_rows=query_rows,
        evicted_cache=evicted_cache,
        active_cache=active_cache,
        pooling=pooling,
    )


def _select_repair_positions(
    *,
    condition: str,
    model,
    active_cache,
    evicted_cache,
    prepared,
    previous_prepared,
    relevant_positions: tuple[int, ...],
    relevant_groups: tuple[tuple[int, ...], ...],
    turn_n_scores: dict[int, float],
    k: int,
    query_scoring_mode: str,
    pooling: str,
    burst_left: int,
    burst_right: int,
    seed: int,
) -> tuple[list[int], dict[str, float]]:
    evicted_positions = tuple(int(position) for position in evicted_cache.positions)
    timings: dict[str, float] = {}

    if condition == "Random-K":
        return (
            select_random_positions(
                evicted_positions=evicted_positions,
                k=int(k),
                left=int(burst_left),
                right=int(burst_right),
                seed=int(seed),
            ),
            timings,
        )
    if condition == "Oldest-K":
        return (
            select_oldest_positions(
                evicted_positions=evicted_positions,
                k=int(k),
                left=int(burst_left),
                right=int(burst_right),
            ),
            timings,
        )

    score_question_ids = prepared.question_ids
    if condition in {"StaleQ-K", "StaleQOnly-K"}:
        if previous_prepared is None:
            return [], timings
        score_question_ids = previous_prepared.question_ids

    query_start = time.perf_counter()
    q_scores = _compute_query_scores(
        model=model,
        active_cache=active_cache,
        evicted_cache=evicted_cache,
        question_ids=score_question_ids,
        query_scoring_mode=query_scoring_mode,
        pooling=pooling,
    )
    timings["query_score_s"] = time.perf_counter() - query_start

    if condition in {"IdleKV", "StaleQ-K", "CurrentQOnly-K", "StaleQOnly-K"}:
        secondary_scores = {} if condition in {"CurrentQOnly-K", "StaleQOnly-K"} else turn_n_scores
        return (
            select_idlekv_positions(
                evicted_positions=evicted_positions,
                q2_scores=q_scores,
                turn_n_scores=secondary_scores,
                k=int(k),
                left=int(burst_left),
                right=int(burst_right),
            ),
            timings,
        )
    if condition == "Gold-K":
        return (
            select_oracle_positions(
                evicted_positions=evicted_positions,
                relevant_positions=relevant_positions,
                relevant_position_groups=relevant_groups,
                q2_scores=q_scores,
                turn_n_scores=turn_n_scores,
                k=int(k),
                left=int(burst_left),
                right=int(burst_right),
            ),
            timings,
        )
    raise ValueError(f"Unsupported repair condition: {condition!r}.")


def _run_condition_trajectory(
    *,
    model,
    tokenizer,
    turn_prepareds,
    turn_span_names: tuple[tuple[str, ...], ...],
    schedule: MultiTurnSchedule,
    condition: str,
    k: int,
    example_index: int,
    initial_cache,
    context_len: int,
    base_context_budget: int,
    sink_size: int,
    recency_window: int,
    pooling: str,
    initial_compressor: str,
    query_scoring_mode: str,
    burst_left: int,
    burst_right: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pool_cache = initial_cache
    previous_token_ids: torch.Tensor | None = None
    previous_prepared = None
    key_count = max(max(turn) for turn in schedule.turns) + 1

    for turn_index, prepared in enumerate(turn_prepareds):
        active_context_positions: tuple[int, ...]
        selected_positions: list[int] = []
        repair_timings: dict[str, float] = {}
        restore_timing = {"transfer_ms": 0.0, "inject_ms": 0.0, "restored_count": 0.0}
        cache_for_turn = pool_cache
        target_active_context_count = int(context_len)
        base_active_context_count = int(context_len)
        evicted_buffer_count = 0
        evicted_buffer_bytes = 0

        relevant_positions = relevant_positions_for_spans(prepared, turn_span_names[turn_index])
        relevant_groups = relevant_position_groups_for_spans(prepared, turn_span_names[turn_index])

        if turn_index == 0 or condition == "Full":
            active_context_positions = tuple(range(int(context_len)))
        else:
            keep_plan = build_turn_n_keep_plan(
                post_q1_cache=pool_cache,
                q1_answer_ids=previous_token_ids if previous_token_ids is not None else torch.empty((0,), dtype=torch.long),
                context_len=int(context_len),
                sink_size=int(sink_size),
                recency_window=int(recency_window),
                pooling=pooling,
                initial_compressor=initial_compressor,
            )
            if condition == "Matched":
                matched_partition = materialize_context_partition(
                    full_post_q1_cache=pool_cache,
                    keep_plan=keep_plan,
                    context_budget=int(base_context_budget) + int(k),
                )
                cache_for_turn = matched_partition.compressed
                active_context_positions = tuple(matched_partition.kept_context_positions)
                target_active_context_count = int(base_context_budget) + int(k)
                base_active_context_count = int(target_active_context_count)
            else:
                base_partition = materialize_context_partition(
                    full_post_q1_cache=pool_cache,
                    keep_plan=keep_plan,
                    context_budget=int(base_context_budget),
                )
                target_active_context_count = int(base_context_budget) + int(k)
                base_active_context_count = int(len(base_partition.kept_context_positions))
                evicted_buffer_count = int(len(base_partition.evicted.positions))
                evicted_buffer_bytes = _cache_nbytes(base_partition.evicted)
                selected_positions, repair_timings = _select_repair_positions(
                    condition=condition,
                    model=model,
                    active_cache=base_partition.compressed,
                    evicted_cache=base_partition.evicted,
                    prepared=prepared,
                    previous_prepared=previous_prepared,
                    relevant_positions=relevant_positions,
                    relevant_groups=relevant_groups,
                    turn_n_scores=keep_plan.importance_scores,
                    k=int(k),
                    query_scoring_mode=query_scoring_mode,
                    pooling=pooling,
                    burst_left=int(burst_left),
                    burst_right=int(burst_right),
                    seed=(int(example_index) + 1) * 100_000 + int(turn_index) * 1000 + int(k),
                )
                cache_for_turn, restore_timing = _restore_positions(
                    active_cache=base_partition.compressed,
                    evicted_cache=base_partition.evicted,
                    selected_positions=selected_positions,
                )
                active_context_positions = tuple(
                    sorted(set(base_partition.kept_context_positions) | set(selected_positions))
                )

        generation_start = time.perf_counter()
        generated = generate_turn(model, tokenizer, prepared, cache_for_turn)
        generation_s = time.perf_counter() - generation_start
        score = _score_text(generated.text, prepared.example.outputs)
        active_keys = _active_key_indices(
            prepared=prepared,
            active_context_positions=active_context_positions,
            key_count=key_count,
        )

        rows.append(
            {
                "example_index": int(example_index),
                "turn": int(turn_index),
                "condition": condition,
                "k": int(k),
                "score": round(score, 6),
                "output": generated.text,
                "requested_key_indices": list(schedule.turns[turn_index]),
                "active_key_indices": active_keys,
                "active_requested_key_fraction": round(
                    len(set(active_keys) & set(schedule.turns[turn_index])) / max(1, len(schedule.turns[turn_index])),
                    6,
                ),
                "active_context_count": int(len(active_context_positions)),
                "selected_count": int(len(selected_positions)),
                "selected_positions": selected_positions,
                "selected_overlap_fraction": round(_overlap_fraction(selected_positions, relevant_positions), 6),
                "active_overlap_fraction": round(_overlap_fraction(active_context_positions, relevant_positions), 6),
                "generation_s": round(generation_s, 6),
                "query_score_s": round(float(repair_timings.get("query_score_s", 0.0)), 6),
                "transfer_ms": round(float(restore_timing["transfer_ms"]), 6),
                "inject_ms": round(float(restore_timing["inject_ms"]), 6),
                "answer_tokens": int(generated.token_ids.numel()),
                **_budget_audit_fields(
                    active_context_count=int(len(active_context_positions)),
                    target_active_context_count=int(target_active_context_count),
                    base_active_context_count=int(base_active_context_count),
                    evicted_buffer_count=int(evicted_buffer_count),
                    evicted_buffer_bytes=int(evicted_buffer_bytes),
                ),
            }
        )

        pool_cache = _append_new_rows(pool_cache, generated.cache)
        previous_token_ids = generated.token_ids
        previous_prepared = prepared
    return rows


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    schedule = SCHEDULES[str(args.schedule)]
    validate_schedule(schedule, key_count=int(args.key_count))

    model_dir = Path(args.model_dir).expanduser()
    model = load_model(model_dir)
    tokenizer = load_tokenizer(model_dir)
    turn_spans = span_names_by_turn(schedule)
    all_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for example_index in range(int(args.num_samples)):
        base_example = build_task_example(
            schedule.base_task_key,
            example_index,
            int(args.context_length),
            tokenizer,
            dataset_seed_offset=int(args.dataset_seed_offset),
        )
        turn_prepareds = tuple(
            prepare_example_for_model(
                split_example_for_turn(
                    base_example,
                    query_indices=turn,
                    split_name=f"{schedule.name}:turn{turn_index + 1}",
                    max_new_tokens=int(schedule.max_new_tokens),
                ),
                tokenizer,
            )
            for turn_index, turn in enumerate(schedule.turns)
        )
        initial_cache = build_position_tracked_cache(model, turn_prepareds[0].context_ids)
        context_len = int(turn_prepareds[0].context_ids.shape[1])

        for k in args.k:
            for condition in args.conditions:
                trajectory_rows = _run_condition_trajectory(
                    model=model,
                    tokenizer=tokenizer,
                    turn_prepareds=turn_prepareds,
                    turn_span_names=turn_spans,
                    schedule=schedule,
                    condition=str(condition),
                    k=int(k),
                    example_index=example_index,
                    initial_cache=initial_cache,
                    context_len=context_len,
                    base_context_budget=int(args.base_context_budget),
                    sink_size=int(args.sink_size),
                    recency_window=int(args.recency_window),
                    pooling=str(args.pooling),
                    initial_compressor=str(args.initial_compressor),
                    query_scoring_mode=str(args.query_scoring_mode),
                    burst_left=int(args.burst_left),
                    burst_right=int(args.burst_right),
                )
                all_rows.extend(trajectory_rows)
                last = trajectory_rows[-1]
                print(
                    f"[multi-turn] ex={example_index + 1:03d} k={int(k)} "
                    f"condition={condition} final_score={float(last['score']):.3f}",
                    flush=True,
                )

    revisit_turns = tuple(event["revisit_turn"] for event in revisit_events(schedule))
    summary_rows: list[dict[str, Any]] = []
    for k in args.k:
        rows_for_k = [row for row in all_rows if int(row["k"]) == int(k)]
        for row in summarize_score_trajectory(
            rows_for_k,
            matched_condition="Matched",
            revisit_turns=revisit_turns,
            condition_order=args.conditions,
        ):
            enriched = dict(row)
            enriched["k"] = int(k)
            summary_rows.append(enriched)

    recommendations = list(evaluate_multiturn_summary_rows(summary_rows))
    output_csv = Path(args.output_csv)
    summary_csv = Path(args.summary_csv)
    raw_json = Path(args.raw_json)
    _write_csv(all_rows, output_csv)
    _write_csv(summary_rows, summary_csv)
    raw_json.parent.mkdir(parents=True, exist_ok=True)
    raw_json.write_text(
        json.dumps(
            {
                "schema_version": "phase10-multiturn-smoke-v1",
                "config": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "schedule": {
                    "name": schedule.name,
                    "base_task_key": schedule.base_task_key,
                    "turns": [list(turn) for turn in schedule.turns],
                    "revisit_events": list(revisit_events(schedule)),
                },
                "rows": all_rows,
                "summary_rows": summary_rows,
                "recommendations": recommendations,
                "elapsed_s": round(time.perf_counter() - started, 6),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "output_csv": str(output_csv),
        "summary_csv": str(summary_csv),
        "raw_json": str(raw_json),
        "recommendations": recommendations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", choices=sorted(SCHEDULES), default=DEFAULT_8Q_SHIFT_REVISIT.name)
    parser.add_argument("--key-count", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=32_768)
    parser.add_argument("--dataset-seed-offset", type=int, default=0)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--base-context-budget", type=int, default=18_432)
    parser.add_argument("--k", nargs="+", type=int, default=[96])
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--sink-size", type=int, default=4)
    parser.add_argument("--recency-window", type=int, default=128)
    parser.add_argument("--pooling", choices=("max", "mean"), default="max")
    parser.add_argument("--initial-compressor", choices=("snapkv", "streaming_llm", "h2o"), default="snapkv")
    parser.add_argument("--query-scoring-mode", choices=("exact_q", "proxy"), default="exact_q")
    parser.add_argument("--burst-left", type=int, default=2)
    parser.add_argument("--burst-right", type=int, default=20)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PHASE_ROOT / "results" / "multiturn_smoke_rows_n1.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=PHASE_ROOT / "results" / "multiturn_smoke_summary_n1.csv",
    )
    parser.add_argument(
        "--raw-json",
        type=Path,
        default=PHASE_ROOT / "results" / "multiturn_smoke_n1_raw.json",
    )
    return parser.parse_args()


def main() -> int:
    result = run_smoke(parse_args())
    print(result["output_csv"])
    print(result["summary_csv"])
    print(result["raw_json"])
    for recommendation in result["recommendations"]:
        print(recommendation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
