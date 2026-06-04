#!/usr/bin/env python3
"""Run the evict -> full repair -> SnapKV re-evict sweep."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

import torch

PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_ROOT.parents[1]
for root in (PHASE_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from phases.phase1_degradation.phase1.config import TASK_SPECS  # noqa: E402
from phases.phase1_degradation.phase1.evaluation import sample_score, task_prefix  # noqa: E402
from phases.phase1_degradation.phase1.inference import prepare_example_for_model  # noqa: E402
from phases.phase1_degradation.phase1.task_registry import build_task_example, get_task_spec  # noqa: E402
from phases.phase2_kv_cache.src.kv_utils import inject_kv  # noqa: E402
from phases.phase2_kv_cache.src.runtime import generate_from_cache, load_model, load_tokenizer, model_device  # noqa: E402
from phases.phase3_eviction.src.benchmark import (  # noqa: E402
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_OBS_WINDOW_SIZE,
    DEFAULT_POOLING,
    DEFAULT_SINK_SIZE,
)
from phases.phase3_eviction.src.eviction import SnapKV, StreamingLLM  # noqa: E402
from phases.phase3_eviction.src.runtime import build_position_tracked_cache, write_json  # noqa: E402

RESULTS_DIR = PHASE_ROOT / "results" / "phase5_repair_then_reevict"
LOG_DIR = RESULTS_DIR / "logs"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
PROGRESS_PATH = RESULTS_DIR / "progress.json"

DEFAULT_TASKS = tuple(TASK_SPECS.keys())
DEFAULT_METHODS = ("snapkv", "streaming_llm")
DEFAULT_BUDGETS = (256, 512, 1024)
DEFAULT_NUM_SAMPLES = 100
DEFAULT_DATASET_SEED_OFFSET = 0
DEFAULT_SNAPKV_RECENCY_CAP = 1024
SCHEMA_VERSION = "phase5-repair-then-reevict-v1"
SLICE_SCHEMA_VERSION = "phase5-repair-then-reevict-slice-v1"


def ensure_results_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _sync_if_cuda(device: torch.device | str) -> None:
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.synchronize(target)


def _normalize_tasks(tasks: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    for task_key in tasks:
        get_task_spec(task_key)
        if task_key not in ordered:
            ordered.append(task_key)
    return ordered


def _normalize_methods(methods: Iterable[str]) -> list[str]:
    allowed = {"snapkv", "streaming_llm"}
    ordered: list[str] = []
    for method in methods:
        normalized = str(method).lower()
        if normalized not in allowed:
            raise ValueError(f"Unsupported initial eviction method: {method}")
        if normalized not in ordered:
            ordered.append(normalized)
    return ordered


def _normalize_budgets(budgets: Iterable[int | str], *, context_length: int) -> list[int]:
    ordered: list[int] = []
    for budget in budgets:
        normalized = min(int(budget), int(context_length))
        if normalized <= 0:
            raise ValueError(f"Budgets must be positive, got {budget!r}.")
        if normalized not in ordered:
            ordered.append(normalized)
    return ordered


def _budget_label(budget: int) -> str:
    return f"k{int(budget)}"


def _snapkv_recency_window(k_budget: int) -> int:
    return max(0, min(DEFAULT_SNAPKV_RECENCY_CAP, int(k_budget) - DEFAULT_SINK_SIZE))


def _streaming_recency_window(k_budget: int) -> int:
    return max(0, int(k_budget) - DEFAULT_SINK_SIZE)


def _artifact_path(task_key: str, initial_method: str, budget: int) -> Path:
    display_name = get_task_spec(task_key).display_name
    return RESULTS_DIR / f"{task_prefix(display_name)}_{initial_method}_{_budget_label(budget)}_repair_then_reevict.json"


def _generation_kwargs(prepared) -> dict[str, int]:
    return {
        "logical_position_base": int(prepared.context_ids.shape[1]),
        "max_new_tokens": int(prepared.example.max_new_tokens),
    }


def _generate_answer(model, tokenizer, prepared, cache) -> str:
    kwargs = _generation_kwargs(prepared)
    return generate_from_cache(
        model,
        tokenizer,
        prepared.question_ids,
        cache,
        logical_position_base=kwargs["logical_position_base"],
        dense_cache_position_base=len(cache),
        max_new_tokens=kwargs["max_new_tokens"],
    )


def _score_prediction(prediction: str, outputs: list[str]) -> float:
    return float(sample_score(prediction, outputs))


def _slice_payload_matches(
    payload: dict[str, Any] | None,
    *,
    task_key: str,
    initial_method: str,
    budget: int,
    num_samples: int,
    context_length: int,
    dataset_seed_offset: int,
) -> bool:
    if not payload:
        return False
    aggregate = payload.get("aggregate", {})
    return (
        payload.get("schema_version") == SLICE_SCHEMA_VERSION
        and payload.get("task_key") == task_key
        and payload.get("initial_method") == initial_method
        and payload.get("reevict_method") == "snapkv"
        and int(payload.get("k_budget", -1)) == int(budget)
        and int(payload.get("context_length", -1)) == int(context_length)
        and int(payload.get("num_samples", -1)) == int(num_samples)
        and int(payload.get("dataset_seed_offset", -1)) == int(dataset_seed_offset)
        and int(aggregate.get("n_examples", -1)) == int(num_samples)
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(fmean(values)) if values else 0.0


def _pct_true(values: Iterable[bool]) -> float:
    values = list(values)
    return float(sum(bool(value) for value in values) / len(values)) if values else 0.0


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    b_scores = [float(row["condition_b_score"]) for row in rows]
    reevict_scores = [float(row["reevict_score"]) for row in rows]
    lifts = [reevict - base for reevict, base in zip(reevict_scores, b_scores)]
    same_score_flags = [abs(lift) < 1e-9 for lift in lifts]

    return {
        "mean_condition_b": round(_mean(b_scores), 6),
        "mean_reevict": round(_mean(reevict_scores), 6),
        "mean_reevict_lift_over_b": round(_mean(lifts), 6),
        "pct_improved_over_b": round(_pct_true(lift > 0.0 for lift in lifts), 6),
        "pct_equal_to_b": round(_pct_true(same_score_flags), 6),
        "pct_worse_than_b": round(_pct_true(lift < 0.0 for lift in lifts), 6),
        "pct_same_output_as_b": round(_pct_true(row["same_output_as_b"] for row in rows), 6),
        "pct_same_compressed_positions_as_initial": round(
            _pct_true(row["same_compressed_positions_as_initial"] for row in rows),
            6,
        ),
        "mean_compressed_position_jaccard": round(
            _mean(float(row["compressed_position_jaccard"]) for row in rows),
            6,
        ),
        "mean_initial_eviction_s": round(_mean(float(row["initial_eviction_s"]) for row in rows), 6),
        "mean_repair_transfer_ms": round(_mean(float(row["repair_transfer_ms"]) for row in rows), 6),
        "mean_repair_inject_ms": round(_mean(float(row["repair_inject_ms"]) for row in rows), 6),
        "mean_repair_total_ms": round(_mean(float(row["repair_total_ms"]) for row in rows), 6),
        "mean_reevict_policy_s": round(_mean(float(row["reevict_policy_s"]) for row in rows), 6),
        "mean_condition_b_generation_s": round(_mean(float(row["condition_b_generation_s"]) for row in rows), 6),
        "mean_reevict_generation_s": round(_mean(float(row["reevict_generation_s"]) for row in rows), 6),
        "mean_example_wall_s": round(_mean(float(row["example_wall_s"]) for row in rows), 6),
        "mean_restored_token_count": round(_mean(float(row["restored_token_count"]) for row in rows), 2),
        "n_examples": len(rows),
    }


def _position_jaccard(left: list[int], right: list[int]) -> float:
    left_set = set(int(value) for value in left)
    right_set = set(int(value) for value in right)
    union = left_set | right_set
    if not union:
        return 1.0
    return float(len(left_set & right_set) / len(union))


def _run_initial_eviction(*, method: str, full_cache, budget: int):
    if method == "snapkv":
        precompute_policy = SnapKV(
            obs_window_size=DEFAULT_OBS_WINDOW_SIZE,
            sink_size=DEFAULT_SINK_SIZE,
            recency_window=0,
            pooling=DEFAULT_POOLING,
        )
        prepare_start = time.perf_counter()
        snap_cache, importance, obs_q_vecs = precompute_policy.prepare_eviction_inputs(full_cache)
        prepare_s = time.perf_counter() - prepare_start

        policy = SnapKV(
            obs_window_size=DEFAULT_OBS_WINDOW_SIZE,
            sink_size=DEFAULT_SINK_SIZE,
            recency_window=_snapkv_recency_window(budget),
            pooling=DEFAULT_POOLING,
        )
        evict_start = time.perf_counter()
        result = policy.evict_from_precomputed(
            full_cache=snap_cache,
            k_budget=budget,
            importance=importance,
            obs_window_q_vecs=obs_q_vecs,
        )
        policy_s = prepare_s + (time.perf_counter() - evict_start)
        del snap_cache, importance, obs_q_vecs
        return result, policy_s

    if method == "streaming_llm":
        policy = StreamingLLM(
            sink_size=DEFAULT_SINK_SIZE,
            recency_window=_streaming_recency_window(budget),
        )
        evict_start = time.perf_counter()
        result = policy.evict(full_cache, k_budget=budget)
        return result, time.perf_counter() - evict_start

    raise ValueError(f"Unsupported initial eviction method: {method}")


def _run_snapkv_reevict(*, repaired_cache, budget: int):
    precompute_policy = SnapKV(
        obs_window_size=DEFAULT_OBS_WINDOW_SIZE,
        sink_size=DEFAULT_SINK_SIZE,
        recency_window=0,
        pooling=DEFAULT_POOLING,
    )
    prepare_start = time.perf_counter()
    snap_cache, importance, obs_q_vecs = precompute_policy.prepare_eviction_inputs(repaired_cache)
    prepare_s = time.perf_counter() - prepare_start

    policy = SnapKV(
        obs_window_size=DEFAULT_OBS_WINDOW_SIZE,
        sink_size=DEFAULT_SINK_SIZE,
        recency_window=_snapkv_recency_window(budget),
        pooling=DEFAULT_POOLING,
    )
    evict_start = time.perf_counter()
    result = policy.evict_from_precomputed(
        full_cache=snap_cache,
        k_budget=budget,
        importance=importance,
        obs_window_q_vecs=obs_q_vecs,
    )
    policy_s = prepare_s + (time.perf_counter() - evict_start)
    del snap_cache, importance, obs_q_vecs
    return result, policy_s


def _run_slice(
    *,
    model,
    tokenizer,
    task_key: str,
    initial_method: str,
    budget: int,
    num_samples: int,
    context_length: int,
    dataset_seed_offset: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for index in range(int(num_samples)):
        example_start = time.perf_counter()
        example = build_task_example(
            task_key,
            index,
            context_length,
            tokenizer,
            dataset_seed_offset=dataset_seed_offset,
        )
        prepared = prepare_example_for_model(example, tokenizer)
        example_id = f"ex{index + 1:03d}"
        full_cache = build_position_tracked_cache(model, prepared.context_ids)
        gold_outputs = list(prepared.example.outputs)

        eviction_result, initial_eviction_s = _run_initial_eviction(
            method=initial_method,
            full_cache=full_cache,
            budget=budget,
        )

        b_start = time.perf_counter()
        condition_b_output = _generate_answer(model, tokenizer, prepared, eviction_result.compressed)
        condition_b_generation_s = time.perf_counter() - b_start
        condition_b_score = _score_prediction(condition_b_output, gold_outputs)

        device = model_device(model)
        _sync_if_cuda(device)
        transfer_start = time.perf_counter()
        evicted_gpu = eviction_result.evicted.to_device(device, non_blocking=True)
        _sync_if_cuda(device)
        repair_transfer_ms = (time.perf_counter() - transfer_start) * 1000.0

        inject_start = time.perf_counter()
        repaired_cache = inject_kv(
            eviction_result.compressed,
            evicted_gpu,
            evicted_gpu.positions,
        )
        _sync_if_cuda(device)
        repair_inject_ms = (time.perf_counter() - inject_start) * 1000.0
        repair_total_ms = repair_transfer_ms + repair_inject_ms

        reevict_result, reevict_policy_s = _run_snapkv_reevict(
            repaired_cache=repaired_cache,
            budget=budget,
        )

        reevict_start = time.perf_counter()
        reevict_output = _generate_answer(model, tokenizer, prepared, reevict_result.compressed)
        reevict_generation_s = time.perf_counter() - reevict_start
        reevict_score = _score_prediction(reevict_output, gold_outputs)

        example_wall_s = time.perf_counter() - example_start
        same_positions = list(eviction_result.compressed.positions) == list(reevict_result.compressed.positions)
        position_jaccard = _position_jaccard(
            list(eviction_result.compressed.positions),
            list(reevict_result.compressed.positions),
        )

        row = {
            "example_id": example_id,
            "task_key": task_key,
            "initial_method": initial_method,
            "reevict_method": "snapkv",
            "k_budget": int(budget),
            "condition_b_score": round(condition_b_score, 6),
            "reevict_score": round(reevict_score, 6),
            "condition_b_output": condition_b_output,
            "reevict_output": reevict_output,
            "same_output_as_b": bool(condition_b_output == reevict_output),
            "same_compressed_positions_as_initial": bool(same_positions),
            "compressed_position_jaccard": round(position_jaccard, 6),
            "initial_compressed_context_length": int(len(eviction_result.compressed)),
            "reevict_compressed_context_length": int(len(reevict_result.compressed)),
            "restored_token_count": int(len(eviction_result.evicted)),
            "initial_eviction_s": round(initial_eviction_s, 6),
            "repair_transfer_ms": round(repair_transfer_ms, 6),
            "repair_inject_ms": round(repair_inject_ms, 6),
            "repair_total_ms": round(repair_total_ms, 6),
            "reevict_policy_s": round(reevict_policy_s, 6),
            "condition_b_generation_s": round(condition_b_generation_s, 6),
            "reevict_generation_s": round(reevict_generation_s, 6),
            "example_wall_s": round(example_wall_s, 6),
        }
        rows.append(row)

        print(
            f"[{task_key} {initial_method} k={int(budget)} {index + 1:03d}/{int(num_samples):03d}] "
            f"B={row['condition_b_score']:.3f} "
            f"R2={row['reevict_score']:.3f} "
            f"lift={row['reevict_score'] - row['condition_b_score']:.3f} "
            f"repair_ms={row['repair_total_ms']:.3f} "
            f"same_pos={row['same_compressed_positions_as_initial']}",
            flush=True,
        )

        del full_cache, eviction_result, evicted_gpu, repaired_cache, reevict_result
        torch.cuda.empty_cache()

    return rows


def _write_progress(
    *,
    tasks: list[str],
    methods: list[str],
    budgets: list[int],
    num_samples: int,
    context_length: int,
    dataset_seed_offset: int,
    completed_slices: int,
    expected_slices: int,
    last_completed: dict[str, Any] | None,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tasks": tasks,
        "initial_methods": methods,
        "reevict_method": "snapkv",
        "budgets": budgets,
        "num_samples": int(num_samples),
        "context_length": int(context_length),
        "dataset_seed_offset": int(dataset_seed_offset),
        "completed_slices": int(completed_slices),
        "expected_slices": int(expected_slices),
        "completed_examples": int(completed_slices * int(num_samples)),
        "expected_examples": int(expected_slices * int(num_samples)),
        "last_completed": last_completed,
    }
    write_json(PROGRESS_PATH, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Number of deterministic examples per task.")
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH, help="Context length to generate.")
    parser.add_argument("--dataset-seed-offset", type=int, default=DEFAULT_DATASET_SEED_OFFSET, help="Deterministic seed offset.")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), help="Task keys to run.")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), help="Initial eviction methods to run.")
    parser.add_argument("--budgets", nargs="+", default=[str(value) for value in DEFAULT_BUDGETS], help="Budget list.")
    parser.add_argument(
        "--reuse-matching-artifacts",
        action="store_true",
        help="Reuse completed slice artifacts that already match the requested config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_results_dirs()

    normalized_tasks = _normalize_tasks(args.tasks)
    normalized_methods = _normalize_methods(args.methods)
    normalized_budgets = _normalize_budgets(args.budgets, context_length=int(args.context_length))
    expected_slices = len(normalized_tasks) * len(normalized_methods) * len(normalized_budgets)

    model = load_model()
    tokenizer = load_tokenizer()

    summary_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tasks": {},
        "task_keys": normalized_tasks,
        "initial_methods": normalized_methods,
        "reevict_method": "snapkv",
        "budgets": normalized_budgets,
        "num_samples": int(args.num_samples),
        "context_length": int(args.context_length),
        "dataset_seed_offset": int(args.dataset_seed_offset),
    }

    completed_slices = 0
    last_completed: dict[str, Any] | None = None

    for task_key in normalized_tasks:
        task_payload = summary_payload["tasks"].setdefault(
            task_key,
            {"display_name": get_task_spec(task_key).display_name},
        )
        for method in normalized_methods:
            method_payload = task_payload.setdefault(method, {})
            for budget in normalized_budgets:
                artifact_path = _artifact_path(task_key, method, budget)
                payload = None
                if args.reuse_matching_artifacts:
                    payload = _load_json(artifact_path)
                    if not _slice_payload_matches(
                        payload,
                        task_key=task_key,
                        initial_method=method,
                        budget=budget,
                        num_samples=int(args.num_samples),
                        context_length=int(args.context_length),
                        dataset_seed_offset=int(args.dataset_seed_offset),
                    ):
                        payload = None

                if payload is None:
                    print(
                        f"Starting slice task={task_key} initial_method={method} k={int(budget)} "
                        f"examples={int(args.num_samples)}",
                        flush=True,
                    )
                    rows = _run_slice(
                        model=model,
                        tokenizer=tokenizer,
                        task_key=task_key,
                        initial_method=method,
                        budget=budget,
                        num_samples=int(args.num_samples),
                        context_length=int(args.context_length),
                        dataset_seed_offset=int(args.dataset_seed_offset),
                    )
                    aggregate = _summarize_rows(rows)
                    payload = {
                        "schema_version": SLICE_SCHEMA_VERSION,
                        "task_key": task_key,
                        "display_name": get_task_spec(task_key).display_name,
                        "initial_method": method,
                        "reevict_method": "snapkv",
                        "k_budget": int(budget),
                        "num_samples": int(args.num_samples),
                        "context_length": int(args.context_length),
                        "dataset_seed_offset": int(args.dataset_seed_offset),
                        "aggregate": aggregate,
                        "per_example": rows,
                    }
                    write_json(artifact_path, payload)
                else:
                    print(
                        f"Reusing slice task={task_key} initial_method={method} k={int(budget)} from {artifact_path}",
                        flush=True,
                    )

                completed_slices += 1
                aggregate = payload["aggregate"]
                method_payload[_budget_label(budget)] = {
                    "aggregate": aggregate,
                    "artifact_path": str(artifact_path),
                }
                last_completed = {
                    "task_key": task_key,
                    "initial_method": method,
                    "reevict_method": "snapkv",
                    "k_budget": int(budget),
                    "artifact_path": str(artifact_path),
                    "mean_condition_b": aggregate["mean_condition_b"],
                    "mean_reevict": aggregate["mean_reevict"],
                    "mean_reevict_lift_over_b": aggregate["mean_reevict_lift_over_b"],
                    "mean_example_wall_s": aggregate["mean_example_wall_s"],
                }
                _write_progress(
                    tasks=normalized_tasks,
                    methods=normalized_methods,
                    budgets=normalized_budgets,
                    num_samples=int(args.num_samples),
                    context_length=int(args.context_length),
                    dataset_seed_offset=int(args.dataset_seed_offset),
                    completed_slices=completed_slices,
                    expected_slices=expected_slices,
                    last_completed=last_completed,
                )
                write_json(SUMMARY_PATH, summary_payload)

    write_json(SUMMARY_PATH, summary_payload)
    print(f"Completed {completed_slices}/{expected_slices} slices. Summary written to {SUMMARY_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
