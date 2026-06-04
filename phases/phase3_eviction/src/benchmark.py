"""Resumable Phase 3 benchmark runner."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from ._repo import PHASE_ROOT, REPO_ROOT
from .eviction import EvictionResult, QueryAwareSnapKV, SnapKV, StreamingLLM, log_eviction
from .runtime import DEGRADATION_DIR, EVICTION_LOG_DIR, build_position_tracked_cache, ensure_results_dirs, write_json
from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache, slice_kv
from phases.phase2_kv_cache.src.runtime import generate_from_cache, inspect_environment, load_model, load_tokenizer
from phases.phase1_degradation.phase1.evaluation import breakdown, classify_error, first_broken_hop, matched_outputs, sample_score, task_prefix
from phases.phase1_degradation.phase1.inference import prepare_example_for_model
from phases.phase1_degradation.phase1.models import PredictionRecord, SpanSurvival, TaskExample
from phases.phase1_degradation.phase1.task_registry import build_task_example, get_task_spec

RUN_SCHEMA_VERSION = "phase3-benchmark-v1"
GENERATOR_COMPAT_VERSION = "phase1-current-generator-noise-filler-2026-04-21"
DEFAULT_CONTEXT_LENGTH = 32768
DEFAULT_TASKS = ("vt_4hop", "mq_niah_4q", "s_niah")
DEFAULT_METHODS = ("fullkv", "snapkv", "query_aware_snapkv", "streaming_llm")
DEFAULT_BUDGETS = (128, 256, 512, 1024, 2048, DEFAULT_CONTEXT_LENGTH)
DEFAULT_SINK_SIZE = 4
DEFAULT_OBS_WINDOW_SIZE = 32
DEFAULT_POOLING = "max"
RAW_RESULTS_DIR = PHASE_ROOT / "results" / "phase3_raw_examples" / GENERATOR_COMPAT_VERSION
BENCHMARK_LOG_DIR = EVICTION_LOG_DIR / "benchmark" / GENERATOR_COMPAT_VERSION


def _normalize_tasks(tasks: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    for task_name in tasks:
        if task_name not in ordered:
            ordered.append(task_name)
    return ordered


def _normalize_methods(methods: Iterable[str]) -> list[str]:
    allowed = {"fullkv", "snapkv", "query_aware_snapkv", "streaming_llm"}
    ordered: list[str] = []
    for method in methods:
        normalized = str(method).lower()
        if normalized not in allowed:
            raise ValueError(f"Unsupported Phase 3 method: {method}")
        if normalized not in ordered:
            ordered.append(normalized)
    return ordered


def _normalize_budgets(budgets: Iterable[int | str], *, context_length: int) -> list[int]:
    ordered: list[int] = []
    for budget in budgets:
        if isinstance(budget, str) and budget.lower() == "full":
            normalized = int(context_length)
        else:
            normalized = int(budget)
        if normalized <= 0:
            raise ValueError(f"Budgets must be positive, got {budget!r}.")
        normalized = min(normalized, int(context_length))
        if normalized not in ordered:
            ordered.append(normalized)
    return ordered


def budget_label(budget: int, *, context_length: int) -> str:
    return "FullKV" if int(budget) >= int(context_length) else f"k{int(budget)}"


def _raw_example_path(task_key: str, example_index: int) -> Path:
    return RAW_RESULTS_DIR / task_key / f"ex{example_index + 1:03d}.json"


def _empty_cpu_cache_like(full_cache: PositionTrackedCache) -> PositionTrackedCache:
    layers = tuple(
        (
            key[:, :, :0, :].detach().to("cpu").contiguous(),
            value[:, :, :0, :].detach().to("cpu").contiguous(),
        )
        for key, value in full_cache.kv
    )
    return PositionTrackedCache(layers, [])


def _full_kv_result(full_cache: PositionTrackedCache) -> EvictionResult:
    return EvictionResult(
        compressed=full_cache,
        evicted=_empty_cpu_cache_like(full_cache),
        importance_scores={int(position): 1.0 for position in full_cache.positions},
        obs_window_q_vecs=torch.zeros((len(full_cache.kv), 1, int(full_cache.kv[0][0].shape[-1])), dtype=torch.float32),
    )


def _build_examples(
    task_key: str,
    *,
    tokenizer,
    context_length: int,
    num_samples: int,
    dataset_seed_offset: int,
) -> list[TaskExample]:
    return [
        build_task_example(
            task_key,
            index,
            context_length,
            tokenizer,
            dataset_seed_offset=dataset_seed_offset,
        )
        for index in range(num_samples)
    ]


def _compute_span_survival(example: TaskExample, prepared, kept_positions: set[int]) -> list[SpanSurvival]:
    spans: list[SpanSurvival] = []
    for span in example.relevant_spans:
        token_positions = [int(position) for position in prepared.span_token_positions.get(span.name, [])]
        kept_count = sum(position in kept_positions for position in token_positions)
        total_count = len(token_positions)
        spans.append(
            SpanSurvival(
                name=span.name,
                kind=span.kind,
                depth_fraction=span.depth_fraction,
                survival_fraction=(kept_count / total_count) if total_count else 0.0,
                kept_token_count=kept_count,
                total_token_count=total_count,
                metadata=span.metadata,
            )
        )
    return spans


def _make_prediction_record(
    *,
    example: TaskExample,
    prediction: str,
    method: str,
    budget: int,
    compressed_context_length: int,
    spans: list[SpanSurvival],
) -> PredictionRecord:
    matched = matched_outputs(prediction, example.outputs)
    return PredictionRecord(
        index=example.index,
        task_name=example.task_name,
        task_family=example.task_family,
        context_length=example.target_context_length,
        algorithm=method,
        budget=budget,
        condition="phase3",
        outputs=list(example.outputs),
        prediction=prediction,
        sample_score=sample_score(prediction, example.outputs),
        error_type=classify_error(example, prediction, matched, spans),
        matched_outputs=matched,
        span_survival=spans,
        compressed_context_length=compressed_context_length,
        trace_path=None,
        metadata=dict(example.metadata),
    )


def _log_dir(task_display_name: str, method: str, budget: int, *, context_length: int) -> Path:
    return BENCHMARK_LOG_DIR / task_prefix(task_display_name) / method / budget_label(budget, context_length=context_length)


def _result_row(
    *,
    task_display_name: str,
    prepared,
    result: EvictionResult,
    prediction_record: PredictionRecord,
    budget: int,
    method: str,
    generation_duration_s: float,
    policy_duration_s: float,
    total_duration_s: float,
    context_length: int,
) -> dict[str, Any]:
    relevant_positions = [
        int(token_positions[0])
        for token_positions in (prepared.span_token_positions.get(span.name, []) for span in prepared.example.relevant_spans)
        if token_positions
    ]
    example_id = f"ex{prepared.example.index + 1:03d}"
    log_dir = _log_dir(task_display_name, method, budget, context_length=context_length)
    log_eviction(
        result,
        example_id=example_id,
        task=task_display_name,
        task_relevant_positions=relevant_positions,
        log_dir=log_dir,
        metadata={
            "method": method,
            "context_length": context_length,
            "budget": budget,
            "budget_label": budget_label(budget, context_length=context_length),
        },
    )
    row = {
        "example_id": prepared.example.index + 1,
        "task": task_display_name,
        "task_key": prepared.example.task_name,
        "method": method,
        "k_budget": budget,
        "budget_label": budget_label(budget, context_length=context_length),
        "context_length": context_length,
        "raw_model_output": prediction_record.prediction,
        "gold_answer": ", ".join(prepared.example.outputs),
        "gold_outputs": list(prepared.example.outputs),
        "correct": round(prediction_record.sample_score, 6)
        if len(prepared.example.outputs) > 1
        else bool(prediction_record.sample_score == 1.0),
        "score_fraction": round(prediction_record.sample_score, 6),
        "matched_count": len(prediction_record.matched_outputs),
        "matched_outputs": list(prediction_record.matched_outputs),
        "error_type": prediction_record.error_type,
        "task_relevant_positions": [positions[0] if positions else -1 for positions in (prepared.span_token_positions.get(span.name, []) for span in prepared.example.relevant_spans)],
        "task_relevant_survived": [span.survival_fraction > 0.0 for span in prediction_record.span_survival],
        "task_relevant_spans": [
            {
                "name": span.name,
                "kind": span.kind,
                "depth_fraction": span.depth_fraction,
                "survival_fraction": round(span.survival_fraction, 6),
                "kept_token_count": int(span.kept_token_count),
                "total_token_count": int(span.total_token_count),
                "metadata": dict(span.metadata),
            }
            for span in prediction_record.span_survival
        ],
        "compressed_context_length": int(prediction_record.compressed_context_length),
        "first_broken_hop": None if first_broken_hop(prepared.example, prediction_record.span_survival) is None else first_broken_hop(prepared.example, prediction_record.span_survival)[0],
        "first_broken_hop_depth": None
        if first_broken_hop(prepared.example, prediction_record.span_survival) is None
        else round(first_broken_hop(prepared.example, prediction_record.span_survival)[1], 4),
        "eviction_log_path": str(log_dir / f"{example_id}.json"),
        "q_vectors_path": str(log_dir / f"{example_id}_qvecs.pt"),
        "policy_duration_s": round(policy_duration_s, 6),
        "generation_duration_s": round(generation_duration_s, 6),
        "total_duration_s": round(total_duration_s, 6),
    }
    return row


def _row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["method"]), int(row["k_budget"])


def _load_cached_example_rows(
    task_key: str,
    *,
    example_index: int,
    context_length: int,
    dataset_seed_offset: int,
) -> list[dict[str, Any]]:
    path = _raw_example_path(task_key, example_index)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != RUN_SCHEMA_VERSION:
        return []
    if payload.get("generator_compat_version") != GENERATOR_COMPAT_VERSION:
        return []
    if int(payload.get("context_length", -1)) != int(context_length):
        return []
    if int(payload.get("dataset_seed_offset", -1)) != int(dataset_seed_offset):
        return []
    if str(payload.get("task_key")) != task_key:
        return []
    return list(payload.get("records", []))


def _save_example_rows(
    task_key: str,
    *,
    example_index: int,
    context_length: int,
    dataset_seed_offset: int,
    task_display_name: str,
    rows: list[dict[str, Any]],
) -> None:
    path = _raw_example_path(task_key, example_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": RUN_SCHEMA_VERSION,
        "generator_compat_version": GENERATOR_COMPAT_VERSION,
        "task": task_display_name,
        "task_key": task_key,
        "example_id": example_index + 1,
        "context_length": context_length,
        "dataset_seed_offset": int(dataset_seed_offset),
        "records": sorted(rows, key=lambda row: (str(row["method"]), int(row["k_budget"]))),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _generate_row(
    *,
    model,
    tokenizer,
    prepared,
    task_display_name: str,
    method: str,
    budget: int,
    result: EvictionResult,
    policy_duration_s: float,
) -> dict[str, Any]:
    logical_position_base = int(prepared.context_ids.shape[1])
    dense_cache_position_base = int(len(result.compressed))
    generation_start = time.perf_counter()
    prediction = generate_from_cache(
        model,
        tokenizer,
        prepared.question_ids,
        result.compressed,
        logical_position_base=logical_position_base,
        dense_cache_position_base=dense_cache_position_base,
        max_new_tokens=prepared.example.max_new_tokens,
    )
    generation_duration_s = time.perf_counter() - generation_start
    kept_positions = set(int(position) for position in result.compressed.positions)
    spans = _compute_span_survival(prepared.example, prepared, kept_positions)
    prediction_record = _make_prediction_record(
        example=prepared.example,
        prediction=prediction,
        method=method,
        budget=budget,
        compressed_context_length=len(result.compressed),
        spans=spans,
    )
    return _result_row(
        task_display_name=task_display_name,
        prepared=prepared,
        result=result,
        prediction_record=prediction_record,
        budget=budget,
        method=method,
        generation_duration_s=generation_duration_s,
        policy_duration_s=policy_duration_s,
        total_duration_s=policy_duration_s + generation_duration_s,
        context_length=int(prepared.example.target_context_length),
    )


def _run_example(
    *,
    model,
    tokenizer,
    prepared,
    methods: list[str],
    budgets: list[int],
    force: bool,
    dataset_seed_offset: int,
) -> list[dict[str, Any]]:
    task_key = prepared.example.task_name
    task_display_name = get_task_spec(task_key).display_name
    context_length = int(prepared.example.target_context_length)
    cached_rows = [] if force else _load_cached_example_rows(
        task_key,
        example_index=prepared.example.index,
        context_length=context_length,
        dataset_seed_offset=dataset_seed_offset,
    )
    required_keys: set[tuple[str, int]] = set()
    if "fullkv" in methods and context_length in budgets:
        required_keys.add(("fullkv", context_length))
    non_full_budgets = [budget for budget in budgets if budget < context_length]
    for method in methods:
        if method == "fullkv":
            continue
        for budget in non_full_budgets:
            required_keys.add((method, budget))

    cached_by_key = {_row_key(row): row for row in cached_rows}
    missing_keys = required_keys.difference(cached_by_key)
    if not missing_keys:
        return [cached_by_key[key] for key in sorted(cached_by_key, key=lambda item: (item[0], item[1])) if key in required_keys]

    prefill_start = time.perf_counter()
    full_cache = build_position_tracked_cache(model, prepared.context_ids)
    prefill_duration_s = time.perf_counter() - prefill_start
    print(
        f"[phase3] task={task_display_name} example={prepared.example.index + 1:03d} "
        f"prefill={prefill_duration_s:.2f}s missing={len(missing_keys)}",
        flush=True,
    )

    rows = list(cached_by_key.values())
    try:
        if ("fullkv", context_length) in missing_keys:
            baseline_start = time.perf_counter()
            baseline_result = _full_kv_result(full_cache)
            rows.append(
                _generate_row(
                    model=model,
                    tokenizer=tokenizer,
                    prepared=prepared,
                    task_display_name=task_display_name,
                    method="fullkv",
                    budget=context_length,
                    result=baseline_result,
                    policy_duration_s=time.perf_counter() - baseline_start,
                )
            )

        if "snapkv" in methods and any(("snapkv", budget) in missing_keys for budget in non_full_budgets):
            snap_policy = SnapKV(
                obs_window_size=DEFAULT_OBS_WINDOW_SIZE,
                sink_size=DEFAULT_SINK_SIZE,
                recency_window=max(0, max(non_full_budgets, default=0) - DEFAULT_SINK_SIZE),
                pooling=DEFAULT_POOLING,
            )
            snap_start = time.perf_counter()
            snap_cache, snap_importance, snap_obs = snap_policy.prepare_eviction_inputs(full_cache)
            snap_prepare_s = time.perf_counter() - snap_start
            for budget in non_full_budgets:
                if ("snapkv", budget) not in missing_keys:
                    continue
                policy_start = time.perf_counter()
                result = snap_policy.evict_from_precomputed(
                    full_cache=snap_cache,
                    k_budget=budget,
                    importance=snap_importance,
                    obs_window_q_vecs=snap_obs,
                )
                rows.append(
                    _generate_row(
                        model=model,
                        tokenizer=tokenizer,
                        prepared=prepared,
                        task_display_name=task_display_name,
                        method="snapkv",
                        budget=budget,
                        result=result,
                        policy_duration_s=snap_prepare_s + (time.perf_counter() - policy_start),
                    )
                )

        if "query_aware_snapkv" in methods and any(("query_aware_snapkv", budget) in missing_keys for budget in non_full_budgets):
            qa_policy = QueryAwareSnapKV(
                model,
                obs_window_size=DEFAULT_OBS_WINDOW_SIZE,
                sink_size=DEFAULT_SINK_SIZE,
                recency_window=max(0, max(non_full_budgets, default=0) - DEFAULT_SINK_SIZE),
                pooling=DEFAULT_POOLING,
            )
            qa_start = time.perf_counter()
            qa_cache, qa_importance, qa_obs = qa_policy.prepare_eviction_inputs(full_cache, obs_window=prepared.question_ids)
            qa_prepare_s = time.perf_counter() - qa_start
            for budget in non_full_budgets:
                if ("query_aware_snapkv", budget) not in missing_keys:
                    continue
                policy_start = time.perf_counter()
                result = qa_policy.evict_from_precomputed(
                    full_cache=qa_cache,
                    k_budget=budget,
                    importance=qa_importance,
                    obs_window_q_vecs=qa_obs,
                )
                rows.append(
                    _generate_row(
                        model=model,
                        tokenizer=tokenizer,
                        prepared=prepared,
                        task_display_name=task_display_name,
                        method="query_aware_snapkv",
                        budget=budget,
                        result=result,
                        policy_duration_s=qa_prepare_s + (time.perf_counter() - policy_start),
                    )
                )

        if "streaming_llm" in methods:
            for budget in non_full_budgets:
                if ("streaming_llm", budget) not in missing_keys:
                    continue
                streaming_start = time.perf_counter()
                streaming_policy = StreamingLLM(
                    sink_size=DEFAULT_SINK_SIZE,
                    recency_window=max(0, budget - DEFAULT_SINK_SIZE),
                )
                result = streaming_policy.evict(full_cache, k_budget=budget)
                rows.append(
                    _generate_row(
                        model=model,
                        tokenizer=tokenizer,
                        prepared=prepared,
                        task_display_name=task_display_name,
                        method="streaming_llm",
                        budget=budget,
                        result=result,
                        policy_duration_s=time.perf_counter() - streaming_start,
                    )
                )
    finally:
        del full_cache
        torch.cuda.empty_cache()

    deduped = {_row_key(row): row for row in rows}
    final_rows = [deduped[key] for key in sorted(required_keys, key=lambda item: (item[0], item[1]))]
    _save_example_rows(
        task_key,
        example_index=prepared.example.index,
        context_length=context_length,
        dataset_seed_offset=dataset_seed_offset,
        task_display_name=task_display_name,
        rows=final_rows,
    )
    return final_rows


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row["score_fraction"]) for row in rows]
    failure_types = [row["error_type"] for row in rows if row["error_type"] is not None]
    span_survivals = [
        float(span["survival_fraction"])
        for row in rows
        for span in row["task_relevant_spans"]
    ]
    summary: dict[str, Any] = {
        "num_examples": len(rows),
        "mean_score_fraction": round(sum(scores) / len(scores), 6) if scores else 0.0,
        "mean_score_percent": round((sum(scores) / len(scores) * 100.0), 4) if scores else 0.0,
        "error_breakdown": breakdown(failure_types),
        "eviction_survival_rate": round(sum(span_survivals) / len(span_survivals), 6) if span_survivals else 0.0,
        "mean_policy_duration_s": round(sum(float(row["policy_duration_s"]) for row in rows) / len(rows), 6) if rows else 0.0,
        "mean_generation_duration_s": round(sum(float(row["generation_duration_s"]) for row in rows) / len(rows), 6) if rows else 0.0,
        "mean_total_duration_s": round(sum(float(row["total_duration_s"]) for row in rows) / len(rows), 6) if rows else 0.0,
    }
    if rows and len(rows[0]["gold_outputs"]) > 1:
        matched_counts = [int(row["matched_count"]) for row in rows]
        summary["mean_recall"] = round(sum(matched_counts) / len(matched_counts), 6)
        summary["full_recall_rate"] = round(sum(score == 1.0 for score in scores) / len(scores), 6)
    else:
        summary["accuracy"] = round(sum(scores) / len(scores), 6) if scores else 0.0
    hop_groups: dict[str, list[float]] = {}
    broken_counts: dict[str, int] = {}
    for row in rows:
        for span in row["task_relevant_spans"]:
            if span["kind"] == "hop":
                hop_groups.setdefault(span["name"], []).append(float(span["survival_fraction"]))
        if row["first_broken_hop"] is not None:
            key = f"hop_{int(row['first_broken_hop'])}"
            broken_counts[key] = broken_counts.get(key, 0) + 1
    if hop_groups:
        summary["hop_survival"] = {
            hop_name: round(sum(values) / len(values), 6)
            for hop_name, values in sorted(hop_groups.items())
        }
    if broken_counts:
        total_broken = sum(broken_counts.values())
        summary["chain_break_hop_distribution"] = {
            hop_name: round(count / total_broken, 6)
            for hop_name, count in sorted(broken_counts.items())
        }
    return summary


def _build_task_degradation(
    *,
    task_key: str,
    rows: list[dict[str, Any]],
    budgets: list[int],
    methods: list[str],
    context_length: int,
    num_samples: int,
) -> dict[str, Any]:
    task_spec = get_task_spec(task_key)
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), {}).setdefault(int(row["k_budget"]), []).append(row)

    summary_methods: dict[str, Any] = {}
    for method in methods:
        method_rows = grouped.get(method, {})
        if method == "fullkv":
            target_budgets = [context_length] if context_length in budgets else []
        else:
            target_budgets = [budget for budget in budgets if budget < context_length]
        budget_summary = {
            budget_label(budget, context_length=context_length): _summarize_rows(method_rows.get(budget, []))
            for budget in target_budgets
        }
        if budget_summary:
            summary_methods[method] = budget_summary

    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "generator_compat_version": GENERATOR_COMPAT_VERSION,
        "task": task_spec.display_name,
        "task_key": task_key,
        "context_length": context_length,
        "num_samples_requested": num_samples,
        "budget_order": [budget_label(budget, context_length=context_length) for budget in budgets],
        "methods": summary_methods,
        "raw_record_count": len(rows),
    }


def run_phase3_benchmark(
    *,
    num_samples: int,
    tasks: Iterable[str] = DEFAULT_TASKS,
    methods: Iterable[str] = DEFAULT_METHODS,
    budgets: Iterable[int | str] = DEFAULT_BUDGETS,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    dataset_seed_offset: int = 0,
    label: str = "manual",
    force: bool = False,
) -> dict[str, Any]:
    ensure_results_dirs()
    normalized_tasks = _normalize_tasks(tasks)
    normalized_methods = _normalize_methods(methods)
    normalized_budgets = _normalize_budgets(budgets, context_length=context_length)
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")

    tokenizer = load_tokenizer()
    model = load_model()
    start_time = time.perf_counter()
    all_rows: list[dict[str, Any]] = []

    try:
        environment = inspect_environment(model, tokenizer)
        for task_key in normalized_tasks:
            task_spec = get_task_spec(task_key)
            examples = _build_examples(
                task_key,
                tokenizer=tokenizer,
                context_length=context_length,
                num_samples=num_samples,
                dataset_seed_offset=dataset_seed_offset,
            )
            for example in examples:
                prepared = prepare_example_for_model(example, tokenizer)
                rows = _run_example(
                    model=model,
                    tokenizer=tokenizer,
                    prepared=prepared,
                    methods=normalized_methods,
                    budgets=normalized_budgets,
                    force=force,
                    dataset_seed_offset=dataset_seed_offset,
                )
                all_rows.extend(rows)
                print(
                    f"[phase3] completed task={task_spec.display_name} example={example.index + 1:03d}/{num_samples}",
                    flush=True,
                )
    finally:
        del model
        torch.cuda.empty_cache()

    summary_dir = DEGRADATION_DIR / str(label)
    summary_dir.mkdir(parents=True, exist_ok=True)
    task_summaries: dict[str, Any] = {}
    for task_key in normalized_tasks:
        task_rows = [row for row in all_rows if row["task_key"] == task_key]
        task_summary = _build_task_degradation(
            task_key=task_key,
            rows=task_rows,
            budgets=normalized_budgets,
            methods=normalized_methods,
            context_length=context_length,
            num_samples=num_samples,
        )
        task_summaries[task_key] = task_summary
        task_path = summary_dir / f"{task_prefix(get_task_spec(task_key).display_name)}_degradation.json"
        write_json(task_path, task_summary)

    elapsed_s = time.perf_counter() - start_time
    summary = {
        "schema_version": RUN_SCHEMA_VERSION,
        "generator_compat_version": GENERATOR_COMPAT_VERSION,
        "label": label,
        "context_length": context_length,
        "num_samples": num_samples,
        "dataset_seed_offset": int(dataset_seed_offset),
        "tasks": normalized_tasks,
        "methods": normalized_methods,
        "budgets": [budget_label(budget, context_length=context_length) for budget in normalized_budgets],
        "environment": environment,
        "elapsed_s": round(elapsed_s, 6),
        "elapsed_minutes": round(elapsed_s / 60.0, 4),
        "raw_record_count": len(all_rows),
        "summary_dir": str(summary_dir),
        "task_summaries": task_summaries,
    }
    write_json(summary_dir / "phase3_summary.json", summary)
    return summary


__all__ = [
    "DEFAULT_BUDGETS",
    "DEFAULT_CONTEXT_LENGTH",
    "DEFAULT_METHODS",
    "DEFAULT_TASKS",
    "GENERATOR_COMPAT_VERSION",
    "RUN_SCHEMA_VERSION",
    "budget_label",
    "run_phase3_benchmark",
]
