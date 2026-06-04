#!/usr/bin/env python3
"""Run the K -> query-attention top-up repair sweep."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
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
from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache, inject_kv, slice_kv  # noqa: E402
from phases.phase2_kv_cache.src.runtime import generate_from_cache, load_model, load_tokenizer, model_device  # noqa: E402
from phases.phase3_eviction.src.benchmark import (  # noqa: E402
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_OBS_WINDOW_SIZE,
    DEFAULT_POOLING,
    DEFAULT_SINK_SIZE,
)
from phases.phase3_eviction.src.eviction import QueryAwareSnapKV, SnapKV, StreamingLLM  # noqa: E402
from phases.phase3_eviction.src.runtime import build_position_tracked_cache, write_json  # noqa: E402

RESULTS_DIR = PHASE_ROOT / "results" / "phase5_budgeted_repair"
LOG_DIR = RESULTS_DIR / "logs"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
PROGRESS_PATH = RESULTS_DIR / "progress.json"

DEFAULT_TASKS = tuple(TASK_SPECS.keys())
DEFAULT_METHODS = ("snapkv", "streaming_llm")
DEFAULT_BUDGETS = (256, 512, 1024)
DEFAULT_NUM_SAMPLES = 100
DEFAULT_DATASET_SEED_OFFSET = 0
DEFAULT_SNAPKV_RECENCY_CAP = 1024
DEFAULT_REPAIR_RATIO = 0.10
REPAIR_SELECTOR = "query_aware_attention_topup"
SCHEMA_VERSION = "phase5-budgeted-repair-v1"
SLICE_SCHEMA_VERSION = "phase5-budgeted-repair-slice-v1"


@dataclass(frozen=True)
class _SlicePlan:
    initial_method: str
    budget: int
    target_total_budget: int
    repair_needed: bool
    artifact_path: Path


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
    return RESULTS_DIR / f"{task_prefix(display_name)}_{initial_method}_{_budget_label(budget)}_budgeted_repair.json"


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


def _target_total_budget(*, budget: int, context_length: int, repair_ratio: float) -> int:
    return min(int(context_length), int(math.ceil(int(budget) * (1.0 + float(repair_ratio)))))


def _move_tensor_to_cpu_pinned(tensor: torch.Tensor) -> torch.Tensor:
    cpu_tensor = tensor.detach().to("cpu").contiguous()
    try:
        return cpu_tensor.pin_memory()
    except RuntimeError:
        return cpu_tensor


def _cache_to_cpu_pinned(cache: PositionTrackedCache) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (
            _move_tensor_to_cpu_pinned(key),
            _move_tensor_to_cpu_pinned(value),
        )
        for key, value in cache.kv
    )


def _materialize_partition(
    *,
    full_cache: PositionTrackedCache,
    keep_indices: list[int],
) -> tuple[PositionTrackedCache, PositionTrackedCache]:
    keep_indices = sorted(dict.fromkeys(int(index) for index in keep_indices))
    keep_set = set(keep_indices)
    evict_indices = [index for index in range(len(full_cache)) if index not in keep_set]

    compressed = slice_kv(full_cache, keep_indices)
    evicted = slice_kv(full_cache, evict_indices)
    if not isinstance(compressed, PositionTrackedCache) or not isinstance(evicted, PositionTrackedCache):
        raise RuntimeError("Phase 2 slice_kv did not preserve position tracking.")

    evicted_cpu = PositionTrackedCache(_cache_to_cpu_pinned(evicted), list(evicted.positions))
    return compressed, evicted_cpu


def _prepare_snapkv_importance(full_cache: PositionTrackedCache) -> tuple[torch.Tensor, torch.Tensor, float]:
    policy = SnapKV(
        obs_window_size=DEFAULT_OBS_WINDOW_SIZE,
        sink_size=DEFAULT_SINK_SIZE,
        recency_window=0,
        pooling=DEFAULT_POOLING,
    )
    device = full_cache.device
    _sync_if_cuda(device)
    start = time.perf_counter()
    importance = policy._normalize_importance(
        policy._score_tokens(full_cache),
        seq_len=len(full_cache),
        device=full_cache.device,
    )
    importance_cpu = importance.detach().to("cpu", dtype=torch.float32)
    _sync_if_cuda(device)
    return importance, importance_cpu, time.perf_counter() - start


def _materialize_snapkv_eviction_from_importance(
    *,
    full_cache: PositionTrackedCache,
    budget: int,
    importance: torch.Tensor,
) -> tuple[PositionTrackedCache, PositionTrackedCache, float]:
    policy = SnapKV(
        obs_window_size=DEFAULT_OBS_WINDOW_SIZE,
        sink_size=DEFAULT_SINK_SIZE,
        recency_window=_snapkv_recency_window(budget),
        pooling=DEFAULT_POOLING,
    )
    device = full_cache.device
    _sync_if_cuda(device)
    start = time.perf_counter()

    seq_len = len(full_cache)
    budget = policy._normalize_budget(seq_len, budget)
    mandatory_indices = policy._structural_keep_indices(seq_len=seq_len, k_budget=budget)
    mandatory_set = set(mandatory_indices)
    remaining_slots = max(0, budget - len(mandatory_indices))
    candidate_indices = [index for index in range(seq_len) if index not in mandatory_set]

    selected_indices: list[int] = []
    if remaining_slots > 0 and candidate_indices:
        candidate_tensor = torch.tensor(candidate_indices, device=importance.device, dtype=torch.long)
        candidate_scores = torch.index_select(importance, 0, candidate_tensor)
        topk = min(remaining_slots, len(candidate_indices))
        topk_indices = torch.topk(candidate_scores, k=topk, largest=True, sorted=False).indices
        selected_indices = torch.index_select(candidate_tensor, 0, topk_indices).tolist()

    keep_indices = sorted(mandatory_set | set(selected_indices))
    compressed, evicted_cpu = _materialize_partition(full_cache=full_cache, keep_indices=keep_indices)
    _sync_if_cuda(device)
    return compressed, evicted_cpu, time.perf_counter() - start


def _materialize_streaming_eviction(
    *,
    full_cache: PositionTrackedCache,
    budget: int,
) -> tuple[PositionTrackedCache, PositionTrackedCache, float]:
    policy = StreamingLLM(
        sink_size=DEFAULT_SINK_SIZE,
        recency_window=_streaming_recency_window(budget),
    )
    device = full_cache.device
    _sync_if_cuda(device)
    start = time.perf_counter()
    keep_indices = policy._structural_keep_indices(seq_len=len(full_cache), k_budget=budget)
    compressed, evicted_cpu = _materialize_partition(full_cache=full_cache, keep_indices=keep_indices)
    _sync_if_cuda(device)
    return compressed, evicted_cpu, time.perf_counter() - start


def _slice_payload_matches(
    payload: dict[str, Any] | None,
    *,
    task_key: str,
    initial_method: str,
    budget: int,
    num_samples: int,
    context_length: int,
    dataset_seed_offset: int,
    repair_ratio: float,
) -> bool:
    if not payload:
        return False
    aggregate = payload.get("aggregate", {})
    return (
        payload.get("schema_version") == SLICE_SCHEMA_VERSION
        and payload.get("task_key") == task_key
        and payload.get("initial_method") == initial_method
        and payload.get("repair_selector") == REPAIR_SELECTOR
        and abs(float(payload.get("repair_ratio", -1.0)) - float(repair_ratio)) < 1e-12
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
    repaired_scores = [float(row["repaired_score"]) for row in rows]
    lifts = [repaired - base for repaired, base in zip(repaired_scores, b_scores)]
    same_score_flags = [abs(lift) < 1e-9 for lift in lifts]

    return {
        "mean_condition_b": round(_mean(b_scores), 6),
        "mean_repaired": round(_mean(repaired_scores), 6),
        "mean_lift_over_b": round(_mean(lifts), 6),
        "pct_improved_over_b": round(_pct_true(lift > 0.0 for lift in lifts), 6),
        "pct_equal_to_b": round(_pct_true(same_score_flags), 6),
        "pct_worse_than_b": round(_pct_true(lift < 0.0 for lift in lifts), 6),
        "pct_same_output_as_b": round(_pct_true(row["same_output_as_b"] for row in rows), 6),
        "mean_initial_eviction_s": round(_mean(float(row["initial_eviction_s"]) for row in rows), 6),
        "mean_attention_recompute_s": round(_mean(float(row["attention_recompute_s"]) for row in rows), 6),
        "mean_repair_selection_s": round(_mean(float(row["repair_selection_s"]) for row in rows), 6),
        "mean_repair_transfer_ms": round(_mean(float(row["repair_transfer_ms"]) for row in rows), 6),
        "mean_repair_inject_ms": round(_mean(float(row["repair_inject_ms"]) for row in rows), 6),
        "mean_repair_total_ms": round(_mean(float(row["repair_total_ms"]) for row in rows), 6),
        "mean_condition_b_generation_s": round(_mean(float(row["condition_b_generation_s"]) for row in rows), 6),
        "mean_repaired_generation_s": round(_mean(float(row["repaired_generation_s"]) for row in rows), 6),
        "mean_example_wall_s": round(_mean(float(row["example_wall_s"]) for row in rows), 6),
        "mean_restored_token_count": round(_mean(float(row["restored_token_count"]) for row in rows), 2),
        "mean_target_total_budget": round(_mean(float(row["target_total_budget"]) for row in rows), 2),
        "mean_repaired_context_length": round(_mean(float(row["repaired_context_length"]) for row in rows), 2),
        "mean_selected_attention_score": round(_mean(float(row["selected_attention_score_mean"]) for row in rows), 6),
        "n_examples": len(rows),
    }


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


def _recompute_query_attention_scores(*, model, full_cache, question_ids) -> tuple[torch.Tensor, float]:
    policy = QueryAwareSnapKV(
        model,
        obs_window_size=DEFAULT_OBS_WINDOW_SIZE,
        sink_size=DEFAULT_SINK_SIZE,
        recency_window=0,
        pooling=DEFAULT_POOLING,
    )
    device = model_device(model)
    _sync_if_cuda(device)
    start = time.perf_counter()
    _, attention_scores, _ = policy.prepare_eviction_inputs(full_cache, obs_window=question_ids)
    _sync_if_cuda(device)
    return attention_scores.detach().to("cpu", dtype=torch.float32), time.perf_counter() - start


def _select_repair_fragment(*, eviction_result, full_cache, attention_scores: torch.Tensor, target_restore_count: int):
    if target_restore_count <= 0 or len(eviction_result.evicted) == 0:
        return None, [], []

    position_to_dense = {int(position): dense_idx for dense_idx, position in enumerate(full_cache.positions)}
    candidates: list[tuple[float, float, int, int]] = []
    for evicted_dense_idx, position in enumerate(eviction_result.evicted.positions):
        dense_idx = position_to_dense[int(position)]
        attention_score = float(attention_scores[dense_idx].item())
        original_importance = float(eviction_result.importance_scores.get(int(position), 0.0))
        candidates.append((attention_score, original_importance, int(position), evicted_dense_idx))

    top_k = min(int(target_restore_count), len(candidates))
    ranked = sorted(candidates, key=lambda item: (-item[0], -item[1], item[2]))[:top_k]
    selected_evicted_dense_indices = [item[3] for item in ranked]
    selected_attention_scores = [item[0] for item in ranked]
    selected_fragment = slice_kv(eviction_result.evicted, selected_evicted_dense_indices)
    return selected_fragment, [item[2] for item in ranked], selected_attention_scores


def _run_slice(
    *,
    model,
    tokenizer,
    task_key: str,
    initial_method: str,
    budget: int,
    repair_ratio: float,
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

        target_total_budget = min(int(context_length), int(math.ceil(float(budget) * (1.0 + float(repair_ratio)))))
        target_restore_count = max(0, target_total_budget - int(len(eviction_result.compressed)))

        attention_scores = None
        attention_recompute_s = 0.0
        repair_selection_s = 0.0
        selected_cpu = None
        selected_positions: list[int] = []
        selected_attention_scores: list[float] = []
        if target_restore_count > 0:
            attention_scores, attention_recompute_s = _recompute_query_attention_scores(
                model=model,
                full_cache=full_cache,
                question_ids=prepared.question_ids,
            )

            selection_start = time.perf_counter()
            selected_cpu, selected_positions, selected_attention_scores = _select_repair_fragment(
                eviction_result=eviction_result,
                full_cache=full_cache,
                attention_scores=attention_scores,
                target_restore_count=target_restore_count,
            )
            repair_selection_s = time.perf_counter() - selection_start

        device = model_device(model)
        repair_transfer_ms = 0.0
        repair_inject_ms = 0.0
        selected_gpu = None
        if selected_cpu is not None:
            _sync_if_cuda(device)
            transfer_start = time.perf_counter()
            selected_gpu = selected_cpu.to_device(device, non_blocking=True)
            _sync_if_cuda(device)
            repair_transfer_ms = (time.perf_counter() - transfer_start) * 1000.0

            inject_start = time.perf_counter()
            repaired_cache = inject_kv(
                eviction_result.compressed,
                selected_gpu,
                selected_gpu.positions,
            )
            _sync_if_cuda(device)
            repair_inject_ms = (time.perf_counter() - inject_start) * 1000.0
        else:
            repaired_cache = eviction_result.compressed

        repair_total_ms = (attention_recompute_s + repair_selection_s) * 1000.0 + repair_transfer_ms + repair_inject_ms

        repaired_start = time.perf_counter()
        repaired_output = _generate_answer(model, tokenizer, prepared, repaired_cache)
        repaired_generation_s = time.perf_counter() - repaired_start
        repaired_score = _score_prediction(repaired_output, gold_outputs)

        example_wall_s = time.perf_counter() - example_start
        row = {
            "example_id": example_id,
            "task_key": task_key,
            "initial_method": initial_method,
            "repair_selector": REPAIR_SELECTOR,
            "repair_ratio": float(repair_ratio),
            "k_budget": int(budget),
            "target_total_budget": int(target_total_budget),
            "condition_b_score": round(condition_b_score, 6),
            "repaired_score": round(repaired_score, 6),
            "condition_b_output": condition_b_output,
            "repaired_output": repaired_output,
            "same_output_as_b": bool(condition_b_output == repaired_output),
            "initial_compressed_context_length": int(len(eviction_result.compressed)),
            "repaired_context_length": int(len(repaired_cache)),
            "restored_token_count": int(len(selected_positions)),
            "selected_attention_score_mean": round(_mean(selected_attention_scores), 6),
            "selected_attention_score_max": round(max(selected_attention_scores) if selected_attention_scores else 0.0, 6),
            "selected_attention_score_min": round(min(selected_attention_scores) if selected_attention_scores else 0.0, 6),
            "initial_eviction_s": round(initial_eviction_s, 6),
            "attention_recompute_s": round(attention_recompute_s, 6),
            "repair_selection_s": round(repair_selection_s, 6),
            "repair_transfer_ms": round(repair_transfer_ms, 6),
            "repair_inject_ms": round(repair_inject_ms, 6),
            "repair_total_ms": round(repair_total_ms, 6),
            "condition_b_generation_s": round(condition_b_generation_s, 6),
            "repaired_generation_s": round(repaired_generation_s, 6),
            "example_wall_s": round(example_wall_s, 6),
        }
        rows.append(row)

        print(
            f"[{task_key} {initial_method} k={int(budget)} {index + 1:03d}/{int(num_samples):03d}] "
            f"B={row['condition_b_score']:.3f} "
            f"R={row['repaired_score']:.3f} "
            f"lift={row['repaired_score'] - row['condition_b_score']:.3f} "
            f"added={row['restored_token_count']:d}/{target_restore_count:d} "
            f"repair_ms={row['repair_total_ms']:.3f}",
            flush=True,
        )

        del full_cache, eviction_result, repaired_cache
        if attention_scores is not None:
            del attention_scores
        if selected_cpu is not None:
            del selected_cpu
        if selected_gpu is not None:
            del selected_gpu
        torch.cuda.empty_cache()

    return rows


def _select_repair_fragment_cached(
    *,
    evicted_cache: PositionTrackedCache,
    attention_scores: torch.Tensor,
    target_restore_count: int,
    position_to_dense: dict[int, int],
    original_importance_scores: torch.Tensor | None,
):
    if target_restore_count <= 0 or len(evicted_cache) == 0:
        return None, [], []

    candidates: list[tuple[float, float, int, int]] = []
    for evicted_dense_idx, position in enumerate(evicted_cache.positions):
        dense_idx = position_to_dense[int(position)]
        attention_score = float(attention_scores[dense_idx].item())
        original_importance = 0.0
        if original_importance_scores is not None:
            original_importance = float(original_importance_scores[dense_idx].item())
        candidates.append((attention_score, original_importance, int(position), evicted_dense_idx))

    top_k = min(int(target_restore_count), len(candidates))
    ranked = sorted(candidates, key=lambda item: (-item[0], -item[1], item[2]))[:top_k]
    selected_evicted_dense_indices = [item[3] for item in ranked]
    selected_attention_scores = [item[0] for item in ranked]
    selected_fragment = slice_kv(evicted_cache, selected_evicted_dense_indices)
    if not isinstance(selected_fragment, PositionTrackedCache):
        raise RuntimeError("Phase 2 slice_kv did not preserve position tracking for the selected repair fragment.")
    return selected_fragment, [item[2] for item in ranked], selected_attention_scores


def _run_shared_slice(
    *,
    model,
    tokenizer,
    prepared,
    gold_outputs: list[str],
    full_cache: PositionTrackedCache,
    task_key: str,
    repair_ratio: float,
    example_id: str,
    example_index: int,
    num_samples: int,
    initial_method: str,
    budget: int,
    target_total_budget: int,
    common_share_s: float,
    attention_scores: torch.Tensor | None,
    attention_share_s: float,
    position_to_dense: dict[int, int] | None,
    snap_importance_gpu: torch.Tensor | None,
    snap_importance_cpu: torch.Tensor | None,
    snap_prepare_share_s: float,
) -> dict[str, Any]:
    if initial_method == "snapkv":
        if snap_importance_gpu is None:
            raise RuntimeError("Shared SnapKV execution requires cached importance scores.")
        compressed, evicted_cpu, materialize_s = _materialize_snapkv_eviction_from_importance(
            full_cache=full_cache,
            budget=budget,
            importance=snap_importance_gpu,
        )
        initial_eviction_s = snap_prepare_share_s + materialize_s
        original_importance_scores = snap_importance_cpu
    elif initial_method == "streaming_llm":
        compressed, evicted_cpu, materialize_s = _materialize_streaming_eviction(
            full_cache=full_cache,
            budget=budget,
        )
        initial_eviction_s = materialize_s
        original_importance_scores = None
    else:
        raise ValueError(f"Unsupported initial eviction method: {initial_method}")

    b_start = time.perf_counter()
    condition_b_output = _generate_answer(model, tokenizer, prepared, compressed)
    condition_b_generation_s = time.perf_counter() - b_start
    condition_b_score = _score_prediction(condition_b_output, gold_outputs)

    target_restore_count = max(0, int(target_total_budget) - int(len(compressed)))
    attention_recompute_s = float(attention_share_s) if target_restore_count > 0 else 0.0

    repair_selection_s = 0.0
    selected_cpu = None
    selected_positions: list[int] = []
    selected_attention_scores: list[float] = []
    if target_restore_count > 0:
        if attention_scores is None or position_to_dense is None:
            raise RuntimeError("Repair-required rows need cached query attention scores and dense position lookup.")
        selection_start = time.perf_counter()
        selected_cpu, selected_positions, selected_attention_scores = _select_repair_fragment_cached(
            evicted_cache=evicted_cpu,
            attention_scores=attention_scores,
            target_restore_count=target_restore_count,
            position_to_dense=position_to_dense,
            original_importance_scores=original_importance_scores,
        )
        repair_selection_s = time.perf_counter() - selection_start

    device = model_device(model)
    repair_transfer_ms = 0.0
    repair_inject_ms = 0.0
    selected_gpu = None
    if selected_cpu is not None:
        _sync_if_cuda(device)
        transfer_start = time.perf_counter()
        selected_gpu = selected_cpu.to_device(device, non_blocking=True)
        _sync_if_cuda(device)
        repair_transfer_ms = (time.perf_counter() - transfer_start) * 1000.0

        inject_start = time.perf_counter()
        repaired_cache = inject_kv(
            compressed,
            selected_gpu,
            selected_gpu.positions,
        )
        _sync_if_cuda(device)
        repair_inject_ms = (time.perf_counter() - inject_start) * 1000.0
    else:
        repaired_cache = compressed

    repair_total_ms = (attention_recompute_s + repair_selection_s) * 1000.0 + repair_transfer_ms + repair_inject_ms

    repaired_start = time.perf_counter()
    repaired_output = _generate_answer(model, tokenizer, prepared, repaired_cache)
    repaired_generation_s = time.perf_counter() - repaired_start
    repaired_score = _score_prediction(repaired_output, gold_outputs)

    example_wall_s = (
        float(common_share_s)
        + float(initial_eviction_s)
        + float(condition_b_generation_s)
        + float(attention_recompute_s)
        + float(repair_selection_s)
        + (float(repair_transfer_ms) / 1000.0)
        + (float(repair_inject_ms) / 1000.0)
        + float(repaired_generation_s)
    )

    row = {
        "example_id": example_id,
        "task_key": task_key,
        "initial_method": initial_method,
        "repair_selector": REPAIR_SELECTOR,
        "repair_ratio": float(repair_ratio),
        "k_budget": int(budget),
        "target_total_budget": int(target_total_budget),
        "condition_b_score": round(condition_b_score, 6),
        "repaired_score": round(repaired_score, 6),
        "condition_b_output": condition_b_output,
        "repaired_output": repaired_output,
        "same_output_as_b": bool(condition_b_output == repaired_output),
        "initial_compressed_context_length": int(len(compressed)),
        "repaired_context_length": int(len(repaired_cache)),
        "restored_token_count": int(len(selected_positions)),
        "selected_attention_score_mean": round(_mean(selected_attention_scores), 6),
        "selected_attention_score_max": round(max(selected_attention_scores) if selected_attention_scores else 0.0, 6),
        "selected_attention_score_min": round(min(selected_attention_scores) if selected_attention_scores else 0.0, 6),
        "initial_eviction_s": round(initial_eviction_s, 6),
        "attention_recompute_s": round(attention_recompute_s, 6),
        "repair_selection_s": round(repair_selection_s, 6),
        "repair_transfer_ms": round(repair_transfer_ms, 6),
        "repair_inject_ms": round(repair_inject_ms, 6),
        "repair_total_ms": round(repair_total_ms, 6),
        "condition_b_generation_s": round(condition_b_generation_s, 6),
        "repaired_generation_s": round(repaired_generation_s, 6),
        "example_wall_s": round(example_wall_s, 6),
    }

    print(
        f"[{task_key} {initial_method} k={int(budget)} {int(example_index) + 1:03d}/{int(num_samples):03d}] "
        f"B={row['condition_b_score']:.3f} "
        f"R={row['repaired_score']:.3f} "
        f"lift={row['repaired_score'] - row['condition_b_score']:.3f} "
        f"added={row['restored_token_count']:d}/{target_restore_count:d} "
        f"repair_ms={row['repair_total_ms']:.3f}",
        flush=True,
    )

    del compressed, evicted_cpu, repaired_cache
    if selected_cpu is not None:
        del selected_cpu
    if selected_gpu is not None:
        del selected_gpu
    return row


def _run_task_shared(
    *,
    model,
    tokenizer,
    task_key: str,
    slice_plans: list[_SlicePlan],
    repair_ratio: float,
    num_samples: int,
    context_length: int,
    dataset_seed_offset: int,
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    rows_by_slice = {(plan.initial_method, plan.budget): [] for plan in slice_plans}
    if not slice_plans:
        return rows_by_slice

    pending_slice_count = len(slice_plans)
    snap_slice_count = sum(1 for plan in slice_plans if plan.initial_method == "snapkv")
    repair_slice_count = sum(1 for plan in slice_plans if plan.repair_needed)

    for index in range(int(num_samples)):
        data_start = time.perf_counter()
        example = build_task_example(
            task_key,
            index,
            context_length,
            tokenizer,
            dataset_seed_offset=dataset_seed_offset,
        )
        data_generation_s = time.perf_counter() - data_start

        prepare_start = time.perf_counter()
        prepared = prepare_example_for_model(example, tokenizer)
        prepare_example_s = time.perf_counter() - prepare_start

        prefill_start = time.perf_counter()
        full_cache = build_position_tracked_cache(model, prepared.context_ids)
        prefill_full_cache_s = time.perf_counter() - prefill_start

        gold_outputs = list(prepared.example.outputs)
        example_id = f"ex{index + 1:03d}"
        common_share_s = (data_generation_s + prepare_example_s + prefill_full_cache_s) / float(pending_slice_count)

        attention_scores = None
        attention_share_s = 0.0
        position_to_dense = None
        if repair_slice_count > 0:
            attention_scores, attention_recompute_s = _recompute_query_attention_scores(
                model=model,
                full_cache=full_cache,
                question_ids=prepared.question_ids,
            )
            attention_share_s = attention_recompute_s / float(repair_slice_count)
            position_to_dense = {int(position): dense_idx for dense_idx, position in enumerate(full_cache.positions)}

        snap_importance_gpu = None
        snap_importance_cpu = None
        snap_prepare_share_s = 0.0
        if snap_slice_count > 0:
            snap_importance_gpu, snap_importance_cpu, snap_prepare_s = _prepare_snapkv_importance(full_cache)
            snap_prepare_share_s = snap_prepare_s / float(snap_slice_count)

        for plan in slice_plans:
            row = _run_shared_slice(
                model=model,
                tokenizer=tokenizer,
                prepared=prepared,
                gold_outputs=gold_outputs,
                full_cache=full_cache,
                task_key=task_key,
                repair_ratio=repair_ratio,
                example_id=example_id,
                example_index=index,
                num_samples=int(num_samples),
                initial_method=plan.initial_method,
                budget=plan.budget,
                target_total_budget=plan.target_total_budget,
                common_share_s=common_share_s,
                attention_scores=attention_scores,
                attention_share_s=attention_share_s if plan.repair_needed else 0.0,
                position_to_dense=position_to_dense,
                snap_importance_gpu=snap_importance_gpu,
                snap_importance_cpu=snap_importance_cpu,
                snap_prepare_share_s=snap_prepare_share_s if plan.initial_method == "snapkv" else 0.0,
            )
            rows_by_slice[(plan.initial_method, plan.budget)].append(row)

        del full_cache
        if attention_scores is not None:
            del attention_scores
        if snap_importance_gpu is not None:
            del snap_importance_gpu
        if snap_importance_cpu is not None:
            del snap_importance_cpu
        torch.cuda.empty_cache()

    return rows_by_slice


def _collect_slice_plans(
    *,
    tasks: list[str],
    methods: list[str],
    budgets: list[int],
    num_samples: int,
    context_length: int,
    dataset_seed_offset: int,
    repair_ratio: float,
    reuse_matching_artifacts: bool,
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, list[_SlicePlan]]]:
    existing_payloads: dict[tuple[str, str, int], dict[str, Any]] = {}
    pending_by_task = {task_key: [] for task_key in tasks}

    for task_key in tasks:
        for method in methods:
            for budget in budgets:
                artifact_path = _artifact_path(task_key, method, budget)
                payload = None
                if reuse_matching_artifacts:
                    payload = _load_json(artifact_path)
                    if not _slice_payload_matches(
                        payload,
                        task_key=task_key,
                        initial_method=method,
                        budget=budget,
                        num_samples=int(num_samples),
                        context_length=int(context_length),
                        dataset_seed_offset=int(dataset_seed_offset),
                        repair_ratio=float(repair_ratio),
                    ):
                        payload = None

                if payload is not None:
                    existing_payloads[(task_key, method, budget)] = payload
                    continue

                target_total_budget = _target_total_budget(
                    budget=budget,
                    context_length=context_length,
                    repair_ratio=repair_ratio,
                )
                pending_by_task[task_key].append(
                    _SlicePlan(
                        initial_method=method,
                        budget=int(budget),
                        target_total_budget=int(target_total_budget),
                        repair_needed=int(target_total_budget) > min(int(budget), int(context_length)),
                        artifact_path=artifact_path,
                    )
                )

    return existing_payloads, pending_by_task


def _write_progress(
    *,
    tasks: list[str],
    methods: list[str],
    budgets: list[int],
    num_samples: int,
    context_length: int,
    dataset_seed_offset: int,
    repair_ratio: float,
    completed_slices: int,
    expected_slices: int,
    last_completed: dict[str, Any] | None,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tasks": tasks,
        "initial_methods": methods,
        "repair_selector": REPAIR_SELECTOR,
        "repair_ratio": float(repair_ratio),
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
    parser.add_argument("--repair-ratio", type=float, default=DEFAULT_REPAIR_RATIO, help="Extra token ratio to add back on top of K.")
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
    existing_payloads, pending_by_task = _collect_slice_plans(
        tasks=normalized_tasks,
        methods=normalized_methods,
        budgets=normalized_budgets,
        num_samples=int(args.num_samples),
        context_length=int(args.context_length),
        dataset_seed_offset=int(args.dataset_seed_offset),
        repair_ratio=float(args.repair_ratio),
        reuse_matching_artifacts=bool(args.reuse_matching_artifacts),
    )
    total_pending_slices = sum(len(plans) for plans in pending_by_task.values())

    model = None
    tokenizer = None
    if total_pending_slices > 0:
        model = load_model()
        tokenizer = load_tokenizer()

    summary_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tasks": {},
        "task_keys": normalized_tasks,
        "initial_methods": normalized_methods,
        "repair_selector": REPAIR_SELECTOR,
        "repair_ratio": float(args.repair_ratio),
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
        pending_plans = pending_by_task.get(task_key, [])
        generated_payloads: dict[tuple[str, int], dict[str, Any]] = {}
        if pending_plans:
            if model is None or tokenizer is None:
                raise RuntimeError("Model and tokenizer must be loaded before executing pending slices.")
            print(
                f"Starting task={task_key} pending_slices={len(pending_plans)} "
                f"examples={int(args.num_samples)} shared_execution=1",
                flush=True,
            )
            rows_by_slice = _run_task_shared(
                model=model,
                tokenizer=tokenizer,
                task_key=task_key,
                slice_plans=pending_plans,
                repair_ratio=float(args.repair_ratio),
                num_samples=int(args.num_samples),
                context_length=int(args.context_length),
                dataset_seed_offset=int(args.dataset_seed_offset),
            )
            for plan in pending_plans:
                rows = rows_by_slice[(plan.initial_method, plan.budget)]
                payload = {
                    "schema_version": SLICE_SCHEMA_VERSION,
                    "task_key": task_key,
                    "display_name": get_task_spec(task_key).display_name,
                    "initial_method": plan.initial_method,
                    "repair_selector": REPAIR_SELECTOR,
                    "repair_ratio": float(args.repair_ratio),
                    "k_budget": int(plan.budget),
                    "num_samples": int(args.num_samples),
                    "context_length": int(args.context_length),
                    "dataset_seed_offset": int(args.dataset_seed_offset),
                    "aggregate": _summarize_rows(rows),
                    "per_example": rows,
                }
                write_json(plan.artifact_path, payload)
                generated_payloads[(plan.initial_method, plan.budget)] = payload

        for method in normalized_methods:
            method_payload = task_payload.setdefault(method, {})
            for budget in normalized_budgets:
                artifact_path = _artifact_path(task_key, method, budget)
                payload = existing_payloads.get((task_key, method, budget))
                if payload is None:
                    payload = generated_payloads.get((method, budget))
                if payload is None:
                    raise RuntimeError(f"Missing payload for task={task_key} method={method} budget={budget}.")

                method_payload[_budget_label(budget)] = {
                    "aggregate": payload["aggregate"],
                    "artifact_path": str(artifact_path),
                }
                completed_slices += 1
                last_completed = {
                    "task_key": task_key,
                    "initial_method": method,
                    "k_budget": int(budget),
                    "repair_selector": REPAIR_SELECTOR,
                    "repair_ratio": float(args.repair_ratio),
                    "mean_condition_b": payload["aggregate"]["mean_condition_b"],
                    "mean_repaired": payload["aggregate"]["mean_repaired"],
                    "mean_lift_over_b": payload["aggregate"]["mean_lift_over_b"],
                    "mean_repaired_context_length": payload["aggregate"]["mean_repaired_context_length"],
                    "artifact_path": str(artifact_path),
                }
                _write_progress(
                    tasks=normalized_tasks,
                    methods=normalized_methods,
                    budgets=normalized_budgets,
                    num_samples=int(args.num_samples),
                    context_length=int(args.context_length),
                    dataset_seed_offset=int(args.dataset_seed_offset),
                    repair_ratio=float(args.repair_ratio),
                    completed_slices=completed_slices,
                    expected_slices=expected_slices,
                    last_completed=last_completed,
                )

    write_json(SUMMARY_PATH, summary_payload)
    print(f"Completed {completed_slices}/{expected_slices} slices. Summary written to {SUMMARY_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
