#!/usr/bin/env python3
"""Profile one budgeted-repair example with detailed timing breakdowns."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

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
from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache, inject_kv, slice_kv, to_tuple_cache  # noqa: E402
from phases.phase2_kv_cache.src.runtime import generate_from_cache, load_model, load_tokenizer, model_device  # noqa: E402
from phases.phase3_eviction.src.benchmark import (  # noqa: E402
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_OBS_WINDOW_SIZE,
    DEFAULT_POOLING,
    DEFAULT_SINK_SIZE,
)
from phases.phase3_eviction.src.eviction import QueryAwareSnapKV, SnapKV, StreamingLLM  # noqa: E402
from phases.phase3_eviction.src.eviction.base import _cache_to_cpu_pinned, make_placeholder_obs_q_vecs  # noqa: E402
from phases.phase3_eviction.src.runtime import build_position_tracked_cache, write_json  # noqa: E402

RESULTS_DIR = PHASE_ROOT / "results" / "phase5_budgeted_repair" / "profiling"
DEFAULT_TASK = "vt_8hop_permute_div2"
DEFAULT_METHOD = "snapkv"
DEFAULT_BUDGET = 16384
DEFAULT_REPAIR_RATIO = 0.10


def _sync_if_cuda(device: torch.device | str) -> None:
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.synchronize(target)


def _time_gpu(label: str, fn, *, device: torch.device) -> tuple[Any, float]:
    _sync_if_cuda(device)
    start = time.perf_counter()
    value = fn()
    _sync_if_cuda(device)
    elapsed = time.perf_counter() - start
    return value, elapsed


def _time_cpu(fn) -> tuple[Any, float]:
    start = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - start
    return value, elapsed


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


def _snapkv_recency_window(k_budget: int) -> int:
    return max(0, min(1024, int(k_budget) - DEFAULT_SINK_SIZE))


def _streaming_recency_window(k_budget: int) -> int:
    return max(0, int(k_budget) - DEFAULT_SINK_SIZE)


def _serialize_timings(timings: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 6) for key, value in timings.items()}


def _finalize_result_profile(
    *,
    policy,
    full_cache: PositionTrackedCache,
    keep_indices: list[int],
    importance: torch.Tensor,
    obs_window_q_vecs: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, float]]:
    timings: dict[str, float] = {}
    seq_len = len(full_cache)
    keep_indices = sorted(dict.fromkeys(int(index) for index in keep_indices))
    if any(index < 0 or index >= seq_len for index in keep_indices):
        raise IndexError(f"keep_indices must lie in [0, {seq_len}), got {keep_indices}.")

    device = full_cache.device
    importance, timings["finalize.normalize_importance_s"] = _time_gpu(
        "normalize_importance",
        lambda: policy._normalize_importance(importance, seq_len=seq_len, device=device),
        device=device,
    )
    obs_window_q_vecs, timings["finalize.normalize_obs_q_vecs_s"] = _time_cpu(
        lambda: policy._normalize_obs_window_q_vecs(full_cache, obs_window_q_vecs)
    )

    keep_set, timings["finalize.keep_set_s"] = _time_cpu(lambda: set(keep_indices))
    evict_indices, timings["finalize.build_evict_indices_s"] = _time_cpu(
        lambda: [index for index in range(seq_len) if index not in keep_set]
    )

    compressed, timings["finalize.slice_compressed_s"] = _time_gpu(
        "slice_compressed",
        lambda: slice_kv(full_cache, keep_indices),
        device=device,
    )
    evicted, timings["finalize.slice_evicted_s"] = _time_gpu(
        "slice_evicted",
        lambda: slice_kv(full_cache, evict_indices),
        device=device,
    )
    if not isinstance(compressed, PositionTrackedCache) or not isinstance(evicted, PositionTrackedCache):
        raise RuntimeError("Expected position-tracked caches from slice_kv.")

    evicted_cpu_cache, timings["finalize.copy_evicted_to_cpu_pinned_s"] = _time_gpu(
        "copy_evicted_to_cpu_pinned",
        lambda: PositionTrackedCache(_cache_to_cpu_pinned(evicted.kv), list(evicted.positions)),
        device=device,
    )
    importance_scores, timings["finalize.importance_scores_dict_s"] = _time_cpu(
        lambda: {
            int(position): float(importance[dense_index].item())
            for dense_index, position in enumerate(full_cache.positions)
        }
    )

    result = {
        "compressed": compressed,
        "evicted": evicted_cpu_cache,
        "importance_scores": importance_scores,
        "obs_window_q_vecs": obs_window_q_vecs,
    }
    return result, timings


def _profile_snapkv_initial_eviction(full_cache: PositionTrackedCache, budget: int) -> tuple[dict[str, Any], dict[str, float]]:
    timings: dict[str, float] = {}
    policy = SnapKV(
        obs_window_size=DEFAULT_OBS_WINDOW_SIZE,
        sink_size=DEFAULT_SINK_SIZE,
        recency_window=_snapkv_recency_window(budget),
        pooling=DEFAULT_POOLING,
    )
    device = full_cache.device
    tracked_cache = policy._require_tracked_cache(full_cache)
    importance, timings["initial.score_tokens_s"] = _time_gpu(
        "score_tokens",
        lambda: policy._score_tokens(tracked_cache),
        device=device,
    )
    obs_q_vecs, timings["initial.extract_obs_q_vecs_s"] = _time_cpu(
        lambda: policy._extract_obs_q_vecs(tracked_cache)
    )
    seq_len = len(full_cache)
    budget_norm = policy._normalize_budget(seq_len, budget)
    mandatory_indices, timings["initial.structural_keep_indices_s"] = _time_cpu(
        lambda: policy._structural_keep_indices(seq_len=seq_len, k_budget=budget_norm)
    )
    mandatory_set = set(mandatory_indices)
    remaining_slots = max(0, budget_norm - len(mandatory_indices))
    candidate_indices, timings["initial.build_candidate_indices_s"] = _time_cpu(
        lambda: [index for index in range(seq_len) if index not in mandatory_set]
    )

    selected_indices: list[int] = []
    timings["initial.topk_select_s"] = 0.0
    if remaining_slots > 0 and candidate_indices:
        def _run_topk() -> list[int]:
            candidate_tensor = torch.tensor(candidate_indices, device=importance.device, dtype=torch.long)
            candidate_scores = torch.index_select(importance, 0, candidate_tensor)
            topk = min(remaining_slots, len(candidate_indices))
            topk_indices = torch.topk(candidate_scores, k=topk, largest=True, sorted=False).indices
            return torch.index_select(candidate_tensor, 0, topk_indices).tolist()

        selected_indices, timings["initial.topk_select_s"] = _time_gpu("snapkv_topk", _run_topk, device=device)

    keep_indices, timings["initial.sort_keep_indices_s"] = _time_cpu(
        lambda: sorted(mandatory_set | set(selected_indices))
    )
    result, finalize_timings = _finalize_result_profile(
        policy=policy,
        full_cache=full_cache,
        keep_indices=keep_indices,
        importance=importance,
        obs_window_q_vecs=obs_q_vecs,
    )
    timings.update(finalize_timings)
    return result, timings


def _profile_streaming_initial_eviction(full_cache: PositionTrackedCache, budget: int) -> tuple[dict[str, Any], dict[str, float]]:
    timings: dict[str, float] = {}
    policy = StreamingLLM(
        sink_size=DEFAULT_SINK_SIZE,
        recency_window=_streaming_recency_window(budget),
    )
    device = full_cache.device
    tracked_cache = policy._require_tracked_cache(full_cache)
    keep_indices, timings["initial.structural_keep_indices_s"] = _time_cpu(
        lambda: policy._structural_keep_indices(seq_len=len(tracked_cache), k_budget=budget)
    )

    def _make_importance() -> torch.Tensor:
        importance = torch.zeros(len(tracked_cache), device=device, dtype=torch.float32)
        if keep_indices:
            importance[torch.tensor(keep_indices, device=device, dtype=torch.long)] = 1.0
        return importance

    importance, timings["initial.build_importance_s"] = _time_gpu(
        "streaming_importance",
        _make_importance,
        device=device,
    )
    obs_q_vecs, timings["initial.make_placeholder_obs_q_vecs_s"] = _time_cpu(
        lambda: make_placeholder_obs_q_vecs(tracked_cache)
    )
    result, finalize_timings = _finalize_result_profile(
        policy=policy,
        full_cache=tracked_cache,
        keep_indices=keep_indices,
        importance=importance,
        obs_window_q_vecs=obs_q_vecs,
    )
    timings.update(finalize_timings)
    return result, timings


def _recompute_query_attention_scores_profile(*, model, full_cache, question_ids) -> tuple[torch.Tensor, dict[str, float]]:
    timings: dict[str, float] = {}
    policy = QueryAwareSnapKV(
        model,
        obs_window_size=DEFAULT_OBS_WINDOW_SIZE,
        sink_size=DEFAULT_SINK_SIZE,
        recency_window=0,
        pooling=DEFAULT_POOLING,
    )
    device = model_device(model)

    tracked_cache, timings["repair.queryaware.require_cache_s"] = _time_cpu(
        lambda: policy._require_tracked_cache(full_cache)
    )
    query_ids, timings["repair.queryaware.prepare_query_ids_s"] = _time_cpu(
        lambda: policy._prepare_query_tokens(question_ids)
    )
    (live_cache, outputs), timings["repair.queryaware.forward_query_s"] = _time_gpu(
        "queryaware_forward",
        lambda: policy._forward_query(tracked_cache, query_ids),
        device=device,
    )
    extended_kv, timings["repair.queryaware.to_tuple_cache_s"] = _time_cpu(
        lambda: to_tuple_cache(outputs.past_key_values)
    )

    context_len = len(live_cache)

    def _compute_layer_scores() -> tuple[torch.Tensor, torch.Tensor]:
        layer_scores: list[torch.Tensor] = []
        obs_vecs: list[torch.Tensor] = []
        for key, _ in extended_kv:
            key_float = key.detach().to(dtype=torch.float32)
            query_rows = key_float[:, :, context_len:, :]
            context_rows = key_float[:, :, :context_len, :]
            scores = torch.matmul(query_rows, context_rows.transpose(-2, -1)) / math.sqrt(key_float.shape[-1])
            scores = torch.softmax(scores, dim=-1)
            layer_scores.append(policy._pool_scores(scores))
            obs_vecs.append(query_rows[0].mean(dim=0).cpu())
        return torch.stack(layer_scores, dim=0).mean(dim=0), torch.stack(obs_vecs, dim=0)

    (importance, _obs_q_vecs), timings["repair.queryaware.compute_scores_s"] = _time_gpu(
        "queryaware_compute_scores",
        _compute_layer_scores,
        device=device,
    )
    attention_scores, timings["repair.queryaware.detach_attention_to_cpu_s"] = _time_gpu(
        "queryaware_detach_attention",
        lambda: importance.detach().to("cpu", dtype=torch.float32),
        device=device,
    )
    return attention_scores, timings


def _select_repair_fragment_profile(*, eviction_result, full_cache, attention_scores: torch.Tensor, target_restore_count: int):
    timings: dict[str, float] = {}
    if target_restore_count <= 0 or len(eviction_result["evicted"]) == 0:
        return None, [], [], timings

    position_to_dense, timings["repair.selection.position_to_dense_s"] = _time_cpu(
        lambda: {int(position): dense_idx for dense_idx, position in enumerate(full_cache.positions)}
    )

    def _build_candidates():
        candidates: list[tuple[float, float, int, int]] = []
        for evicted_dense_idx, position in enumerate(eviction_result["evicted"].positions):
            dense_idx = position_to_dense[int(position)]
            attention_score = float(attention_scores[dense_idx].item())
            original_importance = float(eviction_result["importance_scores"].get(int(position), 0.0))
            candidates.append((attention_score, original_importance, int(position), evicted_dense_idx))
        return candidates

    candidates, timings["repair.selection.build_candidates_s"] = _time_cpu(_build_candidates)
    top_k = min(int(target_restore_count), len(candidates))
    ranked, timings["repair.selection.sort_ranked_s"] = _time_cpu(
        lambda: sorted(candidates, key=lambda item: (-item[0], -item[1], item[2]))[:top_k]
    )
    selected_evicted_dense_indices = [item[3] for item in ranked]
    selected_attention_scores = [item[0] for item in ranked]
    selected_fragment, timings["repair.selection.slice_selected_fragment_s"] = _time_cpu(
        lambda: slice_kv(eviction_result["evicted"], selected_evicted_dense_indices)
    )
    return selected_fragment, [item[2] for item in ranked], selected_attention_scores, timings


def profile_one_example(
    *,
    model,
    tokenizer,
    task_key: str,
    initial_method: str,
    budget: int,
    repair_ratio: float,
    context_length: int,
    dataset_seed_offset: int,
    index: int,
) -> dict[str, Any]:
    device = model_device(model)
    timings: dict[str, float] = {}

    example_start = time.perf_counter()
    example, timings["data_generation_s"] = _time_cpu(
        lambda: build_task_example(
            task_key,
            index,
            context_length,
            tokenizer,
            dataset_seed_offset=dataset_seed_offset,
        )
    )
    prepared, timings["prepare_example_s"] = _time_cpu(lambda: prepare_example_for_model(example, tokenizer))
    full_cache, timings["prefill_full_cache_s"] = _time_gpu(
        "prefill_full_cache",
        lambda: build_position_tracked_cache(model, prepared.context_ids),
        device=device,
    )
    gold_outputs = list(prepared.example.outputs)

    if initial_method == "snapkv":
        eviction_result, initial_timings = _profile_snapkv_initial_eviction(full_cache, budget)
    elif initial_method == "streaming_llm":
        eviction_result, initial_timings = _profile_streaming_initial_eviction(full_cache, budget)
    else:
        raise ValueError(f"Unsupported method: {initial_method}")
    timings.update(initial_timings)
    timings["initial.total_s"] = sum(value for key, value in initial_timings.items() if key.startswith("initial.") or key.startswith("finalize."))

    condition_b_output, timings["condition_b_generation_s"] = _time_gpu(
        "condition_b_generation",
        lambda: _generate_answer(model, tokenizer, prepared, eviction_result["compressed"]),
        device=device,
    )
    condition_b_score = _score_prediction(condition_b_output, gold_outputs)

    target_total_budget = min(int(context_length), int(math.ceil(float(budget) * (1.0 + float(repair_ratio)))))
    target_restore_count = max(0, target_total_budget - int(len(eviction_result["compressed"])))

    attention_scores = None
    selected_cpu = None
    selected_gpu = None
    selected_positions: list[int] = []
    selected_attention_scores: list[float] = []

    if target_restore_count > 0:
        attention_scores, repair_query_timings = _recompute_query_attention_scores_profile(
            model=model,
            full_cache=full_cache,
            question_ids=prepared.question_ids,
        )
        timings.update(repair_query_timings)
        selected_cpu, selected_positions, selected_attention_scores, selection_timings = _select_repair_fragment_profile(
            eviction_result=eviction_result,
            full_cache=full_cache,
            attention_scores=attention_scores,
            target_restore_count=target_restore_count,
        )
        timings.update(selection_timings)
    else:
        selected_cpu = None

    if selected_cpu is not None:
        selected_gpu, timings["repair.transfer_selected_to_gpu_s"] = _time_gpu(
            "repair_transfer_selected_to_gpu",
            lambda: selected_cpu.to_device(device, non_blocking=True),
            device=device,
        )
        repaired_cache, timings["repair.inject_kv_s"] = _time_gpu(
            "repair_inject_kv",
            lambda: inject_kv(
                eviction_result["compressed"],
                selected_gpu,
                selected_gpu.positions,
            ),
            device=device,
        )
    else:
        repaired_cache = eviction_result["compressed"]
        timings["repair.transfer_selected_to_gpu_s"] = 0.0
        timings["repair.inject_kv_s"] = 0.0

    repaired_output, timings["repaired_generation_s"] = _time_gpu(
        "repaired_generation",
        lambda: _generate_answer(model, tokenizer, prepared, repaired_cache),
        device=device,
    )
    repaired_score = _score_prediction(repaired_output, gold_outputs)

    _, timings["cleanup_empty_cache_s"] = _time_gpu(
        "cleanup_empty_cache",
        lambda: torch.cuda.empty_cache() if torch.cuda.is_available() else None,
        device=device,
    )
    total_wall_s = time.perf_counter() - example_start
    timings["total_example_wall_s"] = total_wall_s

    repair_component_keys = [
        "repair.queryaware.require_cache_s",
        "repair.queryaware.prepare_query_ids_s",
        "repair.queryaware.forward_query_s",
        "repair.queryaware.to_tuple_cache_s",
        "repair.queryaware.compute_scores_s",
        "repair.queryaware.detach_attention_to_cpu_s",
        "repair.selection.position_to_dense_s",
        "repair.selection.build_candidates_s",
        "repair.selection.sort_ranked_s",
        "repair.selection.slice_selected_fragment_s",
        "repair.transfer_selected_to_gpu_s",
        "repair.inject_kv_s",
    ]
    timings["repair.total_s"] = sum(timings.get(key, 0.0) for key in repair_component_keys)

    payload = {
        "task_key": task_key,
        "task_display_name": get_task_spec(task_key).display_name,
        "example_index": int(index),
        "initial_method": initial_method,
        "repair_ratio": float(repair_ratio),
        "k_budget": int(budget),
        "context_length": int(context_length),
        "condition_b_score": round(condition_b_score, 6),
        "repaired_score": round(repaired_score, 6),
        "condition_b_output": condition_b_output,
        "repaired_output": repaired_output,
        "initial_compressed_context_length": int(len(eviction_result["compressed"])),
        "repaired_context_length": int(len(repaired_cache)),
        "target_total_budget": int(target_total_budget),
        "target_restore_count": int(target_restore_count),
        "restored_token_count": int(len(selected_positions)),
        "selected_attention_score_mean": round(
            float(sum(selected_attention_scores) / len(selected_attention_scores)) if selected_attention_scores else 0.0,
            6,
        ),
        "timings_s": _serialize_timings(timings),
        "largest_components_s": sorted(
            ((key, float(value)) for key, value in timings.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:15],
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK, choices=sorted(TASK_SPECS.keys()))
    parser.add_argument("--method", default=DEFAULT_METHOD, choices=("snapkv", "streaming_llm"))
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--repair-ratio", type=float, default=DEFAULT_REPAIR_RATIO)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--dataset-seed-offset", type=int, default=0)
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    startup_start = time.perf_counter()
    tokenizer = load_tokenizer()
    tokenizer_load_s = time.perf_counter() - startup_start

    model_start = time.perf_counter()
    model = load_model()
    model_load_s = time.perf_counter() - model_start

    payload = profile_one_example(
        model=model,
        tokenizer=tokenizer,
        task_key=args.task,
        initial_method=args.method,
        budget=args.budget,
        repair_ratio=float(args.repair_ratio),
        context_length=int(args.context_length),
        dataset_seed_offset=int(args.dataset_seed_offset),
        index=int(args.example_index),
    )
    payload["startup_s"] = {
        "tokenizer_load_s": round(tokenizer_load_s, 6),
        "model_load_s": round(model_load_s, 6),
    }

    if args.output is not None:
        output_path = args.output
    else:
        output_path = RESULTS_DIR / (
            f"{task_prefix(get_task_spec(args.task).display_name)}_{args.method}_"
            f"k{int(args.budget)}_ex{int(args.example_index):03d}_profile.json"
        )
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2))
    print(f"PROFILE_PATH={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
