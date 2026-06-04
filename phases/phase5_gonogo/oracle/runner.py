"""Phase 5 oracle runner."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from phases.phase1_degradation.phase1.evaluation import sample_score, task_prefix
from phases.phase1_degradation.phase1.inference import prepare_example_for_model
from phases.phase1_degradation.phase1.task_registry import build_task_example, get_task_spec
from phases.phase2_kv_cache.src.kv_utils import inject_kv, load_kv, save_kv, to_dynamic_cache
from phases.phase2_kv_cache.src.runtime import generate_from_cache, load_model, load_tokenizer, model_device
from phases.phase3_eviction.src.benchmark import DEFAULT_CONTEXT_LENGTH, DEFAULT_OBS_WINDOW_SIZE, DEFAULT_POOLING, DEFAULT_SINK_SIZE
from phases.phase3_eviction.src.eviction import SnapKV, StreamingLLM, log_eviction
from phases.phase3_eviction.src.runtime import build_position_tracked_cache, write_json

from .recovery import compute_oracle_recovery, format_go_nogo, plot_oracle_vs_budget, plot_recovery_distribution

PHASE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PHASE_ROOT / "results" / "phase5_oracle"
FIGURES_DIR = RESULTS_DIR / "figures"
DIAGNOSTICS_DIR = RESULTS_DIR / "diagnostics"
LOG_DIR = RESULTS_DIR / "logs"

DEFAULT_TASKS = ("vt_8hop_permute_div2", "mq_niah_4q", "s_niah")
DEFAULT_METHODS = ("snapkv", "streaming_llm")
DEFAULT_BUDGETS = (256, 512, 1024)
DEFAULT_PRIMARY_TASK = "vt_8hop_permute_div2"
DEFAULT_NUM_SAMPLES = 100
DEFAULT_MIN_GAP = 0.05
DEFAULT_SERIALIZATION_SAMPLES = 10
DEFAULT_SERIALIZATION_THRESHOLD = 1e-2
DEFAULT_SNAPKV_RECENCY_CAP = 1024


def ensure_results_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
            raise ValueError(f"Unsupported Phase 5 method: {method}")
        if normalized not in ordered:
            ordered.append(normalized)
    return ordered


def _normalize_budgets(budgets: Iterable[int | str], *, context_length: int) -> list[int]:
    ordered: list[int] = []
    for budget in budgets:
        normalized = int(budget)
        if normalized <= 0:
            raise ValueError(f"Budgets must be positive, got {budget!r}.")
        normalized = min(normalized, int(context_length))
        if normalized not in ordered:
            ordered.append(normalized)
    return ordered


def _budget_label(budget: int) -> str:
    return f"k{int(budget)}"


def _snapkv_recency_window(k_budget: int) -> int:
    return max(0, min(DEFAULT_SNAPKV_RECENCY_CAP, int(k_budget) - DEFAULT_SINK_SIZE))


def _streaming_recency_window(k_budget: int) -> int:
    return max(0, int(k_budget) - DEFAULT_SINK_SIZE)


def _artifact_path(task_key: str, method: str, budget: int) -> Path:
    display_name = get_task_spec(task_key).display_name
    return RESULTS_DIR / f"{task_prefix(display_name)}_{method}_{_budget_label(budget)}_oracle.json"


def _plot_path(task_key: str, method: str, budget: int) -> Path:
    display_name = get_task_spec(task_key).display_name
    return FIGURES_DIR / f"{task_prefix(display_name)}_{method}_{_budget_label(budget)}_recovery_distribution.png"


def _slice_payload_matches(
    payload: dict[str, Any] | None,
    *,
    task_key: str,
    method: str,
    budget: int,
    num_samples: int,
    context_length: int,
    dataset_seed_offset: int,
    min_gap: float,
) -> bool:
    if not payload:
        return False
    aggregate = payload.get("aggregate", {})
    return (
        payload.get("schema_version") == "phase5-oracle-slice-v1"
        and payload.get("task_key") == task_key
        and payload.get("method") == method
        and int(payload.get("k_budget", -1)) == int(budget)
        and int(payload.get("context_length", -1)) == int(context_length)
        and int(payload.get("num_samples", -1)) == int(num_samples)
        and int(payload.get("dataset_seed_offset", -1)) == int(dataset_seed_offset)
        and int(aggregate.get("n_examples", -1)) == int(num_samples)
        and float(aggregate.get("min_gap", float("nan"))) == float(min_gap)
    )


def _serialization_payload_matches(
    payload: dict[str, Any] | None,
    *,
    tasks: Iterable[str],
    num_examples: int,
    context_length: int,
    dataset_seed_offset: int,
) -> bool:
    if not payload:
        return False
    aggregate = payload.get("aggregate", {})
    normalized_tasks = list(tasks)
    return (
        payload.get("tasks") == normalized_tasks
        and int(payload.get("num_examples_per_task", -1)) == int(num_examples)
        and int(payload.get("context_length", -1)) == int(context_length)
        and int(payload.get("dataset_seed_offset", -1)) == int(dataset_seed_offset)
        and int(aggregate.get("n_examples", -1)) == int(num_examples) * len(normalized_tasks)
    )


def _example_id(index: int) -> str:
    return f"ex{index + 1:03d}"


def _generation_kwargs(prepared) -> dict[str, int]:
    logical_position_base = int(prepared.context_ids.shape[1])
    return {
        "logical_position_base": logical_position_base,
        "max_new_tokens": int(prepared.example.max_new_tokens),
    }


def _stop_ids(model) -> list[int]:
    stop_ids = model.generation_config.eos_token_id
    if not isinstance(stop_ids, list):
        stop_ids = [stop_ids]
    return [int(stop_id) for stop_id in stop_ids if stop_id is not None]


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


def _generate_monolithic_answer(model, tokenizer, prepared) -> str:
    """Decode from one full prompt pass so Condition A stays monolithic."""

    device = model_device(model)
    prompt_ids = torch.cat(
        [
            prepared.context_ids.to(device),
            prepared.question_ids.to(device),
        ],
        dim=1,
    )
    max_new_tokens = int(prepared.example.max_new_tokens)
    stop_ids = _stop_ids(model)

    generated: list[torch.Tensor] = []
    with torch.no_grad():
        outputs = model(
            input_ids=prompt_ids,
            use_cache=True,
            logits_to_keep=1,
        )
        live_cache = outputs.past_key_values
        next_token = outputs.logits[0, -1].argmax()
        generated.append(next_token)
        current_position = torch.tensor([[int(prompt_ids.shape[1])]], device=device)
        current_cache_position = torch.tensor([int(prompt_ids.shape[1])], device=device)

        for _ in range(max_new_tokens - 1):
            outputs = model(
                input_ids=generated[-1].reshape(1, 1),
                past_key_values=live_cache,
                position_ids=current_position,
                cache_position=current_cache_position,
                use_cache=True,
            )
            live_cache = outputs.past_key_values
            next_token = outputs.logits[0, -1].argmax()
            generated.append(next_token)
            if int(next_token.item()) in stop_ids:
                break
            current_position = current_position + 1
            current_cache_position = current_cache_position + 1

    return tokenizer.decode(torch.stack(generated), skip_special_tokens=True)


def _score_prediction(prediction: str, outputs: list[str]) -> float:
    return float(sample_score(prediction, outputs))


def _query_logits(model, query_ids: torch.Tensor, cache, *, logical_position_base: int):
    device = model_device(model)
    query_ids = query_ids.to(device)
    seq_len = int(query_ids.shape[1])
    kwargs = {
        "input_ids": query_ids,
        "past_key_values": to_dynamic_cache(cache, config=model.config),
        "position_ids": torch.arange(logical_position_base, logical_position_base + seq_len, device=device).unsqueeze(0),
        "cache_position": torch.arange(len(cache), len(cache) + seq_len, device=device),
        "use_cache": True,
    }
    with torch.no_grad():
        try:
            return model(**kwargs)
        except TypeError:
            kwargs.pop("cache_position", None)
            return model(**kwargs)


def run_exact_serialization_suite(
    *,
    model,
    tokenizer,
    tasks: Iterable[str] = DEFAULT_TASKS,
    num_examples: int = DEFAULT_SERIALIZATION_SAMPLES,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    dataset_seed_offset: int = 0,
    threshold: float = DEFAULT_SERIALIZATION_THRESHOLD,
) -> dict[str, Any]:
    """Verify that the full-cache two-call path matches a monolithic pass."""

    ensure_results_dirs()
    normalized_tasks = _normalize_tasks(tasks)
    per_example: list[dict[str, Any]] = []

    for task_key in normalized_tasks:
        for index in range(int(num_examples)):
            example = build_task_example(
                task_key,
                index,
                context_length,
                tokenizer,
                dataset_seed_offset=dataset_seed_offset,
            )
            prepared = prepare_example_for_model(example, tokenizer)
            full_cache = build_position_tracked_cache(model, prepared.context_ids)
            logical_position_base = int(prepared.context_ids.shape[1])
            with tempfile.TemporaryDirectory(prefix="phase5_exact_kv_") as tmpdir:
                save_kv(full_cache, tmpdir)
                loaded_cache = load_kv(tmpdir, device=str(model_device(model)))

            direct_two_call = _query_logits(
                model,
                prepared.question_ids,
                full_cache,
                logical_position_base=logical_position_base,
            )
            loaded_two_call = _query_logits(
                model,
                prepared.question_ids,
                loaded_cache,
                logical_position_base=logical_position_base,
            )
            with torch.no_grad():
                single_call = model(
                    input_ids=torch.cat([prepared.context_ids.to(model_device(model)), prepared.question_ids.to(model_device(model))], dim=1),
                    use_cache=False,
                )
            query_len = int(prepared.question_ids.shape[1])
            reference_logits = single_call.logits[:, -query_len:, :]
            max_logit_diff = float((loaded_two_call.logits - reference_logits).abs().max().item())
            loaded_vs_direct_diff = float((loaded_two_call.logits - direct_two_call.logits).abs().max().item())
            last_token_diff = float((loaded_two_call.logits[:, -1:, :] - reference_logits[:, -1:, :]).abs().max().item())

            per_example.append(
                {
                    "task": get_task_spec(task_key).display_name,
                    "task_key": task_key,
                    "example_id": _example_id(index),
                    "max_logit_diff": round(max_logit_diff, 8),
                    "loaded_vs_direct_max_logit_diff": round(loaded_vs_direct_diff, 8),
                    "last_token_max_logit_diff": round(last_token_diff, 8),
                    "round_trip_exact": bool(loaded_vs_direct_diff < threshold),
                    "structurally_equivalent": bool(max_logit_diff < threshold),
                }
            )

            del full_cache, loaded_cache, direct_two_call, loaded_two_call, single_call
            torch.cuda.empty_cache()

    aggregate = {
        "threshold": float(threshold),
        "n_examples": len(per_example),
        "n_passed": sum(entry["structurally_equivalent"] for entry in per_example),
        "n_failed": sum(not entry["structurally_equivalent"] for entry in per_example),
        "max_logit_diff": round(max((entry["max_logit_diff"] for entry in per_example), default=0.0), 8),
        "n_round_trip_passed": sum(entry["round_trip_exact"] for entry in per_example),
        "n_round_trip_failed": sum(not entry["round_trip_exact"] for entry in per_example),
        "max_loaded_vs_direct_logit_diff": round(
            max((entry["loaded_vs_direct_max_logit_diff"] for entry in per_example), default=0.0),
            8,
        ),
        "max_last_token_diff": round(max((entry["last_token_max_logit_diff"] for entry in per_example), default=0.0), 8),
    }
    payload = {
        "tasks": normalized_tasks,
        "num_examples_per_task": int(num_examples),
        "context_length": int(context_length),
        "dataset_seed_offset": int(dataset_seed_offset),
        "aggregate": aggregate,
        "per_example": per_example,
    }
    write_json(DIAGNOSTICS_DIR / "exact_serialization.json", payload)
    return payload


def _run_oracle_slice(
    *,
    model,
    tokenizer,
    task_key: str,
    method: str,
    budget: int,
    num_samples: int,
    context_length: int,
    dataset_seed_offset: int,
) -> list[dict[str, Any]]:
    """Run one task/method/budget slice end to end."""

    rows: list[dict[str, Any]] = []
    display_name = get_task_spec(task_key).display_name
    task_log_dir = LOG_DIR / task_prefix(display_name) / method / _budget_label(budget)

    for index in range(int(num_samples)):
        example = build_task_example(
            task_key,
            index,
            context_length,
            tokenizer,
            dataset_seed_offset=dataset_seed_offset,
        )
        prepared = prepare_example_for_model(example, tokenizer)
        example_id = _example_id(index)
        full_cache = build_position_tracked_cache(model, prepared.context_ids)
        gold_outputs = list(prepared.example.outputs)

        a_start = time.perf_counter()
        condition_a_output = _generate_monolithic_answer(model, tokenizer, prepared)
        condition_a_generation_s = time.perf_counter() - a_start
        condition_a_score = _score_prediction(condition_a_output, gold_outputs)

        if method == "snapkv":
            precompute_policy = SnapKV(
                obs_window_size=DEFAULT_OBS_WINDOW_SIZE,
                sink_size=DEFAULT_SINK_SIZE,
                recency_window=0,
                pooling=DEFAULT_POOLING,
            )
            prepare_start = time.perf_counter()
            snap_cache, importance, obs_q_vecs = precompute_policy.prepare_eviction_inputs(full_cache)
            snap_prepare_s = time.perf_counter() - prepare_start

            policy = SnapKV(
                obs_window_size=DEFAULT_OBS_WINDOW_SIZE,
                sink_size=DEFAULT_SINK_SIZE,
                recency_window=_snapkv_recency_window(budget),
                pooling=DEFAULT_POOLING,
            )
            policy_start = time.perf_counter()
            eviction_result = policy.evict_from_precomputed(
                full_cache=snap_cache,
                k_budget=budget,
                importance=importance,
                obs_window_q_vecs=obs_q_vecs,
            )
            policy_duration_s = snap_prepare_s + (time.perf_counter() - policy_start)
        elif method == "streaming_llm":
            policy = StreamingLLM(
                sink_size=DEFAULT_SINK_SIZE,
                recency_window=_streaming_recency_window(budget),
            )
            policy_start = time.perf_counter()
            eviction_result = policy.evict(full_cache, k_budget=budget)
            policy_duration_s = time.perf_counter() - policy_start
        else:
            raise ValueError(f"Unsupported method: {method}")

        relevant_positions = [
            int(token_positions[0])
            for token_positions in (
                prepared.span_token_positions.get(span.name, [])
                for span in prepared.example.relevant_spans
            )
            if token_positions
        ]
        log_eviction(
            eviction_result,
            example_id=example_id,
            task=display_name,
            task_relevant_positions=relevant_positions,
            log_dir=task_log_dir,
            metadata={
                "method": method,
                "task_key": task_key,
                "context_length": int(context_length),
                "k_budget": int(budget),
                "policy_recency_window": _snapkv_recency_window(budget)
                if method == "snapkv"
                else _streaming_recency_window(budget),
            },
        )

        b_start = time.perf_counter()
        condition_b_output = _generate_answer(model, tokenizer, prepared, eviction_result.compressed)
        condition_b_generation_s = time.perf_counter() - b_start
        condition_b_score = _score_prediction(condition_b_output, gold_outputs)

        oracle_start = time.perf_counter()
        evicted_gpu = eviction_result.evicted.to_device(model_device(model), non_blocking=True)
        repaired_cache = inject_kv(
            eviction_result.compressed,
            evicted_gpu,
            evicted_gpu.positions,
        )
        oracle_output = _generate_answer(model, tokenizer, prepared, repaired_cache)
        oracle_generation_s = time.perf_counter() - oracle_start
        oracle_score = _score_prediction(oracle_output, gold_outputs)

        rows.append(
            {
                "example_id": example_id,
                "task": display_name,
                "task_key": task_key,
                "method": method,
                "k_budget": int(budget),
                "context_length": int(context_length),
                "condition_a_score": round(condition_a_score, 6),
                "condition_b_score": round(condition_b_score, 6),
                "oracle_score": round(oracle_score, 6),
                "condition_a_output": condition_a_output,
                "condition_b_output": condition_b_output,
                "oracle_output": oracle_output,
                "condition_a_source": "monolithic_full_prompt",
                "condition_b_source": "matched_pure_eviction_rerun",
                "gold_outputs": gold_outputs,
                "compressed_context_length": int(len(eviction_result.compressed)),
                "restored_token_count": int(len(eviction_result.evicted)),
                "task_relevant_positions": relevant_positions,
                "task_relevant_survived": [
                    position in set(int(value) for value in eviction_result.compressed.positions)
                    for position in relevant_positions
                ],
                "policy_duration_s": round(policy_duration_s, 6),
                "condition_a_generation_s": round(condition_a_generation_s, 6),
                "condition_b_generation_s": round(condition_b_generation_s, 6),
                "oracle_generation_s": round(oracle_generation_s, 6),
                "eviction_log_path": str(task_log_dir / f"{example_id}.json"),
            }
        )

        del full_cache, eviction_result, evicted_gpu, repaired_cache
        if method == "snapkv":
            del snap_cache, importance, obs_q_vecs
        torch.cuda.empty_cache()

    return rows


def run_phase5_oracle(
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    tasks: Iterable[str] = DEFAULT_TASKS,
    methods: Iterable[str] = DEFAULT_METHODS,
    budgets: Iterable[int | str] = DEFAULT_BUDGETS,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    dataset_seed_offset: int = 0,
    min_gap: float = DEFAULT_MIN_GAP,
    run_serialization_diagnostic: bool = True,
    serialization_examples: int = DEFAULT_SERIALIZATION_SAMPLES,
    reuse_matching_artifacts: bool = False,
) -> dict[str, Any]:
    """Run the full Phase 5 oracle sweep and save the expected artifacts."""

    ensure_results_dirs()
    normalized_tasks = _normalize_tasks(tasks)
    normalized_methods = _normalize_methods(methods)
    normalized_budgets = _normalize_budgets(budgets, context_length=context_length)

    tokenizer = load_tokenizer()
    model = load_model()

    serialization_payload = None
    if run_serialization_diagnostic:
        existing_serialization = _load_json(DIAGNOSTICS_DIR / "exact_serialization.json")
        if reuse_matching_artifacts and _serialization_payload_matches(
            existing_serialization,
            tasks=normalized_tasks,
            num_examples=serialization_examples,
            context_length=context_length,
            dataset_seed_offset=dataset_seed_offset,
        ):
            serialization_payload = existing_serialization
            print("[phase5] serialization-ready source=existing", flush=True)
        else:
            print("[phase5] serialization-ready source=rerun", flush=True)
            serialization_payload = run_exact_serialization_suite(
                model=model,
                tokenizer=tokenizer,
                tasks=normalized_tasks,
                num_examples=serialization_examples,
                context_length=context_length,
                dataset_seed_offset=dataset_seed_offset,
            )

    recovery_table: dict[str, Any] = {
        "schema_version": "phase5-oracle-v1",
        "context_length": int(context_length),
        "num_samples_requested": int(num_samples),
        "dataset_seed_offset": int(dataset_seed_offset),
        "primary_task_key": None,
        "comparison_notes": {
            "condition_a": "Recomputed monolithic full-prompt baseline on the exact matched examples.",
            "condition_b": (
                "Reran the pure-eviction baseline on the exact matched examples so the "
                "B->Oracle comparison stays aligned for vt_8hop_permute_div2 and the "
                "Phase 5 SnapKV recency-window policy."
            ),
        },
        "policy_defaults": {
            "obs_window_size": DEFAULT_OBS_WINDOW_SIZE,
            "sink_size": DEFAULT_SINK_SIZE,
            "snapkv_recency_window": "min(1024, k_budget - sink_size)",
            "streaming_recency_window": "k_budget - sink_size",
        },
        "tasks": {},
    }

    if DEFAULT_PRIMARY_TASK in normalized_tasks:
        primary_vt_task = DEFAULT_PRIMARY_TASK
    else:
        primary_vt_task = next((task_key for task_key in normalized_tasks if task_key.startswith("vt_")), normalized_tasks[0])
    recovery_table["primary_task_key"] = primary_vt_task

    for task_key in normalized_tasks:
        display_name = get_task_spec(task_key).display_name
        task_payload: dict[str, Any] = {"display_name": display_name}
        for method in normalized_methods:
            method_payload: dict[str, Any] = {}
            for budget in normalized_budgets:
                artifact_path = _artifact_path(task_key, method, budget)
                existing_slice = _load_json(artifact_path)
                payload = None
                payload_source = "existing"
                if reuse_matching_artifacts and _slice_payload_matches(
                    existing_slice,
                    task_key=task_key,
                    method=method,
                    budget=budget,
                    num_samples=num_samples,
                    context_length=context_length,
                    dataset_seed_offset=dataset_seed_offset,
                    min_gap=min_gap,
                ):
                    payload = existing_slice

                if payload is None:
                    payload_source = "rerun"
                    print(
                        f"[phase5] task={display_name} method={method} budget={budget} num_samples={int(num_samples)}",
                        flush=True,
                    )
                    rows = _run_oracle_slice(
                        model=model,
                        tokenizer=tokenizer,
                        task_key=task_key,
                        method=method,
                        budget=budget,
                        num_samples=num_samples,
                        context_length=context_length,
                        dataset_seed_offset=dataset_seed_offset,
                    )
                    aggregate = compute_oracle_recovery(rows, min_gap=min_gap)
                    payload = {
                        "schema_version": "phase5-oracle-slice-v1",
                        "task": display_name,
                        "task_key": task_key,
                        "method": method,
                        "k_budget": int(budget),
                        "context_length": int(context_length),
                        "num_samples": int(num_samples),
                        "dataset_seed_offset": int(dataset_seed_offset),
                        "aggregate": {key: value for key, value in aggregate.items() if key != "per_example"},
                        "per_example": aggregate["per_example"],
                    }
                    write_json(artifact_path, payload)

                print(
                    f"[phase5] slice-ready source={payload_source} task={display_name} method={method} "
                    f"budget={budget} num_samples={int(payload['num_samples'])}",
                    flush=True,
                )
                plot_recovery_distribution(
                    payload,
                    task_label=display_name,
                    budget=budget,
                    method=method,
                    save_path=_plot_path(task_key, method, budget),
                )
                method_payload[_budget_label(budget)] = {
                    "artifact_path": str(artifact_path),
                    "aggregate": payload["aggregate"],
                }
            task_payload[method] = method_payload
        recovery_table["tasks"][task_key] = task_payload

    write_json(RESULTS_DIR / "recovery_table.json", recovery_table)
    plot_oracle_vs_budget(recovery_table, save_path=FIGURES_DIR / "oracle_vs_k_budget.png")

    go_nogo_text = format_go_nogo(
        recovery_table=recovery_table,
        serialization_diagnostic=serialization_payload,
        primary_task_key=primary_vt_task,
        primary_method="snapkv",
        primary_budget=512,
    )
    (RESULTS_DIR / "go_nogo.txt").write_text(go_nogo_text, encoding="utf-8")

    summary = {
        "results_dir": str(RESULTS_DIR),
        "recovery_table_path": str(RESULTS_DIR / "recovery_table.json"),
        "go_nogo_path": str(RESULTS_DIR / "go_nogo.txt"),
        "serialization_diagnostic_path": None
        if serialization_payload is None
        else str(DIAGNOSTICS_DIR / "exact_serialization.json"),
        "recovery_table": recovery_table,
        "serialization_diagnostic": serialization_payload,
    }
    write_json(RESULTS_DIR / "phase5_summary.json", summary)
    return summary
