"""Top-level orchestration for dataset generation, inference, tracing, and summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch

from .config import DEFAULT_ALGORITHMS, DEFAULT_BUDGETS, DEFAULT_CONTEXT_LENGTHS, DEFAULT_TASKS
from .evaluation import (
    build_condition_b_record,
    build_detailed_eviction_log,
    build_phase1_summary,
    load_trace_payload,
    summarize_trace,
    task_prefix,
    write_json,
)
from .helpers import read_jsonl, write_jsonl
from .inference import prepare_example_for_model, run_example
from .modeling import load_model, load_tokenizer
from .paths import ARTIFACTS_DIR, MODEL_DIR, RESULTS_DIR
from .task_registry import build_task_example, get_task_spec


RUN_ARTIFACTS_DIR = ARTIFACTS_DIR / "reduced_phase1_32k_snapkv"
EVICTION_LOGS_DIR = RESULTS_DIR / "phase1_eviction_logs"
Q_VECTORS_DIR = RESULTS_DIR / "phase1_q_vectors"
CONDITION_B_PATH = RESULTS_DIR / "phase1_condition_b.json"
SUMMARY_PATH = RESULTS_DIR / "phase1_summary.json"

CONDITION_RECORD_KEYS = [
    "example_id",
    "task",
    "task_key",
    "k_budget",
    "context_length",
    "raw_model_output",
    "gold_answer",
    "gold_outputs",
    "correct",
    "score_fraction",
    "matched_count",
    "matched_outputs",
    "error_type",
    "task_relevant_positions",
    "task_relevant_survived",
    "task_relevant_spans",
    "compressed_context_length",
    "first_broken_hop",
    "first_broken_hop_depth",
    "eviction_log_path",
    "q_vectors_path",
]


def dataset_path(task_name: str, context_length: int) -> Path:
    """Location of the cached synthetic dataset for one task/context pair."""
    return RUN_ARTIFACTS_DIR / "data" / task_name / str(context_length) / "validation.jsonl"


def trace_path(task_name: str, context_length: int, budget: int, example_index: int) -> Path:
    """Location of the raw `.pt` eviction trace for one compressed run."""
    return RUN_ARTIFACTS_DIR / "trace" / task_name / str(context_length) / "snapkv" / str(budget) / f"{example_index:05d}.pt"


def eviction_log_path(display_name: str, budget: int, example_index: int) -> Path:
    """Location of the human-readable JSON eviction log for one example."""
    return EVICTION_LOGS_DIR / f"{task_prefix(display_name)}_k{budget}_ex{example_index + 1:03d}.json"


def q_vectors_path(display_name: str, budget: int, example_index: int) -> Path:
    """Location of the saved query-vector snapshot for one example."""
    return Q_VECTORS_DIR / f"{task_prefix(display_name)}_k{budget}_ex{example_index + 1:03d}_qvecs.pt"


def _spec_signature(task_name: str, dataset_seed_offset: int = 0) -> dict:
    """Capture the parts of a task spec that make cached datasets still valid."""
    spec = get_task_spec(task_name)
    return {
        "display_name": spec.display_name,
        "family": spec.family,
        "max_new_tokens": spec.max_new_tokens,
        "params": spec.params,
        "dataset_seed_offset": int(dataset_seed_offset),
    }


def _normalize_tasks(tasks: Iterable[str]) -> list[str]:
    """Deduplicate tasks while preserving the CLI order."""
    ordered: list[str] = []
    for task_name in tasks:
        if task_name not in ordered:
            ordered.append(task_name)
    return ordered


def _select_budgets(task_name: str, budgets: Iterable[int] | None) -> list[int]:
    """Filter requested budgets down to the ones allowed for the given task."""
    spec = get_task_spec(task_name)
    if budgets is None:
        return list(spec.default_budgets)
    allowed = set(spec.default_budgets)
    return [budget for budget in budgets if budget in allowed]


def _select_num_samples(task_name: str, num_samples: int | None) -> int:
    """Use the task default sample count unless the caller overrides it."""
    spec = get_task_spec(task_name)
    return spec.default_num_samples if num_samples is None else int(num_samples)


def _load_or_generate_examples(
    task_name: str,
    context_length: int,
    num_samples: int,
    tokenizer,
    force: bool,
    *,
    dataset_seed_offset: int = 0,
) -> list:
    """Reuse a cached dataset when possible; otherwise regenerate it for the requested dataset seed."""
    path = dataset_path(task_name, context_length)
    expected_signature = _spec_signature(task_name, dataset_seed_offset)
    if path.exists() and not force:
        rows = read_jsonl(path)
        # The spec signature guards against silently reusing stale datasets
        # after task-generation logic changes.
        if len(rows) == num_samples and all(row.get("metadata", {}).get("spec_signature") == expected_signature for row in rows):
            from .models import TaskExample

            return [TaskExample.from_dict(row) for row in rows]

    # No usable cache was found, so rebuild the dataset from the current code.
    examples = [
        build_task_example(
            task_name,
            index,
            context_length,
            tokenizer,
            dataset_seed_offset=dataset_seed_offset,
        )
        for index in range(num_samples)
    ]
    for example in examples:
        example.metadata["spec_signature"] = expected_signature
    write_jsonl(path, [example.to_dict() for example in examples])
    return examples


def _read_condition_record_from_log(path: Path) -> dict:
    """Load just the flattened condition fields from a cached detailed log."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {key: payload[key] for key in CONDITION_RECORD_KEYS}


def _run_task_budget(
    *,
    model,
    tokenizer,
    task_name: str,
    context_length: int,
    budget: int,
    examples,
    query_log_tokens: int,
    force: bool,
) -> list[dict]:
    """Run or reuse every sample for one task at one compression budget."""
    # Section: derive task metadata and tokenize examples once for this sweep.
    spec = get_task_spec(task_name)
    prepared_examples = [prepare_example_for_model(example, tokenizer) for example in examples]
    rows: list[dict] = []
    total_examples = len(prepared_examples)
    print(f"[phase1] start task={spec.display_name} budget={budget} examples={total_examples}", flush=True)

    for example_idx, prepared in enumerate(prepared_examples, start=1):
        # Section: compute artifact paths and decide whether we can reuse cached logs.
        example = prepared.example
        detailed_log = eviction_log_path(spec.display_name, budget, example.index)
        qvec_path = q_vectors_path(spec.display_name, budget, example.index)
        if detailed_log.exists() and qvec_path.exists() and not force:
            # Reuse the JSON log when it already exists; that avoids another
            # model run while still giving the summary code the exact same schema.
            rows.append(_read_condition_record_from_log(detailed_log))
            if example_idx == 1 or example_idx % 10 == 0 or example_idx == total_examples:
                print(
                    f"[phase1] cached task={spec.display_name} budget={budget} example={example_idx}/{total_examples}",
                    flush=True,
                )
            continue

        # Section: run inference and collect the raw eviction trace.
        # Otherwise run the compressed inference path and turn its raw tensors
        # into the human-readable logs used by later analysis.
        record = run_example(
            model=model,
            tokenizer=tokenizer,
            prepared=prepared,
            algorithm="snapkv",
            budget=budget,
            condition="condition_b",
            trace_path=trace_path(task_name, context_length, budget, example.index),
            query_log_tokens=query_log_tokens,
        )
        if record.trace_path is None:
            raise RuntimeError("Condition B run did not emit a trace path.")

        # Section: summarize trace outputs and persist query-vector snapshots.
        trace_payload = load_trace_payload(Path(record.trace_path))
        trace_summary = summarize_trace(trace_payload)
        qvec_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trace_summary["query_vectors"], qvec_path)

        # Section: build flattened condition records plus richer debug logs.
        # Emit both a flat condition record and a richer log with the full
        # eviction mask and token-importance scores.
        condition_record = build_condition_b_record(
            example=example,
            record=record,
            span_token_positions=prepared.span_token_positions,
            detailed_log_path=detailed_log,
            q_vectors_path=qvec_path,
        )
        detailed_payload = build_detailed_eviction_log(
            condition_record=condition_record,
            trace_summary=trace_summary,
        )
        write_json(detailed_log, detailed_payload)
        rows.append(condition_record)
        if example_idx == 1 or example_idx % 10 == 0 or example_idx == total_examples:
            print(
                f"[phase1] done task={spec.display_name} budget={budget} example={example_idx}/{total_examples} "
                f"score={condition_record['score_fraction']}",
                flush=True,
            )
    print(f"[phase1] finish task={spec.display_name} budget={budget}", flush=True)
    return rows


def run_phase1(
    *,
    tasks: Iterable[str] = DEFAULT_TASKS,
    context_lengths: Iterable[int] = DEFAULT_CONTEXT_LENGTHS,
    budgets: Iterable[int] | None = None,
    algorithms: Iterable[str] = DEFAULT_ALGORITHMS,
    num_samples: int | None = None,
    force: bool = False,
    query_log_tokens: int = 64,
) -> dict[str, object]:
    """Run the reduced Phase 1 benchmark end to end and write its JSON outputs."""
    # Section: validate the reduced Phase 1 constraints and normalize inputs.
    algorithms = list(algorithms)
    if set(algorithms) != {"snapkv"}:
        raise ValueError(f"Reduced Phase 1 only supports SnapKV; got {algorithms}.")
    context_lengths = list(context_lengths)
    if set(context_lengths) != {32768}:
        raise ValueError(f"Reduced Phase 1 only supports 32K context; got {context_lengths}.")

    # Section: load the single model/tokenizer pair used by this benchmark.
    # Phase 1 is intentionally narrow right now: one model backend, one context
    # length, and SnapKV budgets for the reduced benchmark sweep.
    tokenizer = load_tokenizer(str(MODEL_DIR))
    model = load_model(str(MODEL_DIR))

    condition_rows: list[dict] = []
    for task_name in _normalize_tasks(tasks):
        # Section: select per-task budgets/samples, then sweep contexts and budgets.
        task_budgets = _select_budgets(task_name, budgets)
        if not task_budgets:
            continue
        sample_count = _select_num_samples(task_name, num_samples)
        for context_length in context_lengths:
            # Reuse the same generated dataset across all budgets so budget is
            # the only thing changing within the sweep.
            examples = _load_or_generate_examples(task_name, context_length, sample_count, tokenizer, force)
            for budget in task_budgets:
                condition_rows.extend(
                    _run_task_budget(
                        model=model,
                        tokenizer=tokenizer,
                        task_name=task_name,
                        context_length=context_length,
                        budget=budget,
                        examples=examples,
                        query_log_tokens=query_log_tokens,
                        force=force,
                    )
                )

    # Section: aggregate the run outputs into the two JSON artifacts.
    summary = build_phase1_summary(condition_rows)
    write_json(CONDITION_B_PATH, condition_rows)
    write_json(SUMMARY_PATH, summary)
    return {
        "condition_b": condition_rows,
        "summary": summary,
    }
