#!/usr/bin/env python3
"""Parallel stress-budget runner for the corrected Phase 1 compressed-cache path."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_ROOT))

from phase1 import DEFAULT_CONTEXT_LENGTHS, DEFAULT_TASKS
from phase1.evaluation import build_phase1_summary, task_prefix, write_json
from phase1.modeling import load_model, load_tokenizer
from phase1.paths import ARTIFACTS_DIR, LOGS_DIR, MODEL_DIR, RESULTS_DIR
from phase1.runner import _load_or_generate_examples, _normalize_tasks, _run_task_budget, _select_num_samples


def _normalize_budgets(budgets: list[int]) -> list[int]:
    """Deduplicate requested budgets while preserving the CLI order."""
    ordered: list[int] = []
    for budget in budgets:
        value = int(budget)
        if value <= 0:
            raise ValueError(f"Budgets must be positive; got {value}.")
        if value not in ordered:
            ordered.append(value)
    return ordered


def _task_condition_path(run_label: str, task_name: str) -> Path:
    """Store one shard's flattened condition rows under a stable run label."""
    return RESULTS_DIR / f"{run_label}_{task_name}_condition_b.json"


def _task_summary_path(run_label: str, task_name: str) -> Path:
    """Store one shard's summary under a stable run label."""
    return RESULTS_DIR / f"{run_label}_{task_name}_summary.json"


def _combined_condition_path(run_label: str) -> Path:
    """Store the merged condition rows for all completed task shards."""
    return RESULTS_DIR / f"{run_label}_condition_b.json"


def _combined_summary_path(run_label: str) -> Path:
    """Store the merged summary for all completed task shards."""
    return RESULTS_DIR / f"{run_label}_summary.json"


def _trace_root(run_label: str) -> Path:
    """Isolate raw trace tensors for one rerun label."""
    return ARTIFACTS_DIR / run_label / "trace"


def _eviction_log_root(run_label: str) -> Path:
    """Isolate human-readable eviction logs for one rerun label."""
    return RESULTS_DIR / f"{run_label}_eviction_logs"


def _q_vectors_root(run_label: str) -> Path:
    """Isolate query-vector snapshots for one rerun label."""
    return RESULTS_DIR / f"{run_label}_q_vectors"


def _worker_log_path(run_label: str, task_name: str) -> Path:
    """Capture each worker's stdout/stderr into a dedicated log file."""
    return LOGS_DIR / f"{run_label}_{task_name}.log"


def _configure_labeled_paths(run_label: str) -> None:
    """Redirect per-example traces/logs into run-specific folders."""
    import phase1.runner as runner

    trace_root = _trace_root(run_label)
    eviction_root = _eviction_log_root(run_label)
    qvec_root = _q_vectors_root(run_label)

    def trace_path(task_name: str, context_length: int, budget: int, example_index: int) -> Path:
        return trace_root / task_name / str(context_length) / "snapkv" / str(budget) / f"{example_index:05d}.pt"

    def eviction_log_path(display_name: str, budget: int, example_index: int) -> Path:
        return eviction_root / f"{task_prefix(display_name)}_k{budget}_ex{example_index + 1:03d}.json"

    def q_vectors_path(display_name: str, budget: int, example_index: int) -> Path:
        return qvec_root / f"{task_prefix(display_name)}_k{budget}_ex{example_index + 1:03d}_qvecs.pt"

    runner.trace_path = trace_path
    runner.eviction_log_path = eviction_log_path
    runner.q_vectors_path = q_vectors_path


def _run_task_worker(
    *,
    task_name: str,
    context_length: int,
    budgets: list[int],
    num_samples: int | None,
    force: bool,
    query_log_tokens: int,
    run_label: str,
    dataset_seed_offset: int,
) -> None:
    """Run one task shard end-to-end and write task-local outputs."""
    _configure_labeled_paths(run_label)
    tokenizer = load_tokenizer(str(MODEL_DIR))
    model = load_model(str(MODEL_DIR))
    sample_count = _select_num_samples(task_name, num_samples)
    examples = _load_or_generate_examples(
        task_name,
        context_length,
        sample_count,
        tokenizer,
        force=False,
        dataset_seed_offset=dataset_seed_offset,
    )
    rows: list[dict] = []
    for budget in budgets:
        rows.extend(
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
    write_json(_task_condition_path(run_label, task_name), rows)
    write_json(_task_summary_path(run_label, task_name), build_phase1_summary(rows))


def _launch_workers(
    *,
    tasks: list[str],
    context_length: int,
    budgets: list[int],
    num_samples: int | None,
    force: bool,
    query_log_tokens: int,
    run_label: str,
    max_parallel: int,
    dataset_seed_offset: int,
) -> None:
    """Run task shards in parallel subprocesses and fail fast on worker errors."""
    script_path = Path(__file__).resolve()
    pending = list(tasks)
    active: dict[str, tuple[subprocess.Popen[str], object, Path]] = {}
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    while pending or active:
        while pending and len(active) < max_parallel:
            task_name = pending.pop(0)
            log_path = _worker_log_path(run_label, task_name)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(log_path, "w", encoding="utf-8")
            cmd = [
                sys.executable,
                str(script_path),
                "--worker",
                "--task",
                task_name,
                "--context-length",
                str(context_length),
                "--run-label",
                run_label,
                "--query-log-tokens",
                str(query_log_tokens),
                "--dataset-seed-offset",
                str(dataset_seed_offset),
                "--budgets",
                *[str(budget) for budget in budgets],
            ]
            if num_samples is not None:
                cmd.extend(["--num-samples", str(num_samples)])
            if force:
                cmd.append("--force")
            proc = subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).resolve().parent),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            active[task_name] = (proc, handle, log_path)
            print(f"[parallel-lowk] started task={task_name} pid={proc.pid} log={log_path}", flush=True)

        time.sleep(5)
        finished: list[str] = []
        for task_name, (proc, handle, log_path) in active.items():
            returncode = proc.poll()
            if returncode is None:
                continue
            handle.close()
            if returncode != 0:
                raise RuntimeError(f"Worker failed for task={task_name} exit={returncode}; see {log_path}")
            print(f"[parallel-lowk] finished task={task_name} log={log_path}", flush=True)
            finished.append(task_name)
        for task_name in finished:
            active.pop(task_name, None)


def _merge_task_outputs(run_label: str, tasks: list[str]) -> None:
    """Merge per-task shard outputs into the final run-level summary files."""
    rows: list[dict] = []
    for task_name in tasks:
        condition_path = _task_condition_path(run_label, task_name)
        with open(condition_path, "r", encoding="utf-8") as handle:
            rows.extend(json.load(handle))
    write_json(_combined_condition_path(run_label), rows)
    write_json(_combined_summary_path(run_label), build_phase1_summary(rows))


def parse_args() -> argparse.Namespace:
    """Expose a small CLI for the low-budget parallel rerun."""
    parser = argparse.ArgumentParser(description="Run corrected Phase 1 low-budget stress sweeps in parallel shards.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTHS[0])
    parser.add_argument("--budgets", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--query-log-tokens", type=int, default=64)
    parser.add_argument("--dataset-seed-offset", type=int, default=0)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--task")
    return parser.parse_args()


def main() -> None:
    """Dispatch to either the worker path or the multi-worker orchestrator."""
    args = parse_args()
    budgets = _normalize_budgets(args.budgets)
    if args.context_length != 32768:
        raise ValueError(f"This harness is only wired for the reduced 32K sweep; got {args.context_length}.")

    if args.worker:
        if not args.task:
            raise ValueError("--worker requires --task.")
        _run_task_worker(
            task_name=args.task,
            context_length=args.context_length,
            budgets=budgets,
            num_samples=args.num_samples,
            force=args.force,
            query_log_tokens=args.query_log_tokens,
            run_label=args.run_label,
            dataset_seed_offset=args.dataset_seed_offset,
        )
        return

    tasks = _normalize_tasks(args.tasks)
    if not tasks:
        raise ValueError("At least one task is required.")
    max_parallel = max(1, min(args.max_parallel, len(tasks)))
    _launch_workers(
        tasks=tasks,
        context_length=args.context_length,
        budgets=budgets,
        num_samples=args.num_samples,
        force=args.force,
        query_log_tokens=args.query_log_tokens,
        run_label=args.run_label,
        max_parallel=max_parallel,
        dataset_seed_offset=args.dataset_seed_offset,
    )
    _merge_task_outputs(args.run_label, tasks)
    print(
        f"[parallel-lowk] wrote {_combined_condition_path(args.run_label)} and {_combined_summary_path(args.run_label)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
