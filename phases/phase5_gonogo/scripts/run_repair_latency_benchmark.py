#!/usr/bin/env python3
"""Benchmark the oracle-style KV repair path without downstream generation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_ROOT.parents[1]
for root in (PHASE_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from oracle.runner import DEFAULT_CONTEXT_LENGTH, DEFAULT_OBS_WINDOW_SIZE, DEFAULT_POOLING, DEFAULT_SINK_SIZE  # noqa: E402
from phases.phase1_degradation.phase1.inference import prepare_example_for_model  # noqa: E402
from phases.phase1_degradation.phase1.task_registry import build_task_example  # noqa: E402
from phases.phase2_kv_cache.src.kv_utils import inject_kv  # noqa: E402
from phases.phase2_kv_cache.src.runtime import load_model, load_tokenizer, model_device  # noqa: E402
from phases.phase3_eviction.src.eviction import SnapKV, StreamingLLM  # noqa: E402
from phases.phase3_eviction.src.runtime import build_position_tracked_cache, write_json  # noqa: E402

DEFAULT_TASK = "vt_8hop_permute_div2"
DEFAULT_METHOD = "snapkv"
DEFAULT_K_BUDGET = 512
DEFAULT_NUM_SAMPLES = 100
DEFAULT_DATASET_SEED_OFFSET = 0
RESULTS_DIR = PHASE_ROOT / "results" / "repair_latency"
DEFAULT_SNAPKV_RECENCY_CAP = 1024


def _snapkv_recency_window(k_budget: int) -> int:
    return max(0, min(DEFAULT_SNAPKV_RECENCY_CAP, int(k_budget) - DEFAULT_SINK_SIZE))


def _streaming_recency_window(k_budget: int) -> int:
    return max(0, int(k_budget) - DEFAULT_SINK_SIZE)


def _percentiles_ms(values_s: list[float]) -> dict[str, float]:
    values = np.asarray(values_s, dtype=np.float64) * 1000.0
    return {
        "mean_ms": float(values.mean()),
        "median_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task key to benchmark.")
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        choices=("snapkv", "streaming_llm"),
        help="Eviction method used before repair.",
    )
    parser.add_argument("--k-budget", type=int, default=DEFAULT_K_BUDGET, help="Eviction budget to benchmark.")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Number of deterministic examples.")
    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help="Context length to generate for each example.",
    )
    parser.add_argument(
        "--dataset-seed-offset",
        type=int,
        default=DEFAULT_DATASET_SEED_OFFSET,
        help="Optional deterministic dataset seed offset.",
    )
    parser.add_argument(
        "--output-path",
        default="",
        help="Optional JSON artifact path. Defaults to results/repair_latency/<task>_<method>_k<budget>.json",
    )
    return parser.parse_args()


def _benchmark_one_example(
    *,
    model,
    tokenizer,
    task_key: str,
    method: str,
    k_budget: int,
    context_length: int,
    dataset_seed_offset: int,
    sample_index: int,
) -> dict[str, Any]:
    example = build_task_example(
        task_key,
        sample_index,
        context_length,
        tokenizer,
        dataset_seed_offset=dataset_seed_offset,
    )
    prepared = prepare_example_for_model(example, tokenizer)
    full_cache = build_position_tracked_cache(model, prepared.context_ids)

    if method == "snapkv":
        precompute_policy = SnapKV(
            obs_window_size=DEFAULT_OBS_WINDOW_SIZE,
            sink_size=DEFAULT_SINK_SIZE,
            recency_window=0,
            pooling=DEFAULT_POOLING,
        )
        snap_cache, importance, obs_q_vecs = precompute_policy.prepare_eviction_inputs(full_cache)
        policy = SnapKV(
            obs_window_size=DEFAULT_OBS_WINDOW_SIZE,
            sink_size=DEFAULT_SINK_SIZE,
            recency_window=_snapkv_recency_window(k_budget),
            pooling=DEFAULT_POOLING,
        )
        eviction_result = policy.evict_from_precomputed(
            full_cache=snap_cache,
            k_budget=k_budget,
            importance=importance,
            obs_window_q_vecs=obs_q_vecs,
        )
        del snap_cache, importance, obs_q_vecs
    elif method == "streaming_llm":
        policy = StreamingLLM(
            sink_size=DEFAULT_SINK_SIZE,
            recency_window=_streaming_recency_window(k_budget),
        )
        eviction_result = policy.evict(full_cache, k_budget=k_budget)
    else:
        raise ValueError(f"Unsupported method: {method}")

    del full_cache

    device = model_device(model)
    torch.cuda.synchronize(device)

    transfer_start = time.perf_counter()
    evicted_gpu = eviction_result.evicted.to_device(device, non_blocking=True)
    torch.cuda.synchronize(device)
    transfer_s = time.perf_counter() - transfer_start

    inject_start = time.perf_counter()
    repaired_cache = inject_kv(
        eviction_result.compressed,
        evicted_gpu,
        evicted_gpu.positions,
    )
    torch.cuda.synchronize(device)
    inject_s = time.perf_counter() - inject_start

    total_s = transfer_s + inject_s
    restored_token_count = int(len(eviction_result.evicted))
    compressed_token_count = int(len(eviction_result.compressed))
    repaired_token_count = int(len(repaired_cache))

    del eviction_result, evicted_gpu, repaired_cache
    torch.cuda.empty_cache()

    return {
        "sample_index": int(sample_index),
        "transfer_ms": round(transfer_s * 1000.0, 6),
        "inject_ms": round(inject_s * 1000.0, 6),
        "repair_total_ms": round(total_s * 1000.0, 6),
        "restored_token_count": restored_token_count,
        "compressed_token_count": compressed_token_count,
        "repaired_token_count": repaired_token_count,
    }


def main() -> int:
    args = _parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output_path)
        if args.output_path
        else RESULTS_DIR / f"{args.task}_{args.method}_k{int(args.k_budget)}_repair_latency.json"
    )

    model = load_model()
    tokenizer = load_tokenizer()

    per_example: list[dict[str, Any]] = []
    for sample_index in range(int(args.num_samples)):
        row = _benchmark_one_example(
            model=model,
            tokenizer=tokenizer,
            task_key=args.task,
            method=args.method,
            k_budget=int(args.k_budget),
            context_length=int(args.context_length),
            dataset_seed_offset=int(args.dataset_seed_offset),
            sample_index=sample_index,
        )
        per_example.append(row)
        print(
            f"[{sample_index + 1:03d}/{int(args.num_samples):03d}] "
            f"repair_total_ms={row['repair_total_ms']:.3f} "
            f"transfer_ms={row['transfer_ms']:.3f} "
            f"inject_ms={row['inject_ms']:.3f} "
            f"restored={row['restored_token_count']}",
            flush=True,
        )

    transfer_values_s = [row["transfer_ms"] / 1000.0 for row in per_example]
    inject_values_s = [row["inject_ms"] / 1000.0 for row in per_example]
    total_values_s = [row["repair_total_ms"] / 1000.0 for row in per_example]
    restored_counts = [int(row["restored_token_count"]) for row in per_example]

    payload = {
        "schema_version": "phase5-repair-latency-v1",
        "task_key": args.task,
        "method": args.method,
        "k_budget": int(args.k_budget),
        "num_samples": int(args.num_samples),
        "context_length": int(args.context_length),
        "dataset_seed_offset": int(args.dataset_seed_offset),
        "repair_definition": (
            "Oracle-style full repair only: move the CPU-resident evicted fragment back to GPU "
            "and rebuild the repaired cache via inject_kv. Downstream generation is excluded."
        ),
        "aggregate": {
            "transfer": _percentiles_ms(transfer_values_s),
            "inject": _percentiles_ms(inject_values_s),
            "repair_total": _percentiles_ms(total_values_s),
            "mean_restored_token_count": float(np.mean(restored_counts)),
            "min_restored_token_count": int(min(restored_counts)),
            "max_restored_token_count": int(max(restored_counts)),
        },
        "per_example": per_example,
    }
    write_json(output_path, payload)
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))
    print(f"Wrote repair-latency artifact to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
