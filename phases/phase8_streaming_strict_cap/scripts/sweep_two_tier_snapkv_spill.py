#!/usr/bin/env python3
"""Coverage sweep for two-tier SnapKV GPU/CPU spill."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Iterable

PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_ROOT.parents[1]
for root in (PHASE_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from phases.phase2_kv_cache.src.runtime import load_model, load_tokenizer  # noqa: E402
from phases.phase3_eviction.src.runtime import write_json  # noqa: E402
from phases.phase6_repair.src.protocol import (  # noqa: E402
    CLEAN_SPLIT_SPECS,
    SPLIT_SPECS_BY_NAME,
    build_base_example,
    build_split_prepared_from_base_example,
    relevant_positions_for_spans,
)
from phases.phase8_streaming_strict_cap.src.streaming import stream_context_two_tier_snapkv_spill  # noqa: E402


RESULTS_DIR = PHASE_ROOT / "results" / "two_tier_snapkv"


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return float(fmean(values)) if values else 0.0


def _overlap_fraction(selected_positions: Iterable[int], relevant_positions: Iterable[int]) -> float:
    selected = {int(position) for position in selected_positions}
    relevant = {int(position) for position in relevant_positions}
    if not relevant:
        return 0.0
    return float(len(selected & relevant) / len(relevant))


def _split_specs(task: str):
    if task == "clean_suite":
        return CLEAN_SPLIT_SPECS
    spec = SPLIT_SPECS_BY_NAME.get(task)
    if spec is None:
        raise ValueError(f"Unsupported task: {task!r}.")
    return (spec,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="clean_suite")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--total-context-length", type=int, default=65_536)
    parser.add_argument("--gpu-cache-cap", type=int, default=32_768)
    parser.add_argument("--turn-headroom", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--cpu-store-fractions", nargs="+", type=float, default=[0.10, 0.25, 0.50, 1.0])
    parser.add_argument("--dataset-seed-offset", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    fractions = tuple(sorted(dict.fromkeys(float(value) for value in args.cpu_store_fractions)))
    if not fractions or any(value < 0.0 or value > 1.0 for value in fractions):
        raise ValueError("cpu-store-fractions must contain values in [0, 1].")

    model = load_model()
    tokenizer = load_tokenizer()
    split_specs = _split_specs(args.task)

    rows: list[dict] = []
    stream_rows: list[dict] = []
    for index in range(int(args.num_samples)):
        base_example = build_base_example(
            split_spec=split_specs[0],
            index=index,
            context_length=int(args.total_context_length),
            tokenizer=tokenizer,
            dataset_seed_offset=int(args.dataset_seed_offset),
        )
        split_views = tuple(
            build_split_prepared_from_base_example(
                base_example=base_example,
                split_spec=split_spec,
                tokenizer=tokenizer,
            )
            for split_spec in split_specs
        )

        for fraction in fractions:
            result = stream_context_two_tier_snapkv_spill(
                model=model,
                context_ids=split_views[0].q1_prepared.context_ids,
                total_context_length=int(args.total_context_length),
                chunk_size=int(args.chunk_size),
                gpu_cache_cap=int(args.gpu_cache_cap),
                turn_headroom=int(args.turn_headroom),
                cpu_store_fraction=float(fraction),
                sink_size=4,
                recency_window=128,
                obs_window_size=128,
                pooling="max",
            )
            active_positions = set(int(position) for position in result.active_cache.positions)
            cpu_positions = set(int(position) for position in result.cpu_spill_positions)
            stream_rows.append(
                {
                    "index": int(index),
                    "cpu_store_fraction": float(fraction),
                    "stream_prefill_s": round(result.stream_prefill_s, 6),
                    "final_active_context_tokens": int(result.final_active_context_tokens),
                    "peak_active_context_tokens": int(result.peak_active_context_tokens),
                    "eviction_events": int(result.eviction_events),
                    "total_cpu_spill_tokens": int(result.total_cpu_spill_tokens),
                    "permanent_evicted_tokens": int(result.permanent_evicted_tokens),
                    "event_summaries": list(result.event_summaries),
                }
            )

            for split in split_views:
                q2_relevant = tuple(relevant_positions_for_spans(split.q2_prepared, split.q2_span_names))
                active_relevant = sorted(active_positions & set(q2_relevant))
                cpu_relevant = sorted(cpu_positions & set(q2_relevant))
                accessible_relevant = sorted((active_positions | cpu_positions) & set(q2_relevant))
                evicted_relevant = [position for position in q2_relevant if int(position) not in active_positions]
                rows.append(
                    {
                        "index": int(index),
                        "task": split.split_spec.name,
                        "cpu_store_fraction": float(fraction),
                        "cpu_store_percent": round(float(fraction) * 100.0, 4),
                        "q2_relevant_tokens": len(q2_relevant),
                        "active_relevant_tokens": len(active_relevant),
                        "cpu_relevant_tokens": len(cpu_relevant),
                        "evicted_q2_relevant_tokens": len(evicted_relevant),
                        "accessible_relevant_tokens": len(accessible_relevant),
                        "active_coverage": round(_overlap_fraction(active_positions, q2_relevant), 6),
                        "cpu_coverage": round(_overlap_fraction(cpu_positions, q2_relevant), 6),
                        "accessible_coverage": round(_overlap_fraction(active_positions | cpu_positions, q2_relevant), 6),
                        "cpu_coverage_of_evicted_relevant": round(_overlap_fraction(cpu_positions, evicted_relevant), 6),
                        "active_relevant_positions": active_relevant,
                        "cpu_relevant_positions": cpu_relevant,
                        "accessible_relevant_positions": accessible_relevant,
                        "final_active_context_tokens": int(result.final_active_context_tokens),
                        "total_cpu_spill_tokens": int(result.total_cpu_spill_tokens),
                        "permanent_evicted_tokens": int(result.permanent_evicted_tokens),
                    }
                )
            print(
                f"[sample {index + 1:03d} x={fraction:g}] "
                f"stream={result.stream_prefill_s:.2f}s "
                f"active={result.final_active_context_tokens} "
                f"cpu={result.total_cpu_spill_tokens} "
                f"perm={result.permanent_evicted_tokens}",
                flush=True,
            )

    aggregate: dict[str, dict] = {}
    for fraction in fractions:
        group = [row for row in rows if float(row["cpu_store_fraction"]) == float(fraction)]
        aggregate[f"x{fraction:g}"] = {
            "mean_active_coverage": round(_mean(row["active_coverage"] for row in group), 6),
            "mean_cpu_coverage": round(_mean(row["cpu_coverage"] for row in group), 6),
            "mean_accessible_coverage": round(_mean(row["accessible_coverage"] for row in group), 6),
            "mean_cpu_coverage_of_evicted_relevant": round(
                _mean(row["cpu_coverage_of_evicted_relevant"] for row in group),
                6,
            ),
            "mean_total_cpu_spill_tokens": round(_mean(row["total_cpu_spill_tokens"] for row in group), 3),
            "mean_permanent_evicted_tokens": round(_mean(row["permanent_evicted_tokens"] for row in group), 3),
            "n_rows": len(group),
        }

    by_task: dict[str, dict] = {}
    for task_name in sorted({row["task"] for row in rows}):
        by_task[task_name] = {}
        for fraction in fractions:
            group = [row for row in rows if row["task"] == task_name and float(row["cpu_store_fraction"]) == float(fraction)]
            by_task[task_name][f"x{fraction:g}"] = {
                "mean_active_coverage": round(_mean(row["active_coverage"] for row in group), 6),
                "mean_cpu_coverage": round(_mean(row["cpu_coverage"] for row in group), 6),
                "mean_accessible_coverage": round(_mean(row["accessible_coverage"] for row in group), 6),
                "mean_cpu_coverage_of_evicted_relevant": round(
                    _mean(row["cpu_coverage_of_evicted_relevant"] for row in group),
                    6,
                ),
                "n_rows": len(group),
            }

    payload = {
        "schema_version": "phase8-two-tier-snapkv-spill-sweep-v1",
        "config": {
            "task": str(args.task),
            "num_samples": int(args.num_samples),
            "total_context_length": int(args.total_context_length),
            "gpu_cache_cap": int(args.gpu_cache_cap),
            "turn_headroom": int(args.turn_headroom),
            "chunk_size": int(args.chunk_size),
            "cpu_store_fractions": list(fractions),
            "dataset_seed_offset": int(args.dataset_seed_offset),
            "policy": "first SnapKV permanent drop to candidate pool, then SnapKV GPU/CPU split",
        },
        "aggregate": aggregate,
        "by_task": by_task,
        "stream_rows": stream_rows,
        "rows": rows,
        "elapsed_s": round(time.perf_counter() - start, 6),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fraction_label = "-".join(f"{value:g}" for value in fractions)
    path = RESULTS_DIR / (
        f"{args.task}_l{args.total_context_length}_cap{args.gpu_cache_cap}_h{args.turn_headroom}"
        f"_chunk{args.chunk_size}_n{args.num_samples}_x{fraction_label}.json"
    )
    write_json(path, payload)
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True), flush=True)
    print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

