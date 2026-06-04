#!/usr/bin/env python3
"""Find the minimum top-X qnorm spill fraction needed to cover Q2 needles."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

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
from phases.phase8_streaming_strict_cap.src.streaming import stream_context_qnorm_spill_sweep  # noqa: E402


RESULTS_DIR = PHASE_ROOT / "results" / "coverage_binary_search"


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return float(fmean(values)) if values else 0.0


def _split_specs(task: str):
    if task == "clean_suite":
        return CLEAN_SPLIT_SPECS
    spec = SPLIT_SPECS_BY_NAME.get(task)
    if spec is None:
        raise ValueError(f"Unsupported task: {task!r}.")
    return (spec,)


def _coverage_at_fraction(
    *,
    fraction: float,
    relevant_positions: Sequence[int],
    active_positions: set[int],
    qnorm_rank_by_position: dict[int, tuple[int, int]],
    count_active_as_covered: bool,
) -> float:
    if not relevant_positions:
        return 0.0
    covered = 0
    for position in relevant_positions:
        position_int = int(position)
        if count_active_as_covered and position_int in active_positions:
            covered += 1
            continue
        rank_pair = qnorm_rank_by_position.get(position_int)
        if rank_pair is None:
            continue
        rank, evicted_count = rank_pair
        keep_count = max(1, int(math.ceil(int(evicted_count) * float(fraction))))
        if int(rank) <= keep_count:
            covered += 1
    return float(covered / len(relevant_positions))


def _binary_search_fraction(
    *,
    relevant_positions: Sequence[int],
    active_positions: set[int],
    qnorm_rank_by_position: dict[int, tuple[int, int]],
    target_coverage: float,
    count_active_as_covered: bool,
    iterations: int,
) -> float | None:
    """Binary-search the smallest fraction reaching target coverage."""
    if _coverage_at_fraction(
        fraction=1.0,
        relevant_positions=relevant_positions,
        active_positions=active_positions,
        qnorm_rank_by_position=qnorm_rank_by_position,
        count_active_as_covered=count_active_as_covered,
    ) < float(target_coverage):
        return None

    lo = 0.0
    hi = 1.0
    for _ in range(int(iterations)):
        mid = (lo + hi) / 2.0
        coverage = _coverage_at_fraction(
            fraction=mid,
            relevant_positions=relevant_positions,
            active_positions=active_positions,
            qnorm_rank_by_position=qnorm_rank_by_position,
            count_active_as_covered=count_active_as_covered,
        )
        if coverage >= float(target_coverage):
            hi = mid
        else:
            lo = mid
    return hi


def _exact_threshold_from_ranks(
    *,
    relevant_positions: Sequence[int],
    active_positions: set[int],
    qnorm_rank_by_position: dict[int, tuple[int, int]],
    target_coverage: float,
    count_active_as_covered: bool,
) -> float | None:
    """Return the exact discrete threshold implied by qnorm event ranks."""
    if not relevant_positions:
        return None
    thresholds: list[float] = []
    active_credit = 0
    for position in relevant_positions:
        position_int = int(position)
        if count_active_as_covered and position_int in active_positions:
            active_credit += 1
            thresholds.append(0.0)
            continue
        rank_pair = qnorm_rank_by_position.get(position_int)
        if rank_pair is None:
            continue
        rank, evicted_count = rank_pair
        thresholds.append(float(int(rank) / int(evicted_count)))

    required_count = int(math.ceil(float(target_coverage) * len(relevant_positions)))
    if active_credit >= required_count:
        return 0.0
    if len(thresholds) < required_count:
        return None
    return sorted(thresholds)[required_count - 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="clean_suite")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--total-context-length", type=int, default=65_536)
    parser.add_argument("--gpu-cache-cap", type=int, default=32_768)
    parser.add_argument("--turn-headroom", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--target-coverages", nargs="+", type=float, default=[0.01, 0.25, 0.50, 1.0])
    parser.add_argument("--binary-iterations", type=int, default=16)
    parser.add_argument("--dataset-seed-offset", type=int, default=0)
    parser.add_argument(
        "--count-active-as-covered",
        action="store_true",
        help="Treat Q2 needles still in final active cache as covered at X=0.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    model = load_model()
    tokenizer = load_tokenizer()
    split_specs = _split_specs(args.task)
    target_coverages = tuple(sorted(dict.fromkeys(float(value) for value in args.target_coverages)))
    if not target_coverages or any(value <= 0.0 or value > 1.0 for value in target_coverages):
        raise ValueError("target-coverages must contain values in (0, 1].")

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
        sweep = stream_context_qnorm_spill_sweep(
            model=model,
            context_ids=split_views[0].q1_prepared.context_ids,
            total_context_length=int(args.total_context_length),
            chunk_size=int(args.chunk_size),
            gpu_cache_cap=int(args.gpu_cache_cap),
            turn_headroom=int(args.turn_headroom),
            spill_fractions=[1.0],
            sink_size=4,
            recency_window=128,
            obs_window_size=128,
            pooling="max",
        )
        active_positions = set()
        if sweep.final_active_context_tokens:
            # The streaming helper returns active positions only indirectly through ranks.
            # Positions never evicted do not appear in qnorm_rank_by_position; infer them
            # as context positions not in the evicted-rank map.
            active_positions = set(range(int(args.total_context_length))) - set(sweep.qnorm_rank_by_position)
        stream_rows.append(
            {
                "index": index,
                "stream_prefill_s": round(sweep.stream_prefill_s, 6),
                "final_active_context_tokens": sweep.final_active_context_tokens,
                "peak_active_context_tokens": sweep.peak_active_context_tokens,
                "eviction_events": sweep.eviction_events,
                "total_evicted_tokens": sweep.total_evicted_tokens,
            }
        )

        for split in split_views:
            q2_relevant = tuple(relevant_positions_for_spans(split.q2_prepared, split.q2_span_names))
            active_relevant = tuple(position for position in q2_relevant if int(position) in active_positions)
            evicted_relevant = tuple(position for position in q2_relevant if int(position) in sweep.qnorm_rank_by_position)
            missing_relevant = tuple(
                position
                for position in q2_relevant
                if int(position) not in active_positions and int(position) not in sweep.qnorm_rank_by_position
            )
            for target in target_coverages:
                binary_threshold = _binary_search_fraction(
                    relevant_positions=q2_relevant,
                    active_positions=active_positions,
                    qnorm_rank_by_position=sweep.qnorm_rank_by_position,
                    target_coverage=target,
                    count_active_as_covered=bool(args.count_active_as_covered),
                    iterations=int(args.binary_iterations),
                )
                exact_threshold = _exact_threshold_from_ranks(
                    relevant_positions=q2_relevant,
                    active_positions=active_positions,
                    qnorm_rank_by_position=sweep.qnorm_rank_by_position,
                    target_coverage=target,
                    count_active_as_covered=bool(args.count_active_as_covered),
                )
                evicted_only_binary_threshold = _binary_search_fraction(
                    relevant_positions=evicted_relevant,
                    active_positions=set(),
                    qnorm_rank_by_position=sweep.qnorm_rank_by_position,
                    target_coverage=target,
                    count_active_as_covered=False,
                    iterations=int(args.binary_iterations),
                )
                evicted_only_exact_threshold = _exact_threshold_from_ranks(
                    relevant_positions=evicted_relevant,
                    active_positions=set(),
                    qnorm_rank_by_position=sweep.qnorm_rank_by_position,
                    target_coverage=target,
                    count_active_as_covered=False,
                )
                rows.append(
                    {
                        "index": index,
                        "task": split.split_spec.name,
                        "target_coverage": float(target),
                        "binary_threshold_fraction": None if binary_threshold is None else round(float(binary_threshold), 8),
                        "exact_threshold_fraction": None if exact_threshold is None else round(float(exact_threshold), 8),
                        "binary_threshold_percent": None if binary_threshold is None else round(float(binary_threshold) * 100.0, 4),
                        "exact_threshold_percent": None if exact_threshold is None else round(float(exact_threshold) * 100.0, 4),
                        "evicted_only_binary_threshold_fraction": None
                        if evicted_only_binary_threshold is None
                        else round(float(evicted_only_binary_threshold), 8),
                        "evicted_only_exact_threshold_fraction": None
                        if evicted_only_exact_threshold is None
                        else round(float(evicted_only_exact_threshold), 8),
                        "evicted_only_binary_threshold_percent": None
                        if evicted_only_binary_threshold is None
                        else round(float(evicted_only_binary_threshold) * 100.0, 4),
                        "evicted_only_exact_threshold_percent": None
                        if evicted_only_exact_threshold is None
                        else round(float(evicted_only_exact_threshold) * 100.0, 4),
                        "coverage_at_1pct": round(
                            _coverage_at_fraction(
                                fraction=0.01,
                                relevant_positions=q2_relevant,
                                active_positions=active_positions,
                                qnorm_rank_by_position=sweep.qnorm_rank_by_position,
                                count_active_as_covered=bool(args.count_active_as_covered),
                            ),
                            6,
                        ),
                        "coverage_at_5pct": round(
                            _coverage_at_fraction(
                                fraction=0.05,
                                relevant_positions=q2_relevant,
                                active_positions=active_positions,
                                qnorm_rank_by_position=sweep.qnorm_rank_by_position,
                                count_active_as_covered=bool(args.count_active_as_covered),
                            ),
                            6,
                        ),
                        "coverage_at_10pct": round(
                            _coverage_at_fraction(
                                fraction=0.10,
                                relevant_positions=q2_relevant,
                                active_positions=active_positions,
                                qnorm_rank_by_position=sweep.qnorm_rank_by_position,
                                count_active_as_covered=bool(args.count_active_as_covered),
                            ),
                            6,
                        ),
                        "coverage_at_25pct": round(
                            _coverage_at_fraction(
                                fraction=0.25,
                                relevant_positions=q2_relevant,
                                active_positions=active_positions,
                                qnorm_rank_by_position=sweep.qnorm_rank_by_position,
                                count_active_as_covered=bool(args.count_active_as_covered),
                            ),
                            6,
                        ),
                        "coverage_at_50pct": round(
                            _coverage_at_fraction(
                                fraction=0.50,
                                relevant_positions=q2_relevant,
                                active_positions=active_positions,
                                qnorm_rank_by_position=sweep.qnorm_rank_by_position,
                                count_active_as_covered=bool(args.count_active_as_covered),
                            ),
                            6,
                        ),
                        "coverage_at_100pct": round(
                            _coverage_at_fraction(
                                fraction=1.00,
                                relevant_positions=q2_relevant,
                                active_positions=active_positions,
                                qnorm_rank_by_position=sweep.qnorm_rank_by_position,
                                count_active_as_covered=bool(args.count_active_as_covered),
                            ),
                            6,
                        ),
                        "evicted_only_coverage_at_1pct": round(
                            _coverage_at_fraction(
                                fraction=0.01,
                                relevant_positions=evicted_relevant,
                                active_positions=set(),
                                qnorm_rank_by_position=sweep.qnorm_rank_by_position,
                                count_active_as_covered=False,
                            ),
                            6,
                        ),
                        "evicted_only_coverage_at_5pct": round(
                            _coverage_at_fraction(
                                fraction=0.05,
                                relevant_positions=evicted_relevant,
                                active_positions=set(),
                                qnorm_rank_by_position=sweep.qnorm_rank_by_position,
                                count_active_as_covered=False,
                            ),
                            6,
                        ),
                        "evicted_only_coverage_at_10pct": round(
                            _coverage_at_fraction(
                                fraction=0.10,
                                relevant_positions=evicted_relevant,
                                active_positions=set(),
                                qnorm_rank_by_position=sweep.qnorm_rank_by_position,
                                count_active_as_covered=False,
                            ),
                            6,
                        ),
                        "evicted_only_coverage_at_25pct": round(
                            _coverage_at_fraction(
                                fraction=0.25,
                                relevant_positions=evicted_relevant,
                                active_positions=set(),
                                qnorm_rank_by_position=sweep.qnorm_rank_by_position,
                                count_active_as_covered=False,
                            ),
                            6,
                        ),
                        "evicted_only_coverage_at_50pct": round(
                            _coverage_at_fraction(
                                fraction=0.50,
                                relevant_positions=evicted_relevant,
                                active_positions=set(),
                                qnorm_rank_by_position=sweep.qnorm_rank_by_position,
                                count_active_as_covered=False,
                            ),
                            6,
                        ),
                        "evicted_only_coverage_at_100pct": round(
                            _coverage_at_fraction(
                                fraction=1.00,
                                relevant_positions=evicted_relevant,
                                active_positions=set(),
                                qnorm_rank_by_position=sweep.qnorm_rank_by_position,
                                count_active_as_covered=False,
                            ),
                            6,
                        ),
                        "q2_relevant_tokens": len(q2_relevant),
                        "active_relevant_tokens": len(active_relevant),
                        "evicted_relevant_tokens": len(evicted_relevant),
                        "missing_relevant_tokens": len(missing_relevant),
                        "active_relevant_positions": list(active_relevant),
                        "evicted_relevant_rank_pairs": {
                            str(position): list(sweep.qnorm_rank_by_position[int(position)])
                            for position in evicted_relevant
                        },
                    }
                )
        print(
            f"[sample {index + 1:03d}] stream={sweep.stream_prefill_s:.2f}s "
            f"active={sweep.final_active_context_tokens} evicted={sweep.total_evicted_tokens}",
            flush=True,
        )

    aggregate: dict[str, dict] = {}
    for target in target_coverages:
        group = [row for row in rows if float(row["target_coverage"]) == float(target)]
        finite_exact = [row["exact_threshold_percent"] for row in group if row["exact_threshold_percent"] is not None]
        finite_evicted_exact = [
            row["evicted_only_exact_threshold_percent"]
            for row in group
            if row["evicted_only_exact_threshold_percent"] is not None
        ]
        aggregate[f"target_{target:g}"] = {
            "mean_exact_threshold_percent": None if not finite_exact else round(_mean(finite_exact), 4),
            "max_exact_threshold_percent": None if not finite_exact else round(max(finite_exact), 4),
            "mean_evicted_only_exact_threshold_percent": None
            if not finite_evicted_exact
            else round(_mean(finite_evicted_exact), 4),
            "max_evicted_only_exact_threshold_percent": None
            if not finite_evicted_exact
            else round(max(finite_evicted_exact), 4),
            "solvable_rows": len(finite_exact),
            "evicted_only_solvable_rows": len(finite_evicted_exact),
            "n_rows": len(group),
            "mean_coverage_at_10pct": round(_mean(row["coverage_at_10pct"] for row in group), 6),
            "mean_coverage_at_25pct": round(_mean(row["coverage_at_25pct"] for row in group), 6),
            "mean_coverage_at_50pct": round(_mean(row["coverage_at_50pct"] for row in group), 6),
            "mean_evicted_only_coverage_at_10pct": round(
                _mean(row["evicted_only_coverage_at_10pct"] for row in group),
                6,
            ),
            "mean_evicted_only_coverage_at_25pct": round(
                _mean(row["evicted_only_coverage_at_25pct"] for row in group),
                6,
            ),
            "mean_evicted_only_coverage_at_50pct": round(
                _mean(row["evicted_only_coverage_at_50pct"] for row in group),
                6,
            ),
        }

    payload = {
        "schema_version": "phase8-qnorm-spill-binary-threshold-v1",
        "config": {
            "task": args.task,
            "num_samples": int(args.num_samples),
            "total_context_length": int(args.total_context_length),
            "gpu_cache_cap": int(args.gpu_cache_cap),
            "turn_headroom": int(args.turn_headroom),
            "chunk_size": int(args.chunk_size),
            "target_coverages": list(target_coverages),
            "binary_iterations": int(args.binary_iterations),
            "dataset_seed_offset": int(args.dataset_seed_offset),
            "count_active_as_covered": bool(args.count_active_as_covered),
            "eviction_mode": "fill_cap",
        },
        "aggregate": aggregate,
        "stream_rows": stream_rows,
        "rows": rows,
        "elapsed_s": round(time.perf_counter() - start, 6),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target_label = "-".join(f"{value:g}" for value in target_coverages)
    active_label = "_activecovered" if args.count_active_as_covered else ""
    path = RESULTS_DIR / (
        f"{args.task}_l{args.total_context_length}_cap{args.gpu_cache_cap}_h{args.turn_headroom}"
        f"_chunk{args.chunk_size}_n{args.num_samples}_targets{target_label}{active_label}.json"
    )
    write_json(path, payload)
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True), flush=True)
    print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
