#!/usr/bin/env python3
"""Run a quality-only low-bit-rowstore precision-promotion smoke.

This is an exploratory Phase 10 diagnostic. It stores selected context rows as
integer low-bit codes with per-row scales, materializes them back to model dtype
at the attention boundary, and tests whether Q2-conditioned high-precision row
promotion improves answer quality at a matched active byte estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

import torch

PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_ROOT.parents[1]
for root in (PHASE_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache, slice_kv  # noqa: E402
from phases.phase2_kv_cache.src.runtime import MODEL_DIR as DEFAULT_MODEL_DIR  # noqa: E402
from phases.phase2_kv_cache.src.runtime import load_model, load_tokenizer  # noqa: E402
from phases.phase3_eviction.src.runtime import build_position_tracked_cache  # noqa: E402
from phases.phase6_repair.src.protocol import (  # noqa: E402
    build_base_example,
    build_split_prepared_from_base_example,
    build_turn_n_keep_plan,
    compute_q2_exact_query_rows,
    generate_turn,
    relevant_position_groups_for_spans,
    relevant_positions_for_spans,
)
from phases.phase6_repair.src.runner import (  # noqa: E402
    TASK_ALIASES,
    _overlap_fraction,
    _run_condition,
)
from phases.phase6_repair.src.selectors import (  # noqa: E402
    pack_anchor_bursts,
    rank_positions,
    score_evicted_positions,
    select_idlekv_positions,
    select_oldest_positions,
    select_oracle_positions,
    select_random_positions,
)
from phases.phase10_expansion.src.precision_promotion import (  # noqa: E402
    PrecisionBudget,
    evaluate_precision_promotion_rows,
    lowbit_row_store_bytes,
    materialize_lowbit_cache,
    mixed_precision_kv_bytes,
    quantize_position_rows,
)


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return float(fmean(values)) if values else 0.0


def _slice_dense(cache: PositionTrackedCache, start: int, stop: int) -> PositionTrackedCache:
    fragment = slice_kv(cache, list(range(int(start), int(stop))))
    if not isinstance(fragment, PositionTrackedCache):
        raise RuntimeError("slice_kv did not preserve position metadata.")
    return fragment


def _score_promoted_cache(
    *,
    model,
    tokenizer,
    prepared,
    high_cache: PositionTrackedCache,
    lowbit_store,
    positions: Iterable[int],
) -> tuple[str, float, float]:
    promoted = materialize_lowbit_cache(high_cache, lowbit_store, promoted_positions=positions)
    return _run_condition(model=model, tokenizer=tokenizer, prepared=prepared, cache=promoted)


def _select_static_positions(
    *,
    context_positions: tuple[int, ...],
    turn_n_scores: dict[int, float],
    k: int,
    left: int,
    right: int,
) -> list[int]:
    ranked = rank_positions(context_positions, primary_scores=turn_n_scores)
    return pack_anchor_bursts(
        anchor_positions=ranked,
        available_positions=context_positions,
        k=int(k),
        left=int(left),
        right=int(right),
        backfill_positions=ranked,
    )


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["nbits"]), int(row["k"]))].append(row)

    aggregate_rows: list[dict[str, Any]] = []
    score_keys = (
        "full_fp16",
        "lowbit_all",
        "static_mixed",
        "random_precision",
        "oldest_precision",
        "idlekv_precision",
        "gold_precision",
        "active_bytes",
        "side_buffer_bytes",
    )
    for (nbits, k), group in sorted(grouped.items()):
        out: dict[str, Any] = {
            "nbits": int(nbits),
            "k": int(k),
            "n_rows": len(group),
            "task": group[0]["suite_task"],
            "row_store_backend": group[0]["row_store_backend"],
            "real_quantized_cache": False,
        }
        for key in score_keys:
            out[key] = round(_mean(float(row[key]) for row in group), 6)
        out["lowbit_store_bytes"] = round(_mean(float(row["lowbit_store_bytes"]) for row in group), 6)
        out["idlekv_overlap_fraction"] = round(_mean(float(row["idlekv_overlap_fraction"]) for row in group), 6)
        out["gold_overlap_fraction"] = round(_mean(float(row["gold_overlap_fraction"]) for row in group), 6)
        aggregate_rows.append(out)
    return aggregate_rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = Path(args.model_dir).expanduser()
    model = load_model(model_dir)
    tokenizer = load_tokenizer(model_dir)
    split_specs = TASK_ALIASES[str(args.task)]
    raw_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index in range(int(args.num_samples)):
        base_example = build_base_example(
            split_spec=split_specs[0],
            index=index,
            context_length=int(args.context_length),
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
        full_cache = build_position_tracked_cache(model, split_views[0].q1_prepared.context_ids)

        for split in split_views:
            context_len = int(split.q1_prepared.context_ids.shape[1])
            context_positions = tuple(range(context_len))
            q1_turn = generate_turn(model, tokenizer, split.q1_prepared, full_cache)
            q1_tail_positions = tuple(range(context_len, len(q1_turn.cache)))
            full_output, full_score, full_generation_s = _run_condition(
                model=model,
                tokenizer=tokenizer,
                prepared=split.q2_prepared,
                cache=q1_turn.cache,
            )
            keep_plan = build_turn_n_keep_plan(
                post_q1_cache=q1_turn.cache,
                q1_answer_ids=q1_turn.token_ids,
                context_len=context_len,
                sink_size=int(args.sink_size),
                recency_window=int(args.recency_window),
                pooling=str(args.pooling),
                initial_compressor="snapkv",
            )
            q2_relevant_positions = relevant_positions_for_spans(split.q2_prepared, split.q2_span_names)
            q2_relevant_groups = relevant_position_groups_for_spans(split.q2_prepared, split.q2_span_names)

            for nbits in args.nbits:
                budget = PrecisionBudget(low_precision_bits=int(nbits), high_precision_bits=16)
                lowbit_store = quantize_position_rows(
                    q1_turn.cache,
                    quantize_positions=context_positions,
                    nbits=int(nbits),
                    backend=str(args.quantizer),
                    axis_key=int(args.axis_key),
                    axis_value=int(args.axis_value),
                    group_size=int(args.q_group_size),
                    hqq_optimize=bool(args.hqq_optimize),
                )
                low_cache = materialize_lowbit_cache(q1_turn.cache, lowbit_store)
                lowbit_store_bytes = lowbit_row_store_bytes(lowbit_store)
                low_output, low_score, low_generation_s = _run_condition(
                    model=model,
                    tokenizer=tokenizer,
                    prepared=split.q2_prepared,
                    cache=low_cache,
                )
                q2_rows = compute_q2_exact_query_rows(
                    model,
                    active_cache=low_cache,
                    question_ids=split.q2_prepared.question_ids,
                )
                context_cache = _slice_dense(low_cache, 0, context_len)
                tail_cache = _slice_dense(low_cache, context_len, len(low_cache)) if q1_tail_positions else None
                q2_scores = score_evicted_positions(
                    query_rows=q2_rows,
                    evicted_cache=context_cache,
                    active_cache=tail_cache,
                    pooling=str(args.pooling),
                )

                for k in args.k:
                    k_int = int(k)
                    static_positions = _select_static_positions(
                        context_positions=context_positions,
                        turn_n_scores=keep_plan.importance_scores,
                        k=k_int,
                        left=int(args.burst_left),
                        right=int(args.burst_right),
                    )
                    random_positions = select_random_positions(
                        evicted_positions=context_positions,
                        k=k_int,
                        left=int(args.burst_left),
                        right=int(args.burst_right),
                        seed=(index + 1) * 1000 + k_int + int(nbits),
                    )
                    oldest_positions = select_oldest_positions(
                        evicted_positions=context_positions,
                        k=k_int,
                        left=int(args.burst_left),
                        right=int(args.burst_right),
                    )
                    idlekv_positions = select_idlekv_positions(
                        evicted_positions=context_positions,
                        q2_scores=q2_scores,
                        turn_n_scores=keep_plan.importance_scores,
                        k=k_int,
                        left=int(args.burst_left),
                        right=int(args.burst_right),
                    )
                    gold_positions = select_oracle_positions(
                        evicted_positions=context_positions,
                        relevant_positions=q2_relevant_positions,
                        relevant_position_groups=q2_relevant_groups,
                        q2_scores=q2_scores,
                        turn_n_scores=keep_plan.importance_scores,
                        k=k_int,
                        left=int(args.burst_left),
                        right=int(args.burst_right),
                    )

                    static_output, static_score, static_generation_s = _score_promoted_cache(
                        model=model,
                        tokenizer=tokenizer,
                        prepared=split.q2_prepared,
                        high_cache=q1_turn.cache,
                        lowbit_store=lowbit_store,
                        positions=static_positions,
                    )
                    random_output, random_score, random_generation_s = _score_promoted_cache(
                        model=model,
                        tokenizer=tokenizer,
                        prepared=split.q2_prepared,
                        high_cache=q1_turn.cache,
                        lowbit_store=lowbit_store,
                        positions=random_positions,
                    )
                    oldest_output, oldest_score, oldest_generation_s = _score_promoted_cache(
                        model=model,
                        tokenizer=tokenizer,
                        prepared=split.q2_prepared,
                        high_cache=q1_turn.cache,
                        lowbit_store=lowbit_store,
                        positions=oldest_positions,
                    )
                    idlekv_output, idlekv_score, idlekv_generation_s = _score_promoted_cache(
                        model=model,
                        tokenizer=tokenizer,
                        prepared=split.q2_prepared,
                        high_cache=q1_turn.cache,
                        lowbit_store=lowbit_store,
                        positions=idlekv_positions,
                    )
                    gold_output, gold_score, gold_generation_s = _score_promoted_cache(
                        model=model,
                        tokenizer=tokenizer,
                        prepared=split.q2_prepared,
                        high_cache=q1_turn.cache,
                        lowbit_store=lowbit_store,
                        positions=gold_positions,
                    )
                    active_bytes = mixed_precision_kv_bytes(
                        low_cache,
                        budget=budget,
                        low_precision_positions=context_positions,
                        promoted_positions=idlekv_positions,
                    )
                    with_side = mixed_precision_kv_bytes(
                        low_cache,
                        budget=budget,
                        low_precision_positions=context_positions,
                        promoted_positions=idlekv_positions,
                        include_high_precision_side_buffer=True,
                    )
                    row = {
                        "example_id": f"{split.split_spec.name}:ex{index + 1:03d}",
                        "task": split.split_spec.name,
                        "suite_task": str(args.task),
                        "index": int(index),
                        "nbits": int(nbits),
                        "k": k_int,
                        "context_length": context_len,
                        "row_store_backend": str(args.quantizer),
                        "axis_key": int(args.axis_key),
                        "axis_value": int(args.axis_value),
                        "q_group_size": int(args.q_group_size),
                        "hqq_optimize": bool(args.hqq_optimize),
                        "full_fp16": round(full_score, 6),
                        "lowbit_all": round(low_score, 6),
                        "static_mixed": round(static_score, 6),
                        "random_precision": round(random_score, 6),
                        "oldest_precision": round(oldest_score, 6),
                        "idlekv_precision": round(idlekv_score, 6),
                        "gold_precision": round(gold_score, 6),
                        "active_bytes": round(active_bytes, 6),
                        "side_buffer_bytes": round(with_side - active_bytes, 6),
                        "lowbit_store_bytes": round(lowbit_store_bytes, 6),
                        "full_output": full_output,
                        "lowbit_output": low_output,
                        "static_output": static_output,
                        "random_output": random_output,
                        "oldest_output": oldest_output,
                        "idlekv_output": idlekv_output,
                        "gold_output": gold_output,
                        "full_generation_s": round(full_generation_s, 6),
                        "lowbit_generation_s": round(low_generation_s, 6),
                        "static_generation_s": round(static_generation_s, 6),
                        "random_generation_s": round(random_generation_s, 6),
                        "oldest_generation_s": round(oldest_generation_s, 6),
                        "idlekv_generation_s": round(idlekv_generation_s, 6),
                        "gold_generation_s": round(gold_generation_s, 6),
                        "static_positions": static_positions,
                        "random_positions": random_positions,
                        "oldest_positions": oldest_positions,
                        "idlekv_positions": idlekv_positions,
                        "gold_positions": gold_positions,
                        "idlekv_overlap_fraction": round(_overlap_fraction(idlekv_positions, q2_relevant_positions), 6),
                        "gold_overlap_fraction": round(_overlap_fraction(gold_positions, q2_relevant_positions), 6),
                        "real_quantized_cache": False,
                    }
                    raw_rows.append(row)
                    print(
                        f"[{row['example_id']}] bits={nbits} k={k_int} "
                        f"F={full_score:.3f} L={low_score:.3f} "
                        f"S={static_score:.3f} R={random_score:.3f} "
                        f"O={oldest_score:.3f} I={idlekv_score:.3f} G={gold_score:.3f}",
                        flush=True,
                    )

    aggregate_rows = _aggregate_rows(raw_rows)
    output_csv = Path(args.output_csv)
    raw_json = Path(args.raw_json)
    _write_csv(aggregate_rows, output_csv)
    raw_json.parent.mkdir(parents=True, exist_ok=True)
    json_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    json_config["model_dir"] = str(model_dir)
    raw_json.write_text(
        json.dumps(
            {
                "schema_version": "phase10-precision-promotion-lowbit-rowstore-v1",
                "config": json_config,
                "rows": raw_rows,
                "aggregate_rows": aggregate_rows,
                "recommendations": evaluate_precision_promotion_rows(aggregate_rows),
                "elapsed_s": round(time.perf_counter() - started, 6),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"aggregate_rows": aggregate_rows, "output_csv": str(output_csv), "raw_json": str(raw_json)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASK_ALIASES), default="clean_suite")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=32_768)
    parser.add_argument("--dataset-seed-offset", type=int, default=0)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--k", nargs="+", type=int, default=[48, 96])
    parser.add_argument("--nbits", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--quantizer", choices=("symmetric_row", "hqq"), default="hqq")
    parser.add_argument("--axis-key", type=int, default=0)
    parser.add_argument("--axis-value", type=int, default=0)
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument("--hqq-optimize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sink-size", type=int, default=4)
    parser.add_argument("--recency-window", type=int, default=128)
    parser.add_argument("--burst-left", type=int, default=2)
    parser.add_argument("--burst-right", type=int, default=20)
    parser.add_argument("--pooling", choices=("max", "mean"), default="max")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PHASE_ROOT / "results" / "precision_promotion_smoke_n1.csv",
    )
    parser.add_argument(
        "--raw-json",
        type=Path,
        default=PHASE_ROOT / "results" / "precision_promotion_smoke_n1_raw.json",
    )
    return parser.parse_args()


def main() -> int:
    result = run_smoke(parse_args())
    print(result["output_csv"])
    print(result["raw_json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
