"""Phase 8 runner: strict-cap streaming with bounded CPU-spill repair."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import shutil
import time
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence

import torch

from phases.phase1_degradation.phase1.evaluation import sample_score
from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache
from phases.phase2_kv_cache.src.runtime import load_model, load_tokenizer
from phases.phase3_eviction.src.runtime import write_json
from phases.phase6_repair.src.protocol import (
    CLEAN_SPLIT_SPECS,
    SPLIT_SPECS_BY_NAME,
    SplitTaskSpec,
    build_base_example,
    build_split_prepared_from_base_example,
    compute_q2_exact_query_rows,
    generate_turn,
    relevant_position_groups_for_spans,
    relevant_positions_for_spans,
)
from phases.phase6_repair.src.selectors import rank_positions, select_oracle_positions

from .streaming import (
    SpillBuffer,
    assert_strict_cap,
    choose_lowest_score_positions,
    score_cache_positions,
    simulate_streaming_geometry,
    slice_cache_by_positions,
    stream_context_with_spill,
    swap_restore_positions,
)


PHASE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PHASE_ROOT / "results"
SCHEMA_VERSION = "phase8-strict-cap-stream-v1"

ALLOWED_CONDITIONS = (
    "B_stream",
    "IdleKV_stream",
    "RandomRepair_stream",
    "OracleSpill_stream",
    "RandomSpill_stream",
)

STAGE_DEFAULTS: dict[str, dict[str, Any]] = {
    "smoke": {
        "conditions": ("B_stream", "IdleKV_stream", "OracleSpill_stream"),
        "num_samples": 2,
        "b_values": (2048,),
    },
    "calibration": {
        "conditions": (
            "B_stream",
            "IdleKV_stream",
            "RandomRepair_stream",
            "OracleSpill_stream",
            "RandomSpill_stream",
        ),
        "num_samples": 8,
        "b_values": (512, 1024, 2048, 4096, 8192),
    },
    "full": {
        "conditions": (
            "B_stream",
            "IdleKV_stream",
            "RandomRepair_stream",
            "OracleSpill_stream",
            "RandomSpill_stream",
        ),
        "num_samples": 100,
        "b_values": (512, 1024, 2048, 4096, 8192),
    },
}

TASK_ALIASES: dict[str, tuple[SplitTaskSpec, ...]] = {
    "clean_suite": CLEAN_SPLIT_SPECS,
}


@dataclass(frozen=True)
class Phase8Config:
    """Frozen configuration for one Phase 8 run."""

    stage: str
    task: str
    split_specs: tuple[SplitTaskSpec, ...]
    num_samples: int
    total_context_length: int = 327_680
    gpu_cache_cap: int = 32_768
    turn_headroom: int = 512
    chunk_size: int = 2048
    keep_fraction: float = 0.10
    spill_fraction: float = 0.10
    spill_hard_cap: int = 32_768
    b_values: tuple[int, ...] = (512, 1024, 2048, 4096, 8192)
    conditions: tuple[str, ...] = (
        "B_stream",
        "IdleKV_stream",
        "RandomRepair_stream",
        "OracleSpill_stream",
    )
    sink_size: int = 4
    recency_window: int = 128
    obs_window_size: int = 128
    pooling: str = "max"
    dataset_seed_offset: int = 0
    seed: int = 8000
    max_runtime_s: int = 10_800


def ensure_results_dirs(stage: str) -> Path:
    stage_dir = RESULTS_DIR / str(stage)
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def _condition_label(conditions: Iterable[str]) -> str:
    parts = [re.sub(r"[^a-z0-9]+", "", str(condition).lower()) for condition in conditions]
    return "-".join(part for part in parts if part)


def _normalize_stage(stage: str) -> str:
    normalized = str(stage).strip().lower()
    if normalized not in STAGE_DEFAULTS:
        raise ValueError(f"Unsupported stage: {stage!r}.")
    return normalized


def _normalize_task(task: str) -> tuple[SplitTaskSpec, ...]:
    normalized = str(task).strip()
    if normalized in TASK_ALIASES:
        return TASK_ALIASES[normalized]
    split_spec = SPLIT_SPECS_BY_NAME.get(normalized)
    if split_spec is None:
        raise ValueError(f"Unsupported Phase 8 task: {task!r}.")
    return (split_spec,)


def build_config(
    *,
    stage: str,
    task: str = "clean_suite",
    num_samples: int | None = None,
    total_context_length: int = 327_680,
    gpu_cache_cap: int = 32_768,
    turn_headroom: int = 512,
    chunk_size: int = 2048,
    keep_fraction: float = 0.10,
    spill_fraction: float = 0.10,
    spill_hard_cap: int = 32_768,
    b_values: Iterable[int] | None = None,
    conditions: Iterable[str] | None = None,
    dataset_seed_offset: int = 0,
    seed: int = 8000,
    max_runtime_s: int = 10_800,
) -> Phase8Config:
    """Construct a Phase 8 run config with stage defaults unless overridden."""
    normalized_stage = _normalize_stage(stage)
    split_specs = _normalize_task(task)
    defaults = STAGE_DEFAULTS[normalized_stage]
    resolved_num_samples = int(num_samples or defaults["num_samples"])
    resolved_b = tuple(dict.fromkeys(int(value) for value in (b_values or defaults["b_values"])))
    resolved_conditions = tuple(dict.fromkeys(str(value) for value in (conditions or defaults["conditions"])))
    invalid_conditions = tuple(value for value in resolved_conditions if value not in ALLOWED_CONDITIONS)
    if invalid_conditions:
        raise ValueError(f"Unsupported Phase 8 condition(s): {invalid_conditions}.")
    if resolved_num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if not resolved_b or any(value <= 0 for value in resolved_b):
        raise ValueError("b_values must contain at least one positive integer.")
    if int(total_context_length) <= 0 or int(gpu_cache_cap) <= 0 or int(chunk_size) <= 0:
        raise ValueError("context length, cap, and chunk size must be positive.")
    if int(turn_headroom) < 0 or int(turn_headroom) >= int(gpu_cache_cap):
        raise ValueError("turn_headroom must be non-negative and smaller than gpu_cache_cap.")
    if not 0.0 < float(keep_fraction) <= 1.0:
        raise ValueError("keep_fraction must lie in (0, 1].")
    if not 0.0 < float(spill_fraction) <= 1.0:
        raise ValueError("spill_fraction must lie in (0, 1].")
    if int(spill_hard_cap) <= 0:
        raise ValueError("spill_hard_cap must be positive.")
    if int(max_runtime_s) <= 0:
        raise ValueError("max_runtime_s must be positive.")

    return Phase8Config(
        stage=normalized_stage,
        task=str(task).strip(),
        split_specs=tuple(split_specs),
        num_samples=resolved_num_samples,
        total_context_length=int(total_context_length),
        gpu_cache_cap=int(gpu_cache_cap),
        turn_headroom=int(turn_headroom),
        chunk_size=int(chunk_size),
        keep_fraction=float(keep_fraction),
        spill_fraction=float(spill_fraction),
        spill_hard_cap=int(spill_hard_cap),
        b_values=resolved_b,
        conditions=resolved_conditions,
        dataset_seed_offset=int(dataset_seed_offset),
        seed=int(seed),
        max_runtime_s=int(max_runtime_s),
    )


def _score_prediction(prediction: str, outputs: Sequence[str]) -> float:
    return float(sample_score(prediction, list(outputs)))


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return float(fmean(values)) if values else 0.0


def _pct(values: Iterable[bool]) -> float:
    values = [bool(value) for value in values]
    return float(sum(values) / len(values)) if values else 0.0


def _overlap_fraction(selected_positions: Iterable[int], relevant_positions: Iterable[int]) -> float:
    selected = {int(position) for position in selected_positions}
    relevant = {int(position) for position in relevant_positions}
    if not relevant:
        return 0.0
    return float(len(selected & relevant) / len(relevant))


def _run_condition(model, tokenizer, prepared, cache: PositionTrackedCache) -> tuple[str, float, float]:
    generation_start = time.perf_counter()
    generated = generate_turn(model, tokenizer, prepared, cache)
    generation_s = time.perf_counter() - generation_start
    score = _score_prediction(generated.text, list(prepared.example.outputs))
    return generated.text, score, generation_s


def _active_context_cache(cache: PositionTrackedCache, *, context_length: int) -> PositionTrackedCache | None:
    positions = [int(position) for position in cache.positions if int(position) < int(context_length)]
    return slice_cache_by_positions(cache, positions)


def _positions_by_random(positions: Sequence[int], *, k: int, seed: int) -> list[int]:
    rng = __import__("random").Random(int(seed))
    selected = [int(position) for position in positions]
    rng.shuffle(selected)
    return selected[: max(0, int(k))]


def _repair_cache(
    *,
    active_cache: PositionTrackedCache,
    spill: SpillBuffer,
    restore_positions: Sequence[int],
    query_rows: torch.Tensor,
    context_length: int,
    gpu_cache_cap: int,
    pooling: str,
) -> tuple[PositionTrackedCache, list[int], list[int]]:
    """Swap restore positions into active cache and evict equally many low-Q2 active context positions."""
    if spill.cache is None or not restore_positions:
        return active_cache, [], []
    active_context = _active_context_cache(active_cache, context_length=context_length)
    if active_context is None or len(active_context) == 0:
        return active_cache, [], []

    restore = [int(position) for position in restore_positions if int(position) in set(spill.cache.positions)]
    restore = restore[: min(len(restore), len(active_context))]
    if not restore:
        return active_cache, [], []

    active_scores = score_cache_positions(
        query_rows=query_rows,
        target_cache=active_context,
        competitor_cache=None,
        pooling=pooling,
    )
    drop = choose_lowest_score_positions(
        positions=active_context.positions,
        scores=active_scores,
        k=len(restore),
    )
    repaired = swap_restore_positions(
        active_cache=active_cache,
        spill_cache=spill.cache,
        restore_positions=restore,
        drop_positions=drop,
        cap=int(gpu_cache_cap),
    )
    return repaired, restore, drop


def _select_oracle_spill_positions(
    *,
    spill_positions: Sequence[int],
    q2_relevant_positions: Sequence[int],
    q2_relevant_groups: Sequence[Sequence[int]],
    q2_scores: dict[int, float],
    k: int,
) -> list[int]:
    return select_oracle_positions(
        evicted_positions=list(spill_positions),
        relevant_positions=list(q2_relevant_positions),
        relevant_position_groups=q2_relevant_groups,
        q2_scores=q2_scores,
        turn_n_scores={},
        k=int(k),
        left=0,
        right=0,
    )


def _run_one_split(
    *,
    model,
    tokenizer,
    config: Phase8Config,
    split,
    stream_result,
    index: int,
) -> list[dict[str, Any]]:
    example_start = time.perf_counter()
    q1_start = time.perf_counter()
    q1_turn = generate_turn(model, tokenizer, split.q1_prepared, stream_result.active_cache)
    q1_generation_s = time.perf_counter() - q1_start
    q1_score = _score_prediction(q1_turn.text, list(split.q1_prepared.example.outputs))
    assert_strict_cap(q1_turn.cache, cap=config.gpu_cache_cap, label="post-Q1 cache")

    condition_b_output, condition_b_score, condition_b_generation_s = _run_condition(
        model,
        tokenizer,
        split.q2_prepared,
        q1_turn.cache,
    )

    q2_query_start = time.perf_counter()
    q2_query_rows = compute_q2_exact_query_rows(
        model,
        active_cache=q1_turn.cache,
        question_ids=split.q2_prepared.question_ids,
    )
    q2_query_s = time.perf_counter() - q2_query_start

    q2_relevant_positions = relevant_positions_for_spans(split.q2_prepared, split.q2_span_names)
    q2_relevant_groups = relevant_position_groups_for_spans(split.q2_prepared, split.q2_span_names)

    spill_scores_start = time.perf_counter()
    qnorm_spill_scores = (
        score_cache_positions(
            query_rows=q2_query_rows,
            target_cache=stream_result.qnorm_spill.cache,
            competitor_cache=q1_turn.cache,
            pooling=config.pooling,
        )
        if stream_result.qnorm_spill.cache is not None and len(stream_result.qnorm_spill) > 0
        else {}
    )
    random_spill_scores = (
        score_cache_positions(
            query_rows=q2_query_rows,
            target_cache=stream_result.random_spill.cache,
            competitor_cache=q1_turn.cache,
            pooling=config.pooling,
        )
        if stream_result.random_spill.cache is not None and len(stream_result.random_spill) > 0
        else {}
    )
    spill_scoring_s = time.perf_counter() - spill_scores_start

    qnorm_spill_positions = list(stream_result.qnorm_spill.positions)
    random_spill_positions = list(stream_result.random_spill.positions)
    qnorm_ranked = rank_positions(qnorm_spill_positions, primary_scores=qnorm_spill_scores)
    random_spill_ranked = rank_positions(random_spill_positions, primary_scores=random_spill_scores)

    rows: list[dict[str, Any]] = []
    for b in config.b_values:
        b_int = int(b)
        row: dict[str, Any] = {
            "example_id": f"{split.split_spec.name}:ex{index + 1:03d}",
            "task": split.split_spec.name,
            "suite_task": config.task,
            "stage": config.stage,
            "index": int(index),
            "b": b_int,
            "q1_indices": list(split.split_spec.q1_indices),
            "q2_indices": list(split.split_spec.q2_indices),
            "q1_score": round(q1_score, 6),
            "b_stream_score": round(condition_b_score, 6),
            "q1_output": q1_turn.text,
            "b_stream_output": condition_b_output,
            "q1_generation_s": round(q1_generation_s, 6),
            "b_stream_generation_s": round(condition_b_generation_s, 6),
            "q2_query_rows_s": round(q2_query_s, 6),
            "spill_scoring_s": round(spill_scoring_s, 6),
            "context_length": int(config.total_context_length),
            "gpu_cache_cap": int(config.gpu_cache_cap),
            "turn_headroom": int(config.turn_headroom),
            "final_active_context_tokens": int(stream_result.final_active_context_tokens),
            "q1_active_tokens": int(len(q1_turn.cache)),
            "qnorm_spill_size": int(len(stream_result.qnorm_spill)),
            "random_spill_size": int(len(stream_result.random_spill)),
            "eviction_events": int(stream_result.eviction_events),
            "q2_relevant_positions": list(q2_relevant_positions),
            "qnorm_spill_coverage": round(_overlap_fraction(qnorm_spill_positions, q2_relevant_positions), 6),
            "random_spill_coverage": round(_overlap_fraction(random_spill_positions, q2_relevant_positions), 6),
        }

        if "IdleKV_stream" in config.conditions:
            select_start = time.perf_counter()
            restore_positions = qnorm_ranked[: min(b_int, len(qnorm_ranked))]
            repaired, restored, dropped = _repair_cache(
                active_cache=q1_turn.cache,
                spill=stream_result.qnorm_spill,
                restore_positions=restore_positions,
                query_rows=q2_query_rows,
                context_length=config.total_context_length,
                gpu_cache_cap=config.gpu_cache_cap,
                pooling=config.pooling,
            )
            select_s = time.perf_counter() - select_start
            output, score, generation_s = _run_condition(model, tokenizer, split.q2_prepared, repaired)
            row.update(
                {
                    "idlekv_stream_score": round(score, 6),
                    "idlekv_stream_output": output,
                    "idlekv_stream_generation_s": round(generation_s, 6),
                    "idlekv_stream_selection_s": round(select_s, 6),
                    "idlekv_stream_restored_count": len(restored),
                    "idlekv_stream_selected_positions": restored,
                    "idlekv_stream_dropped_positions": dropped,
                    "idlekv_stream_repair_coverage": round(_overlap_fraction(restored, q2_relevant_positions), 6),
                }
            )

        if "RandomRepair_stream" in config.conditions:
            select_start = time.perf_counter()
            restore_positions = _positions_by_random(
                qnorm_spill_positions,
                k=min(b_int, len(qnorm_spill_positions)),
                seed=int(config.seed) + (index + 1) * 1000 + b_int,
            )
            repaired, restored, dropped = _repair_cache(
                active_cache=q1_turn.cache,
                spill=stream_result.qnorm_spill,
                restore_positions=restore_positions,
                query_rows=q2_query_rows,
                context_length=config.total_context_length,
                gpu_cache_cap=config.gpu_cache_cap,
                pooling=config.pooling,
            )
            select_s = time.perf_counter() - select_start
            output, score, generation_s = _run_condition(model, tokenizer, split.q2_prepared, repaired)
            row.update(
                {
                    "random_repair_stream_score": round(score, 6),
                    "random_repair_stream_output": output,
                    "random_repair_stream_generation_s": round(generation_s, 6),
                    "random_repair_stream_selection_s": round(select_s, 6),
                    "random_repair_stream_restored_count": len(restored),
                    "random_repair_stream_selected_positions": restored,
                    "random_repair_stream_dropped_positions": dropped,
                    "random_repair_stream_repair_coverage": round(_overlap_fraction(restored, q2_relevant_positions), 6),
                }
            )

        if "OracleSpill_stream" in config.conditions:
            select_start = time.perf_counter()
            restore_positions = _select_oracle_spill_positions(
                spill_positions=qnorm_spill_positions,
                q2_relevant_positions=q2_relevant_positions,
                q2_relevant_groups=q2_relevant_groups,
                q2_scores=qnorm_spill_scores,
                k=b_int,
            )
            repaired, restored, dropped = _repair_cache(
                active_cache=q1_turn.cache,
                spill=stream_result.qnorm_spill,
                restore_positions=restore_positions,
                query_rows=q2_query_rows,
                context_length=config.total_context_length,
                gpu_cache_cap=config.gpu_cache_cap,
                pooling=config.pooling,
            )
            select_s = time.perf_counter() - select_start
            output, score, generation_s = _run_condition(model, tokenizer, split.q2_prepared, repaired)
            row.update(
                {
                    "oracle_spill_stream_score": round(score, 6),
                    "oracle_spill_stream_output": output,
                    "oracle_spill_stream_generation_s": round(generation_s, 6),
                    "oracle_spill_stream_selection_s": round(select_s, 6),
                    "oracle_spill_stream_restored_count": len(restored),
                    "oracle_spill_stream_selected_positions": restored,
                    "oracle_spill_stream_dropped_positions": dropped,
                    "oracle_spill_stream_repair_coverage": round(_overlap_fraction(restored, q2_relevant_positions), 6),
                }
            )

        if "RandomSpill_stream" in config.conditions:
            select_start = time.perf_counter()
            restore_positions = random_spill_ranked[: min(b_int, len(random_spill_ranked))]
            repaired, restored, dropped = _repair_cache(
                active_cache=q1_turn.cache,
                spill=stream_result.random_spill,
                restore_positions=restore_positions,
                query_rows=q2_query_rows,
                context_length=config.total_context_length,
                gpu_cache_cap=config.gpu_cache_cap,
                pooling=config.pooling,
            )
            select_s = time.perf_counter() - select_start
            output, score, generation_s = _run_condition(model, tokenizer, split.q2_prepared, repaired)
            row.update(
                {
                    "random_spill_stream_score": round(score, 6),
                    "random_spill_stream_output": output,
                    "random_spill_stream_generation_s": round(generation_s, 6),
                    "random_spill_stream_selection_s": round(select_s, 6),
                    "random_spill_stream_restored_count": len(restored),
                    "random_spill_stream_selected_positions": restored,
                    "random_spill_stream_dropped_positions": dropped,
                    "random_spill_stream_repair_coverage": round(_overlap_fraction(restored, q2_relevant_positions), 6),
                }
            )

        row["example_wall_s"] = round(time.perf_counter() - example_start, 6)
        rows.append(row)

    return rows


def _summarize_rows_by_b(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_b: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_b.setdefault(int(row["b"]), []).append(row)

    summary: dict[str, Any] = {}
    for b, group in sorted(by_b.items()):
        payload: dict[str, Any] = {
            "mean_q1_score": round(_mean(row["q1_score"] for row in group), 6),
            "mean_b_stream": round(_mean(row["b_stream_score"] for row in group), 6),
            "mean_final_active_context_tokens": round(_mean(row["final_active_context_tokens"] for row in group), 3),
            "mean_qnorm_spill_size": round(_mean(row["qnorm_spill_size"] for row in group), 3),
            "mean_random_spill_size": round(_mean(row["random_spill_size"] for row in group), 3),
            "mean_qnorm_spill_coverage": round(_mean(row["qnorm_spill_coverage"] for row in group), 6),
            "mean_random_spill_coverage": round(_mean(row["random_spill_coverage"] for row in group), 6),
            "n_examples": len(group),
        }
        for name, score_key in (
            ("idlekv_stream", "idlekv_stream_score"),
            ("random_repair_stream", "random_repair_stream_score"),
            ("oracle_spill_stream", "oracle_spill_stream_score"),
            ("random_spill_stream", "random_spill_stream_score"),
        ):
            if score_key in group[0]:
                payload[f"mean_{name}"] = round(_mean(row[score_key] for row in group), 6)
                payload[f"mean_{name}_lift_vs_b"] = round(
                    _mean(float(row[score_key]) - float(row["b_stream_score"]) for row in group),
                    6,
                )
                payload[f"pct_{name}_gt_b"] = round(
                    _pct(float(row[score_key]) > float(row["b_stream_score"]) for row in group),
                    6,
                )
                coverage_key = f"{name}_repair_coverage"
                if coverage_key in group[0]:
                    payload[f"mean_{name}_repair_coverage"] = round(_mean(row[coverage_key] for row in group), 6)
        summary[f"b{b}"] = payload
    return summary


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_names = sorted({str(row["task"]) for row in rows})
    if len(task_names) <= 1:
        return _summarize_rows_by_b(rows)
    return {
        "overall": _summarize_rows_by_b(rows),
        "by_task": {
            task_name: _summarize_rows_by_b([row for row in rows if str(row["task"]) == task_name])
            for task_name in task_names
        },
    }


def _artifact_path(config: Phase8Config) -> Path:
    stage_dir = ensure_results_dirs(config.stage)
    b_label = "-".join(str(value) for value in config.b_values)
    condition_label = _condition_label(config.conditions)
    seed_label = f"_seed{config.dataset_seed_offset}" if int(config.dataset_seed_offset) != 0 else ""
    return stage_dir / (
        f"{config.task}_l{config.total_context_length}_cap{config.gpu_cache_cap}_h{config.turn_headroom}"
        f"_chunk{config.chunk_size}_keep{int(config.keep_fraction * 100)}_spill{int(config.spill_fraction * 100)}"
        f"{seed_label}_n{config.num_samples}_b{b_label}_c{condition_label}.json"
    )


def _backup_existing_artifact(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup_path = path.with_name(f"{path.stem}.prev{path.suffix}")
    shutil.copy2(path, backup_path)
    return backup_path


def _print_progress(first: dict[str, Any], example_rows: list[dict[str, Any]]) -> None:
    parts = []
    for row in sorted(example_rows, key=lambda item: int(item["b"])):
        part = f"b={row['b']}:B={row['b_stream_score']:.3f}"
        if "idlekv_stream_score" in row:
            part += f"/I={row['idlekv_stream_score']:.3f}"
        if "random_repair_stream_score" in row:
            part += f"/RR={row['random_repair_stream_score']:.3f}"
        if "oracle_spill_stream_score" in row:
            part += f"/Or={row['oracle_spill_stream_score']:.3f}"
        if "random_spill_stream_score" in row:
            part += f"/RS={row['random_spill_stream_score']:.3f}"
        parts.append(part)
    print(
        f"[{first['example_id']}] "
        f"active={first['final_active_context_tokens']} "
        f"spill={first['qnorm_spill_size']} "
        f"cov={first['qnorm_spill_coverage']:.3f} "
        + " ".join(parts),
        flush=True,
    )


def run_experiment(config: Phase8Config) -> dict[str, Any]:
    """Run one Phase 8 experiment and persist a JSON artifact."""
    artifact_path = _artifact_path(config)
    overall_start = time.perf_counter()
    model = load_model()
    tokenizer = load_tokenizer()

    geometry = simulate_streaming_geometry(
        total_context_length=config.total_context_length,
        chunk_size=config.chunk_size,
        context_cap=config.gpu_cache_cap - config.turn_headroom,
        keep_fraction=config.keep_fraction,
    )
    print(
        "[phase8] geometry "
        f"final_active={geometry.final_active_tokens} "
        f"peak={geometry.peak_active_tokens} "
        f"events={geometry.eviction_events}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    per_base_stream_times: list[float] = []
    for index in range(int(config.num_samples)):
        if time.perf_counter() - overall_start > int(config.max_runtime_s):
            raise TimeoutError(
                f"Phase 8 run exceeded max_runtime_s={config.max_runtime_s} before sample {index + 1}."
            )
        base_example = build_base_example(
            split_spec=config.split_specs[0],
            index=index,
            context_length=config.total_context_length,
            tokenizer=tokenizer,
            dataset_seed_offset=config.dataset_seed_offset,
        )
        split_views = tuple(
            build_split_prepared_from_base_example(
                base_example=base_example,
                split_spec=split_spec,
                tokenizer=tokenizer,
            )
            for split_spec in config.split_specs
        )
        stream_result = stream_context_with_spill(
            model=model,
            context_ids=split_views[0].q1_prepared.context_ids,
            total_context_length=config.total_context_length,
            chunk_size=config.chunk_size,
            gpu_cache_cap=config.gpu_cache_cap,
            turn_headroom=config.turn_headroom,
            keep_fraction=config.keep_fraction,
            spill_fraction=config.spill_fraction,
            sink_size=config.sink_size,
            recency_window=config.recency_window,
            obs_window_size=config.obs_window_size,
            pooling=config.pooling,
            spill_hard_cap=config.spill_hard_cap,
            seed=int(config.seed) + index * 100_000,
        )
        per_base_stream_times.append(float(stream_result.stream_prefill_s))
        assert_strict_cap(stream_result.active_cache, cap=config.gpu_cache_cap - config.turn_headroom, label="final context cache")

        for split in split_views:
            example_rows = _run_one_split(
                model=model,
                tokenizer=tokenizer,
                config=config,
                split=split,
                stream_result=stream_result,
                index=index,
            )
            rows.extend(example_rows)
            if example_rows:
                _print_progress(example_rows[0], example_rows)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(config),
        "geometry": asdict(geometry),
        "aggregate": summarize_rows(rows),
        "rows": rows,
        "mean_stream_prefill_s": round(_mean(per_base_stream_times), 6),
        "elapsed_s": round(time.perf_counter() - overall_start, 6),
        "artifact_path": str(artifact_path),
    }
    backup_path = _backup_existing_artifact(artifact_path)
    if backup_path is not None:
        payload["previous_artifact_backup_path"] = str(backup_path)
    write_json(artifact_path, payload)
    return payload

