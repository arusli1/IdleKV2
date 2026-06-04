"""Phase 6 runner: two-turn matched-footprint repair on the split MQ-NIAH task."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import keyword
import shutil
import time
from pathlib import Path
import re
from statistics import fmean
from typing import Any, Iterable

import torch

from phases.phase1_degradation.phase1.evaluation import sample_score
from phases.phase1_degradation.phase1.prompting import char_span_to_token_positions
from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache, inject_kv, slice_kv
from phases.phase2_kv_cache.src.runtime import MODEL_DIR as DEFAULT_MODEL_DIR
from phases.phase2_kv_cache.src.runtime import load_model, load_tokenizer
from phases.phase3_eviction.src.runtime import build_position_tracked_cache, write_json

from .protocol import (
    CLEAN_SPLIT_SPECS,
    MQ_NIAH_2Q_CLEAN_SPLIT_SPEC,
    MQ_NIAH_3Q_CLEAN_SPLIT_SPEC,
    MQ_NIAH_6Q_CLEAN_SPLIT_SPECS,
    MQ_NIAH_8Q_CLEAN_SPLIT_SPECS,
    SPLIT_SPECS_BY_NAME,
    TAIL_LEAKY_SPLIT_SPECS,
    SplitTaskSpec,
    build_mismatched_question_ids,
    build_base_example,
    build_split_prepared_from_base_example,
    compute_q2_exact_query_rows,
    build_turn_n_keep_plan,
    compute_q2_query_rows,
    generate_turn,
    materialize_context_partition,
    relevant_position_groups_for_spans,
    relevant_positions_for_spans,
)
from .selectors import (
    contrastive_position_scores,
    pack_anchor_bursts,
    rank_positions,
    score_evicted_positions,
    select_coverage_aware_positions,
    select_idlekv_positions,
    select_mmr_positions,
    select_oldest_positions,
    select_oracle_positions,
    select_random_positions,
    select_refresh_positions,
    score_evicted_positions_budgeted,
)
from phases.phase18_pre_submission.src.page_summary import (
    score_evicted_positions_page_summary,
)

PHASE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PHASE_ROOT / "results"
SCHEMA_VERSION = "phase6-two-turn-v1"
ALLOWED_CONDITIONS = (
    "A",
    "B",
    "B_match",
    "IdleKV",
    "IdleKV-Coverage",
    "IdleKV-MMR",
    "WrongQ-K",
    "StaleQ-K",
    "ContrastiveQ-K",
    "Refresh-K",
    "Refresh-K-budgeted",
    "PageSummary-Quest-inspired",
    "RepairKV-no-burst",
    "RepairKV-chunked",
    "Random-K",
    "Oldest-K",
    "ToolFile-K",
    "AnchorWindow-K",
    "FileGatedIdleKV-K",
    "LexicalAnchor-K",
    "Oracle-K",
)
ALLOWED_QUERY_SCORING_MODES = ("proxy", "exact_q")
ALLOWED_ORACLE_MODES = ("burst_hindsight", "gold_spans")
ALLOWED_WRONG_QUERY_MODES = ("phantom_key", "donor_q2")
ALLOWED_INITIAL_COMPRESSORS = ("snapkv", "streaming_llm", "h2o", "scissorhands")
DEFAULT_WRONG_QUERY_DONOR_OFFSET = 100_000
Q2_SCORE_CONDITIONS = frozenset(
    (
        "IdleKV",
        "IdleKV-Coverage",
        "IdleKV-MMR",
        "ContrastiveQ-K",
        "Refresh-K",
        "Refresh-K-budgeted",
        "PageSummary-Quest-inspired",
        "RepairKV-no-burst",
        "RepairKV-chunked",
        "Oracle-K",
        "FileGatedIdleKV-K",
    )
)
CODE_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
LEXICAL_ANCHOR_STOPWORDS = frozenset(
    {
        "answer",
        "class",
        "constructing",
        "context",
        "declaration",
        "event",
        "executing",
        "exercising",
        "failed",
        "handling",
        "hidden",
        "identifier",
        "includes",
        "missing",
        "module",
        "repository",
        "report",
        "redacted",
        "recover",
        "statement",
        "tool",
        "use",
        "while",
    }
)

STAGE_DEFAULTS: dict[str, dict[str, Any]] = {
    "smoke": {
        "conditions": ("A", "B", "B_match", "IdleKV", "Oracle-K"),
        "num_samples": 8,
        "k_values": (8, 12, 24, 40, 48),
    },
    "pilot": {
        "conditions": ("A", "B", "B_match", "IdleKV", "Random-K", "Oldest-K", "Oracle-K"),
        "num_samples": 20,
        "k_values": (8, 12, 24, 40, 48),
    },
    "full": {
        "conditions": ("A", "B", "B_match", "IdleKV", "Oracle-K"),
        "num_samples": 100,
        "k_values": (8, 12, 24, 40, 48),
    },
}

TASK_ALIASES: dict[str, tuple[SplitTaskSpec, ...]] = {
    "clean_suite": CLEAN_SPLIT_SPECS,
    "diagnostic_suite": TAIL_LEAKY_SPLIT_SPECS,
    "mq_niah_2q_clean_suite": (MQ_NIAH_2Q_CLEAN_SPLIT_SPEC,),
    "mq_niah_3q_clean_suite": (MQ_NIAH_3Q_CLEAN_SPLIT_SPEC,),
    "mq_niah_6q_clean_suite": MQ_NIAH_6Q_CLEAN_SPLIT_SPECS,
    "mq_niah_8q_clean_suite": MQ_NIAH_8Q_CLEAN_SPLIT_SPECS,
}


@dataclass(frozen=True)
class Phase6Config:
    """Frozen configuration for one Phase 6 run."""

    stage: str
    task: str
    split_specs: tuple[SplitTaskSpec, ...]
    num_samples: int
    context_length: int = 32_768
    dataset_seed_offset: int = 0
    k_values: tuple[int, ...] = (8, 12, 24, 40, 48)
    conditions: tuple[str, ...] = ("A", "B", "B_match", "IdleKV", "Oracle-K")
    sink_size: int = 4
    recency_window: int = 128
    base_context_budget: int = 512
    pooling: str = "max"
    burst_left: int = 2
    burst_right: int = 20
    query_scoring_mode: str = "proxy"
    oracle_mode: str = "burst_hindsight"
    wrong_query_mode: str = "phantom_key"
    wrong_query_donor_offset: int = DEFAULT_WRONG_QUERY_DONOR_OFFSET
    model_dir: str = str(DEFAULT_MODEL_DIR)
    initial_compressor: str = "snapkv"
    # Phase 18 W1: TM budget multiplier applied to per-example T_repair when
    # running Refresh-K-budgeted, PageSummary-Quest-inspired, and
    # TM-Recompute-BM25 conditions. 1.05 default; bump to 1.20 if Step 1
    # smoke reports σ(T_repair)/μ(T_repair) > 0.10. Pre-registered in v5.
    tm_budget_multiplier: float = 1.05
    # Phase 18 v5.2: PageSummary-Quest-inspired chunk size for the
    # cheap-stage chunk summaries. Smaller chunks let Stage 2 visit
    # more chunks per budget; larger chunks reduce summary count. 128
    # is the default; sensitivity sweep at {32, 64, 128, 256} addresses
    # the "fixed-hyperparameter strawman" reviewer attack.
    page_summary_chunk_size: int = 128
    # Phase 18 v5.3: optional ABSOLUTE per-K budget for the time-matched
    # conditions, in seconds. When > 0, OVERRIDES the
    # T_repair-derived budget (idlekv_t_repair_s * tm_budget_multiplier).
    # Useful for tight-K-sweep experiments where the n_k amortization
    # would otherwise produce K-dependent budgets — set this to a fixed
    # ms value (e.g. 0.150 = 150 ms = deployment-realistic) to compare
    # at the same absolute budget across K.
    tm_budget_absolute_s: float = 0.0


def ensure_results_dirs(stage: str) -> Path:
    stage_dir = RESULTS_DIR / str(stage)
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def _condition_label(conditions: Iterable[str]) -> str:
    parts = [re.sub(r"[^a-z0-9]+", "", str(condition).lower()) for condition in conditions]
    return "-".join(part for part in parts if part)


def _needs_q2_candidate_scores(conditions: Iterable[str]) -> bool:
    return bool(Q2_SCORE_CONDITIONS.intersection(str(condition) for condition in conditions))


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
        raise ValueError(f"Unsupported Phase 6 task: {task!r}.")
    return (split_spec,)


def build_config(
    *,
    stage: str,
    task: str,
    num_samples: int | None = None,
    context_length: int = 32_768,
    dataset_seed_offset: int = 0,
    k_values: Iterable[int] | None = None,
    conditions: Iterable[str] | None = None,
    base_context_budget: int = 512,
    recency_window: int = 128,
    query_scoring_mode: str = "proxy",
    oracle_mode: str = "burst_hindsight",
    wrong_query_mode: str = "phantom_key",
    wrong_query_donor_offset: int = DEFAULT_WRONG_QUERY_DONOR_OFFSET,
    model_dir: str | Path | None = DEFAULT_MODEL_DIR,
    initial_compressor: str = "snapkv",
    tm_budget_multiplier: float = 1.05,
    page_summary_chunk_size: int = 128,
    tm_budget_absolute_s: float = 0.0,
) -> Phase6Config:
    """Construct one run config with stage defaults unless overridden."""
    normalized_stage = _normalize_stage(stage)
    split_specs = _normalize_task(task)
    defaults = STAGE_DEFAULTS[normalized_stage]
    normalized_k = tuple(dict.fromkeys(int(value) for value in (k_values or defaults["k_values"])))
    if not normalized_k or any(value <= 0 for value in normalized_k):
        raise ValueError("k_values must contain at least one positive integer.")
    normalized_conditions = tuple(dict.fromkeys(str(value) for value in (conditions or defaults["conditions"])))
    invalid_conditions = tuple(value for value in normalized_conditions if value not in ALLOWED_CONDITIONS)
    if invalid_conditions:
        raise ValueError(f"Unsupported Phase 6 condition(s): {invalid_conditions}.")
    resolved_num_samples = int(num_samples or defaults["num_samples"])
    if resolved_num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if int(context_length) <= 0:
        raise ValueError("context_length must be positive.")
    if int(base_context_budget) <= 0:
        raise ValueError("base_context_budget must be positive.")
    if int(recency_window) < 0:
        raise ValueError("recency_window must be non-negative.")
    normalized_query_scoring_mode = str(query_scoring_mode).strip().lower()
    if normalized_query_scoring_mode not in ALLOWED_QUERY_SCORING_MODES:
        raise ValueError(f"Unsupported query_scoring_mode: {query_scoring_mode!r}.")
    normalized_oracle_mode = str(oracle_mode).strip().lower()
    if normalized_oracle_mode not in ALLOWED_ORACLE_MODES:
        raise ValueError(f"Unsupported oracle_mode: {oracle_mode!r}.")
    normalized_wrong_query_mode = str(wrong_query_mode).strip().lower()
    if normalized_wrong_query_mode not in ALLOWED_WRONG_QUERY_MODES:
        raise ValueError(f"Unsupported wrong_query_mode: {wrong_query_mode!r}.")
    normalized_initial_compressor = str(initial_compressor).strip().lower()
    if normalized_initial_compressor not in ALLOWED_INITIAL_COMPRESSORS:
        raise ValueError(f"Unsupported initial_compressor: {initial_compressor!r}.")
    if int(wrong_query_donor_offset) <= 0:
        raise ValueError("wrong_query_donor_offset must be positive.")
    normalized_model_dir = Path(model_dir or DEFAULT_MODEL_DIR).expanduser()
    if not normalized_model_dir.exists():
        raise ValueError(f"model_dir does not exist: {normalized_model_dir}.")
    return Phase6Config(
        stage=normalized_stage,
        task=str(task).strip(),
        split_specs=tuple(split_specs),
        num_samples=resolved_num_samples,
        context_length=int(context_length),
        dataset_seed_offset=int(dataset_seed_offset),
        k_values=normalized_k,
        conditions=normalized_conditions,
        base_context_budget=int(base_context_budget),
        recency_window=int(recency_window),
        query_scoring_mode=normalized_query_scoring_mode,
        oracle_mode=normalized_oracle_mode,
        wrong_query_mode=normalized_wrong_query_mode,
        wrong_query_donor_offset=int(wrong_query_donor_offset),
        model_dir=str(normalized_model_dir),
        initial_compressor=normalized_initial_compressor,
        tm_budget_multiplier=float(tm_budget_multiplier),
        page_summary_chunk_size=int(page_summary_chunk_size),
        tm_budget_absolute_s=float(tm_budget_absolute_s),
    )


def _score_prediction(prediction: str, outputs: list[str]) -> float:
    return float(sample_score(prediction, outputs))


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


def _jaccard_fraction(left_positions: Iterable[int], right_positions: Iterable[int]) -> float:
    left = {int(position) for position in left_positions}
    right = {int(position) for position in right_positions}
    union = left | right
    if not union:
        return 1.0
    return float(len(left & right) / len(union))


def _select_segment_positions(
    *,
    evicted_positions: Iterable[int],
    segment_token_ranges: Iterable[tuple[str, int, int]],
    segment_name: str,
    k: int,
) -> list[int]:
    """Select file-local rows first, then oldest rows to preserve the K footprint."""
    ranges = [(int(start), int(end)) for name, start, end in segment_token_ranges if str(name) == segment_name]
    available = sorted(int(value) for value in evicted_positions)
    selected: list[int] = []
    if not ranges:
        return available[: int(k)]
    for position in available:
        if any(start <= position < end for start, end in ranges):
            selected.append(position)
            if len(selected) >= int(k):
                return selected
    selected_set = set(selected)
    for position in available:
        if position in selected_set:
            continue
        selected.append(position)
        if len(selected) >= int(k):
            return selected
    return selected


def _select_anchor_window_positions(
    *,
    evicted_positions: Iterable[int],
    anchor_positions: Iterable[int],
    k: int,
) -> list[int]:
    """Select evicted rows nearest to annotated relevant spans, then oldest rows.

    This is a label-assisted reference for real-content diagnostics. It does
    not score keys by the next-turn cue; it restores rows nearest to the
    benchmark-annotated relevant positions.
    """
    available = sorted(dict.fromkeys(int(position) for position in evicted_positions))
    target_k = min(max(int(k), 0), len(available))
    if target_k == 0:
        return []
    anchors = sorted(dict.fromkeys(int(position) for position in anchor_positions))
    if not anchors:
        return available[:target_k]

    def _distance_to_anchor(position: int) -> int:
        return min(abs(int(position) - anchor) for anchor in anchors)

    return sorted(available, key=lambda position: (_distance_to_anchor(position), position))[:target_k]


def _segment_ranges_for_name(
    *,
    segment_token_ranges: Iterable[tuple[str, int, int]],
    segment_name: str,
) -> list[tuple[int, int]]:
    return [(int(start), int(end)) for name, start, end in segment_token_ranges if str(name) == str(segment_name)]


def _position_in_ranges(position: int, ranges: Iterable[tuple[int, int]]) -> bool:
    return any(int(start) <= int(position) < int(end) for start, end in ranges)


def _count_positions_in_ranges(positions: Iterable[int], ranges: Iterable[tuple[int, int]]) -> int:
    range_list = list(ranges)
    return sum(1 for position in positions if _position_in_ranges(int(position), range_list))


def _contiguous_run_count(positions: Iterable[int]) -> int:
    """Count contiguous windows in a selected position set."""
    ordered = sorted(dict.fromkeys(int(position) for position in positions))
    if not ordered:
        return 0
    runs = 1
    prev = ordered[0]
    for position in ordered[1:]:
        if position != prev + 1:
            runs += 1
        prev = position
    return runs


def _select_file_gated_idlekv_positions(
    *,
    evicted_positions: Iterable[int],
    segment_token_ranges: Iterable[tuple[str, int, int]],
    segment_name: str,
    q2_scores: dict[int, float],
    turn_n_scores: dict[int, float],
    k: int,
    left: int,
    right: int,
) -> tuple[list[int], dict[str, Any]]:
    """Run the IdleKV selector inside the event file, then backfill globally."""
    available = sorted(dict.fromkeys(int(position) for position in evicted_positions))
    target_k = min(max(int(k), 0), len(available))
    ranges = _segment_ranges_for_name(
        segment_token_ranges=segment_token_ranges,
        segment_name=segment_name,
    )
    file_positions = [position for position in available if _position_in_ranges(position, ranges)]
    selected = select_idlekv_positions(
        evicted_positions=file_positions,
        q2_scores=q2_scores,
        turn_n_scores=turn_n_scores,
        k=target_k,
        left=left,
        right=right,
    )
    selected_set = set(selected)
    if len(selected) < target_k:
        global_idlekv = select_idlekv_positions(
            evicted_positions=available,
            q2_scores=q2_scores,
            turn_n_scores=turn_n_scores,
            k=target_k,
            left=left,
            right=right,
        )
        for position in global_idlekv:
            if int(position) in selected_set:
                continue
            selected.append(int(position))
            selected_set.add(int(position))
            if len(selected) >= target_k:
                break
    if len(selected) < target_k:
        for position in rank_positions(
            available,
            primary_scores=q2_scores,
            secondary_scores=turn_n_scores,
        ):
            if int(position) in selected_set:
                continue
            selected.append(int(position))
            selected_set.add(int(position))
            if len(selected) >= target_k:
                break
    selected_from_file = _count_positions_in_ranges(selected, ranges)
    metadata = {
        "candidate_count": int(len(file_positions)),
        "selected_from_file_count": int(selected_from_file),
        "selected_from_file_fraction": round(selected_from_file / max(1, len(selected)), 6),
        "backfill_count": int(max(0, len(selected) - selected_from_file)),
        "budget_matched": len(selected) == target_k,
    }
    return selected, metadata


def _identifier_parts(value: str) -> set[str]:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", str(value))
    raw_parts = re.split(r"[_\W]+", normalized)
    camel_parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+", str(value))
    return {
        part.lower()
        for part in [normalized, *raw_parts, *camel_parts]
        if len(part) >= 2
    }


def _extract_lexical_anchor_terms(*, repair_cue: str, answer: str, max_terms: int = 24) -> tuple[str, ...]:
    """Extract deployable code-like anchors from the answer-redacted event cue."""
    answer_parts = _identifier_parts(answer)
    answer_flat = re.sub(r"[^A-Za-z0-9]+", "", str(answer)).lower()
    terms: list[str] = []
    seen: set[str] = set()
    for match in CODE_IDENTIFIER_RE.finditer(str(repair_cue)):
        term = match.group(0)
        lower = term.lower()
        if lower in seen or lower in LEXICAL_ANCHOR_STOPWORDS or keyword.iskeyword(term):
            continue
        if len(term) < 3:
            continue
        term_flat = re.sub(r"[^A-Za-z0-9]+", "", term).lower()
        term_parts = _identifier_parts(term)
        if lower == "identifier" or term_flat == answer_flat:
            continue
        if answer_flat and (answer_flat in term_flat or term_flat in answer_flat):
            continue
        if answer_parts.intersection(term_parts):
            continue
        terms.append(term)
        seen.add(lower)
        if len(terms) >= int(max_terms):
            break
    return tuple(terms)


def _lexical_anchor_positions(
    *,
    tokenizer,
    prepared,
    terms: Iterable[str],
    evicted_positions: Iterable[int],
    preferred_ranges: Iterable[tuple[int, int]],
) -> tuple[list[int], int]:
    """Map lexical cue terms to token anchors, preferring the event-named file."""
    rendered_context = prepared.rendered_context
    evicted = {int(position) for position in evicted_positions}
    preferred_range_list = list(preferred_ranges)
    preferred: list[int] = []
    global_positions: list[int] = []
    for term in terms:
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(str(term))}(?![A-Za-z0-9_])")
        for match in pattern.finditer(rendered_context):
            token_positions = char_span_to_token_positions(
                tokenizer,
                rendered_context,
                match.start(),
                match.end(),
            )
            usable_positions = [int(position) for position in token_positions if int(position) in evicted]
            if not usable_positions:
                continue
            if any(_position_in_ranges(position, preferred_range_list) for position in usable_positions):
                preferred.extend(usable_positions)
            else:
                global_positions.extend(usable_positions)
    anchors = list(dict.fromkeys([*preferred, *global_positions]))
    return anchors, len(set(preferred))


def _select_lexical_anchor_positions(
    *,
    tokenizer,
    prepared,
    evicted_positions: Iterable[int],
    segment_token_ranges: Iterable[tuple[str, int, int]],
    segment_name: str,
    repair_cue: str,
    answer: str,
    k: int,
    left: int,
    right: int,
) -> tuple[list[int], dict[str, Any]]:
    """Select rows near deployable lexical anchors from the event cue."""
    available = sorted(dict.fromkeys(int(position) for position in evicted_positions))
    target_k = min(max(int(k), 0), len(available))
    ranges = _segment_ranges_for_name(
        segment_token_ranges=segment_token_ranges,
        segment_name=segment_name,
    )
    terms = _extract_lexical_anchor_terms(repair_cue=repair_cue, answer=answer)
    anchor_positions, preferred_anchor_count = _lexical_anchor_positions(
        tokenizer=tokenizer,
        prepared=prepared,
        terms=terms,
        evicted_positions=available,
        preferred_ranges=ranges,
    )
    anchor_burst_positions = pack_anchor_bursts(
        anchor_positions=anchor_positions,
        available_positions=available,
        k=target_k,
        left=left,
        right=right,
        backfill_positions=[],
    )
    selected = pack_anchor_bursts(
        anchor_positions=anchor_positions,
        available_positions=available,
        k=target_k,
        left=left,
        right=right,
        backfill_positions=available,
    )
    selected_from_file = _count_positions_in_ranges(selected, ranges)
    answer_flat = re.sub(r"[^A-Za-z0-9]+", "", str(answer)).lower()
    leak_terms = [
        term
        for term in terms
        if answer_flat and answer_flat in re.sub(r"[^A-Za-z0-9]+", "", term).lower()
    ]
    metadata = {
        "terms": list(terms),
        "term_count": int(len(terms)),
        "anchor_position_count": int(len(anchor_positions)),
        "window_count": int(_contiguous_run_count(selected)),
        "preferred_file_anchor_position_count": int(preferred_anchor_count),
        "selected_from_file_count": int(selected_from_file),
        "selected_from_file_fraction": round(selected_from_file / max(1, len(selected)), 6),
        "backfill_count": int(max(0, len(selected) - len(anchor_burst_positions))),
        "answer_leak_flag": bool(leak_terms),
        "answer_leak_terms": leak_terms,
        "budget_matched": len(selected) == target_k,
    }
    return selected, metadata


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _slice_fragment_by_positions(evicted_cache: PositionTrackedCache, positions: Iterable[int]) -> PositionTrackedCache | None:
    selected_positions = [int(position) for position in positions]
    if not selected_positions:
        return None
    position_to_dense = {int(position): dense_index for dense_index, position in enumerate(evicted_cache.positions)}
    dense_indices = [position_to_dense[int(position)] for position in selected_positions if int(position) in position_to_dense]
    if not dense_indices:
        return None
    fragment = slice_kv(evicted_cache, dense_indices)
    if not isinstance(fragment, PositionTrackedCache):
        raise RuntimeError("Phase 2 slice_kv did not preserve position tracking for the repair fragment.")
    return fragment


def _slice_by_original_positions(cache: PositionTrackedCache, positions: Iterable[int]) -> PositionTrackedCache:
    """Slice a tracked cache by original position labels rather than dense indices."""
    position_to_dense = {int(position): dense_index for dense_index, position in enumerate(cache.positions)}
    dense_indices = [position_to_dense[int(position)] for position in positions if int(position) in position_to_dense]
    fragment = slice_kv(cache, dense_indices)
    if not isinstance(fragment, PositionTrackedCache):
        raise RuntimeError("Phase 2 slice_kv did not preserve position tracking.")
    return fragment


def _score_context_positions(
    *,
    query_rows: torch.Tensor,
    full_post_q1_cache: PositionTrackedCache,
    context_len: int,
    tail_positions: Iterable[int],
    pooling: str,
) -> dict[int, float]:
    """Score all original context positions for the Refresh-buffered comparator."""
    context_cache = _slice_by_original_positions(full_post_q1_cache, range(int(context_len)))
    tail_cache = _slice_by_original_positions(full_post_q1_cache, tail_positions)
    return score_evicted_positions(
        query_rows=query_rows,
        evicted_cache=context_cache,
        active_cache=tail_cache if len(tail_cache) > 0 else None,
        pooling=pooling,
    )


def _materialize_context_positions(
    *,
    full_post_q1_cache: PositionTrackedCache,
    context_positions: Iterable[int],
    tail_positions: Iterable[int],
) -> PositionTrackedCache:
    """Materialize an active cache from selected context rows plus the Q1 tail."""
    selected_positions = list(sorted(dict.fromkeys(int(position) for position in context_positions)))
    selected_positions.extend(int(position) for position in tail_positions)
    return _slice_by_original_positions(full_post_q1_cache, selected_positions)


def _restore_positions(
    *,
    active_cache: PositionTrackedCache,
    evicted_cache: PositionTrackedCache,
    selected_positions: Iterable[int],
) -> tuple[PositionTrackedCache, dict[str, float]]:
    selected_fragment = _slice_fragment_by_positions(evicted_cache, selected_positions)
    if selected_fragment is None:
        return active_cache, {
            "transfer_ms": 0.0,
            "inject_ms": 0.0,
            "restored_count": 0.0,
        }

    target_device = active_cache.device
    _sync_if_cuda(target_device)
    transfer_start = time.perf_counter()
    selected_gpu = selected_fragment.to_device(target_device, non_blocking=True)
    _sync_if_cuda(target_device)
    transfer_ms = (time.perf_counter() - transfer_start) * 1000.0

    inject_start = time.perf_counter()
    repaired_cache = inject_kv(
        active_cache,
        selected_gpu,
        selected_gpu.positions,
    )
    _sync_if_cuda(target_device)
    inject_ms = (time.perf_counter() - inject_start) * 1000.0
    return repaired_cache, {
        "transfer_ms": transfer_ms,
        "inject_ms": inject_ms,
        "restored_count": float(len(selected_gpu.positions)),
    }


def _run_condition(
    *,
    model,
    tokenizer,
    prepared,
    cache: PositionTrackedCache,
) -> tuple[str, float, float]:
    generation_start = time.perf_counter()
    generated = generate_turn(model, tokenizer, prepared, cache)
    generation_s = time.perf_counter() - generation_start
    score = _score_prediction(generated.text, list(prepared.example.outputs))
    return generated.text, score, generation_s


def _run_selected_position_condition(
    *,
    model,
    tokenizer,
    prepared,
    active_cache: PositionTrackedCache,
    evicted_cache: PositionTrackedCache,
    base_positions: Iterable[int],
    selected_positions: list[int],
    relevant_positions: tuple[int, ...],
    selection_s: float,
    field_prefix: str,
) -> dict[str, Any]:
    repaired_cache, restore_timing = _restore_positions(
        active_cache=active_cache,
        evicted_cache=evicted_cache,
        selected_positions=selected_positions,
    )
    output, score, generation_s = _run_condition(
        model=model,
        tokenizer=tokenizer,
        prepared=prepared,
        cache=repaired_cache,
    )
    active_positions = tuple(sorted(set(int(position) for position in base_positions) | set(selected_positions)))
    return {
        f"{field_prefix}_score": round(score, 6),
        f"{field_prefix}_output": output,
        f"{field_prefix}_generation_s": round(generation_s, 6),
        f"{field_prefix}_selection_s": round(selection_s, 6),
        f"{field_prefix}_transfer_ms": round(restore_timing["transfer_ms"], 6),
        f"{field_prefix}_inject_ms": round(restore_timing["inject_ms"], 6),
        f"{field_prefix}_restored_count": int(restore_timing["restored_count"]),
        f"{field_prefix}_selected_positions": selected_positions,
        f"{field_prefix}_overlap_fraction": round(
            _overlap_fraction(selected_positions, relevant_positions),
            6,
        ),
        f"{field_prefix}_selected_overlap_fraction": round(
            _overlap_fraction(selected_positions, relevant_positions),
            6,
        ),
        f"{field_prefix}_active_overlap_fraction": round(
            _overlap_fraction(active_positions, relevant_positions),
            6,
        ),
    }


def _build_gold_span_oracle_candidates(
    *,
    model,
    tokenizer,
    prepared,
    active_cache: PositionTrackedCache,
    evicted_cache: PositionTrackedCache,
    relevant_position_groups: Iterable[Iterable[int]],
    q2_scores: dict[int, float],
    turn_n_scores: dict[int, float],
    base_output: str,
    base_score: float,
    base_generation_s: float,
) -> list[dict[str, Any]]:
    """Enumerate and score all gold-span group subsets once for one split.

    The returned candidates represent an ``up to K`` oracle over the benchmark's
    gold span groups under actual Q2 generation, not just a token-ranking proxy.
    """
    evicted_available = {int(position) for position in evicted_cache.positions}
    filtered_groups: list[tuple[int, ...]] = []
    seen_groups: set[tuple[int, ...]] = set()
    for group in relevant_position_groups:
        filtered = tuple(sorted(int(position) for position in group if int(position) in evicted_available))
        if filtered and filtered not in seen_groups:
            filtered_groups.append(filtered)
            seen_groups.add(filtered)

    candidates: list[dict[str, Any]] = [
        {
            "positions": (),
            "cost": 0,
            "score": float(base_score),
            "output": str(base_output),
            "generation_s": float(base_generation_s),
            "restore_timing": {
                "transfer_ms": 0.0,
                "inject_ms": 0.0,
                "restored_count": 0.0,
            },
            "q2_sum": 0.0,
            "turn_n_sum": 0.0,
        }
    ]
    seen_position_sets: set[tuple[int, ...]] = {()}

    for subset_bits in itertools.product((0, 1), repeat=len(filtered_groups)):
        chosen_groups = [group for keep, group in zip(subset_bits, filtered_groups, strict=True) if keep]
        chosen_positions = tuple(sorted({position for group in chosen_groups for position in group}))
        if chosen_positions in seen_position_sets:
            continue
        seen_position_sets.add(chosen_positions)
        repaired_cache, restore_timing = _restore_positions(
            active_cache=active_cache,
            evicted_cache=evicted_cache,
            selected_positions=chosen_positions,
        )
        output, score, generation_s = _run_condition(
            model=model,
            tokenizer=tokenizer,
            prepared=prepared,
            cache=repaired_cache,
        )
        candidates.append(
            {
                "positions": chosen_positions,
                "cost": len(chosen_positions),
                "score": float(score),
                "output": str(output),
                "generation_s": float(generation_s),
                "restore_timing": restore_timing,
                "q2_sum": float(sum(q2_scores.get(position, 0.0) for position in chosen_positions)),
                "turn_n_sum": float(sum(turn_n_scores.get(position, 0.0) for position in chosen_positions)),
            }
        )

    return candidates


def _choose_gold_span_oracle_candidate(
    *,
    candidates: Iterable[dict[str, Any]],
    k: int,
) -> dict[str, Any]:
    """Choose the best already-evaluated gold-span subset with cost <= K."""
    target_k = int(k)
    best_candidate: dict[str, Any] | None = None
    best_value: tuple[float, int, float, float] | None = None
    for candidate in candidates:
        cost = int(candidate["cost"])
        if cost > target_k:
            continue
        value = (
            float(candidate["score"]),
            -cost,
            float(candidate["q2_sum"]),
            float(candidate["turn_n_sum"]),
        )
        if best_value is None or value > best_value:
            best_candidate = candidate
            best_value = value
    if best_candidate is None:
        raise RuntimeError("Gold-span oracle search found no candidate within budget.")
    return best_candidate


def _run_one_split(
    *,
    model,
    tokenizer,
    config: Phase6Config,
    split,
    full_cache: PositionTrackedCache,
    index: int,
    wrong_q2_question_ids: torch.Tensor | None = None,
    repair_question_ids: torch.Tensor | None = None,
    stale_question_ids: torch.Tensor | None = None,
    tool_file_q2_path: str | None = None,
) -> list[dict[str, Any]]:
    example_start = time.perf_counter()
    q1_context_ids = split.q1_prepared.context_ids
    context_len = int(q1_context_ids.shape[1])

    q1_start = time.perf_counter()
    q1_turn = generate_turn(model, tokenizer, split.q1_prepared, full_cache)
    q1_generation_s = time.perf_counter() - q1_start
    q1_score = _score_prediction(q1_turn.text, list(split.q1_prepared.example.outputs))

    condition_a_output, condition_a_score, condition_a_generation_s = _run_condition(
        model=model,
        tokenizer=tokenizer,
        prepared=split.q2_prepared,
        cache=q1_turn.cache,
    )

    keep_plan_start = time.perf_counter()
    keep_plan = build_turn_n_keep_plan(
        post_q1_cache=q1_turn.cache,
        q1_answer_ids=q1_turn.token_ids,
        context_len=context_len,
        sink_size=config.sink_size,
        recency_window=config.recency_window,
        pooling=config.pooling,
        initial_compressor=config.initial_compressor,
    )
    keep_plan_s = time.perf_counter() - keep_plan_start

    base_partition = materialize_context_partition(
        full_post_q1_cache=q1_turn.cache,
        keep_plan=keep_plan,
        context_budget=config.base_context_budget,
    )
    condition_b_output, condition_b_score, condition_b_generation_s = _run_condition(
        model=model,
        tokenizer=tokenizer,
        prepared=split.q2_prepared,
        cache=base_partition.compressed,
    )

    q2_query_rows: torch.Tensor | None = None
    q2_scores: dict[int, float] = {}
    q2_query_s = 0.0
    q2_score_s = 0.0
    if _needs_q2_candidate_scores(config.conditions):
        scoring_question_ids = repair_question_ids if repair_question_ids is not None else split.q2_prepared.question_ids
        q2_query_start = time.perf_counter()
        if config.query_scoring_mode == "exact_q":
            q2_query_rows = compute_q2_exact_query_rows(
                model,
                active_cache=base_partition.compressed,
                question_ids=scoring_question_ids,
            )
        else:
            q2_query_rows = compute_q2_query_rows(
                model,
                active_cache=base_partition.compressed,
                question_ids=scoring_question_ids,
            )
        q2_query_s = time.perf_counter() - q2_query_start

        q2_score_start = time.perf_counter()
        q2_scores = score_evicted_positions(
            query_rows=q2_query_rows,
            evicted_cache=base_partition.evicted,
            active_cache=base_partition.compressed,
            pooling=config.pooling,
        )
        q2_score_s = time.perf_counter() - q2_score_start

        # Phase 18 W1: precompute host-CPU FP32 layer keys once per example
        # (NOT per K) so Refresh-K-budgeted and PageSummary-Quest-inspired
        # don't pay 9x redundant CPU<-GPU copies inside the K-loop. Each
        # copy is ~1.8GB on Qwen-shaped 32K context, so this matters.
        precomputed_evicted_keys: list[torch.Tensor] | None = None
        precomputed_active_keys: list[torch.Tensor | None] | None = None
        precomputed_context_keys: list[torch.Tensor] | None = None
        precomputed_q1tail_keys: list[torch.Tensor | None] | None = None
        if any(
            cond in config.conditions
            for cond in ("PageSummary-Quest-inspired", "Refresh-K-budgeted", "RepairKV-chunked")
        ):
            from phases.phase6_repair.src.selectors import precompute_host_layer_keys
            n_layers_local = int(q2_query_rows.shape[0])
            heads_per_layer = [int(q2_query_rows[i].shape[0]) for i in range(n_layers_local)]
            if "PageSummary-Quest-inspired" in config.conditions or "RepairKV-chunked" in config.conditions:
                precomputed_evicted_keys = precompute_host_layer_keys(base_partition.evicted)
                precomputed_active_keys = list(precompute_host_layer_keys(base_partition.compressed))
            if "Refresh-K-budgeted" in config.conditions:
                context_cache_pre = _slice_by_original_positions(q1_turn.cache, range(int(context_len)))
                tail_cache_pre = _slice_by_original_positions(q1_turn.cache, keep_plan.tail_positions)
                precomputed_context_keys = precompute_host_layer_keys(context_cache_pre)
                if len(tail_cache_pre) > 0:
                    precomputed_q1tail_keys = list(precompute_host_layer_keys(tail_cache_pre))
                else:
                    precomputed_q1tail_keys = [None] * n_layers_local


    refresh_scores: dict[int, float] | None = None
    refresh_score_s = 0.0
    if "Refresh-K" in config.conditions:
        assert q2_query_rows is not None
        refresh_score_start = time.perf_counter()
        refresh_scores = _score_context_positions(
            query_rows=q2_query_rows,
            full_post_q1_cache=q1_turn.cache,
            context_len=context_len,
            tail_positions=keep_plan.tail_positions,
            pooling=config.pooling,
        )
        refresh_score_s = time.perf_counter() - refresh_score_start

    wrong_q_scores: dict[int, float] | None = None
    wrong_q_query_s = 0.0
    wrong_q_score_s = 0.0
    if "WrongQ-K" in config.conditions or "ContrastiveQ-K" in config.conditions:
        if wrong_q2_question_ids is None:
            raise ValueError("WrongQ-K and ContrastiveQ-K require donor Q2 question ids.")
        wrong_q_query_start = time.perf_counter()
        if config.query_scoring_mode == "exact_q":
            wrong_q_query_rows = compute_q2_exact_query_rows(
                model,
                active_cache=base_partition.compressed,
                question_ids=wrong_q2_question_ids,
            )
        else:
            wrong_q_query_rows = compute_q2_query_rows(
                model,
                active_cache=base_partition.compressed,
                question_ids=wrong_q2_question_ids,
            )
        wrong_q_query_s = time.perf_counter() - wrong_q_query_start
        wrong_q_score_start = time.perf_counter()
        wrong_q_scores = score_evicted_positions(
            query_rows=wrong_q_query_rows,
            evicted_cache=base_partition.evicted,
            active_cache=base_partition.compressed,
            pooling=config.pooling,
        )
        wrong_q_score_s = time.perf_counter() - wrong_q_score_start

    stale_q_scores: dict[int, float] | None = None
    stale_q_query_s = 0.0
    stale_q_score_s = 0.0
    if "StaleQ-K" in config.conditions:
        stale_scoring_question_ids = stale_question_ids if stale_question_ids is not None else split.q1_prepared.question_ids
        stale_q_query_start = time.perf_counter()
        if config.query_scoring_mode == "exact_q":
            stale_q_query_rows = compute_q2_exact_query_rows(
                model,
                active_cache=base_partition.compressed,
                question_ids=stale_scoring_question_ids,
            )
        else:
            stale_q_query_rows = compute_q2_query_rows(
                model,
                active_cache=base_partition.compressed,
                question_ids=stale_scoring_question_ids,
            )
        stale_q_query_s = time.perf_counter() - stale_q_query_start
        stale_q_score_start = time.perf_counter()
        stale_q_scores = score_evicted_positions(
            query_rows=stale_q_query_rows,
            evicted_cache=base_partition.evicted,
            active_cache=base_partition.compressed,
            pooling=config.pooling,
        )
        stale_q_score_s = time.perf_counter() - stale_q_score_start

    q2_relevant_positions = relevant_positions_for_spans(split.q2_prepared, split.q2_span_names)
    q2_relevant_groups = relevant_position_groups_for_spans(split.q2_prepared, split.q2_span_names)
    evicted_positions = tuple(int(position) for position in base_partition.evicted.positions)
    gold_span_oracle_candidates: list[dict[str, Any]] | None = None
    gold_span_oracle_search_s = 0.0
    if "Oracle-K" in config.conditions and config.oracle_mode == "gold_spans":
        oracle_search_start = time.perf_counter()
        gold_span_oracle_candidates = _build_gold_span_oracle_candidates(
            model=model,
            tokenizer=tokenizer,
            prepared=split.q2_prepared,
            active_cache=base_partition.compressed,
            evicted_cache=base_partition.evicted,
            relevant_position_groups=q2_relevant_groups,
            q2_scores=q2_scores,
            turn_n_scores=keep_plan.importance_scores,
            base_output=condition_b_output,
            base_score=condition_b_score,
            base_generation_s=condition_b_generation_s,
        )
        gold_span_oracle_search_s = time.perf_counter() - oracle_search_start

    rows: list[dict[str, Any]] = []
    for k in config.k_values:
        k_int = int(k)
        row: dict[str, Any] = {
            "example_id": f"{split.split_spec.name}:ex{index + 1:03d}",
            "task": split.split_spec.name,
            "suite_task": config.task,
            "stage": config.stage,
            "index": int(index),
            "k": k_int,
            "q1_indices": list(split.split_spec.q1_indices),
            "q2_indices": list(split.split_spec.q2_indices),
            "q1_score": round(q1_score, 6),
            "condition_a_score": round(condition_a_score, 6),
            "condition_b_score": round(condition_b_score, 6),
            "q1_output": q1_turn.text,
            "condition_a_output": condition_a_output,
            "condition_b_output": condition_b_output,
            "q1_answer_tokens": int(q1_turn.token_ids.numel()),
            "q2_relevant_positions": list(q2_relevant_positions),
            "q1_generation_s": round(q1_generation_s, 6),
            "condition_a_generation_s": round(condition_a_generation_s, 6),
            "turn_n_keep_plan_s": round(keep_plan_s, 6),
            "condition_b_generation_s": round(condition_b_generation_s, 6),
            "q2_query_rows_s": round(q2_query_s, 6),
            "q2_evicted_scoring_s": round(q2_score_s, 6),
            "refresh_context_scoring_s": round(refresh_score_s, 6),
            "wrong_q_query_rows_s": round(wrong_q_query_s, 6),
            "wrong_q_evicted_scoring_s": round(wrong_q_score_s, 6),
            "stale_q_query_rows_s": round(stale_q_query_s, 6),
            "stale_q_evicted_scoring_s": round(stale_q_score_s, 6),
            "wrong_query_mode": config.wrong_query_mode,
            "wrong_query_donor_offset": int(config.wrong_query_donor_offset),
            "base_context_budget": int(config.base_context_budget),
            "context_length": context_len,
            "evicted_context_tokens": int(len(base_partition.evicted.positions)),
            "b_kept_context_positions": list(base_partition.kept_context_positions),
        }

        bmatch_partition = materialize_context_partition(
            full_post_q1_cache=q1_turn.cache,
            keep_plan=keep_plan,
            context_budget=config.base_context_budget + k_int,
        )
        bmatch_output, bmatch_score, bmatch_generation_s = _run_condition(
            model=model,
            tokenizer=tokenizer,
            prepared=split.q2_prepared,
            cache=bmatch_partition.compressed,
        )
        row.update(
            {
                "b_match_score": round(bmatch_score, 6),
                "b_match_output": bmatch_output,
                "b_match_generation_s": round(bmatch_generation_s, 6),
                "b_match_kept_context_positions": list(bmatch_partition.kept_context_positions),
                "b_match_overlap_fraction": round(
                    _overlap_fraction(bmatch_partition.kept_context_positions, q2_relevant_positions),
                    6,
                ),
                "b_match_active_overlap_fraction": round(
                    _overlap_fraction(bmatch_partition.kept_context_positions, q2_relevant_positions),
                    6,
                ),
            }
        )
        row["condition_b_overlap_fraction"] = round(
            _overlap_fraction(base_partition.kept_context_positions, q2_relevant_positions),
            6,
        )

        if "IdleKV" in config.conditions:
            select_start = time.perf_counter()
            idlekv_positions = select_idlekv_positions(
                evicted_positions=evicted_positions,
                q2_scores=q2_scores,
                turn_n_scores=keep_plan.importance_scores,
                k=k_int,
                left=config.burst_left,
                right=config.burst_right,
            )
            select_s = time.perf_counter() - select_start
            repaired_cache, restore_timing = _restore_positions(
                active_cache=base_partition.compressed,
                evicted_cache=base_partition.evicted,
                selected_positions=idlekv_positions,
            )
            idlekv_output, idlekv_score, idlekv_generation_s = _run_condition(
                model=model,
                tokenizer=tokenizer,
                prepared=split.q2_prepared,
                cache=repaired_cache,
            )
            row.update(
                {
                    "idlekv_score": round(idlekv_score, 6),
                    "idlekv_output": idlekv_output,
                    "idlekv_generation_s": round(idlekv_generation_s, 6),
                    "idlekv_selection_s": round(select_s, 6),
                    "idlekv_transfer_ms": round(restore_timing["transfer_ms"], 6),
                    "idlekv_inject_ms": round(restore_timing["inject_ms"], 6),
                    "idlekv_restored_count": int(restore_timing["restored_count"]),
                    "idlekv_selected_positions": idlekv_positions,
                    "idlekv_overlap_fraction": round(
                        _overlap_fraction(idlekv_positions, q2_relevant_positions),
                        6,
                    ),
                    "idlekv_selected_overlap_fraction": round(
                        _overlap_fraction(idlekv_positions, q2_relevant_positions),
                        6,
                    ),
                    "idlekv_active_overlap_fraction": round(
                        _overlap_fraction(
                            tuple(sorted(set(base_partition.kept_context_positions) | set(idlekv_positions))),
                            q2_relevant_positions,
                        ),
                        6,
                    ),
                }
            )

        if "FileGatedIdleKV-K" in config.conditions:
            if not tool_file_q2_path:
                raise ValueError("FileGatedIdleKV-K requires tool_file_q2_path.")
            repair_cue = str(split.q2_prepared.example.metadata.get("repair_cue", ""))
            event_contains_q2_path = str(tool_file_q2_path) in repair_cue
            select_start = time.perf_counter()
            file_gated_positions, file_gated_meta = _select_file_gated_idlekv_positions(
                evicted_positions=evicted_positions,
                segment_token_ranges=split.q2_prepared.segment_token_ranges,
                segment_name=f"file:{tool_file_q2_path}",
                q2_scores=q2_scores,
                turn_n_scores=keep_plan.importance_scores,
                k=k_int,
                left=config.burst_left,
                right=config.burst_right,
            )
            select_s = time.perf_counter() - select_start
            payload = _run_selected_position_condition(
                model=model,
                tokenizer=tokenizer,
                prepared=split.q2_prepared,
                active_cache=base_partition.compressed,
                evicted_cache=base_partition.evicted,
                base_positions=base_partition.kept_context_positions,
                selected_positions=file_gated_positions,
                relevant_positions=q2_relevant_positions,
                selection_s=select_s,
                field_prefix="file_gated_idlekv",
            )
            payload.update(
                {
                    "file_gated_idlekv_q2_path": str(tool_file_q2_path),
                    "file_gated_idlekv_path_source": (
                        "event_repair_cue" if event_contains_q2_path else "q2_metadata"
                    ),
                    "file_gated_idlekv_event_contains_q2_path": bool(event_contains_q2_path),
                    "file_gated_idlekv_candidate_count": int(file_gated_meta["candidate_count"]),
                    "file_gated_idlekv_selected_from_file_count": int(
                        file_gated_meta["selected_from_file_count"]
                    ),
                    "file_gated_idlekv_selected_from_file_fraction": float(
                        file_gated_meta["selected_from_file_fraction"]
                    ),
                    "file_gated_idlekv_backfill_count": int(file_gated_meta["backfill_count"]),
                    "file_gated_idlekv_budget_matched": bool(file_gated_meta["budget_matched"]),
                }
            )
            row.update(payload)

        if "IdleKV-Coverage" in config.conditions:
            select_start = time.perf_counter()
            idlekv_coverage_positions = select_coverage_aware_positions(
                evicted_positions=evicted_positions,
                q2_scores=q2_scores,
                turn_n_scores=keep_plan.importance_scores,
                k=k_int,
                left=config.burst_left,
                right=config.burst_right,
            )
            select_s = time.perf_counter() - select_start
            row.update(
                _run_selected_position_condition(
                    model=model,
                    tokenizer=tokenizer,
                    prepared=split.q2_prepared,
                    active_cache=base_partition.compressed,
                    evicted_cache=base_partition.evicted,
                    base_positions=base_partition.kept_context_positions,
                    selected_positions=idlekv_coverage_positions,
                    relevant_positions=q2_relevant_positions,
                    selection_s=select_s,
                    field_prefix="idlekv_coverage",
                )
            )

        if "IdleKV-MMR" in config.conditions:
            select_start = time.perf_counter()
            idlekv_mmr_positions = select_mmr_positions(
                evicted_positions=evicted_positions,
                q2_scores=q2_scores,
                turn_n_scores=keep_plan.importance_scores,
                k=k_int,
                left=config.burst_left,
                right=config.burst_right,
            )
            select_s = time.perf_counter() - select_start
            row.update(
                _run_selected_position_condition(
                    model=model,
                    tokenizer=tokenizer,
                    prepared=split.q2_prepared,
                    active_cache=base_partition.compressed,
                    evicted_cache=base_partition.evicted,
                    base_positions=base_partition.kept_context_positions,
                    selected_positions=idlekv_mmr_positions,
                    relevant_positions=q2_relevant_positions,
                    selection_s=select_s,
                    field_prefix="idlekv_mmr",
                )
            )

        if "WrongQ-K" in config.conditions:
            assert wrong_q_scores is not None
            select_start = time.perf_counter()
            wrong_q_positions = select_idlekv_positions(
                evicted_positions=evicted_positions,
                q2_scores=wrong_q_scores,
                turn_n_scores=keep_plan.importance_scores,
                k=k_int,
                left=config.burst_left,
                right=config.burst_right,
            )
            select_s = time.perf_counter() - select_start
            repaired_cache, restore_timing = _restore_positions(
                active_cache=base_partition.compressed,
                evicted_cache=base_partition.evicted,
                selected_positions=wrong_q_positions,
            )
            wrong_q_output, wrong_q_score, wrong_q_generation_s = _run_condition(
                model=model,
                tokenizer=tokenizer,
                prepared=split.q2_prepared,
                cache=repaired_cache,
            )
            row.update(
                {
                    "wrong_q_k_score": round(wrong_q_score, 6),
                    "wrong_q_k_output": wrong_q_output,
                    "wrong_q_k_generation_s": round(wrong_q_generation_s, 6),
                    "wrong_q_k_selection_s": round(select_s, 6),
                    "wrong_q_k_transfer_ms": round(restore_timing["transfer_ms"], 6),
                    "wrong_q_k_inject_ms": round(restore_timing["inject_ms"], 6),
                    "wrong_q_k_restored_count": int(restore_timing["restored_count"]),
                    "wrong_q_k_selected_positions": wrong_q_positions,
                    "wrong_q_k_overlap_fraction": round(
                        _overlap_fraction(wrong_q_positions, q2_relevant_positions),
                        6,
                    ),
                    "wrong_q_k_selected_overlap_fraction": round(
                        _overlap_fraction(wrong_q_positions, q2_relevant_positions),
                        6,
                    ),
                    "wrong_q_k_active_overlap_fraction": round(
                        _overlap_fraction(
                            tuple(sorted(set(base_partition.kept_context_positions) | set(wrong_q_positions))),
                            q2_relevant_positions,
                        ),
                        6,
                    ),
                }
            )

        if "StaleQ-K" in config.conditions:
            assert stale_q_scores is not None
            select_start = time.perf_counter()
            stale_q_positions = select_idlekv_positions(
                evicted_positions=evicted_positions,
                q2_scores=stale_q_scores,
                turn_n_scores=keep_plan.importance_scores,
                k=k_int,
                left=config.burst_left,
                right=config.burst_right,
            )
            select_s = time.perf_counter() - select_start
            repaired_cache, restore_timing = _restore_positions(
                active_cache=base_partition.compressed,
                evicted_cache=base_partition.evicted,
                selected_positions=stale_q_positions,
            )
            stale_q_output, stale_q_score, stale_q_generation_s = _run_condition(
                model=model,
                tokenizer=tokenizer,
                prepared=split.q2_prepared,
                cache=repaired_cache,
            )
            row.update(
                {
                    "stale_q_k_score": round(stale_q_score, 6),
                    "stale_q_k_output": stale_q_output,
                    "stale_q_k_generation_s": round(stale_q_generation_s, 6),
                    "stale_q_k_selection_s": round(select_s, 6),
                    "stale_q_k_transfer_ms": round(restore_timing["transfer_ms"], 6),
                    "stale_q_k_inject_ms": round(restore_timing["inject_ms"], 6),
                    "stale_q_k_restored_count": int(restore_timing["restored_count"]),
                    "stale_q_k_selected_positions": stale_q_positions,
                    "stale_q_k_overlap_fraction": round(
                        _overlap_fraction(stale_q_positions, q2_relevant_positions),
                        6,
                    ),
                    "stale_q_k_selected_overlap_fraction": round(
                        _overlap_fraction(stale_q_positions, q2_relevant_positions),
                        6,
                    ),
                    "stale_q_k_active_overlap_fraction": round(
                        _overlap_fraction(
                            tuple(sorted(set(base_partition.kept_context_positions) | set(stale_q_positions))),
                            q2_relevant_positions,
                        ),
                        6,
                    ),
                }
            )

        if "Refresh-K" in config.conditions:
            assert refresh_scores is not None
            select_start = time.perf_counter()
            refresh_positions = select_refresh_positions(
                context_positions=range(context_len),
                mandatory_positions=keep_plan.mandatory_context_positions,
                q2_scores=refresh_scores,
                turn_n_scores=keep_plan.importance_scores,
                context_budget=config.base_context_budget + k_int,
                left=config.burst_left,
                right=config.burst_right,
            )
            select_s = time.perf_counter() - select_start
            materialize_start = time.perf_counter()
            refreshed_cache = _materialize_context_positions(
                full_post_q1_cache=q1_turn.cache,
                context_positions=refresh_positions,
                tail_positions=keep_plan.tail_positions,
            )
            _sync_if_cuda(refreshed_cache.device)
            materialize_ms = (time.perf_counter() - materialize_start) * 1000.0
            refresh_set = {int(position) for position in refresh_positions}
            base_set = {int(position) for position in base_partition.kept_context_positions}
            evicted_set = {int(position) for position in base_partition.evicted_context_positions}
            mandatory_set = {int(position) for position in keep_plan.mandatory_context_positions}
            context_set = set(range(context_len))
            selected_from_base_count = len(refresh_set & base_set)
            selected_from_evicted_count = len(refresh_set & evicted_set)
            dropped_base_count = len(base_set - refresh_set)
            refresh_selected_count = max(1, len(refresh_positions))
            refresh_output, refresh_score, refresh_generation_s = _run_condition(
                model=model,
                tokenizer=tokenizer,
                prepared=split.q2_prepared,
                cache=refreshed_cache,
            )
            row.update(
                {
                    "refresh_k_score": round(refresh_score, 6),
                    "refresh_k_output": refresh_output,
                    "refresh_k_generation_s": round(refresh_generation_s, 6),
                    "refresh_k_selection_s": round(select_s, 6),
                    "refresh_k_materialize_ms": round(materialize_ms, 6),
                    "refresh_k_context_budget": int(config.base_context_budget + k_int),
                    "refresh_k_selected_positions": refresh_positions,
                    "refresh_scope": "buffered_active_plus_evicted",
                    "refresh_uses_prefix_recompute": False,
                    "refresh_scoring_pool": "all_original_context_plus_q1_tail",
                    "refresh_materialization_source": "full_post_q1_cache",
                    "refresh_selection_policy": "q2_score_then_burst_pack",
                    "refresh_selected_from_base_count": int(selected_from_base_count),
                    "refresh_selected_from_evicted_count": int(selected_from_evicted_count),
                    "refresh_selected_from_evicted_fraction": round(
                        selected_from_evicted_count / refresh_selected_count,
                        6,
                    ),
                    "refresh_dropped_base_count": int(dropped_base_count),
                    "refresh_dropped_base_fraction": round(
                        dropped_base_count / max(1, len(base_set)),
                        6,
                    ),
                    "refresh_selected_unique": len(refresh_positions) == len(refresh_set),
                    "refresh_selected_in_context_range": refresh_set <= context_set,
                    "refresh_mandatory_preserved": mandatory_set <= refresh_set,
                    "refresh_budget_invariant": len(refresh_positions) == min(context_len, int(config.base_context_budget + k_int)),
                    "refresh_jaccard_with_b_match": round(
                        _jaccard_fraction(refresh_positions, bmatch_partition.kept_context_positions),
                        6,
                    ),
                    "refresh_k_overlap_fraction": round(
                        _overlap_fraction(refresh_positions, q2_relevant_positions),
                        6,
                    ),
                    "refresh_k_selected_overlap_fraction": round(
                        _overlap_fraction(refresh_positions, q2_relevant_positions),
                        6,
                    ),
                    "refresh_k_active_overlap_fraction": round(
                        _overlap_fraction(refresh_positions, q2_relevant_positions),
                        6,
                    ),
                }
            )

        # ---- Phase 18 W1 NEW CONDITIONS -----------------------------------
        # T_repair, computed per (example, K). Includes the example-level
        # Q2 projection cost q2_query_s and the example-level Q2 scoring
        # cost q2_score_s, both amortized across len(k_values) (because
        # they happen once per example but cover all K's), plus the
        # per-(example, K) selection + movement work. This matches the
        # Phase 18 v5 plan W4.4 stage decomposition.
        idlekv_t_repair_s: float | None = None
        if "IdleKV" in config.conditions and "idlekv_selection_s" in row:
            n_k = max(1, len(config.k_values))
            amortized_q2_s = (float(q2_query_s) + float(q2_score_s)) / float(n_k)
            idlekv_t_repair_s = (
                amortized_q2_s
                + float(row["idlekv_selection_s"])
                + float(row["idlekv_inject_ms"]) / 1000.0
                + float(row["idlekv_transfer_ms"]) / 1000.0
            )
            row["idlekv_t_repair_s"] = round(idlekv_t_repair_s, 6)
            row["idlekv_q2_proj_ms"] = round(float(q2_query_s) * 1000.0, 6)
            row["idlekv_q2_score_ms"] = round(float(q2_score_s) * 1000.0, 6)
            row["idlekv_t_repair_components"] = {
                "amortized_q2_proj_ms": round(float(q2_query_s) * 1000.0 / n_k, 6),
                "amortized_q2_score_ms": round(float(q2_score_s) * 1000.0 / n_k, 6),
                "selection_ms": round(float(row["idlekv_selection_s"]) * 1000.0, 6),
                "transfer_ms": round(float(row["idlekv_transfer_ms"]), 6),
                "inject_ms": round(float(row["idlekv_inject_ms"]), 6),
                "n_k_values": int(n_k),
            }

        if "RepairKV-chunked" in config.conditions:
            # Chunk-restricted softmax variant of RepairKV's scoring,
            # matching PageSummary-Quest-inspired's denominator regime.
            # Disambiguates "lifecycle-slot effect" (algorithmic) from
            # "scorer denominator effect" (chunk_size-dependent
            # normalization). chunk_size matches PageSummary's
            # config.page_summary_chunk_size for the apples-to-apples
            # comparison.
            assert q2_query_rows is not None
            chunked_score_start = time.perf_counter()
            chunked_evicted_scores, chunked_info = score_evicted_positions_budgeted(
                query_rows=q2_query_rows,
                evicted_cache=base_partition.evicted,
                active_cache=base_partition.compressed,
                pooling=config.pooling,
                wallclock_deadline_s=None,  # no cap; isolate denominator effect
                position_chunk_size=int(config.page_summary_chunk_size),
                precomputed_evicted_layer_keys=precomputed_evicted_keys,
                precomputed_active_layer_keys=precomputed_active_keys,
            )
            chunked_score_s = time.perf_counter() - chunked_score_start
            chunked_select_start = time.perf_counter()
            chunked_positions = select_idlekv_positions(
                evicted_positions=evicted_positions,
                q2_scores=chunked_evicted_scores,
                turn_n_scores=keep_plan.importance_scores,
                k=k_int,
                left=config.burst_left,
                right=config.burst_right,
            )
            chunked_select_s = time.perf_counter() - chunked_select_start
            chunked_cache, chunked_timing = _restore_positions(
                active_cache=base_partition.compressed,
                evicted_cache=base_partition.evicted,
                selected_positions=chunked_positions,
            )
            chunked_output, chunked_score, chunked_generation_s = _run_condition(
                model=model,
                tokenizer=tokenizer,
                prepared=split.q2_prepared,
                cache=chunked_cache,
            )
            row.update(
                {
                    "repairkv_chunked_score": round(chunked_score, 6),
                    "repairkv_chunked_output": chunked_output,
                    "repairkv_chunked_generation_s": round(chunked_generation_s, 6),
                    "repairkv_chunked_score_s": round(chunked_score_s, 6),
                    "repairkv_chunked_selection_s": round(chunked_select_s, 6),
                    "repairkv_chunked_transfer_ms": round(chunked_timing["transfer_ms"], 6),
                    "repairkv_chunked_inject_ms": round(chunked_timing["inject_ms"], 6),
                    "repairkv_chunked_restored_count": int(chunked_timing["restored_count"]),
                    "repairkv_chunked_chunk_size": int(config.page_summary_chunk_size),
                    "repairkv_chunked_selected_positions": chunked_positions,
                    "repairkv_chunked_overlap_fraction": round(
                        _overlap_fraction(chunked_positions, q2_relevant_positions), 6,
                    ),
                }
            )

        if "RepairKV-no-burst" in config.conditions:
            select_start = time.perf_counter()
            no_burst_positions = select_idlekv_positions(
                evicted_positions=evicted_positions,
                q2_scores=q2_scores,
                turn_n_scores=keep_plan.importance_scores,
                k=k_int,
                left=0,
                right=0,
            )
            no_burst_select_s = time.perf_counter() - select_start
            no_burst_cache, no_burst_timing = _restore_positions(
                active_cache=base_partition.compressed,
                evicted_cache=base_partition.evicted,
                selected_positions=no_burst_positions,
            )
            no_burst_output, no_burst_score, no_burst_generation_s = _run_condition(
                model=model,
                tokenizer=tokenizer,
                prepared=split.q2_prepared,
                cache=no_burst_cache,
            )
            row.update(
                {
                    "repairkv_no_burst_score": round(no_burst_score, 6),
                    "repairkv_no_burst_output": no_burst_output,
                    "repairkv_no_burst_generation_s": round(no_burst_generation_s, 6),
                    "repairkv_no_burst_selection_s": round(no_burst_select_s, 6),
                    "repairkv_no_burst_transfer_ms": round(no_burst_timing["transfer_ms"], 6),
                    "repairkv_no_burst_inject_ms": round(no_burst_timing["inject_ms"], 6),
                    "repairkv_no_burst_restored_count": int(no_burst_timing["restored_count"]),
                    "repairkv_no_burst_selected_positions": no_burst_positions,
                    "repairkv_no_burst_overlap_fraction": round(
                        _overlap_fraction(no_burst_positions, q2_relevant_positions), 6,
                    ),
                }
            )

        if "Refresh-K-budgeted" in config.conditions:
            assert q2_query_rows is not None
            if idlekv_t_repair_s is None:
                raise ValueError(
                    "Refresh-K-budgeted requires IdleKV in conditions to source per-example T_repair."
                )
            # 1.05 buffer per Phase 18 plan.
            budget_s = (
                float(config.tm_budget_absolute_s)
                if float(config.tm_budget_absolute_s) > 0.0
                else float(idlekv_t_repair_s) * float(config.tm_budget_multiplier)
            )
            score_start = time.perf_counter()
            # Budgeted scorer over (active + evicted-context) — same scope as
            # unbudgeted Refresh-K (entire post-Q1 context plus Q1 tail).
            # Reuse precomputed host-FP32 keys to avoid 9x redundant CPU
            # copies inside the K-loop.
            context_cache = _slice_by_original_positions(q1_turn.cache, range(int(context_len)))
            tail_cache = _slice_by_original_positions(q1_turn.cache, keep_plan.tail_positions)
            partial_scores, refresh_budgeted_info = score_evicted_positions_budgeted(
                query_rows=q2_query_rows,
                evicted_cache=context_cache,
                active_cache=tail_cache if len(tail_cache) > 0 else None,
                pooling=config.pooling,
                wallclock_deadline_s=budget_s,
                position_chunk_size=1024,
                precomputed_evicted_layer_keys=precomputed_context_keys,
                precomputed_active_layer_keys=precomputed_q1tail_keys,
            )
            score_s = time.perf_counter() - score_start
            select_start = time.perf_counter()
            refresh_budgeted_positions = select_refresh_positions(
                context_positions=range(context_len),
                mandatory_positions=keep_plan.mandatory_context_positions,
                q2_scores=partial_scores,
                turn_n_scores=keep_plan.importance_scores,
                context_budget=config.base_context_budget + k_int,
                left=config.burst_left,
                right=config.burst_right,
            )
            select_s = time.perf_counter() - select_start
            materialize_start = time.perf_counter()
            refresh_budgeted_cache = _materialize_context_positions(
                full_post_q1_cache=q1_turn.cache,
                context_positions=refresh_budgeted_positions,
                tail_positions=keep_plan.tail_positions,
            )
            _sync_if_cuda(refresh_budgeted_cache.device)
            materialize_ms = (time.perf_counter() - materialize_start) * 1000.0
            refresh_budgeted_output, refresh_budgeted_score, refresh_budgeted_gen_s = _run_condition(
                model=model,
                tokenizer=tokenizer,
                prepared=split.q2_prepared,
                cache=refresh_budgeted_cache,
            )
            row.update(
                {
                    "refresh_k_budgeted_score": round(refresh_budgeted_score, 6),
                    "refresh_k_budgeted_output": refresh_budgeted_output,
                    "refresh_k_budgeted_generation_s": round(refresh_budgeted_gen_s, 6),
                    "refresh_k_budgeted_score_s": round(score_s, 6),
                    "refresh_k_budgeted_selection_s": round(select_s, 6),
                    "refresh_k_budgeted_materialize_ms": round(materialize_ms, 6),
                    "refresh_k_budgeted_t_repair_s": round(budget_s, 6),
                    "refresh_k_budgeted_positions_scored": int(refresh_budgeted_info["positions_scored"]),
                    "refresh_k_budgeted_positions_total": int(refresh_budgeted_info["positions_total"]),
                    "refresh_k_budgeted_cap_fired": bool(refresh_budgeted_info["cap_fired"]),
                    "refresh_k_budgeted_layer_chunks_completed": int(
                        refresh_budgeted_info["layer_chunks_completed"]
                    ),
                    "refresh_k_budgeted_selected_positions": refresh_budgeted_positions,
                    "refresh_k_budgeted_overlap_fraction": round(
                        _overlap_fraction(refresh_budgeted_positions, q2_relevant_positions), 6,
                    ),
                }
            )

        if "PageSummary-Quest-inspired" in config.conditions:
            assert q2_query_rows is not None
            if idlekv_t_repair_s is None:
                raise ValueError(
                    "PageSummary-Quest-inspired requires IdleKV in conditions for per-example T_repair."
                )
            budget_s = (
                float(config.tm_budget_absolute_s)
                if float(config.tm_budget_absolute_s) > 0.0
                else float(idlekv_t_repair_s) * float(config.tm_budget_multiplier)
            )
            score_start = time.perf_counter()
            page_partial_scores, page_info = score_evicted_positions_page_summary(
                query_rows=q2_query_rows,
                evicted_cache=base_partition.evicted,
                active_cache=base_partition.compressed,
                pooling=config.pooling,
                wallclock_deadline_s=budget_s,
                chunk_size=int(config.page_summary_chunk_size),
                precomputed_evicted_layer_keys=precomputed_evicted_keys,
                precomputed_active_layer_keys=precomputed_active_keys,
            )
            score_s = time.perf_counter() - score_start
            # Build score map for selection. Stage-1 outputs un-softmaxed
            # pooled logits (O(1-10)); Stage-2 outputs post-softmax
            # attention mass (O(1e-4)). They are NOT comparable in the
            # same dict. Earlier code ``dict(stage1).update(stage2)``
            # silently demoted Stage-2-scored chunks below Stage-1-only
            # chunks, so the selector picked the chunks Stage 1 ranked
            # second-best, third-best, ... -- actively avoiding the best
            # chunk. Fix: use Stage-2 scores ONLY when any positions were
            # Stage-2-scored; fall back to Stage-1 ranking only when
            # Stage 2 scored nothing.
            if page_partial_scores:
                page_scores_for_select = dict(page_partial_scores)
            else:
                page_scores_for_select = dict(page_info.get("stage1_only_scores", {}))
            select_start = time.perf_counter()
            # Use the same burst-pack selection as IdleKV but over the
            # page-summary partial-scored set so direct apples-to-apples
            # vs RepairKV.
            page_positions = select_idlekv_positions(
                evicted_positions=evicted_positions,
                q2_scores=page_scores_for_select,
                turn_n_scores=keep_plan.importance_scores,
                k=k_int,
                left=config.burst_left,
                right=config.burst_right,
            )
            select_s = time.perf_counter() - select_start
            page_cache, page_timing = _restore_positions(
                active_cache=base_partition.compressed,
                evicted_cache=base_partition.evicted,
                selected_positions=page_positions,
            )
            page_output, page_score, page_generation_s = _run_condition(
                model=model,
                tokenizer=tokenizer,
                prepared=split.q2_prepared,
                cache=page_cache,
            )
            row.update(
                {
                    "page_summary_score": round(page_score, 6),
                    "page_summary_output": page_output,
                    "page_summary_generation_s": round(page_generation_s, 6),
                    "page_summary_score_s": round(score_s, 6),
                    "page_summary_selection_s": round(select_s, 6),
                    "page_summary_transfer_ms": round(page_timing["transfer_ms"], 6),
                    "page_summary_inject_ms": round(page_timing["inject_ms"], 6),
                    "page_summary_restored_count": int(page_timing["restored_count"]),
                    "page_summary_t_repair_s": round(budget_s, 6),
                    "page_summary_chunks_visited": int(page_info["chunks_visited"]),
                    "page_summary_chunks_total": int(page_info["chunks_total"]),
                    "page_summary_positions_scored": int(page_info["positions_scored"]),
                    "page_summary_positions_total": int(page_info["positions_total"]),
                    "page_summary_stage1_ms": round(float(page_info["stage1_elapsed_s"]) * 1000.0, 6),
                    "page_summary_stage2_ms": round(float(page_info["stage2_elapsed_s"]) * 1000.0, 6),
                    "page_summary_cap_fired": bool(page_info["cap_fired"]),
                    "page_summary_chunk_size": int(config.page_summary_chunk_size),
                    "page_summary_selected_positions": page_positions,
                    "page_summary_overlap_fraction": round(
                        _overlap_fraction(page_positions, q2_relevant_positions), 6,
                    ),
                }
            )
        # ---- end Phase 18 W1 conditions ------------------------------------

        if "ContrastiveQ-K" in config.conditions:
            assert wrong_q_scores is not None
            select_start = time.perf_counter()
            contrastive_scores = contrastive_position_scores(
                evicted_positions,
                positive_scores=q2_scores,
                negative_scores=wrong_q_scores,
            )
            contrastive_positions = select_idlekv_positions(
                evicted_positions=evicted_positions,
                q2_scores=contrastive_scores,
                turn_n_scores=keep_plan.importance_scores,
                k=k_int,
                left=config.burst_left,
                right=config.burst_right,
            )
            select_s = time.perf_counter() - select_start
            repaired_cache, restore_timing = _restore_positions(
                active_cache=base_partition.compressed,
                evicted_cache=base_partition.evicted,
                selected_positions=contrastive_positions,
            )
            contrastive_output, contrastive_score, contrastive_generation_s = _run_condition(
                model=model,
                tokenizer=tokenizer,
                prepared=split.q2_prepared,
                cache=repaired_cache,
            )
            row.update(
                {
                    "contrastive_q_k_score": round(contrastive_score, 6),
                    "contrastive_q_k_output": contrastive_output,
                    "contrastive_q_k_generation_s": round(contrastive_generation_s, 6),
                    "contrastive_q_k_selection_s": round(select_s, 6),
                    "contrastive_q_k_transfer_ms": round(restore_timing["transfer_ms"], 6),
                    "contrastive_q_k_inject_ms": round(restore_timing["inject_ms"], 6),
                    "contrastive_q_k_restored_count": int(restore_timing["restored_count"]),
                    "contrastive_q_k_selected_positions": contrastive_positions,
                    "contrastive_q_k_overlap_fraction": round(
                        _overlap_fraction(contrastive_positions, q2_relevant_positions),
                        6,
                    ),
                    "contrastive_q_k_selected_overlap_fraction": round(
                        _overlap_fraction(contrastive_positions, q2_relevant_positions),
                        6,
                    ),
                    "contrastive_q_k_active_overlap_fraction": round(
                        _overlap_fraction(
                            tuple(sorted(set(base_partition.kept_context_positions) | set(contrastive_positions))),
                            q2_relevant_positions,
                        ),
                        6,
                    ),
                }
            )

        if "Random-K" in config.conditions:
            select_start = time.perf_counter()
            random_positions = select_random_positions(
                evicted_positions=evicted_positions,
                k=k_int,
                left=config.burst_left,
                right=config.burst_right,
                seed=(index + 1) * 1000 + k_int,
            )
            select_s = time.perf_counter() - select_start
            repaired_cache, restore_timing = _restore_positions(
                active_cache=base_partition.compressed,
                evicted_cache=base_partition.evicted,
                selected_positions=random_positions,
            )
            random_output, random_score, random_generation_s = _run_condition(
                model=model,
                tokenizer=tokenizer,
                prepared=split.q2_prepared,
                cache=repaired_cache,
            )
            row.update(
                {
                    "random_k_score": round(random_score, 6),
                    "random_k_output": random_output,
                    "random_k_generation_s": round(random_generation_s, 6),
                    "random_k_selection_s": round(select_s, 6),
                    "random_k_transfer_ms": round(restore_timing["transfer_ms"], 6),
                    "random_k_inject_ms": round(restore_timing["inject_ms"], 6),
                    "random_k_restored_count": int(restore_timing["restored_count"]),
                    "random_k_selected_positions": random_positions,
                    "random_k_overlap_fraction": round(
                        _overlap_fraction(random_positions, q2_relevant_positions),
                        6,
                    ),
                    "random_k_selected_overlap_fraction": round(
                        _overlap_fraction(random_positions, q2_relevant_positions),
                        6,
                    ),
                    "random_k_active_overlap_fraction": round(
                        _overlap_fraction(
                            tuple(sorted(set(base_partition.kept_context_positions) | set(random_positions))),
                            q2_relevant_positions,
                        ),
                        6,
                    ),
                }
            )

        if "Oldest-K" in config.conditions:
            select_start = time.perf_counter()
            oldest_positions = select_oldest_positions(
                evicted_positions=evicted_positions,
                k=k_int,
                left=config.burst_left,
                right=config.burst_right,
            )
            select_s = time.perf_counter() - select_start
            repaired_cache, restore_timing = _restore_positions(
                active_cache=base_partition.compressed,
                evicted_cache=base_partition.evicted,
                selected_positions=oldest_positions,
            )
            oldest_output, oldest_score, oldest_generation_s = _run_condition(
                model=model,
                tokenizer=tokenizer,
                prepared=split.q2_prepared,
                cache=repaired_cache,
            )
            row.update(
                {
                    "oldest_k_score": round(oldest_score, 6),
                    "oldest_k_output": oldest_output,
                    "oldest_k_generation_s": round(oldest_generation_s, 6),
                    "oldest_k_selection_s": round(select_s, 6),
                    "oldest_k_transfer_ms": round(restore_timing["transfer_ms"], 6),
                    "oldest_k_inject_ms": round(restore_timing["inject_ms"], 6),
                    "oldest_k_restored_count": int(restore_timing["restored_count"]),
                    "oldest_k_selected_positions": oldest_positions,
                    "oldest_k_overlap_fraction": round(
                        _overlap_fraction(oldest_positions, q2_relevant_positions),
                        6,
                    ),
                    "oldest_k_selected_overlap_fraction": round(
                        _overlap_fraction(oldest_positions, q2_relevant_positions),
                        6,
                    ),
                    "oldest_k_active_overlap_fraction": round(
                        _overlap_fraction(
                            tuple(sorted(set(base_partition.kept_context_positions) | set(oldest_positions))),
                            q2_relevant_positions,
                        ),
                        6,
                    ),
                }
            )

        if "ToolFile-K" in config.conditions:
            if not tool_file_q2_path:
                raise ValueError("ToolFile-K requires tool_file_q2_path.")
            select_start = time.perf_counter()
            tool_file_positions = _select_segment_positions(
                evicted_positions=evicted_positions,
                segment_token_ranges=split.q2_prepared.segment_token_ranges,
                segment_name=f"file:{tool_file_q2_path}",
                k=k_int,
            )
            select_s = time.perf_counter() - select_start
            repaired_cache, restore_timing = _restore_positions(
                active_cache=base_partition.compressed,
                evicted_cache=base_partition.evicted,
                selected_positions=tool_file_positions,
            )
            tool_file_output, tool_file_score, tool_file_generation_s = _run_condition(
                model=model,
                tokenizer=tokenizer,
                prepared=split.q2_prepared,
                cache=repaired_cache,
            )
            target_segment_name = f"file:{tool_file_q2_path}"
            file_ranges = [
                (int(start), int(end))
                for name, start, end in split.q2_prepared.segment_token_ranges
                if str(name) == target_segment_name
            ]
            selected_from_file_count = sum(
                1
                for position in tool_file_positions
                if any(start <= int(position) < end for start, end in file_ranges)
            )
            row.update(
                {
                    "tool_file_k_score": round(tool_file_score, 6),
                    "tool_file_k_output": tool_file_output,
                    "tool_file_k_generation_s": round(tool_file_generation_s, 6),
                    "tool_file_k_selection_s": round(select_s, 6),
                    "tool_file_k_transfer_ms": round(restore_timing["transfer_ms"], 6),
                    "tool_file_k_inject_ms": round(restore_timing["inject_ms"], 6),
                    "tool_file_k_restored_count": int(restore_timing["restored_count"]),
                    "tool_file_k_selected_positions": tool_file_positions,
                    "tool_file_k_q2_path": str(tool_file_q2_path),
                    "tool_file_k_selected_from_file_count": int(selected_from_file_count),
                    "tool_file_k_selected_from_file_fraction": round(
                        selected_from_file_count / max(1, len(tool_file_positions)),
                        6,
                    ),
                    "tool_file_k_budget_matched": len(tool_file_positions) == min(k_int, len(evicted_positions)),
                    "tool_file_k_overlap_fraction": round(
                        _overlap_fraction(tool_file_positions, q2_relevant_positions),
                        6,
                    ),
                    "tool_file_k_selected_overlap_fraction": round(
                        _overlap_fraction(tool_file_positions, q2_relevant_positions),
                        6,
                    ),
                    "tool_file_k_active_overlap_fraction": round(
                        _overlap_fraction(
                            tuple(sorted(set(base_partition.kept_context_positions) | set(tool_file_positions))),
                            q2_relevant_positions,
                        ),
                        6,
                    ),
                }
            )

        if "LexicalAnchor-K" in config.conditions:
            if not tool_file_q2_path:
                raise ValueError("LexicalAnchor-K requires tool_file_q2_path.")
            repair_cue = str(split.q2_prepared.example.metadata.get("repair_cue", ""))
            answer = str(split.q2_prepared.example.outputs[0])
            event_contains_q2_path = str(tool_file_q2_path) in repair_cue
            select_start = time.perf_counter()
            lexical_positions, lexical_meta = _select_lexical_anchor_positions(
                tokenizer=tokenizer,
                prepared=split.q2_prepared,
                evicted_positions=evicted_positions,
                segment_token_ranges=split.q2_prepared.segment_token_ranges,
                segment_name=f"file:{tool_file_q2_path}",
                repair_cue=repair_cue,
                answer=answer,
                k=k_int,
                left=config.burst_left,
                right=config.burst_right,
            )
            select_s = time.perf_counter() - select_start
            payload = _run_selected_position_condition(
                model=model,
                tokenizer=tokenizer,
                prepared=split.q2_prepared,
                active_cache=base_partition.compressed,
                evicted_cache=base_partition.evicted,
                base_positions=base_partition.kept_context_positions,
                selected_positions=lexical_positions,
                relevant_positions=q2_relevant_positions,
                selection_s=select_s,
                field_prefix="lexical_anchor_k",
            )
            payload.update(
                {
                    "lexical_anchor_k_q2_path": str(tool_file_q2_path),
                    "lexical_anchor_k_path_source": (
                        "event_repair_cue" if event_contains_q2_path else "q2_metadata"
                    ),
                    "lexical_anchor_k_event_contains_q2_path": bool(event_contains_q2_path),
                    "lexical_anchor_k_terms": list(lexical_meta["terms"]),
                    "lexical_anchor_k_term_count": int(lexical_meta["term_count"]),
                    "lexical_anchor_k_anchor_position_count": int(
                        lexical_meta["anchor_position_count"]
                    ),
                    "lexical_anchor_k_window_count": int(
                        lexical_meta["window_count"]
                    ),
                    "lexical_anchor_k_preferred_file_anchor_position_count": int(
                        lexical_meta["preferred_file_anchor_position_count"]
                    ),
                    "lexical_anchor_k_selected_from_file_count": int(
                        lexical_meta["selected_from_file_count"]
                    ),
                    "lexical_anchor_k_selected_from_file_fraction": float(
                        lexical_meta["selected_from_file_fraction"]
                    ),
                    "lexical_anchor_k_backfill_count": int(lexical_meta["backfill_count"]),
                    "lexical_anchor_k_answer_leak_flag": bool(lexical_meta["answer_leak_flag"]),
                    "lexical_anchor_k_answer_leak_terms": list(lexical_meta["answer_leak_terms"]),
                    "lexical_anchor_k_budget_matched": bool(lexical_meta["budget_matched"]),
                }
            )
            row.update(payload)

        if "AnchorWindow-K" in config.conditions:
            select_start = time.perf_counter()
            anchor_window_positions = _select_anchor_window_positions(
                evicted_positions=evicted_positions,
                anchor_positions=q2_relevant_positions,
                k=k_int,
            )
            select_s = time.perf_counter() - select_start
            repaired_cache, restore_timing = _restore_positions(
                active_cache=base_partition.compressed,
                evicted_cache=base_partition.evicted,
                selected_positions=anchor_window_positions,
            )
            anchor_window_output, anchor_window_score, anchor_window_generation_s = _run_condition(
                model=model,
                tokenizer=tokenizer,
                prepared=split.q2_prepared,
                cache=repaired_cache,
            )
            row.update(
                {
                    "anchor_window_k_score": round(anchor_window_score, 6),
                    "anchor_window_k_output": anchor_window_output,
                    "anchor_window_k_generation_s": round(anchor_window_generation_s, 6),
                    "anchor_window_k_selection_s": round(select_s, 6),
                    "anchor_window_k_transfer_ms": round(restore_timing["transfer_ms"], 6),
                    "anchor_window_k_inject_ms": round(restore_timing["inject_ms"], 6),
                    "anchor_window_k_restored_count": int(restore_timing["restored_count"]),
                    "anchor_window_k_selected_positions": anchor_window_positions,
                    "anchor_window_k_budget_matched": len(anchor_window_positions) == min(k_int, len(evicted_positions)),
                    "anchor_window_k_overlap_fraction": round(
                        _overlap_fraction(anchor_window_positions, q2_relevant_positions),
                        6,
                    ),
                    "anchor_window_k_selected_overlap_fraction": round(
                        _overlap_fraction(anchor_window_positions, q2_relevant_positions),
                        6,
                    ),
                    "anchor_window_k_active_overlap_fraction": round(
                        _overlap_fraction(
                            tuple(sorted(set(base_partition.kept_context_positions) | set(anchor_window_positions))),
                            q2_relevant_positions,
                        ),
                        6,
                    ),
                }
            )

        if "Oracle-K" in config.conditions:
            if config.oracle_mode == "gold_spans":
                assert gold_span_oracle_candidates is not None
                oracle_candidate = _choose_gold_span_oracle_candidate(
                    candidates=gold_span_oracle_candidates,
                    k=k_int,
                )
                oracle_positions = list(oracle_candidate["positions"])
                select_s = gold_span_oracle_search_s
                oracle_output = str(oracle_candidate["output"])
                oracle_score = float(oracle_candidate["score"])
                oracle_generation_s = float(oracle_candidate["generation_s"])
                restore_timing = dict(oracle_candidate["restore_timing"])
            else:
                select_start = time.perf_counter()
                oracle_positions = select_oracle_positions(
                    evicted_positions=evicted_positions,
                    relevant_positions=q2_relevant_positions,
                    relevant_position_groups=None,
                    q2_scores=q2_scores,
                    turn_n_scores=keep_plan.importance_scores,
                    k=k_int,
                    left=config.burst_left,
                    right=config.burst_right,
                )
                select_s = time.perf_counter() - select_start
                repaired_cache, restore_timing = _restore_positions(
                    active_cache=base_partition.compressed,
                    evicted_cache=base_partition.evicted,
                    selected_positions=oracle_positions,
                )
                oracle_output, oracle_score, oracle_generation_s = _run_condition(
                    model=model,
                    tokenizer=tokenizer,
                    prepared=split.q2_prepared,
                    cache=repaired_cache,
                )
            row.update(
                {
                    "oracle_k_score": round(oracle_score, 6),
                    "oracle_k_output": oracle_output,
                    "oracle_k_generation_s": round(oracle_generation_s, 6),
                    "oracle_k_selection_s": round(select_s, 6),
                    "oracle_k_transfer_ms": round(restore_timing["transfer_ms"], 6),
                    "oracle_k_inject_ms": round(restore_timing["inject_ms"], 6),
                    "oracle_k_restored_count": int(restore_timing["restored_count"]),
                    "oracle_k_selected_positions": oracle_positions,
                    "oracle_k_overlap_fraction": round(
                        _overlap_fraction(oracle_positions, q2_relevant_positions),
                        6,
                    ),
                    "oracle_k_selected_overlap_fraction": round(
                        _overlap_fraction(oracle_positions, q2_relevant_positions),
                        6,
                    ),
                    "oracle_k_active_overlap_fraction": round(
                        _overlap_fraction(
                            tuple(sorted(set(base_partition.kept_context_positions) | set(oracle_positions))),
                            q2_relevant_positions,
                        ),
                        6,
                    ),
                }
            )

        if "refresh_k_selected_positions" in row and "idlekv_selected_positions" in row:
            row["refresh_jaccard_with_idlekv"] = round(
                _jaccard_fraction(row["refresh_k_selected_positions"], row["idlekv_selected_positions"]),
                6,
            )

        row["example_wall_s"] = round(time.perf_counter() - example_start, 6)
        rows.append(row)

    return rows


def _summarize_condition(rows: list[dict[str, Any]], score_key: str) -> dict[str, float]:
    scores = [float(row[score_key]) for row in rows if score_key in row]
    return {
        "mean_score": round(_mean(scores), 6),
        "n_examples": len(scores),
    }


def _summarize_rows_by_k(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a homogeneous row set by K and condition."""
    by_k: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_k.setdefault(int(row["k"]), []).append(row)

    summary_by_k: dict[str, Any] = {}
    for k, group in sorted(by_k.items()):
        payload: dict[str, Any] = {
            "mean_q1_score": round(_mean(row["q1_score"] for row in group), 6),
            "mean_condition_a": round(_mean(row["condition_a_score"] for row in group), 6),
            "mean_condition_b": round(_mean(row["condition_b_score"] for row in group), 6),
            "mean_b_match": round(_mean(row["b_match_score"] for row in group), 6),
            "n_examples": len(group),
        }
        if "idlekv_score" in group[0]:
            payload.update(
                {
                    "mean_idlekv": round(_mean(row["idlekv_score"] for row in group), 6),
                    "mean_selection_lift": round(_mean(float(row["idlekv_score"]) - float(row["b_match_score"]) for row in group), 6),
                    "pct_idlekv_gt_b_match": round(
                        _pct(float(row["idlekv_score"]) > float(row["b_match_score"]) for row in group),
                        6,
                    ),
                    "pct_idlekv_lt_b_match": round(
                        _pct(float(row["idlekv_score"]) < float(row["b_match_score"]) for row in group),
                        6,
                    ),
                    "mean_idlekv_overlap_fraction": round(_mean(row["idlekv_overlap_fraction"] for row in group), 6),
                    "mean_idlekv_active_overlap_fraction": round(
                        _mean(row.get("idlekv_active_overlap_fraction", row["idlekv_overlap_fraction"]) for row in group),
                        6,
                    ),
                    "mean_idlekv_repair_ms": round(
                        _mean((float(row["idlekv_selection_s"]) * 1000.0) + float(row["idlekv_transfer_ms"]) + float(row["idlekv_inject_ms"]) for row in group),
                        6,
                    ),
                }
            )
        if "idlekv_coverage_score" in group[0]:
            payload.update(
                {
                    "mean_idlekv_coverage": round(_mean(row["idlekv_coverage_score"] for row in group), 6),
                    "mean_idlekv_coverage_lift": round(
                        _mean(float(row["idlekv_coverage_score"]) - float(row["b_match_score"]) for row in group),
                        6,
                    ),
                    "pct_idlekv_coverage_gt_b_match": round(
                        _pct(float(row["idlekv_coverage_score"]) > float(row["b_match_score"]) for row in group),
                        6,
                    ),
                    "mean_idlekv_coverage_overlap_fraction": round(
                        _mean(row["idlekv_coverage_overlap_fraction"] for row in group),
                        6,
                    ),
                }
            )
        if "idlekv_mmr_score" in group[0]:
            payload.update(
                {
                    "mean_idlekv_mmr": round(_mean(row["idlekv_mmr_score"] for row in group), 6),
                    "mean_idlekv_mmr_lift": round(
                        _mean(float(row["idlekv_mmr_score"]) - float(row["b_match_score"]) for row in group),
                        6,
                    ),
                    "pct_idlekv_mmr_gt_b_match": round(
                        _pct(float(row["idlekv_mmr_score"]) > float(row["b_match_score"]) for row in group),
                        6,
                    ),
                    "mean_idlekv_mmr_overlap_fraction": round(
                        _mean(row["idlekv_mmr_overlap_fraction"] for row in group),
                        6,
                    ),
                }
            )
        if "oracle_k_score" in group[0]:
            payload.update(
                {
                    "mean_oracle_k": round(_mean(row["oracle_k_score"] for row in group), 6),
                    "mean_oracle_lift": round(_mean(float(row["oracle_k_score"]) - float(row["b_match_score"]) for row in group), 6),
                    "pct_oracle_gt_b_match": round(
                        _pct(float(row["oracle_k_score"]) > float(row["b_match_score"]) for row in group),
                        6,
                    ),
                }
            )
        if "wrong_q_k_score" in group[0]:
            payload.update(
                {
                    "mean_wrong_q_k": round(_mean(row["wrong_q_k_score"] for row in group), 6),
                    "mean_wrong_q_lift": round(_mean(float(row["wrong_q_k_score"]) - float(row["b_match_score"]) for row in group), 6),
                    "pct_wrong_q_gt_b_match": round(
                        _pct(float(row["wrong_q_k_score"]) > float(row["b_match_score"]) for row in group),
                        6,
                    ),
                }
            )
        if "stale_q_k_score" in group[0]:
            payload.update(
                {
                    "mean_stale_q_k": round(_mean(row["stale_q_k_score"] for row in group), 6),
                    "mean_stale_q_lift": round(
                        _mean(float(row["stale_q_k_score"]) - float(row["b_match_score"]) for row in group),
                        6,
                    ),
                    "pct_stale_q_gt_b_match": round(
                        _pct(float(row["stale_q_k_score"]) > float(row["b_match_score"]) for row in group),
                        6,
                    ),
                }
            )
        if "refresh_k_score" in group[0]:
            payload.update(
                {
                    "mean_refresh_k": round(_mean(row["refresh_k_score"] for row in group), 6),
                    "mean_refresh_lift": round(
                        _mean(float(row["refresh_k_score"]) - float(row["b_match_score"]) for row in group),
                        6,
                    ),
                    "pct_refresh_gt_b_match": round(
                        _pct(float(row["refresh_k_score"]) > float(row["b_match_score"]) for row in group),
                        6,
                    ),
                    "mean_refresh_overlap_fraction": round(
                        _mean(row["refresh_k_overlap_fraction"] for row in group),
                        6,
                    ),
                }
            )
        if "contrastive_q_k_score" in group[0]:
            payload.update(
                {
                    "mean_contrastive_q_k": round(_mean(row["contrastive_q_k_score"] for row in group), 6),
                    "mean_contrastive_q_lift": round(
                        _mean(float(row["contrastive_q_k_score"]) - float(row["b_match_score"]) for row in group),
                        6,
                    ),
                    "pct_contrastive_q_gt_b_match": round(
                        _pct(float(row["contrastive_q_k_score"]) > float(row["b_match_score"]) for row in group),
                        6,
                    ),
                }
            )
        if "random_k_score" in group[0]:
            payload["mean_random_k"] = round(_mean(row["random_k_score"] for row in group), 6)
        if "oldest_k_score" in group[0]:
            payload["mean_oldest_k"] = round(_mean(row["oldest_k_score"] for row in group), 6)
        if "tool_file_k_score" in group[0]:
            payload.update(
                {
                    "mean_tool_file_k": round(_mean(row["tool_file_k_score"] for row in group), 6),
                    "mean_tool_file_lift": round(
                        _mean(float(row["tool_file_k_score"]) - float(row["b_match_score"]) for row in group),
                        6,
                    ),
                }
            )
        if "file_gated_idlekv_score" in group[0]:
            payload.update(
                {
                    "mean_file_gated_idlekv": round(_mean(row["file_gated_idlekv_score"] for row in group), 6),
                    "mean_file_gated_idlekv_lift": round(
                        _mean(float(row["file_gated_idlekv_score"]) - float(row["b_match_score"]) for row in group),
                        6,
                    ),
                    "mean_file_gated_idlekv_minus_idlekv": round(
                        _mean(float(row["file_gated_idlekv_score"]) - float(row["idlekv_score"]) for row in group),
                        6,
                    )
                    if "idlekv_score" in group[0]
                    else 0.0,
                    "mean_file_gated_idlekv_selected_from_file_fraction": round(
                        _mean(row["file_gated_idlekv_selected_from_file_fraction"] for row in group),
                        6,
                    ),
                }
            )
        if "lexical_anchor_k_score" in group[0]:
            payload.update(
                {
                    "mean_lexical_anchor_k": round(_mean(row["lexical_anchor_k_score"] for row in group), 6),
                    "mean_lexical_anchor_lift": round(
                        _mean(float(row["lexical_anchor_k_score"]) - float(row["b_match_score"]) for row in group),
                        6,
                    ),
                    "mean_lexical_anchor_k_minus_idlekv": round(
                        _mean(float(row["lexical_anchor_k_score"]) - float(row["idlekv_score"]) for row in group),
                        6,
                    )
                    if "idlekv_score" in group[0]
                    else 0.0,
                    "mean_lexical_anchor_k_selected_from_file_fraction": round(
                        _mean(row["lexical_anchor_k_selected_from_file_fraction"] for row in group),
                        6,
                    ),
                }
            )
        if "anchor_window_k_score" in group[0]:
            payload.update(
                {
                    "mean_anchor_window_k": round(_mean(row["anchor_window_k_score"] for row in group), 6),
                    "mean_anchor_window_lift": round(
                        _mean(float(row["anchor_window_k_score"]) - float(row["b_match_score"]) for row in group),
                        6,
                    ),
                }
            )
        summary_by_k[f"k{k}"] = payload
    return summary_by_k


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-example rows by K, and by split when needed."""
    task_names = sorted({str(row["task"]) for row in rows})
    if len(task_names) <= 1:
        return _summarize_rows_by_k(rows)
    return {
        "overall": _summarize_rows_by_k(rows),
        "by_task": {
            task_name: _summarize_rows_by_k([row for row in rows if str(row["task"]) == task_name])
            for task_name in task_names
        },
    }


def _artifact_path(config: Phase6Config) -> Path:
    stage_dir = ensure_results_dirs(config.stage)
    k_label = "-".join(str(value) for value in config.k_values)
    condition_label = _condition_label(config.conditions)
    seed_label = f"_seed{config.dataset_seed_offset}" if int(config.dataset_seed_offset) != 0 else ""
    scoring_label = ""
    if config.query_scoring_mode != "proxy":
        scoring_label += f"_q{config.query_scoring_mode}"
    if config.oracle_mode != "burst_hindsight":
        scoring_label += f"_o{config.oracle_mode}"
    if config.wrong_query_mode != "phantom_key":
        scoring_label += f"_wq{config.wrong_query_mode}"
    if int(config.wrong_query_donor_offset) != DEFAULT_WRONG_QUERY_DONOR_OFFSET:
        scoring_label += f"_wqd{config.wrong_query_donor_offset}"
    if config.initial_compressor != "snapkv":
        scoring_label += f"_i{config.initial_compressor}"
    default_model = Path(DEFAULT_MODEL_DIR).resolve()
    configured_model = Path(config.model_dir).expanduser()
    try:
        configured_model = configured_model.resolve()
    except OSError:
        configured_model = configured_model.absolute()
    if configured_model != default_model:
        model_label = re.sub(r"[^a-z0-9]+", "", configured_model.name.lower()) or "custommodel"
        scoring_label += f"_m{model_label}"
    return stage_dir / (
        f"{config.task}_b{config.base_context_budget}_r{config.recency_window}"
        f"{seed_label}{scoring_label}_n{config.num_samples}_k{k_label}_c{condition_label}.json"
    )


def _backup_existing_artifact(path: Path) -> Path | None:
    """Copy an existing artifact aside before overwriting it."""
    if not path.exists():
        return None
    backup_path = path.with_name(f"{path.stem}.prev{path.suffix}")
    shutil.copy2(path, backup_path)
    return backup_path


def _wrong_query_ids_by_split(
    split_views,
    *,
    tokenizer,
    config: Phase6Config,
    index: int,
) -> dict[str, torch.Tensor]:
    """Build wrong-query controls for every split view in the current example."""
    if config.wrong_query_mode == "phantom_key":
        return {
            split.split_spec.name: build_mismatched_question_ids(
                base_example=split.base_example,
                split_spec=split.split_spec,
                tokenizer=tokenizer,
            )
            for split in split_views
        }
    if config.wrong_query_mode == "donor_q2":
        donor_base_example = build_base_example(
            split_spec=config.split_specs[0],
            index=int(index) + int(config.wrong_query_donor_offset),
            context_length=config.context_length,
            tokenizer=tokenizer,
            dataset_seed_offset=config.dataset_seed_offset,
        )
        donor_split_views = tuple(
            build_split_prepared_from_base_example(
                base_example=donor_base_example,
                split_spec=split_spec,
                tokenizer=tokenizer,
            )
            for split_spec in config.split_specs
        )
        return {split.split_spec.name: split.q2_prepared.question_ids for split in donor_split_views}
    raise ValueError(f"Unsupported wrong_query_mode: {config.wrong_query_mode!r}.")


def run_experiment(config: Phase6Config) -> dict[str, Any]:
    """Run one full Phase 6 experiment and persist the JSON artifact."""
    artifact_path = _artifact_path(config)
    overall_start = time.perf_counter()
    model_dir = Path(config.model_dir).expanduser()
    model = load_model(model_dir)
    tokenizer = load_tokenizer(model_dir)

    rows: list[dict[str, Any]] = []
    for index in range(int(config.num_samples)):
        base_example = build_base_example(
            split_spec=config.split_specs[0],
            index=index,
            context_length=config.context_length,
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
        wrong_q2_question_ids_by_split: dict[str, torch.Tensor] = {}
        if "WrongQ-K" in config.conditions or "ContrastiveQ-K" in config.conditions:
            wrong_q2_question_ids_by_split = _wrong_query_ids_by_split(
                split_views,
                tokenizer=tokenizer,
                config=config,
                index=index,
            )
        full_cache = build_position_tracked_cache(model, split_views[0].q1_prepared.context_ids)

        for split in split_views:
            example_rows = _run_one_split(
                model=model,
                tokenizer=tokenizer,
                config=config,
                split=split,
                full_cache=full_cache,
                index=index,
                wrong_q2_question_ids=wrong_q2_question_ids_by_split.get(split.split_spec.name),
            )
            rows.extend(example_rows)
            if not example_rows:
                continue
            first = example_rows[0]
            per_k_parts = []
            for row in sorted(example_rows, key=lambda item: int(item["k"])):
                part = f"k={row['k']}:Bm={row['b_match_score']:.3f}"
                if "idlekv_score" in row:
                    part += f"/I={row['idlekv_score']:.3f}"
                if "idlekv_coverage_score" in row:
                    part += f"/Cov={row['idlekv_coverage_score']:.3f}"
                if "idlekv_mmr_score" in row:
                    part += f"/MMR={row['idlekv_mmr_score']:.3f}"
                if "wrong_q_k_score" in row:
                    part += f"/W={row['wrong_q_k_score']:.3f}"
                if "stale_q_k_score" in row:
                    part += f"/S={row['stale_q_k_score']:.3f}"
                if "refresh_k_score" in row:
                    part += f"/F={row['refresh_k_score']:.3f}"
                if "contrastive_q_k_score" in row:
                    part += f"/C={row['contrastive_q_k_score']:.3f}"
                if "random_k_score" in row:
                    part += f"/R={row['random_k_score']:.3f}"
                if "oldest_k_score" in row:
                    part += f"/O={row['oldest_k_score']:.3f}"
                if "oracle_k_score" in row:
                    part += f"/Or={row['oracle_k_score']:.3f}"
                per_k_parts.append(part)
            print(
                f"[{first['example_id']}] "
                f"A={first['condition_a_score']:.3f} "
                f"B={first['condition_b_score']:.3f} "
                + " ".join(per_k_parts),
                flush=True,
            )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(config),
        "aggregate": summarize_rows(rows),
        "rows": rows,
        "elapsed_s": round(time.perf_counter() - overall_start, 6),
        "artifact_path": str(artifact_path),
    }
    backup_path = _backup_existing_artifact(artifact_path)
    if backup_path is not None:
        payload["previous_artifact_backup_path"] = str(backup_path)
    write_json(artifact_path, payload)
    return payload
