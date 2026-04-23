"""Selection-quality diagnostics for the current Phase 3 artifact set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .eviction_buffer import SelectionStrategy

_PHASE3_LOG_MARKER = "phase3_eviction_logs/"
_PHASE3_RAW_MARKER = "phase3_raw_examples/"
_SHARED_Q_CONTEXT_NOTE = (
    "Current Phase 3 logs capture one shared observation-window q-vector snapshot per eviction event. "
    "That means l2_norm and dot_product are constant within each single-example buffer, so the existing "
    "artifact set is not sufficient for a meaningful selector-quality comparison."
)


def _record_is_correct(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) >= 1.0
    return False


def normalize_phase3_artifact_path(path_str: str, current_root: str | Path) -> Path:
    """Map stale pre-rename Phase 3 artifact paths onto the current repo layout."""
    candidate = Path(path_str)
    if candidate.exists():
        return candidate

    current_root = Path(current_root)
    for marker in (_PHASE3_LOG_MARKER, _PHASE3_RAW_MARKER):
        if marker in path_str:
            suffix = path_str.split(marker, 1)[1]
            return current_root / suffix
    return candidate


def _iter_matching_records(
    raw_examples_dir: Path,
    *,
    task_key: str,
    method: str,
    k_budget: int,
    max_examples: int,
):
    task_dir = raw_examples_dir / task_key
    matched = 0
    if not task_dir.exists():
        return
    for record_path in sorted(task_dir.glob("ex*.json")):
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            if str(record.get("method")) != method:
                continue
            if int(record.get("k_budget", -1)) != int(k_budget):
                continue
            yield record
            matched += 1
            if matched >= max_examples:
                return


def evaluate_selection_quality(
    phase3_log_root: str | Path,
    phase3_raw_examples_dir: str | Path,
    *,
    top_k: int = 250,
    strategies: Sequence[SelectionStrategy] = ("l2_norm", "dot_product", "random", "recency_inverse"),
    task_key: str = "vt_4hop",
    method: str = "snapkv",
    k_budget: int = 512,
    max_examples: int = 50,
) -> dict[str, object]:
    """
    Summarize whether the current Phase 3 artifact set can support selector-quality evaluation.

    The current Phase 3 logs are one-eviction snapshots, so every evicted token from one example
    shares the same q-vector context. That is enough for buffer feasibility profiling but not for
    a meaningful within-example comparison of q-based ranking strategies.
    """
    log_root = Path(phase3_log_root)
    raw_root = Path(phase3_raw_examples_dir)

    matching_records = 0
    incorrect_records = 0
    missing_log_records = 0
    missing_qvec_records = 0
    reparable_records = 0

    for record in _iter_matching_records(
        raw_root,
        task_key=task_key,
        method=method,
        k_budget=k_budget,
        max_examples=max_examples,
    ) or ():
        matching_records += 1
        if _record_is_correct(record.get("correct")):
            continue
        incorrect_records += 1

        log_path = normalize_phase3_artifact_path(str(record.get("eviction_log_path", "")), log_root)
        qvec_path = normalize_phase3_artifact_path(str(record.get("q_vectors_path", "")), log_root)
        if not log_path.exists():
            missing_log_records += 1
            continue
        if not qvec_path.exists():
            missing_qvec_records += 1
            continue

        relevant_positions = [int(position) for position in record.get("task_relevant_positions", [])]
        survived_flags = [bool(flag) for flag in record.get("task_relevant_survived", [])]
        if any((not survived) for _, survived in zip(relevant_positions, survived_flags)):
            reparable_records += 1

    if matching_records == 0:
        status = "no_matching_records"
        reason = "No matching Phase 3 raw-example records were found for the requested selector-quality slice."
    elif incorrect_records == 0:
        status = "insufficient_failures"
        reason = "The requested Phase 3 slice has no incorrect examples, so selector quality cannot be measured."
    elif reparable_records == 0:
        status = "no_reparable_failures"
        reason = "There are incorrect examples, but none expose a reparable broken span in the saved metadata."
    else:
        status = "artifact_limited"
        reason = _SHARED_Q_CONTEXT_NOTE

    return {
        "status": status,
        "reason": reason,
        "top_k": int(top_k),
        "task_key": task_key,
        "method": method,
        "k_budget": int(k_budget),
        "max_examples": int(max_examples),
        "matching_records": int(matching_records),
        "incorrect_records": int(incorrect_records),
        "reparable_records": int(reparable_records),
        "missing_log_records": int(missing_log_records),
        "missing_qvec_records": int(missing_qvec_records),
        "shared_q_context_per_example": True,
        "note": _SHARED_Q_CONTEXT_NOTE,
        "by_strategy": {
            strategy: {
                "selection_precision": None,
                "correct_selections": 0,
                "total_reparable": 0,
            }
            for strategy in strategies
        },
    }


__all__ = [
    "evaluate_selection_quality",
    "normalize_phase3_artifact_path",
]
