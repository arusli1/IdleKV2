"""Exports for scoring, taxonomy, trace summarization, and result formatting."""

# Metric helpers that take model outputs and produce scored summaries.
from .metrics import matched_outputs, sample_score, summarize_scores

# Result builders focus on trace data, summaries, and serialization.
from .results import (
    build_condition_b_record,
    build_detailed_eviction_log,
    build_phase1_summary,
    load_trace_payload,
    summarize_trace,
    task_prefix,
    write_json,
)

# Taxonomy helpers surface structured failure breakdowns.
from .taxonomy import breakdown, classify_error, first_broken_hop

# Public API mirrors the grouped helpers above, ensuring every export remains intentional for evaluation flows.
__all__ = [
    "breakdown",
    "build_condition_b_record",
    "build_detailed_eviction_log",
    "build_phase1_summary",
    "classify_error",
    "first_broken_hop",
    "load_trace_payload",
    "matched_outputs",
    "sample_score",
    "summarize_trace",
    "summarize_scores",
    "task_prefix",
    "write_json",
]
