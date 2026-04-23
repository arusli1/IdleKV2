"""Phase 4 buffer and profiling exports."""

from .eviction_buffer import BufferEntry, EvictionBuffer, SelectionStrategy, extract_recent_q_vecs
from .feasibility import DEFAULT_TOOL_CALL_DURATIONS_S, compute_feasibility_frontier, format_frontier_table
from .profiling import (
    SyntheticKVSpec,
    build_buffer_from_log_artifact,
    build_buffer_from_logs,
    profile_buffer_scoring,
    profile_cpu_to_gpu_transfer,
    profile_end_to_end_repair,
    profile_injection_attention_overhead,
)
from .quality import evaluate_selection_quality, normalize_phase3_artifact_path
from .runtime import LiveRepairFixture, build_snapkv_live_fixture

__all__ = [
    "BufferEntry",
    "EvictionBuffer",
    "SelectionStrategy",
    "SyntheticKVSpec",
    "build_buffer_from_log_artifact",
    "build_buffer_from_logs",
    "build_snapkv_live_fixture",
    "compute_feasibility_frontier",
    "DEFAULT_TOOL_CALL_DURATIONS_S",
    "evaluate_selection_quality",
    "extract_recent_q_vecs",
    "format_frontier_table",
    "LiveRepairFixture",
    "normalize_phase3_artifact_path",
    "profile_buffer_scoring",
    "profile_cpu_to_gpu_transfer",
    "profile_end_to_end_repair",
    "profile_injection_attention_overhead",
]
