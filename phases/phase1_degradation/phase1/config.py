"""Default experiment settings and static task definitions for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Global defaults used by orchestration code when a run does not override them.
# Keep these concise; they act as the "starter pack" for Phase 1 sweeps.
DEFAULT_CONTEXT_LENGTHS = [32768]
# Token budgets considered for the Phase 1 onset-of-regression sweep.
DEFAULT_BUDGETS = [16384, 8192, 4096]
# Algorithms to evaluate unless the caller supplies a custom list.
DEFAULT_ALGORITHMS = ["snapkv"]
# Default task IDs pulled from TASK_SPECS below.
# S-NIAH stays available as an opt-in spot check but is no longer part of the
# main Phase 1 sweep.
DEFAULT_TASKS = ["vt_4hop_permute", "mq_niah_4q"]


@dataclass(frozen=True)
class TaskSpec:
    """Immutable description of how a benchmark task should be generated and scored."""

    # Stable internal identifier used by config selectors.
    name: str
    # Human-facing label for reports and tables.
    display_name: str
    # High-level family key used for grouping (e.g., vt vs niah).
    family: str
    # Generation budget for the task's answer portion.
    max_new_tokens: int
    # Default sample count when the runner does not specify one.
    default_num_samples: int
    # Per-task budget defaults; often mirrors DEFAULT_BUDGETS.
    default_budgets: tuple[int, ...]
    # Task-specific generation/scoring parameters.
    params: dict[str, Any] = field(default_factory=dict)


# Concrete task catalog used by Phase 1. Each entry is a TaskSpec keyed by ID.
# Sectioned to make it easy to add new variants or adjust depths.
TASK_SPECS: dict[str, TaskSpec] = {
    # VT (variable traversal) tasks.
    "vt_2hop": TaskSpec(
        "vt_2hop",
        "VT-2hop",
        "vt",
        24,
        100,
        tuple(DEFAULT_BUDGETS),
        {"num_hops": 2, "depths": [0.12], "terminal_depth": None, "filler_kind": "prose"},
    ),
    "vt_4hop": TaskSpec(
        "vt_4hop",
        "VT-4hop",
        "vt",
        24,
        100,
        tuple(DEFAULT_BUDGETS),
        {"num_hops": 4, "depths": [0.12, 0.37, 0.62], "terminal_depth": None, "filler_kind": "prose"},
    ),
    "vt_4hop_permute": TaskSpec(
        "vt_4hop_permute",
        "VT-4hop-permute",
        "vt",
        24,
        100,
        tuple(DEFAULT_BUDGETS),
        {
            "num_hops": 4,
            "depths": [0.12, 0.37, 0.62],
            "terminal_depth": None,
            "permute": True,
            "filler_kind": "prose",
        },
    ),
    "vt_8hop_permute_div2": TaskSpec(
        "vt_8hop_permute_div2",
        "VT-8hop-permute-div2",
        "vt",
        24,
        100,
        tuple(DEFAULT_BUDGETS),
        {
            "num_hops": 8,
            "depths": [],
            "terminal_depth": None,
            "permute": True,
            "random_permute": True,
            "num_divergences": 2,
            "filler_kind": "prose",
        },
    ),
    # NIAH single-needle task.
    "s_niah": TaskSpec(
        "s_niah",
        "S-NIAH",
        "niah",
        32,
        50,
        tuple(DEFAULT_BUDGETS),
        {"mode": "single", "num_needles": 1, "depths": [0.15]},
    ),
    # NIAH multi-query tasks with increasing needle counts.
    "mq_niah_2q": TaskSpec(
        "mq_niah_2q",
        "MQ-NIAH-2q",
        "niah",
        40,
        100,
        tuple(DEFAULT_BUDGETS),
        {"mode": "multi_query", "num_needles": 2, "depths": [0.35]},
    ),
    "mq_niah_3q": TaskSpec(
        "mq_niah_3q",
        "MQ-NIAH-3q",
        "niah",
        48,
        100,
        tuple(DEFAULT_BUDGETS),
        {"mode": "multi_query", "num_needles": 3, "depths": [0.1, 0.5]},
    ),
    "mq_niah_4q": TaskSpec(
        "mq_niah_4q",
        "MQ-NIAH-4q",
        "niah",
        64,
        100,
        tuple(DEFAULT_BUDGETS),
        {"mode": "multi_query", "num_needles": 4, "depths": [0.10, 0.37, 0.63]},
    ),
    "mq_niah_6q": TaskSpec(
        "mq_niah_6q",
        "MQ-NIAH-6q",
        "niah",
        96,
        100,
        tuple(DEFAULT_BUDGETS),
        {"mode": "multi_query", "num_needles": 6, "depths": [0.08, 0.24, 0.40, 0.56, 0.72]},
    ),
    "mq_niah_8q": TaskSpec(
        "mq_niah_8q",
        "MQ-NIAH-8q",
        "niah",
        128,
        100,
        tuple(DEFAULT_BUDGETS),
        {"mode": "multi_query", "num_needles": 8, "depths": [0.06, 0.18, 0.30, 0.42, 0.54, 0.66]},
    ),
}
