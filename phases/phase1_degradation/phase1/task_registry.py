"""Resolve task names into specs and concrete example builders.

This module keeps the lookup/dispatch logic centralized so that new task
families only require registering metadata and a builder.
"""

from __future__ import annotations

from .config import TASK_SPECS, TaskSpec
from .generators import build_niah_example, build_vt_example
from .models import TaskExample


def get_task_spec(task_name: str) -> TaskSpec:
    """Look up the static metadata for a task and fail loudly on typos."""
    # Defensive guard: avoid silently falling back when callers pass a typo.
    if task_name not in TASK_SPECS:
        raise KeyError(f"Unknown task: {task_name}")
    # Return the canonical spec object used to decide generator/parameters.
    return TASK_SPECS[task_name]


def build_task_example(
    task_name: str,
    index: int,
    target_context_length: int,
    tokenizer,
    *,
    dataset_seed_offset: int = 0,
) -> TaskExample:
    """Dispatch to the right synthetic-data generator for the requested task family."""
    # Resolve the spec once so we can both validate and route the request.
    spec = get_task_spec(task_name)
    # Family-based dispatch keeps generator implementations isolated.
    if spec.family == "vt":
        # "vt" tasks share a single builder and are differentiated by spec params.
        return build_vt_example(
            task_name,
            index,
            target_context_length,
            tokenizer,
            dataset_seed_offset=dataset_seed_offset,
            **spec.params,
        )
    if spec.family == "niah":
        # "niah" tasks have their own builder with similar call signature.
        return build_niah_example(
            task_name,
            index,
            target_context_length,
            tokenizer,
            dataset_seed_offset=dataset_seed_offset,
            **spec.params,
        )
    # Explicit error for unsupported families to surface missing registrations.
    raise ValueError(f"Unsupported task family: {spec.family}")
