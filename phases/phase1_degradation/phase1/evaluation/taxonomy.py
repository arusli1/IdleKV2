"""Error taxonomy that turns wrong answers into more interpretable failure modes."""

from __future__ import annotations

import re
from collections import Counter

from ..models import SpanSurvival, TaskExample


# Maintain a stable ordering for the reporting histogram used later in evaluation.
ERROR_TYPES = ("eviction_miss", "chain_break", "partial_recall", "hallucination", "other")


def first_broken_hop(example: TaskExample, span_survival: list[SpanSurvival]) -> tuple[int, float] | None:
    """Find the earliest variable-tracking hop whose span was fully lost."""
    # Only VT tasks expose hop-level survivals that can break the whole chain.
    if example.task_family != "vt":
        return None

    survival_by_name = {span.name: span for span in span_survival}
    for hop_index in range(1, example.metadata["num_hops"] + 1):
        hop_name = f"hop_{hop_index}"
        hop_span = survival_by_name.get(hop_name)
        if hop_span is not None and hop_span.survival_fraction == 0.0:
            return hop_index, hop_span.depth_fraction
    return None


def classify_error(
    example: TaskExample,
    prediction: str,
    matched: list[str],
    span_survival: list[SpanSurvival],
) -> str | None:
    """Assign a coarse error label that explains why the sample failed."""
    if len(matched) == len(example.outputs):
        return None

    # Early-exit heuristics cover task-specific chain failures before surface checks.
    broken_hop = first_broken_hop(example, span_survival)
    if broken_hop is not None:
        # VT tasks are special: once an earlier assignment disappears, the whole
        # reasoning chain is suspect even if later spans survived.
        return "chain_break"

    if example.task_family == "niah" and 0 < len(matched) < len(example.outputs):
        # Multi-answer NIAH tasks can be partially right, which deserves its
        # own category instead of being lumped into generic wrong answers.
        return "partial_recall"

    # Eviction misses happen when a critical span is fully dropped before the answer.
    if any(span.survival_fraction == 0.0 for span in span_survival):
        # If a task-critical span vanished completely, treat the miss as an
        # eviction failure before considering output-surface heuristics.
        return "eviction_miss"

    prediction_numbers = set(re.findall(r"\d+", prediction))
    context_numbers = set(re.findall(r"\d+", example.context))
    if not prediction_numbers or prediction_numbers.isdisjoint(context_numbers):
        # A brand-new number that never appeared in context is a decent proxy
        # for hallucination in these synthetic numeric tasks.
        return "hallucination"

    return "other"


def breakdown(errors: list[str | None]) -> dict[str, float]:
    """Normalize failure counts into fractions over only the failed examples."""
    # Drop non-failures so the fractions reflect only the taxonomy spread.
    failed = [error for error in errors if error is not None]
    if not failed:
        return {error_type: 0.0 for error_type in ERROR_TYPES}
    counts = Counter(failed)
    total = len(failed)
    return {error_type: round(counts.get(error_type, 0) / total, 6) for error_type in ERROR_TYPES}
