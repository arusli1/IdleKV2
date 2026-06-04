"""Small output-matching helpers used by the Phase 1 evaluator."""

from __future__ import annotations

import statistics

# Prediction-level helpers take a single model output and scan for exact matches
# against the set of expected answers from the task.


def matched_outputs(prediction: str, outputs: list[str]) -> list[str]:
    """Return every gold output string that appears verbatim in the prediction."""
    lowered = prediction.lower()
    # Build the matched list by checking each expected string in case-insensitive form.
    return [output for output in outputs if output.lower() in lowered]


def sample_score(prediction: str, outputs: list[str]) -> float:
    """Score one sample as exact-match recall over the expected outputs."""
    if not outputs:
        return 0.0
    # Count how many references appear in the prediction, then normalize.
    return len(matched_outputs(prediction, outputs)) / len(outputs)

# Aggregation helpers translate the per-sample recall values into the
# summary statistics emitted by the evaluator's reports.


def summarize_scores(scores: list[float]) -> dict[str, float]:
    """Convert raw per-sample scores into the summary stats used in reports."""
    if not scores:
        return {"mean_accuracy": 0.0, "std_accuracy": 0.0}
    return {
        # Express both metrics as percent points with four decimals for reporting.
        "mean_accuracy": round(sum(scores) / len(scores) * 100, 4),
        "std_accuracy": round(statistics.pstdev(scores) * 100, 4),
    }
