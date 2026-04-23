"""Feasibility-frontier helpers for Phase 4 profiling outputs."""

from __future__ import annotations

from typing import Iterable

DEFAULT_TOOL_CALL_DURATIONS_S = (0.1, 0.5, 1.0, 2.0, 5.0, 8.0, 15.0, 30.0, 60.0)


def _as_int_keys(mapping: dict[object, object]) -> dict[int, object]:
    """Normalize possibly stringified numeric keys from JSON payloads."""
    return {int(key): value for key, value in mapping.items()}


def compute_feasibility_frontier(
    transfer_results: dict[object, dict[str, float]],
    scoring_results: dict[str, dict[object, dict[str, float]]],
    *,
    strategy: str = "l2_norm",
    overhead_budget_pct: float = 0.90,
    tool_call_durations_s: Iterable[float] = DEFAULT_TOOL_CALL_DURATIONS_S,
) -> dict[float, dict[str, float | int]]:
    """Map tool-call duration to the largest profiled K that fits the repair budget."""
    if not 0.0 < float(overhead_budget_pct) <= 1.0:
        raise ValueError(f"overhead_budget_pct must lie in (0, 1], got {overhead_budget_pct}.")

    normalized_transfer = _as_int_keys(dict(transfer_results))
    normalized_scoring = _as_int_keys(dict(scoring_results.get(strategy, {})))

    if normalized_scoring:
        largest_bucket = max(normalized_scoring)
        scoring_overhead_ms = max(float(normalized_scoring[largest_bucket].get("p50_ms", 0.0)), 1.0)
    else:
        scoring_overhead_ms = 1.0

    frontier: dict[float, dict[str, float | int]] = {}
    for duration_s in tool_call_durations_s:
        budget_ms = float(duration_s) * 1000.0 * float(overhead_budget_pct)
        available_transfer_ms = budget_ms - scoring_overhead_ms

        max_k = 0
        for n_tokens, timing in sorted(normalized_transfer.items()):
            if float(timing.get("p90_ms", float("inf"))) <= available_transfer_ms:
                max_k = int(n_tokens)

        frontier[float(duration_s)] = {
            "budget_ms": budget_ms,
            "scoring_overhead_ms": scoring_overhead_ms,
            "max_K": max_k,
        }

    return frontier


def format_frontier_table(frontier: dict[float, dict[str, float | int]]) -> str:
    """Render a compact markdown table from a frontier payload."""
    lines = [
        "| Tool Call (s) | Repair Budget (ms) | Max K |",
        "|---:|---:|---:|",
    ]
    for duration_s in sorted(frontier):
        row = frontier[duration_s]
        lines.append(
            f"| {duration_s:.1f} | {float(row['budget_ms']):.1f} | {int(row['max_K'])} |"
        )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_TOOL_CALL_DURATIONS_S",
    "compute_feasibility_frontier",
    "format_frontier_table",
]
