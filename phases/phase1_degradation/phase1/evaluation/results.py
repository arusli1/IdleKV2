"""Trace summarization and JSON formatting for the final Phase 1 outputs."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from ..helpers import ensure_parent
from ..models import PredictionRecord, TaskExample
from ..task_registry import get_task_spec
from .taxonomy import breakdown, first_broken_hop


def write_json(path: Path, payload: Any) -> None:
    """Write a pretty-printed JSON artifact."""
    # Ensure the target directory exists before serializing the payload.
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)


def task_prefix(display_name: str) -> str:
    """Make a filesystem-friendly task prefix from the display name."""
    # Drop punctuation so downstream directories stay simple.
    return "".join(char for char in display_name if char.isalnum())


def load_trace_payload(path: Path) -> dict[str, Any]:
    """Load a saved `.pt` eviction trace onto CPU."""
    # Always pin to CPU-level tensors so downstream logic stays deterministic.
    return torch.load(path, map_location="cpu", weights_only=False)


def summarize_trace(trace_payload: dict[str, Any]) -> dict[str, Any]:
    """Collapse per-layer tensors into a simpler union/average view for reporting."""
    layers = trace_payload.get("layers", {})
    # Guard against traces that failed to capture layers so we still emit valid JSON.
    if not layers:
        # No per-layer data was stored, so return the empty structure downstream
        # consumers expect when traces are missing.
        return {
            "kept_positions": [],
            "evicted_positions": [],
            "token_importance_scores": {},
            "query_vectors": {},
        }

    max_len = max(int(layer["input_kv_length"]) for layer in layers.values())
    kept_union = torch.zeros(max_len, dtype=torch.bool)
    score_sums = torch.zeros(max_len, dtype=torch.float32)
    score_counts = torch.zeros(max_len, dtype=torch.float32)
    query_vectors: dict[str, torch.Tensor] = {}
    # Summaries are built by sweeping through layers in order and accumulating masks/scores.

    for layer_idx, layer in sorted(layers.items(), key=lambda item: int(item[0])):
        # `kept_union` answers the broad question "did any head in any layer
        # preserve this token position?" which is useful for summary plots.
        kept_mask = layer["kept_mask"]
        if kept_mask.ndim == 3:
            kept_any = kept_mask.squeeze(0).any(dim=0)
        elif kept_mask.ndim == 2:
            kept_any = kept_mask.any(dim=0)
        else:
            kept_any = kept_mask.bool()
        kept_union[: kept_any.shape[-1]] |= kept_any.to(torch.bool)

        # Collapse multi-head score tensors into a single per-position average for this layer.
        scores = layer["scores"]
        if scores.ndim == 3:
            score_mean = scores.squeeze(0).mean(dim=0)
        elif scores.ndim == 2:
            score_mean = scores.mean(dim=0)
        else:
            score_mean = scores
        score_sums[: score_mean.shape[-1]] += score_mean.float()
        score_counts[: score_mean.shape[-1]] += 1

        # Capture per-layer query vectors when present so analysts can revisit them later.
        if layer.get("query_vectors") is not None:
            # Keep the raw query-vector snapshots keyed by layer so the paper
            # analysis can revisit them without re-running the model.
            query_vectors[str(layer_idx)] = layer["query_vectors"].detach().cpu()

    average_scores = score_sums / torch.clamp_min(score_counts, 1.0)
    kept_positions = kept_union.nonzero(as_tuple=False).flatten().tolist()
    evicted_positions = (~kept_union).nonzero(as_tuple=False).flatten().tolist()
    token_importance_scores = {
        str(position): round(float(average_scores[position]), 6)
        for position in range(max_len)
    }
    # Final bundle mirrors the reporting schema consumed by Phase 1 tooling.
    return {
        "kept_positions": kept_positions,
        "evicted_positions": evicted_positions,
        "token_importance_scores": token_importance_scores,
        "query_vectors": query_vectors,
    }


def build_task_relevant_spans(
    *,
    example: TaskExample,
    span_token_positions: dict[str, list[int]],
    record: PredictionRecord,
) -> list[dict[str, Any]]:
    """Format per-span survival details into the JSON schema used by the reports."""
    # Map the recorded survival stats so we can look them up by span name.
    survival_by_name = {span.name: span for span in record.span_survival}
    output: list[dict[str, Any]] = []
    # Build the schema that downstream visualizations expect for each relevant span.
    for span in example.relevant_spans:
        token_positions = span_token_positions.get(span.name, [])
        survival = survival_by_name.get(span.name)
        # Record the essential metadata and survival metrics for this span.
        output.append(
            {
                "name": span.name,
                "kind": span.kind,
                "depth_fraction": span.depth_fraction,
                "token_positions": token_positions,
                "survived": bool(survival and survival.survival_fraction > 0.0),
                "survival_fraction": None if survival is None else round(survival.survival_fraction, 6),
                "metadata": span.metadata,
            }
        )
    return output


def build_condition_b_record(
    *,
    example: TaskExample,
    record: PredictionRecord,
    span_token_positions: dict[str, list[int]],
    detailed_log_path: Path,
    q_vectors_path: Path,
) -> dict[str, Any]:
    """Assemble the flattened per-example record expected by downstream analysis."""
    # Resolve the task metadata so we can show human-friendly names.
    spec = get_task_spec(example.task_name)
    # Collect the span-level survival details that will be embedded in the record.
    relevant_spans = build_task_relevant_spans(
        example=example,
        span_token_positions=span_token_positions,
        record=record,
    )
    # Identify where the dependency chain first failed so we can track VT behavior.
    broken_hop = first_broken_hop(example, record.span_survival)
    # Package every required field so the JSON mirrors the expected schema.
    return {
        "example_id": example.index + 1,
        "task": spec.display_name,
        "task_key": example.task_name,
        "k_budget": record.budget,
        "context_length": record.context_length,
        "raw_model_output": record.prediction,
        "gold_answer": ", ".join(example.outputs),
        "gold_outputs": example.outputs,
        # Single-answer tasks want a boolean, while multi-answer tasks keep the
        # fractional notion of recall.
        "correct": round(record.sample_score, 6) if len(example.outputs) > 1 else bool(record.sample_score == 1.0),
        "score_fraction": round(record.sample_score, 6),
        "matched_count": len(record.matched_outputs),
        "matched_outputs": record.matched_outputs,
        "error_type": record.error_type,
        "task_relevant_positions": [positions[0] if positions else -1 for positions in (span["token_positions"] for span in relevant_spans)],
        "task_relevant_survived": [span["survived"] for span in relevant_spans],
        "task_relevant_spans": relevant_spans,
        "compressed_context_length": record.compressed_context_length,
        "first_broken_hop": None if broken_hop is None else broken_hop[0],
        "first_broken_hop_depth": None if broken_hop is None else round(broken_hop[1], 4),
        "eviction_log_path": str(detailed_log_path),
        "q_vectors_path": str(q_vectors_path),
    }


def build_detailed_eviction_log(
    *,
    condition_record: dict[str, Any],
    trace_summary: dict[str, Any],
) -> dict[str, Any]:
    """Augment a flat condition record with the trace-derived eviction details."""
    # Start by cloning the base record before attaching trace artifacts.
    payload = dict(condition_record)
    # Inject the boolean eviction mask derived from the summarized trace.
    payload["eviction_mask"] = {
        "kept_positions": trace_summary["kept_positions"],
        "evicted_positions": trace_summary["evicted_positions"],
    }
    # Preserve the per-position importance scores alongside the original record.
    payload["token_importance_scores"] = trace_summary["token_importance_scores"]
    # Trace summary now lives alongside the flattened record for export.
    return payload


def build_phase1_summary(condition_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Group per-example records into the task-and-budget summary JSON."""
    # Group records by task name and retained budget before computing aggregates.
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in condition_records:
        grouped[row["task"]][int(row["k_budget"])].append(row)

    summary: dict[str, Any] = {}
    for task_name, budget_groups in grouped.items():
        task_summary: dict[str, Any] = {}
        for budget, rows in sorted(budget_groups.items(), key=lambda item: item[0], reverse=True):
            # Summaries are computed per task and per budget because Phase 1 is
            # explicitly about degradation as the retained KV budget shrinks.
            # Metric lists derived from the stored rows form the basis of each summary.
            scores = [float(row["score_fraction"]) for row in rows]
            # Track the error types so downstream breakdowns can count failure modes.
            failed_errors = [row["error_type"] for row in rows if row["error_type"] is not None]
            # Combine survival fractions from all relevant spans to get an average survival picture.
            relevant_survival_fractions = [
                float(span["survival_fraction"] or 0.0)
                for row in rows
                for span in row["task_relevant_spans"]
            ]
            # Build a stable key for the retained budget so summaries stay ordered.
            budget_key = f"k{budget}"
            # Base payload collects counts and survival rate for this task/budget group.
            base_payload: dict[str, Any] = {
                "num_examples": len(rows),
                "error_breakdown": breakdown(failed_errors),
                "eviction_survival_rate": round(
                    sum(relevant_survival_fractions) / len(relevant_survival_fractions),
                    6,
                )
                if relevant_survival_fractions
                else 0.0,
            }
            if task_name.startswith("MQ-NIAH"):
                # Multi-query NIAH is fundamentally a recall task, so expose
                # recall-oriented aggregates instead of accuracy alone.
                matched_counts = [int(row["matched_count"]) for row in rows]
                base_payload["mean_recall"] = round(sum(matched_counts) / len(matched_counts), 6)
                base_payload["mean_recall_fraction"] = round(sum(scores) / len(scores), 6)
                base_payload["full_recall_rate"] = round(sum(score == 1.0 for score in scores) / len(scores), 6)
                base_payload["std_recall_fraction"] = round(statistics.pstdev(scores), 6) if len(scores) > 1 else 0.0
            else:
                # Single-answer tasks report ordinary accuracy-style metrics.
                base_payload["accuracy"] = round(sum(scores) / len(scores), 6)
                base_payload["std_accuracy"] = round(statistics.pstdev(scores), 6) if len(scores) > 1 else 0.0
                if task_name.startswith("VT"):
                    # VT tasks also track where the dependency chain first snapped.
                    broken_hops = [row["first_broken_hop"] for row in rows if row["first_broken_hop"] is not None]
                    if broken_hops:
                        hop_counts = {f"hop_{hop}": broken_hops.count(hop) for hop in sorted(set(broken_hops))}
                        total_broken = sum(hop_counts.values())
                        base_payload["chain_break_hop_distribution"] = {
                            hop: round(count / total_broken, 6) for hop, count in hop_counts.items()
                        }
            # Record the aggregated metrics for this budget level.
            task_summary[budget_key] = base_payload
        # Store the per-task summary organized by budget keys.
        summary[task_name] = task_summary
    return summary
