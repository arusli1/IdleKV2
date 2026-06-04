"""Dataclasses shared across generation, inference, tracing, and reporting."""

# This module defines the core data shapes that move through the pipeline.
# Each dataclass is a record for a phase (inputs, outputs, or metrics) with
# small serialization helpers so we can persist to JSON without ambiguity.

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RelevantSpan:
    """Character span in the synthetic prompt that we care about surviving eviction."""

    name: str
    kind: str
    char_start: int
    char_end: int
    depth_fraction: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly structure."""
        return asdict(self)


@dataclass
class PrefillSegment:
    """Contiguous chunk of the rendered prompt that should be prefed as one unit."""

    name: str
    char_start: int
    char_end: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly structure."""
        return asdict(self)


@dataclass
class TaskExample:
    """One synthetic benchmark example plus the metadata needed for attribution."""

    index: int
    task_name: str
    task_family: str
    context: str
    question: str
    answer_prefix: str
    outputs: list[str]
    max_new_tokens: int
    target_context_length: int
    relevant_spans: list[RelevantSpan]
    prefill_segments: list[PrefillSegment]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Persist the example while keeping nested dataclasses explicit."""
        payload = asdict(self)
        # Expand nested dataclasses so the JSON stays stable and explicit.
        payload["relevant_spans"] = [span.to_dict() for span in self.relevant_spans]
        # Preserve segment ordering and boundaries explicitly in the payload.
        payload["prefill_segments"] = [segment.to_dict() for segment in self.prefill_segments]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskExample":
        """Rebuild a saved example from the on-disk JSON representation."""
        return cls(
            index=payload["index"],
            task_name=payload["task_name"],
            task_family=payload["task_family"],
            context=payload["context"],
            question=payload["question"],
            answer_prefix=payload["answer_prefix"],
            outputs=[str(value) for value in payload["outputs"]],
            max_new_tokens=int(payload["max_new_tokens"]),
            target_context_length=int(payload["target_context_length"]),
            # Rehydrate nested records from the explicit list-of-dicts form.
            relevant_spans=[RelevantSpan(**span) for span in payload.get("relevant_spans", [])],
            # Prefill segments are reconstructed in order to match the prompt layout.
            prefill_segments=[PrefillSegment(**segment) for segment in payload.get("prefill_segments", [])],
            metadata=payload.get("metadata", {}),
        )


@dataclass
class SpanSurvival:
    """How much of one task-relevant span survived the compression step."""

    name: str
    kind: str
    depth_fraction: float
    survival_fraction: float
    kept_token_count: int
    total_token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly structure."""
        return asdict(self)


@dataclass
class PredictionRecord:
    """Model output plus the attribution metadata collected for one example."""

    index: int
    task_name: str
    task_family: str
    context_length: int
    algorithm: str
    budget: int | None
    condition: str
    outputs: list[str]
    prediction: str
    sample_score: float
    error_type: str | None
    matched_outputs: list[str]
    span_survival: list[SpanSurvival]
    compressed_context_length: int
    trace_path: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Persist the prediction and expand nested survival records explicitly."""
        payload = asdict(self)
        # Serialize nested SpanSurvival entries so consumers can load them directly.
        payload["span_survival"] = [span.to_dict() for span in self.span_survival]
        return payload
