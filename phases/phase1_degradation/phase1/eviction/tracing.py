"""Trace capture for per-layer eviction decisions during compressed prefill."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class LayerTrace:
    """Recorded eviction tensors for one transformer layer.

    This is a lightweight, CPU-resident snapshot of the tensors needed to
    analyze which tokens were kept or evicted for a single layer.
    """

    layer_idx: int
    scores: torch.Tensor
    kept_mask: torch.Tensor
    kept_indices: torch.Tensor
    query_vectors: torch.Tensor | None
    input_kv_length: int
    kept_kv_length: int


@dataclass
class EvictionTraceRecorder:
    """Collect layer traces in memory and flush them to a `.pt` file at the end.

    The recorder holds per-layer tensor snapshots plus run metadata, and emits
    a torch-serialized payload for offline inspection or debugging.
    """

    trace_path: Path
    algorithm: str
    budget: int
    context_length: int
    sample_index: int
    layers: dict[int, LayerTrace] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def record(
        self,
        *,
        layer_idx: int,
        scores: torch.Tensor,
        kept_mask: torch.Tensor,
        kept_indices: torch.Tensor,
        query_vectors: torch.Tensor | None,
        input_kv_length: int,
        kept_kv_length: int,
    ) -> None:
        """Store a CPU copy of one layer's scores, masks, and query vectors."""
        # Record flow: normalize to CPU tensors so the trace is device-agnostic.
        self.layers[layer_idx] = LayerTrace(
            layer_idx=layer_idx,
            scores=scores.detach().cpu(),
            kept_mask=kept_mask.detach().cpu(),
            kept_indices=kept_indices.detach().cpu(),
            query_vectors=None if query_vectors is None else query_vectors.detach().cpu(),
            input_kv_length=input_kv_length,
            kept_kv_length=kept_kv_length,
        )

    def save(self) -> None:
        """Persist the full trace payload in a torch-friendly format."""
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        # Save flow: mirror the in-memory layout so analysis code stays simple.
        payload = {
            "algorithm": self.algorithm,
            "budget": self.budget,
            "context_length": self.context_length,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
            # Save flow: per-layer records keyed by layer index.
            "layers": {
                layer_idx: {
                    "layer_idx": trace.layer_idx,
                    "scores": trace.scores,
                    "kept_mask": trace.kept_mask,
                    "kept_indices": trace.kept_indices,
                    "query_vectors": trace.query_vectors,
                    "input_kv_length": trace.input_kv_length,
                    "kept_kv_length": trace.kept_kv_length,
                }
                for layer_idx, trace in self.layers.items()
            },
        }
        torch.save(payload, self.trace_path)
