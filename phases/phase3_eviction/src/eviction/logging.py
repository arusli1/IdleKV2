"""Eviction log helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .base import EvictionResult


def log_eviction(
    result: EvictionResult,
    example_id: str,
    task: str,
    task_relevant_positions: Sequence[int],
    log_dir: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the JSON and q-vector artifacts requested by the Phase 3 spec."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    relevant_positions = [int(position) for position in task_relevant_positions]
    kept_positions = set(result.compressed.positions)
    survived = [position in kept_positions for position in relevant_positions]

    entry: dict[str, Any] = {
        "example_id": str(example_id),
        "task": str(task),
        "k_budget": len(result.compressed.positions),
        "seq_len": len(result.compressed.positions) + len(result.evicted.positions),
        "kept_positions": list(result.compressed.positions),
        "evicted_positions": list(result.evicted.positions),
        "task_relevant_positions": relevant_positions,
        "task_relevant_survived": survived,
        "task_relevant_survival_fraction": [1.0 if alive else 0.0 for alive in survived],
        "importance_scores": {str(position): float(score) for position, score in result.importance_scores.items()},
        "obs_window_q_vecs_shape": list(result.obs_window_q_vecs.shape),
    }
    if metadata:
        entry["metadata"] = dict(metadata)

    (log_path / f"{example_id}.json").write_text(
        json.dumps(entry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.save(result.obs_window_q_vecs, log_path / f"{example_id}_qvecs.pt")
    return entry


__all__ = ["log_eviction"]
