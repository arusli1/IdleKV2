"""Shared eviction abstractions and tensor utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from .._repo import REPO_ROOT as _REPO_ROOT  # noqa: F401
from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache, slice_kv, to_tuple_cache


@dataclass
class EvictionResult:
    """Common output contract for every Phase 3 eviction policy."""

    compressed: PositionTrackedCache
    evicted: PositionTrackedCache
    importance_scores: dict[int, float]
    obs_window_q_vecs: torch.Tensor


def make_placeholder_obs_q_vecs(full_cache: PositionTrackedCache, *, length: int = 1) -> torch.Tensor:
    """Create a minimal CPU tensor when a policy has no learned observation rows."""
    head_dim = int(full_cache.kv[0][0].shape[-1])
    n_layers = len(full_cache.kv)
    return torch.zeros((n_layers, max(1, int(length)), head_dim), dtype=torch.float32)


def _move_tensor_to_cpu_pinned(tensor: torch.Tensor) -> torch.Tensor:
    """Detach one tensor onto CPU pinned memory when available."""
    cpu_tensor = tensor.detach().to("cpu").contiguous()
    try:
        return cpu_tensor.pin_memory()
    except RuntimeError:
        return cpu_tensor


def _cache_to_cpu_pinned(cache: object) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Move a cache tuple onto CPU pinned memory layer by layer."""
    return tuple(
        (
            _move_tensor_to_cpu_pinned(key),
            _move_tensor_to_cpu_pinned(value),
        )
        for key, value in to_tuple_cache(cache)
    )


class EvictionPolicy(ABC):
    """Base class for policies that compress a tracked KV cache."""

    def __init__(self, *, sink_size: int = 4, recency_window: int = 1024) -> None:
        if sink_size < 0:
            raise ValueError(f"sink_size must be non-negative, got {sink_size}.")
        if recency_window < 0:
            raise ValueError(f"recency_window must be non-negative, got {recency_window}.")
        self.sink_size = int(sink_size)
        self.recency_window = int(recency_window)

    @abstractmethod
    def evict(
        self,
        full_cache: PositionTrackedCache,
        k_budget: int,
        obs_window: torch.Tensor | None = None,
    ) -> EvictionResult:
        """Return the compressed and evicted cache partitions."""

    def _require_tracked_cache(self, full_cache: object) -> PositionTrackedCache:
        if not isinstance(full_cache, PositionTrackedCache):
            raise TypeError(
                "Phase 3 eviction policies require a PositionTrackedCache from Phase 2 "
                f"so kept/evicted slots retain their original absolute positions, got {type(full_cache)!r}."
            )
        return full_cache

    def _normalize_budget(self, seq_len: int, k_budget: int) -> int:
        if k_budget < 0:
            raise ValueError(f"k_budget must be non-negative, got {k_budget}.")
        return min(int(k_budget), int(seq_len))

    def _structural_keep_indices(self, *, seq_len: int, k_budget: int) -> list[int]:
        """
        Compute mandatory sink/recency slots without overrunning the target budget.

        If the requested budget is smaller than the configured sink count, sinks
        consume the entire budget and no recency rows are kept.
        """
        budget = self._normalize_budget(seq_len, k_budget)
        if budget == 0:
            return []
        sink_count = min(seq_len, budget, self.sink_size)
        remaining_budget = budget - sink_count
        recency_count = min(self.recency_window, max(0, seq_len - sink_count), remaining_budget)
        recency_start = max(sink_count, seq_len - recency_count)
        keep_indices = sorted(set(range(sink_count)) | set(range(recency_start, seq_len)))
        return keep_indices

    def _normalize_importance(
        self,
        importance: torch.Tensor,
        *,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(importance, torch.Tensor):
            raise TypeError(f"importance must be a torch.Tensor, got {type(importance)!r}.")
        if importance.ndim != 1 or int(importance.shape[0]) != seq_len:
            raise ValueError(
                "importance must be a one-dimensional tensor with one score per dense cache slot: "
                f"expected {seq_len}, got shape {tuple(importance.shape)}."
            )
        return importance.detach().to(device=device, dtype=torch.float32).contiguous()

    def _normalize_obs_window_q_vecs(
        self,
        full_cache: PositionTrackedCache,
        obs_window_q_vecs: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(obs_window_q_vecs, torch.Tensor) or obs_window_q_vecs.ndim != 3:
            raise ValueError("obs_window_q_vecs must be a rank-3 tensor [n_layers, obs_len, head_dim].")
        expected_layers = len(full_cache.kv)
        expected_head_dim = int(full_cache.kv[0][0].shape[-1])
        if int(obs_window_q_vecs.shape[0]) != expected_layers or int(obs_window_q_vecs.shape[2]) != expected_head_dim:
            raise ValueError(
                "obs_window_q_vecs shape mismatch: "
                f"expected leading/trailing dims ({expected_layers}, *, {expected_head_dim}), "
                f"got {tuple(obs_window_q_vecs.shape)}."
            )
        return obs_window_q_vecs.detach().to("cpu", dtype=torch.float32).contiguous()

    def _finalize_result(
        self,
        *,
        full_cache: PositionTrackedCache,
        keep_indices: list[int],
        importance: torch.Tensor,
        obs_window_q_vecs: torch.Tensor,
    ) -> EvictionResult:
        seq_len = len(full_cache)
        keep_indices = sorted(dict.fromkeys(int(index) for index in keep_indices))
        if any(index < 0 or index >= seq_len for index in keep_indices):
            raise IndexError(f"keep_indices must lie in [0, {seq_len}), got {keep_indices}.")
        importance = self._normalize_importance(importance, seq_len=seq_len, device=full_cache.device)
        obs_window_q_vecs = self._normalize_obs_window_q_vecs(full_cache, obs_window_q_vecs)

        keep_set = set(keep_indices)
        evict_indices = [index for index in range(seq_len) if index not in keep_set]

        compressed = slice_kv(full_cache, keep_indices)
        evicted = slice_kv(full_cache, evict_indices)
        if not isinstance(compressed, PositionTrackedCache) or not isinstance(evicted, PositionTrackedCache):
            raise RuntimeError("Phase 2 slice_kv did not preserve position tracking.")

        evicted_cpu = PositionTrackedCache(_cache_to_cpu_pinned(evicted.kv), list(evicted.positions))
        importance_scores = {
            int(position): float(importance[dense_index].item())
            for dense_index, position in enumerate(full_cache.positions)
        }
        return EvictionResult(
            compressed=compressed,
            evicted=evicted_cpu,
            importance_scores=importance_scores,
            obs_window_q_vecs=obs_window_q_vecs,
        )


__all__ = ["EvictionPolicy", "EvictionResult", "make_placeholder_obs_q_vecs"]
