"""StreamingLLM structural baseline."""

from __future__ import annotations

import torch

from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache
from .base import EvictionPolicy, EvictionResult, make_placeholder_obs_q_vecs


class StreamingLLM(EvictionPolicy):
    """Keep sink tokens and a trailing recency window, discard the middle."""

    def evict(
        self,
        full_cache: PositionTrackedCache,
        k_budget: int,
        obs_window: torch.Tensor | None = None,
    ) -> EvictionResult:
        tracked_cache = self._require_tracked_cache(full_cache)
        keep_indices = self._structural_keep_indices(seq_len=len(tracked_cache), k_budget=k_budget)
        importance = torch.zeros(len(tracked_cache), device=tracked_cache.device, dtype=torch.float32)
        if keep_indices:
            importance[torch.tensor(keep_indices, device=tracked_cache.device, dtype=torch.long)] = 1.0
        return self._finalize_result(
            full_cache=tracked_cache,
            keep_indices=keep_indices,
            importance=importance,
            obs_window_q_vecs=make_placeholder_obs_q_vecs(tracked_cache),
        )


__all__ = ["StreamingLLM"]
