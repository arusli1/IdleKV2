"""SnapKV eviction policies."""

from __future__ import annotations

import math

import torch

from .._repo import REPO_ROOT as _REPO_ROOT  # noqa: F401
from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache, to_dynamic_cache, to_tuple_cache
from .base import EvictionPolicy, EvictionResult


class SnapKV(EvictionPolicy):
    """Observation-window scoring over a full tracked KV cache."""

    def __init__(
        self,
        *,
        obs_window_size: int = 32,
        sink_size: int = 4,
        recency_window: int = 1024,
        pooling: str = "max",
    ) -> None:
        super().__init__(sink_size=sink_size, recency_window=recency_window)
        if obs_window_size <= 0:
            raise ValueError(f"obs_window_size must be positive, got {obs_window_size}.")
        if pooling not in {"max", "mean"}:
            raise ValueError(f"pooling must be 'max' or 'mean', got {pooling!r}.")
        self.obs_window_size = int(obs_window_size)
        self.pooling = pooling

    def evict(
        self,
        full_cache: PositionTrackedCache,
        k_budget: int,
        obs_window: torch.Tensor | None = None,
    ) -> EvictionResult:
        tracked_cache, importance, obs_window_q_vecs = self.prepare_eviction_inputs(full_cache, obs_window=obs_window)
        return self.evict_from_precomputed(
            full_cache=tracked_cache,
            k_budget=k_budget,
            importance=importance,
            obs_window_q_vecs=obs_window_q_vecs,
        )

    def prepare_eviction_inputs(
        self,
        full_cache: PositionTrackedCache,
        *,
        obs_window: torch.Tensor | None = None,
    ) -> tuple[PositionTrackedCache, torch.Tensor, torch.Tensor]:
        """Compute reusable eviction inputs once for a fixed cache/query pair."""
        del obs_window
        tracked_cache = self._require_tracked_cache(full_cache)
        importance = self._score_tokens(tracked_cache)
        obs_window_q_vecs = self._extract_obs_q_vecs(tracked_cache)
        return tracked_cache, importance, obs_window_q_vecs

    def evict_from_precomputed(
        self,
        *,
        full_cache: PositionTrackedCache,
        k_budget: int,
        importance: torch.Tensor,
        obs_window_q_vecs: torch.Tensor,
    ) -> EvictionResult:
        """Materialize one eviction result from precomputed scores."""
        return self._evict_with_scores(
            full_cache=full_cache,
            k_budget=k_budget,
            importance=importance,
            obs_window_q_vecs=obs_window_q_vecs,
        )

    def _pool_scores(self, scores: torch.Tensor) -> torch.Tensor:
        if self.pooling == "max":
            return scores.amax(dim=2).mean(dim=(0, 1))
        return scores.mean(dim=(0, 1, 2))

    def _score_tokens(self, full_cache: PositionTrackedCache) -> torch.Tensor:
        cache = to_tuple_cache(full_cache.kv)
        seq_len = len(full_cache)
        obs_len = min(self.obs_window_size, seq_len)
        obs_start = seq_len - obs_len
        layer_scores: list[torch.Tensor] = []

        for key, _ in cache:
            key_float = key.detach().to(dtype=torch.float32)
            obs_rows = key_float[:, :, obs_start:, :]
            scores = torch.matmul(obs_rows, key_float.transpose(-2, -1)) / math.sqrt(key_float.shape[-1])
            scores = torch.softmax(scores, dim=-1)
            layer_scores.append(self._pool_scores(scores))

        return torch.stack(layer_scores, dim=0).mean(dim=0)

    def _extract_obs_q_vecs(self, full_cache: PositionTrackedCache) -> torch.Tensor:
        cache = to_tuple_cache(full_cache.kv)
        seq_len = len(full_cache)
        obs_len = min(self.obs_window_size, seq_len)
        obs_start = seq_len - obs_len
        obs_vecs: list[torch.Tensor] = []

        for key, _ in cache:
            obs_vecs.append(key[0, :, obs_start:, :].detach().to(dtype=torch.float32).mean(dim=0).cpu())

        return torch.stack(obs_vecs, dim=0)

    def _evict_with_scores(
        self,
        *,
        full_cache: PositionTrackedCache,
        k_budget: int,
        importance: torch.Tensor,
        obs_window_q_vecs: torch.Tensor,
    ) -> EvictionResult:
        seq_len = len(full_cache)
        budget = self._normalize_budget(seq_len, k_budget)
        importance = self._normalize_importance(importance, seq_len=seq_len, device=full_cache.device)

        mandatory_indices = self._structural_keep_indices(seq_len=seq_len, k_budget=budget)
        mandatory_set = set(mandatory_indices)
        remaining_slots = max(0, budget - len(mandatory_indices))
        candidate_indices = [index for index in range(seq_len) if index not in mandatory_set]

        selected_indices: list[int] = []
        if remaining_slots > 0 and candidate_indices:
            candidate_tensor = torch.tensor(candidate_indices, device=importance.device, dtype=torch.long)
            candidate_scores = torch.index_select(importance, 0, candidate_tensor)
            topk = min(remaining_slots, len(candidate_indices))
            topk_indices = torch.topk(candidate_scores, k=topk, largest=True, sorted=False).indices
            selected_indices = torch.index_select(candidate_tensor, 0, topk_indices).tolist()

        keep_indices = sorted(mandatory_set | set(selected_indices))
        return self._finalize_result(
            full_cache=full_cache,
            keep_indices=keep_indices,
            importance=importance,
            obs_window_q_vecs=obs_window_q_vecs,
        )


class QueryAwareSnapKV(SnapKV):
    """SnapKV variant that scores context slots against actual query tokens."""

    def __init__(
        self,
        model,
        *,
        obs_window_size: int = 32,
        sink_size: int = 4,
        recency_window: int = 1024,
        pooling: str = "max",
    ) -> None:
        super().__init__(
            obs_window_size=obs_window_size,
            sink_size=sink_size,
            recency_window=recency_window,
            pooling=pooling,
        )
        self.model = model

    def _model_device(self) -> torch.device:
        device = getattr(self.model, "device", None)
        if device is not None:
            return torch.device(device)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _prepare_query_tokens(self, obs_window: torch.Tensor) -> torch.Tensor:
        if not isinstance(obs_window, torch.Tensor):
            raise TypeError(f"obs_window must be a torch.Tensor, got {type(obs_window)!r}.")
        query_ids = obs_window.detach()
        if query_ids.ndim == 1:
            query_ids = query_ids.unsqueeze(0)
        if query_ids.ndim != 2 or int(query_ids.shape[0]) != 1:
            raise ValueError(f"obs_window must have shape [1, query_len] or [query_len], got {tuple(query_ids.shape)}.")
        if int(query_ids.shape[1]) == 0:
            raise ValueError("obs_window must contain at least one token.")
        if int(query_ids.shape[1]) > self.obs_window_size:
            query_ids = query_ids[:, -self.obs_window_size :]
        return query_ids.to(dtype=torch.long)

    def _forward_query(self, cache: PositionTrackedCache, query_ids: torch.Tensor):
        device = self._model_device()
        tracked_cache = cache if cache.device == device else cache.to_device(device)
        query_ids = query_ids.to(device)
        logical_base = int(max(tracked_cache.positions) + 1) if tracked_cache.positions else 0
        dense_base = len(tracked_cache.positions)
        query_len = int(query_ids.shape[1])

        model_cache = to_dynamic_cache(tracked_cache.kv, config=getattr(self.model, "config", None))
        kwargs = {
            "input_ids": query_ids,
            "past_key_values": model_cache,
            "position_ids": torch.arange(logical_base, logical_base + query_len, device=device).unsqueeze(0),
            "cache_position": torch.arange(dense_base, dense_base + query_len, device=device),
            "use_cache": True,
        }
        with torch.no_grad():
            try:
                return tracked_cache, self.model(**kwargs, num_logits_to_keep=1)
            except TypeError:
                try:
                    return tracked_cache, self.model(**kwargs)
                except TypeError:
                    kwargs.pop("cache_position", None)
                    return tracked_cache, self.model(**kwargs)

    def evict(
        self,
        full_cache: PositionTrackedCache,
        k_budget: int,
        obs_window: torch.Tensor | None = None,
    ) -> EvictionResult:
        tracked_cache, importance, obs_window_q_vecs = self.prepare_eviction_inputs(full_cache, obs_window=obs_window)
        return self.evict_from_precomputed(
            full_cache=tracked_cache,
            k_budget=k_budget,
            importance=importance,
            obs_window_q_vecs=obs_window_q_vecs,
        )

    def prepare_eviction_inputs(
        self,
        full_cache: PositionTrackedCache,
        *,
        obs_window: torch.Tensor | None = None,
    ) -> tuple[PositionTrackedCache, torch.Tensor, torch.Tensor]:
        if obs_window is None:
            raise ValueError(
                "QueryAwareSnapKV requires obs_window query token ids. "
                "Use standard SnapKV when the query is unavailable."
            )

        tracked_cache = self._require_tracked_cache(full_cache)
        query_ids = self._prepare_query_tokens(obs_window)
        live_cache, outputs = self._forward_query(tracked_cache, query_ids)
        extended_kv = to_tuple_cache(outputs.past_key_values)
        context_len = len(live_cache)

        layer_scores: list[torch.Tensor] = []
        obs_vecs: list[torch.Tensor] = []
        for key, _ in extended_kv:
            key_float = key.detach().to(dtype=torch.float32)
            query_rows = key_float[:, :, context_len:, :]
            context_rows = key_float[:, :, :context_len, :]
            scores = torch.matmul(query_rows, context_rows.transpose(-2, -1)) / math.sqrt(key_float.shape[-1])
            scores = torch.softmax(scores, dim=-1)
            layer_scores.append(self._pool_scores(scores))
            obs_vecs.append(query_rows[0].mean(dim=0).cpu())

        importance = torch.stack(layer_scores, dim=0).mean(dim=0)
        obs_window_q_vecs = torch.stack(obs_vecs, dim=0)
        return live_cache, importance, obs_window_q_vecs


__all__ = ["QueryAwareSnapKV", "SnapKV"]
