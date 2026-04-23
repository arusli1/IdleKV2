"""Eviction-buffer storage and selection utilities for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from phases.phase2_kv_cache.src.kv_utils import LayerKV, PositionTrackedCache
from phases.phase3_eviction.src.eviction.base import EvictionResult

SelectionStrategy = Literal["l2_norm", "dot_product", "random", "recency_inverse"]


def _pin_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Detach one tensor to CPU pinned memory when available."""
    cpu_tensor = tensor.detach().to("cpu").contiguous()
    try:
        return cpu_tensor.pin_memory()
    except RuntimeError:
        return cpu_tensor


def _pin_single_token_layer(layer: LayerKV, *, index: int) -> LayerKV:
    """Normalize one per-token layer payload onto CPU pinned memory."""
    key, value = layer
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(f"Layer {index} tensors must have rank 4.")
    if key.shape != value.shape:
        raise ValueError(f"Layer {index} key/value shape mismatch: {tuple(key.shape)} vs {tuple(value.shape)}.")
    if int(key.shape[2]) != 1:
        raise ValueError(
            f"Buffer entries must store exactly one token per layer; layer {index} uses seq_len={int(key.shape[2])}."
        )
    return _pin_tensor(key), _pin_tensor(value)


@dataclass
class BufferEntry:
    """One evicted token's pinned KV payload plus selection metadata."""

    position: int
    kv: tuple[LayerKV, ...]
    importance_score: float
    q_vec: torch.Tensor

    def __post_init__(self) -> None:
        self.position = int(self.position)
        self.importance_score = float(self.importance_score)
        self.kv = tuple(_pin_single_token_layer(layer, index=i) for i, layer in enumerate(self.kv))
        if not isinstance(self.q_vec, torch.Tensor) or self.q_vec.ndim != 2:
            raise ValueError("q_vec must be a rank-2 tensor with shape [n_layers, head_dim].")
        self.q_vec = self.q_vec.detach().to("cpu", dtype=torch.float32).contiguous()
        if int(self.q_vec.shape[0]) != len(self.kv):
            raise ValueError(
                "q_vec layer count must match the KV layer count: "
                f"{int(self.q_vec.shape[0])} vs {len(self.kv)}."
            )


class EvictionBuffer:
    """CPU-resident buffer of evicted single-token KV payloads."""

    def __init__(
        self,
        max_tokens: int = 10_000,
        selection_strategy: SelectionStrategy = "l2_norm",
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}.")
        self.max_tokens = int(max_tokens)
        self.selection_strategy: SelectionStrategy = selection_strategy
        self._entries: dict[int, BufferEntry] = {}

    def push_from_result(self, eviction_result: EvictionResult) -> None:
        """Populate the buffer from one Phase 3 eviction result."""
        evicted = eviction_result.evicted
        q_vecs_mean = eviction_result.obs_window_q_vecs.mean(dim=1).detach().to("cpu", dtype=torch.float32).contiguous()

        for token_idx, position in enumerate(evicted.positions):
            per_layer_kv = tuple(
                (
                    key[:, :, token_idx : token_idx + 1, :],
                    value[:, :, token_idx : token_idx + 1, :],
                )
                for key, value in evicted.kv
            )
            self.push(
                BufferEntry(
                    position=int(position),
                    kv=per_layer_kv,
                    importance_score=float(eviction_result.importance_scores.get(int(position), 0.0)),
                    q_vec=q_vecs_mean,
                )
            )

    def push(self, entry: BufferEntry) -> None:
        """Add one entry and trim the buffer by importance when over capacity."""
        self._entries[int(entry.position)] = entry
        self._trim_to_capacity()

    def clear(self) -> None:
        """Drop all buffered entries."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> list[BufferEntry]:
        """Return buffered entries in insertion-independent dictionary order."""
        return list(self._entries.values())

    def _trim_to_capacity(self) -> None:
        while len(self._entries) > self.max_tokens:
            min_pos = min(
                self._entries,
                key=lambda position: (self._entries[position].importance_score, self._entries[position].position),
            )
            del self._entries[min_pos]

    def query(self, recent_q_vecs: torch.Tensor, top_k: int) -> list[BufferEntry]:
        """Score all entries and return the top-k most relevant ones."""
        if len(self._entries) == 0:
            return []

        if not isinstance(recent_q_vecs, torch.Tensor) or recent_q_vecs.ndim != 3:
            raise ValueError("recent_q_vecs must be a rank-3 tensor [n_layers, M, head_dim].")

        requested_top_k = min(max(int(top_k), 0), len(self._entries))
        if requested_top_k == 0:
            return []

        entries = list(self._entries.values())
        if self.selection_strategy == "l2_norm":
            scores = self._score_l2_norm(entries)
        elif self.selection_strategy == "dot_product":
            scores = self._score_dot_product(entries, recent_q_vecs)
        elif self.selection_strategy == "random":
            scores = self._score_random(entries)
        elif self.selection_strategy == "recency_inverse":
            scores = self._score_recency_inverse(entries)
        else:
            raise ValueError(f"Unknown strategy: {self.selection_strategy}")

        if requested_top_k == len(entries):
            ordered_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -requested_top_k)[-requested_top_k:]
            ordered_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [entries[int(index)] for index in ordered_indices]

    def _score_l2_norm(self, entries: list[BufferEntry]) -> np.ndarray:
        """Score entries by the L2 norm of the stored eviction-time query vectors."""
        return np.asarray([float(entry.q_vec.norm(p=2).item()) for entry in entries], dtype=np.float32)

    def _score_dot_product(self, entries: list[BufferEntry], recent_q_vecs: torch.Tensor) -> np.ndarray:
        """Score entries by dot product with the current recent-context summary."""
        query_mean = recent_q_vecs.detach().to("cpu", dtype=torch.float32).mean(dim=(0, 1))
        stored_vecs = torch.stack([entry.q_vec.mean(dim=0) for entry in entries], dim=0)
        scores = torch.mv(stored_vecs, query_mean)
        return scores.numpy()

    def _score_random(self, entries: list[BufferEntry]) -> np.ndarray:
        """Random ablation baseline."""
        return np.random.random_sample(len(entries)).astype(np.float32)

    def _score_recency_inverse(self, entries: list[BufferEntry]) -> np.ndarray:
        """Prefer older positions by assigning lower absolute positions higher scores."""
        positions = np.asarray([float(entry.position) for entry in entries], dtype=np.float32)
        return 1.0 / np.maximum(positions + 1.0, 1.0)

    def to_gpu(
        self,
        entries: list[BufferEntry],
        device: str | torch.device = "cuda",
    ) -> PositionTrackedCache:
        """Move selected entries to GPU and return them as a tracked cache fragment."""
        if not entries:
            raise ValueError("to_gpu requires at least one selected entry.")

        target_device = torch.device(device)
        positions = [entry.position for entry in entries]
        n_layers = len(entries[0].kv)

        merged_layers: list[LayerKV] = []
        for layer_idx in range(n_layers):
            keys = torch.cat([entry.kv[layer_idx][0] for entry in entries], dim=2).to(target_device, non_blocking=True)
            values = torch.cat([entry.kv[layer_idx][1] for entry in entries], dim=2).to(
                target_device,
                non_blocking=True,
            )
            merged_layers.append((keys, values))

        if target_device.type == "cuda":
            torch.cuda.synchronize()
        return PositionTrackedCache(tuple(merged_layers), positions)


def extract_recent_q_vecs(active_cache: PositionTrackedCache, m: int = 64) -> torch.Tensor:
    """Extract a recent key-vector proxy for the active cache's current focus."""
    if not isinstance(active_cache, PositionTrackedCache):
        raise TypeError(f"extract_recent_q_vecs expects PositionTrackedCache, got {type(active_cache)!r}.")
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}.")

    recent_vectors: list[torch.Tensor] = []
    for key, _ in active_cache.kv:
        seq_len = int(key.shape[2])
        start = max(0, seq_len - int(m))
        recent_key = key[0, :, start:, :].mean(dim=0)
        recent_vectors.append(recent_key.detach().to("cpu", dtype=torch.float32).contiguous())
    return torch.stack(recent_vectors, dim=0)


__all__ = [
    "BufferEntry",
    "EvictionBuffer",
    "SelectionStrategy",
    "extract_recent_q_vecs",
]
