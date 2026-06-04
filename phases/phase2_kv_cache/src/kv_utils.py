"""KV-cache utilities for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from transformers import DynamicCache

LayerKV = tuple[torch.Tensor, torch.Tensor]
CacheTuple = tuple[LayerKV, ...]


def _require_layer_pair(layer: object, *, index: int | None = None) -> LayerKV:
    """Validate one cache layer and return it as a `(key, value)` pair."""
    label = f"layer {index}" if index is not None else "layer"
    if not isinstance(layer, (tuple, list)) or len(layer) < 2:
        raise TypeError(f"{label} must be a tuple/list with at least two entries.")
    key, value = layer[0], layer[1]
    if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must contain torch tensors.")
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(f"{label} tensors must have rank 4, got {key.ndim} and {value.ndim}.")
    if key.shape != value.shape:
        raise ValueError(f"{label} key/value shape mismatch: {tuple(key.shape)} vs {tuple(value.shape)}.")
    return key, value


def _validate_cache_tuple(cache: CacheTuple) -> CacheTuple:
    """Check that all layers share the same non-sequence dimensions."""
    if not isinstance(cache, tuple) or not cache:
        raise ValueError("KV cache must be a non-empty tuple of layers.")
    batch_size = cache[0][0].shape[0]
    num_heads = cache[0][0].shape[1]
    head_dim = cache[0][0].shape[3]
    for layer_idx, (key, value) in enumerate(cache):
        if key.shape[0] != batch_size or key.shape[1] != num_heads or key.shape[3] != head_dim:
            raise ValueError(
                "All cache layers must share batch size, KV heads, and head dimension. "
                f"Layer 0 uses {(batch_size, num_heads, head_dim)} but layer {layer_idx} uses "
                f"{(key.shape[0], key.shape[1], key.shape[3])}."
            )
        if key.shape != value.shape:
            raise ValueError(f"Layer {layer_idx} key/value shape mismatch.")
    return cache


def sequence_length(past_key_values: object) -> int:
    """Return the current dense sequence length of a cache."""
    cache = to_tuple_cache(past_key_values)
    return int(cache[0][0].shape[2])


@dataclass
class PositionTrackedCache:
    """KV cache paired with the original absolute positions of each dense slot."""

    kv: CacheTuple
    positions: list[int]

    def __post_init__(self) -> None:
        self.kv = _validate_cache_tuple(tuple(_require_layer_pair(layer, index=i) for i, layer in enumerate(self.kv)))
        self.positions = [int(position) for position in self.positions]
        if len(self.positions) != sequence_length(self.kv):
            raise ValueError(
                "Position metadata length must equal cache sequence length: "
                f"{len(self.positions)} vs {sequence_length(self.kv)}."
            )

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def device(self) -> torch.device:
        return self.kv[0][0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.kv[0][0].dtype

    def to_device(self, device: str | torch.device, *, non_blocking: bool = False) -> "PositionTrackedCache":
        """Move all layers to a device while preserving position metadata."""
        target = torch.device(device)
        if target == self.device:
            return PositionTrackedCache(self.kv, list(self.positions))
        moved = tuple(
            (
                key.to(device=target, non_blocking=non_blocking),
                value.to(device=target, non_blocking=non_blocking),
            )
            for key, value in self.kv
        )
        return PositionTrackedCache(moved, list(self.positions))


def to_tuple_cache(past_key_values: object) -> CacheTuple:
    """Normalize supported cache formats into a tuple-of-tuples representation."""
    if isinstance(past_key_values, PositionTrackedCache):
        return past_key_values.kv
    if isinstance(past_key_values, tuple):
        return _validate_cache_tuple(tuple(_require_layer_pair(layer, index=i) for i, layer in enumerate(past_key_values)))
    if isinstance(past_key_values, list):
        return _validate_cache_tuple(tuple(_require_layer_pair(layer, index=i) for i, layer in enumerate(past_key_values)))
    if isinstance(past_key_values, DynamicCache):
        return _validate_cache_tuple(
            tuple((layer.keys, layer.values) for layer in past_key_values.layers)
        )
    if hasattr(past_key_values, "layers"):
        layers = []
        for layer_idx, layer in enumerate(getattr(past_key_values, "layers")):
            if not hasattr(layer, "keys") or not hasattr(layer, "values"):
                raise TypeError(f"Cache layer {layer_idx} does not expose `.keys` and `.values`.")
            layers.append((layer.keys, layer.values))
        return _validate_cache_tuple(tuple(layers))
    raise TypeError(f"Unsupported cache type: {type(past_key_values)!r}.")


def to_dynamic_cache(past_key_values: object, *, config: object | None = None) -> DynamicCache:
    """Convert a tuple-style cache back into a `DynamicCache` for model resumes."""
    if isinstance(past_key_values, DynamicCache):
        return past_key_values
    cache = to_tuple_cache(past_key_values)
    ddp_data = tuple((key, value) for key, value in cache)
    if config is not None:
        return DynamicCache(ddp_cache_data=ddp_data, config=config)
    return DynamicCache(ddp_cache_data=ddp_data)


def _normalize_dense_indices(seq_len: int, token_indices: slice | Sequence[int] | torch.Tensor) -> tuple[slice | torch.Tensor, list[int]]:
    """Prepare a sequence selector for the KV sequence dimension."""
    if isinstance(token_indices, slice):
        dense_indices = list(range(seq_len))[token_indices]
        return token_indices, dense_indices
    if isinstance(token_indices, torch.Tensor):
        if token_indices.ndim != 1:
            raise ValueError("Tensor token indices must be one-dimensional.")
        raw_indices = [int(index) for index in token_indices.tolist()]
    else:
        raw_indices = [int(index) for index in token_indices]
    if any(index < 0 or index >= seq_len for index in raw_indices):
        raise IndexError(f"Token indices must lie in [0, {seq_len}), got {raw_indices}.")
    dense_indices = sorted(dict.fromkeys(raw_indices))
    index_tensor = torch.tensor(dense_indices, dtype=torch.long)
    return index_tensor, dense_indices


def _select_layer_tokens(layer: LayerKV, selector: slice | torch.Tensor) -> LayerKV:
    """Slice the sequence dimension of a single KV layer."""
    key, value = layer
    if isinstance(selector, slice):
        return key[:, :, selector, :].contiguous(), value[:, :, selector, :].contiguous()
    selector = selector.to(device=key.device)
    return torch.index_select(key, 2, selector), torch.index_select(value, 2, selector)


def save_kv(past_key_values: object, path: str) -> None:
    """
    Serialize a KV cache to disk, one layer per file.

    If `past_key_values` is position-tracked, the positions are saved in the
    metadata and restored by `load_kv`.
    """
    cache = to_tuple_cache(past_key_values)
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    tracked_positions = list(past_key_values.positions) if isinstance(past_key_values, PositionTrackedCache) else None

    metadata = {
        "n_layers": len(cache),
        "dtype": str(cache[0][0].dtype),
        "key_shapes": [list(key.shape) for key, _ in cache],
        "value_shapes": [list(value.shape) for _, value in cache],
        "positions": tracked_positions,
    }

    for layer_idx, (key, value) in enumerate(cache):
        torch.save(
            {
                "key": key.detach().to("cpu"),
                "value": value.detach().to("cpu"),
            },
            path_obj / f"layer_{layer_idx:03d}.pt",
        )

    torch.save(metadata, path_obj / "meta.pt")


def load_kv(path: str, device: str = "cuda") -> CacheTuple | PositionTrackedCache:
    """Load a KV cache from disk and move it to the requested device."""
    path_obj = Path(path)
    metadata = torch.load(path_obj / "meta.pt", map_location="cpu")
    layers: list[LayerKV] = []
    target_device = torch.device(device)

    for layer_idx in range(int(metadata["n_layers"])):
        payload = torch.load(path_obj / f"layer_{layer_idx:03d}.pt", map_location="cpu")
        key = payload["key"]
        value = payload["value"]
        expected_key_shape = tuple(metadata["key_shapes"][layer_idx])
        expected_value_shape = tuple(metadata["value_shapes"][layer_idx])
        if tuple(key.shape) != expected_key_shape or tuple(value.shape) != expected_value_shape:
            raise ValueError(
                f"Layer {layer_idx} shape mismatch on load: "
                f"{tuple(key.shape)} / {tuple(value.shape)} vs "
                f"{expected_key_shape} / {expected_value_shape}."
            )
        layers.append((key.to(target_device), value.to(target_device)))

    cache = _validate_cache_tuple(tuple(layers))
    positions = metadata.get("positions")
    if positions is not None:
        return PositionTrackedCache(cache, list(positions))
    return cache


def slice_kv(past_key_values: object, token_indices: slice | Sequence[int] | torch.Tensor) -> CacheTuple | PositionTrackedCache:
    """
    Extract a subset of dense sequence positions from a KV cache.

    When the input is `PositionTrackedCache`, the returned cache preserves the
    corresponding original absolute positions.
    """
    cache = to_tuple_cache(past_key_values)
    selector, dense_indices = _normalize_dense_indices(sequence_length(cache), token_indices)
    sliced = tuple(_select_layer_tokens(layer, selector) for layer in cache)
    if isinstance(past_key_values, PositionTrackedCache):
        selected_positions = [past_key_values.positions[index] for index in dense_indices]
        return PositionTrackedCache(sliced, selected_positions)
    return sliced


def _validate_concat_compatibility(cache_a: CacheTuple, cache_b: CacheTuple) -> None:
    """Check that two caches can be concatenated along the sequence axis."""
    if len(cache_a) != len(cache_b):
        raise ValueError(f"Layer count mismatch: {len(cache_a)} vs {len(cache_b)}.")
    for layer_idx, ((key_a, value_a), (key_b, value_b)) in enumerate(zip(cache_a, cache_b)):
        if key_a.device != key_b.device or value_a.device != value_b.device:
            raise ValueError(f"Layer {layer_idx} device mismatch: {key_a.device} vs {key_b.device}.")
        if key_a.dtype != key_b.dtype or value_a.dtype != value_b.dtype:
            raise ValueError(f"Layer {layer_idx} dtype mismatch: {key_a.dtype} vs {key_b.dtype}.")
        if key_a.shape[:2] != key_b.shape[:2] or key_a.shape[3] != key_b.shape[3]:
            raise ValueError(
                "Only the sequence dimension may differ when merging caches. "
                f"Layer {layer_idx} uses {tuple(key_a.shape)} vs {tuple(key_b.shape)}."
            )


def merge_kv(cache_a: object, cache_b: object) -> CacheTuple | PositionTrackedCache:
    """
    Concatenate two KV caches along the dense sequence dimension.

    If both caches are position-tracked, the position lists are concatenated in
    the same order as the tensors.
    """
    tracked_a = isinstance(cache_a, PositionTrackedCache)
    tracked_b = isinstance(cache_b, PositionTrackedCache)
    if tracked_a != tracked_b:
        raise ValueError("Either both caches or neither cache must carry position metadata.")
    left = to_tuple_cache(cache_a)
    right = to_tuple_cache(cache_b)
    _validate_concat_compatibility(left, right)
    merged = tuple(
        (
            torch.cat([key_a, key_b], dim=2),
            torch.cat([value_a, value_b], dim=2),
        )
        for (key_a, value_a), (key_b, value_b) in zip(left, right)
    )
    if tracked_a:
        return PositionTrackedCache(merged, list(cache_a.positions) + list(cache_b.positions))
    return merged


def inject_kv(
    past_key_values: PositionTrackedCache,
    new_pairs: object,
    positions: Sequence[int],
) -> PositionTrackedCache:
    """
    Insert repaired KV pairs at their original absolute sequence positions.

    `past_key_values` must already track the original positions of its dense
    cache slots. `positions` gives the original absolute positions for
    `new_pairs`.
    """
    if not isinstance(past_key_values, PositionTrackedCache):
        raise TypeError("inject_kv requires `past_key_values` to be a PositionTrackedCache.")

    new_positions = [int(position) for position in positions]
    new_cache = to_tuple_cache(new_pairs)
    if len(new_positions) != sequence_length(new_cache):
        raise ValueError(
            "Injected position count must match injected cache length: "
            f"{len(new_positions)} vs {sequence_length(new_cache)}."
        )
    if isinstance(new_pairs, PositionTrackedCache) and list(new_pairs.positions) != new_positions:
        raise ValueError("Injected position metadata does not match the supplied `positions` argument.")

    overlap = set(past_key_values.positions).intersection(new_positions)
    if overlap:
        raise ValueError(f"Injected positions overlap the active cache: {sorted(overlap)}.")

    merged = merge_kv(past_key_values.kv, new_cache)
    if isinstance(merged, PositionTrackedCache):
        raise RuntimeError("Internal merge returned an unexpected tracked cache.")
    merged_positions = list(past_key_values.positions) + new_positions
    sort_order = sorted(range(len(merged_positions)), key=merged_positions.__getitem__)
    sort_tensor = torch.tensor(sort_order, dtype=torch.long, device=merged[0][0].device)
    sorted_cache = tuple(
        (
            torch.index_select(key, 2, sort_tensor),
            torch.index_select(value, 2, sort_tensor),
        )
        for key, value in merged
    )
    sorted_positions = [merged_positions[index] for index in sort_order]
    if len(sorted_positions) != len(set(sorted_positions)):
        raise ValueError("Injected cache contains duplicate absolute positions after merge.")
    return PositionTrackedCache(sorted_cache, sorted_positions)


__all__ = [
    "CacheTuple",
    "LayerKV",
    "PositionTrackedCache",
    "inject_kv",
    "load_kv",
    "merge_kv",
    "save_kv",
    "sequence_length",
    "slice_kv",
    "to_dynamic_cache",
    "to_tuple_cache",
]
