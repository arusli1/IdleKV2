"""Low-bit KV utilities for Phase 10 precision-promotion smokes.

These helpers include a storage-level low-bit row store and simpler
quantize/dequantize utilities. They still materialize tensors back to model
dtype before attention, so they do not claim real low-bit attention latency. The
point is to test whether query-conditioned promotion of selected KV rows has a
measurable quality signal before implementing custom packed-cache kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch

from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache, to_tuple_cache


def _normalize_nbits(nbits: int) -> int:
    nbits = int(nbits)
    if nbits < 2 or nbits > 8:
        raise ValueError(f"nbits must be in [2, 8] for fake KV quantization, got {nbits}.")
    return nbits


def fake_quantize_tensor(tensor: torch.Tensor, *, nbits: int, eps: float = 1e-12) -> torch.Tensor:
    """Symmetric per-KV-row quantize/dequantize along the head dimension.

    Input tensors are expected to follow the local KV shape convention
    `[batch, kv_heads, seq, head_dim]`. Scale is computed independently for
    each `(batch, kv_head, position)` row.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor)!r}.")
    if tensor.ndim != 4:
        raise ValueError(f"KV tensors must be rank 4, got shape {tuple(tensor.shape)}.")
    nbits = _normalize_nbits(nbits)
    qmax = float((2 ** (nbits - 1)) - 1)
    source_dtype = tensor.dtype
    working = tensor.detach().to(dtype=torch.float32)
    max_abs = working.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(max_abs / qmax, min=float(eps))
    quantized = torch.clamp(torch.round(working / scale), min=-qmax, max=qmax)
    dequantized = quantized * scale
    zero_rows = max_abs <= float(eps)
    if bool(zero_rows.any()):
        dequantized = torch.where(zero_rows, torch.zeros_like(dequantized), dequantized)
    return dequantized.to(dtype=source_dtype)


def fake_quantize_cache(cache: object, *, nbits: int) -> tuple[tuple[torch.Tensor, torch.Tensor], ...] | PositionTrackedCache:
    """Fake-quantize every K/V tensor in a cache while preserving positions."""
    tuple_cache = to_tuple_cache(cache)
    quantized = tuple(
        (
            fake_quantize_tensor(key, nbits=nbits),
            fake_quantize_tensor(value, nbits=nbits),
        )
        for key, value in tuple_cache
    )
    if isinstance(cache, PositionTrackedCache):
        return PositionTrackedCache(quantized, list(cache.positions))
    return quantized


def _position_to_dense_index(positions: Sequence[int]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for dense_index, position in enumerate(positions):
        position = int(position)
        if position in mapping:
            raise ValueError(f"Duplicate absolute position in cache metadata: {position}.")
        mapping[position] = dense_index
    return mapping


def fake_quantize_positions(
    cache: PositionTrackedCache,
    *,
    quantize_positions: Iterable[int],
    nbits: int,
) -> PositionTrackedCache:
    """Fake-quantize selected absolute-position rows and leave others intact."""
    if not isinstance(cache, PositionTrackedCache):
        raise TypeError("cache must be a PositionTrackedCache.")
    position_to_dense = _position_to_dense_index(cache.positions)
    selected = sorted(dict.fromkeys(int(position) for position in quantize_positions))
    missing = [position for position in selected if position not in position_to_dense]
    if missing:
        raise ValueError(f"Quantized positions missing from cache: {missing}.")
    if not selected:
        return PositionTrackedCache(cache.kv, list(cache.positions))

    dense_indices = [position_to_dense[position] for position in selected]
    quantized_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for key, value in cache.kv:
        next_key = key.clone()
        next_value = value.clone()
        key_rows = key[:, :, dense_indices, :]
        value_rows = value[:, :, dense_indices, :]
        next_key[:, :, dense_indices, :] = fake_quantize_tensor(key_rows, nbits=nbits)
        next_value[:, :, dense_indices, :] = fake_quantize_tensor(value_rows, nbits=nbits)
        quantized_layers.append((next_key, next_value))
    return PositionTrackedCache(tuple(quantized_layers), list(cache.positions))


def promote_high_precision_rows(
    low_precision_cache: PositionTrackedCache,
    high_precision_cache: PositionTrackedCache,
    promote_positions: Iterable[int],
) -> PositionTrackedCache:
    """Replace selected low-precision rows with high-precision originals.

    `low_precision_cache` is the active dequantized cache. `high_precision_cache`
    is the side buffer or original cache containing full-precision rows. The
    promoted positions must exist in both caches.
    """
    if not isinstance(low_precision_cache, PositionTrackedCache):
        raise TypeError("low_precision_cache must be a PositionTrackedCache.")
    if not isinstance(high_precision_cache, PositionTrackedCache):
        raise TypeError("high_precision_cache must be a PositionTrackedCache.")
    if len(low_precision_cache.kv) != len(high_precision_cache.kv):
        raise ValueError(
            f"Layer count mismatch: {len(low_precision_cache.kv)} vs {len(high_precision_cache.kv)}."
        )

    low_index = _position_to_dense_index(low_precision_cache.positions)
    high_index = _position_to_dense_index(high_precision_cache.positions)
    selected = sorted(dict.fromkeys(int(position) for position in promote_positions))
    missing_low = [position for position in selected if position not in low_index]
    missing_high = [position for position in selected if position not in high_index]
    if missing_low:
        raise ValueError(f"Promoted positions missing from low-precision cache: {missing_low}.")
    if missing_high:
        raise ValueError(f"Promoted positions missing from high-precision cache: {missing_high}.")

    promoted_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx, ((low_key, low_value), (high_key, high_value)) in enumerate(
        zip(low_precision_cache.kv, high_precision_cache.kv)
    ):
        if low_key.shape != low_value.shape or high_key.shape != high_value.shape:
            raise ValueError(f"Layer {layer_idx} has mismatched key/value shapes.")
        if low_key.shape[:2] != high_key.shape[:2] or low_key.shape[3] != high_key.shape[3]:
            raise ValueError(
                "Only the sequence length may differ between low/high caches. "
                f"Layer {layer_idx} uses {tuple(low_key.shape)} vs {tuple(high_key.shape)}."
            )
        next_key = low_key.clone()
        next_value = low_value.clone()
        for position in selected:
            dst = low_index[position]
            src = high_index[position]
            next_key[:, :, dst : dst + 1, :] = high_key[:, :, src : src + 1, :].to(
                device=next_key.device,
                dtype=next_key.dtype,
            )
            next_value[:, :, dst : dst + 1, :] = high_value[:, :, src : src + 1, :].to(
                device=next_value.device,
                dtype=next_value.dtype,
            )
        promoted_layers.append((next_key, next_value))
    return PositionTrackedCache(tuple(promoted_layers), list(low_precision_cache.positions))


@dataclass(frozen=True)
class PrecisionBudget:
    """Byte accounting for a fake mixed-precision active KV cache."""

    low_precision_bits: int
    high_precision_bits: int = 16

    def __post_init__(self) -> None:
        _normalize_nbits(self.low_precision_bits)
        if self.high_precision_bits <= self.low_precision_bits:
            raise ValueError(
                "high_precision_bits should be larger than low_precision_bits: "
                f"{self.high_precision_bits} <= {self.low_precision_bits}."
            )


@dataclass(frozen=True)
class LowBitLayerRows:
    """Integer-coded low-bit rows for one KV layer."""

    key_codes: torch.Tensor
    key_scales: torch.Tensor
    value_codes: torch.Tensor
    value_scales: torch.Tensor
    key_meta: dict[str, Any] | None = None
    value_meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class LowBitRowStore:
    """Storage-level emulation of selected low-bit KV rows.

    Codes are stored as int8 tensors because PyTorch has no native int2/int4
    tensor dtype. Byte accounting below charges them at the requested packed
    bit-width plus scale metadata, so this is closer to low-bit storage than
    rounded FP tensors but still not a low-bit attention kernel.
    """

    layers: tuple[LowBitLayerRows, ...]
    positions: tuple[int, ...]
    nbits: int
    source_dtype: torch.dtype
    backend: str = "symmetric_row"
    axis_key: int = 0
    axis_value: int = 0
    group_size: int | None = None


def _normalize_row_store_backend(backend: str) -> str:
    backend = str(backend).lower().replace("-", "_")
    if backend not in {"symmetric_row", "hqq"}:
        raise ValueError(f"Unsupported low-bit row-store backend: {backend!r}.")
    return backend


def _quantize_tensor_to_codes(tensor: torch.Tensor, *, nbits: int, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor.ndim != 4:
        raise ValueError(f"KV tensors must be rank 4, got shape {tuple(tensor.shape)}.")
    nbits = _normalize_nbits(nbits)
    qmax = float((2 ** (nbits - 1)) - 1)
    working = tensor.detach().to(dtype=torch.float32)
    max_abs = working.abs().amax(dim=-1, keepdim=True)
    scales = torch.clamp(max_abs / qmax, min=float(eps))
    codes = torch.clamp(torch.round(working / scales), min=-qmax, max=qmax).to(dtype=torch.int8)
    zero_rows = max_abs <= float(eps)
    if bool(zero_rows.any()):
        codes = torch.where(zero_rows.expand_as(codes), torch.zeros_like(codes), codes)
    return codes.contiguous(), scales.contiguous()


def _dequantize_codes(codes: torch.Tensor, scales: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    return (codes.to(dtype=torch.float32) * scales.to(dtype=torch.float32)).to(dtype=dtype)


def _meta_to_device(meta: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in meta.items():
        if key == "compute_dtype":
            moved[key] = dtype
        elif isinstance(value, torch.Tensor):
            moved[key] = value.to(device=device)
        else:
            moved[key] = value
    moved["compute_dtype"] = dtype
    return moved


def _meta_to_cpu(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach().cpu().contiguous() if isinstance(value, torch.Tensor) else value
        for key, value in meta.items()
    }


def _hqq_quantize_tensor(
    tensor: torch.Tensor,
    *,
    nbits: int,
    axis: int,
    group_size: int,
    optimize: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if axis not in {0, 1}:
        raise ValueError(f"HQQ row-store axis must be 0 or 1, got {axis}.")
    if group_size > 0 and tensor.numel() % int(group_size) != 0:
        raise ValueError(
            f"HQQ group_size must divide the selected tensor size: {tensor.numel()} % {group_size} != 0."
        )
    try:
        from hqq.core.quantize import Quantizer as HQQQuantizer
    except Exception as exc:  # pragma: no cover - depends on optional local package
        raise ImportError("Install `hqq` to use backend='hqq' for low-bit row stores.") from exc

    qtensor, meta = HQQQuantizer.quantize(
        tensor.contiguous(),
        axis=int(axis),
        device=str(tensor.device),
        compute_dtype=tensor.dtype,
        nbits=int(nbits),
        group_size=int(group_size),
        optimize=bool(optimize),
        bitpack=True,
    )
    meta["compute_dtype"] = tensor.dtype
    return qtensor.detach().cpu().contiguous(), _meta_to_cpu(meta)


def _hqq_dequantize_tensor(
    qtensor: torch.Tensor,
    meta: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    try:
        from hqq.core.quantize import Quantizer as HQQQuantizer
    except Exception as exc:  # pragma: no cover - depends on optional local package
        raise ImportError("Install `hqq` to materialize backend='hqq' low-bit row stores.") from exc
    return HQQQuantizer.dequantize(
        qtensor.to(device=device).contiguous(),
        _meta_to_device(meta, device=device, dtype=dtype),
    ).to(dtype=dtype)


def quantize_position_rows(
    cache: PositionTrackedCache,
    *,
    quantize_positions: Iterable[int],
    nbits: int,
    backend: str = "symmetric_row",
    axis_key: int = 0,
    axis_value: int = 0,
    group_size: int = 64,
    hqq_optimize: bool = True,
) -> LowBitRowStore:
    """Store selected absolute-position rows as integer low-bit codes."""
    if not isinstance(cache, PositionTrackedCache):
        raise TypeError("cache must be a PositionTrackedCache.")
    nbits = _normalize_nbits(nbits)
    backend = _normalize_row_store_backend(backend)
    position_to_dense = _position_to_dense_index(cache.positions)
    selected = tuple(sorted(dict.fromkeys(int(position) for position in quantize_positions)))
    missing = [position for position in selected if position not in position_to_dense]
    if missing:
        raise ValueError(f"Quantized positions missing from cache: {missing}.")
    dense_indices = [position_to_dense[position] for position in selected]

    layers: list[LowBitLayerRows] = []
    for key, value in cache.kv:
        key_rows = key[:, :, dense_indices, :]
        value_rows = value[:, :, dense_indices, :]
        if backend == "hqq":
            key_codes, key_meta = _hqq_quantize_tensor(
                key_rows,
                nbits=nbits,
                axis=int(axis_key),
                group_size=int(group_size),
                optimize=bool(hqq_optimize),
            )
            value_codes, value_meta = _hqq_quantize_tensor(
                value_rows,
                nbits=nbits,
                axis=int(axis_value),
                group_size=int(group_size),
                optimize=bool(hqq_optimize),
            )
            key_scales = torch.tensor([], dtype=torch.float32)
            value_scales = torch.tensor([], dtype=torch.float32)
        else:
            key_codes, key_scales = _quantize_tensor_to_codes(key_rows, nbits=nbits)
            value_codes, value_scales = _quantize_tensor_to_codes(value_rows, nbits=nbits)
            key_meta = None
            value_meta = None
        layers.append(
            LowBitLayerRows(
                key_codes=key_codes.cpu().contiguous(),
                key_scales=key_scales.cpu().contiguous(),
                value_codes=value_codes.cpu().contiguous(),
                value_scales=value_scales.cpu().contiguous(),
                key_meta=key_meta,
                value_meta=value_meta,
            )
        )
    return LowBitRowStore(
        layers=tuple(layers),
        positions=selected,
        nbits=nbits,
        source_dtype=cache.dtype,
        backend=backend,
        axis_key=int(axis_key),
        axis_value=int(axis_value),
        group_size=int(group_size),
    )


def materialize_lowbit_cache(
    high_precision_cache: PositionTrackedCache,
    lowbit_store: LowBitRowStore,
    *,
    promoted_positions: Iterable[int] = (),
) -> PositionTrackedCache:
    """Materialize a cache by dequantizing low-bit rows except promotions."""
    if not isinstance(high_precision_cache, PositionTrackedCache):
        raise TypeError("high_precision_cache must be a PositionTrackedCache.")
    if len(high_precision_cache.kv) != len(lowbit_store.layers):
        raise ValueError(
            f"Layer count mismatch: {len(high_precision_cache.kv)} vs {len(lowbit_store.layers)}."
        )
    position_to_dense = _position_to_dense_index(high_precision_cache.positions)
    selected = set(int(position) for position in lowbit_store.positions)
    promoted = set(int(position) for position in promoted_positions)
    missing_selected = sorted(selected - set(position_to_dense))
    missing_promoted = sorted(promoted - set(position_to_dense))
    if missing_selected:
        raise ValueError(f"Low-bit positions missing from high-precision cache: {missing_selected}.")
    if missing_promoted:
        raise ValueError(f"Promoted positions missing from high-precision cache: {missing_promoted}.")
    low_positions_to_materialize = [position for position in lowbit_store.positions if position not in promoted]
    if not low_positions_to_materialize:
        return PositionTrackedCache(high_precision_cache.kv, list(high_precision_cache.positions))

    store_index = {position: dense_index for dense_index, position in enumerate(lowbit_store.positions)}
    store_dense_indices = [store_index[position] for position in low_positions_to_materialize]
    destination_indices = [position_to_dense[position] for position in low_positions_to_materialize]
    materialized_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for (high_key, high_value), layer_rows in zip(high_precision_cache.kv, lowbit_store.layers, strict=True):
        next_key = high_key.clone()
        next_value = high_value.clone()
        if lowbit_store.backend == "hqq":
            if layer_rows.key_meta is None or layer_rows.value_meta is None:
                raise ValueError("HQQ row store is missing quantization metadata.")
            dequant_key = _hqq_dequantize_tensor(
                layer_rows.key_codes,
                layer_rows.key_meta,
                device=next_key.device,
                dtype=next_key.dtype,
            )
            dequant_value = _hqq_dequantize_tensor(
                layer_rows.value_codes,
                layer_rows.value_meta,
                device=next_value.device,
                dtype=next_value.dtype,
            )
            key_rows = dequant_key[:, :, store_dense_indices, :]
            value_rows = dequant_value[:, :, store_dense_indices, :]
        else:
            key_rows = _dequantize_codes(
                layer_rows.key_codes[:, :, store_dense_indices, :].to(device=next_key.device),
                layer_rows.key_scales[:, :, store_dense_indices, :].to(device=next_key.device),
                dtype=next_key.dtype,
            )
            value_rows = _dequantize_codes(
                layer_rows.value_codes[:, :, store_dense_indices, :].to(device=next_value.device),
                layer_rows.value_scales[:, :, store_dense_indices, :].to(device=next_value.device),
                dtype=next_value.dtype,
            )
        next_key[:, :, destination_indices, :] = key_rows
        next_value[:, :, destination_indices, :] = value_rows
        materialized_layers.append((next_key, next_value))
    return PositionTrackedCache(tuple(materialized_layers), list(high_precision_cache.positions))


def lowbit_row_store_bytes(lowbit_store: LowBitRowStore, *, include_scales: bool = True) -> float:
    """Estimate packed bytes for an emulated low-bit row store."""
    if lowbit_store.backend == "hqq":
        total = 0.0
        for layer in lowbit_store.layers:
            total += float(layer.key_codes.numel() * layer.key_codes.element_size())
            total += float(layer.value_codes.numel() * layer.value_codes.element_size())
            if include_scales:
                for meta in (layer.key_meta, layer.value_meta):
                    if meta is None:
                        continue
                    total += float(
                        sum(value.numel() * value.element_size() for value in meta.values() if isinstance(value, torch.Tensor))
                    )
        return total

    total_code_values = 0
    total_scale_values = 0
    for layer in lowbit_store.layers:
        total_code_values += int(layer.key_codes.numel() + layer.value_codes.numel())
        total_scale_values += int(layer.key_scales.numel() + layer.value_scales.numel())
    code_bits = total_code_values * int(lowbit_store.nbits)
    scale_bytes = total_scale_values * 4 if include_scales else 0
    return float(code_bits) / 8.0 + float(scale_bytes)


def effective_kv_bytes(
    cache: object,
    *,
    budget: PrecisionBudget,
    promoted_positions: Iterable[int] = (),
    include_high_precision_side_buffer: bool = False,
) -> float:
    """Estimate active KV bytes for low-bit cache plus promoted high-bit rows.

    The return value is a byte-equivalent accounting estimate. It is valid for
    comparing low-bit operating points, not for real allocator behavior.
    """
    tuple_cache = to_tuple_cache(cache)
    seq_len = int(tuple_cache[0][0].shape[2])
    positions = list(cache.positions) if isinstance(cache, PositionTrackedCache) else list(range(seq_len))
    position_set = set(int(position) for position in positions)
    promoted = sorted(dict.fromkeys(int(position) for position in promoted_positions))
    missing = [position for position in promoted if position not in position_set]
    if missing:
        raise ValueError(f"Promoted positions missing from cache: {missing}.")

    total_scalars = 0
    scalars_per_row = 0
    for key, value in tuple_cache:
        total_scalars += int(key.numel() + value.numel())
        scalars_per_row += int(key.shape[0] * key.shape[1] * key.shape[3] * 2)
    promoted_scalars = int(len(promoted) * scalars_per_row)
    low_bits = int(total_scalars * budget.low_precision_bits)
    promoted_extra_bits = int(promoted_scalars * (budget.high_precision_bits - budget.low_precision_bits))
    side_buffer_bits = int(total_scalars * budget.high_precision_bits) if include_high_precision_side_buffer else 0
    return float(low_bits + promoted_extra_bits + side_buffer_bits) / 8.0


def mixed_precision_kv_bytes(
    cache: object,
    *,
    budget: PrecisionBudget,
    low_precision_positions: Iterable[int],
    promoted_positions: Iterable[int] = (),
    include_high_precision_side_buffer: bool = False,
) -> float:
    """Estimate bytes when only selected rows are stored at low precision.

    Rows in ``low_precision_positions`` are counted at ``low_precision_bits``
    unless they also appear in ``promoted_positions``. All other active rows are
    counted at ``high_precision_bits``. If requested, the side buffer counts one
    high-precision copy for every low-precision row.
    """
    tuple_cache = to_tuple_cache(cache)
    seq_len = int(tuple_cache[0][0].shape[2])
    positions = list(cache.positions) if isinstance(cache, PositionTrackedCache) else list(range(seq_len))
    position_set = set(int(position) for position in positions)
    low_positions = set(int(position) for position in low_precision_positions)
    promoted = set(int(position) for position in promoted_positions)
    missing_low = sorted(low_positions - position_set)
    missing_promoted = sorted(promoted - position_set)
    if missing_low:
        raise ValueError(f"Low-precision positions missing from cache: {missing_low}.")
    if missing_promoted:
        raise ValueError(f"Promoted positions missing from cache: {missing_promoted}.")
    promoted &= low_positions

    scalars_per_row = 0
    for key, value in tuple_cache:
        scalars_per_row += int(key.shape[0] * key.shape[1] * key.shape[3] + value.shape[0] * value.shape[1] * value.shape[3])

    high_rows = len(position_set) - len(low_positions) + len(promoted)
    low_rows = len(low_positions) - len(promoted)
    active_bits = int(
        high_rows * scalars_per_row * budget.high_precision_bits
        + low_rows * scalars_per_row * budget.low_precision_bits
    )
    side_buffer_bits = (
        int(len(low_positions) * scalars_per_row * budget.high_precision_bits)
        if include_high_precision_side_buffer
        else 0
    )
    return float(active_bits + side_buffer_bits) / 8.0


def _row_float(row: dict[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def evaluate_precision_promotion_rows(
    rows: Sequence[dict[str, object]],
    *,
    min_full_score: float = 0.90,
    min_lowbit_drop: float = 0.15,
    min_idle_gain_vs_lowbit: float = 0.10,
    min_idle_margin_vs_controls: float = 0.10,
) -> list[dict[str, object]]:
    """Gate compact precision-promotion result rows.

    Each input row is one operating point with compact columns such as
    ``full_fp16``, ``lowbit_all``, ``static_mixed``, ``random_precision``,
    ``idlekv_precision``, and ``gold_precision``. This gate intentionally
    treats low-bit row-store results as appendix/future-work evidence unless
    the row explicitly marks ``real_quantized_cache`` as true.
    """
    recommendations: list[dict[str, object]] = []
    for row in rows:
        full = _row_float(row, "full_fp16")
        lowbit = _row_float(row, "lowbit_all")
        static = _row_float(row, "static_mixed", lowbit)
        random_precision = _row_float(row, "random_precision", lowbit)
        oldest_precision = _row_float(row, "oldest_precision", lowbit)
        idlekv = _row_float(row, "idlekv_precision", lowbit)
        gold = _row_float(row, "gold_precision", idlekv)
        active_bytes = _row_float(row, "active_bytes", 0.0)
        side_buffer_bytes = _row_float(row, "side_buffer_bytes", 0.0)
        real_quantized_cache = str(row.get("real_quantized_cache", "")).lower() in {"1", "true", "yes"}

        lowbit_drop = full - lowbit
        idle_gain_vs_lowbit = idlekv - lowbit
        idle_margin_vs_static = idlekv - static
        idle_margin_vs_random = idlekv - random_precision
        idle_margin_vs_oldest = idlekv - oldest_precision
        worst_control_margin = min(idle_margin_vs_static, idle_margin_vs_random, idle_margin_vs_oldest)
        gold_headroom = gold - idlekv

        full_ok = full >= min_full_score
        degradation_ok = lowbit_drop >= min_lowbit_drop
        idle_gain_ok = idle_gain_vs_lowbit >= min_idle_gain_vs_lowbit
        controls_ok = worst_control_margin >= min_idle_margin_vs_controls
        gold_ok = gold + 1e-9 >= idlekv
        bytes_ok = active_bytes > 0.0
        appendix_ok = all([full_ok, degradation_ok, idle_gain_ok, controls_ok, gold_ok, bytes_ok])
        main_ok = appendix_ok and real_quantized_cache

        if main_ok:
            action = "main_candidate_if_paired_ci_passes"
        elif appendix_ok:
            action = "appendix_quality_only"
        elif not degradation_ok:
            action = "do_not_promote_lowbit_not_degraded"
        elif not controls_ok:
            action = "do_not_promote_controls_match_or_beat_idlekv"
        elif not bytes_ok:
            action = "do_not_promote_missing_byte_accounting"
        else:
            action = "do_not_promote"

        recommendations.append(
            {
                "nbits": int(_row_float(row, "nbits", 0.0)),
                "k": int(_row_float(row, "k", _row_float(row, "promoted_rows", 0.0))),
                "full_score": round(full, 6),
                "lowbit_score": round(lowbit, 6),
                "idlekv_score": round(idlekv, 6),
                "gold_score": round(gold, 6),
                "lowbit_drop": round(lowbit_drop, 6),
                "idle_gain_vs_lowbit": round(idle_gain_vs_lowbit, 6),
                "idle_margin_vs_static": round(idle_margin_vs_static, 6),
                "idle_margin_vs_random": round(idle_margin_vs_random, 6),
                "idle_margin_vs_oldest": round(idle_margin_vs_oldest, 6),
                "gold_headroom": round(gold_headroom, 6),
                "active_bytes": round(active_bytes, 6),
                "side_buffer_bytes": round(side_buffer_bytes, 6),
                "appendix_ok": appendix_ok,
                "main_ok": main_ok,
                "action": action,
            }
        )
    return recommendations
