"""Synthetic KV movement capacity profiling helpers.

These helpers measure the systems path that does not require answer scoring:
moving Qwen-shaped restored KV rows from host memory to GPU and reinserting
them into an active position-tracked cache.  The tensors are synthetic, but the
shape, dtype, transfer direction, and current ``inject_kv`` implementation are
real.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache, inject_kv


@dataclass(frozen=True)
class KVRuntimeSpec:
    """KV tensor shape for one model family."""

    n_layers: int = 28
    n_query_heads: int = 28
    n_kv_heads: int = 4
    head_dim: int = 128
    dtype: torch.dtype = torch.bfloat16

    @property
    def bytes_per_token(self) -> int:
        return int(
            self.n_layers
            * 2
            * self.n_kv_heads
            * self.head_dim
            * torch.empty((), dtype=self.dtype).element_size()
        )


def parse_dtype(name: str) -> torch.dtype:
    """Parse a small CLI-facing dtype vocabulary."""
    normalized = name.lower().replace("torch.", "")
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def percentile_summary(values_ms: Sequence[float]) -> dict[str, float]:
    """Return p50/p95/p99/mean for millisecond samples."""
    if not values_ms:
        raise ValueError("Cannot summarize an empty timing sample.")
    values = np.asarray([float(value) for value in values_ms], dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(np.mean(values)),
    }


def synchronize(device: str | torch.device) -> None:
    """Synchronize CUDA timings when needed."""
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.synchronize(target)


def _h2d_score_keys(
    host_chunk: torch.Tensor,
    *,
    target: torch.device,
    pin_memory: bool,
) -> torch.Tensor:
    """BF16/FP16 host -> device async copy, then on-device upcast to FP32.

    A single ``.to(device, dtype=torch.float32)`` from a pinned BF16 source
    silently demotes the H2D to a blocking copy because cudart cannot do an
    async cross-dtype copy from pinned host. We avoid that by transferring
    in source dtype first (truly async on pinned), then upcasting on-device
    where the cast is free.
    """
    non_blocking = pin_memory and target.type == "cuda"
    if host_chunk.dtype == torch.float32:
        return host_chunk.to(device=target, non_blocking=non_blocking)
    device_chunk = host_chunk.to(device=target, non_blocking=non_blocking)
    return device_chunk.float()


def _enforce_coverage(
    *,
    host_pool_tokens: int,
    candidate_tokens: int,
    allow_partial_coverage: bool,
) -> float:
    """Refuse to emit a row whose pinned host pool does not cover the full candidate set.

    Phase 17 numbers were computed with ``source_pool_chunks=1`` at very
    large ``candidate_tokens``, which silently re-read a 16K-token pool 64x
    instead of measuring 1M of pinned-DRAM faults. Phase 18 forbids that.
    """
    coverage = float(host_pool_tokens) / float(candidate_tokens)
    if coverage < 1.0 and not allow_partial_coverage:
        raise ValueError(
            f"host_pool_coverage={coverage:.3f} < 1.0; refusing to emit a "
            f"non-full-pool runtime row. Set source_pool_chunks high enough "
            f"so pool_chunks * chunk_tokens >= candidate_tokens, or pass "
            f"allow_partial_coverage=True if you really want a re-read pool."
        )
    return coverage


def make_synthetic_cache(
    *,
    seq_len: int,
    positions: Sequence[int],
    spec: KVRuntimeSpec,
    device: str | torch.device,
    pin_memory: bool = False,
) -> PositionTrackedCache:
    """Build a zero-valued position-tracked KV cache with real model shape."""
    if len(positions) != int(seq_len):
        raise ValueError(f"positions length {len(positions)} does not match seq_len {seq_len}")
    target = torch.device(device)
    layers = []
    for _ in range(spec.n_layers):
        key = torch.zeros((1, spec.n_kv_heads, int(seq_len), spec.head_dim), dtype=spec.dtype, device=target)
        value = torch.zeros((1, spec.n_kv_heads, int(seq_len), spec.head_dim), dtype=spec.dtype, device=target)
        if pin_memory and target.type == "cpu":
            try:
                key = key.pin_memory()
                value = value.pin_memory()
            except RuntimeError:
                pass
        layers.append((key, value))
    return PositionTrackedCache(tuple(layers), list(positions))


def interleaved_active_positions(active_tokens: int) -> list[int]:
    """Use even absolute positions so restored rows can be interleaved."""
    return [2 * index for index in range(int(active_tokens))]


def interleaved_restore_positions(k_tokens: int) -> list[int]:
    """Use odd absolute positions to stress non-append reinsertion."""
    return [2 * index + 1 for index in range(int(k_tokens))]


def profile_transfer_inject_capacity(
    *,
    active_tokens: int,
    k_tokens: int,
    spec: KVRuntimeSpec,
    device: str | torch.device = "cuda",
    trials: int = 20,
    warmup_trials: int = 2,
    pin_memory: bool = True,
) -> dict[str, float | int | str]:
    """Measure host-to-device transfer and reinjection for one operating point."""
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cuda runtime-capacity profiling.")
    if int(active_tokens) <= 0 or int(k_tokens) <= 0:
        raise ValueError("active_tokens and k_tokens must both be positive.")
    if int(trials) <= 0:
        raise ValueError("trials must be positive.")

    active_cache = make_synthetic_cache(
        seq_len=int(active_tokens),
        positions=interleaved_active_positions(int(active_tokens)),
        spec=spec,
        device=target,
    )
    host_restore = make_synthetic_cache(
        seq_len=int(k_tokens),
        positions=interleaved_restore_positions(int(k_tokens)),
        spec=spec,
        device="cpu",
        pin_memory=pin_memory,
    )

    transfer_ms: list[float] = []
    inject_ms: list[float] = []
    total_ms: list[float] = []

    for trial_idx in range(int(warmup_trials) + int(trials)):
        synchronize(target)
        total_start = time.perf_counter()
        transfer_start = total_start
        restored = host_restore.to_device(target, non_blocking=pin_memory and target.type == "cuda")
        synchronize(target)
        transfer_elapsed_ms = (time.perf_counter() - transfer_start) * 1000.0

        inject_start = time.perf_counter()
        repaired = inject_kv(active_cache, restored, restored.positions)
        synchronize(target)
        inject_elapsed_ms = (time.perf_counter() - inject_start) * 1000.0
        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0

        if trial_idx >= int(warmup_trials):
            transfer_ms.append(transfer_elapsed_ms)
            inject_ms.append(inject_elapsed_ms)
            total_ms.append(total_elapsed_ms)

        del repaired
        del restored

    if target.type == "cuda":
        torch.cuda.empty_cache()

    transfer = percentile_summary(transfer_ms)
    inject = percentile_summary(inject_ms)
    total = percentile_summary(total_ms)
    return {
        "active_tokens": int(active_tokens),
        "k": int(k_tokens),
        "trials": int(trials),
        "device": str(target),
        "dtype": str(spec.dtype).replace("torch.", ""),
        "bytes_per_token": int(spec.bytes_per_token),
        "restore_bytes": int(spec.bytes_per_token * int(k_tokens)),
        "active_bytes": int(spec.bytes_per_token * int(active_tokens)),
        "p50_transfer_ms": transfer["p50_ms"],
        "p95_transfer_ms": transfer["p95_ms"],
        "p99_transfer_ms": transfer["p99_ms"],
        "p50_inject_ms": inject["p50_ms"],
        "p95_inject_ms": inject["p95_ms"],
        "p99_inject_ms": inject["p99_ms"],
        "p50_total_ms": total["p50_ms"],
        "p95_total_ms": total["p95_ms"],
        "p99_total_ms": total["p99_ms"],
    }


def _pin_tensor_if_requested(tensor: torch.Tensor, *, pin_memory: bool) -> torch.Tensor:
    if not pin_memory:
        return tensor
    try:
        return tensor.pin_memory()
    except RuntimeError:
        return tensor


def profile_chunked_selection_capacity(
    *,
    candidate_tokens: int,
    k_tokens: int,
    spec: KVRuntimeSpec,
    query_len: int = 64,
    chunk_tokens: int = 16_384,
    source_pool_chunks: int = 1,
    device: str | torch.device = "cuda",
    trials: int = 10,
    warmup_trials: int = 3,
    pin_memory: bool = True,
    allow_partial_coverage: bool = False,
) -> dict[str, float | int | str]:
    """Measure chunked Q2-key scanning and top-K selection over a large store.

    The measured operation is the scalable selection part of repair: stream
    Qwen-shaped offloaded key rows in chunks, score them against synthetic
    exact-Q-shaped query rows with GPU matmuls, and select the top K rows.  It
    does not materialize the full offloaded KV store in GPU memory and does not
    include final KV row movement or reinsertion.  ``source_pool_chunks``
    controls how many distinct pinned host chunks are cycled through during the
    scan; values greater than one avoid repeatedly reading the same host
    allocation in large-store timing probes.
    """
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cuda selection-capacity profiling.")
    if int(candidate_tokens) <= 0 or int(k_tokens) <= 0:
        raise ValueError("candidate_tokens and k_tokens must both be positive.")
    if int(k_tokens) > int(candidate_tokens):
        raise ValueError("k_tokens cannot exceed candidate_tokens.")
    if int(query_len) <= 0 or int(chunk_tokens) <= 0 or int(source_pool_chunks) <= 0:
        raise ValueError("query_len, chunk_tokens, and source_pool_chunks must all be positive.")
    if int(trials) <= 0:
        raise ValueError("trials must be positive.")
    if int(spec.n_query_heads) % int(spec.n_kv_heads) != 0:
        raise ValueError("n_query_heads must be divisible by n_kv_heads.")

    heads_per_kv = int(spec.n_query_heads) // int(spec.n_kv_heads)
    max_chunk = min(int(chunk_tokens), int(candidate_tokens))
    candidate_chunks = int(np.ceil(int(candidate_tokens) / max_chunk))
    pool_chunks = min(int(source_pool_chunks), candidate_chunks)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    host_key_pool = torch.randn(
        (pool_chunks, int(spec.n_kv_heads), max_chunk, int(spec.head_dim)),
        generator=generator,
        dtype=spec.dtype,
        device="cpu",
    )
    host_key_pool = _pin_tensor_if_requested(host_key_pool, pin_memory=pin_memory)
    pin_memory_effective = bool(host_key_pool.is_pinned()) if hasattr(host_key_pool, "is_pinned") else False

    query_layers = []
    for layer_idx in range(int(spec.n_layers)):
        layer_generator = torch.Generator(device="cpu")
        layer_generator.manual_seed(10_000 + layer_idx)
        query_layer = torch.randn(
            (int(spec.n_query_heads), int(query_len), int(spec.head_dim)),
            generator=layer_generator,
            dtype=torch.float32,
            device="cpu",
        ).to(target)
        query_layers.append(
            query_layer.reshape(int(spec.n_kv_heads), heads_per_kv, int(query_len), int(spec.head_dim))
        )

    total_ms: list[float] = []
    for trial_idx in range(int(warmup_trials) + int(trials)):
        synchronize(target)
        start_time = time.perf_counter()
        scores = torch.zeros(int(candidate_tokens), dtype=torch.float32, device=target)
        for layer_query in query_layers:
            for chunk_idx, start in enumerate(range(0, int(candidate_tokens), max_chunk)):
                stop = min(start + max_chunk, int(candidate_tokens))
                current = stop - start
                host_key_chunk = host_key_pool[chunk_idx % pool_chunks]
                key_chunk = _h2d_score_keys(
                    host_key_chunk[:, :current, :],
                    target=target,
                    pin_memory=pin_memory,
                )
                logits = torch.einsum("hgqd,htd->hgqt", layer_query, key_chunk)
                pooled = logits.amax(dim=2).mean(dim=(0, 1))
                scores[start:stop].add_(pooled)
        scores.div_(float(spec.n_layers))
        torch.topk(scores, k=int(k_tokens), largest=True, sorted=False)
        synchronize(target)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if trial_idx >= int(warmup_trials):
            total_ms.append(elapsed_ms)
        del scores

    if target.type == "cuda":
        torch.cuda.empty_cache()

    total = percentile_summary(total_ms)
    key_bytes_per_token = int(
        spec.n_layers
        * spec.n_kv_heads
        * spec.head_dim
        * torch.empty((), dtype=spec.dtype).element_size()
    )
    layer_key_bytes_per_token = int(
        spec.n_kv_heads * spec.head_dim * torch.empty((), dtype=spec.dtype).element_size()
    )
    host_pool_tokens = min(int(candidate_tokens), int(pool_chunks) * int(max_chunk))
    coverage = _enforce_coverage(
        host_pool_tokens=int(host_pool_tokens),
        candidate_tokens=int(candidate_tokens),
        allow_partial_coverage=bool(allow_partial_coverage),
    )
    return {
        "candidate_tokens": int(candidate_tokens),
        "k": int(k_tokens),
        "query_len": int(query_len),
        "chunk_tokens": int(chunk_tokens),
        "candidate_chunks": int(candidate_chunks),
        "source_pool_chunks": int(pool_chunks),
        "host_pool_tokens": int(host_pool_tokens),
        "host_pool_coverage": coverage,
        "trials": int(trials),
        "device": str(target),
        "dtype": str(spec.dtype).replace("torch.", ""),
        "n_layers": int(spec.n_layers),
        "n_query_heads": int(spec.n_query_heads),
        "n_kv_heads": int(spec.n_kv_heads),
        "head_dim": int(spec.head_dim),
        "key_bytes_per_token": int(key_bytes_per_token),
        "layer_key_bytes_per_token": int(layer_key_bytes_per_token),
        "host_key_pool_bytes": int(layer_key_bytes_per_token * host_pool_tokens),
        "streamed_key_bytes": int(key_bytes_per_token * int(candidate_tokens)),
        "offloaded_kv_bytes": int(spec.bytes_per_token * int(candidate_tokens)),
        "selected_kv_bytes": int(spec.bytes_per_token * int(k_tokens)),
        "pin_memory_requested": bool(pin_memory),
        "pin_memory_effective": bool(pin_memory_effective),
        "p50_total_ms": total["p50_ms"],
        "p95_total_ms": total["p95_ms"],
        "p99_total_ms": total["p99_ms"],
        "mean_total_ms": total["mean_ms"],
    }


def _make_host_key_pool(
    *,
    candidate_tokens: int,
    chunk_tokens: int,
    source_pool_chunks: int,
    spec: KVRuntimeSpec,
    pin_memory: bool,
) -> tuple[torch.Tensor, int, int, int]:
    max_chunk = min(int(chunk_tokens), int(candidate_tokens))
    candidate_chunks = int(np.ceil(int(candidate_tokens) / max_chunk))
    pool_chunks = min(int(source_pool_chunks), candidate_chunks)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    host_key_pool = torch.randn(
        (pool_chunks, int(spec.n_kv_heads), max_chunk, int(spec.head_dim)),
        generator=generator,
        dtype=spec.dtype,
        device="cpu",
    )
    return _pin_tensor_if_requested(host_key_pool, pin_memory=pin_memory), max_chunk, candidate_chunks, pool_chunks


def _make_query_layers(*, spec: KVRuntimeSpec, query_len: int, target: torch.device) -> list[torch.Tensor]:
    heads_per_kv = int(spec.n_query_heads) // int(spec.n_kv_heads)
    query_layers = []
    for layer_idx in range(int(spec.n_layers)):
        layer_generator = torch.Generator(device="cpu")
        layer_generator.manual_seed(10_000 + layer_idx)
        query_layer = torch.randn(
            (int(spec.n_query_heads), int(query_len), int(spec.head_dim)),
            generator=layer_generator,
            dtype=torch.float32,
            device="cpu",
        ).to(target)
        query_layers.append(
            query_layer.reshape(int(spec.n_kv_heads), heads_per_kv, int(query_len), int(spec.head_dim))
        )
    return query_layers


def _scan_candidate_scores(
    *,
    candidate_tokens: int,
    max_chunk: int,
    pool_chunks: int,
    host_key_pool: torch.Tensor,
    query_layers: Sequence[torch.Tensor],
    target: torch.device,
    pin_memory: bool,
) -> torch.Tensor:
    scores = torch.zeros(int(candidate_tokens), dtype=torch.float32, device=target)
    for layer_query in query_layers:
        for chunk_idx, start in enumerate(range(0, int(candidate_tokens), int(max_chunk))):
            stop = min(start + int(max_chunk), int(candidate_tokens))
            current = stop - start
            host_key_chunk = host_key_pool[chunk_idx % int(pool_chunks)]
            key_chunk = _h2d_score_keys(
                host_key_chunk[:, :current, :],
                target=target,
                pin_memory=pin_memory,
            )
            logits = torch.einsum("hgqd,htd->hgqt", layer_query, key_chunk)
            pooled = logits.amax(dim=2).mean(dim=(0, 1))
            scores[start:stop].add_(pooled)
    scores.div_(float(len(query_layers)))
    return scores


def profile_chunked_selection_capacity_multi_k(
    *,
    candidate_tokens: int,
    k_tokens_values: Sequence[int],
    spec: KVRuntimeSpec,
    query_len: int = 64,
    chunk_tokens: int = 16_384,
    source_pool_chunks: int = 1,
    device: str | torch.device = "cuda",
    trials: int = 10,
    warmup_trials: int = 3,
    pin_memory: bool = True,
    allow_partial_coverage: bool = False,
) -> list[dict[str, float | int | str]]:
    """Measure one chunked score scan and multiple top-K budgets per trial."""
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cuda selection-capacity profiling.")
    if int(candidate_tokens) <= 0:
        raise ValueError("candidate_tokens must be positive.")
    k_values = tuple(sorted({int(k) for k in k_tokens_values}))
    if not k_values:
        raise ValueError("At least one k value is required.")
    if any(k <= 0 for k in k_values):
        raise ValueError("All k values must be positive.")
    if any(k > int(candidate_tokens) for k in k_values):
        raise ValueError("k_tokens cannot exceed candidate_tokens.")
    if int(query_len) <= 0 or int(chunk_tokens) <= 0 or int(source_pool_chunks) <= 0:
        raise ValueError("query_len, chunk_tokens, and source_pool_chunks must all be positive.")
    if int(trials) <= 0:
        raise ValueError("trials must be positive.")
    if int(spec.n_query_heads) % int(spec.n_kv_heads) != 0:
        raise ValueError("n_query_heads must be divisible by n_kv_heads.")

    host_key_pool, max_chunk, candidate_chunks, pool_chunks = _make_host_key_pool(
        candidate_tokens=int(candidate_tokens),
        chunk_tokens=int(chunk_tokens),
        source_pool_chunks=int(source_pool_chunks),
        spec=spec,
        pin_memory=pin_memory,
    )
    pin_memory_effective = bool(host_key_pool.is_pinned()) if hasattr(host_key_pool, "is_pinned") else False
    host_pool_tokens = min(int(candidate_tokens), int(pool_chunks) * int(max_chunk))
    query_layers = _make_query_layers(spec=spec, query_len=int(query_len), target=target)

    scan_ms: list[float] = []
    topk_ms: dict[int, list[float]] = {k: [] for k in k_values}
    total_ms: dict[int, list[float]] = {k: [] for k in k_values}
    for trial_idx in range(int(warmup_trials) + int(trials)):
        synchronize(target)
        scan_start = time.perf_counter()
        scores = _scan_candidate_scores(
            candidate_tokens=int(candidate_tokens),
            max_chunk=int(max_chunk),
            pool_chunks=int(pool_chunks),
            host_key_pool=host_key_pool,
            query_layers=query_layers,
            target=target,
            pin_memory=pin_memory,
        )
        synchronize(target)
        scan_elapsed_ms = (time.perf_counter() - scan_start) * 1000.0

        trial_topk_ms: dict[int, float] = {}
        for k_value in k_values:
            topk_start = time.perf_counter()
            torch.topk(scores, k=int(k_value), largest=True, sorted=False)
            synchronize(target)
            trial_topk_ms[k_value] = (time.perf_counter() - topk_start) * 1000.0

        if trial_idx >= int(warmup_trials):
            scan_ms.append(scan_elapsed_ms)
            for k_value in k_values:
                topk_ms[k_value].append(trial_topk_ms[k_value])
                total_ms[k_value].append(scan_elapsed_ms + trial_topk_ms[k_value])
        del scores

    if target.type == "cuda":
        torch.cuda.empty_cache()

    scan = percentile_summary(scan_ms)
    coverage = _enforce_coverage(
        host_pool_tokens=int(host_pool_tokens),
        candidate_tokens=int(candidate_tokens),
        allow_partial_coverage=bool(allow_partial_coverage),
    )
    key_bytes_per_token = int(
        spec.n_layers
        * spec.n_kv_heads
        * spec.head_dim
        * torch.empty((), dtype=spec.dtype).element_size()
    )
    rows = []
    for k_value in k_values:
        topk = percentile_summary(topk_ms[k_value])
        total = percentile_summary(total_ms[k_value])
        rows.append(
            {
                "candidate_tokens": int(candidate_tokens),
                "k": int(k_value),
                "query_len": int(query_len),
                "chunk_tokens": int(chunk_tokens),
                "candidate_chunks": int(candidate_chunks),
                "source_pool_chunks": int(pool_chunks),
                "host_pool_tokens": int(host_pool_tokens),
                "host_pool_coverage": coverage,
                "trials": int(trials),
                "device": str(target),
                "dtype": str(spec.dtype).replace("torch.", ""),
                "n_layers": int(spec.n_layers),
                "n_query_heads": int(spec.n_query_heads),
                "n_kv_heads": int(spec.n_kv_heads),
                "head_dim": int(spec.head_dim),
                "key_bytes_per_token": int(key_bytes_per_token),
                "streamed_key_bytes": int(key_bytes_per_token * int(candidate_tokens)),
                "offloaded_kv_bytes": int(spec.bytes_per_token * int(candidate_tokens)),
                "selected_kv_bytes": int(spec.bytes_per_token * int(k_value)),
                "pin_memory_requested": bool(pin_memory),
                "pin_memory_effective": bool(pin_memory_effective),
                "p50_scan_ms": scan["p50_ms"],
                "p95_scan_ms": scan["p95_ms"],
                "p99_scan_ms": scan["p99_ms"],
                "p50_topk_ms": topk["p50_ms"],
                "p95_topk_ms": topk["p95_ms"],
                "p99_topk_ms": topk["p99_ms"],
                "p50_total_ms": total["p50_ms"],
                "p95_total_ms": total["p95_ms"],
                "p99_total_ms": total["p99_ms"],
                "mean_total_ms": total["mean_ms"],
            }
        )
    return rows


def profile_end_to_end_repair_capacity(
    *,
    active_tokens: int,
    candidate_tokens: int,
    k_tokens: int,
    spec: KVRuntimeSpec,
    query_len: int = 64,
    chunk_tokens: int = 16_384,
    source_pool_chunks: int = 1,
    device: str | torch.device = "cuda",
    trials: int = 10,
    warmup_trials: int = 3,
    pin_memory: bool = True,
    allow_partial_coverage: bool = False,
) -> dict[str, float | int | str]:
    """Measure one synthetic repair path: select from host store, move, inject.

    Each trial streams offloaded Qwen-shaped key rows through a GPU scorer,
    takes a real top-K, transfers a full selected KV block from host memory,
    and reinserts that block into an active ``PositionTrackedCache``. Tensor
    values are synthetic, but shapes, device transfers, scoring kernels, and
    reinsertion mechanics match the runtime path being claimed.
    """
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cuda end-to-end repair profiling.")
    if int(active_tokens) <= 0 or int(candidate_tokens) <= 0 or int(k_tokens) <= 0:
        raise ValueError("active_tokens, candidate_tokens, and k_tokens must all be positive.")
    if int(k_tokens) > int(candidate_tokens):
        raise ValueError("k_tokens cannot exceed candidate_tokens.")
    if int(query_len) <= 0 or int(chunk_tokens) <= 0 or int(source_pool_chunks) <= 0:
        raise ValueError("query_len, chunk_tokens, and source_pool_chunks must all be positive.")
    if int(trials) <= 0:
        raise ValueError("trials must be positive.")
    if int(spec.n_query_heads) % int(spec.n_kv_heads) != 0:
        raise ValueError("n_query_heads must be divisible by n_kv_heads.")

    active_cache = make_synthetic_cache(
        seq_len=int(active_tokens),
        positions=interleaved_active_positions(int(active_tokens)),
        spec=spec,
        device=target,
    )
    host_restore = make_synthetic_cache(
        seq_len=int(k_tokens),
        positions=interleaved_restore_positions(int(k_tokens)),
        spec=spec,
        device="cpu",
        pin_memory=pin_memory,
    )
    host_key_pool, max_chunk, candidate_chunks, pool_chunks = _make_host_key_pool(
        candidate_tokens=int(candidate_tokens),
        chunk_tokens=int(chunk_tokens),
        source_pool_chunks=int(source_pool_chunks),
        spec=spec,
        pin_memory=pin_memory,
    )
    pin_memory_effective = bool(host_key_pool.is_pinned()) if hasattr(host_key_pool, "is_pinned") else False
    query_layers = _make_query_layers(spec=spec, query_len=int(query_len), target=target)

    select_ms: list[float] = []
    move_inject_ms: list[float] = []
    total_ms: list[float] = []
    for trial_idx in range(int(warmup_trials) + int(trials)):
        synchronize(target)
        total_start = time.perf_counter()

        select_start = total_start
        scores = _scan_candidate_scores(
            candidate_tokens=int(candidate_tokens),
            max_chunk=int(max_chunk),
            pool_chunks=int(pool_chunks),
            host_key_pool=host_key_pool,
            query_layers=query_layers,
            target=target,
            pin_memory=pin_memory,
        )
        torch.topk(scores, k=int(k_tokens), largest=True, sorted=False)
        synchronize(target)
        select_elapsed_ms = (time.perf_counter() - select_start) * 1000.0
        del scores

        move_start = time.perf_counter()
        restored = host_restore.to_device(target, non_blocking=pin_memory and target.type == "cuda")
        repaired = inject_kv(active_cache, restored, restored.positions)
        synchronize(target)
        move_elapsed_ms = (time.perf_counter() - move_start) * 1000.0
        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0

        if trial_idx >= int(warmup_trials):
            select_ms.append(select_elapsed_ms)
            move_inject_ms.append(move_elapsed_ms)
            total_ms.append(total_elapsed_ms)

        del repaired
        del restored

    if target.type == "cuda":
        torch.cuda.empty_cache()

    select = percentile_summary(select_ms)
    move_inject = percentile_summary(move_inject_ms)
    total = percentile_summary(total_ms)
    key_bytes_per_token = int(
        spec.n_layers
        * spec.n_kv_heads
        * spec.head_dim
        * torch.empty((), dtype=spec.dtype).element_size()
    )
    layer_key_bytes_per_token = int(
        spec.n_kv_heads * spec.head_dim * torch.empty((), dtype=spec.dtype).element_size()
    )
    host_pool_tokens = min(int(candidate_tokens), int(pool_chunks) * int(max_chunk))
    coverage = _enforce_coverage(
        host_pool_tokens=int(host_pool_tokens),
        candidate_tokens=int(candidate_tokens),
        allow_partial_coverage=bool(allow_partial_coverage),
    )
    return {
        "active_tokens": int(active_tokens),
        "candidate_tokens": int(candidate_tokens),
        "k": int(k_tokens),
        "query_len": int(query_len),
        "chunk_tokens": int(chunk_tokens),
        "candidate_chunks": int(candidate_chunks),
        "source_pool_chunks": int(pool_chunks),
        "host_pool_tokens": int(host_pool_tokens),
        "host_pool_coverage": coverage,
        "trials": int(trials),
        "device": str(target),
        "dtype": str(spec.dtype).replace("torch.", ""),
        "n_layers": int(spec.n_layers),
        "n_query_heads": int(spec.n_query_heads),
        "n_kv_heads": int(spec.n_kv_heads),
        "head_dim": int(spec.head_dim),
        "bytes_per_token": int(spec.bytes_per_token),
        "key_bytes_per_token": int(key_bytes_per_token),
        "layer_key_bytes_per_token": int(layer_key_bytes_per_token),
        "host_key_pool_bytes": int(layer_key_bytes_per_token * host_pool_tokens),
        "streamed_key_bytes": int(key_bytes_per_token * int(candidate_tokens)),
        "offloaded_kv_bytes": int(spec.bytes_per_token * int(candidate_tokens)),
        "selected_kv_bytes": int(spec.bytes_per_token * int(k_tokens)),
        "active_bytes": int(spec.bytes_per_token * int(active_tokens)),
        "pin_memory_requested": bool(pin_memory),
        "pin_memory_effective": bool(pin_memory_effective),
        "p50_select_ms": select["p50_ms"],
        "p95_select_ms": select["p95_ms"],
        "p99_select_ms": select["p99_ms"],
        "p50_move_inject_ms": move_inject["p50_ms"],
        "p95_move_inject_ms": move_inject["p95_ms"],
        "p99_move_inject_ms": move_inject["p99_ms"],
        "p50_total_ms": total["p50_ms"],
        "p95_total_ms": total["p95_ms"],
        "p99_total_ms": total["p99_ms"],
        "mean_total_ms": total["mean_ms"],
    }


def profile_end_to_end_repair_capacity_multi_k(
    *,
    active_tokens: int,
    candidate_tokens: int,
    k_tokens_values: Sequence[int],
    spec: KVRuntimeSpec,
    query_len: int = 64,
    chunk_tokens: int = 16_384,
    source_pool_chunks: int = 1,
    device: str | torch.device = "cuda",
    trials: int = 10,
    warmup_trials: int = 3,
    pin_memory: bool = True,
    allow_partial_coverage: bool = False,
) -> list[dict[str, float | int | str]]:
    """Measure integrated repair for several K values after one scan per trial."""
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cuda end-to-end repair profiling.")
    if int(active_tokens) <= 0 or int(candidate_tokens) <= 0:
        raise ValueError("active_tokens and candidate_tokens must both be positive.")
    k_values = tuple(sorted({int(k) for k in k_tokens_values}))
    if not k_values:
        raise ValueError("At least one k value is required.")
    if any(k <= 0 for k in k_values):
        raise ValueError("All k values must be positive.")
    if any(k > int(candidate_tokens) for k in k_values):
        raise ValueError("k_tokens cannot exceed candidate_tokens.")
    if int(query_len) <= 0 or int(chunk_tokens) <= 0 or int(source_pool_chunks) <= 0:
        raise ValueError("query_len, chunk_tokens, and source_pool_chunks must all be positive.")
    if int(trials) <= 0:
        raise ValueError("trials must be positive.")
    if int(spec.n_query_heads) % int(spec.n_kv_heads) != 0:
        raise ValueError("n_query_heads must be divisible by n_kv_heads.")

    active_cache = make_synthetic_cache(
        seq_len=int(active_tokens),
        positions=interleaved_active_positions(int(active_tokens)),
        spec=spec,
        device=target,
    )
    host_restore_by_k = {
        k: make_synthetic_cache(
            seq_len=int(k),
            positions=interleaved_restore_positions(int(k)),
            spec=spec,
            device="cpu",
            pin_memory=pin_memory,
        )
        for k in k_values
    }
    host_key_pool, max_chunk, candidate_chunks, pool_chunks = _make_host_key_pool(
        candidate_tokens=int(candidate_tokens),
        chunk_tokens=int(chunk_tokens),
        source_pool_chunks=int(source_pool_chunks),
        spec=spec,
        pin_memory=pin_memory,
    )
    pin_memory_effective = bool(host_key_pool.is_pinned()) if hasattr(host_key_pool, "is_pinned") else False
    query_layers = _make_query_layers(spec=spec, query_len=int(query_len), target=target)

    scan_ms: list[float] = []
    topk_ms: dict[int, list[float]] = {k: [] for k in k_values}
    move_inject_ms: dict[int, list[float]] = {k: [] for k in k_values}
    total_ms: dict[int, list[float]] = {k: [] for k in k_values}
    for trial_idx in range(int(warmup_trials) + int(trials)):
        synchronize(target)
        scan_start = time.perf_counter()
        scores = _scan_candidate_scores(
            candidate_tokens=int(candidate_tokens),
            max_chunk=int(max_chunk),
            pool_chunks=int(pool_chunks),
            host_key_pool=host_key_pool,
            query_layers=query_layers,
            target=target,
            pin_memory=pin_memory,
        )
        synchronize(target)
        scan_elapsed_ms = (time.perf_counter() - scan_start) * 1000.0

        trial_topk_ms: dict[int, float] = {}
        trial_move_ms: dict[int, float] = {}
        for k_value in k_values:
            topk_start = time.perf_counter()
            torch.topk(scores, k=int(k_value), largest=True, sorted=False)
            synchronize(target)
            trial_topk_ms[k_value] = (time.perf_counter() - topk_start) * 1000.0

            move_start = time.perf_counter()
            host_restore = host_restore_by_k[k_value]
            restored = host_restore.to_device(target, non_blocking=pin_memory and target.type == "cuda")
            repaired = inject_kv(active_cache, restored, restored.positions)
            synchronize(target)
            trial_move_ms[k_value] = (time.perf_counter() - move_start) * 1000.0
            del repaired
            del restored

        if trial_idx >= int(warmup_trials):
            scan_ms.append(scan_elapsed_ms)
            for k_value in k_values:
                topk_ms[k_value].append(trial_topk_ms[k_value])
                move_inject_ms[k_value].append(trial_move_ms[k_value])
                total_ms[k_value].append(scan_elapsed_ms + trial_topk_ms[k_value] + trial_move_ms[k_value])
        del scores

    if target.type == "cuda":
        torch.cuda.empty_cache()

    scan = percentile_summary(scan_ms)
    key_bytes_per_token = int(
        spec.n_layers
        * spec.n_kv_heads
        * spec.head_dim
        * torch.empty((), dtype=spec.dtype).element_size()
    )
    layer_key_bytes_per_token = int(
        spec.n_kv_heads * spec.head_dim * torch.empty((), dtype=spec.dtype).element_size()
    )
    host_pool_tokens = min(int(candidate_tokens), int(pool_chunks) * int(max_chunk))
    coverage = _enforce_coverage(
        host_pool_tokens=int(host_pool_tokens),
        candidate_tokens=int(candidate_tokens),
        allow_partial_coverage=bool(allow_partial_coverage),
    )
    rows = []
    for k_value in k_values:
        topk = percentile_summary(topk_ms[k_value])
        move_inject = percentile_summary(move_inject_ms[k_value])
        total = percentile_summary(total_ms[k_value])
        rows.append(
            {
                "active_tokens": int(active_tokens),
                "candidate_tokens": int(candidate_tokens),
                "k": int(k_value),
                "query_len": int(query_len),
                "chunk_tokens": int(chunk_tokens),
                "candidate_chunks": int(candidate_chunks),
                "source_pool_chunks": int(pool_chunks),
                "host_pool_tokens": int(host_pool_tokens),
                "host_pool_coverage": coverage,
                "trials": int(trials),
                "device": str(target),
                "dtype": str(spec.dtype).replace("torch.", ""),
                "n_layers": int(spec.n_layers),
                "n_query_heads": int(spec.n_query_heads),
                "n_kv_heads": int(spec.n_kv_heads),
                "head_dim": int(spec.head_dim),
                "bytes_per_token": int(spec.bytes_per_token),
                "key_bytes_per_token": int(key_bytes_per_token),
                "layer_key_bytes_per_token": int(layer_key_bytes_per_token),
                "host_key_pool_bytes": int(layer_key_bytes_per_token * host_pool_tokens),
                "streamed_key_bytes": int(key_bytes_per_token * int(candidate_tokens)),
                "offloaded_kv_bytes": int(spec.bytes_per_token * int(candidate_tokens)),
                "selected_kv_bytes": int(spec.bytes_per_token * int(k_value)),
                "active_bytes": int(spec.bytes_per_token * int(active_tokens)),
                "pin_memory_requested": bool(pin_memory),
                "pin_memory_effective": bool(pin_memory_effective),
                "p50_scan_ms": scan["p50_ms"],
                "p95_scan_ms": scan["p95_ms"],
                "p99_scan_ms": scan["p99_ms"],
                "p50_topk_ms": topk["p50_ms"],
                "p95_topk_ms": topk["p95_ms"],
                "p99_topk_ms": topk["p99_ms"],
                "p50_move_inject_ms": move_inject["p50_ms"],
                "p95_move_inject_ms": move_inject["p95_ms"],
                "p99_move_inject_ms": move_inject["p99_ms"],
                "p50_total_ms": total["p50_ms"],
                "p95_total_ms": total["p95_ms"],
                "p99_total_ms": total["p99_ms"],
                "mean_total_ms": total["mean_ms"],
            }
        )
    return rows


def feasibility_rows(
    rows: Sequence[dict[str, float | int | str]],
    *,
    idle_windows_s: Iterable[float] = (0.1, 0.5, 1.0, 2.0, 5.0),
    budget_fraction: float = 0.90,
    percentile_field: str = "p95_total_ms",
    group_fields: Sequence[str] = ("active_tokens",),
) -> list[dict[str, float | int]]:
    """Summarize the largest measured K that fits each idle-window budget."""
    if not 0.0 < float(budget_fraction) <= 1.0:
        raise ValueError("budget_fraction must lie in (0, 1].")
    if not group_fields:
        raise ValueError("group_fields must be non-empty.")
    output: list[dict[str, float | int]] = []
    group_values = sorted({tuple(int(row[field]) for field in group_fields) for row in rows})
    for group_value in group_values:
        subset = [
            row
            for row in rows
            if tuple(int(row[field]) for field in group_fields) == group_value
        ]
        for idle_window_s in idle_windows_s:
            budget_ms = float(idle_window_s) * 1000.0 * float(budget_fraction)
            fitting = [
                int(row["k"])
                for row in subset
                if float(row.get(percentile_field, float("inf"))) <= budget_ms
            ]
            result = {field: int(value) for field, value in zip(group_fields, group_value, strict=True)}
            result.update(
                {
                    "idle_window_s": float(idle_window_s),
                    "budget_ms": budget_ms,
                    "max_measured_k": max(fitting) if fitting else 0,
                }
            )
            output.append(result)
    return output


def write_rows_csv(rows: Sequence[dict[str, object]], path: str | Path) -> None:
    """Write a non-empty row set to CSV."""
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "KVRuntimeSpec",
    "feasibility_rows",
    "interleaved_active_positions",
    "interleaved_restore_positions",
    "make_synthetic_cache",
    "parse_dtype",
    "percentile_summary",
    "profile_chunked_selection_capacity",
    "profile_chunked_selection_capacity_multi_k",
    "profile_end_to_end_repair_capacity",
    "profile_end_to_end_repair_capacity_multi_k",
    "profile_transfer_inject_capacity",
    "write_rows_csv",
]
