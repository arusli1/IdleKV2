"""Profiling helpers for the Phase 4 CPU eviction buffer."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache, inject_kv
from phases.phase2_kv_cache.src.runtime import resume_forward

from .eviction_buffer import BufferEntry, EvictionBuffer, SelectionStrategy, extract_recent_q_vecs


@dataclass(frozen=True)
class SyntheticKVSpec:
    """Synthetic one-token KV shape used for log-based profiling."""

    n_layers: int = 28
    n_kv_heads: int = 4
    head_dim: int = 128
    dtype: torch.dtype = torch.float16


def _iter_log_jsons(log_root: Path) -> list[Path]:
    """Recursively enumerate example-log JSON files below one Phase 3 log root."""
    return sorted(
        path
        for path in log_root.rglob("*.json")
        if path.stem.startswith("ex") or path.stem == "live_smoke"
    )


def _percentiles_ms(times_s: list[float]) -> dict[str, float]:
    """Summarize timing samples in milliseconds."""
    return {
        "p50_ms": float(np.percentile(times_s, 50) * 1000.0),
        "p90_ms": float(np.percentile(times_s, 90) * 1000.0),
        "p99_ms": float(np.percentile(times_s, 99) * 1000.0),
        "mean_ms": float(np.mean(times_s) * 1000.0),
    }


def _synchronize_if_cuda(device: str | torch.device) -> None:
    target_device = torch.device(device)
    if target_device.type == "cuda":
        torch.cuda.synchronize()


def profile_cpu_to_gpu_transfer(
    buffer: EvictionBuffer,
    n_tokens_list: Sequence[int] = (50, 100, 250, 500, 1000, 2000),
    n_trials: int = 50,
    device: str = "cuda",
) -> dict[int, dict[str, float]]:
    """Profile the dominant small-K cost: CPU -> GPU restoration latency."""
    results: dict[int, dict[str, float]] = {}
    all_entries = buffer.entries()

    for n_tokens in n_tokens_list:
        requested = int(n_tokens)
        entries = all_entries[:requested]
        if len(entries) < requested:
            continue

        warmup = buffer.to_gpu(entries, device=device)
        del warmup
        times_s: list[float] = []
        for _ in range(int(n_trials)):
            _synchronize_if_cuda(device)
            t0 = time.perf_counter()
            restored = buffer.to_gpu(entries, device=device)
            _synchronize_if_cuda(device)
            times_s.append(time.perf_counter() - t0)
            del restored

        results[requested] = _percentiles_ms(times_s)
    return results


def profile_buffer_scoring(
    buffer: EvictionBuffer,
    recent_q_vecs: torch.Tensor,
    top_k: int = 250,
    strategies: Sequence[SelectionStrategy] = ("l2_norm", "dot_product", "random", "recency_inverse"),
    buffer_sizes: Sequence[int] = (500, 1000, 2000, 5000),
    n_trials: int = 50,
) -> dict[str, dict[int, dict[str, float]]]:
    """Profile CPU-side scoring and top-k selection latency."""
    results: dict[str, dict[int, dict[str, float]]] = {}
    all_entries = buffer.entries()

    for strategy in strategies:
        buffer.selection_strategy = strategy
        strategy_results: dict[int, dict[str, float]] = {}
        for requested_size in buffer_sizes:
            actual_size = min(int(requested_size), len(all_entries))
            if actual_size == 0:
                continue
            entries = all_entries[:actual_size]

            times_s: list[float] = []
            for _ in range(int(n_trials)):
                t0 = time.perf_counter()
                if strategy == "dot_product":
                    scores = buffer._score_dot_product(entries, recent_q_vecs)
                elif strategy == "l2_norm":
                    scores = buffer._score_l2_norm(entries)
                elif strategy == "random":
                    scores = buffer._score_random(entries)
                elif strategy == "recency_inverse":
                    scores = buffer._score_recency_inverse(entries)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

                top_k_actual = min(int(top_k), len(entries))
                if top_k_actual:
                    if top_k_actual == len(entries):
                        _ = np.argsort(scores)[::-1]
                    else:
                        _ = np.argpartition(scores, -top_k_actual)[-top_k_actual:]
                times_s.append(time.perf_counter() - t0)

            strategy_results[int(requested_size)] = _percentiles_ms(times_s)
            strategy_results[int(requested_size)]["actual_entries"] = int(actual_size)
        results[strategy] = strategy_results

    return results


def profile_injection_attention_overhead(
    model,
    base_cache: PositionTrackedCache,
    repair_buffer: EvictionBuffer,
    query_ids: torch.Tensor,
    extra_token_counts: Sequence[int] = (50, 100, 250, 500, 1000),
    n_trials: int = 20,
) -> dict[str, object]:
    """Profile the added forward cost of carrying K restored tokens in cache."""
    device = base_cache.device
    query_ids = query_ids.to(device)

    _ = resume_forward(model, query_ids, base_cache, num_logits_to_keep=1)
    _synchronize_if_cuda(device)
    baseline_times_s: list[float] = []
    for _ in range(int(n_trials)):
        _synchronize_if_cuda(device)
        t0 = time.perf_counter()
        _ = resume_forward(model, query_ids, base_cache, num_logits_to_keep=1)
        _synchronize_if_cuda(device)
        baseline_times_s.append(time.perf_counter() - t0)
    baseline_p50_ms = float(np.percentile(baseline_times_s, 50) * 1000.0)

    by_k: dict[int, dict[str, float]] = {}
    all_entries = repair_buffer.entries()
    for extra_tokens in extra_token_counts:
        actual_k = min(int(extra_tokens), len(all_entries))
        if actual_k == 0:
            continue
        restored = repair_buffer.to_gpu(all_entries[:actual_k], device=str(device))
        extended_cache = inject_kv(base_cache, restored, restored.positions)

        _ = resume_forward(model, query_ids, extended_cache, num_logits_to_keep=1)
        _synchronize_if_cuda(device)
        times_s: list[float] = []
        for _ in range(int(n_trials)):
            _synchronize_if_cuda(device)
            t0 = time.perf_counter()
            _ = resume_forward(model, query_ids, extended_cache, num_logits_to_keep=1)
            _synchronize_if_cuda(device)
            times_s.append(time.perf_counter() - t0)

        p50_ms = float(np.percentile(times_s, 50) * 1000.0)
        by_k[int(extra_tokens)] = {
            "actual_k": float(actual_k),
            "total_p50_ms": p50_ms,
            "overhead_vs_base_ms": p50_ms - baseline_p50_ms,
            "overhead_pct": ((p50_ms - baseline_p50_ms) / baseline_p50_ms * 100.0) if baseline_p50_ms else 0.0,
        }
        del restored
        del extended_cache

    return {
        "baseline_p50_ms": baseline_p50_ms,
        "query_tokens": int(query_ids.shape[1]),
        "by_k": by_k,
    }


def profile_end_to_end_repair(
    buffer: EvictionBuffer,
    active_cache: PositionTrackedCache,
    top_k_values: Sequence[int] = (50, 100, 250, 500),
    n_trials: int = 20,
    device: str = "cuda",
) -> dict[int, dict[str, float]]:
    """Profile extract -> score -> restore -> inject as one wall-clock path."""
    results: dict[int, dict[str, float]] = {}

    for top_k in top_k_values:
        top_k_int = int(top_k)
        warmup_recent_q = extract_recent_q_vecs(active_cache, m=64)
        warmup_selected = buffer.query(warmup_recent_q, top_k=top_k_int)
        if warmup_selected:
            warmup_restored = buffer.to_gpu(warmup_selected, device=device)
            warmup_repaired = inject_kv(active_cache, warmup_restored, warmup_restored.positions)
            _synchronize_if_cuda(device)
            del warmup_repaired
            del warmup_restored
        times_s: list[float] = []
        for _ in range(int(n_trials)):
            t0 = time.perf_counter()
            recent_q = extract_recent_q_vecs(active_cache, m=64)
            selected = buffer.query(recent_q, top_k=top_k_int)
            if not selected:
                times_s.append(time.perf_counter() - t0)
                continue
            restored = buffer.to_gpu(selected, device=device)
            repaired = inject_kv(active_cache, restored, restored.positions)
            _synchronize_if_cuda(device)
            times_s.append(time.perf_counter() - t0)
            del restored
            del repaired

        results[top_k_int] = _percentiles_ms(times_s)
    return results


def _make_synthetic_token(spec: SyntheticKVSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a one-token CPU-pinned KV pair of the requested shape."""
    key = torch.zeros((1, spec.n_kv_heads, 1, spec.head_dim), dtype=spec.dtype)
    value = torch.zeros((1, spec.n_kv_heads, 1, spec.head_dim), dtype=spec.dtype)
    try:
        return key.pin_memory(), value.pin_memory()
    except RuntimeError:
        return key, value


def _load_log_q_vec_mean(q_vec_path: Path) -> torch.Tensor | None:
    """Load one q-vector artifact and normalize it to `[n_layers, head_dim]` on CPU."""
    q_vecs = torch.load(q_vec_path, map_location="cpu")
    if q_vecs.ndim == 3:
        return q_vecs.mean(dim=1).detach().to("cpu", dtype=torch.float32).contiguous()
    if q_vecs.ndim == 2:
        return q_vecs.detach().to("cpu", dtype=torch.float32).contiguous()
    return None


def _iter_entries_from_log_artifact(
    log_path: Path,
    *,
    synthetic_spec: SyntheticKVSpec,
):
    """Yield synthetic profiling entries reconstructed from one Phase 3 log artifact."""
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    q_vec_path = log_path.with_name(f"{log_path.stem}_qvecs.pt")
    if not q_vec_path.exists():
        return

    q_vec_mean = _load_log_q_vec_mean(q_vec_path)
    if q_vec_mean is None:
        return

    score_map = payload.get("importance_scores", {})
    for position in payload.get("evicted_positions", []):
        token_position = int(position)
        score = float(score_map.get(str(token_position), 0.0))
        synthetic_kv = tuple(_make_synthetic_token(synthetic_spec) for _ in range(synthetic_spec.n_layers))
        yield BufferEntry(
            position=token_position,
            kv=synthetic_kv,
            importance_score=score,
            q_vec=q_vec_mean,
        )


def build_buffer_from_log_artifact(
    log_path: str | Path,
    strategy: SelectionStrategy = "l2_norm",
    max_tokens: int = 10_000,
    synthetic_spec: SyntheticKVSpec | None = None,
) -> EvictionBuffer:
    """Reconstruct one profiling buffer from a single Phase 3 log artifact."""
    spec = synthetic_spec or SyntheticKVSpec()
    buffer = EvictionBuffer(max_tokens=max_tokens, selection_strategy=strategy)
    for entry in _iter_entries_from_log_artifact(Path(log_path), synthetic_spec=spec) or ():
        buffer.push(entry)
    return buffer


def build_buffer_from_logs(
    log_dir: str | Path,
    strategy: SelectionStrategy = "l2_norm",
    max_tokens: int = 10_000,
    synthetic_spec: SyntheticKVSpec | None = None,
    stop_when_full: bool = True,
) -> EvictionBuffer:
    """Reconstruct a profiling buffer from Phase 3 eviction logs and q-vector artifacts."""
    spec = synthetic_spec or SyntheticKVSpec()
    buffer = EvictionBuffer(max_tokens=max_tokens, selection_strategy=strategy)
    log_root = Path(log_dir)

    for log_path in _iter_log_jsons(log_root):
        for entry in _iter_entries_from_log_artifact(log_path, synthetic_spec=spec) or ():
            buffer.push(entry)
            if stop_when_full and len(buffer) >= buffer.max_tokens:
                return buffer

    return buffer


__all__ = [
    "SyntheticKVSpec",
    "build_buffer_from_log_artifact",
    "build_buffer_from_logs",
    "profile_buffer_scoring",
    "profile_cpu_to_gpu_transfer",
    "profile_end_to_end_repair",
    "profile_injection_attention_overhead",
]
