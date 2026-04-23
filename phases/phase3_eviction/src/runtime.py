"""Runtime helpers for Phase 3 smoke tests and runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ._repo import PHASE_ROOT, REPO_ROOT
from .eviction import QueryAwareSnapKV, SnapKV, StreamingLLM, log_eviction
from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache, to_tuple_cache
from phases.phase2_kv_cache.src.runtime import (
    MODEL_DIR,
    load_model,
    load_tokenizer,
    make_exact_length_input_ids,
    model_device,
    prefill_cache,
)

RESULTS_DIR = PHASE_ROOT / "results"
DEGRADATION_DIR = RESULTS_DIR / "phase3_degradation"
FIGURE_DIR = DEGRADATION_DIR / "figures"
EVICTION_LOG_DIR = RESULTS_DIR / "phase3_eviction_logs"


def ensure_results_dirs() -> None:
    """Create the result directories declared in the phase instructions."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    EVICTION_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable.")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON artifact with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def build_position_tracked_cache(model, input_ids: torch.Tensor) -> PositionTrackedCache:
    """Prefill a dense cache and attach absolute positions 0..N-1."""
    cache = to_tuple_cache(prefill_cache(model, input_ids))
    seq_len = int(cache[0][0].shape[2])
    return PositionTrackedCache(cache, list(range(seq_len)))


def make_live_smoke_inputs(
    model,
    tokenizer,
    *,
    context_tokens: int = 768,
) -> tuple[PositionTrackedCache, torch.Tensor]:
    """Create one moderate-size live cache and a short query prompt."""
    device = model_device(model)
    context_ids = make_exact_length_input_ids(
        tokenizer,
        target_tokens=context_tokens,
        device=device,
        base_text=(
            "Idle GPU windows, deferred tool calls, cache repair buffers, maintenance "
            "records, telemetry, and diagnostics entries are reviewed in this synthetic "
            "phase-three smoke prompt. "
        ),
    )
    full_cache = build_position_tracked_cache(model, context_ids)
    query_ids = tokenizer(
        "Which diagnostic topic appears in the smoke prompt?",
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)
    return full_cache, query_ids


def run_live_smoke(
    model,
    tokenizer,
    *,
    context_tokens: int = 768,
    k_budget: int = 96,
) -> dict[str, Any]:
    """Exercise all Phase 3 policies against the local Qwen model once."""
    ensure_results_dirs()
    full_cache, query_ids = make_live_smoke_inputs(model, tokenizer, context_tokens=context_tokens)
    recency_window = max(0, min(64, k_budget - 4))
    policies = {
        "snapkv": SnapKV(obs_window_size=16, sink_size=4, recency_window=recency_window, pooling="max"),
        "query_aware_snapkv": QueryAwareSnapKV(
            model,
            obs_window_size=16,
            sink_size=4,
            recency_window=recency_window,
            pooling="max",
        ),
        "streaming_llm": StreamingLLM(sink_size=4, recency_window=max(0, k_budget - 4)),
    }
    task_relevant_positions = [
        full_cache.positions[0],
        full_cache.positions[len(full_cache) // 2],
        full_cache.positions[-1],
    ]

    summary: dict[str, Any] = {
        "repo_root": REPO_ROOT,
        "context_tokens": len(full_cache),
        "k_budget": int(k_budget),
        "query_tokens": int(query_ids.shape[1]),
        "policies": {},
    }

    for name, policy in policies.items():
        result = policy.evict(full_cache, k_budget=k_budget, obs_window=query_ids if "query_aware" in name else None)
        log_dir = EVICTION_LOG_DIR / "live_smoke" / name
        log_entry = log_eviction(
            result,
            example_id="live_smoke",
            task="phase3_live_smoke",
            task_relevant_positions=task_relevant_positions,
            log_dir=log_dir,
            metadata={"policy": name},
        )
        summary["policies"][name] = {
            "kept_count": len(result.compressed.positions),
            "evicted_count": len(result.evicted.positions),
            "obs_window_q_vecs_shape": list(result.obs_window_q_vecs.shape),
            "kept_positions_head": list(result.compressed.positions[:8]),
            "evicted_device": str(result.evicted.device),
            "log_path": log_dir / "live_smoke.json",
            "task_relevant_survived": log_entry["task_relevant_survived"],
        }

    artifact_path = RESULTS_DIR / "phase3_live_smoke.json"
    write_json(artifact_path, summary)
    summary["artifact_path"] = artifact_path
    return summary


__all__ = [
    "DEGRADATION_DIR",
    "EVICTION_LOG_DIR",
    "FIGURE_DIR",
    "MODEL_DIR",
    "RESULTS_DIR",
    "build_position_tracked_cache",
    "ensure_results_dirs",
    "load_model",
    "load_tokenizer",
    "make_live_smoke_inputs",
    "run_live_smoke",
    "write_json",
]
