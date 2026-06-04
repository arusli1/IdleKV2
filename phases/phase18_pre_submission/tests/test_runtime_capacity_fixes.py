"""Unit tests for Step 0a fixes in runtime_capacity.py.

Two regressions we want to prevent:

1. ``.to(device, dtype=torch.float32)`` from a pinned BF16 host source
   silently demotes the H2D to a blocking copy. Phase 18 fixes this by
   doing BF16 H2D first, then on-device ``.float()``. The result must be
   numerically identical to the old call.
2. ``host_pool_coverage < 1.0`` was emitted silently in Phase 17 because
   ``source_pool_chunks=1`` re-read a 16K-token pool 64x at 1M
   candidate_tokens. Phase 18 forbids this by default.
"""

from __future__ import annotations

import pytest
import torch

from phases.phase4_eviction_buffer.src.buffer.runtime_capacity import (
    KVRuntimeSpec,
    _enforce_coverage,
    _h2d_score_keys,
    profile_chunked_selection_capacity_multi_k,
)


def test_h2d_score_keys_matches_old_path_bf16() -> None:
    """New BF16 H2D + on-device float() matches old single .to(device, fp32).

    Numerical equivalence is required: the old call did the same upcast,
    just blocking. The new call should produce bit-equal results.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    target = torch.device("cuda")
    host = torch.randn((4, 16384, 128), dtype=torch.bfloat16).pin_memory()

    new_path = _h2d_score_keys(host, target=target, pin_memory=True)

    old_path = host.to(device=target, dtype=torch.float32, non_blocking=True)
    torch.cuda.synchronize(target)

    assert new_path.dtype == torch.float32
    assert new_path.device.type == target.type
    assert torch.equal(new_path, old_path), "BF16->FP32 upcast must be value-identical"


def test_h2d_score_keys_matches_old_path_fp16() -> None:
    """FP16 path also produces correct upcast."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    target = torch.device("cuda")
    host = torch.randn((4, 1024, 128), dtype=torch.float16).pin_memory()

    new_path = _h2d_score_keys(host, target=target, pin_memory=True)
    old_path = host.to(device=target, dtype=torch.float32, non_blocking=True)
    torch.cuda.synchronize(target)

    assert new_path.dtype == torch.float32
    assert torch.equal(new_path, old_path)


def test_h2d_score_keys_passthrough_fp32() -> None:
    """FP32 host tensors do not need an on-device upcast; passthrough."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    target = torch.device("cuda")
    host = torch.randn((2, 64, 128), dtype=torch.float32).pin_memory()
    out = _h2d_score_keys(host, target=target, pin_memory=True)
    torch.cuda.synchronize(target)
    assert out.dtype == torch.float32
    assert out.device.type == target.type


def test_enforce_coverage_full_pool_returns_one() -> None:
    """Full pool coverage returns 1.0 without raising."""
    coverage = _enforce_coverage(
        host_pool_tokens=32_768,
        candidate_tokens=32_768,
        allow_partial_coverage=False,
    )
    assert coverage == 1.0


def test_enforce_coverage_partial_pool_raises_by_default() -> None:
    """Partial coverage raises ValueError by default. This is the Phase 17 bug guard."""
    with pytest.raises(ValueError, match="host_pool_coverage"):
        _enforce_coverage(
            host_pool_tokens=16_384,
            candidate_tokens=1_048_576,
            allow_partial_coverage=False,
        )


def test_enforce_coverage_partial_allowed_when_flagged() -> None:
    """Partial coverage is allowed when explicitly flagged (back-compat path)."""
    coverage = _enforce_coverage(
        host_pool_tokens=16_384,
        candidate_tokens=1_048_576,
        allow_partial_coverage=True,
    )
    assert 0.0 < coverage < 1.0


def test_chunked_selection_refuses_partial_pool_by_default() -> None:
    """End-to-end: profile_chunked_selection_capacity_multi_k refuses partial pool."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    spec = KVRuntimeSpec()
    with pytest.raises(ValueError, match="host_pool_coverage"):
        profile_chunked_selection_capacity_multi_k(
            candidate_tokens=65_536,
            k_tokens_values=(96,),
            spec=spec,
            query_len=64,
            chunk_tokens=16_384,
            source_pool_chunks=1,  # 16384 < 65536 -> partial coverage
            trials=2,
            warmup_trials=1,
        )


def test_chunked_selection_full_pool_succeeds() -> None:
    """Full pool coverage path runs end-to-end and returns the expected schema."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    spec = KVRuntimeSpec()
    rows = profile_chunked_selection_capacity_multi_k(
        candidate_tokens=32_768,
        k_tokens_values=(96,),
        spec=spec,
        query_len=64,
        chunk_tokens=16_384,
        source_pool_chunks=2,  # 2 * 16384 = 32768 -> full coverage
        trials=2,
        warmup_trials=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["host_pool_coverage"] == pytest.approx(1.0)
    assert row["k"] == 96
    assert row["candidate_tokens"] == 32_768
