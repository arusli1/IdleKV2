"""Tests for ShadowBuffer."""

import torch
import pytest
from idlekv.core.shadow_buffer import ShadowBuffer


def test_push_and_get():
    buf = ShadowBuffer(num_layers=2, max_size=4, num_kv_heads=2, head_dim=8, device="cpu", dtype=torch.float32)

    k = torch.randn(2, 3, 8)  # 3 evicted tokens
    v = torch.randn(2, 3, 8)
    buf.push(0, k, v)

    got_k, got_v = buf.get(0)
    assert got_k.shape == (2, 3, 8)
    assert torch.allclose(got_k, k)


def test_fifo_eviction():
    buf = ShadowBuffer(num_layers=1, max_size=4, num_kv_heads=1, head_dim=4, device="cpu")

    # Push 3, then push 3 more (should wrap, keeping last 4)
    k1 = torch.ones(1, 3, 4) * 1.0
    buf.push(0, k1, k1)

    k2 = torch.ones(1, 3, 4) * 2.0
    buf.push(0, k2, k2)

    got_k, _ = buf.get(0)
    assert got_k.shape == (1, 4, 4)
    # Should contain: last 1 from k1 and all 3 from k2
    # (FIFO: oldest overwritten first)


def test_empty_buffer():
    buf = ShadowBuffer(num_layers=1, max_size=4, num_kv_heads=2, head_dim=8, device="cpu")
    k, v = buf.get(0)
    assert k.shape[1] == 0


def test_clear():
    buf = ShadowBuffer(num_layers=2, max_size=4, num_kv_heads=1, head_dim=4, device="cpu")
    buf.push(0, torch.randn(1, 2, 4), torch.randn(1, 2, 4))
    buf.push(1, torch.randn(1, 2, 4), torch.randn(1, 2, 4))

    buf.clear(0)
    k0, _ = buf.get(0)
    k1, _ = buf.get(1)
    assert k0.shape[1] == 0
    assert k1.shape[1] == 2

    buf.clear()
    k1, _ = buf.get(1)
    assert k1.shape[1] == 0


def test_overflow():
    """Push more tokens than max_size at once."""
    buf = ShadowBuffer(num_layers=1, max_size=4, num_kv_heads=1, head_dim=4, device="cpu", dtype=torch.float32)
    k = torch.randn(1, 10, 4)  # 10 > max_size=4
    v = torch.randn(1, 10, 4)
    buf.push(0, k, v)

    got_k, got_v = buf.get(0)
    assert got_k.shape == (1, 4, 4)
    # Should keep the last 4 tokens
    assert torch.allclose(got_k, k[:, -4:, :])


def test_memory_bytes():
    buf = ShadowBuffer(num_layers=32, max_size=256, num_kv_heads=8, head_dim=128,
                       device="cpu", dtype=torch.float16)
    expected = 32 * 2 * 8 * 256 * 128 * 2  # layers * (k+v) * heads * size * dim * fp16
    assert buf.memory_bytes == expected
