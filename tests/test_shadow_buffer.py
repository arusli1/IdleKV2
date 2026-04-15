"""Tests for ShadowBuffer."""

import torch
import pytest
from idlekv.core.shadow_buffer import ShadowBuffer


def test_push_and_get():
    buf = ShadowBuffer(num_layers=2, max_size=4, num_kv_heads=2, head_dim=8, device="cpu", dtype=torch.float32)

    k = torch.randn(2, 3, 8)  # 3 evicted tokens
    v = torch.randn(2, 3, 8)
    positions = torch.tensor([5, 6, 7])
    buf.push(0, k, v, positions=positions)

    got_k, got_v, got_pos = buf.get(0)
    assert got_k.shape == (2, 3, 8)
    assert torch.allclose(got_k, k)
    assert torch.equal(got_pos, positions)


def test_fifo_eviction():
    buf = ShadowBuffer(num_layers=1, max_size=4, num_kv_heads=1, head_dim=4, device="cpu", dtype=torch.float32)

    # Push 3, then push 3 more (should wrap, keeping last 4)
    k1 = torch.ones(1, 3, 4) * 1.0
    buf.push(0, k1, k1, positions=torch.tensor([0, 1, 2]))

    k2 = torch.ones(1, 3, 4) * 2.0
    buf.push(0, k2, k2, positions=torch.tensor([3, 4, 5]))

    got_k, _, got_pos = buf.get(0)
    assert got_k.shape == (1, 4, 4)
    assert torch.equal(got_pos, torch.tensor([2, 3, 4, 5]))
    assert torch.allclose(got_k[:, 0, :], k1[:, 2, :])
    assert torch.allclose(got_k[:, 1:, :], k2)


def test_empty_buffer():
    buf = ShadowBuffer(num_layers=1, max_size=4, num_kv_heads=2, head_dim=8, device="cpu")
    k, v, pos = buf.get(0)
    assert k.shape[1] == 0
    assert pos.numel() == 0


def test_clear():
    buf = ShadowBuffer(num_layers=2, max_size=4, num_kv_heads=1, head_dim=4, device="cpu")
    buf.push(0, torch.randn(1, 2, 4), torch.randn(1, 2, 4), positions=torch.tensor([0, 1]))
    buf.push(1, torch.randn(1, 2, 4), torch.randn(1, 2, 4), positions=torch.tensor([2, 3]))

    buf.clear(0)
    k0, _, pos0 = buf.get(0)
    k1, _, pos1 = buf.get(1)
    assert k0.shape[1] == 0
    assert pos0.numel() == 0
    assert k1.shape[1] == 2
    assert torch.equal(pos1, torch.tensor([2, 3]))

    buf.clear()
    k1, _, pos1 = buf.get(1)
    assert k1.shape[1] == 0
    assert pos1.numel() == 0


def test_overflow():
    """Push more tokens than max_size at once."""
    buf = ShadowBuffer(num_layers=1, max_size=4, num_kv_heads=1, head_dim=4, device="cpu", dtype=torch.float32)
    k = torch.randn(1, 10, 4)  # 10 > max_size=4
    v = torch.randn(1, 10, 4)
    positions = torch.arange(10)
    buf.push(0, k, v, positions=positions)

    got_k, got_v, got_pos = buf.get(0)
    assert got_k.shape == (1, 4, 4)
    # Should keep the last 4 tokens
    assert torch.allclose(got_k, k[:, -4:, :])
    assert torch.equal(got_pos, positions[-4:])


def test_memory_bytes():
    buf = ShadowBuffer(num_layers=32, max_size=256, num_kv_heads=8, head_dim=128,
                       device="cpu", dtype=torch.float16)
    expected = 32 * 2 * 8 * 256 * 128 * 2  # layers * (k+v) * heads * size * dim * fp16
    assert buf.memory_bytes == expected
