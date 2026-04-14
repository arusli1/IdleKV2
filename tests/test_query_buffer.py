"""Tests for QueryBuffer."""

import torch
from idlekv.core.query_buffer import QueryBuffer


def test_append_and_get():
    qb = QueryBuffer(buffer_size=4, hidden_dim=8, device="cpu")

    for i in range(3):
        qb.append(torch.ones(8) * (i + 1))

    result = qb.get()
    assert result.shape == (3, 8)
    assert result[0, 0].item() == 1.0  # oldest
    assert result[2, 0].item() == 3.0  # newest


def test_wrap_around():
    qb = QueryBuffer(buffer_size=3, hidden_dim=4, device="cpu")

    for i in range(5):
        qb.append(torch.ones(4) * (i + 1))

    result = qb.get()
    assert result.shape == (3, 4)
    # Should contain tokens 3, 4, 5 (oldest-to-newest)
    assert result[0, 0].item() == 3.0
    assert result[1, 0].item() == 4.0
    assert result[2, 0].item() == 5.0


def test_empty():
    qb = QueryBuffer(buffer_size=4, hidden_dim=8, device="cpu")
    result = qb.get()
    assert result.shape == (0, 8)


def test_clear():
    qb = QueryBuffer(buffer_size=4, hidden_dim=8, device="cpu")
    qb.append(torch.ones(8))
    qb.clear()
    result = qb.get()
    assert result.shape == (0, 8)


def test_multilayer_get_by_layer():
    qb = QueryBuffer(buffer_size=3, num_layers=2, hidden_dim=4, device="cpu")

    qb.append(torch.stack([
        torch.full((4,), 1.0),
        torch.full((4,), 10.0),
    ], dim=0))
    qb.append(torch.stack([
        torch.full((4,), 2.0),
        torch.full((4,), 20.0),
    ], dim=0))

    full = qb.get()
    layer0 = qb.get(layer_idx=0)
    layer1 = qb.get(layer_idx=1)

    assert full.shape == (2, 2, 4)
    assert layer0.shape == (2, 4)
    assert layer1.shape == (2, 4)
    assert layer0[0, 0].item() == 1.0
    assert layer0[1, 0].item() == 2.0
    assert layer1[0, 0].item() == 10.0
    assert layer1[1, 0].item() == 20.0
