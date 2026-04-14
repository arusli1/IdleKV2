"""Tests for idle-time scheduler tier gating."""

import torch
from unittest.mock import Mock

from idlekv.core.scheduler import IdleScheduler


def _make_mock_model():
    model = Mock()
    model.config = Mock()
    model.config.num_hidden_layers = 2
    model.config.num_attention_heads = 2
    model.config.num_key_value_heads = 2
    model.config.hidden_size = 8
    param = torch.tensor([1.0], device="cpu", dtype=torch.float32)
    model.parameters.return_value = iter([param])
    return model


def test_scheduler_skips_phase2_for_100ms_budget(monkeypatch):
    shadow_buffer = Mock()
    query_buffer = Mock()
    query_buffer.count = 1
    full_kv_store = Mock()
    full_kv_store.__len__ = Mock(return_value=1)

    model = _make_mock_model()
    scheduler = IdleScheduler(
        shadow_buffer=shadow_buffer,
        query_buffer=query_buffer,
        full_kv_store=full_kv_store,
        model=model,
        budget_per_layer=8,
        num_layers=2,
    )

    phase1_called = {"value": False}
    phase2_called = {"value": False}

    def fake_phase1(**kwargs):
        phase1_called["value"] = True
        return kwargs["past_key_values"]

    def fake_phase2(**kwargs):
        phase2_called["value"] = True
        return kwargs["past_key_values"], 2

    monkeypatch.setattr("idlekv.core.scheduler.phase1_rescore", fake_phase1)
    monkeypatch.setattr("idlekv.core.scheduler.phase2_refresh", fake_phase2)

    result = scheduler.run(
        past_key_values=((torch.zeros(1, 2, 4, 4), torch.zeros(1, 2, 4, 4)),),
        generated_kv=[],
        max_time_ms=100,
        num_generated=0,
    )

    assert phase1_called["value"] is True
    assert phase2_called["value"] is False
    assert result.phase1_ran is True
    assert result.phase2_ran is False
