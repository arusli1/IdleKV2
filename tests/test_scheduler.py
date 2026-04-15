"""Tests for idle-time scheduler tier gating."""

import torch
from unittest.mock import Mock

from idlekv.core.scheduler import IdleScheduler
from idlekv.core.anytime_repair import ColdCandidateBatch, LayerRepairOutcome


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


def test_scheduler_phase1_only_skips_phase2_even_with_large_budget(monkeypatch):
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
        return kwargs["past_key_values"], 1

    monkeypatch.setattr("idlekv.core.scheduler.phase1_rescore", fake_phase1)
    monkeypatch.setattr("idlekv.core.scheduler.phase2_refresh", fake_phase2)

    result = scheduler.run(
        past_key_values=((torch.zeros(1, 2, 4, 4), torch.zeros(1, 2, 4, 4)),),
        generated_kv=[],
        max_time_ms=1000,
        num_generated=0,
        phases=1,
    )

    assert phase1_called["value"] is True
    assert phase2_called["value"] is False
    assert result.phase1_ran is True
    assert result.phase2_ran is False


def test_scheduler_phase2_only_skips_phase1(monkeypatch):
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
        max_time_ms=1000,
        num_generated=0,
        phases=2,
    )

    assert phase1_called["value"] is False
    assert phase2_called["value"] is True
    assert result.phase1_ran is False
    assert result.phase2_ran is True


def test_sampled_policy_skips_work_below_min_idle():
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
        retained_positions=[torch.arange(4), torch.arange(4)],
        prefill_importance_scores=[torch.ones(16), torch.ones(16)],
    )

    result = scheduler.run(
        past_key_values=((torch.zeros(1, 2, 4, 4), torch.zeros(1, 2, 4, 4)),) * 2,
        generated_kv=[],
        max_time_ms=10,
        num_generated=0,
        policy="sampled_spans",
    )

    assert result.phase1_ran is False
    assert result.sampled_rounds == 0
    assert result.policy == "sampled_spans"


def test_sampled_policy_records_cold_span_metrics(monkeypatch):
    shadow_buffer = Mock()
    shadow_buffer.get = Mock(return_value=(
        torch.empty(2, 0, 4),
        torch.empty(2, 0, 4),
        torch.empty(0, dtype=torch.long),
    ))
    shadow_buffer.clear = Mock()
    shadow_buffer.push = Mock()

    query_buffer = Mock()
    query_buffer.count = 1
    query_buffer.get = Mock(return_value=torch.randn(2, 8))
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
        retained_positions=[torch.arange(4), torch.arange(4)],
        prefill_importance_scores=[torch.ones(16), torch.ones(16)],
        sample_spans_per_layer=1,
    )

    phase1_called = {"value": False}

    def fake_phase1(**kwargs):
        phase1_called["value"] = True
        return kwargs["past_key_values"]

    def fake_sample(**kwargs):
        return ColdCandidateBatch(
            keys=torch.randn(2, 2, 4),
            values=torch.randn(2, 2, 4),
            positions=torch.tensor([6, 7]),
            bytes_loaded=64,
            span_count=1,
        )

    def fake_repair(**kwargs):
        current_k = kwargs["current_k"]
        current_v = kwargs["current_v"]
        retained_positions = kwargs["retained_positions"]
        return LayerRepairOutcome(
            new_k=current_k,
            new_v=current_v,
            retained_positions=retained_positions,
            shadow_k=torch.empty(2, 0, 4),
            shadow_v=torch.empty(2, 0, 4),
            shadow_positions=torch.empty(0, dtype=torch.long),
            candidate_count=6,
            shadow_candidate_count=0,
            sampled_candidate_count=2,
        )

    monkeypatch.setattr("idlekv.core.scheduler.phase1_rescore", fake_phase1)
    monkeypatch.setattr("idlekv.core.scheduler.sample_cold_spans", fake_sample)
    monkeypatch.setattr("idlekv.core.scheduler.repair_layer_pool", fake_repair)

    result = scheduler.run(
        past_key_values=((torch.zeros(1, 2, 4, 4), torch.zeros(1, 2, 4, 4)),) * 2,
        generated_kv=[],
        max_time_ms=100,
        num_generated=0,
        policy="sampled_spans",
    )

    assert phase1_called["value"] is True
    assert result.sampled_rounds >= 1
    assert result.sampled_layers >= 1
    assert result.sampled_tokens >= 2
    assert result.cpu_bytes_loaded >= 64
