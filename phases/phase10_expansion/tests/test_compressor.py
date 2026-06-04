"""Tests for first-stage retention-rule smoke gates."""

from __future__ import annotations

from phases.phase10_expansion.src.compressor import evaluate_compressor_smoke


def test_compressor_smoke_recommends_locked_followup_for_clean_budget() -> None:
    rows = [
        {
            "task": "clean_suite",
            "base_context_budget": 16384,
            "k": k,
            "condition_a": 1.0,
            "b_match": 0.25,
            "idlekv": idlekv,
            "random_k": 0.26,
            "oldest_k": 0.24,
            "gold_k": gold,
        }
        for k, idlekv, gold in [(48, 0.40, 0.80), (96, 0.62, 0.95), (128, 0.72, 1.0)]
    ]

    rec = evaluate_compressor_smoke(rows)[0]

    assert rec["base_context_budget"] == 16384
    assert rec["best_k"] == 128
    assert rec["best_gain"] == 0.47
    assert rec["lock_followup"] is True
    assert rec["action"] == "lock_followup"


def test_compressor_smoke_rejects_when_controls_explain_gain() -> None:
    rows = [
        {
            "task": "clean_suite",
            "base_context_budget": 16384,
            "k": k,
            "condition_a": 1.0,
            "b_match": 0.25,
            "idlekv": idlekv,
            "random_k": random_k,
            "oldest_k": 0.25,
            "gold_k": 1.0,
        }
        for k, idlekv, random_k in [(48, 0.40, 0.36), (96, 0.62, 0.60), (128, 0.72, 0.70)]
    ]

    rec = evaluate_compressor_smoke(rows)[0]

    assert rec["lock_followup"] is False
    assert rec["max_control_lift"] == 0.45
    assert rec["action"] == "do_not_lock_controls_explain_gain"


def test_compressor_smoke_rejects_saturated_budget() -> None:
    rows = [
        {
            "task": "clean_suite",
            "base_context_budget": 18432,
            "k": k,
            "condition_a": 1.0,
            "b_match": 0.95,
            "idlekv": 1.0,
            "random_k": 0.95,
            "oldest_k": 0.95,
            "gold_k": 1.0,
        }
        for k in [48, 96, 128]
    ]

    rec = evaluate_compressor_smoke(rows)[0]

    assert rec["lock_followup"] is False
    assert rec["full_vs_matched_gap"] == 0.05
    assert rec["action"] == "do_not_lock_low_gain"
