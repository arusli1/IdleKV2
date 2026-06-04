"""Tests for cross-model repair gates."""

from __future__ import annotations

from phases.phase10_expansion.src.model_transfer import evaluate_model_transfer_rows


def test_model_transfer_recommends_appendix_candidate_for_clean_budget() -> None:
    rows = [
        {
            "task": "clean_suite",
            "base_context_budget": 8192,
            "k": k,
            "condition_a": 1.0,
            "b_match": 0.35,
            "idlekv": idlekv,
            "random_k": 0.38,
            "oldest_k": 0.35,
            "gold_k": 1.0,
        }
        for k, idlekv in [(48, 0.65), (96, 0.95)]
    ]

    rec = evaluate_model_transfer_rows(rows)[0]

    assert rec["base_context_budget"] == 8192
    assert rec["best_k"] == 96
    assert rec["best_gain"] == 0.6
    assert rec["k96_gain"] == 0.6
    assert rec["appendix_candidate"] is True
    assert rec["action"] == "appendix_candidate"


def test_model_transfer_rejects_when_full_cache_fails() -> None:
    rows = [
        {
            "task": "clean_suite",
            "base_context_budget": 16384,
            "k": k,
            "condition_a": 0.25,
            "b_match": 0.0,
            "idlekv": 0.5,
            "random_k": 0.0,
            "oldest_k": 0.0,
            "gold_k": 1.0,
        }
        for k in [48, 96]
    ]

    rec = evaluate_model_transfer_rows(rows)[0]

    assert rec["appendix_candidate"] is False
    assert rec["action"] == "reject_model_cannot_solve_full_cache"


def test_model_transfer_rejects_when_controls_explain_gain() -> None:
    rows = [
        {
            "task": "clean_suite",
            "base_context_budget": 8192,
            "k": k,
            "condition_a": 1.0,
            "b_match": 0.25,
            "idlekv": idlekv,
            "random_k": random_k,
            "oldest_k": 0.25,
            "gold_k": 1.0,
        }
        for k, idlekv, random_k in [(48, 0.50, 0.48), (96, 0.80, 0.78)]
    ]

    rec = evaluate_model_transfer_rows(rows)[0]

    assert rec["appendix_candidate"] is False
    assert rec["max_control_lift"] == 0.53
    assert rec["action"] == "reject_controls_explain_gain"
