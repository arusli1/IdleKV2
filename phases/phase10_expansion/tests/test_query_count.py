"""Tests for query-count breadth gates."""

from __future__ import annotations

from phases.phase10_expansion.src.query_count import evaluate_query_count_breadth


def test_evaluate_query_count_breadth_promotes_appendix_when_controls_stay_matched() -> None:
    rows = [
        {
            "task": "mq_niah_8q_clean_suite",
            "k": 48,
            "condition_a": 1.0,
            "b_match": 0.6,
            "idlekv": 0.7,
            "random_k": 0.6,
            "oldest_k": 0.6,
            "gold_k": 1.0,
        },
        {
            "task": "mq_niah_8q_clean_suite",
            "k": 96,
            "condition_a": 1.0,
            "b_match": 0.6,
            "idlekv": 0.95,
            "random_k": 0.6,
            "oldest_k": 0.6,
            "gold_k": 1.0,
        },
    ]

    recommendation = evaluate_query_count_breadth(rows)[0]

    assert recommendation["query_count"] == 8
    assert recommendation["best_k"] == 96
    assert recommendation["appendix_ok"] is True
    assert recommendation["main_ok"] is True


def test_evaluate_query_count_breadth_rejects_content_agnostic_control_lift() -> None:
    rows = [
        {
            "task": "mq_niah_3q_clean_suite",
            "k": 96,
            "condition_a": 1.0,
            "b_match": 0.25,
            "idlekv": 0.75,
            "random_k": 0.55,
            "oldest_k": 0.25,
            "gold_k": 1.0,
        }
    ]

    recommendation = evaluate_query_count_breadth(rows)[0]

    assert recommendation["appendix_ok"] is False
    assert recommendation["action"] == "do_not_promote"


def test_evaluate_query_count_breadth_rejects_saturated_matched_baseline() -> None:
    rows = [
        {
            "task": "mq_niah_2q_clean_suite",
            "k": 48,
            "condition_a": 1.0,
            "b_match": 0.9,
            "idlekv": 1.0,
            "random_k": 0.9,
            "oldest_k": 0.9,
            "gold_k": 1.0,
        }
    ]

    recommendation = evaluate_query_count_breadth(rows)[0]

    assert recommendation["appendix_ok"] is False
    assert recommendation["full_vs_matched_gap"] == 0.1
