"""Tests for full frontier promotion gates."""

from __future__ import annotations

from phases.phase10_expansion.src.frontier import evaluate_frontier_promotion


def test_frontier_promotion_accepts_clean_increasing_curve() -> None:
    rows = [
        {
            "task": "mq_niah_8q_clean_suite",
            "k": k,
            "b_match": 0.5,
            "idlekv": idlekv,
            "random_k": 0.5,
            "oldest_k": 0.5,
            "gold_k": gold,
        }
        for k, idlekv, gold in [
            (8, 0.50, 0.55),
            (16, 0.51, 0.70),
            (32, 0.58, 0.82),
            (48, 0.68, 0.95),
            (96, 0.94, 0.98),
        ]
    ]

    rec = evaluate_frontier_promotion(rows)

    assert rec["query_count"] == 8
    assert rec["points"] == 5
    assert rec["best_k"] == 96
    assert rec["best_gain"] == 0.44
    assert rec["promote"] is True


def test_frontier_promotion_rejects_endpoint_only_rows() -> None:
    rows = [
        {
            "task": "mq_niah_8q_clean_suite",
            "k": 48,
            "b_match": 0.5,
            "idlekv": 0.8,
            "random_k": 0.5,
            "oldest_k": 0.5,
            "gold_k": 1.0,
        },
        {
            "task": "mq_niah_8q_clean_suite",
            "k": 96,
            "b_match": 0.5,
            "idlekv": 0.95,
            "random_k": 0.5,
            "oldest_k": 0.5,
            "gold_k": 1.0,
        },
    ]

    rec = evaluate_frontier_promotion(rows)

    assert rec["promote"] is False
    assert rec["points"] == 2


def test_frontier_promotion_rejects_content_agnostic_controls() -> None:
    rows = [
        {
            "task": "mq_niah_2q_clean_suite",
            "k": k,
            "b_match": 0.0,
            "idlekv": idlekv,
            "random_k": random_k,
            "oldest_k": 0.0,
            "gold_k": 1.0,
        }
        for k, idlekv, random_k in [
            (8, 0.0, 0.0),
            (16, 0.2, 0.0),
            (32, 0.4, 0.3),
            (48, 0.7, 0.4),
            (96, 1.0, 0.6),
        ]
    ]

    rec = evaluate_frontier_promotion(rows)

    assert rec["promote"] is False
    assert rec["max_control_lift"] == 0.6


def test_frontier_promotion_marks_2q_as_review_boundary() -> None:
    rows = [
        {
            "task": "mq_niah_2q_clean_suite",
            "k": k,
            "b_match": 0.0,
            "idlekv": idlekv,
            "random_k": 0.0,
            "oldest_k": 0.0,
            "gold_k": 1.0,
        }
        for k, idlekv in [
            (8, 0.0),
            (16, 0.3),
            (32, 0.7),
            (48, 1.0),
            (96, 1.0),
        ]
    ]

    rec = evaluate_frontier_promotion(rows)

    assert rec["query_count"] == 2
    assert rec["boundary_review"] is True
    assert rec["promote"] is False
    assert rec["action"] == "review_easy_boundary_frontier"
