"""Tests for Phase 10 specificity summaries."""

from __future__ import annotations

import csv
from pathlib import Path

from phases.phase10_expansion.src.specificity import (
    recommend_specificity_followup,
    summarize_specificity_payload,
    write_specificity_csv,
)


def test_summarize_specificity_payload_reports_gains_and_overlap() -> None:
    payload = {
        "artifact_path": "artifact.json",
        "config": {
            "task": "clean_suite",
            "base_context_budget": 16384,
            "num_samples": 1,
            "query_scoring_mode": "exact_q",
            "wrong_query_mode": "donor_q2",
        },
        "rows": [
            {
                "k": 48,
                "b_match_score": 0.25,
                "b_match_active_overlap_fraction": 0.1,
                "stale_q_k_score": 0.25,
                "stale_q_k_active_overlap_fraction": 0.1,
                "wrong_q_k_score": 0.5,
                "wrong_q_k_active_overlap_fraction": 0.2,
                "refresh_k_score": 0.75,
                "refresh_k_active_overlap_fraction": 0.6,
                "refresh_selected_from_evicted_fraction": 0.5,
                "refresh_dropped_base_fraction": 0.25,
                "refresh_jaccard_with_b_match": 0.75,
                "refresh_jaccard_with_idlekv": 0.5,
                "idlekv_score": 1.0,
                "idlekv_active_overlap_fraction": 0.8,
                "oracle_k_score": 1.0,
                "oracle_k_active_overlap_fraction": 1.0,
            },
            {
                "k": 48,
                "b_match_score": 0.5,
                "b_match_active_overlap_fraction": 0.2,
                "stale_q_k_score": 0.25,
                "stale_q_k_active_overlap_fraction": 0.1,
                "wrong_q_k_score": 0.25,
                "wrong_q_k_active_overlap_fraction": 0.1,
                "refresh_k_score": 0.5,
                "refresh_k_active_overlap_fraction": 0.4,
                "refresh_selected_from_evicted_fraction": 0.25,
                "refresh_dropped_base_fraction": 0.125,
                "refresh_jaccard_with_b_match": 0.5,
                "refresh_jaccard_with_idlekv": 0.25,
                "idlekv_score": 0.75,
                "idlekv_active_overlap_fraction": 0.7,
                "oracle_k_score": 1.0,
                "oracle_k_active_overlap_fraction": 1.0,
            },
        ],
    }

    rows = summarize_specificity_payload(payload)
    by_condition = {row["condition"]: row for row in rows}

    assert by_condition["Matched"]["mean_score"] == 0.375
    assert by_condition["Matched"]["mean_gain_vs_matched"] == 0.0
    assert by_condition["Matched"]["score_ci95_low"] <= 0.375
    assert by_condition["Matched"]["score_ci95_high"] >= 0.375
    assert by_condition["IdleKV"]["mean_score"] == 0.875
    assert by_condition["IdleKV"]["mean_gain_vs_matched"] == 0.5
    assert by_condition["IdleKV"]["gain_ci95_low"] <= 0.5
    assert by_condition["IdleKV"]["gain_ci95_high"] >= 0.5
    assert by_condition["IdleKV"]["win_rate_vs_matched"] == 1.0
    assert by_condition["IdleKV"]["loss_rate_vs_matched"] == 0.0
    assert by_condition["Refresh-K"]["mean_score"] == 0.625
    assert by_condition["Refresh-K"]["refresh_scope"] == "buffered_active_plus_evicted"
    assert by_condition["Refresh-K"]["mean_refresh_selected_from_evicted_fraction"] == 0.375
    assert by_condition["Refresh-K"]["mean_refresh_dropped_base_fraction"] == 0.1875
    assert by_condition["Refresh-K"]["mean_refresh_jaccard_with_b_match"] == 0.625
    assert by_condition["Refresh-K"]["mean_refresh_jaccard_with_idlekv"] == 0.375
    assert by_condition["WrongQ-K"]["wrong_query_mode"] == "donor_q2"


def test_write_specificity_csv(tmp_path: Path) -> None:
    output = tmp_path / "specificity.csv"
    write_specificity_csv(
        [
            {
                "task": "clean_suite",
                "base_context_budget": 16384,
                "k": 48,
                "condition": "IdleKV",
                "mean_score": 0.875,
                "mean_gain_vs_matched": 0.5,
                "mean_overlap_fraction": 0.75,
                "n_rows": 2,
                "num_samples": 1,
                "query_scoring_mode": "exact_q",
                "wrong_query_mode": "donor_q2",
                "refresh_scope": "",
                "mean_refresh_selected_from_evicted_fraction": "",
                "mean_refresh_dropped_base_fraction": "",
                "mean_refresh_jaccard_with_b_match": "",
                "mean_refresh_jaccard_with_idlekv": "",
                "artifact_path": "artifact.json",
            }
        ],
        output,
    )

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["condition"] == "IdleKV"
    assert rows[0]["mean_score"] == "0.875"
    assert "score_ci95_low" in rows[0]
    assert "gain_ci95_high" in rows[0]
    assert "win_rate_vs_matched" in rows[0]


def test_recommend_specificity_followup_promotes_clean_separation() -> None:
    rows = [
        {"k": 48, "condition": "Matched", "mean_score": 0.25},
        {"k": 48, "condition": "StaleQ-K", "mean_score": 0.30},
        {"k": 48, "condition": "WrongQ-K", "mean_score": 0.35},
        {"k": 48, "condition": "Refresh-K", "mean_score": 0.72},
        {"k": 48, "condition": "IdleKV", "mean_score": 0.75},
        {"k": 48, "condition": "Gold-K", "mean_score": 0.90},
    ]

    recommendation = recommend_specificity_followup(rows)[0]

    assert recommendation["action"] == "promote_locked_specificity_run"
    assert recommendation["idle_vs_stale"] == 0.45
    assert recommendation["saturated"] is False


def test_recommend_specificity_followup_detects_saturation_and_stale_failure() -> None:
    saturated = recommend_specificity_followup(
        [
            {"k": 96, "condition": "Matched", "mean_score": 0.30},
            {"k": 96, "condition": "StaleQ-K", "mean_score": 0.40},
            {"k": 96, "condition": "WrongQ-K", "mean_score": 0.40},
            {"k": 96, "condition": "Refresh-K", "mean_score": 1.00},
            {"k": 96, "condition": "IdleKV", "mean_score": 1.00},
            {"k": 96, "condition": "Gold-K", "mean_score": 1.00},
        ]
    )[0]
    stale_failure = recommend_specificity_followup(
        [
            {"k": 48, "condition": "Matched", "mean_score": 0.30},
            {"k": 48, "condition": "StaleQ-K", "mean_score": 0.71},
            {"k": 48, "condition": "WrongQ-K", "mean_score": 0.40},
            {"k": 48, "condition": "Refresh-K", "mean_score": 0.80},
            {"k": 48, "condition": "IdleKV", "mean_score": 0.75},
            {"k": 48, "condition": "Gold-K", "mean_score": 0.90},
        ]
    )[0]

    assert saturated["action"] == "rerun_lower_k_before_promotion"
    assert stale_failure["action"] == "demote_specificity_no_stale_separation"
