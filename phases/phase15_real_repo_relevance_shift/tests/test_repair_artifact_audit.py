from __future__ import annotations

import json
import subprocess
import sys

from phases.phase15_real_repo_relevance_shift.scripts.audit_phase15_repair_artifact import (
    audit_repair_artifact,
    followup_gate,
    repair_gate,
)


def test_audit_repair_artifact_reports_control_lifts() -> None:
    payload = {
        "manifest_hash": "abc",
        "protocol_hash": "def",
        "stage": "repair_smoke",
        "rows": [
            {
                "k": 96,
                "example_id": "a",
                "repo_id": "repo1",
                "idlekv_score": 1.0,
                "condition_a_score": 1.0,
                "q1_score": 1.0,
                "b_match_score": 0.0,
                "random_k_score": 0.0,
                "tool_file_k_score": 0.0,
                "anchor_window_k_score": 0.0,
                "file_gated_idlekv_score": 1.0,
                "lexical_anchor_k_score": 0.0,
                "tool_file_k_selected_from_file_fraction": 1.0,
                "tool_file_k_budget_matched": True,
                "anchor_window_k_budget_matched": True,
                "file_gated_idlekv_selected_from_file_fraction": 1.0,
                "file_gated_idlekv_backfill_count": 0,
                "file_gated_idlekv_budget_matched": True,
                "file_gated_idlekv_event_contains_q2_path": True,
                "lexical_anchor_k_selected_from_file_fraction": 0.5,
                "lexical_anchor_k_term_count": 4,
                "lexical_anchor_k_answer_leak_flag": False,
                "lexical_anchor_k_budget_matched": True,
                "lexical_anchor_k_event_contains_q2_path": True,
                "phase15_manifest_audit": {"passed": True},
                "wrong_event_donor_example_id": "donor-a",
                "wrong_event_donor_repo_id": "repo3",
                "wrong_event_donor_answer": "OtherA",
                "wrong_event_donor_tool_event_sha256": "sha-a",
            },
            {
                "k": 96,
                "example_id": "b",
                "repo_id": "repo2",
                "idlekv_score": 0.0,
                "condition_a_score": 1.0,
                "q1_score": 1.0,
                "b_match_score": 0.0,
                "random_k_score": 1.0,
                "tool_file_k_score": 0.0,
                "anchor_window_k_score": 0.0,
                "file_gated_idlekv_score": 1.0,
                "lexical_anchor_k_score": 0.0,
                "tool_file_k_selected_from_file_fraction": 0.5,
                "tool_file_k_budget_matched": True,
                "anchor_window_k_budget_matched": True,
                "file_gated_idlekv_selected_from_file_fraction": 0.75,
                "file_gated_idlekv_backfill_count": 1,
                "file_gated_idlekv_budget_matched": True,
                "file_gated_idlekv_event_contains_q2_path": True,
                "lexical_anchor_k_selected_from_file_fraction": 0.25,
                "lexical_anchor_k_term_count": 2,
                "lexical_anchor_k_answer_leak_flag": False,
                "lexical_anchor_k_budget_matched": True,
                "lexical_anchor_k_event_contains_q2_path": False,
                "phase15_manifest_audit": {"passed": True},
                "wrong_event_donor_example_id": "donor-b",
                "wrong_event_donor_repo_id": "repo4",
                "wrong_event_donor_answer": "OtherB",
                "wrong_event_donor_tool_event_sha256": "sha-b",
            },
        ],
    }

    result = audit_repair_artifact(payload, bootstrap_draws=50, bootstrap_seed=1)

    assert result["manifest_hash"] == "abc"
    assert result["artifact_checks"]["all_manifest_audits_passed"]
    assert not result["artifact_checks"]["has_duplicate_example_rows_by_k"]
    assert result["artifact_checks"]["wrong_event_donor_metadata_complete"]
    k96 = result["k_results"]["k96"]
    assert k96["n_rows"] == 2
    assert k96["repo_count"] == 2
    assert k96["mean_idlekv"] == 0.5
    assert k96["mean_idlekv_minus_b_match"] == 0.5
    assert k96["wins_vs_b_match"] == 1
    assert k96["losses_vs_random_k"] == 1
    assert k96["repo_positive_count_vs_b_match"] == 1
    assert k96["repo_nonnegative_count_vs_b_match"] == 2
    assert k96["repo_median_lift_vs_b_match"] == 0.5
    assert k96["repo_min_lift_vs_b_match"] == 0.0
    assert k96["repo_max_lift_vs_b_match"] == 1.0
    assert k96["repo_lift_vs_b_match"]["median"] == 0.5
    assert k96["repo_lift_vs_b_match"]["positive_repos"] == 1
    assert k96["bootstrap_idlekv_minus_b_match"]["draws"] == 50
    assert k96["mean_idlekv_minus_anchor_window_k"] == 0.5
    assert k96["mean_file_gated_idlekv"] == 1.0
    assert k96["mean_file_gated_idlekv_minus_idlekv"] == 0.5
    assert k96["wins_file_gated_idlekv_vs_idlekv"] == 1
    assert k96["repo_positive_count_file_gated_idlekv_vs_idlekv"] == 1
    assert k96["bootstrap_file_gated_idlekv_minus_b_match"]["draws"] == 50
    assert k96["mean_lexical_anchor_k"] == 0.0
    assert k96["mean_lexical_anchor_k_minus_idlekv"] == -0.5
    assert k96["min_tool_file_k_selected_from_file_fraction"] == 0.5
    assert k96["fraction_tool_file_k_budget_matched"] == 1.0
    assert k96["mean_file_gated_idlekv_selected_from_file_fraction"] == 0.875
    assert k96["min_file_gated_idlekv_selected_from_file_fraction"] == 0.75
    assert k96["mean_file_gated_idlekv_backfill_count"] == 0.5
    assert k96["fraction_file_gated_idlekv_event_contains_q2_path"] == 1.0
    assert k96["mean_lexical_anchor_k_selected_from_file_fraction"] == 0.375
    assert k96["mean_lexical_anchor_k_term_count"] == 3.0
    assert k96["lexical_anchor_k_answer_sanitized"]
    assert "answer-sanitized diagnostic" in k96["lexical_anchor_k_deployability_note"]
    assert k96["lexical_anchor_k_answer_leak_rows"] == 0
    assert k96["fraction_lexical_anchor_k_event_contains_q2_path"] == 0.5
    assert k96["fraction_file_gated_idlekv_budget_matched"] == 1.0
    assert k96["fraction_lexical_anchor_k_budget_matched"] == 1.0
    assert k96["fraction_anchor_window_k_budget_matched"] == 1.0
    assert result["sensitivity"]["exclude_cue_and_answer_retention"]["n_examples"] == 2
    assert "k96" in result["sensitivity"]["strict_repair_eligible"]["k_results"]


def test_followup_gate_reports_filegated_promotion_fields() -> None:
    audit = {
        "k_results": {
            "k96": {
                "mean_file_gated_idlekv": 0.50,
                "mean_file_gated_idlekv_minus_idlekv": 0.00,
            },
            "k192": {
                "mean_file_gated_idlekv": 0.80,
                "mean_file_gated_idlekv_minus_idlekv": 0.10,
                "mean_file_gated_idlekv_minus_tool_file_k": 0.50,
                "mean_file_gated_idlekv_minus_anchor_window_k": -0.05,
                "repo_positive_count_file_gated_idlekv_vs_idlekv": 7,
                "repo_negative_count_file_gated_idlekv_vs_idlekv": 2,
                "repo_positive_count_file_gated_idlekv_vs_tool_file_k": 9,
                "repo_negative_count_file_gated_idlekv_vs_tool_file_k": 1,
                "repo_positive_count_file_gated_idlekv_vs_anchor_window_k": 1,
                "repo_negative_count_file_gated_idlekv_vs_anchor_window_k": 8,
                "bootstrap_file_gated_idlekv_minus_idlekv": {
                    "mean": 0.10,
                    "low": 0.02,
                    "high": 0.18,
                    "draws": 50,
                },
                "bootstrap_file_gated_idlekv_minus_tool_file_k": {
                    "mean": 0.50,
                    "low": 0.20,
                    "high": 0.70,
                    "draws": 50,
                },
                "bootstrap_file_gated_idlekv_minus_anchor_window_k": {
                    "mean": -0.05,
                    "low": -0.10,
                    "high": 0.00,
                    "draws": 50,
                },
                "fraction_file_gated_idlekv_event_contains_q2_path": 1.0,
                "mean_file_gated_idlekv_selected_from_file_fraction": 0.75,
                "min_file_gated_idlekv_selected_from_file_fraction": 0.50,
                "fraction_file_gated_idlekv_budget_matched": 1.0,
                "fraction_tool_file_k_budget_matched": 1.0,
                "fraction_anchor_window_k_budget_matched": 1.0,
                "lexical_anchor_k_answer_leak_rows": 0,
                "fraction_lexical_anchor_k_budget_matched": 1.0,
                "lexical_anchor_k_answer_sanitized": True,
                "lexical_anchor_k_deployability_note": "answer-sanitized diagnostic",
            },
        },
        "sensitivity": {
            "exclude_cue_only_hits": {
                "n_examples": 12,
                "k_results": {"k192": {"mean_file_gated_idlekv_minus_idlekv": 0.08}},
            },
            "exclude_answer_retention": {
                "n_examples": 10,
                "k_results": {"k192": {"mean_file_gated_idlekv_minus_idlekv": 0.07}},
            },
            "exclude_cue_and_answer_retention": {
                "n_examples": 9,
                "k_results": {"k192": {"mean_file_gated_idlekv_minus_idlekv": 0.05}},
            },
        },
    }

    result = followup_gate(audit, primary_k=192, adjacent_k=96, min_sensitivity_examples=8)

    assert result["passed"]
    assert result["comparisons"]["mean_file_gated_idlekv_minus_idlekv"] == 0.10
    assert result["comparisons"]["mean_file_gated_idlekv_minus_tool_file_k"] == 0.50
    assert result["comparisons"]["mean_file_gated_idlekv_minus_anchor_window_k"] == -0.05
    assert result["metadata"]["fraction_file_gated_idlekv_event_contains_q2_path"] == 1.0
    assert result["metadata"]["lexical_anchor_k_answer_leak_rows"] == 0
    assert result["metadata"]["lexical_anchor_k_answer_sanitized"]
    assert result["repo_counts"]["repo_positive_file_gated_idlekv_vs_idlekv"] == 7
    assert result["repo_counts"]["repo_negative_file_gated_idlekv_vs_anchor_window_k"] == 8
    assert "eligible_for_one_cautious_main_sentence" in result["recommendation"]


def test_followup_gate_rejects_metadata_assisted_filegated_result() -> None:
    audit = {
        "k_results": {
            "k96": {
                "mean_file_gated_idlekv": 0.50,
                "mean_file_gated_idlekv_minus_idlekv": 0.00,
            },
            "k192": {
                "mean_file_gated_idlekv": 0.80,
                "mean_file_gated_idlekv_minus_idlekv": 0.10,
                "mean_file_gated_idlekv_minus_tool_file_k": 0.50,
                "mean_file_gated_idlekv_minus_anchor_window_k": -0.05,
                "fraction_file_gated_idlekv_event_contains_q2_path": 0.75,
                "mean_file_gated_idlekv_selected_from_file_fraction": 0.75,
                "fraction_file_gated_idlekv_budget_matched": 1.0,
                "fraction_tool_file_k_budget_matched": 1.0,
                "fraction_anchor_window_k_budget_matched": 1.0,
                "lexical_anchor_k_answer_leak_rows": 0,
                "fraction_lexical_anchor_k_budget_matched": 1.0,
            },
        },
        "sensitivity": {
            "exclude_cue_only_hits": {
                "n_examples": 12,
                "k_results": {"k192": {"mean_file_gated_idlekv_minus_idlekv": 0.08}},
            },
            "exclude_answer_retention": {
                "n_examples": 10,
                "k_results": {"k192": {"mean_file_gated_idlekv_minus_idlekv": 0.07}},
            },
            "exclude_cue_and_answer_retention": {
                "n_examples": 9,
                "k_results": {"k192": {"mean_file_gated_idlekv_minus_idlekv": 0.05}},
            },
        },
    }

    result = followup_gate(audit, primary_k=192, adjacent_k=96, min_sensitivity_examples=8)

    assert not result["passed"]
    assert not result["gate_results"]["event_path_available_for_all_rows"]
    assert "do_not_change_main_text" in result["recommendation"]


def test_repair_gate_requires_primary_lift_adjacent_and_toolfile_margin() -> None:
    audit = {
        "k_results": {
            "k96": {
                "mean_idlekv_minus_b_match": 0.0,
            },
            "k192": {
                "mean_idlekv_minus_b_match": 0.25,
                "wins_vs_tool_file_k": 3,
                "losses_vs_tool_file_k": 1,
                "wins_vs_anchor_window_k": 3,
                "losses_vs_anchor_window_k": 1,
                "repo_lift_vs_b_match": {"median": 0.25},
                "min_tool_file_k_selected_from_file_fraction": 0.5,
                "fraction_tool_file_k_budget_matched": 1.0,
                "fraction_anchor_window_k_budget_matched": 1.0,
                "bootstrap_idlekv_minus_b_match": {"low": 0.05},
                "bootstrap_idlekv_minus_random_k": {"low": 0.02},
                "bootstrap_idlekv_minus_oldest_k": {"low": 0.03},
                "bootstrap_idlekv_minus_stale_q_k": {"low": 0.04},
                "bootstrap_idlekv_minus_wrong_q_k": {"low": 0.04},
                "bootstrap_idlekv_minus_tool_file_k": {"low": 0.01},
            },
        },
        "artifact_checks": {
            "all_manifest_audits_passed": True,
            "has_duplicate_example_rows_by_k": False,
            "wrong_event_donor_metadata_complete": True,
        },
        "sensitivity": {
            "exclude_cue_only_hits": {
                "n_examples": 8,
                "k_results": {"k192": {"mean_idlekv_minus_b_match": 0.20}},
            },
            "exclude_answer_retention": {
                "n_examples": 8,
                "k_results": {"k192": {"mean_idlekv_minus_b_match": 0.20}},
            },
            "exclude_cue_and_answer_retention": {
                "n_examples": 8,
                "k_results": {"k192": {"mean_idlekv_minus_b_match": 0.20}},
            },
            "strict_repair_eligible": {
                "n_examples": 4,
                "repo_count": 4,
                "k_results": {"k192": {"mean_idlekv_minus_b_match": 0.25}},
            },
        },
    }

    result = repair_gate(
        audit,
        primary_k=192,
        adjacent_k=96,
        min_primary_lift=0.10,
        require_positive_ci=True,
        require_sensitivity=True,
        min_sensitivity_examples=8,
        toolfile_margin_rows=1,
        min_toolfile_file_fraction=0.10,
        anchor_window_margin_rows=0,
    )

    assert result["passed"]
    assert result["toolfile_win_loss_margin"] == 2
    assert result["anchor_window_win_loss_margin"] == 2
    assert result["repo_median_lift_vs_b_match"] == 0.25
    assert result["gate_results"]["sensitivity_ok"]
    assert "diagnostic" in result["anchor_window_reference_note"]


def test_repair_gate_fails_main_gate_when_label_assisted_anchor_window_dominates() -> None:
    audit = {
        "k_results": {
            "k96": {
                "mean_idlekv_minus_b_match": 0.0,
            },
            "k192": {
                "mean_idlekv_minus_b_match": 0.25,
                "wins_vs_tool_file_k": 3,
                "losses_vs_tool_file_k": 1,
                "wins_vs_anchor_window_k": 0,
                "losses_vs_anchor_window_k": 4,
                "repo_lift_vs_b_match": {"median": 0.25},
                "min_tool_file_k_selected_from_file_fraction": 0.5,
                "fraction_tool_file_k_budget_matched": 1.0,
                "fraction_anchor_window_k_budget_matched": 1.0,
                "bootstrap_idlekv_minus_b_match": {"low": 0.05},
                "bootstrap_idlekv_minus_random_k": {"low": 0.02},
                "bootstrap_idlekv_minus_oldest_k": {"low": 0.03},
                "bootstrap_idlekv_minus_stale_q_k": {"low": 0.04},
                "bootstrap_idlekv_minus_wrong_q_k": {"low": 0.04},
                "bootstrap_idlekv_minus_tool_file_k": {"low": 0.01},
            },
        },
        "artifact_checks": {
            "all_manifest_audits_passed": True,
            "has_duplicate_example_rows_by_k": False,
            "wrong_event_donor_metadata_complete": True,
        },
        "sensitivity": {
            "exclude_cue_only_hits": {
                "n_examples": 8,
                "k_results": {"k192": {"mean_idlekv_minus_b_match": 0.20}},
            },
            "exclude_answer_retention": {
                "n_examples": 8,
                "k_results": {"k192": {"mean_idlekv_minus_b_match": 0.20}},
            },
            "exclude_cue_and_answer_retention": {
                "n_examples": 8,
                "k_results": {"k192": {"mean_idlekv_minus_b_match": 0.20}},
            },
            "strict_repair_eligible": {"n_examples": 0, "repo_count": 0, "k_results": {}},
        },
    }

    result = repair_gate(
        audit,
        primary_k=192,
        adjacent_k=96,
        min_primary_lift=0.10,
        require_positive_ci=True,
        require_sensitivity=True,
        min_sensitivity_examples=8,
        toolfile_margin_rows=1,
        min_toolfile_file_fraction=0.10,
        anchor_window_margin_rows=0,
    )

    assert not result["passed"]
    assert not result["gate_results"]["anchor_window_margin_ok"]
    assert result["anchor_window_win_loss_margin"] == -4
    assert "AnchorWindow-K is label-assisted" in result["anchor_window_reference_note"]


def test_repair_gate_fails_when_toolfile_control_is_not_real_file_selection() -> None:
    audit = {
        "k_results": {
            "k96": {
                "mean_idlekv_minus_b_match": 0.0,
            },
            "k192": {
                "mean_idlekv_minus_b_match": 0.25,
                "wins_vs_tool_file_k": 3,
                "losses_vs_tool_file_k": 1,
                "wins_vs_anchor_window_k": 3,
                "losses_vs_anchor_window_k": 1,
                "repo_lift_vs_b_match": {"median": 0.25},
                "min_tool_file_k_selected_from_file_fraction": 0.0,
                "fraction_tool_file_k_budget_matched": 1.0,
                "fraction_anchor_window_k_budget_matched": 1.0,
                "bootstrap_idlekv_minus_b_match": {"low": 0.05},
                "bootstrap_idlekv_minus_random_k": {"low": 0.02},
                "bootstrap_idlekv_minus_oldest_k": {"low": 0.03},
            },
        },
        "artifact_checks": {
            "all_manifest_audits_passed": True,
            "has_duplicate_example_rows_by_k": False,
            "wrong_event_donor_metadata_complete": True,
        },
    }

    result = repair_gate(
        audit,
        primary_k=192,
        adjacent_k=96,
        min_primary_lift=0.10,
        require_positive_ci=True,
        require_sensitivity=False,
        min_sensitivity_examples=8,
        toolfile_margin_rows=1,
        min_toolfile_file_fraction=0.10,
        anchor_window_margin_rows=0,
    )

    assert not result["passed"]
    assert not result["gate_results"]["toolfile_selection_ok"]


def test_repair_gate_fails_when_required_ci_keys_are_missing() -> None:
    audit = {
        "k_results": {
            "k96": {
                "mean_idlekv_minus_b_match": 0.0,
            },
            "k192": {
                "mean_idlekv_minus_b_match": 0.25,
                "wins_vs_tool_file_k": 3,
                "losses_vs_tool_file_k": 1,
                "wins_vs_anchor_window_k": 3,
                "losses_vs_anchor_window_k": 1,
                "repo_lift_vs_b_match": {"median": 0.25},
                "min_tool_file_k_selected_from_file_fraction": 0.5,
                "fraction_tool_file_k_budget_matched": 1.0,
                "fraction_anchor_window_k_budget_matched": 1.0,
                "bootstrap_idlekv_minus_b_match": {"low": 0.05},
            },
        },
        "artifact_checks": {
            "all_manifest_audits_passed": True,
            "has_duplicate_example_rows_by_k": False,
            "wrong_event_donor_metadata_complete": True,
        },
    }

    result = repair_gate(
        audit,
        primary_k=192,
        adjacent_k=96,
        min_primary_lift=0.10,
        require_positive_ci=True,
        require_sensitivity=False,
        min_sensitivity_examples=8,
        toolfile_margin_rows=1,
        min_toolfile_file_fraction=0.10,
        anchor_window_margin_rows=0,
    )

    assert not result["passed"]
    assert not result["gate_results"]["required_ci_keys_present"]
    assert not result["gate_results"]["positive_ci_ok"]


def test_repair_gate_fails_when_manifest_static_audit_fails() -> None:
    audit = {
        "k_results": {
            "k96": {"mean_idlekv_minus_b_match": 0.0},
            "k192": {
                "mean_idlekv_minus_b_match": 0.25,
                "wins_vs_tool_file_k": 3,
                "losses_vs_tool_file_k": 1,
                "wins_vs_anchor_window_k": 3,
                "losses_vs_anchor_window_k": 1,
                "repo_lift_vs_b_match": {"median": 0.25},
                "min_tool_file_k_selected_from_file_fraction": 0.5,
                "fraction_tool_file_k_budget_matched": 1.0,
                "fraction_anchor_window_k_budget_matched": 1.0,
                "bootstrap_idlekv_minus_b_match": {"low": 0.05},
                "bootstrap_idlekv_minus_random_k": {"low": 0.02},
                "bootstrap_idlekv_minus_oldest_k": {"low": 0.03},
                "bootstrap_idlekv_minus_stale_q_k": {"low": 0.04},
                "bootstrap_idlekv_minus_wrong_q_k": {"low": 0.04},
                "bootstrap_idlekv_minus_tool_file_k": {"low": 0.01},
            },
        },
        "artifact_checks": {
            "all_manifest_audits_passed": False,
            "has_duplicate_example_rows_by_k": False,
            "wrong_event_donor_metadata_complete": True,
        },
    }

    result = repair_gate(
        audit,
        primary_k=192,
        adjacent_k=96,
        min_primary_lift=0.10,
        require_positive_ci=True,
        require_sensitivity=False,
        min_sensitivity_examples=8,
        toolfile_margin_rows=1,
        min_toolfile_file_fraction=0.10,
        anchor_window_margin_rows=0,
    )

    assert not result["passed"]
    assert not result["gate_results"]["manifest_audit_ok"]


def test_repair_gate_fails_when_wrong_event_donor_metadata_is_missing() -> None:
    audit = {
        "k_results": {
            "k96": {"mean_idlekv_minus_b_match": 0.0},
            "k192": {
                "mean_idlekv_minus_b_match": 0.25,
                "wins_vs_tool_file_k": 3,
                "losses_vs_tool_file_k": 1,
                "wins_vs_anchor_window_k": 3,
                "losses_vs_anchor_window_k": 1,
                "repo_lift_vs_b_match": {"median": 0.25},
                "min_tool_file_k_selected_from_file_fraction": 0.5,
                "fraction_tool_file_k_budget_matched": 1.0,
                "fraction_anchor_window_k_budget_matched": 1.0,
                "bootstrap_idlekv_minus_b_match": {"low": 0.05},
                "bootstrap_idlekv_minus_random_k": {"low": 0.02},
                "bootstrap_idlekv_minus_oldest_k": {"low": 0.03},
                "bootstrap_idlekv_minus_stale_q_k": {"low": 0.04},
                "bootstrap_idlekv_minus_wrong_q_k": {"low": 0.04},
                "bootstrap_idlekv_minus_tool_file_k": {"low": 0.01},
            },
        },
        "artifact_checks": {
            "all_manifest_audits_passed": True,
            "has_duplicate_example_rows_by_k": False,
            "wrong_event_donor_metadata_complete": False,
        },
    }

    result = repair_gate(
        audit,
        primary_k=192,
        adjacent_k=96,
        min_primary_lift=0.10,
        require_positive_ci=True,
        require_sensitivity=False,
        min_sensitivity_examples=8,
        toolfile_margin_rows=1,
        min_toolfile_file_fraction=0.10,
        anchor_window_margin_rows=0,
    )

    assert not result["passed"]
    assert not result["gate_results"]["wrong_event_donor_metadata_ok"]


def test_repair_audit_cli_returns_nonzero_when_gate_fails(tmp_path) -> None:
    artifact = {
        "stage": "unit",
        "manifest_hash": "abc",
        "protocol_hash": "def",
        "rows": [
            {
                "k": 96,
                "example_id": "a",
                "repo_id": "repo1",
                "idlekv_score": 0.0,
                "b_match_score": 0.0,
                "random_k_score": 0.0,
                "oldest_k_score": 0.0,
                "tool_file_k_score": 0.0,
                "anchor_window_k_score": 0.0,
                "tool_file_k_selected_from_file_fraction": 1.0,
                "tool_file_k_budget_matched": True,
                "anchor_window_k_budget_matched": True,
            }
        ],
    }
    artifact_path = tmp_path / "repair.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "phases.phase15_real_repo_relevance_shift.scripts.audit_phase15_repair_artifact",
            str(artifact_path),
            "--gate",
            "--primary-k",
            "96",
            "--adjacent-k",
            "96",
            "--bootstrap-draws",
            "20",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert '"passed": false' in completed.stdout


def test_repair_gate_fails_when_required_ci_is_missing() -> None:
    audit = {
        "k_results": {
            "k96": {
                "mean_idlekv_minus_b_match": 0.0,
            },
            "k192": {
                "mean_idlekv_minus_b_match": 0.25,
                "wins_vs_tool_file_k": 3,
                "losses_vs_tool_file_k": 1,
                "wins_vs_anchor_window_k": 3,
                "losses_vs_anchor_window_k": 1,
                "repo_lift_vs_b_match": {"median": 0.25},
                "min_tool_file_k_selected_from_file_fraction": 0.5,
                "fraction_tool_file_k_budget_matched": 1.0,
                "fraction_anchor_window_k_budget_matched": 1.0,
                "bootstrap_idlekv_minus_b_match": {"low": 0.05},
                "bootstrap_idlekv_minus_random_k": {"low": 0.02},
            },
        },
        "artifact_checks": {
            "all_manifest_audits_passed": True,
            "has_duplicate_example_rows_by_k": False,
            "wrong_event_donor_metadata_complete": True,
        },
    }

    result = repair_gate(
        audit,
        primary_k=192,
        adjacent_k=96,
        min_primary_lift=0.10,
        require_positive_ci=True,
        require_sensitivity=False,
        min_sensitivity_examples=8,
        toolfile_margin_rows=1,
        min_toolfile_file_fraction=0.10,
        anchor_window_margin_rows=0,
    )

    assert not result["passed"]
    assert not result["gate_results"]["positive_ci_ok"]
