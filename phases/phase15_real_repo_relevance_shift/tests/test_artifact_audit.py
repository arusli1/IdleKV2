from __future__ import annotations

from phases.phase15_real_repo_relevance_shift.scripts.audit_phase15_artifact import audit_artifact


def test_audit_artifact_passes_clean_ability_gate() -> None:
    payload = {
        "manifest_hash": "abc",
        "protocol_hash": "def",
        "rows": [
            {
                "k": 96,
                "condition_a_score": 1.0,
                "b_match_score": 0.0,
                "cue_only_score": 0.0,
                "q1_score": 1.0,
                "evicted_context_tokens": 1000,
                "b_answer_token_overlap_fraction": 0.0,
                "b_match_answer_token_overlap_fraction": 0.0,
            },
            {
                "k": 96,
                "condition_a_score": 1.0,
                "b_match_score": 0.5,
                "cue_only_score": 0.0,
                "q1_score": 1.0,
                "evicted_context_tokens": 2000,
                "b_answer_token_overlap_fraction": 0.0,
                "b_match_answer_token_overlap_fraction": 0.0,
            },
        ],
    }

    result = audit_artifact(payload, primary_k=96, min_full_cache=0.8, max_cue_only=0.2, min_gap=0.15)

    assert result["passed"]
    assert result["mean_condition_a"] == 1.0
    assert result["mean_a_minus_b_match"] == 0.75
    assert result["cue_only_hits"] == 0


def test_audit_artifact_rejects_zero_eviction() -> None:
    payload = {
        "rows": [
            {
                "k": 96,
                "condition_a_score": 1.0,
                "b_match_score": 0.0,
                "cue_only_score": 0.0,
                "q1_score": 1.0,
                "evicted_context_tokens": 0,
                "b_answer_token_overlap_fraction": 0.0,
                "b_match_answer_token_overlap_fraction": 0.0,
            }
        ],
    }

    result = audit_artifact(payload, primary_k=96, min_full_cache=0.8, max_cue_only=0.2, min_gap=0.15)

    assert not result["passed"]
    assert not result["gate_results"]["eviction_ok"]


def test_audit_artifact_rejects_answer_token_retention() -> None:
    payload = {
        "rows": [
            {
                "k": 96,
                "condition_a_score": 1.0,
                "b_match_score": 0.0,
                "cue_only_score": 0.0,
                "q1_score": 1.0,
                "evicted_context_tokens": 1000,
                "b_answer_token_overlap_fraction": 0.0,
                "b_match_answer_token_overlap_fraction": 1.0,
            }
        ],
    }

    result = audit_artifact(
        payload,
        primary_k=96,
        min_full_cache=0.8,
        max_cue_only=0.2,
        min_gap=0.15,
        max_answer_overlap=0.0,
    )

    assert not result["passed"]
    assert not result["gate_results"]["answer_retention_ok"]


def test_audit_artifact_can_require_per_row_ability_gap() -> None:
    payload = {
        "rows": [
            {
                "k": 96,
                "condition_a_score": 1.0,
                "b_match_score": 0.0,
                "cue_only_score": 0.0,
                "q1_score": 1.0,
                "evicted_context_tokens": 1000,
                "b_answer_token_overlap_fraction": 0.0,
                "b_match_answer_token_overlap_fraction": 0.0,
            },
            {
                "k": 96,
                "condition_a_score": 1.0,
                "b_match_score": 1.0,
                "cue_only_score": 0.0,
                "q1_score": 1.0,
                "evicted_context_tokens": 1000,
                "b_answer_token_overlap_fraction": 0.0,
                "b_match_answer_token_overlap_fraction": 0.0,
            },
        ],
    }

    result = audit_artifact(
        payload,
        primary_k=96,
        min_full_cache=0.8,
        max_cue_only=0.2,
        min_gap=0.15,
        require_individual_gap=True,
    )

    assert not result["passed"]
    assert not result["gate_results"]["individual_gap_ok"]


def test_audit_artifact_can_limit_cue_only_hits() -> None:
    payload = {
        "rows": [
            {
                "k": 96,
                "condition_a_score": 1.0,
                "b_match_score": 0.0,
                "cue_only_score": 1.0,
                "q1_score": 1.0,
                "evicted_context_tokens": 1000,
                "b_answer_token_overlap_fraction": 0.0,
                "b_match_answer_token_overlap_fraction": 0.0,
            }
        ],
    }

    result = audit_artifact(
        payload,
        primary_k=96,
        min_full_cache=0.8,
        max_cue_only=1.0,
        min_gap=0.15,
        max_cue_only_hits=0,
    )

    assert not result["passed"]
    assert result["cue_only_hits"] == 1
    assert not result["gate_results"]["cue_only_hit_count_ok"]
