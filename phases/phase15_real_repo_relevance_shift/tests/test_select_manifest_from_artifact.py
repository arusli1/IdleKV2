from __future__ import annotations

import pytest

from phases.phase15_real_repo_relevance_shift.scripts.select_phase15_manifest_from_artifact import (
    select_rows,
    validate_artifact_manifest_hash,
)


class _Repo:
    def __init__(self, repo_id: str = "repo") -> None:
        self.repo_id = repo_id


class _Row:
    def __init__(self, example_id: str, answer: str = "Target", repo_id: str = "repo") -> None:
        self.example_id = example_id
        self.repo = _Repo(repo_id)
        self.answer = answer
        self.q2 = {"edge_type": "callsite_leaf_callee"}


def test_select_rows_keeps_only_full_cache_gap_rows() -> None:
    manifest_rows = [_Row("a"), _Row("b"), _Row("c"), _Row("d")]
    artifact = {
        "rows": [
            {
                "k": 96,
                "example_id": "a",
                "condition_a_score": 1.0,
                "b_match_score": 0.0,
                "cue_only_score": 0.0,
                "q1_score": 1.0,
                "evicted_context_tokens": 100,
                "b_answer_token_overlap_fraction": 0.0,
                "b_match_answer_token_overlap_fraction": 0.0,
            },
            {
                "k": 96,
                "example_id": "b",
                "condition_a_score": 1.0,
                "b_match_score": 1.0,
                "cue_only_score": 0.0,
                "q1_score": 1.0,
                "evicted_context_tokens": 100,
                "b_answer_token_overlap_fraction": 0.0,
                "b_match_answer_token_overlap_fraction": 0.0,
            },
            {
                "k": 96,
                "example_id": "c",
                "condition_a_score": 1.0,
                "b_match_score": 0.0,
                "cue_only_score": 0.0,
                "q1_score": 1.0,
                "evicted_context_tokens": 100,
                "b_answer_token_overlap_fraction": 0.0,
                "b_match_answer_token_overlap_fraction": 0.5,
            },
        ]
    }

    selected, diagnostics, reasons = select_rows(
        manifest_rows=manifest_rows,
        artifact_payload=artifact,
        k=96,
        max_cue_only=0.0,
        max_answer_overlap=0.0,
    )

    assert [row.example_id for row in selected] == ["a"]
    assert reasons["selected"] == 1
    assert reasons["b_match_success"] == 1
    assert reasons["b_match_answer_retained"] == 1
    assert reasons["missing_artifact_row"] == 1
    assert [item["reason"] for item in diagnostics] == [
        "selected",
        "b_match_success",
        "b_match_answer_retained",
        "missing_artifact_row",
    ]


def test_select_rows_can_require_q1_success() -> None:
    manifest_rows = [_Row("a")]
    artifact = {
        "rows": [
            {
                "k": 96,
                "example_id": "a",
                "condition_a_score": 1.0,
                "b_match_score": 0.0,
                "cue_only_score": 0.0,
                "q1_score": 0.0,
                "evicted_context_tokens": 100,
                "b_answer_token_overlap_fraction": 0.0,
                "b_match_answer_token_overlap_fraction": 0.0,
            }
        ]
    }

    selected, _diagnostics, reasons = select_rows(
        manifest_rows=manifest_rows,
        artifact_payload=artifact,
        k=96,
        max_cue_only=0.0,
        max_answer_overlap=0.0,
        require_q1=True,
    )

    assert selected == []
    assert reasons["q1_miss"] == 1


def test_select_rows_can_balance_by_repo_and_row_count() -> None:
    manifest_rows = [
        _Row("a", repo_id="repo1"),
        _Row("b", repo_id="repo1"),
        _Row("c", repo_id="repo2"),
        _Row("d", repo_id="repo3"),
    ]
    artifact = {
        "rows": [
            {
                "k": 96,
                "example_id": row.example_id,
                "condition_a_score": 1.0,
                "b_match_score": 0.0,
                "cue_only_score": 0.0,
                "q1_score": 1.0,
                "evicted_context_tokens": 100,
                "b_answer_token_overlap_fraction": 0.0,
                "b_match_answer_token_overlap_fraction": 0.0,
            }
            for row in manifest_rows
        ]
    }

    selected, diagnostics, reasons = select_rows(
        manifest_rows=manifest_rows,
        artifact_payload=artifact,
        k=96,
        max_cue_only=0.0,
        max_answer_overlap=0.0,
        max_selected_rows=3,
        max_rows_per_repo=1,
    )

    assert [row.example_id for row in selected] == ["a", "c", "d"]
    assert reasons["selected"] == 3
    assert reasons["repo_balance_truncated"] == 1
    assert [item["reason"] for item in diagnostics] == [
        "selected",
        "repo_balance_truncated",
        "selected",
        "selected",
    ]


def test_validate_artifact_manifest_hash_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        validate_artifact_manifest_hash(
            source_manifest_hash="manifest-a",
            artifact_payload={"manifest_hash": "manifest-b"},
        )
