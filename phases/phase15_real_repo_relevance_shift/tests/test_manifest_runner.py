from __future__ import annotations

from types import SimpleNamespace

import pytest

from phases.phase15_real_repo_relevance_shift.scripts.run_phase15_manifest import (
    _donor_row,
    add_answer_retention_fields,
    map_phase15_conditions,
    needs_wrong_event_donor,
    strict_rescore_row,
    summarize_rows,
    validate_wrong_event_donors,
)
from phases.phase15_real_repo_relevance_shift.scripts import backfill_wrong_event_metadata


def test_phase15_condition_mapping_includes_mandatory_controls() -> None:
    assert map_phase15_conditions([
        "A",
        "IdleKV-EventOnly-K",
        "StaleCue-K",
        "ToolFile-K",
        "AnchorWindow-K",
        "FileGatedIdleKV-K",
        "LexicalAnchor-K",
    ]) == (
        "A",
        "IdleKV",
        "StaleQ-K",
        "ToolFile-K",
        "AnchorWindow-K",
        "FileGatedIdleKV-K",
        "LexicalAnchor-K",
    )


def test_followup_conditions_do_not_require_wrong_event_donor() -> None:
    assert not needs_wrong_event_donor([
        "A",
        "B_match",
        "IdleKV-EventOnly-K",
        "FileGatedIdleKV-K",
        "LexicalAnchor-K",
    ])
    assert needs_wrong_event_donor(["IdleKV-EventOnly-K", "WrongEvent-K"])


def _row(example_id: str, repo_id: str, event: str, answer: str):
    return SimpleNamespace(
        example_id=example_id,
        repo=SimpleNamespace(repo_id=repo_id),
        tool_event=event,
        answer=answer,
    )


def test_wrong_event_donor_skips_same_repo_event_and_answer() -> None:
    rows = [
        _row("a", "django", "event-1", "target"),
        _row("b", "django", "event-2", "other"),
        _row("c", "pytest", "event-3", "target"),
        _row("d", "sympy", "event-4", "other"),
    ]

    donor = _donor_row(rows, 0)

    assert donor.example_id == "d"
    assert donor.repo.repo_id != rows[0].repo.repo_id
    assert donor.tool_event != rows[0].tool_event
    assert donor.answer != rows[0].answer


def test_wrong_event_donor_requires_distinct_repo_event_and_answer() -> None:
    rows = [
        _row("a", "django", "event-1", "target"),
        _row("b", "pytest", "event-2", "target"),
        _row("c", "django", "event-3", "other"),
    ]

    with pytest.raises(ValueError, match="different repo"):
        _donor_row(rows, 0)


def test_wrong_event_preflight_rejects_nondiverse_smoke_prefix() -> None:
    rows = [
        _row("a", "django", "event-1", "target"),
        _row("b", "django", "event-2", "other"),
    ]

    with pytest.raises(ValueError, match="different repo"):
        validate_wrong_event_donors(rows)


def test_wrong_event_preflight_accepts_repo_diverse_rows() -> None:
    rows = [
        _row("a", "django", "event-1", "target"),
        _row("b", "django", "event-2", "other"),
        _row("c", "pytest", "event-3", "third"),
    ]

    validate_wrong_event_donors(rows)


def test_wrong_event_metadata_backfill_adds_donor_provenance(monkeypatch) -> None:
    rows = [
        _row("a", "django", "event-1", "target"),
        _row("b", "django", "event-2", "other"),
        _row("c", "pytest", "event-3", "third"),
    ]
    monkeypatch.setattr(backfill_wrong_event_metadata, "stable_manifest_hash", lambda _rows: "manifest-hash")

    payload = {
        "manifest_hash": "manifest-hash",
        "rows": [
            {"k": 96, "example_id": "a"},
            {"k": 192, "example_id": "a"},
        ],
    }

    enriched = backfill_wrong_event_metadata.backfill_wrong_event_metadata(
        manifest_rows=rows,
        artifact_payload=payload,
    )

    for row in enriched["rows"]:
        assert row["wrong_event_donor_example_id"] == "c"
        assert row["wrong_event_donor_repo_id"] == "pytest"
        assert row["wrong_event_donor_answer"] == "third"
        assert "wrong_event_donor_tool_event_sha256" in row
    assert enriched["wrong_event_donor_metadata_backfilled"]


def test_strict_rescore_rejects_substring_and_case_mismatch() -> None:
    row = {
        "k": 96,
        "q1_output": "AlphaWindow",
        "condition_a_output": "The answer is BuildRow.",
        "b_match_output": "buildrow",
    }

    rescored = strict_rescore_row(row, q1_gold="AlphaWindow", q2_gold="BuildRow")

    assert rescored["q1_score"] == 1.0
    assert rescored["condition_a_score"] == 0.0
    assert rescored["condition_a_failure_type"] == "format_error"
    assert rescored["b_match_score"] == 0.0
    assert rescored["b_match_failure_type"] == "case_mismatch"


def test_summarize_rows_reports_idlekv_lift() -> None:
    summary = summarize_rows(
        [
            {"k": 96, "condition_a_score": 1.0, "b_match_score": 0.0, "idlekv_score": 1.0},
            {"k": 96, "condition_a_score": 0.0, "b_match_score": 0.0, "idlekv_score": 0.0},
        ]
    )

    assert summary["k96"]["mean_condition_a"] == 0.5
    assert summary["k96"]["mean_idlekv_minus_b_match"] == 0.5


class _Audit:
    answer_token_start = 10
    answer_token_end = 12
    answer_token_count = 2


def test_answer_retention_fields_track_gold_token_overlap() -> None:
    row = {
        "b_kept_context_positions": [0, 9, 12],
        "b_match_kept_context_positions": [0, 10, 11, 12],
        "idlekv_selected_positions": [11],
        "anchor_window_k_selected_positions": [10],
    }

    enriched = add_answer_retention_fields(row, audit=_Audit())

    assert enriched["b_answer_token_overlap_fraction"] == 0.0
    assert enriched["b_match_answer_token_overlap_fraction"] == 1.0
    assert enriched["idlekv_answer_token_overlap_fraction"] == 0.5
    assert enriched["anchor_window_k_answer_token_overlap_fraction"] == 0.5
