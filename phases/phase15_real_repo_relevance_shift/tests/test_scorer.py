from __future__ import annotations

import pytest

from phases.phase15_real_repo_relevance_shift.src.scorer import (
    score_identifier_prediction,
    validate_gold_identifier,
)


def test_strict_identifier_scorer_accepts_exact_first_line() -> None:
    result = score_identifier_prediction("  `RenderPayload`\nextra text", "RenderPayload")

    assert result.score == 1.0
    assert result.failure_type is None
    assert result.normalized_prediction == "RenderPayload"


def test_strict_identifier_scorer_rejects_substrings_and_case_mismatch() -> None:
    substring = score_identifier_prediction("RenderPayload()", "RenderPayload")
    case = score_identifier_prediction("renderpayload", "RenderPayload")

    assert substring.score == 0.0
    assert substring.failure_type == "format_error"
    assert case.score == 0.0
    assert case.failure_type == "case_mismatch"


def test_validate_gold_identifier_rejects_non_identifier() -> None:
    with pytest.raises(ValueError):
        validate_gold_identifier("not-a-valid-id")

