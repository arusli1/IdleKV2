"""Strict identifier scoring for Phase 15 exact-answer diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import re

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class IdentifierScore:
    """Exact identifier score plus a compact failure label."""

    score: float
    normalized_prediction: str
    failure_type: str | None


def normalize_identifier_prediction(prediction: str) -> str:
    """Return the first decoded line after whitespace/backtick trim.

    This intentionally does not strip punctuation, parentheses, explanations, or
    case differences. Repo/code identifiers are case-sensitive, and substring
    credit would make Phase 15 too easy to audit incorrectly.
    """
    text = str(prediction).lstrip()
    first_line = text.splitlines()[0] if text else ""
    return first_line.strip().strip("`").strip()


def score_identifier_prediction(prediction: str, gold: str) -> IdentifierScore:
    """Score one exact identifier answer.

    A prediction is correct only if the normalized first decoded line matches the
    gold identifier byte-for-byte.
    """
    gold_text = str(gold).strip()
    normalized = normalize_identifier_prediction(prediction)
    if not normalized:
        return IdentifierScore(0.0, normalized, "empty")
    if normalized == gold_text:
        return IdentifierScore(1.0, normalized, None)
    if not IDENTIFIER_RE.match(normalized):
        return IdentifierScore(0.0, normalized, "format_error")
    if normalized.lower() == gold_text.lower():
        return IdentifierScore(0.0, normalized, "case_mismatch")
    return IdentifierScore(0.0, normalized, "wrong_identifier")


def validate_gold_identifier(identifier: str) -> None:
    """Reject gold labels that cannot be scored by the strict identifier scorer."""
    if not IDENTIFIER_RE.match(str(identifier)):
        raise ValueError(f"Gold answer is not a simple identifier: {identifier!r}")

