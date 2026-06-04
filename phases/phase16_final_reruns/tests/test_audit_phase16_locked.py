from __future__ import annotations

from phases.phase16_final_reruns.scripts.audit_phase16_locked import audit_rows


def _row(k: int, *, full: float, matched: float, idle: float, random: float, oldest: float) -> dict[str, float | int]:
    return {
        "k": k,
        "condition_a_score": full,
        "b_match_score": matched,
        "idlekv_score": idle,
        "random_k_score": random,
        "oldest_k_score": oldest,
    }


def test_audit_rows_recommends_main_reference_for_clean_locked_result() -> None:
    rows = []
    for k in (48, 96):
        rows.extend(
            _row(k, full=1.0, matched=0.25, idle=0.75, random=0.25, oldest=0.25)
            for _ in range(8)
        )

    result = audit_rows(rows, draws=100)

    assert result["status"] == "locked_pass"
    assert result["recommendation"] == "main_reference_plus_appendix"
    assert result["clean_k"] == 2


def test_audit_rows_defers_when_full_context_is_weak() -> None:
    rows = [_row(96, full=0.75, matched=0.0, idle=0.5, random=0.0, oldest=0.0) for _ in range(8)]

    result = audit_rows(rows, draws=100)

    assert result["status"] == "locked_fail"
    assert result["recommendation"] == "defer_do_not_include"
    assert result["decisions"][0]["failures"] == ["full_context_not_reliable"]


def test_audit_rows_keeps_saturated_result_in_appendix() -> None:
    rows = []
    for k in (24, 48):
        rows.extend(
            _row(k, full=1.0, matched=0.4, idle=1.0, random=0.4, oldest=0.4)
            for _ in range(8)
        )

    result = audit_rows(rows, draws=100)

    assert result["status"] == "locked_partial"
    assert result["recommendation"] == "appendix_only"
    assert result["saturated"] is True

