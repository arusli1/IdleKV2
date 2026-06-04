from __future__ import annotations

import csv
from pathlib import Path

from phases.phase13_iteration_framework.scripts import audit_live_branches as audit


def _write_multiturn_summary(path: Path) -> None:
    rows = [
        {
            "condition": "IdleKV",
            "mean_noninitial_gain_vs_matched": 0.60,
            "mean_revisit_gain_vs_matched": 0.85,
            "k": 80,
        },
        {
            "condition": "Random-K",
            "mean_noninitial_gain_vs_matched": 0.02,
            "mean_revisit_gain_vs_matched": 0.00,
            "k": 80,
        },
        {
            "condition": "Oldest-K",
            "mean_noninitial_gain_vs_matched": 0.04,
            "mean_revisit_gain_vs_matched": 0.00,
            "k": 80,
        },
        {
            "condition": "StaleQ-K",
            "mean_noninitial_gain_vs_matched": 0.20,
            "mean_revisit_gain_vs_matched": 0.25,
            "k": 80,
        },
        {
            "condition": "CurrentQOnly-K",
            "mean_noninitial_gain_vs_matched": 0.50,
            "mean_revisit_gain_vs_matched": 0.85,
            "k": 80,
        },
        {
            "condition": "StaleQOnly-K",
            "mean_noninitial_gain_vs_matched": 0.25,
            "mean_revisit_gain_vs_matched": 0.25,
            "k": 80,
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "condition",
                "mean_noninitial_gain_vs_matched",
                "mean_revisit_gain_vs_matched",
                "k",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_multiturn_rows(path: Path, *, current_margin: float = 0.25) -> None:
    rows = []
    conditions = {
        "Matched": (0.0, 0.0),
        "IdleKV": (0.60, 0.80),
        "Random-K": (0.05, 0.05),
        "Oldest-K": (0.04, 0.04),
        "CurrentQOnly-K": (0.55, 0.75),
        "StaleQOnly-K": (0.55 - current_margin, 0.75 - current_margin),
    }
    for example_index in range(1, 7):
        for turn in (1, 2):
            for condition, scores in conditions.items():
                rows.append(
                    {
                        "example_index": example_index,
                        "turn": turn,
                        "k": 80,
                        "condition": condition,
                        "score": scores[turn - 1],
                    }
                )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("example_index", "turn", "k", "condition", "score"))
        writer.writeheader()
        writer.writerows(rows)


def test_full_multiturn_audit_requires_paired_uncertainty_for_main_claim(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(audit, "REPO_DIR", tmp_path)
    summary_path = tmp_path / "phases" / "phase13_iteration_framework" / "results" / "multiturn_hard_locked_summary_n24_k80.csv"
    summary_path.parent.mkdir(parents=True)
    _write_multiturn_summary(summary_path)

    result = audit._audit_multiturn_summary(
        summary_path,
        stage="full",
        paired_uncertainty_reported=False,
    )

    assert result["source"] == "phases/phase13_iteration_framework/results/multiturn_hard_locked_summary_n24_k80.csv"
    rigor = result["k_decisions"][0]["result_rigor"]
    assert rigor["action"] == "add_paired_uncertainty_or_demote"
    assert "paired_uncertainty_missing" in rigor["failures"]


def test_full_multiturn_audit_accepts_clean_locked_result_with_uncertainty(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(audit, "REPO_DIR", tmp_path)
    summary_path = tmp_path / "phases" / "phase13_iteration_framework" / "results" / "multiturn_hard_locked_summary_n24_k80.csv"
    summary_path.parent.mkdir(parents=True)
    _write_multiturn_summary(summary_path)
    _write_multiturn_rows(summary_path.with_name("multiturn_hard_locked_rows_n24_k80.csv"))

    result = audit._audit_multiturn_summary(
        summary_path,
        stage="full",
        paired_uncertainty_reported=True,
    )

    rigor = result["k_decisions"][0]["result_rigor"]
    assert result["paired_uncertainty_reported"] is True
    assert rigor["action"] == "main_ready_result"
    assert rigor["main_ready"] is True
    assert result["k_decisions"][0]["paired_uncertainty_gate"]["passed"] is True


def test_full_multiturn_audit_rejects_uncertain_query_only_separation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(audit, "REPO_DIR", tmp_path)
    summary_path = tmp_path / "phases" / "phase13_iteration_framework" / "results" / "multiturn_hard_locked_summary_n24_k80.csv"
    summary_path.parent.mkdir(parents=True)
    _write_multiturn_summary(summary_path)
    _write_multiturn_rows(
        summary_path.with_name("multiturn_hard_locked_rows_n24_k80.csv"),
        current_margin=0.0,
    )

    result = audit._audit_multiturn_summary(
        summary_path,
        stage="full",
        paired_uncertainty_reported=True,
    )

    decision = result["k_decisions"][0]
    assert decision["result_rigor"]["action"] == "demote_or_rerun_for_uncertainty"
    assert "paired_uncertainty_not_positive" in decision["result_rigor"]["failures"]
    assert "current_only_vs_stale_only_noninitial" in decision["paired_uncertainty_gate"]["failures"]


def test_multiturn_figure_quality_requires_real_rows_and_stale_controls() -> None:
    rows = []
    for turn in range(5):
        for condition in ("IdleKV", "Random-K", "Oldest-K", "StaleQ-K", "StaleQOnly-K", "Gold-K"):
            rows.append({"turn": str(turn), "condition": condition, "k": "80"})

    decision = audit._multiturn_figure_quality(rows, exists=True)

    assert decision["main_ready"] is True
    assert decision["data_points"] == 25


def test_multiturn_figure_quality_rejects_missing_stale_control() -> None:
    rows = []
    for turn in range(5):
        for condition in ("IdleKV", "Random-K", "Oldest-K", "Gold-K"):
            rows.append({"turn": str(turn), "condition": condition, "k": "80"})

    decision = audit._multiturn_figure_quality(rows, exists=True)

    assert decision["main_ready"] is False
    assert decision["action"] == "revise_before_main"
    assert "missing_controls" in decision["failures"]
