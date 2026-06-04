from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

from phases.phase13_iteration_framework.src import (
    bootstrap_mean_interval,
    multiturn_uncertainty_rows,
    paired_condition_difference_values,
    paired_gain_values,
)
from phases.phase13_iteration_framework.scripts.postprocess_multiturn_locked import (
    latest_locked_summary,
    locked_paths,
    write_locked_uncertainty,
)

REPO_DIR = Path(__file__).resolve().parents[3]


def test_paired_gain_values_matches_by_example_turn_and_k() -> None:
    rows = [
        {"example_index": "1", "turn": "0", "k": "80", "condition": "Matched", "score": "0.25"},
        {"example_index": "1", "turn": "0", "k": "80", "condition": "IdleKV", "score": "0.75"},
        {"example_index": "1", "turn": "1", "k": "80", "condition": "Matched", "score": "0.00"},
        {"example_index": "1", "turn": "1", "k": "80", "condition": "IdleKV", "score": "1.00"},
        {"example_index": "1", "turn": "1", "k": "96", "condition": "IdleKV", "score": "1.00"},
        {"example_index": "2", "turn": "1", "k": "80", "condition": "IdleKV", "score": "1.00"},
    ]

    assert paired_gain_values(rows, condition="IdleKV", k=80) == (0.5, 1.0)
    assert paired_gain_values(rows, condition="IdleKV", k=80, turns=(1,)) == (1.0,)


def test_paired_condition_difference_values_compares_two_conditions() -> None:
    rows = [
        {"example_index": "1", "turn": "1", "k": "80", "condition": "Random-K", "score": "0.25"},
        {"example_index": "1", "turn": "1", "k": "80", "condition": "IdleKV", "score": "0.75"},
        {"example_index": "2", "turn": "1", "k": "80", "condition": "Random-K", "score": "0.50"},
        {"example_index": "2", "turn": "1", "k": "80", "condition": "IdleKV", "score": "1.00"},
        {"example_index": "2", "turn": "2", "k": "80", "condition": "IdleKV", "score": "1.00"},
    ]

    assert paired_condition_difference_values(
        rows,
        condition="IdleKV",
        baseline_condition="Random-K",
        k=80,
        turns=(1,),
    ) == (0.5, 0.5)


def test_bootstrap_mean_interval_is_deterministic_and_contains_mean() -> None:
    first = bootstrap_mean_interval([0.0, 1.0, 1.0, 0.0], n_bootstrap=200, seed=9)
    second = bootstrap_mean_interval([0.0, 1.0, 1.0, 0.0], n_bootstrap=200, seed=9)

    assert first == second
    assert first["mean"] == 0.5
    assert first["lo"] <= first["mean"] <= first["hi"]
    assert first["n"] == 4


def test_multiturn_uncertainty_rows_reports_noninitial_and_revisit_intervals() -> None:
    rows = []
    for example_index, matched_scores, idle_scores in [
        (1, (0.25, 0.0, 0.5), (0.25, 1.0, 1.0)),
        (2, (0.25, 0.5, 0.5), (0.25, 0.5, 1.0)),
    ]:
        for turn, matched_score in enumerate(matched_scores):
            rows.append(
                {
                    "example_index": str(example_index),
                    "turn": str(turn),
                    "k": "80",
                    "condition": "Matched",
                    "score": str(matched_score),
                }
            )
        for turn, idle_score in enumerate(idle_scores):
            rows.append(
                {
                    "example_index": str(example_index),
                    "turn": str(turn),
                    "k": "80",
                    "condition": "IdleKV",
                    "score": str(idle_score),
                }
            )

    summary = multiturn_uncertainty_rows(
        rows,
        revisit_turns=(2,),
        conditions=("IdleKV",),
        n_bootstrap=200,
        seed=3,
    )

    assert len(summary) == 1
    row = summary[0]
    assert row["condition"] == "IdleKV"
    assert row["mean_gain"] == 0.333333
    assert row["mean_noninitial_gain"] == 0.5
    assert row["mean_revisit_gain"] == 0.5
    assert row["n_paired"] == 6
    assert row["n_noninitial"] == 4
    assert row["n_revisit"] == 2


def test_summarize_multiturn_uncertainty_cli_reads_revisit_turns(tmp_path) -> None:
    rows_path = tmp_path / "rows.csv"
    raw_path = tmp_path / "raw.json"
    out_path = tmp_path / "uncertainty.csv"

    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("example_index", "turn", "k", "condition", "score"))
        writer.writeheader()
        writer.writerows(
            [
                {"example_index": 1, "turn": 0, "k": 80, "condition": "Matched", "score": 0.0},
                {"example_index": 1, "turn": 0, "k": 80, "condition": "IdleKV", "score": 0.5},
                {"example_index": 1, "turn": 1, "k": 80, "condition": "Matched", "score": 0.0},
                {"example_index": 1, "turn": 1, "k": 80, "condition": "IdleKV", "score": 1.0},
            ]
        )
    raw_path.write_text(json.dumps({"schedule": {"revisit_events": [{"revisit_turn": 1}]}}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_DIR / "phases/phase13_iteration_framework/scripts/summarize_multiturn_uncertainty.py"),
            "--rows-csv",
            str(rows_path),
            "--raw-json",
            str(raw_path),
            "--conditions",
            "IdleKV",
            "--bootstrap-samples",
            "20",
            "--out-csv",
            str(out_path),
        ],
        check=True,
        cwd=str(REPO_DIR),
        text=True,
        capture_output=True,
    )

    assert str(out_path) in result.stdout
    with out_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["condition"] == "IdleKV"
    assert rows[0]["mean_revisit_gain"] == "1.0"


def test_locked_postprocess_derives_expected_paths_and_writes_uncertainty(tmp_path) -> None:
    summary_path = tmp_path / "multiturn_hard_locked_summary_n24_k80.csv"
    rows_path = tmp_path / "multiturn_hard_locked_rows_n24_k80.csv"
    raw_path = tmp_path / "multiturn_hard_locked_n24_k80_raw.json"
    summary_path.write_text("condition,k\nIdleKV,80\n", encoding="utf-8")
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("example_index", "turn", "k", "condition", "score"))
        writer.writeheader()
        writer.writerows(
            [
                {"example_index": 1, "turn": 0, "k": 80, "condition": "Matched", "score": 0.0},
                {"example_index": 1, "turn": 0, "k": 80, "condition": "IdleKV", "score": 0.5},
                {"example_index": 1, "turn": 1, "k": 80, "condition": "Matched", "score": 0.0},
                {"example_index": 1, "turn": 1, "k": 80, "condition": "IdleKV", "score": 1.0},
            ]
        )
    raw_path.write_text(json.dumps({"schedule": {"revisit_events": [{"revisit_turn": 1}]}}), encoding="utf-8")

    paths = locked_paths(summary_path)
    out_path = write_locked_uncertainty(
        summary_path,
        conditions=("IdleKV",),
        bootstrap_samples=20,
    )

    assert latest_locked_summary(tmp_path) == summary_path
    assert paths["rows"] == rows_path
    assert paths["raw"] == raw_path
    assert out_path == tmp_path / "multiturn_hard_locked_uncertainty_n24_k80.csv"
    with out_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["condition"] == "IdleKV"
