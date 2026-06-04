from __future__ import annotations

from phases.phase15_real_repo_relevance_shift.src.bootstrap import paired_cluster_bootstrap


def test_paired_cluster_bootstrap_reports_positive_lift() -> None:
    rows = [
        {"repo": "a", "example": "a1", "idlekv": 1.0, "matched": 0.0},
        {"repo": "a", "example": "a2", "idlekv": 1.0, "matched": 0.0},
        {"repo": "b", "example": "b1", "idlekv": 0.0, "matched": 0.0},
        {"repo": "b", "example": "b2", "idlekv": 1.0, "matched": 0.0},
    ]

    ci = paired_cluster_bootstrap(
        rows,
        repo_field="repo",
        example_field="example",
        treatment_field="idlekv",
        baseline_field="matched",
        draws=200,
        seed=7,
    )

    assert ci.mean == 0.75
    assert ci.low <= ci.mean <= ci.high
    assert ci.draws == 200


def test_paired_cluster_bootstrap_averages_duplicate_example_rows() -> None:
    rows = [
        {"repo": "a", "example": "a1", "idlekv": 1.0, "matched": 0.0},
        {"repo": "a", "example": "a1", "idlekv": 0.0, "matched": 0.0},
        {"repo": "b", "example": "b1", "idlekv": 1.0, "matched": 0.0},
    ]

    ci = paired_cluster_bootstrap(
        rows,
        repo_field="repo",
        example_field="example",
        treatment_field="idlekv",
        baseline_field="matched",
        draws=50,
        seed=7,
    )

    assert ci.mean == 0.75
