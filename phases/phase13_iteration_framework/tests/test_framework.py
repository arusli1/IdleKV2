from phases.phase13_iteration_framework.src import (
    classify_figure_quality,
    classify_multiturn_candidate,
    classify_policy_curve,
    classify_result_rigor,
    consecutive_positive_budget_count,
)


def test_consecutive_positive_budget_count_uses_observed_grid_order() -> None:
    rows = [
        {"k": 8, "idle_gain": 0.01},
        {"k": 16, "idle_gain": 0.20},
        {"k": 32, "idle_gain": 0.21},
        {"k": 96, "idle_gain": 0.00},
        {"k": 128, "idle_gain": 0.30},
    ]

    assert consecutive_positive_budget_count(rows, min_gain=0.15) == 2


def test_policy_curve_rejects_endpoint_only_positive() -> None:
    rows = [
        {
            "k": k,
            "condition_a": 1.0,
            "b_match": 0.25,
            "idlekv": 0.25 if k < 128 else 0.75,
            "random_k": 0.25,
            "oldest_k": 0.25,
            "gold_k": 1.0,
        }
        for k in [8, 16, 32, 64, 96, 128]
    ]

    decision = classify_policy_curve(rows)

    assert decision["main_candidate"] is False
    assert decision["action"] == "appendix_only_endpoint_positive"


def test_policy_curve_accepts_clean_multi_point_curve() -> None:
    rows = [
        {"k": 8, "condition_a": 1.0, "b_match": 0.25, "idlekv": 0.25, "random_k": 0.25, "oldest_k": 0.25, "gold_k": 0.4},
        {"k": 16, "condition_a": 1.0, "b_match": 0.25, "idlekv": 0.30, "random_k": 0.25, "oldest_k": 0.25, "gold_k": 0.6},
        {"k": 32, "condition_a": 1.0, "b_match": 0.25, "idlekv": 0.55, "random_k": 0.26, "oldest_k": 0.25, "gold_k": 0.8},
        {"k": 64, "condition_a": 1.0, "b_match": 0.25, "idlekv": 0.75, "random_k": 0.26, "oldest_k": 0.25, "gold_k": 1.0},
        {"k": 96, "condition_a": 1.0, "b_match": 0.25, "idlekv": 0.95, "random_k": 0.26, "oldest_k": 0.25, "gold_k": 1.0},
    ]

    decision = classify_policy_curve(rows)

    assert decision["main_candidate"] is True
    assert decision["adjacent_positive"] >= 2


def test_result_rigor_rejects_positive_smoke_for_main() -> None:
    decision = classify_result_rigor(
        {
            "stage": "smoke",
            "n": 2,
            "grid_points": 5,
            "effect_size": 0.5,
            "matched_budget_audited": True,
            "full_cache_ok": True,
            "controls_clean": True,
            "non_saturated": True,
            "confound_checked": True,
        }
    )

    assert decision["main_ready"] is False
    assert decision["action"] == "run_locked_full_before_main"
    assert "smoke_only" in decision["failures"]


def test_result_rigor_accepts_clean_full_grid_result() -> None:
    decision = classify_result_rigor(
        {
            "stage": "full",
            "n": 24,
            "grid_points": 9,
            "effect_size": 0.43,
            "paired_or_shared_examples": True,
            "matched_budget_audited": True,
            "full_cache_ok": True,
            "controls_clean": True,
            "non_saturated": True,
            "confound_checked": True,
        }
    )

    assert decision["main_ready"] is True
    assert decision["action"] == "main_ready_result"


def test_result_rigor_requires_uncertainty_for_primary_claim() -> None:
    decision = classify_result_rigor(
        {
            "stage": "full",
            "n": 24,
            "grid_points": 9,
            "effect_size": 0.43,
            "paired_or_shared_examples": True,
            "matched_budget_audited": True,
            "full_cache_ok": True,
            "controls_clean": True,
            "non_saturated": True,
            "confound_checked": True,
            "primary_claim": True,
        }
    )

    assert decision["main_ready"] is False
    assert decision["action"] == "add_paired_uncertainty_or_demote"
    assert "paired_uncertainty_missing" in decision["failures"]


def test_figure_quality_rejects_wrong_graph_type_and_overlap() -> None:
    decision = classify_figure_quality(
        {
            "real_data": True,
            "no_fake_data": True,
            "data_points": 27,
            "graph_type_fits_claim": False,
            "one_column_fit": True,
            "legend_outside_data": False,
            "labels_readable": True,
            "caption_scopes_claim": True,
            "controls_visible": True,
            "not_redundant": True,
            "top_paper_style": False,
        }
    )

    assert decision["main_ready"] is False
    assert decision["action"] == "redesign_graph_type"
    assert "wrong_graph_type" in decision["failures"]


def test_figure_quality_accepts_dense_one_column_graph() -> None:
    decision = classify_figure_quality(
        {
            "real_data": True,
            "no_fake_data": True,
            "data_points": 54,
            "graph_type_fits_claim": True,
            "one_column_fit": True,
            "legend_outside_data": True,
            "labels_readable": True,
            "caption_scopes_claim": True,
            "controls_visible": True,
            "not_redundant": True,
            "top_paper_style": True,
        }
    )

    assert decision["main_ready"] is True
    assert decision["action"] == "main_ready_figure"


def test_multiturn_gate_rejects_when_stale_query_is_too_strong() -> None:
    rows = [
        {
            "k": 96,
            "condition": "IdleKV",
            "mean_noninitial_gain_vs_matched": 0.60,
            "mean_revisit_gain_vs_matched": 1.0,
        },
        {"k": 96, "condition": "Random-K", "mean_noninitial_gain_vs_matched": 0.01},
        {"k": 96, "condition": "Oldest-K", "mean_noninitial_gain_vs_matched": 0.03},
        {"k": 96, "condition": "StaleQ-K", "mean_noninitial_gain_vs_matched": 0.36},
    ]

    decision = classify_multiturn_candidate(rows)[0]

    assert decision["main_candidate"] is False
    assert decision["action"] == "appendix_only_stale_query_too_strong"


def test_multiturn_gate_accepts_clean_current_query_separation() -> None:
    rows = [
        {
            "k": 80,
            "condition": "IdleKV",
            "mean_noninitial_gain_vs_matched": 0.60,
            "mean_revisit_gain_vs_matched": 0.85,
        },
        {"k": 80, "condition": "Random-K", "mean_noninitial_gain_vs_matched": 0.02},
        {"k": 80, "condition": "Oldest-K", "mean_noninitial_gain_vs_matched": 0.04},
        {"k": 80, "condition": "StaleQ-K", "mean_noninitial_gain_vs_matched": 0.20},
        {
            "k": 80,
            "condition": "CurrentQOnly-K",
            "mean_noninitial_gain_vs_matched": 0.50,
        },
        {
            "k": 80,
            "condition": "StaleQOnly-K",
            "mean_noninitial_gain_vs_matched": 0.25,
        },
    ]

    decision = classify_multiturn_candidate(rows)[0]

    assert decision["main_candidate"] is True
    assert decision["action"] == "main_candidate"
    assert decision["query_only_margin"] == 0.25


def test_multiturn_gate_rejects_when_query_only_controls_are_too_close() -> None:
    rows = [
        {
            "k": 80,
            "condition": "IdleKV",
            "mean_noninitial_gain_vs_matched": 0.60,
            "mean_revisit_gain_vs_matched": 0.85,
        },
        {"k": 80, "condition": "Random-K", "mean_noninitial_gain_vs_matched": 0.02},
        {"k": 80, "condition": "Oldest-K", "mean_noninitial_gain_vs_matched": 0.04},
        {"k": 80, "condition": "StaleQ-K", "mean_noninitial_gain_vs_matched": 0.20},
        {
            "k": 80,
            "condition": "CurrentQOnly-K",
            "mean_noninitial_gain_vs_matched": 0.36,
        },
        {
            "k": 80,
            "condition": "StaleQOnly-K",
            "mean_noninitial_gain_vs_matched": 0.31,
        },
    ]

    decision = classify_multiturn_candidate(rows)[0]

    assert decision["main_candidate"] is False
    assert decision["action"] == "appendix_only_query_only_controls_too_close"
    assert decision["query_only_margin"] == 0.05
