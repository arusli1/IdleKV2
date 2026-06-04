"""Tests for publication figure rendering helpers."""

from __future__ import annotations

import pandas as pd
import numpy as np

import paper.scripts.render_paper_figures as renderer
from paper.scripts.render_paper_figures import SPAN_REF_LABEL, _line_style, _pivot_heatmap


def test_pivot_heatmap_orders_budget_and_restore_budget() -> None:
    df = pd.DataFrame(
        [
            {"base_context_budget": 18432, "k": 96, "idlekv_lift": 0.7, "gold_headroom": 0.1},
            {"base_context_budget": 14336, "k": 16, "idlekv_lift": -0.01, "gold_headroom": 0.4},
            {"base_context_budget": 14336, "k": 96, "idlekv_lift": 0.6, "gold_headroom": 0.2},
            {"base_context_budget": 18432, "k": 16, "idlekv_lift": 0.02, "gold_headroom": 0.3},
        ]
    )

    budgets, ks, values, headroom = _pivot_heatmap(df)

    assert budgets == [14336, 18432]
    assert ks == [16, 96]
    np.testing.assert_allclose(values, [[0.0, 0.75], [0.0625, 0.875]])
    assert headroom.tolist() == [[0.4, 0.2], [0.3, 0.1]]


def test_line_styles_use_distinct_marker_channels() -> None:
    styles = {name: _line_style(name) for name in ["IdleKV", SPAN_REF_LABEL, "Matched", "Random-K", "Oldest-K"]}

    assert styles["IdleKV"]["marker"] != styles[SPAN_REF_LABEL]["marker"]
    assert styles["Matched"]["linestyle"] != styles["Random-K"]["linestyle"]
    assert styles["Oldest-K"]["color"] != styles["Random-K"]["color"]


def test_runtime_opportunity_helpers_use_measured_fit_only() -> None:
    select = pd.DataFrame(
        [
            {"candidate_tokens": 32_768, "k": 5000, "p95_total_ms": 35.0},
            {"candidate_tokens": 250_000, "k": 5000, "p95_total_ms": 340.0},
            {"candidate_tokens": 500_000, "k": 5000, "p95_total_ms": 730.0},
            {"candidate_tokens": 1_000_000, "k": 5000, "p95_total_ms": 1160.0},
        ]
    )
    move = pd.DataFrame(
        [
            {"active_tokens": 32768, "k": 5000, "p95_total_ms": 12.0},
            {"active_tokens": 100000, "k": 5000, "p95_total_ms": 24.0},
        ]
    )

    combined = renderer._runtime_select_with_move(select, move)
    assert combined.loc[combined["candidate_tokens"] == 32_768, "repair_ms"].iloc[0] == 47.0

    fit = renderer._max_measured_candidates_by_idle(
        combined,
        k_value=5000,
        idle_windows_s=(0.1, 0.5, 1.0, 2.0),
        budget_fraction=0.9,
    )
    assert fit["max_candidate_tokens"].tolist() == [32_768, 250_000, 500_000, 1_000_000]


def test_runtime_e2e_helper_uses_integrated_total() -> None:
    e2e = pd.DataFrame(
        [
            {"active_tokens": 32768, "candidate_tokens": 32_768, "k": 96, "p95_total_ms": 55.0},
            {"active_tokens": 32768, "candidate_tokens": 1_000_000, "k": 5000, "p95_total_ms": 1180.0},
            {"active_tokens": 100000, "candidate_tokens": 1_000_000, "k": 5000, "p95_total_ms": 1300.0},
        ]
    )
    combined = renderer._runtime_e2e_repair_frame(e2e)
    assert combined["repair_ms"].tolist() == [55.0, 1180.0]
    assert combined["repair_s"].tolist() == [0.055, 1.18]


def test_runtime_component_shares_decompose_p95_path() -> None:
    select = pd.DataFrame(
        [
            {
                "candidate_tokens": 32_768,
                "k": 5000,
                "query_len": 64,
                "p95_scan_ms": 75.0,
                "p95_topk_ms": 5.0,
            },
            {
                "candidate_tokens": 65_536,
                "k": 5000,
                "query_len": 128,
                "p95_scan_ms": 150.0,
                "p95_topk_ms": 5.0,
            },
        ]
    )
    move = pd.DataFrame(
        [
            {
                "active_tokens": 32768,
                "k": 5000,
                "p95_transfer_ms": 10.0,
                "p95_inject_ms": 10.0,
            },
        ]
    )

    shares = renderer._runtime_component_shares(select, move, query_len=64, k_value=5000)

    assert shares["candidate_tokens"].tolist() == [32_768]
    np.testing.assert_allclose(
        shares[["scan_share", "topk_share", "copy_insert_share"]].iloc[0].to_numpy(dtype=float),
        [0.75, 0.05, 0.20],
    )


def test_runtime_labels_use_compact_paper_units() -> None:
    assert renderer._format_candidate_rows(32_768) == "32K"
    assert renderer._format_candidate_rows(1_048_576) == "1M"
    assert renderer._format_candidate_rows(4_194_304) == "4M"
    assert renderer._format_runtime_cell(0.0889) == "89ms"
    assert renderer._format_runtime_cell(1.204) == "1.20s"


def test_runtime_repair_grid_orders_restore_budget_and_candidates() -> None:
    select = pd.DataFrame(
        [
            {"candidate_tokens": 65_536, "k": 512, "query_len": 64, "p95_total_ms": 80.0},
            {"candidate_tokens": 32_768, "k": 512, "query_len": 64, "p95_total_ms": 40.0},
            {"candidate_tokens": 32_768, "k": 96, "query_len": 64, "p95_total_ms": 35.0},
            {"candidate_tokens": 65_536, "k": 96, "query_len": 128, "p95_total_ms": 100.0},
        ]
    )
    move = pd.DataFrame(
        [
            {"active_tokens": 32768, "k": 96, "p95_total_ms": 5.0},
            {"active_tokens": 32768, "k": 512, "p95_total_ms": 7.0},
        ]
    )

    ks, candidates, values = renderer._runtime_repair_grid(select, move, query_len=64)

    assert ks == [96, 512]
    assert candidates == [32_768, 65_536]
    np.testing.assert_allclose(
        values,
        [[40.0 / 1000.0, np.nan], [47.0 / 1000.0, 87.0 / 1000.0]],
        equal_nan=True,
    )


def test_proxy_controlled_renderer_prefers_phase14_locked_csv(tmp_path, monkeypatch) -> None:
    figure_dir = tmp_path / "figures"
    phase14_dir = tmp_path / "phase14"
    figure_dir.mkdir()
    phase14_dir.mkdir()
    csv_path = phase14_dir / "proxy_controlled_locked_n100.csv"
    csv_path.write_text(
        "task,k,b_match,random_k,oldest_k,idlekv,gold_k\n"
        "clean_suite,48,0.20,0.21,0.20,0.70,0.90\n"
        "clean_suite,96,0.22,0.23,0.22,0.85,1.00\n"
        "mq_niah_6q_clean_suite,48,0.35,0.36,0.35,0.65,0.92\n"
        "mq_niah_6q_clean_suite,96,0.40,0.41,0.40,0.80,1.00\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(renderer, "FIGURE_DIR", figure_dir)
    monkeypatch.setattr(renderer, "PHASE14_DIR", phase14_dir)
    renderer.configure_matplotlib()

    assert renderer.render_proxy_controlled_frontier() is True
    assert (figure_dir / "proxy_controlled_frontier.pdf").exists()
    assert (figure_dir / "proxy_controlled_frontier.png").exists()


def test_model_transfer_candidates_prefer_non_saturated_full_grid(tmp_path, monkeypatch) -> None:
    figure_dir = tmp_path / "figures"
    phase14_dir = tmp_path / "phase14"
    phase11_dir = tmp_path / "phase11"
    phase13_dir = tmp_path / "phase13"
    phase10_dir = tmp_path / "phase10"
    for path in (figure_dir, phase14_dir, phase11_dir, phase13_dir, phase10_dir):
        path.mkdir()

    stale_short_grid = figure_dir / "llama31_8b_6q_locked_n12_b18432_k64-96-128.csv"
    preferred_full_grid = phase11_dir / "llama31_8b_4q_fullgrid_n24.csv"
    stale_short_grid.write_text("stale\n", encoding="utf-8")
    preferred_full_grid.write_text("preferred\n", encoding="utf-8")

    monkeypatch.setattr(renderer, "FIGURE_DIR", figure_dir)
    monkeypatch.setattr(renderer, "PHASE14_DIR", phase14_dir)
    monkeypatch.setattr(renderer, "PHASE11_DIR", phase11_dir)
    monkeypatch.setattr(renderer, "PHASE13_DIR", phase13_dir)
    monkeypatch.setattr(renderer, "PHASE10_DIR", phase10_dir)

    candidates = renderer._model_transfer_candidate_paths()
    chosen = next(path for path in candidates if path.exists())

    assert chosen == preferred_full_grid


def test_model_transfer_candidates_prefer_tracked_llama_before_phase14(tmp_path, monkeypatch) -> None:
    figure_dir = tmp_path / "figures"
    phase14_dir = tmp_path / "phase14"
    phase11_dir = tmp_path / "phase11"
    phase13_dir = tmp_path / "phase13"
    phase10_dir = tmp_path / "phase10"
    for path in (figure_dir, phase14_dir, phase11_dir, phase13_dir, phase10_dir):
        path.mkdir()

    preferred = figure_dir / "llama31_8b_6q_locked_n24_b16384_k24-32-48-64.csv"
    phase14_candidate = phase14_dir / "llama_calibrated_locked_n24_b16384.csv"
    fallback = phase11_dir / "llama31_8b_4q_fullgrid_n24.csv"
    preferred.write_text("preferred\n", encoding="utf-8")
    phase14_candidate.write_text("phase14\n", encoding="utf-8")
    fallback.write_text("fallback\n", encoding="utf-8")

    monkeypatch.setattr(renderer, "FIGURE_DIR", figure_dir)
    monkeypatch.setattr(renderer, "PHASE14_DIR", phase14_dir)
    monkeypatch.setattr(renderer, "PHASE11_DIR", phase11_dir)
    monkeypatch.setattr(renderer, "PHASE13_DIR", phase13_dir)
    monkeypatch.setattr(renderer, "PHASE10_DIR", phase10_dir)

    candidates = renderer._model_transfer_candidate_paths()
    chosen = next(path for path in candidates if path.exists())

    assert chosen == preferred


def test_policy_breadth_renderer_requires_streamingllm_full_grid(tmp_path, monkeypatch) -> None:
    figure_dir = tmp_path / "figures"
    phase11_dir = tmp_path / "phase11"
    phase12_dir = tmp_path / "phase12"
    figure_dir.mkdir()
    phase11_dir.mkdir()
    phase12_dir.mkdir()

    snapkv_csv = figure_dir / "phase7_clean_suite_b16384_exact_overall.csv"
    h2o_csv = phase11_dir / "h2o_4q_fullgrid_n24.csv"
    for path in (snapkv_csv, h2o_csv):
        path.write_text(
            "k,b_match,idlekv,random_k,oldest_k,gold_k\n"
            "8,0.20,0.21,0.20,0.20,0.40\n"
            "32,0.20,0.45,0.20,0.21,0.70\n"
            "96,0.20,0.80,0.20,0.20,1.00\n"
            "128,0.20,0.90,0.21,0.20,1.00\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(renderer, "FIGURE_DIR", figure_dir)
    monkeypatch.setattr(renderer, "PHASE11_DIR", phase11_dir)
    monkeypatch.setattr(renderer, "PHASE12_DIR", phase12_dir)
    renderer.configure_matplotlib()

    assert renderer.render_policy_breadth_delta() is False
    assert not (figure_dir / "policy_breadth_delta.pdf").exists()

    (phase12_dir / "streamingllm_4q_fullgrid_n24_b16384.csv").write_text(
        "k,b_match,idlekv,random_k,oldest_k,gold_k\n"
        "8,0.30,0.30,0.30,0.30,0.35\n"
        "32,0.30,0.40,0.30,0.31,0.60\n"
        "96,0.30,0.70,0.30,0.30,0.90\n"
        "128,0.30,0.85,0.31,0.30,0.95\n",
        encoding="utf-8",
    )

    assert renderer.render_policy_breadth_delta() is True
    assert (figure_dir / "policy_breadth_delta.pdf").exists()
    assert (figure_dir / "policy_breadth_delta.png").exists()


def test_multiturn_renderer_prefers_phase13_locked_query_only_controls(tmp_path, monkeypatch) -> None:
    figure_dir = tmp_path / "figures"
    phase10_dir = tmp_path / "phase10"
    phase13_dir = tmp_path / "phase13"
    figure_dir.mkdir()
    phase10_dir.mkdir()
    phase13_dir.mkdir()

    rows = [
        ("Matched", 64, 0, 0.50),
        ("Matched", 64, 1, 0.00),
        ("IdleKV", 64, 0, 0.50),
        ("IdleKV", 64, 1, 0.75),
        ("CurrentQOnly-K", 64, 0, 0.50),
        ("CurrentQOnly-K", 64, 1, 0.70),
        ("StaleQOnly-K", 64, 0, 0.50),
        ("StaleQOnly-K", 64, 1, 0.10),
        ("Random-K", 64, 0, 0.50),
        ("Random-K", 64, 1, 0.00),
        ("Oldest-K", 64, 0, 0.50),
        ("Oldest-K", 64, 1, 0.00),
        ("Gold-K", 64, 0, 0.50),
        ("Gold-K", 64, 1, 1.00),
    ]
    csv_lines = ["example_index,turn,k,condition,score"]
    csv_lines.extend(f"0,{turn},{k},{condition},{score}" for condition, k, turn, score in rows)
    (phase13_dir / "multiturn_hard_locked_rows_n24_k64.csv").write_text(
        "\n".join(csv_lines) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(renderer, "FIGURE_DIR", figure_dir)
    monkeypatch.setattr(renderer, "PHASE10_DIR", phase10_dir)
    monkeypatch.setattr(renderer, "PHASE13_DIR", phase13_dir)
    renderer.configure_matplotlib()

    assert renderer.render_multiturn_hard_trajectory() is True
    assert (figure_dir / "multiturn_hard_trajectory.pdf").exists()
    assert (figure_dir / "multiturn_hard_trajectory.png").exists()
