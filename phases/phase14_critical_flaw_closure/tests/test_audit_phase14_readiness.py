from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from phases.phase14_critical_flaw_closure.scripts import audit_phase14_readiness as audit
from phases.phase14_critical_flaw_closure.scripts import evaluate_phase14_smokes as smoke_eval
from phases.phase14_critical_flaw_closure.scripts import evaluate_proxy_controlled_smoke as proxy_eval
from phases.phase14_critical_flaw_closure.scripts.monitor_proxy_progress import (
    parse_proxy_log,
    parse_run_metadata,
    render_eta,
    render_summary,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_proxy_audit_requires_controls_even_when_speed_and_quality_pass(tmp_path: Path) -> None:
    exact = tmp_path / "exact.csv"
    proxy = tmp_path / "proxy.csv"
    _write_csv(
        exact,
        [
            {
                "k": 96,
                "b_match": 0.25,
                "idlekv": 0.90,
                "p50_total_ms": 6000,
                "p50_score_ms": 5900,
            }
        ],
    )
    _write_csv(
        proxy,
        [
            {
                "k": 96,
                "b_match": 0.25,
                "idlekv": 0.88,
                "p50_total_ms": 700,
                "p50_score_ms": 500,
            }
        ],
    )

    result = audit.audit_proxy_pair(label="4q", exact_csv=exact, proxy_csv=proxy)

    assert result["status"] == "needs_controlled_proxy_smoke_or_locked_run"
    assert "missing_random_oldest_gold_controls" in result["failures"]


def test_proxy_audit_accepts_controlled_multik_artifact(tmp_path: Path) -> None:
    exact = tmp_path / "exact.csv"
    proxy = tmp_path / "proxy.csv"
    _write_csv(
        exact,
        [
            {"k": 48, "b_match": 0.25, "idlekv": 0.55, "p50_total_ms": 6000, "p50_score_ms": 5900},
            {"k": 96, "b_match": 0.25, "idlekv": 0.90, "p50_total_ms": 6000, "p50_score_ms": 5900},
            {"k": 128, "b_match": 0.25, "idlekv": 0.99, "p50_total_ms": 6000, "p50_score_ms": 5900},
        ],
    )
    _write_csv(
        proxy,
        [
            {
                "k": k,
                "b_match": 0.25,
                "idlekv": value,
                "random_k": 0.25,
                "oldest_k": 0.24,
                "gold_k": 1.0,
                "p50_total_ms": 700,
                "p50_score_ms": 500,
            }
            for k, value in ((48, 0.50), (96, 0.86), (128, 0.96))
        ],
    )

    result = audit.audit_proxy_pair(label="4q", exact_csv=exact, proxy_csv=proxy)

    assert result["status"] == "main_ready_proxy_evidence"
    assert result["has_random_oldest_gold_controls"] is True


def test_proxy_audit_uses_controlled_quality_with_separate_latency_source(tmp_path: Path) -> None:
    exact = tmp_path / "exact.csv"
    controlled = tmp_path / "controlled.csv"
    latency = tmp_path / "latency.csv"
    _write_csv(
        exact,
        [
            {"k": 96, "b_match": 0.40, "idlekv": 0.90, "p50_total_ms": 7000, "p50_score_ms": 6500},
        ],
    )
    _write_csv(
        controlled,
        [
            {
                "task": "clean_suite",
                "k": 48,
                "b_match": 0.25,
                "idlekv": 0.70,
                "random_k": 0.26,
                "oldest_k": 0.24,
                "gold_k": 1.0,
            },
            {
                "task": "clean_suite",
                "k": 96,
                "b_match": 0.25,
                "idlekv": 0.92,
                "random_k": 0.26,
                "oldest_k": 0.24,
                "gold_k": 1.0,
            },
            {
                "task": "clean_suite",
                "k": 128,
                "b_match": 0.25,
                "idlekv": 0.96,
                "random_k": 0.26,
                "oldest_k": 0.24,
                "gold_k": 1.0,
            },
            {
                "task": "mq_niah_6q_clean_suite",
                "k": 96,
                "b_match": 0.45,
                "idlekv": 0.80,
                "random_k": 0.46,
                "oldest_k": 0.44,
                "gold_k": 0.95,
            },
        ],
    )
    _write_csv(
        latency,
        [
            {"k": 96, "p50_total_ms": 700, "p50_score_ms": 650},
        ],
    )

    result = audit.audit_proxy_pair(
        label="4q_proxy",
        exact_csv=exact,
        proxy_csv=controlled,
        latency_proxy_csv=latency,
        task="clean_suite",
    )

    assert result["status"] == "main_ready_proxy_evidence"
    assert result["proxy_lift"] == 0.67
    assert result["p50_total_speedup"] == 10.0
    assert result["quality_source"] == str(controlled)
    assert result["latency_source"] == str(latency)


def test_audit_all_uses_prespecified_lower_6q_proxy_retention_gate(tmp_path: Path) -> None:
    figures = tmp_path / "paper" / "figures"
    phase14 = tmp_path / "phases" / "phase14_critical_flaw_closure" / "results"
    figures.mkdir(parents=True)
    phase14.mkdir(parents=True)

    _write_csv(
        figures / "phase9_proxy_exact_4q_reference.csv",
        [
            {
                "task": "clean_suite",
                "k": 96,
                "b_match": 0.25,
                "idlekv": 0.75,
                "p50_total_ms": 7000,
                "p50_score_ms": 6500,
            }
        ],
    )
    _write_csv(
        figures / "phase9_proxy_exact_6q_reference.csv",
        [
            {
                "task": "mq_niah_6q_clean_suite",
                "k": 96,
                "b_match": 0.40,
                "idlekv": 0.90,
                "p50_total_ms": 7000,
                "p50_score_ms": 6500,
            }
        ],
    )
    controlled_rows = [
        {
            "task": task,
            "k": k,
            "b_match": b_match,
            "idlekv": idlekv,
            "random_k": b_match,
            "oldest_k": b_match,
            "gold_k": 1.0,
        }
        for task, b_match, idlekv in (
            ("clean_suite", 0.25, 0.70),
            ("mq_niah_6q_clean_suite", 0.40, 0.81),
        )
        for k in (48, 96, 128)
    ]
    _write_csv(phase14 / "proxy_controlled_locked_n100.csv", controlled_rows)
    _write_csv(
        figures / "phase9_proxy_4q_full_n100.csv",
        [
            {
                "task": "clean_suite",
                "k": 96,
                "p50_total_ms": 700,
                "p50_score_ms": 650,
            }
        ],
    )
    _write_csv(
        figures / "phase9_proxy_6q_full_n100.csv",
        [
            {
                "task": "mq_niah_6q_clean_suite",
                "k": 96,
                "p50_total_ms": 700,
                "p50_score_ms": 650,
            }
        ],
    )

    result = audit.audit_all(tmp_path)
    proxies = {item["label"]: item for item in result["proxy_deployability"]}

    assert proxies["4q_proxy"]["retention_gate"] == 0.85
    assert proxies["6q_proxy"]["retention_gate"] == 0.80
    assert proxies["6q_proxy"]["retained_gain"] == 0.82
    assert proxies["6q_proxy"]["status"] == "main_ready_proxy_evidence"


def test_proxy_audit_does_not_request_controlled_run_after_controlled_quality_fails(tmp_path: Path) -> None:
    exact = tmp_path / "exact.csv"
    controlled = tmp_path / "controlled.csv"
    latency = tmp_path / "latency.csv"
    _write_csv(
        exact,
        [
            {"k": 96, "b_match": 0.25, "idlekv": 0.95, "p50_total_ms": 7000, "p50_score_ms": 6500},
        ],
    )
    _write_csv(
        controlled,
        [
            {
                "k": k,
                "b_match": 0.25,
                "idlekv": value,
                "random_k": 0.25,
                "oldest_k": 0.24,
                "gold_k": 1.0,
            }
            for k, value in ((48, 0.30), (96, 0.70), (128, 0.76))
        ],
    )
    _write_csv(
        latency,
        [
            {"k": 96, "p50_total_ms": 700, "p50_score_ms": 650},
        ],
    )

    result = audit.audit_proxy_pair(
        label="6q_proxy",
        exact_csv=exact,
        proxy_csv=controlled,
        latency_proxy_csv=latency,
    )

    assert result["status"] == "needs_proxy_redesign"
    assert "retained_gain_below_gate" in result["failures"]


def test_specificity_audit_flags_refresh_boundary(tmp_path: Path) -> None:
    path = tmp_path / "specificity.csv"
    _write_csv(
        path,
        [
            {"condition": "Matched", "mean_score": 0.20},
            {"condition": "StaleQ-K", "mean_score": 0.24},
            {"condition": "WrongQ-K", "mean_score": 0.25},
            {"condition": "IdleKV", "mean_score": 0.55},
            {"condition": "Refresh-K", "mean_score": 1.00},
            {"condition": "Gold-K", "mean_score": 1.00},
        ],
    )

    result = audit.audit_specificity(path)

    assert result["status"] == "boundary_result_needs_framing"
    assert "refresh_dominates_idlekv" in result["failures"]


def test_llama_audit_demotes_saturated_short_grid(tmp_path: Path) -> None:
    path = tmp_path / "llama.csv"
    _write_csv(
        path,
        [
            {
                "k": k,
                "condition_a": 1.0,
                "b_match": 0.45,
                "idlekv": 1.0,
                "num_samples": 12,
            }
            for k in (64, 96, 128)
        ],
    )

    result = audit.audit_llama(path)

    assert result["status"] == "appendix_only_portability"
    assert "idlekv_saturates_across_grid" in result["failures"]
    assert "n_below_main_claim_threshold" in result["failures"]


def test_llama_audit_accepts_non_saturated_full_grid(tmp_path: Path) -> None:
    path = tmp_path / "llama.csv"
    _write_csv(
        path,
        [
            {
                "k": k,
                "condition_a": 0.99,
                "b_match": 0.48,
                "idlekv": score,
                "num_samples": 24,
            }
            for k, score in ((8, 0.56), (16, 0.63), (24, 0.94), (32, 0.99), (64, 1.00))
        ],
    )

    result = audit.audit_llama(path)

    assert result["status"] == "candidate_main_model_evidence"
    assert result["failures"] == []


def test_controlled_proxy_evaluator_rejects_high_control_lift() -> None:
    decisions = proxy_eval.evaluate_rows(
        [
            {
                "task": "clean_suite",
                "k": 96,
                "b_match": 0.25,
                "idlekv": 0.85,
                "random_k": 0.50,
                "oldest_k": 0.25,
                "gold_k": 1.0,
            }
        ]
    )

    assert decisions[0]["status"] == "controlled_proxy_smoke_fail"
    assert "content_agnostic_control_lift_too_high" in decisions[0]["failures"]


def test_controlled_proxy_evaluator_accepts_clean_controlled_proxy() -> None:
    decisions = proxy_eval.evaluate_rows(
        [
            {
                "task": "clean_suite",
                "k": 96,
                "b_match": 0.25,
                "idlekv": 0.85,
                "random_k": 0.27,
                "oldest_k": 0.24,
                "gold_k": 1.0,
            }
        ]
    )

    assert decisions[0]["status"] == "controlled_proxy_smoke_pass"


def test_proxy_progress_monitor_summarizes_wrapped_log_events() -> None:
    text = "\n".join(
        [
            "[phase14-proxy-locked] start 2026-05-03 19:48:54 UTC",
            "[phase14-proxy-locked] n=100 K=48 96 conditions=A/B/B_match/Random-K/Oldest-K/IdleKV/Oracle-K",
            "[mq_niah_4q_split_14_to_23:ex001] A=1.000 B=0.000 "
            "k=48:Bm=0.000/I=1.000/R=0.000/O=0.000/Or=1.000 "
            "k=96:Bm=0.000/I=1.000/R=0.000/O=0.000/Or=1.000",
            "[mq_niah_4q_split_24_to_13:ex001] A=1.000 B=0.500 "
            "k=48:Bm=0.500/I=1.000/R=0.500/O=0.500/Or=1.000 "
            "k=96:Bm=0.500/I=1.000/R=0.500/O=0.500/Or=1.000",
        ]
    )
    progress, summaries = parse_proxy_log(text)

    assert progress == {"mq_niah_4q": 1}
    assert summaries[("mq_niah_4q", 48)].count == 2
    assert summaries[("mq_niah_4q", 48)].mean("b_match") == 0.25
    rendered = render_summary(progress, summaries)
    assert "K=96" in rendered
    assert "max_control_lift=0.000" in rendered
    start, num_samples = parse_run_metadata(text)
    assert num_samples == 100
    assert start == datetime(2026, 5, 3, 19, 48, 54, tzinfo=timezone.utc)
    eta = render_eta(
        start=start,
        num_samples=num_samples,
        summaries=summaries,
        now=datetime(2026, 5, 3, 19, 49, 54, tzinfo=timezone.utc),
    )
    assert "completed 2/700" in eta
    assert "rough remaining" in eta


def test_refresh_evaluator_classifies_refresh_boundary() -> None:
    rows = []
    for k in (24, 48, 96):
        rows.extend(
            [
                {"k": k, "condition": "Matched", "mean_score": 0.20},
                {"k": k, "condition": "StaleQ-K", "mean_score": 0.25},
                {"k": k, "condition": "WrongQ-K", "mean_score": 0.28},
                {"k": k, "condition": "IdleKV", "mean_score": 0.60},
                {"k": k, "condition": "Refresh-K", "mean_score": 0.92},
                {"k": k, "condition": "Gold-K", "mean_score": 0.95},
            ]
        )

    result = smoke_eval.evaluate_refresh_rows(rows)

    assert result["status"] == "refresh_boundary_confirmed"
    assert result["refresh_dominates_k"] == 3


def test_llama_evaluator_demotes_saturated_grid() -> None:
    rows = [
        {
            "k": k,
            "condition_a": 1.0,
            "b_match": 0.50,
            "random_k": 0.55,
            "oldest_k": 0.50,
            "idlekv": 1.0,
            "gold_k": 1.0,
            "num_samples": 2,
        }
        for k in (24, 32, 48, 64)
    ]

    result = smoke_eval.evaluate_llama_rows(rows)

    assert result["status"] == "appendix_portability_only_saturated"
    assert result["saturated"] is True


def test_llama_evaluator_promotes_non_saturated_useful_grid() -> None:
    rows = [
        {
            "k": k,
            "condition_a": 1.0,
            "b_match": 0.45,
            "random_k": 0.48,
            "oldest_k": 0.46,
            "idlekv": score,
            "gold_k": 0.95,
            "num_samples": 2,
        }
        for k, score in ((24, 0.58), (32, 0.66), (48, 0.75), (64, 0.82))
    ]

    result = smoke_eval.evaluate_llama_rows(rows)

    assert result["status"] == "llama_smoke_pass_run_locked"
    assert result["useful_k"] == 4


def test_llama_evaluator_treats_span_ref_shortfall_as_warning() -> None:
    rows = [
        {
            "k": k,
            "condition_a": 1.0,
            "b_match": 0.35,
            "random_k": 0.36,
            "oldest_k": 0.35,
            "idlekv": score,
            "gold_k": gold,
            "num_samples": 24,
        }
        for k, score, gold in ((24, 0.80, 0.70), (32, 0.89, 0.83), (48, 0.98, 1.0), (64, 1.0, 1.0))
    ]

    result = smoke_eval.evaluate_llama_rows(rows)

    assert result["status"] == "llama_smoke_pass_run_locked"
    assert result["useful_k"] == 4
    assert result["decisions"][0]["warnings"] == ["span_ref_diagnostic_below_idlekv"]
    assert result["decisions"][0]["failures"] == []


def test_selector_evaluator_requires_mid_gain_without_high_loss() -> None:
    rows = [
        {"k": 48, "idlekv": 0.50, "idlekv_coverage": 0.58, "idlekv_mmr": 0.49, "gold_k": 0.90},
        {"k": 96, "idlekv": 0.85, "idlekv_coverage": 0.84, "idlekv_mmr": 0.80, "gold_k": 1.00},
    ]

    result = smoke_eval.evaluate_selector_rows(rows)

    assert result["status"] == "selector_smoke_pass"
    assert result["candidates"] == ["coverage"]


def test_selector_evaluator_accepts_coverage_only_locked_summary() -> None:
    rows = [
        {"k": 48, "idlekv": 0.65, "idlekv_coverage": 0.97, "gold_k": 1.00},
        {"k": 96, "idlekv": 0.99, "idlekv_coverage": 0.99, "gold_k": 1.00},
    ]

    result = smoke_eval.evaluate_selector_rows(rows)

    assert result["status"] == "selector_smoke_pass"
    assert result["candidates"] == ["coverage"]
    missing = next(item for item in result["decisions"] if item["variant"] == "mmr")
    assert missing["status"] == "missing"
    assert missing["failures"] == ["missing_variant_metric"]
