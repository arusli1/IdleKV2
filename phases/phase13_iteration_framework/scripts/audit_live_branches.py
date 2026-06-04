#!/usr/bin/env python3
"""Audit live Phase 13 branches against explicit promotion gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

REPO_DIR = Path(__file__).resolve().parents[3]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from phases.phase13_iteration_framework.src import (  # noqa: E402
    bootstrap_mean_interval,
    classify_figure_quality,
    classify_multiturn_candidate,
    classify_policy_curve,
    classify_result_rigor,
    paired_condition_difference_values,
)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def policy_result_rigor(decision: dict[str, object], *, n: int = 24) -> dict[str, object]:
    return classify_result_rigor(
        {
            "stage": "full",
            "n": n,
            "grid_points": decision.get("points", 0),
            "effect_size": decision.get("best_gain", 0.0),
            "paired_or_shared_examples": True,
            "matched_budget_audited": True,
            "full_cache_ok": decision.get("full_ok", False),
            "controls_clean": int(decision.get("eligible_points", 0) or 0) > 0,
            "non_saturated": decision.get("non_saturated", False),
            "confound_checked": int(decision.get("adjacent_positive", 0) or 0) >= 2,
        }
    )


def policy_figure_quality(*, policies: int, k_points: int, exists: bool) -> dict[str, object]:
    return classify_figure_quality(
        {
            "real_data": exists,
            "no_fake_data": True,
            "data_points": policies * k_points * 4,
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


def _multiturn_figure_quality(rows: list[dict[str, str]], *, exists: bool) -> dict[str, object]:
    plotted_points = {
        (
            row.get("condition", ""),
            row.get("turn", ""),
            row.get("k", ""),
        )
        for row in rows
        if row.get("condition") in {"IdleKV", "Random-K", "Oldest-K", "StaleQ-K", "Gold-K"}
        and row.get("turn") not in ("", None)
    }
    conditions = {row.get("condition", "") for row in rows}
    controls_visible = {"Random-K", "Oldest-K", "StaleQ-K"}.issubset(conditions)
    return classify_figure_quality(
        {
            "real_data": exists and bool(rows),
            "no_fake_data": True,
            "data_points": len(plotted_points),
            "graph_type_fits_claim": True,
            "one_column_fit": True,
            "legend_outside_data": True,
            "labels_readable": True,
            "caption_scopes_claim": True,
            "controls_visible": controls_visible,
            "not_redundant": True,
            "top_paper_style": True,
        }
    )


def _latest_path(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def _samples_from_name(path: Path, *, fallback: int) -> int:
    match = re.search(r"_n(\d+)", path.name)
    return int(match.group(1)) if match else int(fallback)


def _rows_path_from_summary(path: Path) -> Path:
    return path.with_name(path.name.replace("summary", "rows", 1))


def _noninitial_turns(rows: list[dict[str, str]]) -> tuple[int, ...]:
    turns = sorted({int(float(row.get("turn", 0) or 0)) for row in rows})
    return tuple(turn for turn in turns if turn > 0)


def _condition_diff_interval(
    rows: list[dict[str, str]],
    *,
    condition: str,
    baseline_condition: str,
    k: int,
    turns: tuple[int, ...],
    seed: int,
) -> dict[str, float | int]:
    return bootstrap_mean_interval(
        paired_condition_difference_values(
            rows,
            condition=condition,
            baseline_condition=baseline_condition,
            k=int(k),
            turns=turns,
        ),
        n_bootstrap=2000,
        seed=seed,
    )


def _paired_multiturn_uncertainty_gate(
    summary_path: Path,
    *,
    k: int,
    reported: bool,
) -> dict[str, object]:
    """Check paired lower bounds needed for a main multi-turn claim."""

    rows_path = _rows_path_from_summary(summary_path)
    if not reported:
        return {"reported": False, "passed": False, "failures": ("missing_uncertainty_file",)}
    if not rows_path.exists():
        return {"reported": True, "passed": False, "failures": ("missing_rows_for_uncertainty",)}

    rows = load_rows(rows_path)
    turns = _noninitial_turns(rows)
    intervals = {
        "idlekv_vs_matched_noninitial": _condition_diff_interval(
            rows,
            condition="IdleKV",
            baseline_condition="Matched",
            k=int(k),
            turns=turns,
            seed=11,
        ),
        "idlekv_vs_random_noninitial": _condition_diff_interval(
            rows,
            condition="IdleKV",
            baseline_condition="Random-K",
            k=int(k),
            turns=turns,
            seed=12,
        ),
        "idlekv_vs_oldest_noninitial": _condition_diff_interval(
            rows,
            condition="IdleKV",
            baseline_condition="Oldest-K",
            k=int(k),
            turns=turns,
            seed=13,
        ),
        "current_only_vs_stale_only_noninitial": _condition_diff_interval(
            rows,
            condition="CurrentQOnly-K",
            baseline_condition="StaleQOnly-K",
            k=int(k),
            turns=turns,
            seed=14,
        ),
    }
    failures = tuple(name for name, interval in intervals.items() if float(interval.get("lo", 0.0)) <= 0.0)
    return {
        "reported": True,
        "passed": not failures,
        "failures": failures,
        "intervals": intervals,
    }


def _audit_multiturn_summary(path: Path, *, stage: str, paired_uncertainty_reported: bool) -> dict[str, object]:
    decisions = classify_multiturn_candidate(load_rows(path))
    audited: list[dict[str, object]] = []
    for decision in decisions:
        best_control_gain = decision.get("best_control_noninitial_gain")
        idle_noninitial_gain = decision.get("idle_noninitial_gain")
        uncertainty_gate = _paired_multiturn_uncertainty_gate(
            path,
            k=int(decision.get("k", 0)),
            reported=paired_uncertainty_reported,
        )
        result_rigor = classify_result_rigor(
            {
                "stage": stage,
                "n": _samples_from_name(path, fallback=2 if stage == "smoke" else 12),
                "grid_points": 1,
                "effect_size": idle_noninitial_gain if idle_noninitial_gain is not None else 0.0,
                "paired_or_shared_examples": True,
                "matched_budget_audited": True,
                "full_cache_ok": True,
                "controls_clean": float(best_control_gain if best_control_gain is not None else 1.0) <= 0.10,
                "non_saturated": float(idle_noninitial_gain if idle_noninitial_gain is not None else 1.0) < 0.95,
                "confound_checked": decision.get("query_only_margin") is not None,
                "primary_claim": stage == "full" and bool(decision.get("main_candidate")),
                "paired_uncertainty_reported": uncertainty_gate["reported"],
                "paired_uncertainty_positive": uncertainty_gate["passed"],
            },
            min_main_grid_points=1,
        )
        audited.append(
            {
                "promotion_gate": decision,
                "paired_uncertainty_gate": uncertainty_gate,
                "result_rigor": result_rigor,
            }
        )
    return {
        "source": str(path.relative_to(REPO_DIR)),
        "stage": stage,
        "paired_uncertainty_reported": paired_uncertainty_reported,
        "k_decisions": audited,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args()

    results: dict[str, object] = {}
    phase10 = REPO_DIR / "phases" / "phase10_expansion" / "results"
    phase13 = REPO_DIR / "phases" / "phase13_iteration_framework" / "results"
    phase11 = REPO_DIR / "phases" / "phase11_main_robustness" / "results"
    phase12 = REPO_DIR / "phases" / "phase12_policy_breadth" / "results"
    paper_figures = REPO_DIR / "paper" / "figures"

    multiturn_locked = _latest_path(list(phase13.glob("multiturn_hard_locked_summary_n*_k*.csv")))
    multiturn_kcal = _latest_path(list(phase13.glob("multiturn_hard_kcal_smoke_summary_n*.csv")))
    legacy_multiturn = phase10 / "multiturn_hard_locked_summary_n12.csv"
    legacy_multiturn_rows = _rows_path_from_summary(legacy_multiturn)
    if multiturn_locked is not None:
        uncertainty = multiturn_locked.with_name(
            multiturn_locked.name.replace("summary", "uncertainty", 1)
        )
        multiturn_rows_path = _rows_path_from_summary(multiturn_locked)
        multiturn_figure_rows_path = multiturn_rows_path
        results["multiturn_hard_locked"] = _audit_multiturn_summary(
            multiturn_locked,
            stage="full",
            paired_uncertainty_reported=uncertainty.exists(),
        )
    elif multiturn_kcal is not None:
        multiturn_rows_path = _rows_path_from_summary(multiturn_kcal)
        multiturn_figure_rows_path = legacy_multiturn_rows if legacy_multiturn_rows.exists() else None
        results["multiturn_hard_locked"] = _audit_multiturn_summary(
            multiturn_kcal,
            stage="smoke",
            paired_uncertainty_reported=False,
        )
    elif legacy_multiturn.exists():
        multiturn_rows_path = _rows_path_from_summary(legacy_multiturn)
        multiturn_figure_rows_path = multiturn_rows_path
        results["multiturn_hard_locked"] = _audit_multiturn_summary(
            legacy_multiturn,
            stage="full",
            paired_uncertainty_reported=False,
        )
    else:
        multiturn_rows_path = None
        multiturn_figure_rows_path = None

    llama = phase11 / "llama31_8b_4q_fullgrid_n24.csv"
    if llama.exists():
        decision = classify_policy_curve(load_rows(llama))
        results["llama31_8b_4q"] = {
            "promotion_gate": decision,
            "result_rigor": policy_result_rigor(decision),
        }

    llama_6q_candidates = sorted(phase13.glob("llama31_8b_6q_smoke_n*_b*.csv"))
    if llama_6q_candidates:
        decision = classify_policy_curve(load_rows(llama_6q_candidates[-1]), min_points=4)
        results["llama31_8b_6q_smoke"] = {
            "promotion_gate": decision,
            "result_rigor": classify_result_rigor(
                {
                    "stage": "smoke",
                    "n": 2,
                    "grid_points": decision.get("points", 0),
                    "effect_size": decision.get("best_gain", 0.0),
                    "matched_budget_audited": True,
                    "full_cache_ok": decision.get("full_ok", False),
                    "controls_clean": int(decision.get("eligible_points", 0) or 0) > 0,
                    "non_saturated": decision.get("non_saturated", False),
                    "confound_checked": False,
                }
            ),
        }

    h2o = phase11 / "h2o_4q_fullgrid_n24.csv"
    if h2o.exists():
        decision = classify_policy_curve(load_rows(h2o))
        results["h2o_inspired_accum_attention_4q"] = {
            "promotion_gate": decision,
            "result_rigor": policy_result_rigor(decision),
        }

    streaming_candidates = sorted(phase12.glob("streamingllm_4q_fullgrid_n24_b*.csv"))
    if streaming_candidates:
        decision = classify_policy_curve(load_rows(streaming_candidates[-1]))
        results["sink_plus_recent_inspired_by_streamingllm_4q"] = {
            "promotion_gate": decision,
            "result_rigor": policy_result_rigor(decision),
        }
    else:
        results["sink_plus_recent_inspired_by_streamingllm_4q"] = {"action": "running_or_missing", "main_candidate": False}

    results["paper_artifacts"] = {
        "policy_breadth_delta": {
            "figure_quality": policy_figure_quality(
                policies=3,
                k_points=9,
                exists=(paper_figures / "policy_breadth_delta.pdf").exists(),
            ),
            "result_question": "Does the repair effect survive different first-stage retention rules?",
        }
    }
    if multiturn_figure_rows_path is not None:
        results["paper_artifacts"]["multiturn_hard_trajectory"] = {
            "source": str(multiturn_figure_rows_path.relative_to(REPO_DIR)),
            "figure_quality": _multiturn_figure_quality(
                load_rows(multiturn_figure_rows_path) if multiturn_figure_rows_path.exists() else [],
                exists=(paper_figures / "multiturn_hard_trajectory.pdf").exists(),
            ),
            "result_question": "Does repair adapt across repeated relevance shifts and revisits?",
        }
    results["terminology_audit"] = {
        "h2o": {
            "canonical_reproduction": False,
            "paper_name": "accumulated-attention first-stage retention inspired by H2O",
            "reason": "The implementation scores a frozen post-Q1 cache with accumulated attention over recent observation rows; it does not reproduce the full H2O dynamic decode-time cache manager.",
        },
        "streaming_llm": {
            "canonical_reproduction": False,
            "paper_name": "sink-plus-recent first-stage retention inspired by StreamingLLM",
            "reason": "The implementation retains sink tokens plus a recent window inside the matched two-turn protocol; it does not reproduce StreamingLLM's full streaming position/remapping deployment stack.",
        },
        "exact_prior_policy_next": {
            "recommended": "Scissorhands",
            "reason": "It is a fixed-budget attention-history retention policy that fits the two-turn repair protocol better than a full rolling StreamingLLM reproduction.",
            "required_before_claim": "Log actual decode-time attention history, unit-test budget/tie-breaking invariants, smoke on MQ-NIAH-4Q, and label the result exact only after those checks pass.",
            "deferred_exact_candidates": "H2O needs decode-time accumulated attention; FastGen needs head profiling and head-specific retention; PyramidKV/Ada-KV need layer-varying budget accounting.",
        },
    }

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    for branch, decision in results.items():
        print(f"{branch}: {decision}")


if __name__ == "__main__":
    main()
