#!/usr/bin/env python3
"""Audit paper-critical Phase 14 risks against current artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return int(float(value))


def _row_by_k(rows: Iterable[dict[str, str]], k: int) -> dict[str, str] | None:
    for row in rows:
        if _int(row, "k", -1) == int(k):
            return row
    return None


def _filter_task(rows: list[dict[str, str]], task: str | None) -> list[dict[str, str]]:
    if task is None or not any("task" in row for row in rows):
        return rows
    return [row for row in rows if row.get("task") == task]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1e-12:
        return float("inf") if numerator > 0 else 0.0
    return numerator / denominator


def _column_has_data(rows: list[dict[str, str]], column: str) -> bool:
    return any(row.get(column, "") not in ("", None) for row in rows)


def _first_existing(paths: Iterable[Path]) -> Path:
    paths = list(paths)
    return next((path for path in paths if path.exists()), paths[0])


def audit_proxy_pair(
    *,
    label: str,
    exact_csv: Path,
    proxy_csv: Path,
    latency_proxy_csv: Path | None = None,
    task: str | None = None,
    headline_k: int = 96,
    retention_gate: float = 0.85,
) -> dict[str, object]:
    latency_proxy_csv = latency_proxy_csv or proxy_csv
    required_paths = (exact_csv, proxy_csv, latency_proxy_csv)
    if not all(path.exists() for path in required_paths):
        return {
            "label": label,
            "status": "missing_artifact",
            "missing": [str(path) for path in required_paths if not path.exists()],
        }

    exact_rows = _filter_task(_read_csv(exact_csv), task)
    proxy_rows = _filter_task(_read_csv(proxy_csv), task)
    latency_proxy_rows = _filter_task(_read_csv(latency_proxy_csv), task)
    exact = _row_by_k(exact_rows, headline_k)
    proxy = _row_by_k(proxy_rows, headline_k)
    latency_proxy = _row_by_k(latency_proxy_rows, headline_k)
    if exact is None or proxy is None or latency_proxy is None:
        return {
            "label": label,
            "status": "missing_headline_k",
            "headline_k": headline_k,
            "task": task,
        }

    exact_lift = _float(exact, "idlekv") - _float(exact, "b_match")
    proxy_lift = _float(proxy, "idlekv") - _float(proxy, "b_match")
    retained_gain = _safe_ratio(proxy_lift, exact_lift)
    absolute_loss = _float(exact, "idlekv") - _float(proxy, "idlekv")
    total_speedup = _safe_ratio(_float(exact, "p50_total_ms"), _float(latency_proxy, "p50_total_ms"))
    score_speedup = _safe_ratio(_float(exact, "p50_score_ms"), _float(latency_proxy, "p50_score_ms"))
    has_controls = all(_column_has_data(proxy_rows, column) for column in ("random_k", "oldest_k", "gold_k"))
    k_points = len({_int(row, "k", -1) for row in proxy_rows if _int(row, "k", -1) >= 0})

    failures: list[str] = []
    if proxy_lift < 0.10:
        failures.append("weak_proxy_lift")
    if retained_gain < retention_gate:
        failures.append("retained_gain_below_gate")
    if absolute_loss > 0.10:
        failures.append("proxy_quality_loss_too_large")
    if total_speedup < 3.0:
        failures.append("total_speedup_below_3x")
    if score_speedup < 3.0:
        failures.append("score_speedup_below_3x")
    if not has_controls:
        failures.append("missing_random_oldest_gold_controls")
    if k_points < 3:
        failures.append("too_few_proxy_k_points")

    if not failures:
        status = "main_ready_proxy_evidence"
    elif not has_controls or k_points < 3:
        status = "needs_controlled_proxy_smoke_or_locked_run"
    else:
        status = "needs_proxy_redesign"

    return {
        "label": label,
        "status": status,
        "headline_k": headline_k,
        "task": task,
        "quality_source": str(proxy_csv),
        "latency_source": str(latency_proxy_csv),
        "exact_lift": round(exact_lift, 6),
        "proxy_lift": round(proxy_lift, 6),
        "retained_gain": round(retained_gain, 6),
        "retention_gate": round(retention_gate, 6),
        "absolute_loss": round(absolute_loss, 6),
        "p50_total_speedup": round(total_speedup, 3),
        "p50_score_speedup": round(score_speedup, 3),
        "has_random_oldest_gold_controls": has_controls,
        "proxy_k_points": k_points,
        "failures": failures,
    }


def audit_specificity(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"status": "missing_artifact", "source": str(path)}
    rows = _read_csv(path)
    by_condition = {row.get("condition", ""): row for row in rows}
    matched = _float(by_condition.get("Matched", {}), "mean_score")
    idle = _float(by_condition.get("IdleKV", {}), "mean_score")
    refresh = _float(by_condition.get("Refresh-K", {}), "mean_score")
    gold = _float(by_condition.get("Gold-K", {}), "mean_score")
    stale = _float(by_condition.get("StaleQ-K", {}), "mean_score")
    wrong = _float(by_condition.get("WrongQ-K", {}), "mean_score")
    failures: list[str] = []
    if idle - matched < 0.20:
        failures.append("weak_idlekv_specificity_gain")
    if idle - max(stale, wrong) < 0.15:
        failures.append("weak_true_query_specificity")
    if refresh - idle > 0.10:
        failures.append("refresh_dominates_idlekv")
    if gold + 1e-9 < idle:
        failures.append("gold_reference_below_idlekv")
    return {
        "status": "boundary_result_needs_framing" if "refresh_dominates_idlekv" in failures else "specificity_result_clean",
        "source": str(path),
        "matched": matched,
        "idlekv": idle,
        "stale_q": stale,
        "wrong_q": wrong,
        "refresh": refresh,
        "gold": gold,
        "idlekv_gain": round(idle - matched, 6),
        "true_minus_best_stale_or_wrong": round(idle - max(stale, wrong), 6),
        "refresh_minus_idlekv": round(refresh - idle, 6),
        "failures": failures,
    }


def audit_llama(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"status": "missing_artifact", "source": str(path)}
    rows = _read_csv(path)
    n = max((_int(row, "num_samples", 0) for row in rows), default=0)
    k_points = len({_int(row, "k", -1) for row in rows if _int(row, "k", -1) >= 0})
    idle_scores = [_float(row, "idlekv") for row in rows]
    saturated = bool(idle_scores) and min(idle_scores) >= 0.95
    matched_gap = max((_float(row, "condition_a") - _float(row, "b_match") for row in rows), default=0.0)
    status = "appendix_only_portability"
    failures: list[str] = []
    if n < 24:
        failures.append("n_below_main_claim_threshold")
    if saturated:
        failures.append("idlekv_saturates_across_grid")
    if k_points < 4:
        failures.append("short_k_grid")
    if not failures:
        status = "candidate_main_model_evidence"
    return {
        "status": status,
        "source": str(path),
        "num_samples": n,
        "k_points": k_points,
        "matched_gap": round(matched_gap, 6),
        "idlekv_min": round(min(idle_scores), 6) if idle_scores else None,
        "idlekv_max": round(max(idle_scores), 6) if idle_scores else None,
        "failures": failures,
    }


def audit_policy_breadth(paths: list[Path]) -> dict[str, object]:
    present = [path for path in paths if path.exists()]
    if len(present) != len(paths):
        return {
            "status": "missing_artifact",
            "missing": [str(path) for path in paths if not path.exists()],
        }
    policy_summaries = []
    for path in present:
        rows = _read_csv(path)
        best_gain = max((_float(row, "idlekv") - _float(row, "b_match") for row in rows), default=0.0)
        n = max((_int(row, "num_samples", 0) for row in rows), default=0)
        k_points = len({_int(row, "k", -1) for row in rows if _int(row, "k", -1) >= 0})
        policy_summaries.append(
            {
                "source": str(path),
                "num_samples": n,
                "k_points": k_points,
                "best_gain": round(best_gain, 6),
            }
        )
    return {
        "status": "protocol_matched_not_faithful_prior_reproduction",
        "policies": policy_summaries,
        "next_exact_policy_if_needed": "Scissorhands",
    }


def audit_all(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    figures = repo_root / "paper" / "figures"
    phase14_results = repo_root / "phases" / "phase14_critical_flaw_closure" / "results"
    controlled_proxy = phase14_results / "proxy_controlled_locked_n100.csv"
    use_controlled_proxy = controlled_proxy.exists()
    proxy = [
        audit_proxy_pair(
            label="4q_proxy",
            exact_csv=figures / "phase9_proxy_exact_4q_reference.csv",
            proxy_csv=controlled_proxy if use_controlled_proxy else figures / "phase9_proxy_4q_full_n100.csv",
            latency_proxy_csv=figures / "phase9_proxy_4q_full_n100.csv",
            task="clean_suite" if use_controlled_proxy else None,
            retention_gate=0.85,
        ),
        audit_proxy_pair(
            label="6q_proxy",
            exact_csv=figures / "phase9_proxy_exact_6q_reference.csv",
            proxy_csv=controlled_proxy if use_controlled_proxy else figures / "phase9_proxy_6q_full_n100.csv",
            latency_proxy_csv=figures / "phase9_proxy_6q_full_n100.csv",
            task="mq_niah_6q_clean_suite" if use_controlled_proxy else None,
            retention_gate=0.80,
        ),
    ]
    status = {
        "proxy_deployability": proxy,
        "specificity_refresh_boundary": audit_specificity(figures / "specificity_locked_n24_k48.csv"),
        "llama_cross_model": audit_llama(
            _first_existing(
                [
                    figures / "llama31_8b_4q_fullgrid_n24.csv",
                    repo_root / "phases" / "phase11_main_robustness" / "results" / "llama31_8b_4q_fullgrid_n24.csv",
                    figures / "llama31_8b_6q_locked_n12_b18432_k64-96-128.csv",
                    repo_root
                    / "phases"
                    / "phase13_iteration_framework"
                    / "results"
                    / "llama31_8b_6q_locked_n12_b18432_k64-96-128.csv",
                ]
            )
        ),
        "policy_breadth": audit_policy_breadth(
            [
                figures / "h2o_4q_fullgrid_n24.csv",
                figures / "streamingllm_4q_fullgrid_n24_b16384.csv",
            ]
        ),
    }
    open_priorities: list[str] = []
    if any(item["status"] != "main_ready_proxy_evidence" for item in proxy):
        open_priorities.append("P0 controlled proxy scorer evidence")
    if status["specificity_refresh_boundary"]["status"] == "boundary_result_needs_framing":
        open_priorities.append("P1 Refresh-K frontier or explicit boundary framing")
    if status["llama_cross_model"]["status"] != "candidate_main_model_evidence":
        open_priorities.append("P2 calibrated non-saturating Llama smoke before any main model claim")
    if status["policy_breadth"]["status"] == "protocol_matched_not_faithful_prior_reproduction":
        open_priorities.append("P3 exact Scissorhands only if policy breadth must be a main claim")
    status["open_priorities"] = open_priorities
    return status


def _print_text(report: dict[str, object]) -> None:
    print("[phase14 readiness]")
    for item in report["proxy_deployability"]:
        print(
            "{label}: {status}; retained_gain={retained_gain} "
            "(gate={retention_gate}); "
            "speedup={p50_total_speedup}x; failures={failures}".format(**item)
        )
    spec = report["specificity_refresh_boundary"]
    print(
        "specificity: {status}; IdleKV gain={idlekv_gain}; "
        "Refresh-Idle={refresh_minus_idlekv}; failures={failures}".format(**spec)
    )
    llama = report["llama_cross_model"]
    print(
        "llama: {status}; n={num_samples}; k_points={k_points}; "
        "failures={failures}".format(**llama)
    )
    policy = report["policy_breadth"]
    print(f"policy breadth: {policy['status']}; next={policy.get('next_exact_policy_if_needed', '')}")
    print("[open priorities]")
    for priority in report["open_priorities"]:
        print(f"- {priority}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_all()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
