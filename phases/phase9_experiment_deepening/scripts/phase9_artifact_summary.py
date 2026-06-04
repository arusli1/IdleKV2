#!/usr/bin/env python3
"""Summarize Phase 9 candidate artifacts and print smoke-promotion gates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phases.phase6_repair.src.reporting import runtime_rows as phase6_runtime_rows  # noqa: E402


def _k_value(k_label: str) -> int:
    if not k_label.startswith("k"):
        raise ValueError(f"Expected k-prefixed label, got: {k_label}")
    return int(k_label[1:])


def _overall_curve(artifact: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    aggregate = artifact.get("aggregate", {})
    if "overall" in aggregate:
        return aggregate["overall"]
    return aggregate


def _config_value(artifact: Mapping[str, Any], key: str, default: object = "") -> object:
    config = artifact.get("config", {})
    if isinstance(config, Mapping) and key in config:
        return config[key]
    if key in artifact:
        return artifact[key]
    return default


def _metric(curve: Mapping[str, Mapping[str, Any]], *, k: int, key: str) -> float | None:
    metrics = curve.get(f"k{k}", {})
    if key not in metrics:
        return None
    return float(metrics[key])


def _optional_float(metrics: Mapping[str, Any], key: str) -> float | None:
    if key not in metrics:
        return None
    return float(metrics[key])


def frontier_rows_for_artifact(artifact: Mapping[str, Any], *, artifact_path: str = "") -> list[dict[str, object]]:
    """Return paper/appendix-ready rows for one artifact's overall frontier."""
    curve = _overall_curve(artifact)
    config = artifact.get("config", {})
    task = str(_config_value(artifact, "task", artifact.get("task", "")))
    runtime_by_k = {
        int(row["k"]): row
        for row in phase6_runtime_rows(artifact.get("rows", []), by_task=False)
    }
    rows: list[dict[str, object]] = []
    for k_label in sorted(curve, key=_k_value):
        metrics = curve[k_label]
        k = _k_value(k_label)
        b_match = _optional_float(metrics, "mean_b_match")
        idlekv = _optional_float(metrics, "mean_idlekv")
        oracle = _optional_float(metrics, "mean_oracle_k")
        wrong_q = _optional_float(metrics, "mean_wrong_q_k")
        stale_q = _optional_float(metrics, "mean_stale_q_k")
        contrastive_q = _optional_float(metrics, "mean_contrastive_q_k")
        row: dict[str, object] = {
            "artifact": artifact_path,
            "task": task,
            "stage": _config_value(artifact, "stage"),
            "query_scoring_mode": _config_value(artifact, "query_scoring_mode"),
            "oracle_mode": _config_value(artifact, "oracle_mode"),
            "wrong_query_mode": _config_value(artifact, "wrong_query_mode"),
            "wrong_query_donor_offset": _config_value(artifact, "wrong_query_donor_offset"),
            "base_context_budget": _config_value(artifact, "base_context_budget"),
            "recency_window": _config_value(artifact, "recency_window"),
            "initial_compressor": _config_value(artifact, "initial_compressor", "snapkv"),
            "num_samples": _config_value(artifact, "num_samples"),
            "dataset_seed_offset": _config_value(artifact, "dataset_seed_offset", 0),
            "k": k,
        }
        field_map = (
            ("condition_a", "mean_condition_a"),
            ("condition_b", "mean_condition_b"),
            ("b_match", "mean_b_match"),
            ("idlekv", "mean_idlekv"),
            ("idlekv_coverage", "mean_idlekv_coverage"),
            ("idlekv_mmr", "mean_idlekv_mmr"),
            ("wrong_q_k", "mean_wrong_q_k"),
            ("stale_q_k", "mean_stale_q_k"),
            ("contrastive_q_k", "mean_contrastive_q_k"),
            ("random_k", "mean_random_k"),
            ("oldest_k", "mean_oldest_k"),
            ("gold_k", "mean_oracle_k"),
        )
        for out_key, metric_key in field_map:
            value = _optional_float(metrics, metric_key)
            if value is not None:
                row[out_key] = value
        if b_match is not None and idlekv is not None:
            row["idlekv_lift"] = idlekv - b_match
        idlekv_coverage = _optional_float(metrics, "mean_idlekv_coverage")
        idlekv_mmr = _optional_float(metrics, "mean_idlekv_mmr")
        if b_match is not None and idlekv_coverage is not None:
            row["idlekv_coverage_lift"] = idlekv_coverage - b_match
        if b_match is not None and idlekv_mmr is not None:
            row["idlekv_mmr_lift"] = idlekv_mmr - b_match
        if idlekv is not None and idlekv_coverage is not None:
            row["coverage_minus_idlekv"] = idlekv_coverage - idlekv
        if idlekv is not None and idlekv_mmr is not None:
            row["mmr_minus_idlekv"] = idlekv_mmr - idlekv
        if b_match is not None and wrong_q is not None:
            row["wrong_q_lift"] = wrong_q - b_match
        if b_match is not None and stale_q is not None:
            row["stale_q_lift"] = stale_q - b_match
        if idlekv is not None and wrong_q is not None:
            row["true_minus_wrong_q"] = idlekv - wrong_q
        if idlekv is not None and stale_q is not None:
            row["true_minus_stale_q"] = idlekv - stale_q
        if b_match is not None and contrastive_q is not None:
            row["contrastive_q_lift"] = contrastive_q - b_match
        if idlekv is not None and contrastive_q is not None:
            row["contrastive_minus_idlekv"] = contrastive_q - idlekv
        if oracle is not None and idlekv is not None:
            row["gold_headroom"] = oracle - idlekv
        if oracle is not None and b_match is not None and idlekv is not None and oracle > b_match:
            row["gold_normalized_recovery"] = (idlekv - b_match) / (oracle - b_match)
        if isinstance(config, Mapping) and "conditions" in config:
            row["conditions"] = " ".join(str(value) for value in config["conditions"])
        runtime = runtime_by_k.get(k)
        if runtime is not None:
            for key, value in runtime.items():
                if key not in {"k", "n", "task"}:
                    row[key] = value
        rows.append(row)
    return rows


def future_query_gate(
    artifact: Mapping[str, Any],
    *,
    mid_k: int = 48,
    high_k: int = 96,
    min_mid_separation: float = 0.05,
    min_high_separation: float = 0.10,
    max_wrong_q_lift_high: float = 0.10,
) -> list[tuple[str, bool, str]]:
    """Evaluate whether a WrongQ-K smoke is worth promoting."""
    curve = _overall_curve(artifact)
    checks: list[tuple[str, bool, str]] = []
    for k, threshold in ((mid_k, min_mid_separation), (high_k, min_high_separation)):
        idlekv = _metric(curve, k=k, key="mean_idlekv")
        wrong_q = _metric(curve, k=k, key="mean_wrong_q_k")
        if idlekv is None or wrong_q is None:
            checks.append((f"true Q2 minus wrong query @ K={k}", False, "missing mean_idlekv or mean_wrong_q_k"))
            continue
        delta = idlekv - wrong_q
        checks.append((f"true Q2 minus wrong query @ K={k} >= {threshold:.3f}", delta >= threshold, f"delta={delta:.3f}"))
    wrong_q = _metric(curve, k=high_k, key="mean_wrong_q_k")
    b_match = _metric(curve, k=high_k, key="mean_b_match")
    if wrong_q is None or b_match is None:
        checks.append((f"wrong-query lift @ K={high_k} <= {max_wrong_q_lift_high:.3f}", False, "missing mean_wrong_q_k or mean_b_match"))
    else:
        lift = wrong_q - b_match
        checks.append((f"wrong-query lift @ K={high_k} <= {max_wrong_q_lift_high:.3f}", lift <= max_wrong_q_lift_high, f"lift={lift:.3f}"))
    return checks


def proxy_gate(
    artifact: Mapping[str, Any],
    *,
    k: int = 96,
    min_sixq_lift: float = 0.10,
    min_fourq_lift: float = 0.20,
) -> list[tuple[str, bool, str]]:
    """Evaluate whether a proxy-scoring smoke is worth promoting."""
    task = str(_config_value(artifact, "task", "")).lower()
    threshold = min_sixq_lift if "6q" in task else min_fourq_lift
    curve = _overall_curve(artifact)
    idlekv = _metric(curve, k=k, key="mean_idlekv")
    b_match = _metric(curve, k=k, key="mean_b_match")
    if idlekv is None or b_match is None:
        return [(f"proxy IdleKV lift @ K={k} >= {threshold:.3f}", False, "missing mean_idlekv or mean_b_match")]
    lift = idlekv - b_match
    return [(f"proxy IdleKV lift @ K={k} >= {threshold:.3f}", lift >= threshold, f"lift={lift:.3f}")]


def contrastive_gate(
    artifact: Mapping[str, Any],
    *,
    mid_k: int = 48,
    high_k: int = 96,
    min_mid_gain_over_idlekv: float = 0.05,
    min_mid_gain_over_wrong_q: float = 0.05,
    max_high_loss_vs_idlekv: float = 0.02,
) -> list[tuple[str, bool, str]]:
    """Evaluate whether ContrastiveQ-K is worth promoting beyond smoke tests."""
    curve = _overall_curve(artifact)
    checks: list[tuple[str, bool, str]] = []
    mid_contrastive = _metric(curve, k=mid_k, key="mean_contrastive_q_k")
    mid_idlekv = _metric(curve, k=mid_k, key="mean_idlekv")
    mid_wrong = _metric(curve, k=mid_k, key="mean_wrong_q_k")
    if mid_contrastive is None or mid_idlekv is None:
        checks.append((f"ContrastiveQ-K minus IdleKV @ K={mid_k}", False, "missing contrastive or idlekv metric"))
    else:
        gain = mid_contrastive - mid_idlekv
        checks.append(
            (
                f"ContrastiveQ-K minus IdleKV @ K={mid_k} >= {min_mid_gain_over_idlekv:.3f}",
                gain >= min_mid_gain_over_idlekv,
                f"gain={gain:.3f}",
            )
        )
    if mid_contrastive is None or mid_wrong is None:
        checks.append((f"ContrastiveQ-K minus donor WrongQ-K @ K={mid_k}", False, "missing contrastive or wrong-query metric"))
    else:
        gain = mid_contrastive - mid_wrong
        checks.append(
            (
                f"ContrastiveQ-K minus donor WrongQ-K @ K={mid_k} >= {min_mid_gain_over_wrong_q:.3f}",
                gain >= min_mid_gain_over_wrong_q,
                f"gain={gain:.3f}",
            )
        )
    high_contrastive = _metric(curve, k=high_k, key="mean_contrastive_q_k")
    high_idlekv = _metric(curve, k=high_k, key="mean_idlekv")
    if high_contrastive is None or high_idlekv is None:
        checks.append((f"ContrastiveQ-K high-budget loss @ K={high_k}", False, "missing contrastive or idlekv metric"))
    else:
        loss = high_idlekv - high_contrastive
        checks.append(
            (
                f"ContrastiveQ-K loss vs IdleKV @ K={high_k} <= {max_high_loss_vs_idlekv:.3f}",
                loss <= max_high_loss_vs_idlekv,
                f"loss={loss:.3f}",
            )
        )
    return checks


def stale_query_gate(
    artifact: Mapping[str, Any],
    *,
    mid_k: int = 48,
    high_k: int = 96,
    min_mid_separation: float = 0.15,
    max_stale_fraction_of_true_lift: float = 0.50,
) -> list[tuple[str, bool, str]]:
    """Evaluate whether future-query repair beats previous-query repair."""
    curve = _overall_curve(artifact)
    checks: list[tuple[str, bool, str]] = []
    idlekv = _metric(curve, k=mid_k, key="mean_idlekv")
    stale = _metric(curve, k=mid_k, key="mean_stale_q_k")
    if idlekv is None or stale is None:
        checks.append((f"true Q2 minus stale query @ K={mid_k}", False, "missing mean_idlekv or mean_stale_q_k"))
    else:
        delta = idlekv - stale
        checks.append((f"true Q2 minus stale query @ K={mid_k} >= {min_mid_separation:.3f}", delta >= min_mid_separation, f"delta={delta:.3f}"))
    idlekv_high = _metric(curve, k=high_k, key="mean_idlekv")
    stale_high = _metric(curve, k=high_k, key="mean_stale_q_k")
    b_match_high = _metric(curve, k=high_k, key="mean_b_match")
    if idlekv_high is None or stale_high is None or b_match_high is None:
        checks.append((f"stale-query lift fraction @ K={high_k}", False, "missing high-budget metrics"))
    else:
        true_lift = idlekv_high - b_match_high
        stale_lift = stale_high - b_match_high
        fraction = stale_lift / true_lift if true_lift > 1e-12 else float("inf")
        checks.append(
            (
                f"stale-query lift fraction @ K={high_k} <= {max_stale_fraction_of_true_lift:.2f}",
                fraction <= max_stale_fraction_of_true_lift + 1e-6,
                f"fraction={fraction:.3f}",
            )
        )
    return checks


def _print_checks(title: str, checks: Iterable[tuple[str, bool, str]]) -> bool:
    print(f"[{title}]")
    all_passed = True
    for label, passed, detail in checks:
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"{status} {label} ({detail})")
    return all_passed


def write_csv(rows: list[Mapping[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("Cannot write empty CSV")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, action="append", required=True, help="Phase 6 JSON artifact to summarize")
    parser.add_argument("--out-csv", type=Path, default=None, help="Optional combined frontier CSV")
    parser.add_argument(
        "--gate",
        choices=("summary", "future-query", "proxy", "contrastive", "stale-query"),
        default="summary",
        help="Promotion gate to print for each artifact",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_rows: list[dict[str, object]] = []
    all_passed = True
    for path in args.artifact:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        rows = frontier_rows_for_artifact(artifact, artifact_path=str(path))
        all_rows.extend(rows)
        task = str(_config_value(artifact, "task", path.stem))
        print(f"[artifact] {path}")
        print(f"task={task} rows={len(rows)}")
        if args.gate == "future-query":
            all_passed = _print_checks("future-query gate", future_query_gate(artifact)) and all_passed
        elif args.gate == "proxy":
            all_passed = _print_checks("proxy gate", proxy_gate(artifact)) and all_passed
        elif args.gate == "contrastive":
            all_passed = _print_checks("contrastive gate", contrastive_gate(artifact)) and all_passed
        elif args.gate == "stale-query":
            all_passed = _print_checks("stale-query gate", stale_query_gate(artifact)) and all_passed
    if args.out_csv is not None:
        write_csv(all_rows, args.out_csv)
        print(args.out_csv)
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
