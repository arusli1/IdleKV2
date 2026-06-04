#!/usr/bin/env python3
"""Quick audit for Phase 7 exact-evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _k_value(label: str) -> int:
    if not label.startswith("k"):
        raise ValueError(f"Unexpected K label: {label}")
    return int(label[1:])


def _summarize_curve(curve: dict[str, Any]) -> dict[str, Any]:
    ordered = [(label, curve[label]) for label in sorted(curve, key=_k_value)]
    monotone_breaks: list[str] = []
    previous_idlekv = None
    previous_oracle = None
    max_random_gap: float | None = None
    max_oldest_gap: float | None = None
    for label, metrics in ordered:
        idlekv = float(metrics.get("mean_idlekv", 0.0))
        oracle = float(metrics.get("mean_oracle_k", 0.0))
        if "mean_random_k" in metrics:
            random_gap = idlekv - float(metrics["mean_random_k"])
            max_random_gap = random_gap if max_random_gap is None else max(max_random_gap, random_gap)
        if "mean_oldest_k" in metrics:
            oldest_gap = idlekv - float(metrics["mean_oldest_k"])
            max_oldest_gap = oldest_gap if max_oldest_gap is None else max(max_oldest_gap, oldest_gap)
        if previous_idlekv is not None and idlekv + 1e-9 < previous_idlekv:
            monotone_breaks.append(f"IdleKV drops at {label}: {previous_idlekv:.3f} -> {idlekv:.3f}")
        if previous_oracle is not None and oracle + 1e-9 < previous_oracle:
            monotone_breaks.append(f"Gold-K drops at {label}: {previous_oracle:.3f} -> {oracle:.3f}")
        previous_idlekv = idlekv
        previous_oracle = oracle

    start_label, start_metrics = ordered[0]
    end_label, end_metrics = ordered[-1]
    return {
        "k_start": _k_value(start_label),
        "k_end": _k_value(end_label),
        "condition_a_start": float(start_metrics.get("mean_condition_a", 0.0)),
        "condition_a_end": float(end_metrics.get("mean_condition_a", 0.0)),
        "condition_b_start": float(start_metrics.get("mean_condition_b", 0.0)),
        "condition_b_end": float(end_metrics.get("mean_condition_b", 0.0)),
        "b_match_start": float(start_metrics.get("mean_b_match", 0.0)),
        "b_match_end": float(end_metrics.get("mean_b_match", 0.0)),
        "idlekv_start": float(start_metrics.get("mean_idlekv", 0.0)),
        "idlekv_end": float(end_metrics.get("mean_idlekv", 0.0)),
        "oracle_end": float(end_metrics.get("mean_oracle_k", 0.0)),
        "max_random_gap": float(max_random_gap) if max_random_gap is not None else None,
        "max_oldest_gap": float(max_oldest_gap) if max_oldest_gap is not None else None,
        "pct_idlekv_gt_b_match_end": float(end_metrics.get("pct_idlekv_gt_b_match", 0.0)),
        "monotone_breaks": monotone_breaks,
    }


def _print_summary(name: str, summary: dict[str, Any]) -> None:
    print(f"[{name}]")
    print(
        f"  K {summary['k_start']} -> {summary['k_end']}: "
        f"A {summary['condition_a_start']:.3f} -> {summary['condition_a_end']:.3f}, "
        f"B {summary['condition_b_start']:.3f} -> {summary['condition_b_end']:.3f}, "
        f"matched no-repair {summary['b_match_start']:.3f} -> {summary['b_match_end']:.3f}, "
        f"IdleKV {summary['idlekv_start']:.3f} -> {summary['idlekv_end']:.3f}, "
        f"Gold-K end {summary['oracle_end']:.3f}"
    )
    random_gap = (
        f"{summary['max_random_gap']:.3f}"
        if summary["max_random_gap"] is not None
        else "n/a"
    )
    oldest_gap = (
        f"{summary['max_oldest_gap']:.3f}"
        if summary["max_oldest_gap"] is not None
        else "n/a"
    )
    print(
        f"  End pct(IdleKV > matched no-repair)={summary['pct_idlekv_gt_b_match_end']:.3f}, "
        f"max gaps: IdleKV-Random={random_gap}, IdleKV-Oldest={oldest_gap}"
    )
    if summary["monotone_breaks"]:
        print("  Warnings:")
        for warning in summary["monotone_breaks"]:
            print(f"    - {warning}")


def _metric_at_k(curve: dict[str, Any], *, k: int, key: str) -> float | None:
    for label, metrics in curve.items():
        if _k_value(label) == int(k):
            value = metrics.get(key)
            return None if value is None else float(value)
    return None


def _print_sixq_acceptance(curve: dict[str, Any]) -> None:
    checks: list[tuple[str, bool | None, str]] = []
    a_end = _metric_at_k(curve, k=128, key="mean_condition_a")
    if a_end is not None:
        checks.append(("A @ K=128 >= 0.95", a_end >= 0.95, f"{a_end:.3f}"))
    b_match_end = _metric_at_k(curve, k=128, key="mean_b_match")
    if b_match_end is not None:
        checks.append(("matched no-repair @ K=128 >= 0.10", b_match_end >= 0.10, f"{b_match_end:.3f}"))
    for k, threshold in ((48, 0.10), (96, 0.20)):
        idlekv = _metric_at_k(curve, k=k, key="mean_idlekv")
        b_match = _metric_at_k(curve, k=k, key="mean_b_match")
        if idlekv is not None and b_match is not None:
            lift = idlekv - b_match
            checks.append((f"IdleKV - matched no-repair @ K={k} >= {threshold:.2f}", lift >= threshold, f"{lift:.3f}"))
    idlekv_96 = _metric_at_k(curve, k=96, key="mean_idlekv")
    random_96 = _metric_at_k(curve, k=96, key="mean_random_k")
    oldest_96 = _metric_at_k(curve, k=96, key="mean_oldest_k")
    if idlekv_96 is not None and random_96 is not None:
        checks.append(("IdleKV-Random @ K=96 >= 0.10", (idlekv_96 - random_96) >= 0.10, f"{(idlekv_96 - random_96):.3f}"))
    if idlekv_96 is not None and oldest_96 is not None:
        checks.append(("IdleKV-Oldest @ K=96 >= 0.10", (idlekv_96 - oldest_96) >= 0.10, f"{(idlekv_96 - oldest_96):.3f}"))
    oracle_48 = _metric_at_k(curve, k=48, key="mean_oracle_k")
    idlekv_48 = _metric_at_k(curve, k=48, key="mean_idlekv")
    if oracle_48 is not None and idlekv_48 is not None:
        gold_gap = oracle_48 - idlekv_48
        checks.append(("Gold-K - IdleKV @ K=48 >= 0.05", gold_gap >= 0.05, f"{gold_gap:.3f}"))

    print("[6q acceptance]")
    for label, passed, detail in checks:
        verdict = "PASS" if passed else "FAIL"
        print(f"  {verdict} {label} ({detail})")


def _csv_metric_at_k(rows: list[dict[str, str]], *, k: int, key: str) -> float | None:
    for row in rows:
        if int(row["k"]) == int(k):
            value = row.get(key)
            if value in (None, ""):
                return None
            return float(value)
    return None


def _print_sixq_ci_gate(overall_csv: Path) -> None:
    with overall_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    idlekv_lo = _csv_metric_at_k(rows, k=96, key="idlekv_lo")
    b_match_hi = _csv_metric_at_k(rows, k=96, key="b_match_hi")
    print("[6q bootstrap CI gate]")
    if idlekv_lo is None or b_match_hi is None:
        print("  NOTE missing idlekv_lo or b_match_hi at K=96 in exported overall CSV.")
        return
    passed = idlekv_lo > b_match_hi
    verdict = "PASS" if passed else "FAIL"
    print(
        f"  {verdict} IdleKV_lo @ K=96 > matched-no-repair_hi @ K=96 "
        f"({idlekv_lo:.3f} > {b_match_hi:.3f})"
    )


def _print_sixq_split_gates(by_task: dict[str, dict[str, Any]]) -> None:
    print("[6q per-split gates]")
    for task_name in sorted(by_task):
        curve = by_task[task_name]
        lift_96 = None
        lift_128 = None
        rand_gap_128 = None
        oldest_gap_128 = None
        idlekv_96 = _metric_at_k(curve, k=96, key="mean_idlekv")
        bmatch_96 = _metric_at_k(curve, k=96, key="mean_b_match")
        idlekv_128 = _metric_at_k(curve, k=128, key="mean_idlekv")
        bmatch_128 = _metric_at_k(curve, k=128, key="mean_b_match")
        random_128 = _metric_at_k(curve, k=128, key="mean_random_k")
        oldest_128 = _metric_at_k(curve, k=128, key="mean_oldest_k")
        if idlekv_96 is not None and bmatch_96 is not None:
            lift_96 = idlekv_96 - bmatch_96
        if idlekv_128 is not None and bmatch_128 is not None:
            lift_128 = idlekv_128 - bmatch_128
        if idlekv_128 is not None and random_128 is not None:
            rand_gap_128 = idlekv_128 - random_128
        if idlekv_128 is not None and oldest_128 is not None:
            oldest_gap_128 = idlekv_128 - oldest_128
        pieces = []
        if lift_96 is not None:
            verdict = "PASS" if lift_96 > 0.0 else "FAIL"
            pieces.append(f"{verdict} lift@96>0 ({lift_96:.3f})")
        if lift_128 is not None:
            verdict = "PASS" if lift_128 >= 0.10 else "FAIL"
            pieces.append(f"{verdict} lift@128>=0.10 ({lift_128:.3f})")
        if rand_gap_128 is not None:
            verdict = "PASS" if rand_gap_128 > 0.0 else "FAIL"
            pieces.append(f"{verdict} idle-rand@128>0 ({rand_gap_128:.3f})")
        if oldest_gap_128 is not None:
            verdict = "PASS" if oldest_gap_128 > 0.0 else "FAIL"
            pieces.append(f"{verdict} idle-oldest@128>0 ({oldest_gap_128:.3f})")
        summary = "; ".join(pieces) if pieces else "NOTE missing required metrics"
        print(f"  {task_name}: {summary}")


def _example_level_monotonicity(rows: list[dict[str, Any]], *, score_key: str) -> tuple[int, int]:
    by_example: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if score_key not in row:
            continue
        example_key = f"{row.get('task', '')}:{row['example_id']}"
        by_example.setdefault(example_key, []).append(row)
    total = len(by_example)
    non_monotone = 0
    for example_rows in by_example.values():
        ordered = sorted(example_rows, key=lambda row: int(row["k"]))
        previous = None
        dropped = False
        for row in ordered:
            current = float(row[score_key])
            if previous is not None and current + 1e-9 < previous:
                dropped = True
                break
            previous = current
        if dropped:
            non_monotone += 1
    return non_monotone, total


def _example_level_comparison(
    rows: list[dict[str, Any]],
    *,
    lhs_key: str,
    rhs_key: str,
    k: int | None = None,
) -> tuple[int, int]:
    by_example: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if lhs_key not in row or rhs_key not in row:
            continue
        if k is not None and int(row["k"]) != int(k):
            continue
        example_key = f"{row.get('task', '')}:{row['example_id']}"
        by_example.setdefault(example_key, []).append(row)
    total = len(by_example)
    count = 0
    for example_rows in by_example.values():
        if k is None:
            row = max(example_rows, key=lambda item: int(item["k"]))
        else:
            row = example_rows[0]
        if float(row[lhs_key]) + 1e-9 < float(row[rhs_key]):
            count += 1
    return count, total


def _example_level_gt_comparison(
    rows: list[dict[str, Any]],
    *,
    lhs_key: str,
    rhs_key: str,
    k: int | None = None,
) -> tuple[int, int]:
    by_example: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if lhs_key not in row or rhs_key not in row:
            continue
        if k is not None and int(row["k"]) != int(k):
            continue
        example_key = f"{row.get('task', '')}:{row['example_id']}"
        by_example.setdefault(example_key, []).append(row)
    total = len(by_example)
    count = 0
    for example_rows in by_example.values():
        if k is None:
            row = max(example_rows, key=lambda item: int(item["k"]))
        else:
            row = example_rows[0]
        if float(row[lhs_key]) > float(row[rhs_key]) + 1e-9:
            count += 1
    return count, total


def _example_level_condition_a_below_one(rows: list[dict[str, Any]]) -> tuple[int, int]:
    if not rows:
        return 0, 0
    smallest_k = min(int(row["k"]) for row in rows if "condition_a_score" in row)
    by_example: dict[str, dict[str, Any]] = {}
    for row in rows:
        if "condition_a_score" not in row or int(row["k"]) != smallest_k:
            continue
        example_key = f"{row.get('task', '')}:{row['example_id']}"
        by_example[example_key] = row
    total = len(by_example)
    failures = sum(1 for row in by_example.values() if float(row["condition_a_score"]) + 1e-9 < 1.0)
    return failures, total


def _artifact_task_name(artifact: dict[str, Any]) -> str:
    """Return the task name for both legacy and Phase 6 result shapes."""
    task = artifact.get("task")
    if task:
        return str(task)
    config = artifact.get("config")
    if isinstance(config, dict):
        config_task = config.get("task")
        if config_task:
            return str(config_task)
    return ""


def _restored_count_mismatches(
    rows: list[dict[str, Any]], *, count_key: str, mode: str = "eq"
) -> tuple[int, int]:
    checked = 0
    mismatches = 0
    for row in rows:
        if count_key not in row:
            continue
        checked += 1
        restored = int(row[count_key])
        budget = int(row["k"])
        if mode == "eq":
            bad = restored != budget
        elif mode == "le":
            bad = restored > budget
        else:
            raise ValueError(f"Unsupported restored-count mode: {mode}")
        if bad:
            mismatches += 1
    return mismatches, checked


def _display_metric_name(metric_key: str) -> str:
    labels = {
        "b_match_score": "matched no-repair score",
        "idlekv_score": "IdleKV score",
        "oracle_k_score": "Gold-K score",
        "condition_a_score": "full-cache score",
        "idlekv_restored_count": "IdleKV restored count",
        "random_k_restored_count": "Random-K restored count",
        "oldest_k_restored_count": "Oldest-K restored count",
        "oracle_k_restored_count": "Gold-K restored count",
    }
    return labels.get(metric_key, metric_key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--overall-csv", type=Path)
    args = parser.parse_args()

    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    aggregate = artifact["aggregate"]
    rows = artifact.get("rows", [])
    artifact_task_name = _artifact_task_name(artifact)
    if "overall" in aggregate and "by_task" in aggregate:
        _print_summary("overall", _summarize_curve(aggregate["overall"]))
        for split_task_name in sorted(aggregate["by_task"]):
            _print_summary(split_task_name, _summarize_curve(aggregate["by_task"][split_task_name]))
    else:
        _print_summary(str(artifact.get("task", "overall")), _summarize_curve(aggregate))
    if "6q" in artifact_task_name:
        curve = aggregate["overall"] if "overall" in aggregate else aggregate
        _print_sixq_acceptance(curve)
        if "by_task" in aggregate:
            _print_sixq_split_gates(aggregate["by_task"])
        if args.overall_csv is not None:
            _print_sixq_ci_gate(args.overall_csv)
        else:
            print("  NOTE CI gate requires the exported overall CSV with bootstrap bounds.")
    if rows:
        for score_key in ("b_match_score", "idlekv_score", "oracle_k_score"):
            non_monotone, total = _example_level_monotonicity(rows, score_key=score_key)
            if total:
                print(f"[example-level] {_display_metric_name(score_key)}: {non_monotone}/{total} non-monotone")
        oracle_lt_bm, total = _example_level_comparison(rows, lhs_key="oracle_k_score", rhs_key="b_match_score")
        if total:
            print(
                "[example-level] Gold-K score < matched no-repair score "
                f"at max K: {oracle_lt_bm}/{total}"
            )
        idlekv_gt_a, total = _example_level_gt_comparison(rows, lhs_key="idlekv_score", rhs_key="condition_a_score")
        if total:
            print(f"[example-level] IdleKV score > full-cache score at max K: {idlekv_gt_a}/{total}")
        oracle_gt_a, total = _example_level_gt_comparison(rows, lhs_key="oracle_k_score", rhs_key="condition_a_score")
        if total:
            print(f"[example-level] Gold-K score > full-cache score at max K: {oracle_gt_a}/{total}")
        a_lt_one, total = _example_level_condition_a_below_one(rows)
        if total:
            print(f"[example-level] full-cache score < 1.0: {a_lt_one}/{total}")
        for count_key in (
            "idlekv_restored_count",
            "random_k_restored_count",
            "oldest_k_restored_count",
        ):
            mismatches, checked = _restored_count_mismatches(rows, count_key=count_key, mode="eq")
            if checked:
                print(f"[row-level] {_display_metric_name(count_key)} != K: {mismatches}/{checked}")
        oracle_mismatches, oracle_checked = _restored_count_mismatches(
            rows, count_key="oracle_k_restored_count", mode="le"
        )
        if oracle_checked:
            print(f"[row-level] Gold-K restored count > K: {oracle_mismatches}/{oracle_checked}")


if __name__ == "__main__":
    main()
