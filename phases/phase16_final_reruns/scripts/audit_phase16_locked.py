#!/usr/bin/env python3
"""Audit Phase 16 locked-run artifacts for paper integration decisions."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def _int(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return int(float(value))


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(max(float(q), 0.0), 1.0) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    draws: int = 2000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Return a deterministic percentile bootstrap interval for a mean."""

    samples = [float(value) for value in values]
    if not samples:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    mean = sum(samples) / len(samples)
    if len(samples) == 1 or int(draws) <= 0:
        return {"mean": round(mean, 6), "lo": round(mean, 6), "hi": round(mean, 6), "n": len(samples)}
    rng = random.Random(int(seed))
    boot_means: list[float] = []
    for _ in range(int(draws)):
        boot_means.append(sum(samples[rng.randrange(len(samples))] for _ in samples) / len(samples))
    return {
        "mean": round(mean, 6),
        "lo": round(_percentile(boot_means, 0.025), 6),
        "hi": round(_percentile(boot_means, 0.975), 6),
        "n": len(samples),
    }


def _read_artifact(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError(f"Expected Phase 6 artifact with row list: {path}")
    return payload


def _artifact_from_summary(path: Path) -> Path:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not rows[0].get("artifact"):
        raise ValueError(f"Summary CSV does not contain an artifact column: {path}")
    artifact = Path(rows[0]["artifact"])
    return artifact if artifact.is_absolute() else REPO_ROOT / artifact


def _rows_for_k(rows: Iterable[Mapping[str, Any]], k: int) -> list[Mapping[str, Any]]:
    return [row for row in rows if _int(row, "k") == int(k)]


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(_float(row, key) for row in rows) / len(rows)


def audit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_full_score: float = 0.90,
    min_matched_gap: float = 0.20,
    min_idlekv_gain: float = 0.10,
    min_control_gap: float = 0.05,
    max_saturation_floor: float = 0.97,
    draws: int = 2000,
    seed: int = 20260505,
) -> dict[str, Any]:
    """Return per-K locked-run gates plus a paper placement recommendation."""

    k_values = sorted({_int(row, "k") for row in rows})
    decisions: list[dict[str, Any]] = []
    for k in k_values:
        k_rows = _rows_for_k(rows, k)
        full = _mean(k_rows, "condition_a_score")
        matched = _mean(k_rows, "b_match_score")
        idlekv = _mean(k_rows, "idlekv_score")
        random_k = _mean(k_rows, "random_k_score")
        oldest_k = _mean(k_rows, "oldest_k_score")
        best_control = max(random_k, oldest_k)
        idlekv_vs_matched = [_float(row, "idlekv_score") - _float(row, "b_match_score") for row in k_rows]
        idlekv_vs_random = [_float(row, "idlekv_score") - _float(row, "random_k_score") for row in k_rows]
        idlekv_vs_oldest = [_float(row, "idlekv_score") - _float(row, "oldest_k_score") for row in k_rows]
        idlekv_vs_best_row_control = [
            _float(row, "idlekv_score") - max(_float(row, "random_k_score"), _float(row, "oldest_k_score"))
            for row in k_rows
        ]
        failures: list[str] = []
        if full < min_full_score:
            failures.append("full_context_not_reliable")
        if full - matched < min_matched_gap:
            failures.append("matched_gap_too_small")
        if idlekv - matched < min_idlekv_gain:
            failures.append("weak_idlekv_gain")
        if idlekv - best_control < min_control_gap:
            failures.append("content_agnostic_controls_too_close")
        matched_ci = bootstrap_mean_interval(idlekv_vs_matched, draws=draws, seed=seed + k)
        best_control_ci = bootstrap_mean_interval(idlekv_vs_best_row_control, draws=draws, seed=seed + 1000 + k)
        if float(matched_ci["lo"]) <= 0.0:
            failures.append("paired_gain_ci_touches_zero")
        decisions.append(
            {
                "k": k,
                "n": len(k_rows),
                "full": round(full, 6),
                "matched": round(matched, 6),
                "idlekv": round(idlekv, 6),
                "random_k": round(random_k, 6),
                "oldest_k": round(oldest_k, 6),
                "best_control": round(best_control, 6),
                "matched_gap": round(full - matched, 6),
                "idlekv_gain": round(idlekv - matched, 6),
                "idlekv_minus_best_control": round(idlekv - best_control, 6),
                "idlekv_vs_matched_ci": matched_ci,
                "idlekv_vs_random_ci": bootstrap_mean_interval(idlekv_vs_random, draws=draws, seed=seed + 2000 + k),
                "idlekv_vs_oldest_ci": bootstrap_mean_interval(idlekv_vs_oldest, draws=draws, seed=seed + 3000 + k),
                "idlekv_vs_best_row_control_ci": best_control_ci,
                "failures": failures,
            }
        )

    clean = [item for item in decisions if not item["failures"]]
    positive_matched_ci = [item for item in decisions if float(item["idlekv_vs_matched_ci"]["lo"]) > 0.0]
    positive_control_ci = [item for item in decisions if float(item["idlekv_vs_best_row_control_ci"]["lo"]) > 0.0]
    saturated = bool(decisions) and min(float(item["idlekv"]) for item in decisions) >= max_saturation_floor
    if not decisions:
        recommendation = "rerun_missing_artifact"
        status = "missing_rows"
    elif len(clean) >= 2 and len(positive_matched_ci) >= 2 and len(positive_control_ci) >= 2 and not saturated:
        recommendation = "main_reference_plus_appendix"
        status = "locked_pass"
    elif len(clean) >= 2 and not any("full_context_not_reliable" in item["failures"] for item in decisions):
        recommendation = "appendix_only"
        status = "locked_partial"
    else:
        recommendation = "defer_do_not_include"
        status = "locked_fail"

    return {
        "status": status,
        "recommendation": recommendation,
        "clean_k": len(clean),
        "positive_matched_ci_k": len(positive_matched_ci),
        "positive_control_ci_k": len(positive_control_ci),
        "num_k": len(decisions),
        "saturated": saturated,
        "decisions": decisions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact", type=Path)
    source.add_argument("--summary-csv", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260505)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_path = args.artifact if args.artifact is not None else _artifact_from_summary(args.summary_csv)
    if artifact_path is None:
        raise AssertionError("argparse should require an artifact source")
    artifact_path = artifact_path if artifact_path.is_absolute() else REPO_ROOT / artifact_path
    artifact = _read_artifact(artifact_path)
    result = audit_rows(artifact["rows"], draws=args.draws, seed=args.seed)
    result["artifact"] = str(artifact_path)
    result["config"] = artifact.get("config", {})
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0 if result["status"] in {"locked_pass", "locked_partial"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

