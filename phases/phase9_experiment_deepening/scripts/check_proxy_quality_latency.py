#!/usr/bin/env python3
"""Check fixed-K proxy quality-latency gates against an exact-scoring reference."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Check:
    label: str
    passed: bool
    detail: str


def _load_rows(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            numeric: dict[str, float] = {}
            for key, value in row.items():
                if not value:
                    continue
                try:
                    numeric[key] = float(value)
                except ValueError:
                    continue
            if "k" in numeric:
                rows[int(numeric["k"])] = numeric
    return rows


def _require_row(rows: dict[int, dict[str, float]], k: int, *, label: str) -> dict[str, float]:
    if k not in rows:
        raise ValueError(f"{label} CSV has no row for K={k}")
    return rows[k]


def _lift(row: dict[str, float]) -> float:
    return float(row["idlekv"]) - float(row["b_match"])


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1e-12:
        return float("inf") if numerator > 0 else 0.0
    return numerator / denominator


def evaluate_proxy_quality_latency(
    *,
    exact_csv: Path,
    proxy_csv: Path,
    headline_k: int = 96,
    guardrail_k: int | None = 48,
    min_headline_lift: float = 0.10,
    min_headline_retention: float = 0.85,
    max_headline_loss: float = 0.10,
    min_total_speedup: float = 3.0,
    min_score_speedup: float = 3.0,
    min_guardrail_lift: float = 0.0,
    min_guardrail_retention: float = 0.50,
) -> list[Check]:
    exact_rows = _load_rows(exact_csv)
    proxy_rows = _load_rows(proxy_csv)
    exact = _require_row(exact_rows, headline_k, label="exact")
    proxy = _require_row(proxy_rows, headline_k, label="proxy")

    exact_lift = _lift(exact)
    proxy_lift = _lift(proxy)
    retention = _ratio(proxy_lift, exact_lift)
    score_loss = float(exact["idlekv"]) - float(proxy["idlekv"])
    total_speedup = _ratio(float(exact["p50_total_ms"]), float(proxy["p50_total_ms"]))
    score_speedup = _ratio(float(exact["p50_score_ms"]), float(proxy["p50_score_ms"]))

    checks = [
        Check(
            f"headline proxy lift @ K={headline_k} >= {min_headline_lift:.3f}",
            proxy_lift >= min_headline_lift,
            f"lift={proxy_lift:.3f}",
        ),
        Check(
            f"headline lift retention @ K={headline_k} >= {min_headline_retention:.2f}",
            retention >= min_headline_retention,
            f"retention={retention:.3f}; exact_lift={exact_lift:.3f}; proxy_lift={proxy_lift:.3f}",
        ),
        Check(
            f"headline absolute loss @ K={headline_k} <= {max_headline_loss:.3f}",
            score_loss <= max_headline_loss,
            f"loss={score_loss:.3f}; exact={exact['idlekv']:.3f}; proxy={proxy['idlekv']:.3f}",
        ),
        Check(
            f"p50 total speedup @ K={headline_k} >= {min_total_speedup:.1f}x",
            total_speedup >= min_total_speedup,
            f"speedup={total_speedup:.2f}x",
        ),
        Check(
            f"p50 scoring speedup @ K={headline_k} >= {min_score_speedup:.1f}x",
            score_speedup >= min_score_speedup,
            f"speedup={score_speedup:.2f}x",
        ),
    ]

    if guardrail_k is not None:
        exact_guardrail = _require_row(exact_rows, guardrail_k, label="exact")
        proxy_guardrail = _require_row(proxy_rows, guardrail_k, label="proxy")
        exact_guardrail_lift = _lift(exact_guardrail)
        proxy_guardrail_lift = _lift(proxy_guardrail)
        guardrail_retention = _ratio(proxy_guardrail_lift, exact_guardrail_lift)
        checks.extend(
            [
                Check(
                    f"guardrail proxy lift @ K={guardrail_k} >= {min_guardrail_lift:.3f}",
                    proxy_guardrail_lift >= min_guardrail_lift,
                    f"lift={proxy_guardrail_lift:.3f}",
                ),
                Check(
                    f"guardrail lift retention @ K={guardrail_k} >= {min_guardrail_retention:.2f}",
                    guardrail_retention >= min_guardrail_retention,
                    (
                        f"retention={guardrail_retention:.3f}; "
                        f"exact_lift={exact_guardrail_lift:.3f}; proxy_lift={proxy_guardrail_lift:.3f}"
                    ),
                ),
            ]
        )
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-csv", type=Path, required=True)
    parser.add_argument("--proxy-csv", type=Path, required=True)
    parser.add_argument("--headline-k", type=int, default=96)
    parser.add_argument("--guardrail-k", type=int, default=48)
    parser.add_argument("--no-guardrail", action="store_true")
    parser.add_argument("--min-headline-lift", type=float, default=0.10)
    parser.add_argument("--min-headline-retention", type=float, default=0.85)
    parser.add_argument("--max-headline-loss", type=float, default=0.10)
    parser.add_argument("--min-total-speedup", type=float, default=3.0)
    parser.add_argument("--min-score-speedup", type=float, default=3.0)
    parser.add_argument("--min-guardrail-lift", type=float, default=0.0)
    parser.add_argument("--min-guardrail-retention", type=float, default=0.50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = evaluate_proxy_quality_latency(
        exact_csv=args.exact_csv,
        proxy_csv=args.proxy_csv,
        headline_k=args.headline_k,
        guardrail_k=None if args.no_guardrail else args.guardrail_k,
        min_headline_lift=args.min_headline_lift,
        min_headline_retention=args.min_headline_retention,
        max_headline_loss=args.max_headline_loss,
        min_total_speedup=args.min_total_speedup,
        min_score_speedup=args.min_score_speedup,
        min_guardrail_lift=args.min_guardrail_lift,
        min_guardrail_retention=args.min_guardrail_retention,
    )
    all_passed = True
    print("[proxy quality-latency gate]")
    for check in checks:
        all_passed = all_passed and check.passed
        status = "PASS" if check.passed else "FAIL"
        print(f"{status} {check.label} ({check.detail})")
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
