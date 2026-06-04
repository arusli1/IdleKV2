#!/usr/bin/env python3
"""Bootstrap paired exact-vs-proxy quality deltas from Phase 6 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping


PairKey = tuple[str, int, int]


def _load_rows(path: Path) -> list[Mapping[str, Any]]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    rows = artifact.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"Artifact has no row list: {path}")
    return rows


def _row_key(row: Mapping[str, Any]) -> PairKey:
    return (str(row["task"]), int(row["index"]), int(row["k"]))


def _rows_by_key(rows: Iterable[Mapping[str, Any]]) -> dict[PairKey, Mapping[str, Any]]:
    return {_row_key(row): row for row in rows}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float((1.0 - weight) * ordered[lower] + weight * ordered[upper])


def _mean_stats(pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, float]:
    exact_scores = [float(exact["idlekv_score"]) for exact, _ in pairs]
    proxy_scores = [float(proxy["idlekv_score"]) for _, proxy in pairs]
    exact_b_match = [float(exact["b_match_score"]) for exact, _ in pairs]
    proxy_b_match = [float(proxy["b_match_score"]) for _, proxy in pairs]
    exact_lift = fmean(exact - b_match for exact, b_match in zip(exact_scores, exact_b_match))
    proxy_lift = fmean(proxy - b_match for proxy, b_match in zip(proxy_scores, proxy_b_match))
    return {
        "exact_idlekv": fmean(exact_scores),
        "proxy_idlekv": fmean(proxy_scores),
        "exact_b_match": fmean(exact_b_match),
        "proxy_b_match": fmean(proxy_b_match),
        "exact_lift": exact_lift,
        "proxy_lift": proxy_lift,
        "retained_lift": proxy_lift / exact_lift if exact_lift > 1e-12 else 0.0,
        "proxy_minus_exact": fmean(proxy - exact for exact, proxy in zip(exact_scores, proxy_scores)),
    }


def _bootstrap(
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    num_bootstrap: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    if not pairs:
        return {}
    if len(pairs) == 1:
        stats = _mean_stats(pairs)
        return {key: (value, value) for key, value in stats.items()}
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(num_bootstrap):
        sampled = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        for key, value in _mean_stats(sampled).items():
            samples.setdefault(key, []).append(value)
    return {key: (_percentile(values, 0.025), _percentile(values, 0.975)) for key, values in samples.items()}


def paired_rows(
    *,
    exact_artifact: Path,
    proxy_artifact: Path,
    k_values: Iterable[int],
    num_bootstrap: int = 1000,
    seed: int = 0,
) -> list[dict[str, float | int | str]]:
    exact_by_key = _rows_by_key(_load_rows(exact_artifact))
    proxy_by_key = _rows_by_key(_load_rows(proxy_artifact))
    output: list[dict[str, float | int | str]] = []
    for k in k_values:
        keys = sorted(key for key in exact_by_key if key[2] == int(k) and key in proxy_by_key)
        pairs = [(exact_by_key[key], proxy_by_key[key]) for key in keys]
        if not pairs:
            raise ValueError(f"No paired rows found for K={k}")
        stats = _mean_stats(pairs)
        bounds = _bootstrap(pairs, num_bootstrap=num_bootstrap, seed=seed + int(k))
        row: dict[str, float | int | str] = {
            "k": int(k),
            "n_pairs": len(pairs),
            "exact_artifact": str(exact_artifact),
            "proxy_artifact": str(proxy_artifact),
        }
        for key, value in stats.items():
            row[key] = value
            if key in bounds:
                lo, hi = bounds[key]
                row[f"{key}_lo"] = lo
                row[f"{key}_hi"] = hi
        output.append(row)
    return output


def write_csv(rows: list[Mapping[str, object]], path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-artifact", type=Path, required=True)
    parser.add_argument("--proxy-artifact", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = paired_rows(
        exact_artifact=args.exact_artifact,
        proxy_artifact=args.proxy_artifact,
        k_values=args.k,
        num_bootstrap=args.num_bootstrap,
        seed=args.seed,
    )
    write_csv(rows, args.out_csv)
    print(args.out_csv)


if __name__ == "__main__":
    main()
