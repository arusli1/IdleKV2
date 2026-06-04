#!/usr/bin/env python3
"""Summarize exact question-query repair runtime from a Phase 6/7 JSON artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(pct)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="Phase 6/7 JSON artifact path")
    args = parser.parse_args()

    data = json.loads(args.artifact.read_text())
    rows = data["rows"]

    by_k: dict[int, list[dict]] = {}
    for row in rows:
        by_k.setdefault(int(row["k"]), []).append(row)

    print("k,n,p50_ms,p95_ms,p99_ms,p50_query_ms,p50_score_ms,p50_select_ms,p50_transfer_ms,p50_inject_ms")
    for k in sorted(by_k):
        group = by_k[k]
        query_ms = [float(r["q2_query_rows_s"]) * 1000.0 for r in group]
        score_ms = [float(r["q2_evicted_scoring_s"]) * 1000.0 for r in group]
        select_ms = [float(r["idlekv_selection_s"]) * 1000.0 for r in group]
        transfer_ms = [float(r["idlekv_transfer_ms"]) for r in group]
        inject_ms = [float(r["idlekv_inject_ms"]) for r in group]
        total_ms = [
            q + s + se + t + i
            for q, s, se, t, i in zip(query_ms, score_ms, select_ms, transfer_ms, inject_ms)
        ]
        print(
            ",".join(
                [
                    str(k),
                    str(len(group)),
                    f"{_percentile(total_ms, 0.50):.6f}",
                    f"{_percentile(total_ms, 0.95):.6f}",
                    f"{_percentile(total_ms, 0.99):.6f}",
                    f"{_percentile(query_ms, 0.50):.6f}",
                    f"{_percentile(score_ms, 0.50):.6f}",
                    f"{_percentile(select_ms, 0.50):.6f}",
                    f"{_percentile(transfer_ms, 0.50):.6f}",
                    f"{_percentile(inject_ms, 0.50):.6f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
