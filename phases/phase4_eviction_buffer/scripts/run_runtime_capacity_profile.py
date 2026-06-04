#!/usr/bin/env python3
"""Profile host-to-GPU KV repair capacity.

The modes separate mechanical movement, chunked candidate selection, and an
integrated synthetic repair path that scans an offloaded candidate store,
selects top-K rows, moves selected KV, and reinserts it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_ROOT.parents[1]
for root in (PHASE_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from phases.phase4_eviction_buffer.src.buffer.runtime_capacity import (  # noqa: E402
    KVRuntimeSpec,
    feasibility_rows,
    parse_dtype,
    profile_chunked_selection_capacity,
    profile_chunked_selection_capacity_multi_k,
    profile_end_to_end_repair_capacity,
    profile_end_to_end_repair_capacity_multi_k,
    profile_transfer_inject_capacity,
    write_rows_csv,
)


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(chunk.strip()) for chunk in raw.split(",") if chunk.strip())


def _parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(chunk.strip()) for chunk in raw.split(",") if chunk.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "move_inject",
            "chunked_select",
            "chunked_select_multi_k",
            "end_to_end_repair",
            "end_to_end_repair_multi_k",
        ),
        default="move_inject",
        help="Profile movement, candidate scoring, or integrated synthetic repair.",
    )
    parser.add_argument("--active-tokens", default="8192,32768,100000")
    parser.add_argument("--candidate-tokens", default="32768,100000,250000,500000,1000000")
    parser.add_argument("--k", default="96,500,1000,2000,5000")
    parser.add_argument("--query-len", type=int, default=64)
    parser.add_argument("--chunk-tokens", type=int, default=16384)
    parser.add_argument(
        "--source-pool-chunks",
        type=int,
        default=1,
        help="Distinct pinned host chunks cycled by chunked_select mode.",
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--warmup-trials", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--n-layers", type=int, default=28)
    parser.add_argument("--n-query-heads", type=int, default=28)
    parser.add_argument("--n-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--idle-windows-s", default="0.1,0.5,1,2,5")
    parser.add_argument("--budget-fraction", type=float, default=0.90)
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable pinned host memory for the restored KV block.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Override to a cheap validation run.",
    )
    parser.add_argument(
        "--allow-partial-coverage",
        action="store_true",
        help=(
            "Allow host_pool_coverage<1.0 (re-read pool). Off by default for Phase 18+ "
            "headline runs so we never silently re-read a 16K pool 64x at 1M candidates. "
            "Set this only for legacy compatibility runs."
        ),
    )
    parser.add_argument(
        "--out-prefix",
        default="phases/phase4_eviction_buffer/results/runtime_capacity/runtime_capacity",
        help="Output prefix without extension.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    active_tokens = _parse_int_list(args.active_tokens)
    candidate_tokens = _parse_int_list(args.candidate_tokens)
    k_values = _parse_int_list(args.k)
    idle_windows_s = _parse_float_list(args.idle_windows_s)
    trials = int(args.trials)
    warmup_trials = int(args.warmup_trials)
    if args.smoke:
        active_tokens = tuple(value for value in active_tokens if value <= 8192) or (8192,)
        candidate_tokens = tuple(value for value in candidate_tokens if value <= 32768) or (32768,)
        k_values = tuple(value for value in k_values if value <= 500) or (96,)
        trials = min(trials, 3)
        warmup_trials = min(warmup_trials, 1)

    spec = KVRuntimeSpec(
        n_layers=int(args.n_layers),
        n_query_heads=int(args.n_query_heads),
        n_kv_heads=int(args.n_kv_heads),
        head_dim=int(args.head_dim),
        dtype=parse_dtype(args.dtype),
    )

    rows = []
    if args.mode == "move_inject":
        for active in active_tokens:
            for k in k_values:
                print(f"[runtime-capacity] active={active} k={k} trials={trials}", flush=True)
                row = profile_transfer_inject_capacity(
                    active_tokens=active,
                    k_tokens=k,
                    spec=spec,
                    device=args.device,
                    trials=trials,
                    warmup_trials=warmup_trials,
                    pin_memory=not args.no_pin_memory,
                )
                rows.append(row)
                print(
                    "[runtime-capacity] "
                    f"p50={float(row['p50_total_ms']):.2f}ms "
                    f"p95={float(row['p95_total_ms']):.2f}ms "
                    f"p99={float(row['p99_total_ms']):.2f}ms",
                    flush=True,
                )
    elif args.mode == "chunked_select":
        for candidates in candidate_tokens:
            for k in k_values:
                if k > candidates:
                    continue
                print(
                    f"[runtime-capacity] candidates={candidates} k={k} "
                    f"q_len={args.query_len} chunk={args.chunk_tokens} trials={trials}",
                    flush=True,
                )
                row = profile_chunked_selection_capacity(
                    candidate_tokens=candidates,
                    k_tokens=k,
                    spec=spec,
                    query_len=int(args.query_len),
                    chunk_tokens=int(args.chunk_tokens),
                    source_pool_chunks=int(args.source_pool_chunks),
                    device=args.device,
                    trials=trials,
                    warmup_trials=warmup_trials,
                    pin_memory=not args.no_pin_memory,
                    allow_partial_coverage=bool(args.allow_partial_coverage),
                )
                rows.append(row)
                print(
                    "[runtime-capacity] "
                    f"p50={float(row['p50_total_ms']):.2f}ms "
                    f"p95={float(row['p95_total_ms']):.2f}ms "
                    f"p99={float(row['p99_total_ms']):.2f}ms",
                    flush=True,
                )
    elif args.mode == "chunked_select_multi_k":
        for candidates in candidate_tokens:
            fitting_k = tuple(k for k in k_values if k <= candidates)
            if not fitting_k:
                continue
            print(
                f"[runtime-capacity] candidates={candidates} k={','.join(str(k) for k in fitting_k)} "
                f"q_len={args.query_len} chunk={args.chunk_tokens} "
                f"pool={args.source_pool_chunks} trials={trials}",
                flush=True,
            )
            candidate_rows = profile_chunked_selection_capacity_multi_k(
                candidate_tokens=candidates,
                k_tokens_values=fitting_k,
                spec=spec,
                query_len=int(args.query_len),
                chunk_tokens=int(args.chunk_tokens),
                source_pool_chunks=int(args.source_pool_chunks),
                device=args.device,
                trials=trials,
                warmup_trials=warmup_trials,
                pin_memory=not args.no_pin_memory,
                allow_partial_coverage=bool(args.allow_partial_coverage),
            )
            rows.extend(candidate_rows)
            for row in candidate_rows:
                print(
                    "[runtime-capacity] "
                    f"k={int(row['k'])} "
                    f"p50={float(row['p50_total_ms']):.2f}ms "
                    f"p95={float(row['p95_total_ms']):.2f}ms "
                    f"p99={float(row['p99_total_ms']):.2f}ms",
                    flush=True,
                )
    elif args.mode == "end_to_end_repair":
        for active in active_tokens:
            for candidates in candidate_tokens:
                for k in k_values:
                    if k > candidates:
                        continue
                    print(
                        f"[runtime-capacity] active={active} candidates={candidates} k={k} "
                        f"q_len={args.query_len} chunk={args.chunk_tokens} "
                        f"pool={args.source_pool_chunks} trials={trials}",
                        flush=True,
                    )
                    row = profile_end_to_end_repair_capacity(
                        active_tokens=active,
                        candidate_tokens=candidates,
                        k_tokens=k,
                        spec=spec,
                        query_len=int(args.query_len),
                        chunk_tokens=int(args.chunk_tokens),
                        source_pool_chunks=int(args.source_pool_chunks),
                        device=args.device,
                        trials=trials,
                        warmup_trials=warmup_trials,
                        pin_memory=not args.no_pin_memory,
                        allow_partial_coverage=bool(args.allow_partial_coverage),
                    )
                    rows.append(row)
                    print(
                        "[runtime-capacity] "
                        f"p50={float(row['p50_total_ms']):.2f}ms "
                        f"p95={float(row['p95_total_ms']):.2f}ms "
                        f"p99={float(row['p99_total_ms']):.2f}ms "
                        f"(select p95={float(row['p95_select_ms']):.2f}ms, "
                        f"move+inject p95={float(row['p95_move_inject_ms']):.2f}ms)",
                        flush=True,
                    )
    else:
        for active in active_tokens:
            for candidates in candidate_tokens:
                fitting_k = tuple(k for k in k_values if k <= candidates)
                if not fitting_k:
                    continue
                print(
                    f"[runtime-capacity] active={active} candidates={candidates} "
                    f"k={','.join(str(k) for k in fitting_k)} "
                    f"q_len={args.query_len} chunk={args.chunk_tokens} "
                    f"pool={args.source_pool_chunks} trials={trials}",
                    flush=True,
                )
                candidate_rows = profile_end_to_end_repair_capacity_multi_k(
                    active_tokens=active,
                    candidate_tokens=candidates,
                    k_tokens_values=fitting_k,
                    spec=spec,
                    query_len=int(args.query_len),
                    chunk_tokens=int(args.chunk_tokens),
                    source_pool_chunks=int(args.source_pool_chunks),
                    device=args.device,
                    trials=trials,
                    warmup_trials=warmup_trials,
                    pin_memory=not args.no_pin_memory,
                    allow_partial_coverage=bool(args.allow_partial_coverage),
                )
                rows.extend(candidate_rows)
                for row in candidate_rows:
                    print(
                        "[runtime-capacity] "
                        f"k={int(row['k'])} "
                        f"p50={float(row['p50_total_ms']):.2f}ms "
                        f"p95={float(row['p95_total_ms']):.2f}ms "
                        f"p99={float(row['p99_total_ms']):.2f}ms "
                        f"(scan p95={float(row['p95_scan_ms']):.2f}ms, "
                        f"move+inject p95={float(row['p95_move_inject_ms']):.2f}ms)",
                        flush=True,
                    )

    prefix = Path(args.out_prefix)
    capacity_csv = prefix.with_suffix(".csv")
    feasibility_csv = prefix.with_name(prefix.name + "_feasibility.csv")
    metadata_json = prefix.with_name(prefix.name + "_metadata.json")
    write_rows_csv(rows, capacity_csv)
    fit_rows = []
    if args.mode == "move_inject":
        fit_rows = feasibility_rows(
            rows,
            idle_windows_s=idle_windows_s,
            budget_fraction=float(args.budget_fraction),
        )
        write_rows_csv(fit_rows, feasibility_csv)
    elif args.mode in {"end_to_end_repair", "end_to_end_repair_multi_k"}:
        fit_rows = feasibility_rows(
            rows,
            idle_windows_s=idle_windows_s,
            budget_fraction=float(args.budget_fraction),
            group_fields=("active_tokens", "candidate_tokens"),
        )
        write_rows_csv(fit_rows, feasibility_csv)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.write_text(
        json.dumps(
            {
                "active_tokens": list(active_tokens),
                "candidate_tokens": list(candidate_tokens),
                "k": list(k_values),
                "trials": trials,
                "warmup_trials": warmup_trials,
                "mode": args.mode,
                "device": args.device,
                "dtype": args.dtype,
                "n_layers": args.n_layers,
                "n_query_heads": args.n_query_heads,
                "n_kv_heads": args.n_kv_heads,
                "head_dim": args.head_dim,
                "query_len": args.query_len,
                "chunk_tokens": args.chunk_tokens,
                "source_pool_chunks": args.source_pool_chunks,
                "bytes_per_token": spec.bytes_per_token,
                "idle_windows_s": list(idle_windows_s),
                "budget_fraction": float(args.budget_fraction),
                "pin_memory": not args.no_pin_memory,
                "capacity_csv": str(capacity_csv),
                "feasibility_csv": str(feasibility_csv) if fit_rows else None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[runtime-capacity] wrote {capacity_csv}")
    if fit_rows:
        print(f"[runtime-capacity] wrote {feasibility_csv}")
    print(f"[runtime-capacity] wrote {metadata_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
