#!/usr/bin/env python3
"""Historical CLI scaffold for the log-based Phase 4 profiling path.

The current paper runtime-capacity evidence is produced by
``run_runtime_capacity_profile.py``. This script is retained for the older
Phase 3 log-backed prototype and should be rerun/validated before its outputs
are used in paper claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_ROOT.parents[1]
PHASE3_RESULTS = REPO_ROOT / "phases" / "phase3_eviction" / "results"
DEFAULT_PHASE3_LOG_ROOT = PHASE3_RESULTS / "phase3_eviction_logs"
DEFAULT_BENCHMARK_RUN = "phase1-current-generator-noise-filler-2026-04-21"
DEFAULT_PROFILING_LOG_DIR = (
    DEFAULT_PHASE3_LOG_ROOT / "benchmark" / DEFAULT_BENCHMARK_RUN / "MQNIAH4q" / "snapkv" / "k512"
)
DEFAULT_RAW_EXAMPLES_DIR = PHASE3_RESULTS / "phase3_raw_examples" / DEFAULT_BENCHMARK_RUN
for root in (PHASE_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from src.buffer import (  # noqa: E402
    build_buffer_from_logs,
    build_snapkv_live_fixture,
    compute_feasibility_frontier,
    evaluate_selection_quality,
    format_frontier_table,
    profile_buffer_scoring,
    profile_cpu_to_gpu_transfer,
    profile_end_to_end_repair,
    profile_injection_attention_overhead,
)
from phases.phase2_kv_cache.src.runtime import load_model, load_tokenizer  # noqa: E402


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(chunk.strip()) for chunk in raw.split(",") if chunk.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiling-log-dir",
        default=str(DEFAULT_PROFILING_LOG_DIR),
        help="Phase 3 eviction-log directory root or one leaf task directory to ingest.",
    )
    parser.add_argument(
        "--phase3-log-root",
        default=str(DEFAULT_PHASE3_LOG_ROOT),
        help="Root directory for the current Phase 3 eviction-log tree.",
    )
    parser.add_argument(
        "--raw-examples-dir",
        default=str(DEFAULT_RAW_EXAMPLES_DIR),
        help="Root directory containing the Phase 3 raw-example records.",
    )
    parser.add_argument(
        "--strategy",
        default="l2_norm",
        choices=["l2_norm", "dot_product", "random", "recency_inverse"],
        help="Selection strategy to use for the feasibility summary.",
    )
    parser.add_argument("--max-buffer-tokens", type=int, default=10_000, help="Maximum number of buffered entries.")
    parser.add_argument("--transfer-trials", type=int, default=50, help="Trial count for CPU->GPU transfer profiling.")
    parser.add_argument("--scoring-trials", type=int, default=50, help="Trial count for CPU scoring profiling.")
    parser.add_argument("--attention-trials", type=int, default=20, help="Trial count for attention-overhead profiling.")
    parser.add_argument("--repair-trials", type=int, default=20, help="Trial count for end-to-end repair profiling.")
    parser.add_argument(
        "--transfer-token-counts",
        default="50,100,250,500,1000,2000",
        help="Comma-separated token counts for transfer profiling.",
    )
    parser.add_argument(
        "--scoring-buffer-sizes",
        default="500,1000,2000,5000",
        help="Comma-separated buffer sizes for CPU scoring profiling.",
    )
    parser.add_argument(
        "--attention-extra-token-counts",
        default="50,100,250,500,1000",
        help="Comma-separated extra-token counts for post-injection attention profiling.",
    )
    parser.add_argument(
        "--repair-top-k-values",
        default="50,100,250,500",
        help="Comma-separated repair sizes for end-to-end profiling.",
    )
    parser.add_argument("--live-context-tokens", type=int, default=32_768, help="Context length for the live Phase 4 fixture.")
    parser.add_argument("--live-k-budget", type=int, default=4_096, help="Compressed-cache budget for the live fixture.")
    parser.add_argument("--live-query-len", type=int, default=20, help="Query length for the live attention-overhead fixture.")
    parser.add_argument("--quality-task-key", default="vt_4hop", help="Task key to inspect for selector-quality diagnostics.")
    parser.add_argument("--quality-method", default="snapkv", help="Method to inspect for selector-quality diagnostics.")
    parser.add_argument("--quality-budget", type=int, default=512, help="Budget to inspect for selector-quality diagnostics.")
    parser.add_argument("--quality-top-k", type=int, default=250, help="Top-k value to record in selector-quality diagnostics.")
    parser.add_argument("--quality-max-examples", type=int, default=50, help="Maximum number of Phase 3 records to inspect.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a reduced-count profiling pass for fast validation.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PHASE_ROOT / "results" / "phase4_profiling"),
        help="Directory where JSON profiling artifacts should be written.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _apply_smoke_overrides(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.max_buffer_tokens = min(args.max_buffer_tokens, 2000)
    args.transfer_trials = min(args.transfer_trials, 3)
    args.scoring_trials = min(args.scoring_trials, 3)
    args.attention_trials = min(args.attention_trials, 2)
    args.repair_trials = min(args.repair_trials, 2)
    args.transfer_token_counts = "50,100"
    args.scoring_buffer_sizes = "500,1000"
    args.attention_extra_token_counts = "50,100"
    args.repair_top_k_values = "50,100"
    args.live_context_tokens = min(args.live_context_tokens, 4096)
    args.live_k_budget = min(args.live_k_budget, 512)


def main() -> int:
    args = parse_args()
    _apply_smoke_overrides(args)
    output_dir = Path(args.output_dir)
    transfer_token_counts = _parse_int_list(args.transfer_token_counts)
    scoring_buffer_sizes = _parse_int_list(args.scoring_buffer_sizes)
    attention_extra_counts = _parse_int_list(args.attention_extra_token_counts)
    repair_top_k_values = _parse_int_list(args.repair_top_k_values)
    buffer = build_buffer_from_logs(args.profiling_log_dir, strategy=args.strategy, max_tokens=args.max_buffer_tokens)

    if len(buffer) == 0:
        raise SystemExit("Phase 4 profiling scaffold found no log-backed buffer entries to profile.")

    seed_recent_q = next(iter(buffer._entries.values())).q_vec.unsqueeze(1)
    transfer_results = profile_cpu_to_gpu_transfer(
        buffer,
        n_tokens_list=transfer_token_counts,
        n_trials=args.transfer_trials,
    )
    scoring_results = profile_buffer_scoring(
        buffer,
        seed_recent_q,
        buffer_sizes=scoring_buffer_sizes,
        n_trials=args.scoring_trials,
    )
    frontier = compute_feasibility_frontier(transfer_results, scoring_results, strategy=args.strategy)
    model = load_model()
    tokenizer = load_tokenizer()
    live_fixture = build_snapkv_live_fixture(
        model,
        tokenizer,
        context_tokens=args.live_context_tokens,
        k_budget=args.live_k_budget,
        selection_strategy=args.strategy,
        max_buffer_tokens=args.max_buffer_tokens,
        query_len=args.live_query_len,
    )
    attention_results = profile_injection_attention_overhead(
        model,
        live_fixture.active_cache,
        live_fixture.buffer,
        live_fixture.query_ids,
        extra_token_counts=attention_extra_counts,
        n_trials=args.attention_trials,
    )
    end_to_end_results = profile_end_to_end_repair(
        live_fixture.buffer,
        live_fixture.active_cache,
        top_k_values=repair_top_k_values,
        n_trials=args.repair_trials,
        device=str(live_fixture.active_cache.device),
    )
    selection_quality = evaluate_selection_quality(
        args.phase3_log_root,
        args.raw_examples_dir,
        top_k=args.quality_top_k,
        task_key=args.quality_task_key,
        method=args.quality_method,
        k_budget=args.quality_budget,
        max_examples=args.quality_max_examples,
    )
    run_metadata = {
        "profiling_log_dir": str(Path(args.profiling_log_dir)),
        "phase3_log_root": str(Path(args.phase3_log_root)),
        "raw_examples_dir": str(Path(args.raw_examples_dir)),
        "strategy": args.strategy,
        "max_buffer_tokens": int(args.max_buffer_tokens),
        "transfer_token_counts": list(transfer_token_counts),
        "scoring_buffer_sizes": list(scoring_buffer_sizes),
        "attention_extra_token_counts": list(attention_extra_counts),
        "repair_top_k_values": list(repair_top_k_values),
        "transfer_trials": int(args.transfer_trials),
        "scoring_trials": int(args.scoring_trials),
        "attention_trials": int(args.attention_trials),
        "repair_trials": int(args.repair_trials),
        "smoke": bool(args.smoke),
        "log_buffer_entries": len(buffer),
        "live_fixture": {
            "context_tokens": int(live_fixture.context_tokens),
            "k_budget": int(live_fixture.k_budget),
            "evicted_tokens": int(live_fixture.evicted_tokens),
            "query_tokens": int(live_fixture.query_ids.shape[1]),
            "obs_window_size": int(live_fixture.obs_window_size),
        },
    }

    write_json(output_dir / "transfer_latency.json", transfer_results)
    write_json(output_dir / "scoring_latency.json", scoring_results)
    write_json(output_dir / "attention_overhead.json", attention_results)
    write_json(output_dir / "end_to_end_repair.json", end_to_end_results)
    write_json(output_dir / "feasibility_frontier.json", frontier)
    write_json(output_dir / "selection_quality.json", selection_quality)
    write_json(output_dir / "run_metadata.json", run_metadata)
    (output_dir / "feasibility_frontier.md").write_text(format_frontier_table(frontier) + "\n", encoding="utf-8")
    print(format_frontier_table(frontier))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
