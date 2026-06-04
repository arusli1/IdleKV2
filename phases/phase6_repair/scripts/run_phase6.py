#!/usr/bin/env python3
"""Run the two-turn matched active-cache-budget IdleKV experiment.

Legacy CLI defaults preserve the earlier proxy/hindsight path.
Current exact-mode runs should pass explicit
``--query-scoring-mode exact_q --oracle-mode gold_spans`` flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_ROOT.parents[1]
for root in (PHASE_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from phases.phase6_repair.src.runner import (  # noqa: E402
    DEFAULT_WRONG_QUERY_DONOR_OFFSET,
    STAGE_DEFAULTS,
    build_config,
    run_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGE_DEFAULTS), default="smoke")
    parser.add_argument("--task", default="clean_suite")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=32_768)
    parser.add_argument("--dataset-seed-offset", type=int, default=0)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Local model directory. Defaults to the Phase 2 Qwen2.5-7B-Instruct path.",
    )
    parser.add_argument("--k", nargs="+", type=int, default=None)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--base-context-budget", type=int, default=512)
    parser.add_argument("--recency-window", type=int, default=128)
    parser.add_argument(
        "--initial-compressor",
        choices=("snapkv", "streaming_llm", "h2o", "scissorhands"),
        default="snapkv",
        help="First-stage post-Q1 context compressor used before idle repair.",
    )
    parser.add_argument(
        "--query-scoring-mode",
        choices=("proxy", "exact_q"),
        default="proxy",
        help="Repair scorer mode. Legacy default is proxy; current exact-mode runs should pass exact_q explicitly.",
    )
    parser.add_argument(
        "--oracle-mode",
        choices=("burst_hindsight", "gold_spans"),
        default="burst_hindsight",
        help="Hindsight-reference path. Legacy default is burst_hindsight; current exact-mode runs should pass gold_spans explicitly.",
    )
    parser.add_argument(
        "--wrong-query-mode",
        choices=("phantom_key", "donor_q2"),
        default="phantom_key",
        help="WrongQ-K scorer query. donor_q2 uses another example's true Q2 query as the specificity control.",
    )
    parser.add_argument(
        "--wrong-query-donor-offset",
        type=int,
        default=DEFAULT_WRONG_QUERY_DONOR_OFFSET,
        help="Example-index offset used when --wrong-query-mode donor_q2 is selected.",
    )
    parser.add_argument(
        "--tm-budget-multiplier",
        type=float,
        default=1.05,
        help="Per-example T_repair multiplier for the time-matched conditions "
        "(Refresh-K-budgeted, PageSummary-Quest-inspired). 1.05 default; lower "
        "values force the wall-clock cap to actually bind for the budgeted "
        "scorers. e.g. 0.10 -> ~150ms budget.",
    )
    parser.add_argument(
        "--page-summary-chunk-size",
        type=int,
        default=128,
        help="PageSummary-Quest-inspired chunk size. Smaller -> more Stage 2 "
        "chunks visited per budget. Sensitivity sweep {32, 64, 128, 256} "
        "addresses the fixed-hyperparameter strawman attack.",
    )
    parser.add_argument(
        "--tm-budget-absolute-s",
        type=float,
        default=0.0,
        help="Absolute per-K budget for the time-matched conditions, in "
        "seconds. When > 0, OVERRIDES the T_repair * multiplier. Use this "
        "for tight K-sweep experiments where the n_k amortization would "
        "otherwise produce K-dependent budgets. e.g., 0.150 = 150 ms = "
        "deployment-realistic.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = build_config(
        stage=args.stage,
        task=args.task,
        num_samples=args.num_samples,
        context_length=args.context_length,
        dataset_seed_offset=args.dataset_seed_offset,
        k_values=args.k,
        conditions=args.conditions,
        base_context_budget=args.base_context_budget,
        recency_window=args.recency_window,
        query_scoring_mode=args.query_scoring_mode,
        oracle_mode=args.oracle_mode,
        wrong_query_mode=args.wrong_query_mode,
        wrong_query_donor_offset=args.wrong_query_donor_offset,
        model_dir=args.model_dir if args.model_dir is not None else None,
        initial_compressor=args.initial_compressor,
        tm_budget_multiplier=args.tm_budget_multiplier,
        page_summary_chunk_size=args.page_summary_chunk_size,
        tm_budget_absolute_s=args.tm_budget_absolute_s,
    )
    payload = run_experiment(config)
    print(payload["artifact_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
