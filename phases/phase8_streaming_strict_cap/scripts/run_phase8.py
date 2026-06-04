#!/usr/bin/env python3
"""Run the Phase 8 strict-cap streaming + bounded CPU-spill experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_ROOT.parents[1]
for root in (PHASE_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from phases.phase8_streaming_strict_cap.src.runner import STAGE_DEFAULTS, build_config, run_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGE_DEFAULTS), default="smoke")
    parser.add_argument("--task", default="clean_suite")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--total-context-length", type=int, default=327_680)
    parser.add_argument("--gpu-cache-cap", type=int, default=32_768)
    parser.add_argument("--turn-headroom", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--keep-fraction", type=float, default=0.10)
    parser.add_argument("--spill-fraction", type=float, default=0.10)
    parser.add_argument("--spill-hard-cap", type=int, default=32_768)
    parser.add_argument("--b", nargs="+", type=int, default=None)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--dataset-seed-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=8000)
    parser.add_argument("--max-runtime-s", type=int, default=10_800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = build_config(
        stage=args.stage,
        task=args.task,
        num_samples=args.num_samples,
        total_context_length=args.total_context_length,
        gpu_cache_cap=args.gpu_cache_cap,
        turn_headroom=args.turn_headroom,
        chunk_size=args.chunk_size,
        keep_fraction=args.keep_fraction,
        spill_fraction=args.spill_fraction,
        spill_hard_cap=args.spill_hard_cap,
        b_values=args.b,
        conditions=args.conditions,
        dataset_seed_offset=args.dataset_seed_offset,
        seed=args.seed,
        max_runtime_s=args.max_runtime_s,
    )
    payload = run_experiment(config)
    print(payload["artifact_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

