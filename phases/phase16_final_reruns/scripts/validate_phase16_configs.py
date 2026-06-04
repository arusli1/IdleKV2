#!/usr/bin/env python3
"""CPU-only validation for the Phase 16 planned run configs."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phases.phase6_repair.src.runner import build_config  # noqa: E402


RUN_SPECS = (
    {
        "name": "qwen_4q_final",
        "kwargs": {
            "stage": "full",
            "task": "clean_suite",
            "num_samples": 24,
            "base_context_budget": 16384,
            "k_values": (8, 16, 24, 32, 48, 64, 80, 96, 128),
            "conditions": ("A", "B", "B_match", "Random-K", "Oldest-K", "IdleKV", "Oracle-K"),
            "query_scoring_mode": "exact_q",
            "oracle_mode": "gold_spans",
        },
    },
    {
        "name": "qwen_6q_final",
        "kwargs": {
            "stage": "full",
            "task": "mq_niah_6q_clean_suite",
            "num_samples": 24,
            "base_context_budget": 18432,
            "k_values": (16, 24, 32, 48, 64, 80, 96, 128),
            "conditions": ("A", "B", "B_match", "Random-K", "Oldest-K", "IdleKV", "Oracle-K"),
            "query_scoring_mode": "exact_q",
            "oracle_mode": "gold_spans",
        },
    },
    {
        "name": "qwen_8q_final",
        "kwargs": {
            "stage": "full",
            "task": "mq_niah_8q_clean_suite",
            "num_samples": 24,
            "base_context_budget": 18432,
            "k_values": (16, 24, 32, 48, 64, 80, 96, 128),
            "conditions": ("A", "B", "B_match", "Random-K", "Oldest-K", "IdleKV", "Oracle-K"),
            "query_scoring_mode": "exact_q",
            "oracle_mode": "gold_spans",
        },
    },
    {
        "name": "mistral_smoke",
        "kwargs": {
            "stage": "smoke",
            "task": "mq_niah_6q_clean_suite",
            "num_samples": 2,
            "base_context_budget": 16384,
            "k_values": (24, 48, 96),
            "conditions": ("A", "B", "B_match", "Random-K", "Oldest-K", "IdleKV", "Oracle-K"),
            "query_scoring_mode": "exact_q",
            "oracle_mode": "gold_spans",
            "model_dir": REPO_ROOT / "models" / "Mistral-7B-Instruct-v0.3",
        },
    },
    {
        "name": "scissorhands_smoke",
        "kwargs": {
            "stage": "smoke",
            "task": "mq_niah_6q_clean_suite",
            "num_samples": 2,
            "base_context_budget": 18432,
            "k_values": (48, 96, 128),
            "conditions": ("A", "B", "B_match", "Random-K", "Oldest-K", "IdleKV", "Oracle-K"),
            "query_scoring_mode": "exact_q",
            "oracle_mode": "gold_spans",
            "initial_compressor": "scissorhands",
        },
    },
    {
        "name": "refresh_boundary_smoke",
        "kwargs": {
            "stage": "smoke",
            "task": "clean_suite",
            "num_samples": 2,
            "base_context_budget": 16384,
            "k_values": (24, 48, 80, 96),
            "conditions": ("A", "B", "B_match", "StaleQ-K", "WrongQ-K", "Refresh-K", "IdleKV", "Oracle-K"),
            "query_scoring_mode": "exact_q",
            "oracle_mode": "gold_spans",
            "wrong_query_mode": "donor_q2",
        },
    },
)


def main() -> int:
    for spec in RUN_SPECS:
        config = build_config(**spec["kwargs"])
        print(
            f"{spec['name']}: task={config.task} n={config.num_samples} "
            f"B={config.base_context_budget} K={','.join(str(k) for k in config.k_values)} "
            f"model={Path(config.model_dir).name} compressor={config.initial_compressor}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

