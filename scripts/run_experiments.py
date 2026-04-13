#!/usr/bin/env python3
"""
Run the full IdleKV experiment suite.

Reads config from YAML file. Runs all baselines, IdleKV configurations,
and ablations. Saves structured JSON results.

Usage:
    python scripts/run_experiments.py --config configs/main.yaml
    python scripts/run_experiments.py --config configs/main.yaml --only-baselines
    python scripts/run_experiments.py --config configs/main.yaml --model llama8b --ratio 0.5
"""

import argparse
import json
import yaml
import sys
import time
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_kvpress_baseline(model, tokenizer, benchmark_fn, method, ratio, **kwargs):
    """Run a kvpress-based baseline (SnapKV, H2O, StreamingLLM)."""
    import kvpress

    press_map = {
        "snapkv": kvpress.SnapKVPress,
        "h2o": kvpress.ObservedAttentionPress,
        "streaminglm": kvpress.StreamingLLMPress,
    }

    if method not in press_map:
        raise ValueError(f"Unknown method: {method}")

    press_cls = press_map[method]
    press_kwargs = {"compression_ratio": ratio} if ratio else {}
    press = press_cls(**press_kwargs)

    # Run benchmark with kvpress
    return benchmark_fn(model, tokenizer, press=press, **kwargs)


def run_idlekv(model, tokenizer, benchmark_fn, ratio, idle_budget_ms, phases, **kwargs):
    """Run IdleKV with specified idle budget and phases."""
    from idlekv.core.compression import CompressedKVManager

    manager = CompressedKVManager(model, compression_ratio=ratio)

    # Run benchmark with IdleKV manager
    return benchmark_fn(
        model, tokenizer,
        manager=manager,
        idle_budget_ms=idle_budget_ms,
        phases=phases,
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main.yaml")
    parser.add_argument("--only-baselines", action="store_true")
    parser.add_argument("--only-idlekv", action="store_true")
    parser.add_argument("--only-ablations", action="store_true")
    parser.add_argument("--model", type=str, default=None, help="Run only this model (short name)")
    parser.add_argument("--ratio", type=float, default=None, help="Run only this ratio")
    parser.add_argument("--seed", type=int, default=None, help="Run only this seed")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter models
    models = config["models"]
    if args.model:
        models = [m for m in models if m["short"] == args.model]

    seeds = [args.seed] if args.seed else config["evaluation"]["seeds"]

    # Build experiment matrix
    experiments = []

    for model_cfg in models:
        for seed in seeds:
            if not args.only_idlekv and not args.only_ablations:
                # Baselines
                for bl in config["baselines"]:
                    experiments.append({
                        "type": "baseline",
                        "model": model_cfg,
                        "baseline": bl,
                        "seed": seed,
                    })

            if not args.only_baselines and not args.only_ablations:
                # IdleKV at primary ratio across idle budgets
                ratio = args.ratio or config["compression"]["primary_ratio"]
                for budget in config["idlekv"]["idle_budgets_ms"]:
                    for phases in config["idlekv"]["phases"]:
                        experiments.append({
                            "type": "idlekv",
                            "model": model_cfg,
                            "ratio": ratio,
                            "idle_budget_ms": budget,
                            "phases": phases,
                            "seed": seed,
                        })

            if not args.only_baselines and not args.only_idlekv:
                # Ablations: shadow buffer size
                for buf_size in config["ablations"]["shadow_buffer_sizes"]:
                    experiments.append({
                        "type": "ablation_buffer",
                        "model": model_cfg,
                        "ratio": config["compression"]["primary_ratio"],
                        "shadow_buffer_size": buf_size,
                        "seed": seed,
                    })

    print(f"Total experiments: {len(experiments)}")
    if args.dry_run:
        for i, exp in enumerate(experiments):
            print(f"  [{i+1}] {exp['type']}: model={exp['model']['short']}, seed={exp['seed']}, "
                  f"{json.dumps({k:v for k,v in exp.items() if k not in ('type','model','seed')})}")
        return

    # Run experiments
    run_log = {
        "config": config,
        "start_time": datetime.now().isoformat(),
        "results": [],
    }

    for i, exp in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(experiments)}: {exp['type']}")
        print(f"  Model: {exp['model']['short']}, Seed: {exp['seed']}")
        print(f"{'='*60}")

        # TODO: implement actual benchmark execution
        # This is the skeleton — fill in with actual RULER/LongBench calls
        result = {"experiment": exp, "status": "TODO"}
        run_log["results"].append(result)

    run_log["end_time"] = datetime.now().isoformat()

    # Save run log
    log_path = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(run_log, f, indent=2)
    print(f"\nRun log saved to {log_path}")


if __name__ == "__main__":
    main()
