#!/usr/bin/env python3
"""
Run the real IdleKV experiment suite.

This script executes actual benchmark evaluations and saves one JSON file per
experiment so long sweeps can be resumed safely after interruption.
"""

import argparse
import gc
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from idlekv.baselines.kvpress_baselines import get_press
from idlekv.core.compression import CompressedKVManager
from idlekv.core.scheduler import IdleScheduler
from idlekv.eval.longbench import evaluate_longbench
from idlekv.eval.ruler import evaluate_ruler_niah


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_model_and_tokenizer(model_cfg: dict, attn_implementation: str = "sdpa"):
    """Load a model directly onto GPU when available to avoid offload churn."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading model: {model_cfg['name']} (attn={attn_implementation})")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["name"],
            dtype=torch.float16,
            device_map={"": 0},
            attn_implementation=attn_implementation,
        )
        device = "cuda"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["name"],
            dtype=torch.float32,
            device_map="cpu",
        )
        device = "cpu"

    model.eval()
    return model, tokenizer, device


def aggregate_ruler(results) -> dict:
    subtask_scores = {
        result.subtask: {
            "accuracy": result.accuracy,
            "samples": result.num_samples,
        }
        for result in results
    }
    avg_accuracy = (
        sum(result.accuracy for result in results) / len(results)
        if results else 0.0
    )
    return {
        **subtask_scores,
        "avg_accuracy": avg_accuracy,
        "num_subtasks": len(results),
    }


def aggregate_longbench(results) -> dict:
    subtask_scores = {
        result.subtask: {
            "score": result.score,
            "samples": result.num_samples,
        }
        for result in results
    }
    avg_score = (
        sum(result.score for result in results) / len(results)
        if results else 0.0
    )
    return {
        **subtask_scores,
        "avg_score": avg_score,
        "num_subtasks": len(results),
    }


def phases_label(phases) -> str:
    return IdleScheduler.normalize_phases(phases)[2]


def policy_label(policy) -> str:
    return "legacy" if policy is None else str(policy).strip().lower()


def benchmark_list(config: dict, args, allowed: list[str] | None = None) -> list[str]:
    if args.benchmarks:
        requested = [name.strip() for name in args.benchmarks.split(",") if name.strip()]
        if allowed is None:
            return requested
        return [name for name in requested if name in allowed]
    return list(allowed) if allowed is not None else list(config["evaluation"]["benchmarks"])


def benchmarks_for_experiment(exp: dict, config: dict, args) -> list[str]:
    default = list(config["evaluation"]["benchmarks"])
    by_type = config["evaluation"].get("benchmarks_by_type", {})
    allowed = list(by_type.get(exp["type"], default))

    if exp["type"] == "baseline":
        baseline = exp["baseline"]
        if baseline.get("benchmarks"):
            allowed = list(baseline["benchmarks"])
        excluded = set(baseline.get("exclude_benchmarks", []))
        if excluded:
            allowed = [name for name in allowed if name not in excluded]

    selected = benchmark_list(config, args, allowed=allowed)
    if not selected:
        raise ValueError(
            f"No benchmarks selected for experiment type {exp['type']!r} "
            f"after applying config/CLI filters."
        )
    return selected


def benchmark_num_samples(args) -> int:
    return args.num_samples if args.num_samples is not None else 50


def ruler_context_lengths(config: dict, args) -> list[int]:
    if args.ruler_context_lengths:
        return [int(value.strip()) for value in args.ruler_context_lengths.split(",") if value.strip()]
    return list(config["evaluation"]["ruler"]["context_lengths"])


def longbench_max_input_length(config: dict, args) -> int:
    if args.longbench_max_input_length is not None:
        return args.longbench_max_input_length
    return int(config["evaluation"].get("longbench", {}).get("max_input_length", 4096))


def build_experiment_matrix(config: dict, args) -> list[dict]:
    models = config["models"]
    if args.model:
        models = [model for model in models if model["short"] == args.model]
    if not models:
        raise ValueError(f"No models matched --model {args.model!r}")

    seeds = [args.seed] if args.seed is not None else config["evaluation"]["seeds"]
    experiments = []

    for model_cfg in models:
        for seed in seeds:
            if not args.only_idlekv and not args.only_ablations:
                for baseline in config["baselines"]:
                    experiments.append({
                        "type": "baseline",
                        "model": model_cfg,
                        "seed": seed,
                        "baseline": baseline,
                    })

            if not args.only_baselines and not args.only_ablations:
                ratio = args.ratio or config["compression"]["primary_ratio"]
                idlekv_cfg = config["idlekv"]
                conditions = idlekv_cfg.get("conditions")
                if conditions:
                    condition_iter = [
                        (
                            condition["idle_budget_ms"],
                            condition.get("phases", 1),
                            condition.get("policy", idlekv_cfg.get("refinement_policy")),
                        )
                        for condition in conditions
                    ]
                else:
                    condition_iter = [
                        (budget, phases, idlekv_cfg.get("refinement_policy"))
                        for budget in idlekv_cfg["idle_budgets_ms"]
                        for phases in idlekv_cfg["phases"]
                    ]
                for budget, phases, policy in condition_iter:
                        experiments.append({
                            "type": "idlekv",
                            "model": model_cfg,
                            "seed": seed,
                            "ratio": ratio,
                            "idle_budget_ms": budget,
                            "phases": phases,
                            "policy": policy,
                        })

            if not args.only_baselines and not args.only_idlekv:
                for shadow_size in config["ablations"]["shadow_buffer_sizes"]:
                    experiments.append({
                        "type": "ablation_buffer",
                        "model": model_cfg,
                        "seed": seed,
                        "ratio": args.ratio or config["compression"]["primary_ratio"],
                        "shadow_buffer_size": shadow_size,
                    })

    type_order = {"baseline": 0, "idlekv": 1, "ablation_buffer": 2}
    experiments.sort(
        key=lambda exp: (
            exp["model"]["short"],
            exp["seed"],
            type_order[exp["type"]],
        )
    )
    if args.max_experiments is not None:
        experiments = experiments[:args.max_experiments]
    return experiments


def make_manager(model, config: dict, ratio: float, shadow_size: int) -> CompressedKVManager:
    compression_cfg = config["compression"]
    idlekv_cfg = config.get("idlekv", {})
    return CompressedKVManager(
        model,
        compression_ratio=ratio,
        shadow_size=shadow_size,
        query_buffer_size=compression_cfg["query_buffer_size"],
        offload_full_kv=compression_cfg["offload_full_kv"],
        default_refinement_policy=idlekv_cfg.get("refinement_policy"),
        anytime_min_idle_ms=idlekv_cfg.get("min_idle_ms", 20.0),
        anytime_shadow_only_max_ms=idlekv_cfg.get("shadow_only_max_ms", 80.0),
        sample_span_size=idlekv_cfg.get("sample_span_size", 16),
        sample_spans_per_layer=idlekv_cfg.get("sample_spans_per_layer", 2),
        sample_sampler_seed=idlekv_cfg.get("sample_sampler_seed", 0),
    )


def serialize_experiment(exp: dict) -> dict:
    serialized = {}
    for key, value in exp.items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized


def experiment_slug(exp: dict) -> str:
    model_short = exp["model"]["short"]
    seed = exp["seed"]

    if exp["type"] == "baseline":
        baseline = exp["baseline"]
        name = baseline["name"]
        return f"{name}_{model_short}_seed{seed}"

    if exp["type"] == "idlekv":
        phases = phases_label(exp["phases"])
        slug = (
            f"idlekv_r{exp['ratio']}_budget{exp['idle_budget_ms']}"
            f"_phases{phases.replace('+', '')}_{model_short}_seed{seed}"
        )
        if exp.get("policy") is not None:
            slug += f"_{policy_label(exp['policy'])}"
        return slug

    if exp["type"] == "ablation_buffer":
        return (
            f"buffer{exp['shadow_buffer_size']}_r{exp['ratio']}"
            f"_{model_short}_seed{seed}"
        )

    raise ValueError(f"Unknown experiment type: {exp['type']}")


def result_path(output_dir: Path, exp: dict) -> Path:
    subdir_map = {
        "baseline": "baselines",
        "idlekv": "idlekv",
        "ablation_buffer": "ablations",
    }
    path = output_dir / subdir_map[exp["type"]] / f"{experiment_slug(exp)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def release_model(model, tokenizer):
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return None, None


def required_attn_implementation(exp: dict) -> str:
    """
    Return the attention implementation required by an experiment.

    kvpress' ObservedAttentionPress relies on eager attention hooks, while the
    rest of the current stack is happiest on SDPA.
    """
    if exp["type"] == "baseline":
        baseline = exp["baseline"]
        if baseline.get("method") == "h2o":
            return "eager"
    return "sdpa"


def run_benchmarks(
    model,
    tokenizer,
    device: str,
    config: dict,
    args,
    seed: int,
    benchmarks: list[str] | None = None,
    *,
    manager=None,
    press=None,
    idle_budget_ms: float = 0.0,
    phases="1+2",
    policy=None,
    sync_refresh_stride=None,
) -> dict:
    benchmark_results = {}
    num_samples = benchmark_num_samples(args)
    selected_benchmarks = benchmarks or benchmark_list(config, args)

    for benchmark in selected_benchmarks:
        if benchmark == "ruler":
            benchmark_results["ruler"] = {}
            for context_length in ruler_context_lengths(config, args):
                results = evaluate_ruler_niah(
                    model=model,
                    tokenizer=tokenizer,
                    context_length=context_length,
                    num_samples=num_samples,
                    device=device,
                    manager=manager,
                    press=press,
                    idle_budget_ms=idle_budget_ms,
                    phases=phases,
                    policy=policy,
                    sync_refresh_stride=sync_refresh_stride,
                    seed=seed,
                )
                benchmark_results["ruler"][str(context_length)] = aggregate_ruler(results)
        elif benchmark == "longbench":
            results = evaluate_longbench(
                model=model,
                tokenizer=tokenizer,
                num_samples=num_samples,
                device=device,
                manager=manager,
                press=press,
                idle_budget_ms=idle_budget_ms,
                phases=phases,
                policy=policy,
                sync_refresh_stride=sync_refresh_stride,
                max_input_length=longbench_max_input_length(config, args),
            )
            benchmark_results["longbench"] = aggregate_longbench(results)
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark}")

    return benchmark_results


def execute_experiment(exp: dict, config: dict, args, model, tokenizer, device: str) -> dict:
    random.seed(exp["seed"])
    torch.manual_seed(exp["seed"])
    selected_benchmarks = benchmarks_for_experiment(exp, config, args)

    started = time.perf_counter()
    result = {
        "status": "completed",
        "experiment": serialize_experiment(exp),
    }

    if exp["type"] == "baseline":
        baseline = exp["baseline"]
        name = baseline["name"]
        ratio = baseline.get("ratio")
        result.update({
            "method": name,
            "ratio": ratio,
            "seed": exp["seed"],
        })

        if name == "full_cache":
            benchmark_results = run_benchmarks(
                model=model,
                tokenizer=tokenizer,
                device=device,
                config=config,
                args=args,
                seed=exp["seed"],
                benchmarks=selected_benchmarks,
            )
        elif name == "sync_refresh":
            ratio = ratio if ratio is not None else config["compression"]["primary_ratio"]
            manager = make_manager(model, config, ratio=ratio, shadow_size=0)
            benchmark_results = run_benchmarks(
                model=model,
                tokenizer=tokenizer,
                device=device,
                config=config,
                args=args,
                seed=exp["seed"],
                benchmarks=selected_benchmarks,
                manager=manager,
                idle_budget_ms=0.0,
                phases="2",
                policy=None,
                sync_refresh_stride=baseline.get("refresh_stride", 15),
            )
        else:
            compression_ratio = ratio if ratio is not None else config["compression"]["primary_ratio"]
            press = get_press(
                baseline["method"],
                compression_ratio=compression_ratio,
            )
            benchmark_results = run_benchmarks(
                model=model,
                tokenizer=tokenizer,
                device=device,
                config=config,
                args=args,
                seed=exp["seed"],
                benchmarks=selected_benchmarks,
                press=press,
            )

        result["benchmark_results"] = benchmark_results

    elif exp["type"] == "idlekv":
        manager = make_manager(
            model,
            config,
            ratio=exp["ratio"],
            shadow_size=config["compression"]["shadow_buffer_size"],
        )
        result.update({
            "method": "idlekv",
            "ratio": exp["ratio"],
            "idle_budget_ms": exp["idle_budget_ms"],
            "phases": phases_label(exp["phases"]),
            "policy": exp.get("policy"),
            "seed": exp["seed"],
        })
        result["benchmark_results"] = run_benchmarks(
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=config,
            args=args,
            seed=exp["seed"],
            benchmarks=selected_benchmarks,
            manager=manager,
            idle_budget_ms=exp["idle_budget_ms"],
            phases=exp["phases"],
            policy=exp.get("policy"),
        )

    elif exp["type"] == "ablation_buffer":
        manager = make_manager(
            model,
            config,
            ratio=exp["ratio"],
            shadow_size=exp["shadow_buffer_size"],
        )
        result.update({
            "method": "idlekv_buffer_ablation",
            "ratio": exp["ratio"],
            "shadow_buffer_size": exp["shadow_buffer_size"],
            "idle_budget_ms": 100,
            "phases": "1",
            "seed": exp["seed"],
        })
        result["benchmark_results"] = run_benchmarks(
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=config,
            args=args,
            seed=exp["seed"],
            benchmarks=selected_benchmarks,
            manager=manager,
            idle_budget_ms=100,
            phases=1,
        )

    else:
        raise ValueError(f"Unknown experiment type: {exp['type']}")

    result["wall_time_sec"] = time.perf_counter() - started
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main.yaml")
    parser.add_argument("--only-baselines", action="store_true")
    parser.add_argument("--only-idlekv", action="store_true")
    parser.add_argument("--only-ablations", action="store_true")
    parser.add_argument("--model", type=str, default=None, help="Run only this model short name")
    parser.add_argument("--ratio", type=float, default=None, help="Override compression ratio")
    parser.add_argument("--seed", type=int, default=None, help="Run only this seed")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--num-samples", type=int, default=None, help="Override samples per benchmark subtask")
    parser.add_argument("--benchmarks", type=str, default=None, help="Comma-separated subset, e.g. ruler,longbench")
    parser.add_argument("--ruler-context-lengths", type=str, default=None, help="Comma-separated RULER context lengths, e.g. 4096,8192")
    parser.add_argument("--longbench-max-input-length", type=int, default=None, help="Override LongBench prompt budget, e.g. 4096")
    parser.add_argument("--max-experiments", type=int, default=None, help="Cap the experiment count for smoke tests")
    parser.add_argument("--skip-existing", action="store_true", help="Skip experiments whose result JSON already exists")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print("Loaded config:")
        print(json.dumps(config, indent=2))

    experiments = build_experiment_matrix(config, args)
    print(f"Total experiments: {len(experiments)}")
    if args.dry_run:
        for index, exp in enumerate(experiments, start=1):
            benchmarks = benchmarks_for_experiment(exp, config, args)
            print(
                f"  [{index}] {exp['type']}: model={exp['model']['short']}, "
                f"seed={exp['seed']}, benchmarks={benchmarks}, "
                f"out={result_path(output_dir, exp)}"
            )
        return

    run_log = {
        "config_path": str(args.config),
        "output_dir": str(output_dir),
        "start_time": datetime.now().isoformat(),
        "results": [],
    }
    run_log_path = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    current_model_key = None
    model = None
    tokenizer = None
    device = "cpu"

    try:
        for index, exp in enumerate(experiments, start=1):
            out_path = result_path(output_dir, exp)
            if args.skip_existing and out_path.exists():
                print(f"[{index}/{len(experiments)}] Skipping existing {out_path.name}")
                run_log["results"].append({
                    "experiment": serialize_experiment(exp),
                    "status": "skipped_existing",
                    "result_path": str(out_path),
                })
                continue

            model_key = (
                exp["model"]["short"],
                required_attn_implementation(exp),
            )
            if current_model_key != model_key:
                model, tokenizer = release_model(model, tokenizer)
                model, tokenizer, device = load_model_and_tokenizer(
                    exp["model"],
                    attn_implementation=model_key[1],
                )
                current_model_key = model_key

            print(f"\n{'=' * 72}")
            print(f"Experiment {index}/{len(experiments)}")
            print(f"  Type: {exp['type']}")
            print(f"  Model: {exp['model']['short']}")
            print(f"  Seed: {exp['seed']}")
            print(f"  Output: {out_path}")
            print(f"{'=' * 72}")
            if args.verbose:
                print(json.dumps(exp, indent=2))

            try:
                result = execute_experiment(exp, config, args, model, tokenizer, device)
                with open(out_path, "w", encoding="utf-8") as handle:
                    json.dump(result, handle, indent=2)
                run_log["results"].append({
                    "experiment": serialize_experiment(exp),
                    "status": "completed",
                    "result_path": str(out_path),
                })
            except Exception as exc:
                failure = {
                    "experiment": serialize_experiment(exp),
                    "status": "failed",
                    "error": str(exc),
                    "result_path": str(out_path),
                }
                run_log["results"].append(failure)
                print(f"  ERROR: {exc}")

            with open(run_log_path, "w", encoding="utf-8") as handle:
                json.dump(run_log, handle, indent=2)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        model, tokenizer = release_model(model, tokenizer)

    run_log["end_time"] = datetime.now().isoformat()
    with open(run_log_path, "w", encoding="utf-8") as handle:
        json.dump(run_log, handle, indent=2)
    print(f"\nRun log saved to {run_log_path}")


if __name__ == "__main__":
    main()
