#!/usr/bin/env python3
"""
Measure the decode operating point of shortlisted IdleKV settings.

This script is intentionally small and focused. It is not a full benchmark
runner; it answers one question:

  Does `IdleKV r=0.7, 100ms, phase=1` keep roughly the same decode-time
  operating point as the matching compressed no-idle path?

We report:
  - setup wall time (prefill + optional idle refinement)
  - decode tokens/sec for a fixed-length greedy decode loop
  - mean over a tiny number of trials
"""

import argparse
import gc
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from idlekv.core.compression import CompressedKVManager
from idlekv.eval.ruler import create_niah_test
from idlekv.utils.generation import last_token_logits_kwargs
from idlekv.utils.kv_cache import get_layer_kv


def load_model_and_tokenizer(model_name: str, attn_implementation: str = "sdpa"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map={"": 0},
            attn_implementation=attn_implementation,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map="cpu",
        )
    model.eval()
    return model, tokenizer


def _sync_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _build_trial_input(tokenizer, *, context_length: int, seed: int, device: str):
    rng = random.Random(seed)
    input_ids, _answers = create_niah_test(
        tokenizer=tokenizer,
        context_length=context_length,
        num_needles=3,
        device=device,
        rng=rng,
    )
    return input_ids


def _measure_full_cache(model, input_ids: torch.Tensor, num_measure_tokens: int):
    setup_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            **last_token_logits_kwargs(model),
        )
    setup_sec = time.perf_counter() - setup_start

    current_past = outputs.past_key_values
    logits = outputs.logits[:, -1, :]
    next_token = logits.argmax(dim=-1)

    _sync_if_needed()
    decode_start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(num_measure_tokens):
            outputs = model(
                input_ids=next_token.unsqueeze(0),
                past_key_values=current_past,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            current_past = outputs.past_key_values
            next_token = logits.argmax(dim=-1)
    _sync_if_needed()
    decode_sec = time.perf_counter() - decode_start

    return {
        "setup_sec": setup_sec,
        "decode_sec": decode_sec,
        "decode_tokens_per_sec": num_measure_tokens / decode_sec,
    }


def _measure_manager(
    model,
    input_ids: torch.Tensor,
    *,
    ratio: float,
    shadow_size: int,
    idle_budget_ms: int,
    phases,
    num_measure_tokens: int,
):
    manager = CompressedKVManager(
        model,
        compression_ratio=ratio,
        shadow_size=shadow_size,
        query_buffer_size=32,
        offload_full_kv=True,
    )

    setup_start = time.perf_counter()
    past_key_values = manager.prefill(input_ids)
    manager.seed_query_buffer_from_prefill()
    if idle_budget_ms > 0:
        refinement = manager.idle_refine(
            past_key_values,
            max_time_ms=idle_budget_ms,
            phases=phases,
        )
        past_key_values = refinement.past_key_values
    setup_sec = time.perf_counter() - setup_start

    current_past = past_key_values
    logits = manager.last_prefill_logits
    next_token = logits.argmax(dim=-1)

    _sync_if_needed()
    decode_start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(num_measure_tokens):
            outputs = model(
                input_ids=next_token.unsqueeze(0),
                past_key_values=current_past,
                position_ids=manager.next_position_ids(num_new_tokens=1),
                use_cache=True,
                output_hidden_states=True,
            )
            logits = outputs.logits[:, -1, :]
            current_past = outputs.past_key_values

            query_state = manager.build_query_state(outputs.hidden_states)
            new_kv_per_layer = [
                (
                    get_layer_kv(current_past, layer_idx)[0][:, :, -1:, :],
                    get_layer_kv(current_past, layer_idx)[1][:, :, -1:, :],
                )
                for layer_idx in range(manager.num_layers)
            ]
            manager.on_token_generated(query_state, new_kv_per_layer)
            current_past = manager.maybe_evict_online(current_past)

            next_token = logits.argmax(dim=-1)
    _sync_if_needed()
    decode_sec = time.perf_counter() - decode_start

    return {
        "setup_sec": setup_sec,
        "decode_sec": decode_sec,
        "decode_tokens_per_sec": num_measure_tokens / decode_sec,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF model name",
    )
    parser.add_argument("--ratio", type=float, default=0.7)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--num-measure-tokens", type=int, default=128)
    parser.add_argument("--output", type=str, default="results/throughput_spotcheck.json")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)
    device = next(model.parameters()).device

    results = {
        "model": args.model,
        "ratio": args.ratio,
        "context_length": args.context_length,
        "num_trials": args.num_trials,
        "num_measure_tokens": args.num_measure_tokens,
        "trials": [],
    }

    for trial_idx in range(args.num_trials):
        seed = 1000 + trial_idx
        input_ids = _build_trial_input(
            tokenizer,
            context_length=args.context_length,
            seed=seed,
            device=str(device),
        )
        trial_result = {
            "trial_idx": trial_idx,
            "seed": seed,
            "full_cache": _measure_full_cache(
                model,
                input_ids,
                num_measure_tokens=args.num_measure_tokens,
            ),
            "compressed_no_idle": _measure_manager(
                model,
                input_ids,
                ratio=args.ratio,
                shadow_size=256,
                idle_budget_ms=0,
                phases=1,
                num_measure_tokens=args.num_measure_tokens,
            ),
            "phase1_100ms": _measure_manager(
                model,
                input_ids,
                ratio=args.ratio,
                shadow_size=256,
                idle_budget_ms=100,
                phases=1,
                num_measure_tokens=args.num_measure_tokens,
            ),
        }
        results["trials"].append(trial_result)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {}
    for key in ("full_cache", "compressed_no_idle", "phase1_100ms"):
        summary[key] = {
            metric: statistics.mean(trial[key][metric] for trial in results["trials"])
            for metric in ("setup_sec", "decode_sec", "decode_tokens_per_sec")
        }
    results["summary"] = summary

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
