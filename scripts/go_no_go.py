#!/usr/bin/env python3
"""
Day 3 go/no-go decision gate.

Runs RULER needle-in-a-haystack at r=0.5 on a single model, comparing:
  1. SnapKV r=0.5 (baseline)
  2. SnapKV r=0.5 + Phase 1 re-scoring (100ms idle)

Decision: Phase 1 must gain >= 3% absolute on needle retrieval.
If < 1%: diagnose or pivot framing.

Usage:
    python scripts/go_no_go.py --model meta-llama/Llama-3.1-8B-Instruct --ratio 0.5
"""

import argparse
import json
import sys
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from idlekv.core.compression import CompressedKVManager
from idlekv.eval.metrics import compute_kl_divergence, measure_throughput


def create_needle_test(tokenizer, context_length=4096, needle_depth=0.5):
    """
    Create a simple needle-in-a-haystack test.

    Embeds a fact ("The secret number is 7392") at needle_depth fraction
    through a context of filler text, then asks for it.
    """
    needle = "The secret number is 7392."
    question = "\n\nWhat is the secret number mentioned in the text above?"

    # Generate filler (repeated sentence)
    filler_unit = "This is a passage of text that serves as filler content for testing long-context retrieval. "
    filler_tokens = tokenizer.encode(filler_unit, add_special_tokens=False)

    # Target tokens for context (minus needle and question)
    needle_tokens = tokenizer.encode(needle, add_special_tokens=False)
    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    budget = context_length - len(needle_tokens) - len(question_tokens) - 10

    # Build filler
    repeats = budget // len(filler_tokens) + 1
    all_filler = (filler_tokens * repeats)[:budget]

    # Insert needle at depth
    insert_pos = int(len(all_filler) * needle_depth)
    context_tokens = all_filler[:insert_pos] + needle_tokens + all_filler[insert_pos:]

    # Add question
    full_tokens = context_tokens + question_tokens
    input_ids = torch.tensor([full_tokens[:context_length]], device="cuda")

    return input_ids, "7392"


def evaluate_needle(model, tokenizer, input_ids, past_key_values, expected_answer):
    """Generate a short response and check if it contains the needle."""
    with torch.no_grad():
        output = model.generate(
            input_ids[:, -1:],
            past_key_values=past_key_values,
            max_new_tokens=20,
            do_sample=False,
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return expected_answer in response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--shadow-size", type=int, default=256)
    args = parser.parse_args()

    print(f"=== IdleKV Go/No-Go Gate ===")
    print(f"Model: {args.model}")
    print(f"Compression ratio: {args.ratio}")
    print(f"Context length: {args.context_length}")
    print(f"Trials: {args.num_trials}")
    print()

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()

    # Run trials
    baseline_correct = 0
    idlekv_correct = 0

    for trial in range(args.num_trials):
        depth = (trial + 1) / (args.num_trials + 1)  # vary needle depth
        input_ids, answer = create_needle_test(
            tokenizer, args.context_length, needle_depth=depth
        )

        # --- Baseline: SnapKV compression, no refinement ---
        manager_baseline = CompressedKVManager(
            model, compression_ratio=args.ratio, shadow_size=0
        )
        baseline_kv = manager_baseline.prefill(input_ids)

        if evaluate_needle(model, tokenizer, input_ids, baseline_kv, answer):
            baseline_correct += 1

        # --- IdleKV: SnapKV + Phase 1 re-scoring ---
        manager_idlekv = CompressedKVManager(
            model, compression_ratio=args.ratio, shadow_size=args.shadow_size
        )
        idlekv_kv = manager_idlekv.prefill(input_ids)

        # Simulate: generate a few tokens to populate query buffer
        # (Phase 1 needs recent queries to re-score)
        with torch.no_grad():
            for _ in range(32):
                out = model(input_ids[:, -1:], past_key_values=idlekv_kv, use_cache=True)
                idlekv_kv = out.past_key_values
                if hasattr(out, 'hidden_states') and out.hidden_states:
                    manager_idlekv.on_token_generated(
                        out.hidden_states[-1][:, -1, :], []
                    )

        # Run Phase 1 refinement
        result = manager_idlekv.idle_refine(idlekv_kv, max_time_ms=100)
        idlekv_kv = result.past_key_values

        if evaluate_needle(model, tokenizer, input_ids, idlekv_kv, answer):
            idlekv_correct += 1

        status = f"Trial {trial+1}/{args.num_trials}: depth={depth:.2f}"
        print(f"  {status} | baseline={'✓' if baseline_correct > trial else '✗'} "
              f"| idlekv={'✓' if idlekv_correct > trial else '✗'}")

    # Results
    baseline_acc = baseline_correct / args.num_trials * 100
    idlekv_acc = idlekv_correct / args.num_trials * 100
    delta = idlekv_acc - baseline_acc

    print()
    print(f"=== RESULTS ===")
    print(f"Baseline (SnapKV r={args.ratio}):  {baseline_acc:.1f}%")
    print(f"IdleKV (+ Phase 1):               {idlekv_acc:.1f}%")
    print(f"Delta:                             {delta:+.1f}%")
    print()

    if delta >= 3.0:
        print("✅ GO: Phase 1 gains >= 3%. Proceed with full experiments.")
    elif delta >= 1.0:
        print("⚠️  MARGINAL: Phase 1 gains 1-3%. Consider adjusting shadow buffer "
              "size or switching to Phase 2 as primary contribution.")
    else:
        print("❌ NO-GO: Phase 1 gains < 1%. Investigate:")
        print("   - Is the shadow buffer capturing the right tokens?")
        print("   - Are recent queries informative for re-scoring?")
        print("   - Consider pivoting to iso-quality compression framing.")

    # Save results
    results = {
        "model": args.model,
        "ratio": args.ratio,
        "context_length": args.context_length,
        "num_trials": args.num_trials,
        "baseline_acc": baseline_acc,
        "idlekv_acc": idlekv_acc,
        "delta": delta,
        "decision": "GO" if delta >= 3 else "MARGINAL" if delta >= 1 else "NO-GO",
    }
    outpath = Path("results/go_no_go.json")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
