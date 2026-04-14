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
import gc
import json
import sys
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from idlekv.core.compression import CompressedKVManager
from idlekv.utils.kv_cache import get_layer_kv


def create_needle_test(tokenizer, context_length=4096, needle_depth=0.5, device="cuda" if torch.cuda.is_available() else "cpu"):
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
    input_ids = torch.tensor([full_tokens[:context_length]], device=device)

    return input_ids, "7392"


def _extract_last_token_kv(past_key_values, num_layers):
    """Return just the last generated KV pair from each layer."""
    return [
        (
            get_layer_kv(past_key_values, layer_idx)[0][:, :, -1:, :],
            get_layer_kv(past_key_values, layer_idx)[1][:, :, -1:, :],
        )
        for layer_idx in range(num_layers)
    ]


def decode_with_manager(
    model,
    tokenizer,
    manager,
    past_key_values,
    max_new_tokens=20,
):
    """
    Decode greedily using the manager's semantic position tracking.

    The first token reuses the logits computed during prefill. Subsequent
    tokens use `manager.next_position_ids()` and update the manager with the
    actual generated-token hidden states/KV pairs.
    """
    generated_tokens = []
    current_past_kv = past_key_values
    next_token_id = None

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if next_token_id is None:
                logits = manager.last_prefill_logits
                outputs = None
            else:
                outputs = model(
                    input_ids=next_token_id.unsqueeze(0),
                    past_key_values=current_past_kv,
                    position_ids=manager.next_position_ids(num_new_tokens=1),
                    use_cache=True,
                    output_hidden_states=True,
                )
                logits = outputs.logits[:, -1, :]
                current_past_kv = outputs.past_key_values

                if getattr(outputs, "hidden_states", None) is not None:
                    manager.on_token_generated(
                        outputs.hidden_states[-1][:, -1, :],
                        _extract_last_token_kv(current_past_kv, manager.num_layers),
                    )
                    current_past_kv = manager.maybe_evict_online(current_past_kv)

            next_token_id = logits.argmax(dim=-1)
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token_id.item())

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def evaluate_needle(model, tokenizer, manager, past_key_values, expected_answer):
    """Decode a short answer continuation and check if it contains the needle."""
    response = decode_with_manager(
        model=model,
        tokenizer=tokenizer,
        manager=manager,
        past_key_values=past_key_values,
        max_new_tokens=20,
    )
    return expected_answer in response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--shadow-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use. Defaults to 'cuda' if available, else 'cpu'")
    args = parser.parse_args()

    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Skip gracefully on CPU
    if device == "cpu":
        print("⚠️  Running on CPU. This script is designed for GPU evaluation.")
        print("   Results may not be representative of GPU performance.")
        print("   Consider running on a CUDA-enabled GPU for accurate benchmarks.")
        print("   For CPU verification: script parses correctly and detects device.")
        print("   Exiting gracefully to avoid model compatibility issues.")
        return 0

    print(f"=== IdleKV Go/No-Go Gate ===")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Compression ratio: {args.ratio}")
    print(f"Context length: {args.context_length}")
    print(f"Trials: {args.num_trials}")
    print()

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Set tokenizer pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            attn_implementation="sdpa",
        )
    else:
        # CPU configuration
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu",
        )
    model.eval()

    # Run trials
    baseline_correct = 0
    idlekv_correct = 0

    for trial in range(args.num_trials):
        depth = (trial + 1) / (args.num_trials + 1)  # vary needle depth
        input_ids, answer = create_needle_test(
            tokenizer, args.context_length, needle_depth=depth, device=device
        )

        # --- Baseline: SnapKV compression, no refinement ---
        manager_baseline = CompressedKVManager(
            model, compression_ratio=args.ratio, shadow_size=0
        )
        baseline_kv = manager_baseline.prefill(input_ids)

        if evaluate_needle(model, tokenizer, manager_baseline, baseline_kv, answer):
            baseline_correct += 1
        del baseline_kv, manager_baseline
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # --- IdleKV: SnapKV + Phase 1 re-scoring ---
        manager_idlekv = CompressedKVManager(
            model, compression_ratio=args.ratio, shadow_size=args.shadow_size
        )
        idlekv_kv = manager_idlekv.prefill(input_ids)
        # Seed query states from the tail of the prompt so Phase 1 can score
        # tokens against the actual retrieval question without mutating the
        # cache via a synthetic warmup decode.
        manager_idlekv.seed_query_buffer_from_prefill()

        # Run Phase 1 refinement
        result = manager_idlekv.idle_refine(idlekv_kv, max_time_ms=100)
        idlekv_kv = result.past_key_values

        if evaluate_needle(model, tokenizer, manager_idlekv, idlekv_kv, answer):
            idlekv_correct += 1

        status = f"Trial {trial+1}/{args.num_trials}: depth={depth:.2f}"
        print(f"  {status} | baseline={'✓' if baseline_correct > trial else '✗'} "
              f"| idlekv={'✓' if idlekv_correct > trial else '✗'}")

        del manager_idlekv, idlekv_kv, result, input_ids
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

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
