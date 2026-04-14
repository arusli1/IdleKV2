#!/usr/bin/env python3
"""
Day 3 go/no-go decision gate.

This is a delayed-query stress test for Phase 1. It intentionally compresses
the context before the retrieval query arrives, then appends a longer query
suffix that can trigger online evictions before the 100ms idle window.

Why this setup:
  1. It creates a real query shift after compression, which is the premise of
     Token Importance Recurrence (TIR).
  2. It exercises the shadow-buffer recovery path that IdleKV Phase 1 is meant
     to improve.
  3. It avoids the old ceiling effect where the final question was already in
     the prefill prompt, making the task too easy at r=0.5.

By default the gate uses r=0.7 (30% retention), because Llama-3.1-8B on this
stress test is typically at ceiling for r=0.5 and therefore not diagnostic.
Main experiments should still sweep the planned ratios separately.

Usage:
    python scripts/go_no_go.py --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import gc
import json
import random
import sys
import torch
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from idlekv.core.compression import CompressedKVManager
from idlekv.utils.kv_cache import get_layer_kv


@dataclass
class DelayedQueryTrial:
    context_ids: torch.Tensor
    suffix_ids: torch.Tensor
    full_prompt_ids: torch.Tensor
    answer: str
    target_name: str
    suffix_len: int


def create_delayed_query_trial(
    tokenizer,
    trial_idx: int,
    context_length: int = 4096,
    num_records: int = 24,
    target_depth: float = 0.45,
    suffix_repeats: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> DelayedQueryTrial:
    """
    Create a delayed-query retrieval trial.

    The context contains many project→code records. The target project is
    present in the context, but the disambiguating query arrives only after
    compression. The repeated suffix deliberately grows the cache and triggers
    online evictions before the idle window.
    """
    rng = random.Random(9000 + trial_idx)
    base_names = [
        "Aster", "Beacon", "Cinder", "Drift", "Ember", "Falcon", "Grove",
        "Harbor", "Ion", "Juniper", "Kepler", "Lumen", "Mistral", "Nova",
        "Onyx", "Prairie", "Quartz", "Raven", "Solace", "Tundra", "Umber",
        "Vector", "Willow", "Xenon", "Yarrow", "Zephyr", "Atlas", "Birch",
        "Comet", "Delta",
    ]
    names = base_names[:]
    rng.shuffle(names)
    names = names[:num_records]

    target_name = names[0]
    target_code = str(rng.randint(1000, 9999))
    distractors = [(name, str(rng.randint(1000, 9999))) for name in names[1:]]

    intro = (
        "You are reading a project ledger. Additional instructions will arrive "
        "later. Remember the exact 4-digit vault codes for every project.\n\n"
    )
    filler_unit = (
        "Audit note: routine maintenance entry with no vault code relevance. "
        "These notes exist only to create a long retrieval context. "
    )
    intro_tokens = tokenizer.encode(intro, add_special_tokens=False)
    filler_tokens = tokenizer.encode(filler_unit, add_special_tokens=False)

    other_depths = []
    for idx in range(num_records - 1):
        frac = (idx + 1) / num_records
        # Keep the target isolated so the query shift has to pick the right
        # record instead of benefiting from adjacent duplicates.
        if abs(frac - target_depth) < 0.05:
            frac = min(0.97, frac + 0.08)
        other_depths.append(frac)

    target_record = f"Project {target_name}: vault code {target_code}. "
    records = [(
        target_depth,
        tokenizer.encode(target_record, add_special_tokens=False),
    )]
    for (name, code), depth in zip(distractors, other_depths):
        record = f"Project {name}: vault code {code}. "
        records.append((depth, tokenizer.encode(record, add_special_tokens=False)))
    records.sort(key=lambda item: item[0])

    record_budget = sum(len(tokens) for _, tokens in records)
    filler_budget = max(
        context_length - len(intro_tokens) - record_budget - 32,
        len(filler_tokens) * 2,
    )
    repeats = filler_budget // len(filler_tokens) + 2
    all_filler = (filler_tokens * repeats)[:filler_budget]

    context_tokens = list(intro_tokens)
    prev = 0
    for depth, tokens in records:
        pos = int(depth * filler_budget)
        pos = max(prev, min(pos, filler_budget))
        context_tokens.extend(all_filler[prev:pos])
        context_tokens.extend(tokens)
        prev = pos
    context_tokens.extend(all_filler[prev:])
    context_tokens = context_tokens[:context_length]

    suffix = (
        f"We are now validating Project {target_name}. Ignore every other "
        f"project. The only relevant project is {target_name}. We need the "
        f"exact vault code for {target_name}. Before answering, verify the "
        f"{target_name} record carefully. Respond with digits only for Project "
        f"{target_name}. "
    ) * suffix_repeats + "Answer:"
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    return DelayedQueryTrial(
        context_ids=torch.tensor([context_tokens], device=device),
        suffix_ids=torch.tensor([suffix_tokens], device=device),
        full_prompt_ids=torch.tensor([context_tokens + suffix_tokens], device=device),
        answer=target_code,
        target_name=target_name,
        suffix_len=len(suffix_tokens),
    )


def _extract_suffix_kv(past_key_values, layer_idx: int, start: int, end: int):
    """Return the KV slice for a single suffix token in a given layer."""
    k, v = get_layer_kv(past_key_values, layer_idx)
    return k[:, :, start:end, :], v[:, :, start:end, :]


def ingest_suffix(model, manager, past_key_values, suffix_ids: torch.Tensor):
    """
    Teacher-force the delayed query suffix after compression.

    We track each suffix token as newly appended KV/query state so the manager's
    online-eviction and Phase 1 paths see the same cache semantics as normal
    decode-time ingestion.
    """
    num_suffix_tokens = suffix_ids.shape[1]
    outputs = model(
        input_ids=suffix_ids,
        past_key_values=past_key_values,
        position_ids=manager.next_position_ids(num_new_tokens=num_suffix_tokens),
        use_cache=True,
        output_hidden_states=True,
    )
    current_past_kv = outputs.past_key_values
    total_seq = get_layer_kv(current_past_kv, 0)[0].shape[2]

    for token_idx in range(num_suffix_tokens):
        query_state = torch.stack([
            outputs.hidden_states[layer_idx][0, token_idx, :].detach()
            for layer_idx in range(manager.num_layers)
        ], dim=0)
        start = total_seq - num_suffix_tokens + token_idx
        end = start + 1
        new_kv_per_layer = [
            _extract_suffix_kv(current_past_kv, layer_idx, start, end)
            for layer_idx in range(manager.num_layers)
        ]
        manager.on_token_generated(query_state, new_kv_per_layer)

    current_past_kv = manager.maybe_evict_online(current_past_kv)
    return current_past_kv, outputs.logits[:, -1, :]


def decode_with_manager(
    model,
    tokenizer,
    manager,
    past_key_values,
    initial_logits: torch.Tensor,
    max_new_tokens: int = 8,
) -> str:
    """Greedy decode using the manager's semantic position tracking."""
    generated_tokens = []
    current_past_kv = past_key_values
    next_token_id = None
    logits = initial_logits

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if next_token_id is not None:
                outputs = model(
                    input_ids=next_token_id.unsqueeze(0),
                    past_key_values=current_past_kv,
                    position_ids=manager.next_position_ids(num_new_tokens=1),
                    use_cache=True,
                    output_hidden_states=True,
                )
                logits = outputs.logits[:, -1, :]
                current_past_kv = outputs.past_key_values

                query_state = manager.build_query_state(outputs.hidden_states)
                new_kv_per_layer = [
                    (
                        get_layer_kv(current_past_kv, layer_idx)[0][:, :, -1:, :],
                        get_layer_kv(current_past_kv, layer_idx)[1][:, :, -1:, :],
                    )
                    for layer_idx in range(manager.num_layers)
                ]
                manager.on_token_generated(query_state, new_kv_per_layer)
                current_past_kv = manager.maybe_evict_online(current_past_kv)

            next_token_id = logits.argmax(dim=-1)
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token_id.item())

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def evaluate_with_compression(
    model,
    tokenizer,
    trial: DelayedQueryTrial,
    compression_ratio: float,
    shadow_size: int,
    run_phase1: bool,
):
    manager = CompressedKVManager(
        model,
        compression_ratio=compression_ratio,
        shadow_size=shadow_size if run_phase1 else 0,
    )
    past_key_values = manager.prefill(trial.context_ids)
    past_key_values, initial_logits = ingest_suffix(
        model=model,
        manager=manager,
        past_key_values=past_key_values,
        suffix_ids=trial.suffix_ids,
    )

    refinement = None
    if run_phase1:
        refinement = manager.idle_refine(past_key_values, max_time_ms=100)
        if refinement.phase2_ran:
            raise RuntimeError(
                "The 100ms go/no-go gate must remain Phase 1 only. "
                "Phase 2 unexpectedly ran."
            )
        past_key_values = refinement.past_key_values

    response = decode_with_manager(
        model=model,
        tokenizer=tokenizer,
        manager=manager,
        past_key_values=past_key_values,
        initial_logits=initial_logits,
        max_new_tokens=8,
    )
    return {
        "correct": trial.answer in response,
        "response": response,
        "phase1_time_ms": refinement.phase1_time_ms if refinement else 0.0,
        "shadow_count": manager.shadow_buffer.layers[0].count,
    }


def evaluate_full_cache(model, tokenizer, trial: DelayedQueryTrial):
    """Reference run without compression to confirm the task is solvable."""
    with torch.no_grad():
        outputs = model(
            input_ids=trial.full_prompt_ids,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        generated_tokens = []
        next_token_id = None
        for _ in range(8):
            if next_token_id is not None:
                outputs = model(
                    input_ids=next_token_id.unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]

            next_token_id = logits.argmax(dim=-1)
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            generated_tokens.append(next_token_id.item())

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return {
        "correct": trial.answer in response,
        "response": response,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.7,
        help="Compression ratio. Defaults to 0.7 because this pilot is a stress test.",
    )
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--num-trials", type=int, default=12)
    parser.add_argument("--num-records", type=int, default=24)
    parser.add_argument("--target-depth", type=float, default=0.45)
    parser.add_argument("--suffix-repeats", type=int, default=8)
    parser.add_argument("--shadow-size", type=int, default=256)
    parser.add_argument(
        "--skip-full-cache-ref",
        action="store_true",
        help="Skip the uncompressed reference run to save time.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use. Defaults to 'cuda' if available, else 'cpu'.",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("⚠️  Running on CPU. This stress test is designed for GPU validation.")
        print("   Exiting gracefully.")
        return 0

    print("=== IdleKV Go/No-Go Gate ===")
    print("Benchmark: delayed-query stress test")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Compression ratio: {args.ratio}")
    print(f"Context length: {args.context_length}")
    print(f"Trials: {args.num_trials}")
    print(f"Records per trial: {args.num_records}")
    print(f"Target depth: {args.target_depth:.2f}")
    print(f"Suffix repeats: {args.suffix_repeats}")
    print(f"Shadow size: {args.shadow_size}")
    if args.ratio <= 0.5:
        print("Note: this synthetic gate often saturates at r<=0.5 on Llama-8B.")
        print("      Use the default r=0.7 setting when you need discriminative headroom.")
    print()

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else "cpu",
        attn_implementation="sdpa" if device == "cuda" else None,
    )
    model.eval()

    print("Warming up delayed-query path...")
    warmup_trial = create_delayed_query_trial(
        tokenizer=tokenizer,
        trial_idx=-1,
        context_length=args.context_length,
        num_records=args.num_records,
        target_depth=args.target_depth,
        suffix_repeats=args.suffix_repeats,
        device=device,
    )
    _ = evaluate_with_compression(
        model=model,
        tokenizer=tokenizer,
        trial=warmup_trial,
        compression_ratio=args.ratio,
        shadow_size=args.shadow_size,
        run_phase1=True,
    )
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print("Warmup complete.\n")

    full_cache_correct = 0
    baseline_correct = 0
    idlekv_correct = 0
    phase1_times = []

    for trial_idx in range(args.num_trials):
        trial = create_delayed_query_trial(
            tokenizer=tokenizer,
            trial_idx=trial_idx,
            context_length=args.context_length,
            num_records=args.num_records,
            target_depth=args.target_depth,
            suffix_repeats=args.suffix_repeats,
            device=device,
        )

        if not args.skip_full_cache_ref:
            full_ref = evaluate_full_cache(model, tokenizer, trial)
            full_cache_correct += int(full_ref["correct"])
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        baseline = evaluate_with_compression(
            model=model,
            tokenizer=tokenizer,
            trial=trial,
            compression_ratio=args.ratio,
            shadow_size=0,
            run_phase1=False,
        )
        baseline_correct += int(baseline["correct"])
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        idlekv = evaluate_with_compression(
            model=model,
            tokenizer=tokenizer,
            trial=trial,
            compression_ratio=args.ratio,
            shadow_size=args.shadow_size,
            run_phase1=True,
        )
        idlekv_correct += int(idlekv["correct"])
        phase1_times.append(idlekv["phase1_time_ms"])

        full_symbol = "✓" if args.skip_full_cache_ref or full_ref["correct"] else "✗"
        print(
            f"  Trial {trial_idx + 1}/{args.num_trials}: "
            f"target={trial.target_name:<8} "
            f"| full={full_symbol} "
            f"| baseline={'✓' if baseline['correct'] else '✗'} "
            f"| idlekv={'✓' if idlekv['correct'] else '✗'} "
            f"| p1={idlekv['phase1_time_ms']:.1f}ms"
        )

        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    full_cache_acc = (
        full_cache_correct / args.num_trials * 100
        if not args.skip_full_cache_ref else None
    )
    baseline_acc = baseline_correct / args.num_trials * 100
    idlekv_acc = idlekv_correct / args.num_trials * 100
    delta = idlekv_acc - baseline_acc
    mean_phase1_ms = sum(phase1_times) / len(phase1_times) if phase1_times else 0.0

    print()
    print("=== RESULTS ===")
    if full_cache_acc is not None:
        print(f"Full cache reference:              {full_cache_acc:.1f}%")
    print(f"Compressed baseline (r={args.ratio}):     {baseline_acc:.1f}%")
    print(f"IdleKV + Phase 1:                 {idlekv_acc:.1f}%")
    print(f"Delta:                            {delta:+.1f}%")
    print(f"Mean Phase 1 time:                {mean_phase1_ms:.1f}ms")
    print()

    if delta >= 3.0:
        print("✅ GO: Phase 1 shows a meaningful recovery gain on the delayed-query gate.")
    elif delta >= 1.0:
        print("⚠️  MARGINAL: Phase 1 helps, but the gain is modest on this gate.")
    else:
        print("❌ NO-GO: Phase 1 fails to recover enough accuracy on the delayed-query gate.")

    results = {
        "benchmark": "delayed_query_stress",
        "model": args.model,
        "ratio": args.ratio,
        "context_length": args.context_length,
        "num_trials": args.num_trials,
        "num_records": args.num_records,
        "target_depth": args.target_depth,
        "suffix_repeats": args.suffix_repeats,
        "shadow_size": args.shadow_size,
        "full_cache_acc": full_cache_acc,
        "baseline_acc": baseline_acc,
        "idlekv_acc": idlekv_acc,
        "delta": delta,
        "mean_phase1_ms": mean_phase1_ms,
        "decision": "GO" if delta >= 3 else "MARGINAL" if delta >= 1 else "NO-GO",
    }
    outpath = Path("results/go_no_go.json")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
