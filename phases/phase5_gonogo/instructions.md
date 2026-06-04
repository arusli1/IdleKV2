# Phase 5: Oracle Experiment — The Go/No-Go Gate

**Goal:** Determine the theoretical ceiling of KV cache repair by restoring ALL evicted tokens with no budget constraint. The oracle recovery rate establishes whether the degradation observed in Phase 3 is attributable to KV content loss (recoverable) or something else (not recoverable through repair). This is the most important experiment in the paper and the go/no-go gate for Phase 6.

**This phase is the paper's core novel contribution regardless of Phase 6 results.** A table showing what fraction of eviction-induced degradation is theoretically recoverable does not exist in the published literature. Even if Phase 6 delivers only partial repair, the oracle table is independently publishable and justifies why the gap is hard to close.

---

## What the Oracle Actually Does

The oracle answers one question: if repair were computationally free and perfect — if you could restore every single evicted token before answering the query — do you recover condition A (monolithic pass) quality?

```python
# Oracle: restore ALL evicted tokens, no budget limit, no selection
all_evicted_positions = eviction_result.evicted.positions
all_evicted_kv = buffer.to_gpu(
    list(buffer._entries.values()),   # every entry in the buffer
    device="cuda"
)

repaired_cache = inject_kv_with_positions(
    active_cache=eviction_result.compressed,
    active_positions=eviction_result.compressed.positions,
    new_pairs=all_evicted_kv,
    new_positions=all_evicted_positions,
)

# Answer the query with the fully repaired cache
outputs = model(query_tokens, past_key_values=repaired_cache.kv)
```

With 96GB VRAM and ~36GB used at runtime (model weights + compressed cache), you have roughly 60GB free. The full evicted token set for one 32K example is at most 16GB (the difference between the full cache and the compressed cache). Oracle restoration fits in a single GPU without any CPU offload. This is one of the reasons 96GB makes Phase 5 straightforwardly runnable.

---

## Three Conditions to Run

Phase 5 runs the same 100 examples per task as Phase 3, but now with three conditions measured on the same examples:

**Condition A (already have this):** Monolithic pass scores from P0. Reuse directly — no new inference needed.

**Condition B (already have this):** SnapKV-compressed cache scores from Phase 3. Reuse `results/phase3_degradation/` directly.

**Condition C-oracle (new):** Full restoration of all evicted tokens, then query on the repaired cache.

The oracle recovery rate per example is:

```
oracle_recovery = (oracle_score - condition_B_score) / (condition_A_score - condition_B_score)
```

Where scores of 1.0 = correct, 0.0 = incorrect for VT and S-NIAH, and 0.0–1.0 recall for MQ-NIAH.

An aggregate oracle recovery rate near 1.0 means all degradation is from KV content loss and is theoretically recoverable. Near 0.0 means something else is causing degradation and repair cannot help.

---

## Experimental Configuration

Run oracle at these conditions:

| Parameter | Values |
|-----------|--------|
| Context length | 32K |
| Tasks | VT-4hop, MQ-NIAH-4q, S-NIAH |
| Eviction method | SnapKV (primary), StreamingLLM (secondary) |
| k_budget | 256, 512, 1024 |
| Examples | Same 100 examples used in Phase 3 — match by example_id |
| Condition A | Reuse P0 scores |
| Condition B | Reuse Phase 3 scores |

Using the same 100 examples is critical. Oracle recovery rate is only meaningful when computed on the same examples as conditions A and B — you need the per-example triplet (A_score, B_score, Oracle_score) to compute the recovery rate without confounds from example difficulty variation.

**Total new inference calls:** 3 tasks × 2 eviction methods × 3 k_budgets × 100 examples = 1,800 oracle calls at 32K context. At 5–8 seconds per call, approximately 3–4 hours. This is the cheapest phase in terms of new compute because conditions A and B are already in hand.

---

## Implementation

### Step 1: Reconstruct Eviction Results for Each Example

Phase 3 saved eviction logs (positions, scores, Q vectors) but not the actual KV tensors of evicted tokens. To run the oracle you need the real KV tensors. There are two approaches:

**Approach A — Re-run Phase 3 eviction, capture EvictionResult in memory:**
For each example, run the full context prefill again, apply SnapKV, and hold the `EvictionResult` in GPU memory instead of discarding it. Then immediately run the oracle restoration and answer the query. This doubles the compute for Phase 5 but avoids storing large tensor files.

**Approach B — Store evicted KV tensors to disk during Phase 3:**
If you have sufficient disk space (~512KB × evicted_tokens × n_examples), save the evicted KV tensors alongside the JSON logs during Phase 3. This makes Phase 5 a pure load-and-restore operation.

With 100 examples at 32K context and k_budget=512, roughly 31,500 evicted tokens per example at 512KB each = ~15GB per task × k_budget combination. For 3 tasks × 3 k_budgets = 135GB total. This is large but feasible if you have NVMe storage.

**Recommendation:** Use Approach A. The Phase 3 prefill is cheap relative to the full pipeline, and avoiding the disk storage requirement keeps the codebase simpler. Structure the oracle runner as a single function that takes an example, runs prefill + eviction + oracle restoration + query in sequence:

```python
def run_oracle_example(
    model,
    tokenizer,
    context_tokens: torch.Tensor,    # [1, context_len]
    query_tokens: torch.Tensor,       # [1, query_len]
    eviction_policy: EvictionPolicy,
    k_budget: int,
    task_relevant_positions: list[int],
    example_id: str,
) -> dict:
    """
    Run one oracle example: prefill → evict → restore all → query.
    Returns a dict with oracle_score, condition_b_score, and diagnostics.
    """
    # --- Prefill with full context ---
    with torch.no_grad():
        ctx_out = model(context_tokens, use_cache=True)
    full_cache = PositionTrackedCache(
        kv=to_tuple_cache(ctx_out.past_key_values),
        positions=list(range(context_tokens.shape[1]))
    )

    # --- Apply eviction (Condition B) ---
    eviction_result = eviction_policy.evict(full_cache, k_budget=k_budget)
    log_eviction(eviction_result, example_id, task="oracle", 
                 task_relevant_positions=task_relevant_positions,
                 log_dir="results/phase5_logs/")

    # --- Condition B: answer with compressed cache ---
    with torch.no_grad():
        b_out = model(query_tokens, past_key_values=eviction_result.compressed.kv)
    condition_b_answer = decode_answer(b_out, tokenizer)

    # --- Oracle: restore ALL evicted tokens ---
    evicted_positions = eviction_result.evicted.positions
    evicted_kv_gpu = tuple(
        (k.to("cuda"), v.to("cuda"))
        for k, v in eviction_result.evicted.kv
    )
    evicted_cache = PositionTrackedCache(
        kv=evicted_kv_gpu,
        positions=evicted_positions
    )

    repaired_cache = inject_kv_with_positions(
        active_cache=eviction_result.compressed,
        active_positions=eviction_result.compressed.positions,
        new_pairs=evicted_cache,
        new_positions=evicted_positions,
    )

    with torch.no_grad():
        oracle_out = model(query_tokens, past_key_values=repaired_cache.kv)
    oracle_answer = decode_answer(oracle_out, tokenizer)

    # --- Cleanup: free GPU memory before next example ---
    del evicted_kv_gpu, repaired_cache, evicted_cache
    torch.cuda.empty_cache()

    return {
        "example_id": example_id,
        "condition_b_answer": condition_b_answer,
        "oracle_answer": oracle_answer,
        "eviction_survived": eviction_result.evicted.positions,
        "task_relevant_survived": [
            p in set(eviction_result.compressed.positions)
            for p in task_relevant_positions
        ],
    }
```

### Step 2: Load Condition A Scores

```python
def load_condition_a_scores(
    p0_results_path: str,
    task: str,
    example_ids: list[str],
) -> dict[str, float]:
    """
    Load per-example condition A scores from the P0 baseline results file.
    Returns {example_id: score}.
    """
    with open(p0_results_path) as f:
        p0_data = json.load(f)
    return {
        ex_id: p0_data[task]["per_example"][ex_id]["score"]
        for ex_id in example_ids
        if ex_id in p0_data.get(task, {}).get("per_example", {})
    }
```

If P0 did not record per-example scores (only aggregate accuracy), re-run P0 on the same 100 examples now with `save_per_example=True`. This takes ~1 hour and must be done before the oracle experiment.

### Step 3: Compute Recovery Rate

```python
def compute_oracle_recovery_rate(
    condition_a: dict[str, float],   # example_id → score
    condition_b: dict[str, float],   # example_id → score
    oracle: dict[str, float],        # example_id → score
    min_gap: float = 0.05,           # skip examples where A and B are very close
) -> dict:
    """
    Compute oracle recovery rate across all matched examples.

    Recovery rate = (Oracle - B) / (A - B)
    Aggregate by mean across examples where |A - B| >= min_gap.
    """
    recoveries = []
    skipped = 0

    for ex_id in condition_a:
        if ex_id not in condition_b or ex_id not in oracle:
            continue

        a = condition_a[ex_id]
        b = condition_b[ex_id]
        o = oracle[ex_id]

        if abs(a - b) < min_gap:
            skipped += 1
            continue  # eviction didn't change this example — uninformative

        recovery = (o - b) / (a - b)
        recoveries.append({
            "example_id": ex_id,
            "a": a, "b": b, "oracle": o,
            "recovery": recovery,
            "a_minus_b": a - b,
        })

    aggregate = {
        "mean_recovery": np.mean([r["recovery"] for r in recoveries]),
        "median_recovery": np.median([r["recovery"] for r in recoveries]),
        "pct_full_recovery": np.mean([r["recovery"] >= 0.95 for r in recoveries]),
        "pct_no_recovery": np.mean([r["recovery"] <= 0.05 for r in recoveries]),
        "n_evaluated": len(recoveries),
        "n_skipped": skipped,
        "per_example": recoveries,
    }

    return aggregate
```

**The `min_gap` filter is important.** If condition A and condition B both score 1.0 for a given example (the needle happened to survive eviction), the recovery rate formula is undefined (0/0). The filter removes these uninformative examples. Set `min_gap=0.05` — any example where eviction caused less than a 5-point accuracy drop is not informative for measuring recovery.

---

## Diagnostic Experiments

Run these if the aggregate oracle recovery rate is below 0.6. They identify which non-KV-content factor is responsible for residual degradation.

### Diagnostic 1: Positional Encoding Shift Test

**Hypothesis:** When the model processes context in a single pass, RoPE position IDs are contiguous. When the oracle restores evicted tokens, the position IDs within the repaired cache may be misaligned — tokens that were at positions 5000, 5001, 5002 in the original sequence are still at those IDs after restoration, but the ordering within the `past_key_values` tuple may not match what the model expects.

**Test:** After oracle restoration, explicitly pass `position_ids` aligned to the restored token order and verify it matches what a single-pass would produce.

```python
def test_position_id_alignment(
    model,
    context_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    repaired_cache: PositionTrackedCache,
) -> dict:
    """
    Compare generation with and without explicit position_ids on the repaired cache.
    If explicit position_ids improve oracle recovery, RoPE alignment is the issue.
    """
    cache_len = len(repaired_cache.positions)
    query_len = query_tokens.shape[1]

    # Implicit position_ids (model infers from cache length)
    with torch.no_grad():
        out_implicit = model(
            query_tokens,
            past_key_values=repaired_cache.kv,
        )

    # Explicit position_ids: query tokens start at cache_len
    position_ids = torch.arange(
        cache_len, cache_len + query_len,
        device=query_tokens.device
    ).unsqueeze(0)

    with torch.no_grad():
        out_explicit = model(
            query_tokens,
            past_key_values=repaired_cache.kv,
            position_ids=position_ids,
        )

    return {
        "implicit_logits": out_implicit.logits.cpu(),
        "explicit_logits": out_explicit.logits.cpu(),
        "logit_diff_max": (out_implicit.logits - out_explicit.logits).abs().max().item(),
    }
```

If the logit difference is large (> 0.1), explicit `position_ids` should be used in all oracle and repair experiments going forward.

### Diagnostic 2: Exact Serialization Baseline

**Hypothesis:** The degradation is not from eviction at all — it comes from the two-call structure (context in one call, query in another), independent of which tokens are in the cache.

**Test:** Save the complete uncompressed KV cache after context prefill, load it back without any eviction, and answer the query. If this does not match condition A, the two-call structure itself degrades quality.

```python
def test_exact_serialization_baseline(
    model,
    context_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    save_path: str = "/tmp/exact_kv_test",
) -> dict:
    """
    Two-call structure with exact KV preservation (no eviction).
    Should produce condition A quality if the issue is purely KV content loss.
    """
    # Call 1: context prefill, save complete KV
    with torch.no_grad():
        ctx_out = model(context_tokens, use_cache=True)
    full_kv = to_tuple_cache(ctx_out.past_key_values)
    save_kv(full_kv, save_path)

    # Simulate idle window (optional: sleep briefly)
    del full_kv, ctx_out
    torch.cuda.empty_cache()

    # Call 2: load KV, answer query
    loaded_kv = load_kv(save_path, device="cuda")
    with torch.no_grad():
        out_two_call = model(query_tokens, past_key_values=loaded_kv)

    # Reference: single call
    with torch.no_grad():
        single_call_out = model(
            torch.cat([context_tokens, query_tokens], dim=1),
            use_cache=False
        )

    logit_diff = (out_two_call.logits - single_call_out.logits[:, -query_tokens.shape[1]:, :]).abs().max().item()

    return {
        "two_call_logits": out_two_call.logits.cpu(),
        "single_call_logits": single_call_out.logits[:, -query_tokens.shape[1]:, :].cpu(),
        "max_logit_diff": logit_diff,
        "structurally_equivalent": logit_diff < 1e-2,
    }
```

Run this on 10 examples before the main oracle experiment. If `structurally_equivalent` is True for all 10, the P2 round-trip identity test is confirmed to hold for your actual task inputs and the two-call structure is not a confound. If False, debug the save/load pipeline — something is corrupting the KV cache between calls.

### Diagnostic 3: Attention Pattern Comparison

**Hypothesis:** Even with oracle restoration, the attention patterns differ from condition A because the model's attention is path-dependent — the order in which tokens were seen during the forward pass affects how later layers compute attention.

**Test:** Compare attention heatmaps between condition A (single-pass) and oracle-restored condition at the query tokens. Focus on the layers and heads that attend most strongly to the hop link positions in VT-4hop.

```python
def compare_attention_patterns(
    model,
    context_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    repaired_cache: PositionTrackedCache,
    task_relevant_positions: list[int],
    n_short_context: int = 4096,        # use short context for feasibility
) -> dict:
    """
    Compare query-token attention patterns between:
    (A) single-pass on truncated context
    (B) repaired-cache two-call on same truncated context

    Use SHORT context (4K) to make output_attentions=True feasible.
    The qualitative patterns should transfer to 32K.
    """
    assert context_tokens.shape[1] <= n_short_context, \
        "Use truncated context for attention comparison"

    # Condition A: single pass
    with torch.no_grad():
        out_a = model(
            torch.cat([context_tokens, query_tokens], dim=1),
            use_cache=False,
            output_attentions=True
        )

    # Oracle: two-call with full restoration
    with torch.no_grad():
        ctx_out = model(context_tokens, use_cache=True)
    full_kv = to_tuple_cache(ctx_out.past_key_values)

    with torch.no_grad():
        out_oracle = model(
            query_tokens,
            past_key_values=full_kv,
            output_attentions=True
        )

    # Extract attention weights for query tokens attending to context
    # Shape per layer: [batch, n_heads, query_len, total_seq_len]
    query_len = query_tokens.shape[1]

    pattern_diffs = []
    for layer_idx, (attn_a, attn_oracle) in enumerate(
        zip(out_a.attentions, out_oracle.attentions)
    ):
        # Condition A: last query_len rows of the attention matrix
        a_query_attn = attn_a[0, :, -query_len:, :]           # [heads, q_len, seq_len]
        # Oracle: attention from query tokens to context
        o_query_attn = attn_oracle[0, :, :, :]                # [heads, q_len, ctx_len]

        # Attention to task-relevant positions only
        a_relevant = a_query_attn[:, :, task_relevant_positions].mean().item()
        o_relevant = o_query_attn[:, :, task_relevant_positions].mean().item()

        pattern_diffs.append({
            "layer": layer_idx,
            "condition_a_relevant_attn": a_relevant,
            "oracle_relevant_attn": o_relevant,
            "diff": abs(a_relevant - o_relevant),
        })

    return {
        "layer_diffs": pattern_diffs,
        "mean_attn_diff_at_relevant_positions": np.mean([d["diff"] for d in pattern_diffs]),
    }
```

Run at 4K context on 10 VT-4hop examples. If `mean_attn_diff_at_relevant_positions` is small (< 0.01), attention patterns are well-preserved by oracle restoration. If large, path-dependency is the residual cause of non-perfect recovery.

---

## Interpreting the Results

### Outcome 1: Oracle recovery ≥ 0.85 (full recovery)

KV content loss accounts for at least 85% of condition B degradation. The repair algorithm in Phase 6 has a meaningful ceiling to approach. Proceed to Phase 6 without modification.

**Paper framing:** "We show that oracle restoration recovers X% of eviction-induced degradation on VT-4hop and Y% on MQ-NIAH, establishing that KV content loss is the dominant failure mechanism and that selective token restoration is a theoretically grounded approach."

### Outcome 2: Oracle recovery 0.40–0.85 (partial recovery)

KV content accounts for most but not all degradation. The residual is likely positional encoding shift or attention path-dependency. Run Diagnostics 1 and 2 to attribute the residual.

**Paper framing:** "Oracle restoration recovers X% of degradation, with the remaining gap attributable to [RoPE position shift / attention path-dependency], which represents a direction for future work. Our repair algorithm targets the recoverable X% fraction."

### Outcome 3: Oracle recovery < 0.40 (low recovery)

Degradation is largely not from KV content loss. Do not proceed to Phase 6 until this is diagnosed. The most likely causes:

- P2 round-trip identity test is passing but Diagnostic 2 (exact serialization baseline) fails at the task level. The two-call structure itself is degrading quality in a way that affects these specific tasks.
- The position ID alignment is wrong and needs Diagnostic 1's explicit `position_ids` fix.
- The eviction is so aggressive at the tested k_budget that the model is working from an essentially random cache — try a lighter eviction (k_budget=2048) first to confirm oracle recovery is possible at all.

---

## Results Table Structure

The primary table for the paper. Fill in after running all conditions:

```
Table: Oracle Recovery Rate = (Oracle − B) / (A − B)

                    k_budget=256        k_budget=512        k_budget=1024
Task                SnapKV  Stream      SnapKV  Stream      SnapKV  Stream
VT-4hop             X.XX    X.XX        X.XX    X.XX        X.XX    X.XX
MQ-NIAH-4q          X.XX    X.XX        X.XX    X.XX        X.XX    X.XX
S-NIAH              X.XX    X.XX        X.XX    X.XX        X.XX    X.XX
```

Also report the absolute scores A, B, Oracle as a secondary table:

```
Table: Absolute scores at k_budget=512, SnapKV

Task        Condition A     Condition B     Oracle      Recovery
VT-4hop     X.XX            X.XX            X.XX        X.XX
MQ-NIAH-4q  X.XX            X.XX            X.XX        X.XX
S-NIAH      X.XX            X.XX            X.XX        X.XX
```

**Expected patterns based on task structure:**

VT-4hop: Oracle recovery should be high (0.70–0.95) because each hop link is a discrete localized token. If the link is restored, the chain can be followed. The remaining gap (if any) is likely path-dependency in chain reasoning.

MQ-NIAH-4q: Oracle recovery should be near 1.0 (0.85–0.99) because NIAH is the simplest retrieval task and the oracle restores exactly the tokens that were causing failures. The failure mode is precise — needle evicted → wrong answer, needle restored → correct answer.

S-NIAH: Oracle recovery may be lower than expected if the task is already near-perfect under eviction (condition B ≈ condition A). In this case the recovery formula is undefined for many examples and the task is not a good discriminator.

StreamingLLM: Oracle recovery for StreamingLLM may be lower than for SnapKV because StreamingLLM evicts ALL middle-context tokens, not just the least important ones. Restoring all of them gives the model back its full context, but the model must now integrate a larger repaired cache. For VT-4hop under StreamingLLM with oracle, you are essentially restoring the full 32K cache — similar to condition A.

---

## Per-Example Recovery Distribution

Beyond the aggregate table, plot the distribution of per-example recovery rates as a histogram. This reveals whether recovery is consistently partial (suggesting a structural non-recoverable component) or bimodal (suggesting it is all-or-nothing per example, which would mean the repair algorithm just needs to correctly identify the right tokens).

```python
def plot_recovery_distribution(
    recovery_data: dict,
    task: str,
    k_budget: int,
    save_path: str,
):
    import matplotlib.pyplot as plt

    recoveries = [r["recovery"] for r in recovery_data["per_example"]]
    recoveries = np.clip(recoveries, -0.1, 1.1)  # clip outliers for display

    plt.figure(figsize=(8, 4))
    plt.hist(recoveries, bins=20, range=(-0.1, 1.1), color="#5DCAA5", edgecolor="white")
    plt.axvline(np.mean(recoveries), color="#1D9E75", linestyle="--",
                label=f"Mean: {np.mean(recoveries):.2f}")
    plt.xlabel("Oracle recovery rate per example")
    plt.ylabel("Count")
    plt.title(f"{task} — k_budget={k_budget}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
```

A bimodal distribution (mass near 0.0 and near 1.0, little in between) means the broken token either is or is not in the evicted buffer — if it is there and restored, the example recovers fully; if it was never evicted (condition B already correct) or is missing from the oracle for some other reason, no recovery. This pattern would strongly support the P6 repair algorithm's design: the task reduces to correctly identifying which token to restore, not partially restoring many tokens.

A unimodal distribution centered around 0.6–0.8 means recovery is consistently partial for most examples, suggesting a structural residual that repair can only partially address.

---

## Go/No-Go Decision

Proceed to Phase 6 if the following both hold:

1. Oracle recovery rate ≥ 0.60 on VT-4hop at k_budget=512 under SnapKV.
2. Diagnostic 2 (exact serialization baseline) confirms the two-call structure is not itself causing degradation (i.e., `structurally_equivalent=True` on ≥ 8 of 10 test examples).

If criterion 1 is met but criterion 2 fails: fix the KV serialization pipeline first, then re-run oracle. The oracle recovery rate with a broken serialization pipeline is meaningless.

If criterion 1 is met but recovery is < 0.60 specifically for MQ-NIAH: this is acceptable — the tasks have different structures, and VT is your primary contribution. Report MQ-NIAH oracle recovery as a secondary result.

If neither criterion is met: run Diagnostics 1–3 before any other action. Do not proceed to Phase 6 and do not modify the experimental design to paper over the issue. Understanding *why* oracle recovery is low is as publishable as high oracle recovery — it establishes fundamental limits of KV cache repair.

---

## File Structure

```
src/
  oracle/
    __init__.py
    runner.py               # run_oracle_example(), run_oracle_batch()
    diagnostics.py          # all three diagnostic tests
    recovery.py             # compute_oracle_recovery_rate(), plot_recovery_distribution()

results/
  phase5_oracle/
    VT4hop_snapkv_k512_oracle.json     # per-example {a, b, oracle, recovery}
    VT4hop_snapkv_k256_oracle.json
    VT4hop_snapkv_k1024_oracle.json
    VT4hop_streaming_k512_oracle.json
    MQNIAH_snapkv_k512_oracle.json
    SNIAH_snapkv_k512_oracle.json
    recovery_table.json                # aggregate table across all conditions
    diagnostics/
      serialization_baseline.json
      position_id_test.json
      attention_pattern_diff.json
    figures/
      recovery_table.png               # main paper table as figure
      VT4hop_recovery_distribution.png
      MQNIAH_recovery_distribution.png
      oracle_vs_k_budget.png           # recovery rate vs k_budget curve
```

---

## Deliverable

All oracle scores saved per-example with triplets (A, B, Oracle). Recovery rate table across all task × k_budget × eviction method conditions. Diagnostic tests run and saved. Recovery distribution histograms. Go/no-go decision documented explicitly in `results/phase5_oracle/go_nogo.txt` with the specific criteria and whether they are met.

The oracle recovery rate table, the recovery distribution plots, and the diagnostic results go directly into the paper — they are not scaffolding for Phase 6. They are the paper's central empirical claim about the nature of KV eviction degradation.