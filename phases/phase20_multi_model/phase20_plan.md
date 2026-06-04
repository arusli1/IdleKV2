# Phase 20: Multi-Model Cross-Cut

## Goal

Run RepairKV's headline experiments on Llama-3.1-8B-Instruct and
Mistral-7B-Instruct-v0.3 on the same task cells as the Qwen
headline runs from Phase 18 (and ideally Phase 19's non-NIAH cell).
Defuse the "single-model fragility" attack with symmetric coverage,
not a tack-on K=96 run.

## Why this is its own phase

Phase 18's Step 6 originally bundled multi-model into the Qwen
submission cycle, but at 4Q K=96 only — asymmetric vs the 9-K Qwen
sweep, 6Q sweep, and burst ablation. Reviewer-flagged as a weak
generalization claim. Phase 20 does multi-model *properly*: matched
cells across all three models on the headline task suite.

## Cells to run

| Task | K values | n | Models | Conditions |
|---|---|---|---|---|
| 4Q | {32, 64, 96, 128} (reduced sweep) | 24 | Llama-3.1-8B, Mistral-7B-v0.3 | 6 (full W1 set) |
| 6Q | 96 only | 24 | both | 6 |
| SCBench multi-turn QA *(if Phase 19 has shipped)* | 96 | 24 | both | 6 |

Reduced K-sweep on 4Q (4 K's instead of 9) keeps GPU cost
manageable without losing the frontier-shape evidence. If the
reduced sweep is consistent with Qwen's 9-K shape, the paper can
report "consistent frontier shape across models" without re-running
all 9 K's per model.

## Workstream

### W1 — Model-loading sanity (~30 min)

- Confirm Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3 load
  cleanly under the existing runner's model loader. The Phase 16
  scripts already touched Mistral smoke; reuse that path.
- Verify both models support the protocol's RoPE-position-id custom
  pass (needed for TM-Recompute-BM25). Llama uses the same RoPE
  family as Qwen; Mistral uses a sliding-window variant — confirm
  the prefill helper works without sliding-window cache splits.

### W2 — Smoke per model (~30 min GPU)

n=4 at K=96 on 4Q, all 6 conditions. Sanity-check the sign of
Δ(RepairKV − B_match) before committing to paper-quality runs.

### W3 — Paper-quality runs (~4-5 hr GPU)

Llama 4Q reduced sweep + Llama 6Q K=96, then Mistral same shape.
Sequential to keep T_repair fair (per Phase 18 v5 §Why-sequential).

### W4 — SCBench cells *(if Phase 19 has landed)* (~2 hr GPU)

Llama + Mistral at SCBench K=96, n=24, 6 conditions.

### W5 — Paper edits

Add a multi-model Results subsection or expand the existing
eviction-policy-sensitivity panel into a dual model+policy
robustness panel.

## Acceptance criteria

Same Phase 18 strong/weak/fail tiers, applied per model. The
abstract qualifier upgrades from "Qwen-only with preliminary Llama
evidence" to "consistent across three models tested" if all three
models clear the strong-pass threshold against
PageSummary-Quest-inspired at K=96 on 4Q.

## Sequencing

```
Day 1 — W1 + W2 (loading + smokes)
Day 1-2 — W3 (Llama then Mistral, sequential)
Day 2-3 — W4 (SCBench cells, conditional on Phase 19)
Day 3 — W5 paper edits
```

## Risks

1. **Mistral sliding-window attention** breaks the RoPE-position
   custom pass. Mitigation: smoke catches this; fall back to a
   sliding-window-aware prefill or drop Mistral from W1.
2. **One model shows a clean Phase-18-style win and the other does
   not.** That is *informative* — likely points to a
   model-architectural dependence in the burst-expansion mechanism.
   Honest report: "RepairKV's gain magnitude depends on the model's
   attention concentration; mechanism survives but effect-size
   varies." Don't bury it.
3. **GPU time blows out.** If W3 alone takes ~5 hr and W4 is
   another ~2 hr, Phase 20 needs ~7 hr GPU. Pre-budget; if needed,
   drop the reduced 4Q sweep to K=96 only on the second model.

## Out of scope for Phase 20

- Larger models (70B+). Defer to Phase 21+.
- Quantized models (AWQ/GPTQ). Defer.
- Models without HuggingFace `attn_implementation` plumbing.
