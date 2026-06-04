# Phase 11 Main-Candidate Robustness

Phase 11 asks whether any additional evidence deserves main-paper space beyond
the existing Qwen2.5-7B 2Q/4Q/6Q/8Q matched-budget frontier.

## Audit Decision

- Keep the current Llama-3.1-8B fixed-point result as appendix evidence unless a
  larger same-protocol 4Q K-grid is clean enough to support a compact
  main-text portability statement.
- Treat retention-rule breadth as a portability check, not a new benchmark suite.
  One canonical second policy is enough for the paper. The best candidate is
  the H2O-inspired accumulated-attention first-stage retention rule already
  implemented in Phase 6.
- Do not spend major compute on sink-plus-recent retention inspired by
  StreamingLLM unless a smoke
  suggests the previous weak result was a calibration issue.

## Main-Candidate Runs

1. **H2O-inspired accumulated-attention retention K-grid.**
   - Task: `clean_suite` (MQ-NIAH-4Q split-query suite).
   - Model: Qwen2.5-7B-Instruct.
   - First-stage compressor: `h2o`.
   - Budget: `B_base=16384`.
   - Samples: `n=24`.
   - Restore budgets: `K={8,16,24,32,48,64,80,96,128}`.
   - Main gate: RepairKV must beat matched no-repair and Random/Oldest over the
     curve, with no broad negative split and Gold-K covering RepairKV.

2. **Llama-3.1-8B 4Q K-grid.**
   - Task: `clean_suite` (same 4Q protocol as the Qwen 4Q main panel).
   - Model: Llama-3.1-8B-Instruct.
   - First-stage compressor: `snapkv`.
   - Budget: `B_base=16384`.
   - Samples: `n=24` initially; escalate only if the curve is noisy and still
     promising.
   - Restore budgets: `K={8,16,24,32,48,64,80,96,128}`.
   - Main gate: enough curve shape to show cross-family portability without
     claiming broad multi-model robustness. At least one planned point must be
     non-saturated; if RepairKV and Gold-K are flat at 1.0 across the useful
     region, the result remains appendix-only even if positive.

## Current Run Status

- H2O-inspired accumulated-attention retention K-grid finished on 2026-05-03. Promotion gate output:
  `main_candidate=True`, best gain `+0.764` at `K=128`, and non-saturated
  positive curve shape. Paper decision for now: update main robustness prose
  and render the appendix figure as a full K-grid frontier; do not add a new
  main figure unless Llama also passes cleanly.
- Llama-3.1-8B 4Q K-grid finished on 2026-05-03. Promotion gate output:
  `main_candidate=True`, best gain `+0.521` at `K=64`, and non-saturated
  positive curve shape. Paper decision for now: update main robustness prose
  and render the appendix figure as a full K-grid frontier; consider a compact
  main robustness figure only if the policy-breadth branch also passes.

## Paper Placement Gate

- If only one run passes, mention it in main prose and keep the plot in the
  appendix.
- If both pass with clean curves, consider a compact one-column robustness
  figure with two small panels: one model-transfer panel and one compressor
  panel. Do not add this if it crowds the main frontier or forces overclaiming.
- If a result is positive but sparse/noisy, keep it appendix-only.
- Do not claim broad model-family or compressor-family generalization from
  Phase 11. Safe language is "same-protocol portability check" for Llama and
  "H2O-inspired accumulated-attention first-stage retention check" for the
  compressor branch.
