# Phase 16 Status

Last updated: 2026-05-05.

## Current State

Phase 16 is in the finalization pass. The Scissorhands-style locked run passed
the locked audit. Mistral portability was reopened for one larger same-protocol
smoke because the `n=4` full-cache score missed the gate by less than half a
point while the repair separation remained large; the larger smoke repeated the
same full-cache near miss, so Mistral is deferred.

Implemented in this setup pass:

- Mistral RoPE support in the Phase 6 exact-Q path.
- A Scissorhands-style first-stage compressor option,
  `--initial-compressor scissorhands`, using a persistence/low-attention
  history-window score over the turn-N generated answer window.
- Static Phase 16 config validation.
- Smoke and locked-run wrappers for Mistral, Scissorhands, Refresh-buffered,
  optional Llama retry, and final Qwen reruns.
- A Phase 16 smoke evaluator with paper-action gates.
- A locked-run audit helper,
  `phases/phase16_final_reruns/scripts/audit_phase16_locked.py`, which adds
  paired bootstrap intervals and emits a `main_reference_plus_appendix`,
  `appendix_only`, or `defer_do_not_include` recommendation.

Smoke results completed in this pass:

- Scissorhands-style MQ-NIAH-6Q smoke passed the locked-run gate. At
  `B=18432`, `n=2`, and `K={48,96,128}`, full-cache ability was `0.958`,
  RepairKV exceeded matched no-repair by `+0.333/+0.667/+0.667`, and it
  exceeded the best content-agnostic control by `+0.292/+0.667/+0.625`.
  Artifact:
  `phases/phase16_final_reruns/results/scissorhands_smoke_n2_b18432.csv`.
- Mistral-7B-Instruct-v0.3 MQ-NIAH-6Q smoke was mechanically valid but failed
  the predeclared full-context gate. Full-cache ability was `0.875` at
  `B=16384`, so this should not unlock a locked full run without redesigning
  the task/budget or explicitly revising the gate. Artifact:
  `phases/phase16_final_reruns/results/mistral_smoke_n2_b16384.csv`.
  Audit artifact:
  `phases/phase16_final_reruns/results/mistral_smoke_n2_b16384_audit.json`.
- Mistral `n=4` re-smoke repeated the same outcome. Full-cache ability was
  `0.896`, still below the predeclared `0.90` gate, while RepairKV remained well
  separated from matched no-repair and content-agnostic controls. This is close
  enough to justify one larger same-protocol smoke, but it is not yet paper
  evidence. Artifacts:
  `phases/phase16_final_reruns/results/mistral_smoke_n4_b16384.csv` and
  `phases/phase16_final_reruns/results/mistral_smoke_n4_b16384_audit.json`.
- Mistral `n=8` same-protocol smoke confirmed the near miss rather than
  clearing the gate. Full-cache ability was again `0.896`; matched no-repair
  was `0.135`; and RepairKV reached `0.396/0.833/0.865` at
  `K=24/48/96`, with positive paired intervals over matched no-repair and
  content-agnostic controls. Decision: do not launch a locked Mistral run,
  because the task/model ability gate is not met under the frozen protocol.
  Artifacts:
  `phases/phase16_final_reruns/results/mistral_smoke_n8_b16384.csv` and
  `phases/phase16_final_reruns/results/mistral_smoke_n8_b16384_audit.json`.
- Refresh-buffered boundary smoke confirms the intended framing: Refresh-K is
  a stronger full-budget reselection reference, not a matched incremental
  repair baseline. Artifact:
  `phases/phase16_final_reruns/results/refresh_boundary_smoke_n2.csv`.
- The Scissorhands smoke also passes the locked-run audit helper with positive
  paired intervals at all three smoke K values. Smoke audit artifact:
  `phases/phase16_final_reruns/results/scissorhands_smoke_n2_b18432_audit.json`.
- Scissorhands-style MQ-NIAH-6Q locked run passed the promotion gate. At
  `B=18432`, `n=24`, and `K={48,64,80,96,128}`, full-cache ability was
  `0.990`; RepairKV gains over matched no-repair were
  `+0.347/+0.431/+0.556/+0.618/+0.622`; and paired bootstrap lower bounds
  against matched no-repair were all positive. Recommendation:
  `main_reference_plus_appendix`. Artifacts:
  `phases/phase16_final_reruns/results/scissorhands_locked_n24_b18432.csv`
  and
  `phases/phase16_final_reruns/results/scissorhands_locked_n24_b18432_audit.json`.

Active run: none.

Next planned checks:

1. Keep the Scissorhands-style locked result as a short main-text retention
   stress-test reference plus compact appendix details.
2. Do not promote Mistral portability evidence from Phase 16. It repeatedly
   missed the full-cache ability gate under the frozen MQ-NIAH-6Q protocol, even
   though the repair/control separation was strong.

## Stop Rules

- Do not run locked scripts until the matching smoke artifact passes
  `evaluate_phase16_smokes.py`.
- Do not promote Scissorhands as an exact named reproduction; the current
  implementation is a faithful one-shot persistence stress test, not a full
  per-head online allocator.
- Do not make new main-paper claims from saturated portability curves.
- Do not run full jobs while another GPU experiment is active.
