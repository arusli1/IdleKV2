# Phase 16 Final Reruns Plan

Last updated: 2026-05-05.

## Goal

Phase 16 is the last experiment-preparation pass before submission. It does
not change the paper unless a run clears a predeclared promotion gate. The goal
is to close the two reviewer risks that remain after Phase 15:

- portability beyond the primary Qwen2.5-7B-Instruct setting;
- stronger or named cache-policy comparators, especially Scissorhands-style
  persistence retention and Refresh-buffered full-budget reselection.

Do not run the locked jobs until their smoke gate passes. Do not launch full
runs from this directory without a fresh confirmation.

## Run Order

1. **Static readiness.** Run script syntax checks, config construction, and
   focused unit tests. This is CPU-only and safe.
2. **Mistral smoke.** Test exact-Q repair on `Mistral-7B-Instruct-v0.3` with
   `n=2`, `B=16384`, and `K={24,48,96}` on MQ-NIAH-6Q.
3. **Scissorhands smoke.** Test the Scissorhands-style first-stage compressor
   with `n=2`, `B=18432`, and `K={48,96,128}` on MQ-NIAH-6Q.
4. **Refresh boundary smoke.** Confirm full-budget reselection still dominates
   or bounds RepairKV under exact-Q scoring before any final wording change.
5. **Optional Llama retry smoke.** Only run if Mistral fails for a mechanical
   model-family reason, not just because the result is less pretty.
6. **Locked full runs.** Launch only the branches whose smoke result is clean
   and whose paper action is already known.

## Promotion Gates

A portability or Scissorhands result can enter the main paper only if:

- full-cache ability is at least `0.90`;
- `A - B_match >= 0.20` on at least two K points;
- `RepairKV - B_match >= 0.10` on at least two K points;
- `RepairKV` beats the best content-agnostic control by at least `0.05` on at
  least two K points;
- the result is not fully saturated across all K points.

If a run fails because full-cache ability is weak, redesign the task/budget.
If it passes but saturates, keep it appendix-only. If Scissorhands weakens the
base cache too much, treat it as a stress-test failure rather than a negative
result against the RepairKV mechanism.

Locked artifacts must also pass `audit_phase16_locked.py`, which adds paired
bootstrap intervals for `RepairKV - B_match`, `RepairKV - Random-K`,
`RepairKV - Oldest-K`, and `RepairKV - max(Random-K, Oldest-K)` at each K. A
Phase 16 result is eligible for a main-text reference only when at least two K
points clear the mean gates and have positive paired intervals against both
matched no-repair and content-agnostic controls. If the mean gates pass but
the interval or saturation gate is weak, the result is appendix-only. If the
full-cache gate or matched-gap gate fails, defer the result.

## Execution Loop

For each Phase 16 branch:

1. Run the smallest mechanical smoke that exercises the exact model,
   compressor, K grid, and conditions.
2. Evaluate the smoke with `evaluate_phase16_smokes.py`.
3. Launch the locked tmux run only if the smoke gate passes.
4. Summarize the locked artifact with `phase9_artifact_summary.py`.
5. Audit the locked artifact with `audit_phase16_locked.py`.
6. Decide placement before editing the paper:
   - `main_reference_plus_appendix`: one cautious main sentence plus compact
     appendix details;
   - `appendix_only`: appendix note or figure/table, with at most a main
     appendix pointer;
   - `defer_do_not_include`: no paper integration.
7. Rebuild the PDF and run focused tests after any paper edit.

## Paper Actions

- **Mistral pass:** add one cautious portability sentence or appendix figure.
  Do not claim model-family generality.
- **Mistral near miss:** if `n=4` misses the full-cache ability gate by less
  than one point while the repair/control separation remains large, run one
  larger same-protocol smoke before deferring. Do not change prompts, budgets,
  or scoring to chase the gate.
- **Mistral fail:** if the larger same-protocol smoke still misses the
  full-cache ability gate, defer Mistral rather than spending a locked run.
  Existing Llama artifacts already provide the safer cross-model appendix
  evidence.
- **Scissorhands pass:** add a named retention-policy stress-test sentence.
  Phrase as "Scissorhands-style first-stage retention" unless the implementation
  is upgraded to a full per-head online reproduction.
- **Refresh pass:** keep it as a method-boundary comparator: useful idle-window
  repair is not the strongest possible Q2-aware reselector.
- **Any fail:** leave Phase 16 out of the main paper and report only as future
  work if scientifically informative.
