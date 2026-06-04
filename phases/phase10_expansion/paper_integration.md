# Phase 10 Paper Integration Notes

This file tracks how Phase 10 should change the paper if the planned
experiments pass. It is a writing guide, not an experimental artifact.

## Core Reframe

The paper should not be framed as "we found a trick for one synthetic
two-turn benchmark." The stronger framing is:

> Modern LLM workflows create dynamic context relevance. A cache
> compressed after one turn can be stale for the next turn. Idle windows
> give the runtime a chance to repair compressed KV state using
> information that was unavailable when compression happened.

This supports a workshop-style contribution: define the repair primitive,
show controlled evidence that it works, test whether the effect is
specific to newly revealed turn information, map where it works, and
identify open systems directions.

## Claim Ladder

Use this ladder to keep claims precise.

1. **Established by Phase 7/9:** under matched active GPU KV budgets,
   restoring selected evicted KV rows can recover future-turn answers in
   controlled MQ-NIAH splits.
2. **Established if specificity controls pass:** the effect depends on
   the newly available future-turn query, not just generic buffer
   reinsertion.
3. **Established if multi-turn passes:** the primitive can repeatedly
   adapt cache state across relevance shifts and revisits.
4. **Established if model/compressor smokes pass:** the effect has
   preliminary portability hints, not broad model/compressor generality.
5. **Only exploratory unless real kernels are implemented:** the same
   repair pattern may apply to low-bit KV caches by promoting selected
   rows back to high precision.

## Main Figure Package

Target main-paper figures after Phase 10. The compact live experiment map
is `phase10_high_signal_map.md`.

1. **System schematic:** cache compression, CPU evicted-KV buffer, idle
   scorer, repair before next turn.
2. **Matched-budget frontier:** faceted 2Q/4Q/6Q/8Q raw score versus
   restored rows. Show matched no-repair, Random-K, and Oldest-K controls
   in each facet.
3. **Specificity contrast:** stale-query, donor-query, RepairKV,
   Refresh-buffered, and Gold-K. Plot mean score gain over matched
   no-repair with uncertainty plus paired win/tie/loss rates.
4. **Multi-turn trajectory or cache-state heatmap:** relevance shifts
   over several turns, including a revisit.
5. **Operating-regime heatmap:** where repair helps across base budget
   and restore budget.

If the multi-turn result is positive, it should be a main-paper figure,
not a low-priority appendix item, because it is the cleanest experimental
bridge from the two-turn diagnostic to dynamic agent-style workflows. If
the main text becomes crowded, keep figures 1, 2, 3, and the positive
multi-turn figure main; move the operating-regime heatmap and portability
panels to appendix.

Current integration choice: the strict-cap streaming spill table is
appendix-only because it measures recoverable coverage rather than
answer quality and uses only `n=4` examples per setting. A locked
specificity panel should replace that main-text space if it passes.
The final `n=24` operating-regime heatmap has been promoted to main text
because it answers the budget-cherry-picking objection: it maps where
repair works, saturates, or has too little restore budget. It should be
described as a within-task regime diagnostic, not as a new baseline or
cross-task effect-size comparison.

## Appendix Figure Package

Use appendix for high-signal but secondary evidence:

- Query-count breadth across 2Q/3Q/4Q/6Q/8Q.
- One new-model smoke or compact model-transfer panel.
- One sink-plus-recent retention robustness panel only if a locked follow-up
  shows a clear gain; the current `n=1` smoke is too weak for evidence.
- Precision-promotion quality/byte Pareto if fake-quant signal is clean.
- Per-split and budget-calibration tables.

Avoid appendix dumps. Every appendix figure should answer one reviewer
question.

## Abstract Template If Phase 10 Passes

Use 4-6 sentences. Do not mention "Phase 10."

1. Long-context workflows increasingly shift relevance across turns, so
   a KV cache compressed after one turn may be stale for the next.
2. We introduce RepairKV, an idle-window repair primitive that keeps
   evicted KV rows in a CPU buffer and restores selected rows before the
   next turn under a matched active GPU KV budget.
3. On controlled multi-query retrieval diagnostics, RepairKV recovers a
   large fraction of future-turn answer quality lost to compression.
4. Specificity controls show that repair follows newly revealed
   relevance rather than merely adding arbitrary buffered rows.
5. Preliminary model/retention-rule or precision-promotion results suggest
   idle-time repair may be a useful direction for dynamic KV-state
   maintenance.

Only include sentence 5 if the corresponding Phase 10 evidence passes.

## Terminology

Use these reader-facing terms:

- `active GPU KV budget`: the KV rows resident in the resumed cache.
- `evicted-KV buffer`: CPU-resident KV rows removed by first-stage
  compression.
- `matched no-repair`: a baseline that gets the same active GPU KV
  budget by increasing the first-stage retention budget.
- `next-turn signal specificity`: the requirement that repair uses the
  newly available next-turn signal, not a stale or unrelated query.
- `Refresh-buffered`: a Q2-time reselection comparator under the same
  resumed active GPU KV budget. It reselects from active plus CPU-buffered
  rows without full-prefix recompute.
- `Refresh-recompute`: a stronger comparator that reruns selection after
  recomputing or reloading full-prefix KV. Keep distinct from
  Refresh-buffered because it is a different systems point.
- `precision promotion`: replacing selected low-bit KV rows with their
  high-precision originals under a matched byte budget.

Do not use internal labels such as phase numbers, bridge, extension, or
mockup in the paper.

## Decision Rules

- Main text should contain no result that is only a smoke.
- If a Phase 10 result has `n=2`, it can guide design but should not be
  written as evidence.
- The specificity smoke is design evidence only. It selected `K=48`
  because `K=96` failed stale-query separation. Main text can use the
  specificity panel only after the locked `n=24`, `K=48` follow-up passes.
- If the locked specificity result keeps `Refresh-buffered` above RepairKV,
  the caption and prose must say that RepairKV is an incremental buffered
  repair primitive relative to a stronger Q2-time buffered reselection
  comparator.
- If a result uses fake quantization, label it as quality-only.
- If a cross-model result lacks cache round-trip validation, do not use
  it.
- If a compressor result changes both the compressor and the task
  geometry, do not use it as a clean portability claim.

## Specificity Integration Gate

When `specificity_locked_n24_k48.csv` lands, run
`recommend_specificity_next.py` and promote only if:

- RepairKV gain over matched no-repair is at least `0.15`.
- RepairKV gain CI lower bound is positive.
- RepairKV beats `StaleQ-K` by at least `0.10`.
- RepairKV beats `WrongQ-K` by at least `0.10`, or the donor-query caveat
  is explicitly stated.
- RepairKV paired win rate over matched is at least `0.55`.

If those pass but `Refresh-buffered` beats RepairKV by more than `0.05`,
the result can still enter the main paper as a novelty-boundary panel,
but the claim becomes: RepairKV is an incremental paused-cache repair
primitive; full-budget buffered reselection is a stronger comparator
with different algorithmic behavior and a different systems point.
