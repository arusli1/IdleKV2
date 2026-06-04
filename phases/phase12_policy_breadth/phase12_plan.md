# Phase 12 Policy Breadth Plan

Last updated: 2026-05-03 08:08:16 UTC.

## Goal

Tested whether the idle-window repair effect survives a second non-SnapKV
first-stage retention rule beyond the H2O-inspired accumulated-attention
check.

The best next candidate is sink-plus-recent retention inspired by
StreamingLLM. It is structurally different from SnapKV and
accumulated-attention retention: the first-stage cache is mostly
position-biased rather than content- or attention-mass-biased.
If repair works here, the paper can say the primitive is not only repairing
mistakes from attention-score compressors. If it fails, that is still useful:
it shows that purely structural eviction can over-compress the middle unless a
larger restore budget or stronger selector is used.

## Candidate Policies

1. **SnapKV.** Primary paper policy and current main frontier.
2. **H2O-inspired accumulated-attention retention.** Already passed the
   full-grid main-candidate gate as a cautious compressor robustness check.
3. **Sink-plus-recent retention inspired by StreamingLLM.** Completed as a locked `n=24`,
   `B_base=16384` full K-grid after the calibrated smoke passed.
4. **PyramidKV/Ada-KV-style layer allocation.** High-value but not immediate:
   the current repair implementation assumes one global context-position set
   across layers, so layer-varying retention would need new injection tests.
5. **QUEST query-aware paging.** Important related work, but not a clean
   first-stage retention-rule comparison because it uses query-time retrieval.

Literature sanity check: StreamingLLM is a good structural baseline because it
keeps attention sinks plus a recent window; H2O is the canonical heavy-hitter
attention-retention baseline; PyramidKV/Ada-KV are stronger layer/head-budget
directions but require repair machinery that can handle layer-varying retained
positions; QUEST methods are query-time loading baselines, not clean
post-compression first-stage policies for the current protocol.

## Sink-Plus-Recent Smoke

Run MQ-NIAH-4Q with Qwen2.5-7B-Instruct, exact Q2 scoring, Gold-K reference,
and the same active-cache accounting used by the main paper.

- Budgets: `B_base in {12288, 16384}`.
- Restore grid: `K in {8,16,24,32,48,64,80,96,128}`.
- Samples: `n=2` for the smoke.
- Conditions: `A`, `B`, matched no-repair, Random-K, Oldest-K, RepairKV, Gold-K.

Promotion gate:

- Full-cache score should remain high enough to show task solvability.
- Matched no-repair should leave at least `0.20` absolute score gap.
- RepairKV should gain at least `0.15` at one or more non-saturated K values.
- Random-K and Oldest-K should not explain the gain.
- Gold-K should cover RepairKV, showing the benchmark metadata still bounds the
  selector.

## Full Run Result

The locked `n=24` run used the calibrated `B_base=16384` budget and the same
K-grid. It passed the Phase 11 main-candidate gate:

```
task=clean_suite base_context_budget=16384 points=9 best_k=128
best_gain=0.431 best_eligible_k=128 best_eligible_gain=0.431
full_gap=0.667 non_saturated=True main_candidate=True action=main_candidate
```

The main paper now renders a compact policy-robustness figure:

- One column.
- X-axis: restore budget K.
- Y-axis: score gain over matched no-repair.
- Rows: SnapKV, accumulated-attention retention, sink-plus-recent retention.
- Blue line: RepairKV gain.
- Orange dashed line: Gold-K benchmark-metadata reference gain.
- Gray band: Random-K/Oldest-K control-gain range.

Main-paper promotion required a clean curve, not just a positive endpoint:
The sink-plus-recent branch must pass the Phase 11 gate, show positive RepairKV
gain at two or more adjacent K values, keep Random-K/Oldest-K near matched
no-repair, and remain bounded by Gold-K. The completed run satisfies this
gate, but the paper still describes both non-SnapKV branches as
protocol-matched retention variants rather than canonical reproductions.

## Smoke Result

The calibrated smoke finished on 2026-05-03. `B_base=12288` failed because
Random-K explained part of the gain. `B_base=16384` passed: best gain `+0.250`
at `K=128`, full-vs-matched gap `0.667`, control lift `0.000`, and Gold-K
covers RepairKV. The full `n=24` follow-up then passed and was integrated into
the main paper.
