# Phase 1 Smoke Note

This document is the top-level Phase 1 benchmark note for the main `phases/phase1_degradation` harness.

It captures the two benchmarks we currently treat as the Phase 1 pair:

1. `VT-4hop-permute`
2. `MQ-NIAH-4q`

This file is intentionally written as a stable benchmark definition plus result scaffold. Final averaged numbers should be filled from the repeated-sweep artifacts once they complete.

## Common Setup

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Context length: `32768`
- Eviction policy: `SnapKV`
- Main compressed condition: prefill the full 32K context once, compress the live KV cache during prefill, then answer from the compressed cache without reprefilling the context
- Main reported budgets for the extended curve: `1024`, `2048`, `4096`, `8192`, `16384`, `32768`
- Repeated-curve protocol: `5` independent dataset draws, with no fixed dataset seed

The two main artifacts for the averaged curve are expected to be:

- per-repeat summaries under `results/<run_label>_rXX_summary.json`
- aggregate summary under `results/<run_label>_aggregate_summary.json`

## Benchmark 1: VT-4hop-permute

### Task Definition

This is the main variable-tracking benchmark.

The true chain is:

- `A = value`
- `A = B`
- `B = C`
- `C = D`

One distractor branch is also inserted:

- `C = E`

The rendered context presents these statements in the following order:

- `A = B`
- `C = D`
- `A = value`
- `C = E`
- `B = C`

The query asks:

- `What is the final numeric value of D?`

### Placement Pattern

- `A = B` at depth `0.12`
- `C = D` at depth `0.37`
- `A = value` at depth `0.62`
- `C = E` tail-pinned
- `B = C` tail-pinned

This means the benchmark is no longer the original tail-friendly `VT-4hop`. It is a harder, permuted, single-divergence version meant to remove the easy late-chain shortcut.

### Why We Keep It

This benchmark is still the cleanest single-answer symbolic task in the suite:

- it has a well-defined dependency chain
- it has a single correct numeric answer
- it supports hop-level attribution through the eviction logs
- it is much less forgiving than the original tail-anchored `VT-4hop`

### Primary Metrics

- `accuracy`
- `eviction_survival_rate`
- `chain_break_hop_distribution`

### What A Failure Means

- low `accuracy` with low span survival suggests genuine eviction damage
- low `accuracy` with high span survival suggests a reasoning or distractor problem rather than pure memory loss
- concentration of breaks at early or middle hops is more informative than raw wrong-answer rate alone

### Current Caveat

The benchmark should use complex, non-numeric prose filler rather than a tiny repeated sentence loop. A no-eviction sanity run should still be checked before trusting the compression curve, but the filler must remain rich enough that lower budgets cannot coast on an overly easy background.

## Benchmark 2: MQ-NIAH-4q

### Task Definition

This is the main multi-query retrieval benchmark.

Four key/value needles are hidden in the 32K context, and the query asks for all four values. Unlike VT, this task gives graded recall rather than a single binary chain success/failure.

### Placement Pattern

- needle 1 at depth `0.10`
- needle 2 at depth `0.37`
- needle 3 at depth `0.63`
- needle 4 tail-pinned

### Why We Keep It

This benchmark complements VT:

- it measures retrieval rather than symbolic chain following
- it degrades gradually instead of only as a hard chain break
- it makes partial survival visible through recall fraction
- it provides a second task family, which reduces the risk of overfitting conclusions to one benchmark shape

### Primary Metrics

- `mean_recall_fraction`
- `full_recall_rate`
- `eviction_survival_rate`

### What A Failure Means

- falling `mean_recall_fraction` with preserved tail survival usually means the model is losing earlier needles first
- a large gap between `mean_recall_fraction` and `full_recall_rate` means the task is entering a partial-recall regime rather than all-or-nothing collapse

## Reporting Template

The final averaged table should be filled from the repeated-sweep aggregate summary.

| Benchmark | Metric | k1024 | k2048 | k4096 | k8192 | k16384 | k32768 |
|---|---:|---:|---:|---:|---:|---:|---:|
| VT-4hop-permute | accuracy | TBD | TBD | TBD | TBD | TBD | TBD |
| VT-4hop-permute | eviction_survival_rate | TBD | TBD | TBD | TBD | TBD | TBD |
| MQ-NIAH-4q | mean_recall_fraction | TBD | TBD | TBD | TBD | TBD | TBD |
| MQ-NIAH-4q | full_recall_rate | TBD | TBD | TBD | TBD | TBD | TBD |
| MQ-NIAH-4q | eviction_survival_rate | TBD | TBD | TBD | TBD | TBD | TBD |

## Final Readout

The intended Phase 1 story should come from both benchmarks together:

- `VT-4hop-permute` tells us whether compressed continuation still preserves multi-hop symbolic structure
- `MQ-NIAH-4q` tells us whether recall degrades smoothly or abruptly under the same compression budgets

If both curves move in the same direction as budget shrinks, the Phase 1 result is strong.

If they disagree sharply, inspect the detailed eviction logs before drawing conclusions. In that case, the result is probably being driven by task-specific behavior rather than a clean shared memory effect.

## Current Run Notes

At the time this file was drafted:

- the main default benchmark pair in `phase1/config.py` is `vt_4hop_permute` and `mq_niah_4q`
- the repeated `5x` average sweep is in progress for `VT-4hop-permute` across `1024..32768`
- final averaged values should be copied in only after the aggregate summary is written
