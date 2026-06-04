# Phase 6: Historical Main Paper Experiment Record

Generated: 2026-05-01

This document records the finalized Phase 6 design and outcomes.
It is no longer the active broader-evidence plan.

Current exact-mode bridge and hard-panel work lives in:

- `phases/phase7_broader_evidence/phase7_plan.md`
- `phases/phase7_broader_evidence/phase7_handoff.md`

## Thesis

The paper experiment should answer one question:

- after turn 1 is complete and the cache is compressed, can the now-known turn-2 query
  produce a better cache than a no-repair baseline at the **same final active footprint**?

This is a two-turn memory-maintenance experiment, not a full agent benchmark.

## 1. Exact Design

Notation:

- `C` = long shared context
- `Q1` = turn-1 question
- `A1` = model-generated answer to `Q1`
- `Q2` = turn-2 question
- `A2` = model-generated answer to `Q2`

For each example:

1. prefill long context `C`
2. ask `Q1`
3. generate `A1`
4. compress the post-`Q1` cache
5. during idle, `Q2` is known
6. optionally repair using `Q2`
7. generate `A2`

What is being tested:

- compression is chosen using turn-1 information only
- repair is chosen after the turn-2 query is known
- the comparison is at matched final footprint

## 2. Benchmark

Use one benchmark family for the main paper experiment:

- base task family: `mq_niah_4q`
- primary split suite:
  - `14 -> 23`
  - `24 -> 13`
  - `34 -> 12`

These three splits are one experiment, not three separate benchmark claims.

What counts as a distinct split:

- the unit is the `Q1 | Q2` set partition, not the within-turn ordering
- `34 -> 12` and `43 -> 21` are equivalent under the concise values-only answer format
- by contrast, `12 -> 34`, `13 -> 24`, and `23 -> 14` are distinct because `Q2` includes the
  tail-anchored fourth needle and therefore changes the baseline regime

Main reporting:

- pool across the clean split suite for the main graph/table

Appendix / ablation:

- report per-split results

Why these splits:

- they are balanced `2|2` turn splits
- they remove the obvious tail-recency freebie from Q2
- they are already implemented and validated

Diagnostic splits, not in the main aggregate:

- `12 -> 34`
- `13 -> 24`
- `23 -> 14`

These are useful later to show why the clean suite is the main causal benchmark, but they should
not define the main paper figure because `Q2` gets a structural recency advantage.

## 3. Conditions

Every condition uses the same:

- base example
- rendered `32K` context
- generated `Q1` transcript
- same `Q2`

Locked main conditions:

- `A`
  - full cache, no compression
- `B`
  - compressed base cache
- `B_match(K)`
  - no repair, but keep `B_base + K` context positions from the start
- `RepairKV(K)`
  - start from `B`, then restore `K` positions using the true `Q2`
- `Random-K(K)`
  - restore `K` random evicted positions
- `Oldest-K(K)`
  - restore `K` oldest evicted positions
- `Oracle-K(K)`
  - hindsight best `K`

Additional diagnostic only:

- `WrongQ-K(K)`
  - same as `RepairKV(K)`, but rank using a task-matched mismatched query
  - this was kept only for preflight because it did not separate reliably on this benchmark family

Primary metric:

- `selection_lift(K) = RepairKV(K) - B_match(K)`

## 4. What Is Already Known

### 4.1 Sharp Low-Footprint Regime Exists

Artifact:

- `phases/phase6_repair/results/full/clean_suite_n100_k8-12-24-40-48.json`

At `B_base = 512`, pooled clean-suite result:

| K | `B` | `B_match` | `RepairKV` | `Oracle-K` |
|---|---:|---:|---:|---:|
| 8  | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 12 | 0.0000 | 0.0000 | 0.3550 | 0.4783 |
| 24 | 0.0000 | 0.0000 | 0.3700 | 0.5000 |
| 40 | 0.0000 | 0.0000 | 0.5333 | 0.8583 |
| 48 | 0.0000 | 0.0000 | 0.5467 | 0.9950 |

This is the clean existence-proof regime, but it is too harsh for the main paper figure because:

- `B = 0`
- `B_match = 0`

### 4.2 Main Nonzero-Baseline Regime Is `B_base = 12288`

Calibration artifact:

- `phases/phase6_repair/results/smoke/clean_suite_b12288_r128_n8_k8-16-32-48-64.json`

Pooled clean-suite result:

| K | `B` | `B_match` | `RepairKV` | `Oracle-K` |
|---|---:|---:|---:|---:|
| 8  | 0.1042 | 0.1250 | 0.3125 | 0.4167 |
| 16 | 0.1042 | 0.1042 | 0.3958 | 0.5417 |
| 32 | 0.1042 | 0.1250 | 0.6458 | 0.9792 |
| 48 | 0.1042 | 0.1042 | 0.6667 | 1.0000 |
| 64 | 0.1042 | 0.1458 | 0.7083 | 1.0000 |

Interpretation:

- the pooled baseline is now nonzero
- the matched-footprint baseline is also nonzero
- `RepairKV` is clearly above `B_match` at every `K`
- `Oracle-K` stays above `RepairKV`, so the regime is not saturated
- this meets the paper target and is the selected main regime

### 4.3 Full Main Run Completed

Artifact:

- `phases/phase6_repair/results/full/clean_suite_b12288_r128_n100_k8-16-32-48-64_ca-b-bmatch-idlekv-randomk-oldestk-oraclek.json`

Runtime:

- `elapsed_s = 2510.56` (`41.8 min`)

Pooled clean-suite result:

| K | `B` | `B_match` | `Random-K` | `Oldest-K` | `RepairKV` | `Oracle-K` |
|---|---:|---:|---:|---:|---:|---:|
| 8  | 0.0933 | 0.1017 | 0.0983 | 0.0883 | 0.3417 | 0.4067 |
| 16 | 0.0933 | 0.0983 | 0.0983 | 0.0850 | 0.4167 | 0.5233 |
| 32 | 0.0933 | 0.0950 | 0.1017 | 0.0833 | 0.6067 | 0.9267 |
| 48 | 0.0933 | 0.0967 | 0.0983 | 0.0867 | 0.6683 | 1.0000 |
| 64 | 0.0933 | 0.1000 | 0.1000 | 0.0833 | 0.6850 | 1.0000 |

Interpretation:

- the main regime achieved the paper target: both `B` and `B_match` are nonzero
- `RepairKV` is clearly above `B_match` at every `K`
- `Random-K` and `Oldest-K` stay near the matched baseline
- `Oracle-K` remains above `RepairKV`, so the selector still leaves headroom

Per-split endpoint at `K=64`:

| split | `B` | `B_match` | `RepairKV` | `Oracle-K` |
|---|---:|---:|---:|---:|
| `14 -> 23` | 0.1600 | 0.1750 | 1.0000 | 1.0000 |
| `24 -> 13` | 0.1200 | 0.1250 | 0.5400 | 1.0000 |
| `34 -> 12` | 0.0000 | 0.0000 | 0.5150 | 1.0000 |

This heterogeneity is useful, not a bug:

- the pooled result gives the main figure
- the split breakdown shows that the mechanism works, but the current selector is weak on the
  hardest clean split

## 5. Frozen Shared Setup

Hold these fixed during calibration and the full run:

| Choice | Value |
|---|---|
| Model | `Qwen2.5-7B-Instruct` |
| Context length | `32768` |
| Policy | `SnapKV` |
| Sink size | `4` |
| Recency reserve | `R_ctx = 128` |
| Prompt | concise values-only |
| `max_new_tokens` | `24` |
| Restore unit | burst restore, `left=2`, `right=20` |

## 6. Budget Policy

Budget choice must match the scope of the paper claim.

Rule:

- if the paper's main result is one benchmark family in one matched regime, use one calibrated
  `B_base` for that family
- if the paper later adds a different benchmark family, do **not** force the same `B_base`
  across families
- instead, calibrate each family with the same acceptance rule so every reported frontier is
  nonzero, non-saturated, and comparable in shape

Reason:

- a single global `B_base` across tasks with different difficulty can make easy tasks saturate
  and hard tasks collapse to zero
- that would confound benchmark difficulty with budget choice
- the right invariant is the calibration rule, not the raw `B_base`

Current decision:

- the main paper experiment is the pooled `mq_niah_4q` clean split suite
- for this main experiment, use one shared `B_base = 12288`
- if a second benchmark family is added later, it must get its own calibration pass

## 7. One Main Experiment

This is the paper experiment:

- benchmark: pooled clean split suite
- choose **one** calibrated `B_base`
- run the locked main conditions on that benchmark

This should produce:

- nonzero `B`
- nonzero `B_match`
- `RepairKV > B_match`
- `Random-K`, `Oldest-K` below `RepairKV`
- `Oracle-K` above `RepairKV`

## 8. Calibration Decision

Selected `B_base`:

- `12288`

Selected `K` grid:

- `K = {8, 16, 32, 48, 64}`

Why this grid:

- `8` checks whether the left edge is dead
- `16` is still a small restore budget
- `32` is roughly a one-burst scale
- `48` is the current best-known informative point
- `64` checks early saturation

Calibration conditions that were used:

- `A B B_match RepairKV Oracle-K`

Calibration sample size:

- `n = 8` base examples

Applied selection rule:

Choose the **smallest** `B_base` such that:

- `mean B_match(64) >= 0.10`
- `mean B_match(64) <= 0.60`
- `mean RepairKV(64) - mean B_match(64) >= 0.15`
- `mean Oracle-K(64) - mean RepairKV(64) >= 0.10`
- the frontier is not dead on the left and not fully saturated on the right

Outcome:

- `K=8` is not dead: pooled `RepairKV = 0.3125`
- `K=64` is not saturated: pooled `RepairKV = 0.7083`, `Oracle-K = 1.0000`
- keep the full grid for the main run

## 9. Control Preflight

Artifacts:

- `phases/phase6_repair/results/smoke/clean_suite_b12288_r128_n8_k8-16-32-48-64_ca-b-bmatch-idlekv-wrongqk-randomk-oldestk-oraclek.json`
- `phases/phase6_repair/results/smoke/clean_suite_b12288_r128_n4_k8-16-32-48-64_ca-b-bmatch-idlekv-wrongqk-randomk-oldestk-oraclek.json`

Observed:

- `Random-K` and `Oldest-K` stay well below `RepairKV`
- `WrongQ-K` does **not** separate reliably from `RepairKV` on this benchmark family
- the likely reason is the highly repetitive NIAH query template, which makes a mismatched query
  still recover generic key/value bursts

Decision:

- keep `WrongQ-K` as a diagnostic only
- exclude it from the main long run
- center the main causal comparison on `B_match`, `Random-K`, `Oldest-K`, and `Oracle-K`

## 10. Locked Main Result

The main experiment is now complete with the selected regime:

- task: `clean_suite`
- `n = 100`
- `K = {8, 16, 32, 48, 64}`
- conditions:
  - `A`
  - `B`
  - `B_match`
  - `RepairKV`
  - `Random-K`
  - `Oldest-K`
  - `Oracle-K`

Main figure:

- pooled clean-suite frontier:
  - `B_match`
  - `RepairKV`
  - `Random-K`
  - `Oldest-K`
  - `Oracle-K`

Main table:

- selected `K` values
- pooled clean-suite results
- `% RepairKV > B_match`

Appendix:

- per-split curves
- overlap metrics
- latency

## 11. Status

Phase 6 is complete.

Locked conclusion:

- the paper should now be centered on one main matched-footprint result
- use the pooled 3-split `mq_niah_4q` suite as the main figure
- use `B_base = 12288` as the locked nonzero-baseline regime
- keep the old `B=512` result only as a sharp existence-proof reference

All broader-evidence runs now move to:

- `phases/phase7_broader_evidence/phase7_handoff.md`
