# Phase 7: Broader Same-Family Evidence

Generated: 2026-05-01

## Goal

Phase 6 already established the core claim on `mq_niah_4q`:

- at the same final cache footprint,
- future-query-informed repair beats matched no-repair retention.

Phase 7 should broaden that evidence without changing the core mechanism:

- stay in the same `mq_niah_*` family,
- keep `SnapKV` as the only compressor,
- use the stronger exact-Q scorer,
- and show one harder task, not just one tuned `4q` result.

Because the scorer changed, every Phase 7 panel should be method-consistent end to end.
Do **not** splice older proxy-scored or narrower-`K` results into the final exact-mode graphs.

Phase 7 does **not** replace the locked Phase 6 main result.

## Locked Phase 6 Result

Earlier low-footprint Phase 6 artifact:

- `phases/phase6_repair/results/full/clean_suite_b12288_r128_n100_k8-16-32-48-64_ca-b-bmatch-idlekv-randomk-oldestk-oraclek.json`

Locked pooled numbers:

| K | B_match | RepairKV | Random-K | Oldest-K | Oracle-K |
|---|---:|---:|---:|---:|---:|
| 8  | 0.102 | 0.342 | 0.098 | 0.088 | 0.407 |
| 16 | 0.098 | 0.417 | 0.098 | 0.085 | 0.523 |
| 32 | 0.095 | 0.607 | 0.102 | 0.083 | 0.927 |
| 48 | 0.097 | 0.668 | 0.098 | 0.087 | 1.000 |
| 64 | 0.100 | 0.685 | 0.100 | 0.083 | 1.000 |

Keep this as earlier sharp low-footprint evidence, not as the current exact
bridge panel used for the main broader-evidence figures.

## What Changed Before Phase 7

The old Phase 7 draft is superseded.

Two important method upgrades landed first:

1. The repair scorer can now use **exact extracted Q rows**:
   - real `q_proj`
   - post-RoPE
   - scored against active + evicted keys together
   - specifically, this is an exact **question-query** scorer over the
     `Q_2` prompt tokens, not literal decode-time answer-token attention
2. The hindsight ceiling can now use a **gold-span oracle mode**:
   - enumerates the small set of benchmark gold span-group subsets
   - evaluates each subset under actual `Q2` generation once
   - chooses the best subset whose total span cost is `<= K`

Important wording:

- `Oracle-K` in Phase 7 should be described as a **gold-span hindsight ceiling**.
- It is exact over the benchmark’s gold span-group subsets under budget, using actual `Q2` generation to score those candidates.
- It is **not** an exhaustive optimum over all arbitrary token subsets maximizing final generated task score.
- `A` should be described as the **full-cache reference condition**, not a literal ceiling:
  on harder settings a repaired smaller cache can occasionally outperform the
  full cache by removing distracting context.
  A small number of examples with `RepairKV > A` would therefore be a modeling
  effect to interpret, not an automatic runner bug.

## Why 3, 4, and 6

- `2q` was rejected because a `1 -> 1` split is too binary and gives poor frontier resolution.
- `8q` was rejected because it adds much longer answers, more truncation risk, more split arbitrariness, and more compute before `6q` is even locked.
- `4q` is the validated bridge panel and `6q` is the best harder
  same-family extension.
- `3q` is optional appendix-only sanity evidence.

Use them as:

- `3q`: light sanity panel
- `4q`: bridge panel
- `6q`: hard panel

Do **not** pool these into one difficulty average.

## Why SnapKV Only

Use **SnapKV only** in Phase 7.

Reason:

- the open question here is task breadth, not compressor breadth
- SnapKV is already the best-supported content-aware compressor in the repo
- adding a second compressor now would multiply variables without clarifying the hypothesis

## Stress-Tested Design Conclusions

These are the important audit outcomes from the smokes:

1. A single shared `B_base` across the optional `3q` sanity panel, the
   `4q` bridge panel, and the harder `6q` panel is **not** the best
   exact-mode design.
2. `mq_niah_3q` works, but it is not strong enough to carry the broader-evidence story by itself.
3. `mq_niah_6q` is the right harder extension, but it needs a larger `B_base` than `4q`.
4. `6q` is not a perfect matched-difficulty control for `4q`.
   - It also changes turn lengths:
     - more values in `Q_1`
     - more values in `Q_2`
     - a larger values-only decode budget (`48` tokens instead of `24`)
   - So it should be presented as a calibrated harder same-family extension,
     not as a single-axis difficulty knob or independent confirmatory test.
5. Mixed `3q / 4q / 6q` runs cannot be executed in one runner
   invocation. They must be run as separate jobs and compared offline.
6. The remaining `RepairKV < Oracle-K` gap on `4q` is **not** mainly a missing-attention-signal problem. Under the exact scorer, many gold Q2 tokens already rank near the top of the evicted list; the main bottleneck is the current burst restore policy wasting budget around false-positive anchors and fragmented neighborhoods.
7. The widened high-`K` `4q` run is a **diagnostic**, not the final bridge-panel artifact, because it omits the low-`K` exact points. The final `4q` exact panel must come from one unified rerun.
8. A stronger `6q` design is now available than the original single split.
   - Define the clean suite as the complete family of balanced `3|3` partitions
     where turn 2 excludes the two latest needles `5,6`.
   - In the generator, only needle `6` is truly tail-anchored; needle `5` is a
     late body insertion at depth `0.72`.
   - Concretely, this yields exactly four partitions:
     - `1,5,6 -> 2,3,4`
     - `2,5,6 -> 1,3,4`
     - `3,5,6 -> 1,2,4`
     - `4,5,6 -> 1,2,3`
   - This is a cleaner harder analogue of the `4q` clean split suite than using
     only `4,5,6 -> 1,2,3`.
9. The exact-Q scorer is a stronger method variant than the old proxy path. Quality evidence can be compared directly, but Phase 4 latency numbers should not be reused as exact scorer latency claims without a separate timing pass.
10. The current aggregated `mean_idlekv_repair_ms` in Phase 6/7 artifacts is **partial** runtime.
   - It includes selection + transfer + inject.
   - It does **not** include exact-Q extraction or evicted scoring.
   - So the final systems/runtime figure must be computed separately from the per-row timing fields, not copied directly from the built-in aggregate.
11. The current exact-Q runner is a **quality-first evaluation path**, not yet a production-quality systems path.
   - In the existing implementation, exact evicted scoring is much slower than the old proxy path.
   - So exact-Q results are strong evidence about signal quality and matched-footprint recovery.
   - They are **not yet** sufficient on their own for a “sub-second exact runtime” systems claim without either:
     - integrating the precomputed CPU eviction buffer into the exact scorer path, or
     - moving exact scoring to a faster implementation.
12. Frontier export now supports deterministic bootstrap confidence bounds.
   - The exported overall/by-split CSVs include `*_lo` / `*_hi` columns for the
     score curves used in the paper figures.
   - This makes it straightforward to add uncertainty bands without recomputing
     one-off notebook summaries later.
13. The overlap exporter was patched to prefer **final active overlap** for
    repair conditions.
   - Earlier overlap CSVs could silently mix final-kept overlap for `B_match`
     with restored-only overlap for `RepairKV`, `Random-K`, `Oldest-K`, and
     `Oracle-K` when legacy `*_overlap_fraction` fields were present.
   - Re-exported Phase 7 artifacts now reconstruct final active overlap from
     `b_kept_context_positions ∪ selected_positions` when needed.

So the final Phase 7 suite is **per-task calibrated**, not forced into one shared `B_base`.

## Smoke Calibration Summary

Relevant code/test status before full runs:

- exact-Q scorer path implemented
- gold-span oracle mode implemented
- critical CPU-side suite green on the current code:
  - 53 tests passed across the NIAH generator, SnapKV/eviction invariants,
    eviction-buffer core, and Phase 6 protocol / runner / selector /
    reporting paths

### Task A: `mq_niah_3q` light panel

Artifact:

- `phases/phase6_repair/results/smoke/mq_niah_3q_split_3_to_12_b14336_r128_qexact_q_ogold_spans_n4_k8-16-32-48-64_ca-b-bmatch-idlekv-oraclek.json`

Config:

- split: turn 1 asks `3`, turn 2 asks `1,2`
- `B_base = 14336`
- `R_ctx = 128`
- `K = {8, 16, 32, 48, 64}`

Aggregate:

| K | B_match | RepairKV | Oracle-K |
|---|---:|---:|---:|
| 8  | 0.250 | 0.250 | 0.375 |
| 16 | 0.250 | 0.375 | 0.500 |
| 32 | 0.375 | 0.250 | 0.875 |
| 48 | 0.250 | 0.375 | 1.000 |
| 64 | 0.250 | 0.750 | 1.000 |

Interpretation:

- usable as a light sanity panel
- not a strong headline frontier
- keep as optional / appendix if the main Phase 7 suite already looks strong

### Task B: `mq_niah_4q` bridge panel

Artifact:

- `phases/phase6_repair/results/smoke/clean_suite_b16384_r128_qexact_q_ogold_spans_n4_k8-16-32-48-64_ca-b-bmatch-idlekv-oraclek.json`

Config:

- pooled clean split suite:
  - `1,4 -> 2,3`
  - `2,4 -> 1,3`
  - `3,4 -> 1,2`
- `B_base = 16384`
- `R_ctx = 128`
- `K = {8, 16, 32, 48, 64}`

Aggregate:

| K | B_match | RepairKV | Oracle-K |
|---|---:|---:|---:|
| 8  | 0.125 | 0.125 | 0.417 |
| 16 | 0.125 | 0.125 | 0.625 |
| 32 | 0.167 | 0.458 | 0.917 |
| 48 | 0.167 | 0.500 | 1.000 |
| 64 | 0.167 | 0.583 | 1.000 |

Interpretation:

- better exact-mode bridge panel than `B=14336`
- nonzero matched baseline throughout the sweep
- still clearly below oracle, so the task is not saturated

### Task C: `mq_niah_6q` hard panel

Tested budgets:

| B_base | K=8 B_match / RepairKV | K=32 B_match / RepairKV | K=64 B_match / RepairKV | Verdict |
|---|---:|---:|---:|---|
| 14336 | 0.000 / 0.083 | 0.083 / 0.083 | 0.083 / 0.167 | too harsh |
| 16384 | 0.167 / 0.167 | 0.167 / 0.167 | 0.167 / 0.500 | usable but weak |
| 18432 | 0.250 / 0.250 | 0.250 / 0.333 | 0.250 / 0.667 | best calibrated |

Budget-probe artifact:

- `phases/phase6_repair/results/smoke/mq_niah_6q_split_456_to_123_b18432_r128_qexact_q_ogold_spans_n4_k8-16-32-48-64_ca-b-bmatch-idlekv-oraclek.json`

Budget-probe config:

- split: turn 1 asks `4,5,6`, turn 2 asks `1,2,3`
- `B_base = 18432`
- `R_ctx = 128`
- `K = {8, 16, 32, 48, 64}`

Budget-probe aggregate:

| K | B_match | RepairKV | Oracle-K |
|---|---:|---:|---:|
| 8  | 0.250 | 0.250 | 0.333 |
| 16 | 0.250 | 0.250 | 0.500 |
| 32 | 0.250 | 0.333 | 0.667 |
| 48 | 0.250 | 0.583 | 1.000 |
| 64 | 0.250 | 0.667 | 1.000 |

Interpretation:

- this is the first hard-task budget that gives a clean nonzero matched baseline and a useful rise
- this is the right starting budget for the harder `6q` panel
- but the final Phase 7 hard panel should still be the new `mq_niah_6q_clean_suite`, not this single split

Clean-suite smoke artifact:

- `phases/phase6_repair/results/smoke/mq_niah_6q_clean_suite_b18432_r128_qexact_q_ogold_spans_n4_k8-16-24-32-48-64-80-96-128_ca-b-bmatch-idlekv-randomk-oldestk-oraclek.json`

Clean-suite smoke aggregate:

| K | A | B | B_match | RepairKV | Random-K | Oldest-K | Oracle-K |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8   | 0.979 | 0.312 | 0.312 | 0.375 | 0.354 | 0.354 | 0.500 |
| 16  | 0.979 | 0.312 | 0.333 | 0.333 | 0.312 | 0.292 | 0.562 |
| 24  | 0.979 | 0.312 | 0.354 | 0.438 | 0.312 | 0.312 | 0.833 |
| 32  | 0.979 | 0.312 | 0.312 | 0.458 | 0.312 | 0.333 | 0.833 |
| 48  | 0.979 | 0.312 | 0.312 | 0.583 | 0.333 | 0.354 | 1.000 |
| 64  | 0.979 | 0.312 | 0.354 | 0.771 | 0.333 | 0.354 | 1.000 |
| 80  | 0.979 | 0.312 | 0.354 | 0.938 | 0.312 | 0.312 | 1.000 |
| 96  | 0.979 | 0.312 | 0.354 | 0.979 | 0.396 | 0.312 | 1.000 |
| 128 | 0.979 | 0.312 | 0.354 | 0.979 | 0.354 | 0.312 | 1.000 |

Interpretation:

- the clean-suite gate passed
- `B_base = 18432` is acceptable for the harder panel
- the matched baseline is nonzero overall and the control ordering still holds
- one full-cache example out of 16 scored below `1.0`, so the hard panel should be described as a near-perfect full-cache-reference setting rather than a perfect one
- under the rewritten oracle, the hard panel still has a clean nonzero matched baseline and a useful rise through `K=128`
- the only notable warning is minor example-level non-monotonicity in `RepairKV`, which is expected under greedy generation and does not invalidate the aggregate trend
- the accepted clean-suite splits are the balanced `3 -> 3` partitions whose
  turn-2 side excludes the tail-favored sixth needle:
  - `156 -> 234`
  - `256 -> 134`
  - `356 -> 124`
  - `456 -> 123`
- these are therefore the right `6q` analogue of the recency-clean `4q`
  suite; the complementary partitions with `6` in turn 2 should be treated as
  recency-leaky diagnostics, not as the primary harder panel

### Tiny control check on the chosen main settings

Artifacts:

- `phases/phase6_repair/results/smoke/clean_suite_b16384_r128_qexact_q_ogold_spans_n2_k8-16-32-48-64_ca-b-bmatch-idlekv-randomk-oldestk-oraclek.json`
- `phases/phase6_repair/results/smoke/mq_niah_6q_split_456_to_123_b18432_r128_qexact_q_ogold_spans_n2_k8-16-32-48-64_ca-b-bmatch-idlekv-randomk-oldestk-oraclek.json`

Summary:

- `4q @ 16384`:
  - `Random-K` stays on `B_match`
  - `Oldest-K` stays at or below `B_match`
- `6q @ 18432`:
  - `Random-K` stays on `B_match`
  - `Oldest-K` stays at or below `B_match`

Takeaway:

- the exact-mode main settings preserve the intended control ordering
- the `4q` bridge panel is ready for full runs
- the `6q` hard panel passed the clean-suite smoke gate
- the widened `6q` axis through `K=128` is now accepted

## New Exact-Mode Diagnostics

These diagnostics were run after the exact scorer landed and matter for how Phase 7 should be presented.

### 1. The current `4q` bridge sweep needs a wider K axis

Artifact:

- `phases/phase6_repair/results/smoke/clean_suite_b16384_r128_qexact_q_ogold_spans_n4_k32-48-64-80-96-128_ca-b-bmatch-idlekv-oraclek.json`

Aggregate:

| K | B_match | RepairKV | Oracle-K |
|---|---:|---:|---:|
| 32  | 0.167 | 0.458 | 0.917 |
| 48  | 0.167 | 0.500 | 1.000 |
| 64  | 0.167 | 0.583 | 1.000 |
| 80  | 0.125 | 0.750 | 1.000 |
| 96  | 0.167 | 0.833 | 1.000 |
| 128 | 0.125 | 1.000 | 1.000 |

Interpretation:

- `K <= 64` understates what the exact scorer can do on `4q`
- the widened `K` axis produces the nicer bridge-panel graph we actually want
- `RepairKV` reaches the gold-span oracle by `K = 128` on this smoke

### 2. The hardest split catches up with enough budget

Artifact:

- `phases/phase6_repair/results/smoke/mq_niah_4q_split_34_to_12_b16384_r128_qexact_q_ogold_spans_n4_k32-48-64-80-96-128_ca-b-bmatch-idlekv-oraclek.json`

Aggregate:

| K | B_match | RepairKV | Oracle-K |
|---|---:|---:|---:|
| 32  | 0.000 | 0.125 | 0.750 |
| 48  | 0.000 | 0.250 | 1.000 |
| 64  | 0.000 | 0.375 | 1.000 |
| 80  | 0.000 | 0.500 | 1.000 |
| 96  | 0.000 | 0.750 | 1.000 |
| 128 | 0.000 | 1.000 | 1.000 |

Interpretation:

- the exact scorer is not fundamentally capped below oracle
- on the hard split, the remaining gap is largely budget efficiency

### 3. Exact question-query scoring already ranks many gold tokens highly

Hard-split rank diagnostic, averaged over 4 smoke examples:

- mean relevant tokens inside top-64 evicted scores: `26`
- mean relevant tokens inside top-128 evicted scores: `29`
- median relevant-token rank: `38`
- mean relevant/nonrelevant score ratio: about `82x`

Interpretation:

- the exact question-query signal is already useful
- the current restore policy is leaving performance on the table after ranking
- the right future algorithm work is more likely a better span/burst policy than a totally different query signal

### 4. Large-K random restore can creep up slightly

Artifact:

- `phases/phase6_repair/results/smoke/clean_suite_b16384_r128_qexact_q_ogold_spans_n2_k80-96-128_ca-b-bmatch-idlekv-randomk-oldestk-oraclek.json`

Aggregate:

| K | B_match | RepairKV | Random-K | Oldest-K | Oracle-K |
|---|---:|---:|---:|---:|---:|
| 80  | 0.250 | 0.750 | 0.250 | 0.167 | 1.000 |
| 96  | 0.250 | 0.833 | 0.417 | 0.167 | 1.000 |
| 128 | 0.250 | 1.000 | 0.333 | 0.167 | 1.000 |

Interpretation:

- random restore can begin to recover some useful context at very large `K`
- but it still stays below `RepairKV`
- this means the final paper plot should still show `Random-K`, especially once the `K` axis extends past `64`

### 5. Full high-K exact `4q` diagnostic confirmed the bridge setting

Artifact:

- `phases/phase6_repair/results/full/clean_suite_b16384_r128_qexact_q_ogold_spans_n100_k32-48-64-80-96-128_ca-b-bmatch-idlekv-randomk-oldestk-oraclek.json`

Overall aggregate:

| K | B | B_match | RepairKV | Random-K | Oldest-K | Oracle-K | select+transfer+inject ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32  | 0.243 | 0.252 | 0.365 | 0.247 | 0.223 | 0.877 | 18.57 |
| 48  | 0.243 | 0.243 | 0.535 | 0.240 | 0.225 | 1.000 | 16.28 |
| 64  | 0.243 | 0.248 | 0.660 | 0.242 | 0.222 | 1.000 | 15.93 |
| 80  | 0.243 | 0.240 | 0.795 | 0.248 | 0.217 | 1.000 | 15.40 |
| 96  | 0.243 | 0.245 | 0.910 | 0.255 | 0.228 | 1.000 | 16.74 |
| 128 | 0.243 | 0.243 | 0.998 | 0.253 | 0.223 | 1.000 | 15.72 |

Interpretation:

- `B_base = 16384` is confirmed as the right exact-mode `4q` bridge setting
- the widened `K` axis materially improves the graph
- `RepairKV` nearly reaches `1.0` at `K = 128` on the full run, not just on smoke
- `Random-K` stays close to `B_match` even at large `K`
- the exact bridge result is now locked as the main Phase 7 `4q` quality panel
- the re-exported overlap diagnostic is now usable as a mechanism figure:
  on the finalized exact `4q` bridge run, `RepairKV` reaches `0.998` while
  final-active overlap rises only from about `0.363` to `0.794`, which is
  exactly the expected “good signal, imperfect packing” pattern
- exact total runtime on this bridge run is almost flat across `K`
  (`\approx 6.38`\,s p50, `\approx 6.49`\,s p95), with
  `\approx 6.29`\,s spent in evicted scoring and well under `100`\,ms in
  query extraction plus transfer/injection, so the current exact scorer should
  be treated as a quality-evidence path rather than the final systems path

## Additional Failure Modes To Watch

These are the remaining realistic ways Phase 7 can mislead us even if the code is correct.

1. `4q @ 16384` may still be too heterogeneous at the split level.
   - The easiest split is already near saturation on many examples.
   - The hardest split still needs much more `K`.
   - So the pooled `4q` exact curve is useful, but the paper should still show per-split appendix tables or plots.

2. The widened exact `4q` graph can become visually misleading if `Random-K` is omitted.
   - At large `K`, random restore can recover some useful context by chance.
   - That does not kill the result, but it means the large-`K` regime needs the control curve on the plot.

3. `6q @ 18432` is accepted, but it is still a heterogeneous harder panel.
   - Some splits keep a strong matched baseline, while others remain much harsher.
   - The pooled curve is still useful, but the paper should keep per-split appendix results and should describe `6q` as a harder same-family extension, not a perfectly matched smooth-difficulty control.
   - The full-cache reference condition is near-perfect rather than perfect (`A \approx 0.99` in the smoke), so the harder panel should not be narrated as a fully solved task under the full cache.

4. `3q` is structurally different, not just easier.
   - It uses a `1 -> 2` split, while `4q` uses `2 -> 2` and `6q` uses `3 -> 3`.
   - So `3q` should stay optional / appendix unless we specifically want a light sanity panel.

5. `Oracle-K` remains a gold-span hindsight ceiling, not a literal maximum over arbitrary token subsets and final generations.
   - It is the right diagnostic headroom curve for this benchmark.
   - It should not be described as a globally exact task-score oracle.

6. Exact-Q ranking is not the same thing as an optimal restore policy.
   - The diagnostics already show that many relevant tokens rank highly.
   - The remaining gap is mostly a span construction / packing problem.
   - So if `RepairKV < Oracle-K` persists at moderate `K`, that does not by itself falsify the attention-based hypothesis.

7. The built-in runtime aggregate is easy to misuse.
   - `mean_idlekv_repair_ms` currently excludes query extraction and evicted scoring.
   - If we report runtime, we must explicitly aggregate:
     - `q2_query_rows_s`
     - `q2_evicted_scoring_s`
     - `idlekv_selection_s`
     - `idlekv_transfer_ms`
     - `idlekv_inject_ms`
   - Otherwise the runtime figure will understate the real exact-mode repair path.

8. Exact-Q quality success does not automatically imply exact-Q runtime feasibility in the current runner.
   - The broad quality runs are still worth doing.
   - But the paper must distinguish:
     - quality evidence for the stronger scorer
     - from systems evidence for a practical idle-window implementation
9. Older overlap diagnostics are easy to misread.
   - Earlier exact artifacts mixed two different semantics:
     - full kept-context overlap for `B_match`
     - restored-only overlap for `RepairKV` and the restore controls
   - The runner now records explicit **final active overlap** fields for future runs.
   - The reporting layer can also reconstruct final active overlap from older artifacts when they still contain:
     - `b_kept_context_positions`
     - `*_selected_positions`
     - `q2_relevant_positions`
   - So the current unified exact `4q` rerun remains usable for a mechanism/overlap figure.
   - But do not compare older overlap CSVs across conditions unless they have been regenerated through the patched exporter.
   - Also, final active overlap is only a rough mechanism diagnostic, not a surrogate for task score.
     On the finalized exact `4q` bridge run, `RepairKV` reaches nearly perfect
     task score by `K=128` while final active overlap remains well below `1.0`,
     because answering the values-only NIAH query does not require recovering
     every annotated gold token in each needle span.

## Final Phase 7 Suite

### Main Phase 7 panels

Run these two panels by default:

1. `mq_niah_4q` clean split suite
   - `B_base = 16384`
   - `R_ctx = 128`
2. `mq_niah_6q` clean suite candidate
   - tasks:
     - `mq_niah_6q_split_156_to_234`
     - `mq_niah_6q_split_256_to_134`
     - `mq_niah_6q_split_356_to_124`
     - `mq_niah_6q_split_456_to_123`
   - runner alias: `mq_niah_6q_clean_suite`
   - initial `B_base = 18432`
   - `R_ctx = 128`

Shared settings:

- `K = {8, 16, 32, 48, 64}` for `3q`
- `K = {8, 16, 24, 32, 48, 64, 80, 96, 128}` for the final exact `4q` and `6q` panels
- conditions:
  - `A`
  - `B`
  - `B_match`
  - `RepairKV`
  - `Random-K`
  - `Oldest-K`
  - `Oracle-K`
- `query_scoring_mode = exact_q`
- `oracle_mode = gold_spans`

### Optional light panel

Only if we want a third panel:

- `mq_niah_3q_split_3_to_12`
- `B_base = 14336`
- same `R_ctx`, `K`, and conditions

## Success Criteria

For each task panel, approve the full run only if:

- `B_match` is nonzero somewhere in the sweep
- `RepairKV > B_match` at moderate or high `K`
- `Random-K` and `Oldest-K` stay near `B_match`
- `Oracle-K > RepairKV`

If a task fails those criteria at smoke scale, do not promote it to a full run.

## Run Status

### Unified exact `4q` bridge run: completed

Artifact:

- `phases/phase6_repair/results/full/clean_suite_b16384_r128_qexact_q_ogold_spans_n100_k8-16-24-32-48-64-80-96-128_ca-b-bmatch-idlekv-randomk-oldestk-oraclek.json`

This is now the locked exact `4q` bridge panel.
The old `pre_oracle_rewrite` copy is numerically identical and should be treated
as a redundant reference, not as a competing result line.

Final pooled exact `4q` numbers:

| K | B | B_match | RepairKV | Random-K | Oldest-K | Oracle-K |
|---|---:|---:|---:|---:|---:|---:|
| 8   | 0.243 | 0.240 | 0.240 | 0.240 | 0.223 | 0.442 |
| 16  | 0.243 | 0.247 | 0.247 | 0.243 | 0.225 | 0.580 |
| 24  | 0.243 | 0.247 | 0.298 | 0.243 | 0.223 | 0.857 |
| 32  | 0.243 | 0.252 | 0.365 | 0.247 | 0.223 | 0.877 |
| 48  | 0.243 | 0.243 | 0.535 | 0.240 | 0.225 | 1.000 |
| 64  | 0.243 | 0.248 | 0.660 | 0.242 | 0.222 | 1.000 |
| 80  | 0.243 | 0.240 | 0.795 | 0.248 | 0.217 | 1.000 |
| 96  | 0.243 | 0.245 | 0.910 | 0.255 | 0.228 | 1.000 |
| 128 | 0.243 | 0.243 | 0.998 | 0.253 | 0.223 | 1.000 |

Audit summary:

- `A = 1.0` throughout
- `RepairKV > B_match` on `94.3%` of examples at `K = 128`
- `Random-K` stays on the matched baseline
- `Oldest-K` stays below baseline
- example-level non-monotonicity exists but is low for `Oracle-K` (`4 / 300`) and
  reflects downstream generation instability, not a selector bug

### Current active priority: launch the final exact `6q` hard panel

The sequence is now:

1. the exact `4q` bridge panel is frozen
2. the post-rewrite `6q` clean-suite smoke has already passed at `B = 18432`
3. the widened-`K` `6q` smoke also passed, so the hard panel should extend through `K = 128`
4. the current GPU run is the final exact `6q` hard panel

#### Accepted post-rewrite `6q` smoke state

- task: `mq_niah_6q_clean_suite`
- `B_base = 18432`
- `R_ctx = 128`
- `K = {8, 16, 24, 32, 48, 64, 80, 96, 128}`
- same exact scorer, gold-span oracle, and matched-footprint controls
- smoke scale: `n = 4`

What it established:

- the harder `6q` panel does not need to fall back to one hand-picked split
- one shared `B = 18432` is usable across the four recency-clean `6q` splits
- the widened `K` axis through `128` is worth keeping
- the post-rewrite oracle behaves cleanly enough to promote the full hard panel

#### Immediate order of operations

1. continue the final exact `6q` hard panel
2. audit/postprocess the finished `6q` artifact
3. update paper figures/tables so the broader-evidence package is `4q + 6q`
4. aggregate exact-path runtime from the exported per-row timing fields, not
   from the old transfer-only feasibility table and not from the partial
   built-in `mean_idlekv_repair_ms` summary alone
5. explicitly verify after the run that:
   - `B_match` stays nonzero overall
   - `RepairKV > B_match` overall
   - `Random-K` and `Oldest-K` stay near baseline
   - `idlekv_restored_count == K` across rows
   - `random_k_restored_count == K` across rows
   - `oldest_k_restored_count == K` across rows
   - `oracle_k_restored_count <= K` across rows in `gold_spans` mode
   - there is no systematic `RepairKV > A` pattern

### Full `6q` panel

The accepted full harder-panel run is:

- `mq_niah_6q_clean_suite`
- `B_base = 18432`
- `R_ctx = 128`
- `K = {8, 16, 24, 32, 48, 64, 80, 96, 128}`
- same exact scorer, gold-span oracle, and controls

This should be treated as a harder same-family suite panel, not as a pooled
family average across unrelated tasks or as a single-axis difficulty control.
The right justification for `B_base = 18432` is not aesthetics; it is the
smallest calibrated hard-task budget where the matched no-repair baseline is
nonzero overall, `RepairKV` already lifts above it, and the gold-span oracle
still leaves visible headroom.

#### Acceptance rule for the finished `6q` panel

Promote the completed `6q` run directly into the paper package if all of
the following hold:

- pooled full-cache reference `A` is at least `0.95`;
- pooled `B_match` is at least `0.10` at the right edge of the sweep;
- pooled lift `RepairKV - B_match` is at least `0.10` at `K=48` and at
  least `0.20` at `K=96`;
- pooled `Random-K` and `Oldest-K` are each at least `0.10` below
  pooled `RepairKV` at `K=96`;
- pooled `Oracle-K - RepairKV` is at least `0.05` at `K=48`;
- the bootstrap lower bound for pooled `RepairKV` at `K=96` exceeds the
  bootstrap upper bound for pooled `B_match` at `K=96`; and
- every individual `6q` split has pooled `RepairKV - B_match > 0` at
  `K=96`; and
- every individual `6q` split has pooled `RepairKV - B_match >= 0.10` at
  `K=128`; and
- every individual `6q` split has pooled `RepairKV - Random-K > 0` and
  `RepairKV - Oldest-K > 0` at `K=128`.

If the finished `6q` run instead looks too harsh or too flat overall, the
first fallback is **not** a task change. The first fallback is one calibrated
rerun at a slightly larger base budget (e.g. `B_base = 20480`) with the same
split family and `K` grid.

### What not to run next

- Do **not** prioritize the optional `3q` full panel before the active `6q`
  hard panel is finished and audited.
- Do **not** launch a second compressor before the exact `4q` and `6q` panels are locked.
- Do **not** mix the old Phase 6 `4q @ 12288` numbers with new exact-mode `4q @ 16384` points on one curve.

## Full-Run Commands

## Post-Run Tools

After each completed artifact, use:

- `phases/phase6_repair/scripts/export_phase6_frontier.py`
  - writes overall/by-split frontier CSVs
  - writes exact runtime CSVs
  - writes active-overlap CSVs
- `phases/phase7_broader_evidence/scripts/audit_phase7_artifact.py`
  - prints aggregate curve shape
  - reports per-example monotonicity rates
  - prints the predeclared non-CI `6q` acceptance checks directly for
    `mq_niah_6q_clean_suite` artifacts
  - when given the exported overall CSV, also prints the bootstrap CI
    gate at `K=96`
  - is safe on both single-task and suite artifacts
- `phases/phase7_broader_evidence/scripts/summarize_runtime.py`
  - if a compact latency-only CSV is needed from one finished artifact
- `phases/phase7_broader_evidence/scripts/finalize_phase7_panels.sh`
  - reruns the standard postprocess flow for the frozen exact `4q` artifact
    and one finished `6q` artifact
  - writes the reduced-series main-text SVGs for both panels in one shot

These should be part of the standard Phase 7 run loop, not an afterthought.

### 1. Unified exact `mq_niah_4q` bridge rerun: completed reference

```bash
.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
  --stage full \
  --task clean_suite \
  --num-samples 100 \
  --base-context-budget 16384 \
  --recency-window 128 \
  --k 8 16 24 32 48 64 80 96 128 \
  --conditions A B B_match RepairKV Random-K Oldest-K Oracle-K \
  --query-scoring-mode exact_q \
  --oracle-mode gold_spans
```

### 2. `mq_niah_6q` hard panel: active full run

```bash
.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
  --stage full \
  --task mq_niah_6q_clean_suite \
  --num-samples 100 \
  --base-context-budget 18432 \
  --recency-window 128 \
  --k 8 16 24 32 48 64 80 96 128 \
  --conditions A B B_match RepairKV Random-K Oldest-K Oracle-K \
  --query-scoring-mode exact_q \
  --oracle-mode gold_spans
```

### 3. Optional `mq_niah_3q` light panel

```bash
.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
  --stage full \
  --task mq_niah_3q_split_3_to_12 \
  --num-samples 100 \
  --base-context-budget 14336 \
  --recency-window 128 \
  --k 8 16 32 48 64 \
  --conditions A B B_match RepairKV Random-K Oldest-K Oracle-K \
  --query-scoring-mode exact_q \
  --oracle-mode gold_spans
```

## Reporting Shape For The Paper

Recommended presentation:

- keep the locked `4q @ 12288` Phase 6 result as the earlier sharp low-footprint evidence
- use the unified exact `4q @ 16384` bridge curve as the main broader-evidence quality panel
- add one `6q` hard panel
- optionally add one light `3q` appendix panel

Do **not** average the optional `3q` sanity panel, the `4q` bridge
panel, and the harder `6q` panel into one frontier.

Show them as separate panels with task-specific `B_base`.
For the final exact `4q` plot, keep the widened `K` axis through `128`.
For `6q`, keep the widened `K` axis through `128`.

## Deferred Work

The streaming strict-cap experiment has been moved out of Phase 7:

- `phases/phase8_streaming_strict_cap/phase8_plan.md`
