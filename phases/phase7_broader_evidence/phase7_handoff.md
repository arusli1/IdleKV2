# Phase 7 Handoff

## What Is Locked

Phase 6 is done.

Earlier low-footprint Phase 6 artifact:

- `phases/phase6_repair/results/full/clean_suite_b12288_r128_n100_k8-16-32-48-64_ca-b-bmatch-idlekv-randomk-oldestk-oraclek.json`

Current exact-mode bridge artifact:

- `phases/phase6_repair/results/full/clean_suite_b16384_r128_qexact_q_ogold_spans_n100_k8-16-24-32-48-64-80-96-128_ca-b-bmatch-idlekv-randomk-oldestk-oraclek.json`

This is now the locked exact `4q` bridge panel.
The redundant `pre_oracle_rewrite` copy is numerically identical and should be
treated as an archival reference, not as a competing result.

Main claim supported:

- on `mq_niah_4q`, future-query repair beats a matched no-repair baseline at the same final footprint

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

Current code health after the latest exact-mode and reporting patches:

- targeted Phase 6 + broader supporting suites: `53` tests passed
- this includes:
  - NIAH generator
  - eviction policies and stress tests
  - eviction buffer
  - Phase 6 protocol
  - Phase 6 runner
  - selectors
  - reporting/export helpers

## What Changed After Phase 6

Two important method upgrades are now in place:

1. **Exact-Q scoring**
   - real extracted post-RoPE `Q` rows
   - scored against active + evicted keys together
   - this is an exact **question-query** scorer over the `Q_2` prompt
     tokens, not literal decode-time answer-token attention
2. **Gold-span oracle mode**
   - exact hindsight search over the benchmark’s small set of gold span-group subsets
   - each subset is evaluated once under actual `Q2` generation
   - stronger than the old burst-constrained hindsight selector

Important caveat:

- more precisely, it is exact over the benchmark's gold span-group subsets under budget, scored by actual `Q2` generation
- it is still not exhaustive over all arbitrary token subsets maximizing final generated task score
- `A` is a full-cache reference condition, not a guaranteed upper bound; on
  harder settings a repaired smaller cache can occasionally beat the full cache
  by removing distracting context.

## Phase 7 Recommendation

Do **not** force one shared `B_base` across the optional `3q` sanity
panel, the locked `4q` bridge panel, and the harder `6q` panel.

Use task-specific calibrated budgets instead:

- `mq_niah_4q` bridge panel:
  - `B_base = 16384`
  - `R_ctx = 128`
- `mq_niah_6q` hard panel:
  - `B_base = 18432`
  - `R_ctx = 128`
- optional `mq_niah_3q` light panel:
  - `B_base = 14336`
  - `R_ctx = 128`

Shared settings:

- `4q` K grid: `{8, 16, 24, 32, 48, 64, 80, 96, 128}`
- `6q` K grid: `{8, 16, 24, 32, 48, 64, 80, 96, 128}`
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

## Why This Shape

- `2q` is too binary.
- `8q` is too heavy before `6q` is locked.
- `4q` is the validated bridge panel and `6q` is the right harder
  same-family extension.
- `3q` remains optional appendix-only sanity evidence.
- `6q` is the right harder extension, but it needs a larger `B_base` than `4q`.
- `B_base = 18432` for `6q` should be justified as the smallest calibrated
  hard-task budget with a nonzero matched baseline, visible `RepairKV` lift, and
  preserved oracle headroom, not as an aesthetic choice.
- `6q` should be described as a harder same-family extension, not as a perfectly matched single-axis difficulty control, because it also changes turn lengths and answer budget.
- if `6q` is promoted into the main text, it should be framed as a
  calibrated harder same-family extension, not as an independent
  confirmatory test.
- concretely, the `6q` split turns use a larger values-only decode budget
  (`48` tokens instead of `24`).
- `3q` is useful as a light sanity panel, but not strong enough to carry the whole broader-evidence story.

## Main Smoke Conclusions

- `mq_niah_4q @ 16384` is the better bridge panel under exact scoring.
- `mq_niah_6q @ 14336` is too harsh.
- `mq_niah_6q @ 18432` is the first clearly usable hard-task regime.
- in the accepted post-rewrite `6q` clean-suite smoke, the full-cache
  reference condition is near-perfect rather than perfect:
  `A = 0.979` with one miss out of 16 example-split rows.
- that miss is not a decode-budget artifact; the gold answer is well under the
  split turn's `48`-token cap.
- the post-rewrite `6q` smoke remains usable at the widened `K` axis through
  `128`: pooled `B_match` rises from `0.312` to `0.354`, while `RepairKV` rises
  from `0.375` to `0.979` and `Oracle-K` reaches `1.0`.
- `mq_niah_3q @ 14336` works, but is coarse and should be optional.
- the stronger harder-task design is a small `6q` clean suite, not just one split:
  - `1,5,6 -> 2,3,4`
  - `2,5,6 -> 1,3,4`
  - `3,5,6 -> 1,2,4`
  - `4,5,6 -> 1,2,3`
- these are the balanced `3 -> 3` partitions where turn 2 excludes the
  tail-favored sixth needle, so they are the right `6q` analogue of the
  recency-clean `4q` suite.
- the current exact-Q runner is good for quality evidence but still too slow to reuse the old sub-50ms latency framing without a separate timing aggregation.
- on the finalized exact `4q` bridge run, total exact repair p50 is about `6.38 s`,
  dominated by evicted scoring (`~6.29 s`), so the exact scorer is a
  quality-evidence path rather than the final systems path.
- the current exact path has a fresh green CPU-side suite behind it:
  53 tests passed across the NIAH generator, SnapKV/eviction invariants,
  eviction-buffer core, and the Phase 6 protocol / runner / selector /
  reporting stack.
- the frontier export now includes deterministic bootstrap confidence bounds in
  the `*_overall.csv` and `*_by_split.csv` files via `*_lo` / `*_hi` columns.
- final-active overlap should be treated as a mechanism diagnostic, not a
  surrogate for score: on the exact `4q` bridge run, `RepairKV` reaches nearly
  perfect quality before final-active overlap reaches `1.0`, because answering
  the values-only NIAH question does not require recovering every annotated
  token in each gold span.
- older overlap diagnostics mixed full-kept overlap for `B_match` with
  restored-only overlap for repair conditions; this is now fixed in the
  reporting/export layer by preferring final active overlap reconstructed from
  `b_kept_context_positions ∪ selected_positions` whenever the newer
  `*_active_overlap_fraction` fields are absent.
- on the re-exported exact `4q` bridge run, final-active overlap rises from
  about `0.363` to `0.794` while quality rises from `0.240` to `0.998`, so the
  overlap figure is now a clean mechanism diagnostic rather than a broken mix
  of semantics.

## Acceptance Rule For The Full 6q Panel

Use the completed `6q` panel directly in the paper package if all of the
following hold:

- pooled full-cache reference `A` is at least `0.95`;
- pooled `B_match` is at least `0.10` at the right edge of the sweep;
- pooled lift `RepairKV - B_match` is at least `0.10` at `K=48` and at
  least `0.20` at `K=96`;
- pooled `Random-K` and `Oldest-K` are each at least `0.10` below
  pooled `RepairKV` at `K=96`;
- pooled `Oracle-K - RepairKV` is at least `0.05` at `K=48`, so useful
  headroom still exists before the right edge; and
- the bootstrap lower bound for pooled `RepairKV` at `K=96` exceeds the
  bootstrap upper bound for pooled `B_match` at `K=96`; and
- every individual `6q` split has pooled `RepairKV - B_match > 0` at
  `K=96`; and
- every individual `6q` split has pooled `RepairKV - B_match >= 0.10` at
  `K=128`; and
- every individual `6q` split has pooled `RepairKV - Random-K > 0` and
  `RepairKV - Oldest-K > 0` at `K=128`.

Post-run audit should also confirm the row-level restore-budget invariants:

- `idlekv_restored_count == K`
- `random_k_restored_count == K`
- `oldest_k_restored_count == K`
- `oracle_k_restored_count <= K` in `gold_spans` mode

Useful helper after `6q` lands:

- `phases/phase7_broader_evidence/scripts/finalize_phase7_panels.sh`
  - postprocesses the frozen exact `4q` artifact and one finished `6q` artifact
  - regenerates the reduced-series main-text SVGs for both panels
- `phases/phase7_broader_evidence/scripts/audit_phase7_artifact.py`
  - prints the predeclared non-CI `6q` acceptance checks directly for
    `mq_niah_6q_clean_suite` artifacts
  - when given the exported overall CSV, also prints the bootstrap CI
    gate at `K=96`

## Final Full 6q Result

Completed full artifact:

- `phases/phase6_repair/results/full/mq_niah_6q_clean_suite_b18432_r128_qexact_q_ogold_spans_n100_k8-16-24-32-48-64-80-96-128_ca-b-bmatch-idlekv-randomk-oldestk-oraclek.json`

Export prefix:

- `paper/figures/phase7_mq_niah_6q_clean_suite_b18432_exact_*`

Final pooled exact `6q` numbers:

| K | B | B_match | RepairKV | Random-K | Oldest-K | Oracle-K |
|---|---:|---:|---:|---:|---:|---:|
| 8   | 0.413 | 0.411 | 0.432 | 0.413 | 0.426 | 0.501 |
| 16  | 0.413 | 0.415 | 0.462 | 0.413 | 0.411 | 0.589 |
| 24  | 0.413 | 0.411 | 0.515 | 0.413 | 0.415 | 0.826 |
| 32  | 0.413 | 0.412 | 0.564 | 0.414 | 0.407 | 0.837 |
| 48  | 0.413 | 0.415 | 0.653 | 0.418 | 0.407 | 0.994 |
| 64  | 0.413 | 0.418 | 0.773 | 0.409 | 0.405 | 0.994 |
| 80  | 0.413 | 0.417 | 0.902 | 0.413 | 0.407 | 0.994 |
| 96  | 0.413 | 0.422 | 0.989 | 0.426 | 0.407 | 0.994 |
| 128 | 0.413 | 0.415 | 0.990 | 0.420 | 0.403 | 0.994 |

Audit outcome:

- all predeclared pooled and per-split 6q gates passed;
- bootstrap gate passed: `RepairKV_lo@K=96 = 0.983 > B_match_hi@K=96 = 0.445`;
- row-level restore count invariants passed for RepairKV, Random-K, Oldest-K,
  and Oracle-K;
- exact 6q runtime is quality-path only: total repair p50 is about `6.99 s`,
  dominated by exact evicted scoring at about `6.90 s`.

## Current Status

Phase 7 priority is complete:

1. exact 4q bridge panel is frozen;
2. exact 6q harder same-family extension is complete and accepted;
3. paper figures and text now use the final two-panel package.

Optional:

1. `mq_niah_3q_split_3_to_12 @ B=14336` remains appendix-only sanity evidence.

## Paper Coherence Note

`paper/main.tex` should treat the exact `4q @ 16384` bridge artifact as frozen.
The final `6q` panel has landed; the setup, figures, and runtime language now
reflect the final two-panel Phase 7 evidence package.

Read next:

- `phases/phase7_broader_evidence/phase7_plan.md`
- `phases/phase7_broader_evidence/scripts/postprocess_phase7_artifact.sh`
- `phases/phase8_streaming_strict_cap/phase8_plan.md`
