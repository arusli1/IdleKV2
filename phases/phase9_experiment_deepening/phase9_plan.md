# Phase 9 Experiment Deepening Plan

## Goal

Phase 9 is graph-first. We should decide the paper objects we want before
launching experiments, then run only the missing cells needed to populate those
objects. Smoke tests are for asymmetric-upside ideas: causal controls, faster
scoring, and selector variants that could produce a much stronger figure if
they work.

Phase 9 should make the paper stronger without turning it into a broad
benchmark survey. The core result is already locked: in calibrated two-turn
MQ-NIAH settings, repairing a paused compressed KV cache after the next-turn
query is known can beat matched no-repair retention at the same resumed active
cache budget.

The missing evidence is not more dense `K` points. The missing evidence is:

1. whether the gain is truly tied to the newly revealed future query;
2. whether the repair setting is meaningfully different from existing
   query-aware KV selection and KV-offload systems;
3. whether a plausible runtime path can move from the exact analysis scorer
   toward an idle-window mechanism;
4. whether the current calibrated setup is robust enough to be trusted as a
   workshop paper.

The paper-facing message after Phase 9 should be:

> KV cache compression should not be a one-shot decision. In agent-motivated
> pause/resume workloads, a runtime can compress when the future turn is
> unknown, then use the idle interval to repair the resumed cache once the next
> query arrives.

This is a resource-adaptive inference claim, not an end-to-end agent-quality
claim.

## Current Execution Notes

2026-05-02 update: the main paper has been compressed toward a shorter
high-density workshop format. Exact split/scorer/Gold-K details now live in the
appendix, the main frontier uses paper-facing CSV names, and the guide records
that the 6-page limit is a ceiling rather than a target. The final
operating-regime heatmap run is active in tmux as `phase9_heatmap`, using
`n=24`, exact Q$_2$ scoring, Gold-K span references, 4Q base budgets
`{14336,16384,18432}`, 6Q base budgets `{12288,18432,24576}`, and
`K={16,48,96,128}`. Promote the heatmap only after the final CSVs are checked;
otherwise keep the shorter paper rather than forcing another figure.

Expert review narrowed the first Phase 9 run to a specificity control rather
than another large frontier sweep. The legacy `WrongQ-K` implementation uses
phantom keys and is therefore only a diagnostic for generic prompt-template
effects. Paper-grade specificity evidence should use `donor_q2`, which ranks
with another example's real future query while evaluating on the current
example's answer.

Initial legacy 6Q smoke at `K={48,96}`, `n=2`, failed the strict gate because
the wrong-query control still repaired above matched no-repair at `K=96`
(`wrong-query lift = 0.458`). Treat that artifact as a reason to replace the
control, not as evidence against the thesis.

Donor-query `WrongQ-K` with exact scoring separates at `K=48` but still
saturates at `K=96`, so it is a useful mid-budget specificity control and a
high-budget limitation. `ContrastiveQ-K`, which subtracts standardized
donor-query scores from true future-query scores, is promising in the `n=2` 6Q
suite smoke: pooled score is `1.000` at `K=48` and `K=64` versus default RepairKV
at `0.667` and `0.792`. It narrowly regresses at `K=96` (`0.958` versus
`1.000`), so promote it only as a mid-budget repair-policy candidate unless a
larger run removes that loss.

The next highest-value control is `StaleQ-K`: rank with the previous-turn query
and evaluate on the future turn. This tests whether post-query repair beats
what could have been done at compression time, which is the clearest novelty
boundary against standard query-aware selection.

`StaleQ-K` smoke outcome: one-split smoke was clean, but the `n=2` 6Q suite
does not pass the strict mid-budget gate. At `K=48`, RepairKV is `0.667` and
`StaleQ-K` is `0.583`; at `K=96`, RepairKV is `1.000` and `StaleQ-K` is `0.625`.
Use this as appendix/control evidence unless a larger run improves separation.

Proxy scoring is a strong systems-facing signal. The first 6Q `n=2`, `K=96`
smoke preserved nearly all exact quality (`0.958` versus exact `1.000`) while
dropping p50 total repair time from about `6.75s` to `0.78s`. A follow-up 6Q
`n=8`, `K=96` run still clears the quality gate: proxy RepairKV is `0.906`,
matched no-repair is `0.375`, and the proxy lift is `0.531`. Against the locked
exact 6Q reference at `K=96`, this keeps about `94%` of the exact lift while
cutting p50 total repair latency by about `8x` and p50 scoring latency by about
`11x`. Treat this as a promoted Phase 9 candidate for a quality-latency ladder,
then rerun at larger `n` before using exact/proxy latency claims in the main
paper.

Reviewer-facing caveat: do not present the `n=8` proxy smoke as a definitive
paired comparison against the locked exact `n=100` frontier. The full proxy run
must use the same calibrated tasks, budgets, seed offset, `K` values, and
example count as the locked exact references before making main-text latency
claims.

## Graph-First Target Package

The target package below is the starting point. Experiments should be designed
backward from these figures and tables.

## Visual Style Audit

Status: added after the paper-visual critique on 2026-05-02. Use this as a
checklist before adding or removing main-paper objects.

### Reference Pattern

Primary KV-cache papers are figure-forward:

- SnapKV, NeurIPS 2024 paper: 17 pages, roughly 11 numbered figures and 4
  numbered tables. The main paper uses a workflow figure, mechanism/overlap
  plots, NIAH/benchmark curves, latency curves, and ablations.
- QUEST, MLSys-style paper: 11 pages, roughly 11 numbered figures and 1
  numbered table. It uses a query-awareness schematic, attention-map evidence,
  quality curves, kernel latency panels, end-to-end latency panels, and an
  accuracy-efficiency frontier.
- ShadowKV, ICML 2025 paper: 21 pages, roughly 11 numbered figures and many
  benchmark/latency tables. The visual package mixes method diagrams, memory
  and throughput frontiers, NIAH panels, budget sensitivity, and task
  breakdowns.
- SCBench, benchmark paper: 31 pages, roughly 10 numbered figures and many
  tables. It uses a lifecycle schematic, overview heatmaps/trends, task
  breakdowns, and extensive appendix tables.

Short ICML/AdaptFM-adjacent workshop papers are smaller but still usually have
multiple visuals:

- KVzip, ES-FoMo III 2025: 20 pages including appendix, roughly 23 figures and
  3 tables.
- TinyServe, MOSS 2025: 7 pages, roughly 4 figures and 3 tables.
- AdaInf, ES-FoMo-II 2024: 8 pages, roughly 4 figures and no numbered tables.
- Adaptive CBM, ICML workshop 2025: 8 pages, roughly 1 figure and 3 tables.

Implication for this paper: the current draft should not be table-led. It
should target one method figure plus two or three experimental figures in the
main text, with dense numeric tables moved to appendix.

### Paper Formatting Rules

- Prefer one-column figures unless a figure truly needs the width. Single-column
  multi-panel plots are normal in ICML-style papers and avoid breaking the
  two-column flow.
- Use tables only for exact endpoint values, runtime accounting, or appendix
  robustness. If a table is mostly repeated `K` values or repeated conditions,
  it should usually become a plot.
- Main experimental plots should have a clear visual job: frontier,
  operating-regime map, mechanism diagnostic, or latency-quality tradeoff.
- Legends should not cover data. Put legends below or outside plots.
- Avoid low-signal "all data" tables in the main paper. Tables can exist in the
  appendix if they defend reproducibility or exactness.
- Prefer dense but interpretable plots: small multiples, heatmaps, and
  frontiers. Avoid decorative plots, radar charts, or crowded line plots whose
  only message is "more numbers."

### Terminology Rules

- Paper text should not use internal phase names or unexplained shorthand.
- Define key-value (KV) cache, active cache, evicted-buffer cache, and
  resumption budget before relying on them.
- Prefer "score gain over matched no-repair" to the shorthand "lift" in
  paper-facing prose and axis labels. If a compact symbol is needed, use
  $\Delta_{\mathrm{repair}}$ and define it once.
- Prefer "retained exact gain" to "retained lift" for proxy scoring.
- Define $\goldk{}$ as a "benchmark-metadata hindsight reference" or
  "hindsight span reference"; do not call it an oracle in paper-facing text.
- Define "matched no-repair" as the baseline that keeps the same number of
  active evictable-context tokens at resumption. Be explicit that total
  CPU+GPU retained state is not matched because \idlekv{} also has a CPU
  buffer.
- Use "restore budget $K$" only after saying that $K$ counts restored context
  tokens, not scored anchors, KV rows, or a memory-byte budget.

### Target Visual Package

Main paper target:

1. Pipeline schematic: define the paused-compressed-cache repair setting.
2. Matched resumed-cache frontier: core 4Q/6Q exact evidence.
3. Operating-regime heatmap: show where repair helps, where it is impossible,
   and where matched no-repair saturates. This should replace rather than add a
   dense budget table.
4. Optional mechanism or latency figure only if it displaces text/table space
   and adds a distinct claim.

Appendix target:

1. Endpoint and partition tables for exact values and robustness.
2. Runtime/proxy table or compact latency-quality plot.
3. Gold-span overlap mechanism plot if space allows.
4. Null controls and failed variants, but only when they clarify limitations.

### Needed Data For Target Package

- No rerun needed for the main frontier; the locked 4Q/6Q exact artifacts are
  the best current core evidence.
- Run the operating-regime heatmap at larger `n` after smoke passes:
  4Q `B_base={14336,16384,18432}`, 6Q `B_base={12288,18432,24576}`,
  `K={16,48,96,128}`, conditions `A/B/B_match/RepairKV/Gold-K`.
- Do not run 2Q or 3Q for the main paper now; they are likely lower signal than
  strengthening the operating-regime evidence. Run 8Q only as an appendix or
  future-work stress test after the main visual package is stable.

### Main Paper Objects

1. **Pipeline schematic.**
   - Status: already available.
   - Claim: cache state is compressed after turn `N`, buffered during idle
     time, then repaired before turn `N+1`.
   - No new data.

2. **Matched resumed-cache frontier.**
   - Status: already available from locked 4Q/6Q exact artifacts.
   - Claim: at the same resumed active-cache budget, RepairKV beats matched
     no-repair and content-agnostic restores.
   - Required data: `B_match`, RepairKV, Random-K, Oldest-K, Gold-K, full-cache
     reference over the calibrated `K` grid.
   - Current data quality: strong; do not regenerate unless a Phase 9 control
     exposes a real flaw.

3. **Future-query specificity / contrastive repair figure.**
   - Status: missing.
   - Claim: repair works because the next-turn query is newly available, not
     because any query-like text recovers generic key/value bursts.
   - Ideal visualization: compact 6Q plot at `K={24,48,64,96}` or a full
     score-vs-`K` panel with `B_match`, true-`Q2` RepairKV, donor-query
     `WrongQ-K`, optional `ContrastiveQ-K`, and Gold-K.
   - Required experiment: exact 6Q donor-query smoke, then contrastive smoke if
     donor query exposes generic template recovery.
   - Main-paper rule: promote only if it is clean. If wrong-query is confounded
     by the NIAH template and contrastive scoring does not improve the story,
     keep the result as a limitation and do not force the figure.

4. **Repair phase diagram.**
   - Status: missing; this is the best "cool but honest" figure.
   - Claim: repair has an operating regime. It helps when the base cache is
     stale but the task is still recoverable; it disappears in all-zero and
     recency-saturated regimes.
   - Ideal visualization: two heatmaps, 4Q and 6Q, with base budget on the
     y-axis, `K` on the x-axis, and color equal to
     `RepairKV - matched no-repair`. Mark cells where Gold-K headroom remains.
   - Required experiment: sparse base-budget sweep.
   - Main-paper rule: promote if it cleanly explains the calibrated setting and
     replaces a table; otherwise appendix.

5. **Quality-latency ladder.**
   - Status: partially available; exact runtime and transfer-only runtime exist,
     proxy/two-stage data missing.
   - Claim: exact scoring is mechanistic, but the systems primitive is a
     quality-latency tradeoff that can be improved by faster scoring.
   - Ideal visualization: score vs p50 repair/scoring latency at `K=96`, with
     no-repair, transfer-only annotation, exact RepairKV,
     proxy RepairKV, and two-stage rerank if it clears smoke.
   - Required experiment: runtime microbenchmark first, proxy smoke only if the
     proxy path actually reduces scoring latency.
   - Main-paper rule: promote only if proxy/two-stage preserves meaningful lift
     and cuts latency substantially; otherwise appendix.

Main paper should use at most one of Objects 3-5 in addition to the pipeline
and matched frontier, unless the page budget expands. The preferred main
promotion order is:

1. Future-query specificity, if clean.
2. Repair phase diagram, if the heatmap is interpretable and visually strong.
3. Quality-latency ladder, if proxy/two-stage makes exact scoring look like a
   stepping stone rather than a dead end.

### Appendix Objects

1. **Per-partition endpoint table.**
   - Status: already available.
   - Purpose: shows the pooled result is not carried by one easy split.

2. **Runtime table.**
   - Status: already available for exact and transfer-only paths.
   - Purpose: separates cheap KV movement from expensive exact scoring.

3. **Overlap/mechanism plot.**
   - Status: data already available.
   - Purpose: shows repaired active cache moves toward annotated future-query
     spans.

4. **Robustness/seed overlays.**
   - Status: missing.
   - Purpose: defend calibration if time permits.

5. **Null or failed controls.**
   - Status: depends on smoke tests.
   - Purpose: document wrong-query/proxy/selector failures without bloating the
     main paper.

### Tables To Avoid In Main Unless They Replace Text

- Dense all-`K` numeric tables: redundant with line plots.
- Split-by-split full tables: appendix only.
- Runtime-by-split tables: appendix only.
- Multiple selector tables: only useful if a selector becomes a main algorithmic
  result.

## Novelty Boundary

Checked adjacent work:

- SnapKV: compresses KV from prompt-end observations before generation.
- QUEST: query-aware sparse KV/page selection during long-context inference.
- ShadowKV: stores/offloads KV and reconstructs sparse KV pairs on the fly for
  high-throughput long-context decoding.
- SCBench: benchmarks the full KV-cache lifecycle across generation,
  compression, retrieval, and loading.
- AdaptFM CFP: explicitly values dynamic runtime decisions, token-level
  adaptation, dynamic KV cache compression, quality-resource tradeoffs, and
  systems/hardware-aware adaptive inference.

The paper should not claim that query-aware KV selection or KV lifecycle
benchmarking is new. The defensible novelty is narrower:

- hidden-future-query setup;
- already-compressed paused state;
- no full-prefix recompute after the future query is revealed;
- pre-resume mutation of the active cache during an idle window;
- matched resumed-cache budget against no-repair retention;
- explicit separation between quality evidence and scorer latency.

This positioning fits AdaptFM because it treats KV cache state as a mutable
runtime resource, with quality-resource tradeoffs driven by available memory,
latency, and idle time.

## Paper Figure Budget

### Phase 7 Audit: Keep, Reframe, Or Regenerate?

The locked 4Q/6Q exact suite is good enough to remain the paper's core
empirical result. It has the right shape for the current thesis:

- two calibrated same-family variants rather than one cherry-picked split;
- matched no-repair at `B_base + K`, so the gain is not just "kept more active
  KV";
- random and oldest-token restore controls that stay near no-repair;
- Gold-K hindsight reference showing that mid-budget gaps are real;
- final-active overlap and runtime exports already available for mechanism and
  systems caveats.

Do not regenerate this frontier just to make the graph look different. Rerun it
only if one of the following happens:

- a code/reporting audit finds a semantic error in the exported columns;
- the Phase 9 causal smoke shows the current positive result is likely driven
  by a non-query-specific artifact;
- a faster/proxy scorer clears its smoke gate and becomes strong enough to
  replace the exact scorer as the main deployment-facing line;
- a selector improvement closes a meaningful part of the Gold-K gap without
  hurting high-`K` performance.

Current holes in the Phase 7 data:

- it is conditional `Q2` quality under a fixed full-cache `Q1` transcript, not
  end-to-end agent quality;
- exact scoring is slow, so the current data is mechanistic rather than a
  deployable latency result;
- the task is synthetic and local-span-friendly;
- wrong-query specificity is not resolved on the locked full suite;
- total CPU+GPU memory is not matched, only resumed active cache is matched.

Those holes are real, but they are better addressed by Phase 9 controls than by
rerunning the same frontier.

### Main Figure 1: Pipeline

Status: already in the paper.

Purpose: define the setting quickly. It should show compress, CPU buffer, idle
window, score against `Q2`, restore, and resume. It is the conceptual bridge
from compression-only to compression-plus-repair.

### Main Figure 2: Matched Resumed-Cache Frontier

Status: already in the paper from the final 4Q and 6Q exact artifacts.

Purpose: the core evidence. Keep it as the anchor figure:

- MQ-NIAH-4Q and MQ-NIAH-6Q panels;
- `B_match`, RepairKV, random, oldest, full-cache reference, Gold-K;
- legend outside or below the plots so it does not cover data.

This should remain the strongest main-text figure unless Phase 9 produces a
cleaner causal or latency story.

### Ideal Main Figure 3: Future-Query Specificity

Question: does repair help because the next-turn query is known?

Conditions:

- matched no-repair;
- `StaleQ-K`, scored with the previous-turn query;
- RepairKV scored with true `Q2`;
- `WrongQ-K` scored with a donor future query from another example;
- `ContrastiveQ-K`, scored by standardized true-`Q2` minus donor-`Q2` scores;
- stale/turn-`N` query signal if implemented cleanly;
- Gold-K reference.

Preferred plot:

- 6Q score vs `K` using `K={8,16,24,32,48,64,80,96,128}`;
- or a compact bar/line figure at `K={48,96}` if the full curve is too large.

Promotion rule:

- move to main only if true `Q2` separates from wrong/stale query at mid/high
  budgets and wrong/stale query stays near matched no-repair;
- give `StaleQ-K` higher evidentiary weight than donor `WrongQ-K`, because it
  distinguishes post-query repair from compression-time query-aware selection;
- promote `ContrastiveQ-K` only if it improves mid-budget recovery or creates a
  cleaner specificity story without hurting high-budget recovery;
- otherwise keep as appendix/null diagnostic and do not build the thesis on it.

### Ideal Main or Appendix Figure 4: Quality-Latency Ladder

Question: can repair plausibly fit an idle-window budget?

Conditions:

- no repair;
- exact RepairKV;
- proxy RepairKV;
- two-stage rerank if implemented;
- full recompute/re-prefill estimate or measured baseline if feasible.

Preferred plot:

- x-axis: p50 repair/scoring latency;
- y-axis: mean `Q2` score at `K=96` or `K=128`;
- annotations for transfer/injection-only vs scoring-dominated paths.

Promotion rule:

- main only if proxy or two-stage scoring keeps most of the quality lift with a
  clear latency reduction;
- otherwise appendix, with text saying exact scoring is mechanistic evidence.

### Appendix Figure A: Mechanism Overlap

Question: does repair actually move the active cache toward the hidden
future-relevant spans?

Data already exists:

- `paper/figures/phase7_*_overlap_overall.csv`;
- optional by-split overlap CSVs.

Plot:

- final-active gold-span overlap vs `K`;
- compare matched no-repair, RepairKV, random, oldest, Gold-K.

Use this as a mechanism diagnostic, not a surrogate score.

### Appendix Figure B: Operating-Regime Heatmap

Question: when should repair help or disappear?

Plot:

- rows: base budget;
- columns: `K`;
- color: `RepairKV - matched no-repair`;
- separate 4Q and 6Q panels.

Purpose:

- defend calibration;
- show low-budget all-zero regimes, useful mid regimes, and recency-saturated
  regimes.

### Appendix Figure C: Seed/Partition Robustness

Question: is the effect a single split or seed artifact?

Plot:

- thin per-split or per-seed curves;
- pooled curve highlighted.

Keep in appendix unless a reviewer explicitly asks for robustness in main.

## Execution Iterations

### Iteration 0: Existing-Artifact Paper Package

Purpose: extract every high-signal diagnostic already available before running
more GPU jobs.

Inputs:

- `paper/figures/phase7_*_overall.csv`;
- `paper/figures/phase7_*_by_split.csv`;
- `paper/figures/phase7_*_overlap_overall.csv`;
- `paper/figures/phase7_*_runtime_overall.csv`.

Tasks:

1. Add or generate appendix-only overlap and runtime plots if space permits.
2. Keep the main text compact: pipeline plus matched frontier plus one concise
   table at most.
3. Make all paper-facing names reviewer-legible:
   - `full-cache reference`;
   - `base compressed cache`;
   - `matched no-repair`;
   - `Gold-K hindsight reference`;
   - `matched resumed-cache budget`.

Tests:

- no GPU runs;
- rebuild `paper/main.pdf`;
- check the LaTeX log for undefined refs/cites and overfull boxes.

### Iteration 1: Future-Query Specificity

Purpose: directly test the causal claim.

Smoke A:

```bash
python phases/phase6_repair/scripts/run_phase6.py \
  --stage smoke \
  --task mq_niah_6q_split_456_to_123 \
  --num-samples 1 \
  --base-context-budget 18432 \
  --recency-window 128 \
  --query-scoring-mode exact_q \
  --oracle-mode gold_spans \
  --wrong-query-mode donor_q2 \
  --wrong-query-donor-offset 100000 \
  --k 48 96 \
  --conditions A B B_match RepairKV WrongQ-K Oracle-K
```

Smoke B:

```bash
python phases/phase6_repair/scripts/run_phase6.py \
  --stage smoke \
  --task mq_niah_6q_clean_suite \
  --num-samples 2 \
  --base-context-budget 18432 \
  --recency-window 128 \
  --query-scoring-mode exact_q \
  --oracle-mode gold_spans \
  --wrong-query-mode donor_q2 \
  --wrong-query-donor-offset 100000 \
  --k 48 96 \
  --conditions A B B_match RepairKV WrongQ-K Oracle-K
```

Smoke C:

```bash
python phases/phase6_repair/scripts/run_phase6.py \
  --stage smoke \
  --task mq_niah_6q_clean_suite \
  --num-samples 2 \
  --base-context-budget 18432 \
  --recency-window 128 \
  --query-scoring-mode exact_q \
  --oracle-mode gold_spans \
  --wrong-query-mode donor_q2 \
  --wrong-query-donor-offset 100000 \
  --k 48 96 \
  --conditions A B B_match RepairKV StaleQ-K WrongQ-K Oracle-K
```

Full-run gates:

- run full 6Q donor-query `WrongQ-K` only if true `Q2` beats donor `Q2` by at
  least `0.05` at `K=48` and `0.10` at `K=96`;
- run full `StaleQ-K` only if true `Q2` beats stale query by at least `0.15`
  at `K=48`, and stale-query lift at `K=96` is no more than half of true-query
  lift;
- require donor-query lift over matched no-repair to stay at or below `0.10`
  at `K=96`;
- if donor query does not separate, write it as a limitation of templated NIAH
  and do not overclaim future-query specificity.

Exporter dependency:

- before full `WrongQ-K`, update reporting/export helpers so `wrong_q_k`,
  runtime, and wrong-query mode metadata are emitted to CSV and bootstrap
  intervals can be generated.

Paper use:

- main if clean;
- appendix/null-control note if not clean.

### Cool But Honest Figure Candidate: Repair Phase Diagram

If Phase 9 produces either a base-budget sweep or clean causal controls, the
most visually impactful figure should be a small heatmap rather than another
line plot.

Best version:

- x-axis: restore budget `K`;
- y-axis: base compressed budget `B_base`;
- color: `RepairKV - matched no-repair`;
- contour/marker: where Gold-K still has at least `0.05` headroom;
- two panels: 4Q and 6Q.

This tells the story in one glance: repair helps in a band where the cache is
compressed enough to be stale, but not so compressed that the task is lost or
so large that recency/no-repair already wins. It also makes the broader systems
claim concrete: idle-window repair is an operating-regime primitive, not a
single magic curve.

Cheap fallback from existing data:

- plot normalized recovery
  `(RepairKV - B_match) / (Gold-K - B_match)` vs `K` for 4Q and 6Q;
- add a light background band for "Gold-K headroom remains";
- use this only if space allows, because it repackages the frontier rather than
  adding new evidence.

Do not promote a "cool" graph to the main paper unless it clarifies a claim
better than the current milestone table. Prefer replacing a table over adding
another object.

Smoke outcome:

- 6Q `n=2`, `B_base={12288,18432,24576}`, `K={48,96}` now exists in
  `phases/phase9_experiment_deepening/results/phase9_phase_diagram_6q_n2.csv`.
- The pattern is coherent: `B=12288` has near-zero matched no-repair and weak
  K=48 recovery but improves at K=96; `B=18432` is the calibrated region with
  strong K=96 recovery; `B=24576` is near saturation, where matched no-repair is
  already high and repair adds less.
- 4Q `n=2`, `B_base={14336,16384,18432}`, `K={48,96}` now exists in
  `phases/phase9_experiment_deepening/results/phase9_phase_diagram_4q_n2.csv`.
  It shows the same operating-regime shape: low/mid budgets have large repair
  lifts with Gold-K headroom, while the high-budget `K=96` cell saturates.
- This is currently the best "cool but honest" figure candidate. The 4Q/6Q
  smoke panels are coherent enough to justify a larger run; promote to the main
  paper only after increasing `n` and preferably expanding `K` beyond two
  columns.

### Iteration 2: Runtime Path

Purpose: show whether the repair primitive can move toward an idle-window
systems story.

Smoke:

First compare exact and proxy scorer timing on the same tiny cell. Do not
promote proxy curves unless p50 total repair time and p50 scoring time both
drop by at least `25%`.

```bash
python phases/phase6_repair/scripts/run_phase6.py \
  --stage smoke \
  --task mq_niah_6q_clean_suite \
  --num-samples 2 \
  --base-context-budget 18432 \
  --recency-window 128 \
  --query-scoring-mode proxy \
  --oracle-mode gold_spans \
  --k 96 \
  --conditions A B B_match RepairKV Oracle-K
```

Full-run gates:

- run the exact counterpart on the same cell and compare the merged Phase 9
  quality/runtime CSV before launching any proxy curve;
- promote proxy full curves only if proxy RepairKV beats matched no-repair by at
  least `0.10` at `K=96` on 6Q or `0.20` at `K=96` on 4Q;
- require proxy to preserve at least `70%` of exact RepairKV lift at `K=96`;
- attempt two-stage rerank only after proxy shows nontrivial lift;
- promote two-stage rerank only if it stays within `0.03` of exact RepairKV at
  `K=96` while cutting p50 total repair time by at least `3x`.

Paper use:

- quality-latency appendix by default;
- main only if it materially answers the deployment concern.

Smoke outcome:

- Proxy 6Q `n=2`, `B_base=18432`, `K=96` exists in
  `phases/phase9_experiment_deepening/results/phase9_proxy_suite_n2.csv`.
- Proxy 6Q `n=8`, `B_base=18432`, `K=96` exists in
  `phases/phase9_experiment_deepening/results/phase9_proxy_suite_n8.csv`.
- Compared with the locked exact 6Q reference, proxy preserves about `94%` of
  the exact RepairKV lift while cutting p50 total repair latency by about `8x`.
- Short `n=2` calibrated 4Q/6Q smokes at `K={48,96}` pass the proxy quality
  gate and are summarized in
  `phases/phase9_experiment_deepening/results/phase9_proxy_4q_smoke_n2.csv`
  and `phase9_proxy_6q_smoke_n2.csv`.
- The current long run is
  `tmux attach -t phase9_proxy_full`: calibrated 4Q and 6Q, `n=100`,
  `K={48,96}`, proxy scorer only, summarized against the locked exact
  references. It is intentionally a fixed-operating-point experiment with
  `K=48` as an anti-cherry-pick guardrail and `K=96` as the paper headline
  point.
- Promotion is checked by
  `phases/phase9_experiment_deepening/scripts/check_proxy_quality_latency.py`.
  The gate compares proxy rows against exact-reference rows at the same `K`,
  requiring positive headline lift, retained exact lift, bounded absolute
  score loss, total latency speedup, scoring latency speedup, and no hidden
  `K=48` regression.
- Paired uncertainty is generated by
  `phases/phase9_experiment_deepening/scripts/proxy_paired_bootstrap.py`,
  matching exact and proxy rows by split, example index, and `K`.

### Iteration 3: Repair-Policy Improvement

Purpose: test whether the gap to Gold-K is an algorithmic opportunity.

Priority:

1. `IntervalPack`: select non-overlapping high-scoring bursts under budget.
2. `CoverageRepairKV`: only if smoke suggests multi-query under-coverage.
3. Contrastive or stale-delta scoring only if future-query specificity is weak.

Software tests:

- selector unit tests;
- runner/reporting smoke tests;
- artifact export smoke before any full GPU run.

Live smoke:

```bash
python phases/phase6_repair/scripts/run_phase6.py \
  --stage smoke \
  --task mq_niah_6q_clean_suite \
  --num-samples 4 \
  --base-context-budget 18432 \
  --recency-window 128 \
  --query-scoring-mode exact_q \
  --oracle-mode gold_spans \
  --k 24 32 48 64 \
  --conditions A B B_match RepairKV Oracle-K
```

Full-run gates:

- new selector closes at least `20%` of the default RepairKV-to-Gold-K gap at
  `K=48` on pooled 6Q;
- no loss larger than `0.02` at `K=96` or `K=128`;
- selector overhead below `250 ms` p50, excluding exact scoring.

Paper use:

- main only if it clearly improves the frontier;
- otherwise keep as future work, because weak heuristic variants dilute the
  central protocol contribution.

### Iteration 4: Portability and Robustness

Purpose: defend against "one calibrated setting" without burning time on low
signal breadth.

Sparse base-budget sweep:

- 4Q: `B={14336,16384,18432}`;
- 6Q: `B={16384,18432,20480}`;
- `K={16,48,96,128}`;
- `n=8`;
- conditions: `A B B_match RepairKV Oracle-K`.

Seed offsets:

- `dataset_seed_offset={1000,2000}`;
- `K={48,96,128}`;
- frozen 4Q/6Q exact settings.

Second-compressor portability:

- high value if already available in code;
- otherwise defer rather than introducing a brittle compressor late.

Paper use:

- appendix unless it changes the main conclusion.

## What Not To Run First

- More dense `K` grids: the existing grid already shows onset, rise, and
  saturation.
- Full 3Q panel: mostly duplicates the current 4Q/6Q breadth story.
- Phase 8 strict-cap qnorm spill with current defaults: the smoke gate already
  failed because gold spans never enter the qnorm spill buffer.
- Broad real-agent benchmarks before the causal and runtime controls: valuable
  later, but too noisy to explain the current mechanism.
- Many selector variants in parallel: reviewers will read weak heuristic
  variants as confusion unless one closes the Gold-K gap.

## Critical Flaws To Track

- Exact scorer latency is seconds, not deployable. The exact path is evidence
  for the mechanism, not the final system.
- Matched resumed active cache does not mean matched total memory. RepairKV keeps
  a CPU buffer of evicted KV.
- MQ-NIAH is synthetic and favorable to local span restoration.
- Current evidence measures conditional `Q2` quality under a fixed full-cache
  `Q1` transcript, not end-to-end two-turn agent performance.
- Wrong-query controls can be confounded by templated key/value structure.
- Gold-K is a benchmark-metadata hindsight reference over annotated span
  groups, not a mathematical upper bound over all possible token subsets.
- The paper should make broad agent-systems implications, but only after the
  empirical claim is stated precisely.

## Stop Criteria

Phase 9 is done when the paper has:

1. a precise novelty boundary against SnapKV, QUEST, ShadowKV, and SCBench;
2. one main empirical frontier figure from the locked 4Q/6Q exact suite;
3. either a clean causal-specificity figure or a clear limitation explaining
   why templated wrong-query controls are inconclusive;
4. a runtime appendix that separates transfer/injection feasibility from exact
   scorer latency;
5. no internal "phase" vocabulary in paper-facing text;
6. a compact main text that frames the broader implication as a research agenda
   for dynamic KV cache maintenance in resource-adaptive agent inference.
