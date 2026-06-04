# Phase 9: Experimental Deepening Ideas

Generated: 2026-05-02

## Purpose

The executable staged plan is `phase9_plan.md`. This file remains the broad
idea map and parking lot for variants that may or may not survive smoke gates.

Phase 9 is the paper-evidence expansion phase.

Phase 7 established the current headline result: after a one-shot 32K
compression, future-query-informed repair can outperform matched no-repair
retention at the same final active-cache size on calibrated MQ-NIAH-4Q and
MQ-NIAH-6Q panels.

Phase 8 is separately exploring strict-cap streaming with bounded CPU spill.
Phase 9 should not duplicate that. Instead, Phase 9 asks what additional
experiments, algorithm variants, and figures would most strengthen the current
paper.

The desired outcome is not “more data.” The desired outcome is a tighter
experimental story:

1. **Causal:** show that the newly revealed future query is the reason repair
   helps.
2. **Algorithmic:** show that post-compression repair is an algorithmic design
   space, not just one positive benchmark curve.
3. **Systems-facing:** show where the current implementation sits on a
   quality-latency frontier, and what must change for deployment.
4. **Robustness:** show the effect is not a single tuned operating point.

This file is the idea map. It is not the execution plan.

## Current Evidence Gap

The current paper has one strong experimental axis:

- score as a function of restore budget `K`.

That figure is necessary, but it is visually and scientifically close to 1D.
It answers:

- does RepairKV improve as more evicted context is restored?

It does not fully answer:

- is the improvement specifically caused by future-turn relevance?
- when should the effect disappear?
- can a better repair policy close the gap to the Gold-K reference?
- can the mechanism fit an idle-window latency budget?
- is the result robust to cache budget, seed, and recency structure?

Phase 9 should add the smallest set of high-signal experiments that answer
those questions.

## Candidate Main-Figure Package

### Figure A: Future-Query Specificity

**Question.** Does repair help because the next-turn query is known, or because
any question-like signal recovers generic key/value bursts?

Candidate conditions:

- matched no-repair retention
- current RepairKV using the true `Q2`
- wrong-query repair using a task-matched decoy query
- stale-query or turn-N repair using only the turn-N signal
- `StaleDelta`: `z(Q2) - lambda * z(turnN)`
- `ContrastiveQ`: `z(Q2) - alpha * z(wrong_query)`
- Gold-K reference

Candidate figure:

- score vs `K` on MQ-NIAH-6Q, plus one hard MQ-NIAH-4Q split if needed
- optional companion panel: final-active overlap with annotated `Q2` gold spans

Why this is high value:

- It directly tests the paper thesis.
- It turns the existing weak wrong-query control into a constructive ablation:
  if wrong queries recover generic template bursts, contrastive scoring may
  subtract that generic component.
- It helps distinguish “future query matters” from “more active tokens matter.”

Failure modes:

- Wrong-query repair may still perform well because MQ-NIAH prompts share strong
  generic structure.
- Contrastive scoring may over-subtract and hurt true positives.
- If the ablation is noisy, it should go to appendix rather than main.

### Figure B: Better Repair Policies

**Question.** Is RepairKV an algorithmic design space, or only one heuristic?

The current selector ranks individual evicted tokens by `Q2` score and expands
fixed local bursts. That is simple and legible, but it can waste budget around
false-positive anchors and redundant neighborhoods.

Candidate variants:

- current RepairKV
- `IntervalPack`: score candidate windows and choose non-overlapping windows
  under budget
- `CoverageRepairKV`: encourage coverage across multiple queried values rather
  than letting one hot query token dominate
- `StaleDelta` or `ContrastiveQ` combined with better packing
- Gold-K reference

Candidate figure:

- score vs `K` at moderate budgets, especially `K = 24, 32, 48, 64`
- focus on MQ-NIAH-6Q, where the mid-budget gap to Gold-K is largest

Why this is high value:

- It creates algorithmic novelty.
- It explains the Gold-K gap as a repair-policy problem, not merely as a
  limitation of the hypothesis.
- It gives the paper something constructive to contribute beyond the benchmark
  protocol.

Failure modes:

- New selectors may not improve over the simple heuristic.
- If improvements are small, the figure may still be useful as an ablation, but
  not as a main result.

### Figure C: Quality-Latency Pareto

**Question.** Can repair plausibly fit into an idle window?

Candidate conditions:

- matched no-repair retention
- full-cache reference resume
- exact RepairKV
- proxy RepairKV
- two-stage rerank: cheap screen, exact rerank top `M`
- possibly raw transfer/reinjection only as a lower-bound point

Candidate x-axis:

- p50 turn-2 overhead or p50 repair/scoring latency

Candidate y-axis:

- turn-2 score

Candidate figure:

- scatter or connected Pareto curve
- annotate exact scorer, proxy scorer, and two-stage rerank variants

Why this is high value:

- The current exact scorer is strong mechanistic evidence but slow.
- A quality-latency curve tells reviewers whether the idea has a deployment
  path.
- It connects the paper to systems work: the bottleneck is not transfer or
  injection, it is scoring the evicted pool.

Failure modes:

- Proxy or two-stage rerank may have poor quality.
- If exact remains the only high-quality point, the figure should be framed as
  identifying the bottleneck rather than claiming deployment readiness.

### Figure D: Operating-Regime Heatmap

**Question.** Is the effect robust across cache budgets?

Candidate axes:

- x-axis: restore budget `K`
- y-axis: base context budget `B_base`
- color: `RepairKV - matched no-repair`

Candidate settings:

- MQ-NIAH-4Q first, because it is cheaper
- MQ-NIAH-6Q only if time permits

Why this is high value:

- It shows the regime where repair helps.
- It makes calibration transparent.
- It can show expected null regions:
  - all-zero regimes
  - recency-saturated regimes
  - high-budget regimes where no-repair already keeps enough context

Failure modes:

- Expensive if run at full `n=100` over too many cells.
- A sparse heatmap is acceptable; it should not become a giant table.

### Figure E: Mechanism + Statistics From Existing Data

**Question.** Are gains statistically separated and associated with restoring
future-relevant spans?

This can likely be built from existing exported CSVs.

Candidate panels:

- `RepairKV - matched no-repair` with bootstrap intervals
- final-active overlap with annotated `Q2` span groups vs `K`

Why this is high value:

- No or little new compute.
- Strengthens the current result immediately.
- Helps readers understand what the repair is doing.

Failure modes:

- It may be less novel than the causal and algorithmic figures.
- It should not displace the main frontier unless space is tight.

## Algorithm Ideas

### ContrastiveQ

Score positions by:

```text
score = z(score_Q2) - alpha * z(score_wrong_query)
```

Optional:

```text
score = z(score_Q2) - alpha * z(score_wrong_query) - beta * z(score_turnN)
```

Rationale:

- The wrong-query path already exists in the runner.
- If wrong queries recover generic key/value bursts, subtracting wrong-query
  score may isolate query-specific relevance.

Main risk:

- The decoy query may not be the right negative; it may subtract useful generic
  retrieval structure.

### StaleDelta

Score positions by:

```text
score = z(score_Q2) - lambda * z(score_turnN)
```

or:

```text
score = score_Q2 / (epsilon + score_turnN)
```

Rationale:

- The paper is about stale caches, but the current selection rule mostly uses
  the turn-N score only as a tie-break.
- This ablation directly asks whether repair should favor newly relevant tokens,
  not merely Q2-salient tokens.

Main risk:

- Turn-N scores may be noisy or not semantically aligned enough to subtract.

### IntervalPack

Instead of expanding around top-ranked token anchors, precompute candidate
intervals and select non-overlapping intervals by utility per token.

Possible variants:

- fixed-width windows around candidate anchors
- variable-width windows with a small set of widths
- greedy non-maximum suppression
- weighted interval scheduling if the candidate set is small enough

Rationale:

- Current fixed burst packing can spend budget on redundant neighborhoods.
- Gold-K suggests a small number of value-bearing spans can solve the task.

Main risk:

- More complex selector may overfit to MQ-NIAH span geometry.

### CoverageRepairKV

Attempt to cover multiple future query intents rather than max-pooling all query
tokens into one global score.

Possible variants:

- split the query into key-like token groups and round-robin top windows
- use top query-token clusters rather than one pooled query score
- penalize candidate windows whose query-attention profile is redundant with
  already selected windows

Rationale:

- MQ-NIAH-6Q asks for three values in Q2.
- A single hot query token can dominate the current max-pooled score.

Main risk:

- Identifying query groups robustly without task-specific hacks may be hard.

### TwoStageRerank

Use a cheap scorer to shortlist candidates, then apply exact Q2 scoring only to
the shortlist.

Candidate shortlist sizes:

- `M = 256, 512, 1024, 2048`

Candidate cheap scorers:

- query/key norm
- old proxy scorer
- eviction-time importance
- dot-product surrogate already available in earlier buffer code

Rationale:

- Exact evicted scoring dominates runtime.
- Two-stage rerank can produce a clean quality-latency Pareto.

Main risk:

- If the cheap screen misses relevant spans, exact reranking cannot recover.

### Layer/Query-Token Pruning

Cheap learned-free approximation family:

- score with last `L` layers only
- score only content-bearing query tokens
- score top-`r` query tokens by norm
- compare max vs mean vs top-r pooling

Rationale:

- Current exact scorer uses all layers and all prompt tokens.
- This may reduce scoring cost without changing the repair protocol.

Main risk:

- Could become a pile of low-signal ablations unless tied to the Pareto figure.

## Robustness Ideas

### Seed Robustness

Rerun frozen 4Q and 6Q settings with one or two additional dataset seed offsets.

Candidate figure:

- forest plot of `RepairKV - matched no-repair` at `K = 48, 96, 128`
- or thin-line overlays of score frontier by seed

Main value:

- Shows the effect is not a lucky 100-example sample.

Main risk:

- Expensive if run at full K grid and full conditions.

### Recency Diagnostic

Run the recency-favorable 4Q partitions such as `12 -> 34`.

Candidate figure:

- clean partitions vs recency-favorable partitions
- y-axis: `RepairKV - matched no-repair`

Main value:

- Shows when repair should help less: if Q2 asks for tail-favored needles,
  no-repair already has a recency advantage.

Main risk:

- If repair still helps a lot, the interpretation needs care.

### Base-Budget Sweep

Sweep `B_base` around the current calibrated values.

Candidate grid:

- 4Q: around `12288, 14336, 16384, 18432`
- 6Q: around `16384, 18432, 20480`
- K: sparse grid, e.g. `16, 48, 96, 128`

Main value:

- Shows operating regime and avoids the appearance of cherry-picked budgets.

Main risk:

- Can become too many runs without adding a new mechanism.

## Systems Ideas

### Exact Runtime Breakdown

Use existing artifacts to produce a stacked runtime figure:

- exact query extraction
- evicted scoring
- selection
- transfer
- injection

Expected story:

- exact evicted scoring dominates
- transfer and injection are small
- exact path is mechanistic evidence, not the final deployment path

### Idle-Window Budget Model

Plot available idle time vs max feasible restored tokens under:

- raw transfer/reinjection
- current exact scorer
- proxy or two-stage scorer if implemented

Main value:

- Makes the systems implication concrete.

### CPU Buffer Memory Table

Report bytes/token for the live model configuration and buffer sizes for:

- 1K tokens
- 5K tokens
- 10K tokens
- actual 4Q/6Q evicted pools

Main value:

- Prevents stale or inflated KV memory estimates.
- Shows CPU buffering is feasible at the evaluated scale.

### Recompute Comparison

Small targeted harness or table:

- matched no-repair resume
- full-cache resume
- RepairKV exact repair
- re-prefill full prompt from scratch

Metrics:

- score
- p50 turn-2 latency
- active GPU KV size
- CPU buffer memory

Main value:

- This is the comparison systems reviewers will ask for.

Main risk:

- Re-prefill equivalence and exact timing can be hard to make apples-to-apples.

## Low-Value Or Risky Ideas

### More Dense K Points

The existing K grid already captures onset, rise, and saturation. More K values
will not change the paper unless used for a new algorithmic or latency figure.

### More Tables In Main Text

The main paper should stay compact. New evidence should be figures or concise
appendix tables.

### Second Compressor

Could be valuable later, but it changes the question from “does idle-time repair
work?” to “which compressor is best?” That is probably too much for the current
paper unless implementation is nearly free.

### Learned Scorer

Potentially interesting, but too risky for a few-day sprint. Learned scoring
would require training/evaluation design and would weaken the clean
training-free story.

### Broad New Non-NIAH Benchmark

Very high upside, but high risk. Only attempt if an existing task can be adapted
without disrupting the schedule. Otherwise keep it as future work.

## Preliminary Ranking

Highest-value Phase 9 package:

1. **Future-query specificity:** Q2 vs wrong/stale/contrastive scoring.
2. **Algorithmic repair:** current RepairKV vs IntervalPack/Coverage vs Gold-K.
3. **Quality-latency Pareto:** exact vs proxy/two-stage rerank.
4. **Mechanism/statistics figure:** lift with CI and final-active overlap.
5. **Robustness appendix:** base-budget sweep, seed robustness, recency
   diagnostics.
6. **Systems appendix:** runtime breakdown, idle-window budget model, CPU buffer
   memory.

Best possible main-paper figure set if the runs succeed:

1. Existing MQ-NIAH-4Q/6Q score frontier.
2. New causal/algorithmic figure:
   - panel A: future-query specificity
   - panel B: improved repair policy at mid budgets
3. New systems figure:
   - quality-latency Pareto

Appendix:

- operating-regime heatmaps
- seed robustness
- recency diagnostics
- runtime breakdown
- CPU memory table
- per-split endpoints

## Questions To Resolve Before Execution Plan

1. Should Phase 9 prioritize paper novelty or reviewer robustness?
   - novelty: ContrastiveQ, IntervalPack, TwoStageRerank
   - robustness: seed sweeps, base-budget heatmap, recency diagnostics
2. How much main-paper space can we afford?
   - one extra figure
   - two extra figures
   - one figure plus appendix only
3. Should new algorithms be allowed to replace the current RepairKV line, or only
   appear as ablations?
4. What run budget is acceptable?
   - one overnight GPU job
   - several parallel overnight jobs
   - multi-day run queue
5. Should Phase 9 stay entirely within one-shot 32K repair, leaving strict-cap
   streaming to Phase 8?
