# Phase 10: Robust Dynamic KV-Repair Expansion

Phase 10 is the robustness phase. Phase 7/9 establish the controlled
matched-budget repair effect on Qwen2.5-7B-Instruct with SnapKV-style
turn-N compression. Phase 10 asks whether the effect survives the
dimensions reviewers will naturally probe:

1. More realistic multi-turn relevance shifts.
2. Future-query specificity and novelty-boundary controls.
3. Model-family and compression-policy breadth.
4. Query-count/task breadth.

The compact live map for all high-signal branches is
`phase10_high_signal_map.md`. Treat that file as the source of truth for
what remains live, what is main-paper candidate evidence, and what gates
must pass before a branch enters the paper.

The goal is not to turn the paper into an unfocused benchmark survey.
The goal is to keep every high-signal branch alive until it fails a clear
gate, then promote the strongest few figures into the paper and move
secondary but useful evidence to the appendix.

## Main claim boundary

The current paper should not imply that RepairKV is already validated
across all KV-cache systems. Its strongest present claim is:

> In controlled two-turn MQ-NIAH diagnostics, a compressed cache can be
> stale for the next turn; using idle time to rescore and restore evicted
> KV rows can recover the future-turn answer at the same resumed active
> evictable-context budget.

Phase 10 extends this to a stronger robustness claim:

> Idle-window repair is a general dynamic-cache primitive: it can
> repeatedly adapt a compressed KV cache as relevance changes across
> turns, and it is not tied to one query count, one compressor, or one
> Qwen-family model.

## Paper reframing role

Phase 10 is not only an appendix expansion. If it works, it should
reframe the paper from a narrow "two-turn synthetic repair result" into
a broader technical argument:

> Long-context inference should treat the KV cache as a dynamic state
> that can be compressed and later repaired during idle windows. Repair
> can use information that was unavailable at compression time, such as
> the next turn's query or workflow state, while preserving a matched
> active GPU KV budget.

The earlier phases remain essential because they establish the mechanism
cleanly. Phase 10 decides how far the paper can responsibly generalize.

Main-paper upgrade path:

1. **If multi-turn repair passes:** the abstract and introduction should
   lead with dynamic workflows, not only a two-turn diagnostic. The
   two-turn frontier becomes the controlled mechanism result; the
   multi-turn trajectory becomes the workflow stress test.
2. **If specificity controls pass:** the paper can claim that the gain
   is caused by information revealed after compression, not merely by
   adding more rows from a buffer.
3. **If one new model and one non-SnapKV compressor pass:** the paper can
   claim preliminary portability of the primitive, while still avoiding
   broad benchmark claims.
4. **Precision promotion is not a current main-paper path:** the HQQ
   row-store sweep did not pass the promotion gate. Keep it as
   future-work context unless a redesigned quantization-specific study
   shows a selective repair effect under real or very clearly scoped
   byte accounting.

Main-paper downgrade path:

- If multi-turn is noisy but two-turn remains strong, keep the current
  paper centered on the controlled diagnostic and move multi-turn to
  future work.
- If model or retention-rule breadth fails, state the scope honestly and do
  not dilute the strong Qwen+SnapKV evidence.
- The current precision-promotion sweep is negative: low-bit storage is
  either too destructive to repair selectively (`2/4` bit) or already
  accurate enough that repair is unnecessary (`8` bit). Do not spend
  main-paper space on it.

## What would make the paper outstanding

The strongest possible experimental package is compact:

1. **Core mechanism:** the current 4Q/6Q frontier plus promoted 8Q stress
   frontier shows repair beats matched no-repair under a fixed resumed
   active-cache budget.
2. **Dynamic workflow:** a rolling multi-turn figure shows that repair
   continues to help when relevance shifts over several turns, including
   when a topic returns after being evicted.
3. **Novelty boundary:** a compact control shows that the benefit is not
   captured by stale-query repair, donor-query repair, or simply rerunning
   a query-aware compressor at the next turn.
4. **Portability stress test:** one small model panel and one small
   compressor panel show the primitive is not only a Qwen+SnapKV artifact.
5. **Breadth insurance:** an appendix query-count panel shows the result
   was not chosen only because 4Q/6Q are favorable; full 2Q remains an
   easy-regime boundary unless it improves the main figure without
   clutter.
6. **Compression-format extension:** now future work only. The completed
   HQQ row-store sweep is useful as a negative boundary, but it is not a
   clean paper result because selective promotion did not beat static,
   random, or oldest precision controls.

If Phase 10 can deliver item 2 cleanly, that should outrank most extra
tables. Multi-turn dynamics are closer to the paper's thesis than simply
adding 2Q/3Q/8Q curves.

## Expert stress-test synthesis

The expert critiques converge on a clear priority order.

- The KV-cache critique says the main missing technical evidence is not
  more query counts. It is whether repair remains useful after repeated
  compression/repair cycles, and whether the future query is causally
  necessary rather than a generic reinsertion signal.
- The AdaptFM critique says the paper should avoid becoming a benchmark
  survey. One clean cross-family model or compressor result is more
  reviewer-relevant than many extra synthetic variants, provided cache
  round-trip and memory accounting are validated.
- The systems critique says every promoted result must preserve the
  matched active GPU budget, report the CPU evicted-KV buffer, record
  artifact provenance, and use paired uncertainty. Fixed token budgets
  alone are not enough for cross-model claims.

Resulting Phase 10 priority after adversarial review:

1. Stale/donor/Refresh-K controls for next-turn signal specificity and
   novelty boundary.
2. Rolling multi-turn repair with revisit recovery as a stress test of
   repeated adaptation, not as an end-to-end agent claim.
3. One validated non-SnapKV compressor smoke, written as structural
   retention robustness unless another content-aware compressor is added.
4. One validated model-transfer smoke only after cache round-trip and
   exact/proxy scorer checks pass.
5. Query-count breadth as appendix insurance unless it is exceptionally
   clean.
6. Precision-promotion repair as a high-upside appendix branch, not a
   blocker for the main Phase 10 evidence.

## Why 4Q/6Q stayed main

4Q and 6Q are the best main-paper pair because they are interpretable
and not redundant.

- 4Q is the calibrated base setting: a two-turn 2-to-2 split family with
  multiple non-tail future partitions. It is easy to explain and gives a
  smooth restore-budget frontier.
- 6Q is the higher-query-count stress setting: a 3-to-3 split family
  where the second turn excludes the latest two needles, reducing the
  chance that results are explained by recency.
- 2Q is useful as a sanity check but too easy and too small to carry the
  main story alone.
- 3Q is useful for continuity but currently has only one clean split, so
  it is weaker as a main result than 4Q.
- 8Q is valuable if it works, but it is a stress test: longer answers,
  more splits, larger decode budget, and more ways for exact-match
  scoring to become brittle. It belongs in appendix or as a compact
  breadth figure unless it is very clean.

The likely final paper structure is therefore:

- Main: 4Q/6Q frontier plus controls.
- Main or appendix: multi-turn dynamic repair trajectory if the smoke
  passes.
- Main or appendix: future-query specificity / refresh baseline if it is
  clean.
- Appendix or compact main inset: one new-model result and one
  non-SnapKV compressor result, each with locked budgets and artifact
  provenance.
- Appendix: query-count breadth across 2Q/3Q/4Q/6Q/8Q at one or two
  restore budgets.
- Appendix or future-work result: quantized-cache precision promotion if
  the cheap low-bit row-store smoke shows a real effect.

## Multi-turn dynamic repair

Goal: test the real thesis more directly. Agent workflows are not always
"compress once, answer one follow-up." They can shift topic repeatedly:
tool result, user correction, failed test, new file, return to an older
file, then another topic. A good Phase 10 experiment should show whether
repair helps under repeated relevance changes.

Core benchmark: rolling MQ-NIAH with turn schedule.

- Build a context with `G` key/value needles.
- Define a sequence of turns `Q_1, Q_2, ..., Q_T`, where each turn asks
  for a subset of keys.
- After each answer, compress to the same base active-context budget
  `B_base`, keep the rest in a CPU evicted-KV buffer, and before the next
  answer restore up to `K` rows scored by the newly known turn query.
- The active evictable-context budget before every answer must be
  matched to the no-repair baseline.
- Use `T=3` as the minimal scientific test and `T=4` as the preferred
  paper test. Longer runs are useful only after the accounting is proven,
  because they introduce answer-format, churn, and accumulated-error
  confounds.
- The intended effect is not "accuracy always increases." The useful
  story is conditional: repair should help after relevance shifts and on
  revisits, while it should provide little gain when the compressed cache
  is already saturated or the next query is unchanged.

Candidate schedules:

1. **Shift:** `late -> early -> middle -> early`.
   - Shows topic shift and return.
   - Example 8Q schedule:
     `Q1={7,8}`, `Q2={1,2}`, `Q3={4,5}`, `Q4={1,2}`.
2. **Sweep:** `tail -> body-left -> body-right -> tail`.
   - Tests whether repair can restore old content and later return to
     recency-favored content without permanent damage.
3. **Interleaved agent-like:** answer a retrieval turn, append a short
   synthetic "tool result" segment, then ask about a different file/key
   group.
   - Closer to the paper narrative, but it should come after the pure
     schedule is validated.

Conditions:

- Full cache.
- Matched no-repair with the same active evictable-context budget before
  each turn.
- RepairKV rolling repair.
- Random-K and Oldest-K rolling restores for promoted runs.
- StaleQ-K, scored with the previous turn's query, if available; this is
  the strongest causal control for "new turn text matters."
- Gold-K as a hindsight reference over annotated current-turn span
  groups.

Metrics:

- Per-turn exact score.
- Per-turn score gain over matched no-repair.
- Area under the per-turn score curve.
- Current-turn span overlap after repair.
- Repair churn: number of tokens moved per turn and fraction of restored
  tokens that change between consecutive turns.
- Revisit recovery: score on a topic when it returns after at least one
  intervening turn.
- Harm rate: number of turns where repair underperforms matched no-repair.

Promotion gates:

- RepairKV beats matched no-repair on at least two non-initial turns.
- RepairKV specifically helps on a revisit turn, not only on a single
  first shift.
- Random-K/Oldest-K do not match RepairKV after promotion.
- The result remains interpretable: no turn should fail because of
  answer-format brittleness or decode-budget truncation.

Potential figures:

- **Main candidate:** one-column trajectory plot. X-axis is turn index;
  y-axis is score or score gain. Lines: Full, Matched, RepairKV, Gold-K;
  add Random/Oldest only if visual space allows.
- **Cool dense candidate:** cache-state heatmap. Rows are needle groups;
  columns are turns; cells mark whether the group is active before
  answer under matched no-repair versus RepairKV. This directly visualizes
  "repair follows shifting relevance."
- **Appendix diagnostic:** churn plot, with bars for moved KV rows per
  turn and line for score gain.

Implementation plan:

1. Add a multi-turn task generator without changing the current two-turn
   runner.
2. Unit test schedule construction, query-key/output mapping, and
   per-turn span metadata.
3. Implement a CPU-only synthetic cache test for rolling compression and
   repair accounting before any GPU run.
4. Run a GPU smoke with `T=4`, `G=8`, `n=1`, `B_base` near the current
   6Q operating budget, and `K={48,96}`.
5. Promote only if the smoke shows repeated-turn signal and no artifact
   from answer formatting.

Do not replace the current two-turn results with multi-turn results.
The multi-turn result should be framed as a stronger stress test of the
same primitive.

## Novelty-boundary baselines

Goal: answer the strongest "is this just an existing mechanism under a
new name?" objection.

Baselines:

- `StaleQ-K`: repair using the previous turn's query or answer signal,
  then evaluate on the new turn. This tests whether new turn text is
  actually needed.
- `WrongQ-K` / donor query: repair using a plausible but wrong future
  query. This tests specificity rather than generic reinsertion.
- `Refresh-K`: recompute or rerun the first-stage query-aware selection
  after the next-turn query is known, under the same resumed active-cache
  budget. This is the cleanest comparison against "why not just query
  aware compression at Q2 time?"
- `Refresh-buffered`: a bounded implemented form of `Refresh-K` that
  reselects the whole resumed context budget from active plus CPU-buffered
  context KV rows using the same Q2 scoring rows as RepairKV. It does not
  recompute prefix KV and should be reported separately from a full-prefix
  recompute baseline.
- `Refresh-recompute`: a stronger future comparator that recomputes or
  reloads the full prefix before Q2-time selection. If it wins, the paper
  should narrow the systems claim to low-recompute paused-cache repair and
  report recompute cost explicitly.
- `No-buffer query-aware selection`: if the original evicted KV rows are
  unavailable, run query-aware selection only over the compressed active
  cache. This should be weaker and highlights why retaining a CPU
  evicted-KV buffer matters.

Recommended first implementation:

- Use the existing `StaleQ-K` and `WrongQ-K` machinery first.
- For `Refresh-K`, first run the implemented `Refresh-buffered`
  comparator on the same 4Q operating point before attempting a full
  suite. Add `Refresh-recompute` only if the buffered comparison is
  decisive enough to justify the extra GPU time.
- Report GPU active budget, CPU buffer size, and whether the baseline
  requires recomputing prefix KV. If it recomputes the prefix, it is not
  the same systems point and must be labeled as such.

Potential figure:

- One-column specificity panel at a single operating point or over
  `K={48,96}`.
- Bars or paired dots: Matched, StaleQ-K, WrongQ-K, Refresh-K, RepairKV,
  Gold-K.
- Caption should state which baselines use newly available Q2 and which
  require recomputation.

Promotion gates:

- RepairKV must beat stale/donor query controls by at least 0.10 at a
  promoted operating point.
- If Refresh-buffered beats RepairKV, the result is still valuable but
  changes the claim: RepairKV is a low-recompute buffer repair primitive,
  not the absolute best Q2-time selector over all buffered rows.
- If all Q2-time selectors saturate, move this panel to appendix and use
  latency/recompute cost as the differentiator.

Immediate decision rules after the smoke:

- If `K=48` separates Matched/Stale/Wrong from RepairKV and Gold-K still
  has headroom, run a larger locked specificity experiment at `K=48`.
- If `K=96` saturates but `K=48` separates, use `K=48` for the promoted
  figure and keep `K=96` as a saturation diagnostic.
- If both `K=48` and `K=96` saturate, rerun a smoke at lower `K`
  (`K={16,32,48}`) before spending on a full specificity run.
- If StaleQ-K matches RepairKV, do not claim next-turn signal specificity;
  demote the panel and revert to the narrower matched-budget repair
  story.
- If donor WrongQ-K alone matches RepairKV, do not overinterpret it:
  MQ-NIAH queries are templatic, so the stale-query and Refresh-buffered
  comparisons are the stronger evidence.
- If Refresh-buffered clearly wins, keep it as a main novelty-boundary
  result only if the caption reports that it reselects the whole resumed
  context budget from active plus CPU-buffered rows without full-prefix
  recompute.

## Failure-mode panel

Goal: show reviewers that the paper understands when repair should help
and when it should not.

Failure or low-gain regimes:

- No relevance shift: Q2 asks for the same content as Q1.
- Recency-saturated: the base compressed cache already retains the
  future-turn spans.
- Too-small base cache or too-small K: the needed spans are unavailable
  or cannot fit.
- Wrong/stale future signal: repair targets the wrong subset.
- Fragmented evidence: a future answer requires many small spans and a
  burst-style selector wastes restore budget.
- Repeated-repair harm: rolling repair churns useful state and hurts a
  later return turn.

Potential figure:

- One-column "where repair should not help" strip plot.
- X-axis: regime; y-axis: score gain over matched.
- The ideal pattern is not all positive. A strong paper shows high gain
  only in the intended stale-cache regime and near-zero gain in
  saturated/no-shift regimes.

## Algorithmic upgrade: coverage-aware packing

Goal: add method novelty if experiments show a persistent gap to
Gold-K.

Current RepairKV scores positions and expands local bursts. Phase 7/9
results suggest some remaining gap may come from packing inefficiency:
the selector can spend K on redundant neighboring rows or miss multiple
answer spans.

Candidate algorithm:

- `CoverageRepairKV`: score candidate anchor positions with the Q2 scorer,
  then greedily choose intervals to maximize coverage of separated high
  score mass under the restore budget.
- Penalize overlap with already selected intervals.
- Allow smaller local windows when evidence is fragmented.
- Keep stable tie-breaking by absolute position.

Unit tests:

- Fragmented toy scores: coverage-aware selector should choose two
  separated peaks where burst selector chooses one redundant block.
- Contiguous toy scores: coverage-aware selector should match burst
  selector.
- Budget and determinism tests.

Promotion gate:

- Only promote if it beats RepairKV by at least 0.05 at mid-budget without
  hurting high-budget saturation or latency substantially.

## Query-count breadth

Goal: answer the reviewer question "does this only work for the chosen
4Q/6Q construction?"

Priority: appendix breadth unless the result is unusually clean. This
axis is confounded by different split geometry, answer length, decode
budget, and base-budget calibration, so it should not displace
multi-turn or novelty-boundary evidence.

Implemented task aliases:

- `mq_niah_2q_clean_suite`: one 1-to-1 split, tail Q1 to body Q2.
- `mq_niah_3q_clean_suite`: existing one 1-to-2 split, tail Q1 to body
  Q2.
- `clean_suite`: existing 4Q clean suite, three 2-to-2 splits.
- `mq_niah_6q_clean_suite`: existing 6Q clean suite, four 3-to-3 splits.
- `mq_niah_8q_clean_suite`: five 4-to-4 splits that keep the latest two
  needles in Q1 and exclude them from Q2.

Smoke design:

- Budgets: choose one operating budget per query count, not a full phase
  diagram.
- Restore budgets: `K = 16, 48, 96, 128`.
- Conditions: `A B B_match RepairKV Oracle-K` for smoke; add
  `Random-K Oldest-K` only for promoted final runs.
- Samples: `n=2` for smoke, then `n=12` or `n=24` if the curve is
  monotone and the matched baseline remains meaningfully below full.

Smoke outcome on 2026-05-03:

- 2Q is recoverable but quickly saturates. The clearest budget is
  `B_base=12288`, where matched no-repair is `0.50` and RepairKV reaches
  `1.00` by `K=48`; `B_base=14336` is saturated and should not be used.
- 3Q is recoverable with more K. The clearest budgets are
  `B_base=14336`, where matched no-repair is `0.00` and RepairKV rises
  from `0.25` at `K=48` to `0.75` at `K=96`, and `B_base=12288`, where
  the curve reaches `1.00` only at `K=128`.
- 8Q is the highest-value breadth candidate. At `B_base=18432`, matched
  no-repair is `0.60`; RepairKV is weak at `K=16`, modest at `K=48`,
  and nearly saturates by `K=96`. This gives a visually clean
  task-difficulty extension if it holds under a locked run.

Minimal locked follow-up if breadth becomes a priority:

- Tasks/budgets: 2Q at `B_base=8192`, 3Q at `B_base=14336`, 8Q at
  `B_base=18432`, plus the already-final 4Q/6Q curves for context.
- Restore budgets: `K={16,48,96,128}`.
- Conditions: `A`, `B`, matched no-repair, Random-K, Oldest-K, RepairKV,
  Gold-K.
- Samples: `n=12` for an appendix figure; `n=24` only if the figure is
  likely to replace an existing main-text figure.
- Expected figure: one-column query-count scaling panel with query
  count on x-axis and score gain over matched no-repair on y-axis,
  using one line for `K=48` and one for `K=96`, with Gold-K headroom
  markers. Promote to main only if all query counts show positive
  RepairKV gain and content-agnostic controls stay near matched.

Promotion gates:

- Matched no-repair is below full by at least 0.20 at the chosen budget.
- RepairKV beats matched by at least 0.15 at `K=48` or `K=96`.
- Gold-K remains at or above RepairKV, or the gap is explainable by
  saturation.
- Random-K/Oldest-K stay near matched in promoted runs.

Potential figure:

- One-column "query-count breadth" small multiple.
- X-axis: query count `{2, 3, 4, 6, 8}`.
- Y-axis: score gain over matched.
- Two curves: `K=48` and `K=96`, optionally with faint Gold-K headroom
  markers.
- This figure is high-density and does not repeat Figure 2 because it
  changes the x-axis from restore budget to task difficulty.

## Compression-policy breadth

Goal: separate the repair primitive from the particular first-stage
SnapKV-style retention rule.

Minimum Phase 10 requirement: run at least one non-SnapKV first-stage
retention-rule smoke. The first target should be sink-plus-recent retention
because it is already implemented locally and changes the failure mode from
attention-score retention to sink-plus-recent structural retention.

Sink-plus-recent retention smoke outcome on 2026-05-03:

- The `n=1` smoke is not paper-grade. RepairKV does not improve over
  matched no-repair at `B_base=8192`; at `B_base=12288` and `16384`, it
  improves only from `0.333` to `0.500` at `K=96`, while Gold-K remains
  `1.000`.
- Random-K and Oldest-K stay at matched no-repair, so the small gain is
  directionally compatible with the repair story, but it is too weak and
  too underpowered for a retention-rule portability claim.
- Do not promote a sink-plus-recent panel unless a follow-up recalibrates the
  base budget and demonstrates a larger, repeated gain with `n>=12`.

Priority: one structural baseline plus one content-aware baseline, not a
policy zoo. For every retention rule, recalibrate `B_base`, keep resumed GPU
active budget fixed, and report offloaded-store size. If possible, report KV
bytes or budget fraction in addition to token count.

Priority policies:

1. SnapKV, current baseline.
   - Already implemented and paper-ready.
   - This remains the main compressor because it is standard, simple,
     and query-agnostic at compression time.

2. Sink-plus-recent structural retention inspired by StreamingLLM.
   - Already implemented in Phase 3.
   - Good negative/structural baseline: if the first-stage cache keeps
     only sink and recency, future repair should have large opportunity
     but may need enough evicted context.
   - Cheap to add because the policy exists locally.

3. H2O-inspired accumulated-attention heavy hitters.
   - High-value comparator because it is a canonical eviction baseline.
   - Needs implementation. It should score tokens by accumulated
     attention mass over the observed prompt or turn-N continuation,
     then keep sinks, recency, and heavy hitters.
   - Unit tests: deterministic toy attention weights, budget accounting,
     stable tie-breaking, evicted CPU cache.

4. PyramidKV / Ada-KV-like layer budget allocation.
   - Useful if we want the paper to speak to current KV compression
     methods rather than only token-level global eviction.
   - High implementation risk because our repair machinery currently
     assumes a single global set of context positions. Start with a
     simplified policy that preserves the same position set across
     layers, then only promote real layer-varying retention if injection
     semantics are fully tested.

5. QUEST query-aware paging.
   - Strongly related but not the cleanest initial compressor for this
     paper, because Quest uses query-time information during attention.
     It is still important for the novelty boundary: if included, frame
     it as query-time retrieval/loading rather than post-compression
     pre-resume repair.

Out of scope for immediate Phase 10:

- Full real-kernel quantized KV methods such as KIVI/TurboQuant. Phase
  10 may run a low-bit row-store precision-promotion smoke, but real low-bit
  cache kernels and production memory claims require a separate
  implementation path.
- Full distributed serving methods unless we add a system prototype.

Potential figure:

- One-column grouped bar or small-multiple line figure.
- Rows/panels: initial retention rule `{SnapKV, sink-plus-recent,
  accumulated-attention}`.
- X-axis: restore budget `K`.
- Y-axis: score gain over matched.
- Keep only 4Q or 6Q for this figure to avoid exploding the grid.
- If only one non-SnapKV compressor is promoted, use paired dots rather
  than a large grouped bar: `SnapKV` and sink-plus-recent retention at
  `K=96`, with matched and Gold-K references.

Minimum smoke:

- Qwen2.5-7B-Instruct, 4Q only, one locked operating budget after a
  tiny calibration sweep.
- Conditions: `A B B_match RepairKV Random-K Oldest-K Oracle-K`.
- `K={48,96}`, `n=2` for smoke, then `n=24` if the matched baseline is
  non-saturated and RepairKV beats random/oldest.
- Promotion gate: the non-SnapKV effect should be positive at both K
  values, and the paper must say the compressor changes only the
  first-stage retained rows; the idle repair operation is unchanged.

## Model breadth

Goal: show the repair effect is not a Qwen-only artifact.

Minimum Phase 10 requirement: run at least one new-model smoke after the
runtime supports model selection. A model result is paper-grade only if
tokenizer rendering, cache round-trip, query-row extraction, and memory
accounting are all validated.

Model tiers:

1. Qwen2.5-7B-Instruct.
   - Current anchor model; already local and validated.

2. Qwen2.5-14B-Instruct or Qwen2.5-0.5B-Instruct.
   - Same architecture family, useful to check scale effects.
   - 14B is higher signal if memory permits; 0.5B is useful only as a
     very cheap debugging/model-size trend.

3. Llama-3.1-8B-Instruct.
   - Best cross-family peer for 7B-scale long-context instruction
     models. It changes tokenizer, RoPE, GQA details, and chat template.

4. Mistral-Nemo-Instruct-2407.
   - Good second cross-family model if compute permits. It is 12B and
     has a 128K context window, so it tests a larger model with a
     different stack.

5. Gemma-family model.
   - Lower priority for this paper unless local access is easy. Useful
     only if we need another architecture family.

Engineering work before model runs:

- Add `--model-dir` to Phase 6 runner or route through an environment
  variable instead of hard-coding `models/Qwen2.5-7B-Instruct`.
- Add tokenizer/chat-template smoke tests for each model.
- Add a single-cache round-trip test per model:
  full-cache decode equals resumed-cache decode on a tiny prompt.
- Confirm query-row extraction works for the model architecture after
  RoPE. This is the highest-risk part for Llama/Mistral.
- Store model id, model dir, dtype, context length, and tokenizer hash in
  every artifact.

Model smoke design:

- Use only 4Q at first, `B=16384`, `K={48, 96}`, `n=2`.
- Conditions: `A B B_match RepairKV Oracle-K`.
- Promote a model only if full-cache Q2 is high and matched no-repair is
  meaningfully lower.
- Promoted model panels need paired uncertainty and model-specific
  memory accounting. Do not compare fixed token budgets as if they imply
  the same KV footprint across architectures.

Potential figure:

- One-column "model transfer" dot plot.
- X-axis: model.
- Y-axis: score gain over matched at `K=96`.
- Dot: RepairKV; hollow dot: Gold-K headroom or full-cache reference.
- This is compact, reviewer-friendly, and avoids multi-page tables.
- If only one new model is promoted, use three points: Qwen2.5-7B anchor,
  same-family scale check if cheap, and one cross-family model if
  available. If only two models are available, keep it appendix unless
  the result is exceptionally clean.

Minimum smoke:

- First choice: Llama-3.1-8B-Instruct or another local 7B/8B-class
  long-context instruction model.
- Fallback: Qwen2.5-14B-Instruct if memory permits, or Qwen2.5-0.5B only
  as an engineering debug run, not as a main robustness claim.
- Use 4Q, one locked budget, `K={48,96}`, `n=2`.
- Promote only after full-cache Q2 quality is high, matched no-repair is
  meaningfully below full, and a tiny cache round-trip test passes.

## Quantized-cache precision promotion

Goal: test whether the repair primitive also applies to memory-format
compression, not only row eviction. The high-level idea is:

> Keep a low-bit compressed KV cache active, retain high-precision
> originals in a side buffer, then use idle time after the next query
> arrives to promote selected rows back to high precision.

This is a different axis from SnapKV/StreamingLLM row eviction. It
should be framed as **precision-promotion repair**, not as a new KV
quantization method.

Why this is interesting:

- Existing KV quantization work shows the cache memory format itself is
  a major bottleneck. KIVI studies asymmetric 2-bit KV quantization;
  KVQuant studies low-bit KV quantization with per-channel keys,
  pre-RoPE keys, and non-uniform/sparse handling; KVTuner studies
  mixed-precision layer-wise KV quantization; MiKV is especially close
  because it keeps less-important KV in low precision while preserving
  important KV at higher precision. MixKVQ is an even sharper boundary:
  it is query-aware mixed-precision KV quantization, so RepairKV should not
  claim novelty for query-aware precision assignment in general.
- The possible RepairKV angle is temporal rather than static: after the
  future turn arrives, the system can decide which low-precision rows
  deserve high-precision replacement under a fixed byte budget.
- This supports the broader thesis that idle windows can repair
  compressed KV state dynamically, whether compression drops rows or
  lowers precision.

Prior boundary:

- Do not claim novelty for "mixed precision KV cache" or "low-bit KV
  cache quantization." Those are established areas.
- The novel diagnostic would be query-conditioned, post-compression
  **precision promotion** under an idle window and matched byte budget.
- If the result is only a low-bit row-store smoke, say so. It stores
  integer low-bit rows and scale metadata, but materializes them back to
  model dtype before attention, so it tests quality and byte accounting
  rather than real low-bit attention latency.
- The clean novelty boundary is **post-compression, next-turn precision
  promotion**. If a baseline already chooses high-precision rows from the
  current query before or during attention, it is a related query-aware
  quantization method, not the same paused-cache repair setting.
- Source-checked boundary notes, 2026-05-03:
  - KIVI: `https://arxiv.org/abs/2402.02750`
  - KVQuant: `https://arxiv.org/abs/2401.18079`
  - MiKV: `https://arxiv.org/abs/2402.18096`
  - KVTuner: `https://arxiv.org/abs/2502.04420`
  - MixKVQ: `https://arxiv.org/abs/2512.19206`
  - Hugging Face QuantizedCache docs:
    `https://huggingface.co/docs/transformers/main/kv_cache`

Cheap smoke before any custom row-replacement cache:

- Store selected K/V context rows with a packed HQQ row store at
  `int2`, `int4`, and `int8` where useful. Keep the older symmetric
  row quantizer only as a deterministic unit-test path.
- Store or account for the original high-precision KV tensors as a CPU
  side buffer.
- Materialize dequantized tensors only at the model attention boundary.
  This is still quality-only for latency, but it is closer to the real
  storage path than pure FP quantize/dequantize tensors.
- Conditions:
  - `Full-fp16`: original high-precision cache.
  - `LowBit-all`: all evictable rows quantized/dequantized.
  - `StaticMixed`: sinks/recent or SnapKV-selected rows high precision,
    the rest low precision.
  - `RepairKV-Precision`: low-bit base plus Q2-selected high-precision
    row replacement.
  - `Random-Precision`, `Oldest-Precision`, and `Gold-Precision`
    controls if the smoke passes.
- Use 4Q first, one locked budget, `K={48,96}` high-precision promoted
  rows, `nbits={2,4}`, `n=2`.
- Preferred first locked smoke after unit tests:
  - Task: MQ-NIAH-4Q balanced clean suite.
  - Base setting: no row eviction for the first precision smoke; quantize
    the evictable context rows and keep Q1 tail rows high precision.
    Reuse row-eviction budgets only after the precision-only smoke shows
    a measurable degradation/repair gap.
  - Conditions: `Full-fp16`, `LowBit-all`, `StaticMixed`,
    `Random-Precision`, `Oldest-Precision`, `RepairKV-Precision`,
    `Gold-Precision`.
  - Selection: use the same Q2 scorer as RepairKV for `RepairKV-Precision`;
    use sink/recent or Q1/SnapKV importance for `StaticMixed`.
  - Samples: `n=2` smoke, then `n=12` appendix candidate only if the
    gate below passes on the smoke.

Metrics:

- Q2 exact score.
- Gain over matched low-bit baseline.
- Effective KV bytes: low-bit bytes plus high-precision promoted bytes
  plus high-precision side-buffer bytes if the side buffer is counted.
- Promotion hit rate on annotated future-turn spans.
- Quality/byte Pareto: score versus active high-precision-equivalent KV
  bytes.

Potential figure:

- One-column quality/byte Pareto.
- X-axis: active KV byte budget or high-precision-equivalent bytes.
- Y-axis: Q2 score.
- Points: LowBit-all, StaticMixed, Random-Precision, RepairKV-Precision,
  Gold-Precision, Full-fp16.
- This could be visually strong, but it should enter the paper only if
  it is clean and clearly distinct from existing mixed-precision work.

Promotion gates:

- `LowBit-all` must meaningfully degrade at `int2` or aggressive `int4`;
  otherwise there is no room to repair.
- `RepairKV-Precision` should beat `LowBit-all` by at least `0.10` and
  beat `StaticMixed`, `Random-Precision`, and `Oldest-Precision` by at
  least `0.10` at the same active byte budget.
- `Gold-Precision` should show headroom or validate that the selected
  precision rows are causally important.
- The paper must separate active GPU byte budget from optional CPU
  high-precision side-buffer bytes.
- Use `recommend_precision_promotion.py` on the compact CSV. Low-bit
  row-store rows can be appendix/future-work evidence; a main-text
  precision-promotion result requires a real quantized cache path or
  very explicit language that no low-bit attention kernel is claimed,
  plus paired uncertainty.

Real implementation path and current status:

- Try Hugging Face `QuantizedCache` with `quanto` or `hqq` backends as a
  reference implementation, noting that it supports low-bit cache modes
  but may not support arbitrary high-precision row replacement.
- If row replacement is hard inside `QuantizedCache`, keep the real
  implementation out of the main paper and present the row-store result
  as a future-work diagnostic.
- Current environment check, 2026-05-03: Transformers `5.2.0` exposes
  `QuantizedCache`; `optimum-quanto==0.2.7` and `hqq==0.2.8.post1` are
  installed. A tiny Qwen2.5-7B-Instruct generation smoke passes with
  `QuantizedCache(backend="hqq", nbits=4)`. The `quanto` backend imports
  but its CUDA extension path requires a full CUDA toolkit layout, so it
  is not the practical backend for this repo right now.
- The real `QuantizedCache` smoke establishes that a low-bit KV cache can
  run in this environment. It does not by itself provide arbitrary
  per-row high-precision promotion.
- The row-store precision-promotion smoke now uses HQQ packed
  quantization metadata for selected K/V rows. Tiny diagnostics show that
  partial repair at `K={96,192}` does not recover a 4k 2Q example, while
  all-row promotion recovers to full score.
- The active budget sweep completed on 2026-05-03 with
  `nbits={2,4,8}` and `K={96,192,512,1024,2048,4096}`. Result: do not
  promote. At 2-bit and 4-bit, `LowBit-all` falls to zero and partial
  high-precision promotion does not recover quality; at the all-row
  limit every control recovers equally. At 8-bit, `LowBit-all` already
  preserves the answer, so no repair gap exists. Treat this as a
  quantization-specific future-work note, not evidence for the current
  paper's main thesis.
- Local source inspection: Transformers `QuantizedLayer` stores older
  cache segments in `_quantized_keys/_quantized_values` and keeps a
  residual high-precision tail, then dequantizes the quantized segment
  when returning keys/values for attention. It does not expose an obvious
  public API for arbitrary row-level high-precision promotion, so row
  replacement likely needs either private-cache surgery or a custom
  cache class.

## Statistical and artifact gates

Promoted Phase 10 figures should meet these gates before entering the
main paper:

- Use paired examples whenever conditions share examples.
- Report bootstrap or paired uncertainty for main promoted panels.
- Log model id, model dir, tokenizer hash, dtype, context length, budget,
  `K`, query scoring mode, oracle mode, condition list, split ids, sample
  count, and seed offset in every artifact.
- Report active GPU KV budget and CPU evicted-KV buffer size for any
  systems-facing result.
- Do not retune budgets after inspecting full-run outcomes. Tune on
  smokes, then lock promoted budgets.
- For cross-model results, report KV bytes or budget fraction, not only
  token count.
- For proxy/runtime results, label proxy scoring as a heuristic if it
  outperforms exact query scoring on synthetic structure. Do not call it
  an approximation unless the behavior tracks exact scoring across the
  sweep.

## Execution order

1. Finish Phase 9 6Q operating-regime run.
2. Keep the queued query-count smoke as a background appendix check; do
   not let it block higher-value Phase 10 design work.
3. Design and unit test the multi-turn rolling repair protocol.
4. Run the smallest multi-turn smoke before promoting any broad full
   run.
5. Add or expose stale-query, donor-query, and bounded refresh controls
   at one locked operating point.
6. Summarize 2Q/3Q/8Q smokes and decide whether to promote breadth to
   appendix or discard noisy variants.
7. Implement Phase 6 configurable `--model-dir`; unit test artifact
   provenance.
8. Add sink-plus-recent as a first-stage retention option; unit test budget
   accounting and repair compatibility.
9. Run the required 4Q new-model smoke using the locally cached
   Qwen2.5-0.5B-Instruct model; do not promote unless full-cache
   accuracy makes the repair comparison interpretable.
10. Run the required sink-plus-recent retention smoke.
11. Implement the low-bit-rowstore precision-promotion unit tests and CPU
    synthetic smoke while GPU jobs are occupied.
12. If model or retention-rule breadth works, run one promoted compact
    robustness panel.
13. Only then implement accumulated-attention first-stage retention inspired
    by H2O or a real
    QuantizedCache integration.

## Current recommendation

For the current submission, keep the promoted 4Q/6Q/8Q main frontier
unless a new run is clearly stronger. The highest-value additions are
now:

1. Next-turn specificity contrast with stale/donor/Refresh-buffered
   controls. The locked `n=24`, `K=48` run passed and is integrated into
   the main paper as specificity and method-boundary evidence.
2. Query-count breadth figure in the appendix, now strengthened by the
   promoted 8Q frontier and the active full 2Q easy-boundary run. Move
   2Q to the main frontier only if the full curve is legible and adds
   interpretive value beyond appendix breadth.
3. Multi-turn dynamic repair trajectory, if the protocol and smoke are
   clean.
4. One cross-model follow-up only if a model with nonzero full-cache
   accuracy is available. The cached Qwen2.5-0.5B-Instruct smoke is
   negative and should not appear in the paper.
5. A stronger retention-rule follow-up only after calibrating a
   content-aware alternative such as H2O-inspired accumulated attention.
   The sink-plus-recent `n=1` smoke is too weak for a portability claim.
6. Precision promotion should now stay in future work. The HQQ row-store
   sweep did not beat static/random/oldest precision controls under a
   matched active byte budget, so it is not a current submission result.

## Current implementation status

- Implemented `--model-dir` in the Phase 6 runner so Phase 10 can smoke
  a new model without code edits. Artifacts include a model label when
  the model differs from the default Qwen2.5-7B-Instruct path.
- Implemented `--initial-compressor streaming_llm` in the Phase 6 runner
  as a structural sink-plus-recent first-stage retention rule. SnapKV remains the
  default.
- Added CPU-only low-bit-rowstore precision-promotion utilities and unit
  tests. They store integer low-bit row codes plus scale metadata and
  materialize dequantized rows for attention, so they support
  quality/byte smokes but not real low-bit attention latency claims.
- Added a tested precision-promotion promotion gate and
  `recommend_precision_promotion.py`. It requires meaningful low-bit
  degradation, RepairKV promotion beating static/random/oldest precision
  controls at the same active byte budget, Gold-Precision consistency,
  and explicit byte accounting. Row-store rows are automatically
  appendix/future-work evidence unless a real quantized-cache path is
  also validated.
- Added pure multi-turn schedule helpers for the two preferred 8Q
  schedules: `late -> early -> middle -> early` and
  `tail -> body-left -> body-right -> tail`. Unit tests validate span
  names, immediate-turn overlap, and revisit events before any GPU run.
- Added pure multi-turn accounting helpers for key-level active-state
  recovery, churn, and cache-state grids. These provide tested inputs for
  a future trajectory or cache-state heatmap before implementing the GPU
  rolling runner.
- Added a pure multi-turn score-trajectory summarizer for paired score
  gain, non-initial-turn gain, revisit-turn gain, and matched win rate.
  This is the gate a future rolling smoke should pass before any
  multi-turn figure enters the paper.
- Added the bounded `Refresh-buffered` comparator, specificity summary
  CSV export, automatic follow-up recommendation, and paper-style
  specificity dot-plot renderer. The comparator is quality/specificity
  evidence only unless reimplemented with a systems-fair buffered
  materialization path.
- Specificity smoke and locked follow-up completed. The locked result is
  integrated as a main specificity contrast: RepairKV beats matched by
  `+0.326` with positive paired-gain CI, stale/donor controls stay near
  matched, and Refresh-buffered/Gold-K expose remaining algorithmic
  headroom.
- Added `run_model_transfer_smoke.sh` for the locally cached
  Qwen2.5-0.5B-Instruct model. The smoke completed but full-cache
  accuracy was zero, so it is not usable evidence.
- Sink-plus-recent retention smoke completed under exact-Q scoring and Gold-K
  reference. The result is directionally positive only at `K=96` and too
  weak for a retention-rule portability claim.
- Query-count smoke completed and motivated locked breadth follow-ups.
  The `n=12` 2Q/3Q/8Q run is complete and suggests 2Q is appendix-only,
  while 3Q/8Q are usable breadth candidates. A stronger endpoint run is
  active in tmux as `phase10_even_query_locked_n24`: 2Q at `B=8192` and
  8Q at `B=18432`, `n=24`, `K={48,96}`, exact-Q scoring, Gold-K,
  Random-K, and Oldest-K controls. Promote 2Q/8Q to the main Figure 2
  only if the endpoint curves add a clear scaling story without weakening
  the compact 4Q/6Q main result.
- HQQ row-store precision promotion completed as a negative branch. It
  validates that the implementation can round-trip when all rows are
  promoted, but it does not show selective RepairKV repair under low-bit
  storage. Keep it out of the main paper.

## Red flags to avoid

- Do not make the main paper a table dump. A top workshop paper can have
  many experiments, but the main text should keep only the figures that
  answer distinct reviewer questions.
- Do not call multi-turn results "agent performance." They are still
  controlled diagnostics; they are more agent-like because relevance
  changes across turns.
- Do not promote 2Q if it saturates. It is a sanity check, not a proof.
- Do not promote 8Q if answer-format noise dominates the signal.
- Do not mix model breadth and retention-rule breadth in one large grid. Run
  one axis at a time, then choose the highest-signal compact figure.
- Do not compare against query-aware within-turn systems as if they solve
  the same problem. If included, frame them as related retrieval-time
  mechanisms, while RepairKV is post-compression pre-resume repair.
- Do not claim the quantized-cache branch as a real memory or latency
  result if it uses the low-bit row-store materialization path. That
  smoke tests whether precision promotion has a quality signal and
  reports estimated packed bytes; it does not test low-bit attention
  kernel latency.

## Next Phase 10 Iteration Queue

This queue is deliberately narrow. The paper needs one or two strong
robustness extensions, not a broad but shallow grid.

### Model Breadth

Goal: show the repair primitive is not an artifact of one checkpoint,
while avoiding tiny models that cannot solve the full-cache task.

Do next:

1. Inventory locally available 3B--8B instruct models.
2. Pick one model with a standard Hugging Face cache API and likely
   long-context retrieval ability.
3. Run a cache round-trip smoke before any full experiment.
4. Run one MQ-NIAH-4Q locked smoke only if full-cache score is nonzero.

Preferred model order:

1. Qwen2.5-3B-Instruct or another Qwen2.5/Qwen3 instruct variant, if
   available locally. This minimizes tokenizer/cache integration risk
   while testing scale transfer.
2. Llama-3.1/3.2-8B-Instruct, if local weights exist and the cache path
   works under the existing runner.
3. Mistral-7B-Instruct, if available locally. This is useful because it
   changes model family, but it may introduce tokenizer/prompt-format
   differences.

Minimal model-transfer run:

- Task: MQ-NIAH-4Q only.
- Budget: `B_base=16384`.
- Restore budgets: `K={48,96,128}`.
- Conditions: `A`, `B`, matched no-repair, Random-K, Oldest-K, RepairKV,
  Gold-K.
- Samples: `n=12` for promotion to appendix; `n=24` only if it is a
  main-text candidate.
- Promotion gate: full-cache score `A>=0.90`, matched below full by
  `>=0.20`, RepairKV gain `>=0.15`, and Random/Oldest within `0.05` of
  matched.

Do not run more tiny-model transfer until the model can solve the
full-cache task. The cached Qwen2.5-0.5B-Instruct smoke failed this
gate and should remain absent from the paper.

### Retention-Rule Breadth

Goal: show repair is not tied only to SnapKV's first-stage retention rule.

Do next:

1. Implement one accumulated-attention first-stage retention rule inspired by
   H2O.
2. Unit-test deterministic budget accounting, sink/recency preservation,
   stable tie-breaking, and evicted-buffer compatibility.
3. Run a small calibration smoke over base budgets before any locked
   result.
4. Lock one budget and compare matched no-repair, Random-K, Oldest-K,
   RepairKV, and Gold-K.

Preferred policy order:

1. H2O-inspired accumulated attention heavy hitters. Highest value because
   H2O is canonical and content-aware but still uses past-turn evidence,
   making it a clean contrast with future-turn repair.
2. Sink-plus-recent retention inspired by StreamingLLM. Already implemented;
   current smoke is weak, so do not
   spend more unless a budget recalibration suggests a larger gap.
3. PyramidKV/AdaKV-like allocation. Defer until layer-varying retention
   can be represented without breaking the global-token repair protocol.
4. QUEST query-aware paging. Keep as related work or novelty
   boundary, not as a direct baseline, because it uses query-time
   information during active attention rather than post-compression
   idle repair.

Minimal H2O run:

- Task: MQ-NIAH-4Q first; add MQ-NIAH-6Q only after the 4Q smoke passes.
- Base-budget calibration: try three budgets around the SnapKV operating
  point and keep the one with matched below full by `0.20--0.60`.
- Restore budgets: `K={48,96,128}`.
- Conditions: `A`, `B`, matched no-repair, Random-K, Oldest-K, RepairKV,
  Gold-K.
- Samples: `n=12` for appendix; `n=24` only if it can become a compact
  main robustness panel.

### Repair-Selector Algorithms

Refresh-buffered shows algorithmic headroom: full-budget reselection from
active plus buffered rows can saturate the task. The next algorithmic
question is whether a cheap incremental repair selector can close more
of that gap than the current top-K query scorer.

Candidate selectors:

1. Coverage-aware top-K: penalize near-duplicate selected tokens or span
   neighborhoods so restored rows cover more answer groups.
2. Per-layer or per-head quota top-K: avoid one layer/head dominating
   the restored set.
3. MMR-style diversity repair: combine query score with dissimilarity to
   already selected positions.

Smoke first on CPU/toy selectors, then run only the best candidate on
MQ-NIAH-4Q at `B=16384`, `K={24,48,96}`, `n=12`.
Promotion gate: beat RepairKV by at least `0.05` at a mid restore budget
without reducing high-K saturation or making the method too complex for
the workshop paper.
