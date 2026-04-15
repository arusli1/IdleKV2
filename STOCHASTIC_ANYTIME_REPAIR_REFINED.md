# Stochastic Anytime Repair: Main-Method Proposal

## Current Status

- The repo now contains an explicit policy-driven stochastic path:
  `shadow_only`, `sampled_spans`, `full_refresh`.
- The implementation tracks original prefill positions, samples cold CPU-KV
  spans per layer, merges them with retained and shadow candidates, and
  commits interruptible top-k repairs without touching the decode path.
- Quick validation status:
  - `pytest -q` passes locally (`47 passed`).
  - The official experiment runner dry-runs correctly for a stochastic config:
    `configs/stochastic_core.yaml`.
  - A real 7B smoke run was started and then intentionally stopped before
    completion to avoid a long job.

This means the codebase is structurally ready for the stochastic experiment,
but the first real measured run still needs to be executed explicitly.

## Main Claim

The main method should be:

> Stochastic anytime cache repair during tool-call idle time.

The core story is not "Phase 1 plus an optional Phase 2." The core story is:

1. compress aggressively on the latency-critical path,
2. preserve a small local shadow candidate pool on GPU,
3. maintain a colder backing store off-path,
4. use idle time to repeatedly sample stale context, rescore under updated
   per-layer queries, and commit valid partial repairs until interrupted.

The current `shadow_only` behavior is the smallest candidate-pool instance of
that general mechanism. `sampled_spans` is the main method. `full_refresh`
becomes an explicit long-budget control, not the centerpiece.

## Refined Method

### v1 system instantiation

- Hot state on GPU:
  - compressed retained prefill KV
  - pinned generated tail
  - recent-query buffer
  - small shadow buffer of near-miss evictions
- Cold state off-path:
  - full prefill KV on CPU
- Repair loop:
  - pick a layer, shallow-first
  - sample a few cold spans
  - merge `retained + shadow + sampled`
  - deduplicate by original prefill position
  - rescore under recent per-layer queries
  - keep top-k prefill tokens
  - rebuild shadow from displaced retained tokens and surviving shadow tokens
  - stop immediately if the idle budget expires

### v1 scheduler policy

- `<20 ms`: no-op
- `20-80 ms`: `shadow_only`
- `>80 ms`: one `shadow_only` pass, then repeated `sampled_spans`
- `full_refresh`: explicit long-budget comparison only

### Why this is the right first version

- Fits the current A10G memory model.
- Avoids replay/recompute ambiguity from token-only or checkpoint-only cold
  storage.
- Turns the previous brittle Phase 2 story into a clear control/baseline.
- Preserves the current short-idle success case while giving a stronger story
  for longer idle windows.

## Design Space: What To Modify Next

### Memory variants

1. Full CPU KV
   - Best first baseline.
   - Highest RAM footprint, lowest ambiguity.
   - Good for single-agent or low-concurrency runs.

2. Hidden-state checkpoints
   - Strong next memory optimization.
   - Store sparse hidden states or layer checkpoints instead of full per-layer
     KV, then reconstruct candidate KV lazily.
   - Lower RAM, higher idle-time compute.

3. Hybrid cold store
   - Keep a small exact KV tier for high-probability candidates and a larger
     checkpoint tier for the long tail.
   - Most promising scalability direction after v1.

4. Token replay only
   - Not a good mainline direction for this repo right now.
   - Too much recompute and too much implementation ambiguity.

### Algorithm variants

1. Deterministic shortlist + stochastic tail
   - Take the top-m spans by score, then sample the remaining budget
     stochastically.
   - Reduces variance while preserving exploration.

2. Drift-gated deep repair
   - Skip cold sampling when recent queries have barely changed.
   - Cheap query-drift metric: cosine distance between current and seeded query
     buffer centroids per layer.

3. Acceptance-gated commit
   - Commit only if the new candidate set improves a cheap surrogate score
     over the current retained set.
   - Useful if repeated stochastic commits become noisy at long budgets.

4. Layer schedule variants
   - shallow-first
   - uniform round-robin
   - loss-aware / drift-aware per-layer prioritization

5. Candidate reuse
   - Reuse sampled spans for a short horizon if the idle window remains open.
   - Avoid repeatedly paying PCIe costs for the same cold slices.

### Weighted-random sampling variants

The current implementation uses a simple weighted span sampler. The next
versions should compare:

1. Recency-biased sampling
   - Favor later prefill positions.
   - Cheap and often strong in tool-calling traces.

2. Prefill-importance-biased sampling
   - Favor spans with high prefill-time importance scores.
   - Best default exploit strategy.

3. Drift-aware sampling
   - Favor spans whose historical score was high but are currently absent from
     retained and shadow buffers.
   - Better match to delayed-query correction.

4. Diversity-aware sampling
   - Downweight spans near already-sampled regions.
   - Helps avoid repeatedly sampling one local cluster.

5. Mixture sampler
   - `w = alpha * importance + beta * recency + gamma * novelty`
   - Most likely mainline weighted-random formulation for the paper.

Recommendation:

- v1 paper/main experiments: importance + small recency bonus.
- strongest follow-up ablation: deterministic top spans vs random spans vs
  weighted-random spans at matched transferred bytes.

## Stress Tests That Actually Matter

### Correctness / systems

1. Short idle: `0, 20, 50, 100 ms`
2. Long idle: `200, 500, 1000 ms`
3. High compression: `r=0.7`
4. Lower compression control: `r=0.5`
5. Interruptibility: stop mid-loop and verify decode continues
6. Bytes moved: quality recovered per CPU byte transferred

### Mechanism

1. Shadow-only vs sampled-spans at matched budget
2. Token-level random vs span sampling at matched byte budget
3. Shallow-first vs uniform layer order
4. Buffer `0` vs `256`
5. Deterministic shortlist vs weighted-random sampling

### Failure modes

1. No query drift
   - Repair should help little and should not claim otherwise.
2. Harmful long-budget repair
   - Detect whether repeated stochastic commits start degrading quality.
3. Ceiling slices
   - Exclude or down-weight slices that cannot discriminate methods.
4. CPU-bandwidth saturation
   - Ensure the method still makes sense on the A10G PCIe setup.

## Paper Restructure

## New title-level framing

Stop framing the paper as:

- short-idle Phase 1, plus an exploratory Phase 2 extension

Instead frame it as:

- stochastic anytime repair from a cold store during tool-call idle time

## Suggested contribution rewrite

1. Formulate delayed-query cache repair under unknown idle duration.
2. Introduce stochastic anytime repair with interruptible span sampling and
   merged top-k commits.
3. Show that weighted cold-store sampling improves over local-only repair on
   the informative Qwen RULER-4K slice at short idle budgets.

## Method section rewrite

Replace the current Phase 1 / Phase 2 method arc with:

1. Problem Setup: delayed-query drift + idle-time opportunity
2. System Overview: hot cache, shadow buffer, cold store, query buffer
3. Anytime Repair Operator:
   - sample
   - merge
   - rescore
   - commit
   - interrupt
4. Instantiations:
   - `shadow_only` as minimal candidate-pool case
   - `sampled_spans` as main method
   - `full_refresh` as long-budget control
5. Complexity / memory analysis:
   - quality per ms
   - quality per transferred byte

## Results section rewrite

The main results should answer four questions in order:

1. Does stochastic anytime repair beat the matching compressed baseline?
2. Does sampled cold repair beat shadow-only repair at the same idle budget?
3. Do weighted span samples beat naive/random samples at the same transferred
   bytes?
4. Where does the method fail or saturate as idle budget grows?

That gives a cleaner narrative than the current Phase 1 / Phase 2 split.

## Recommended Experiments

### First run

Use [configs/stochastic_core.yaml](/home/ubuntu/IdleKV/configs/stochastic_core.yaml:1)
as the main Qwen RULER-4K matrix.

### Follow-up ablations

1. `shadow_only` vs `sampled_spans`
2. deterministic top spans vs weighted-random spans
3. buffer `0` vs `256`
4. shallow-first vs uniform layer schedule

### Keep out of the main claim for now

1. Full-refresh wins at long budgets
2. Broad LongBench claims
3. Multi-tenant serving relevance
4. Large-model extrapolation beyond the current 7--8B regime

## Bottom Line

If stochastic anytime repair is the main method, the repo and the paper should
both treat `sampled_spans` as the center of gravity.

- `shadow_only` becomes the fast local baseline.
- `full_refresh` becomes an explicit long-budget control.
- The main metrics become:
  - quality recovered vs no-refinement baseline
  - quality per millisecond
  - quality per transferred byte

That is the cleanest route to a defensible systems paper from the current code.
