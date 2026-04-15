# Stochastic Anytime Repair: Refined Research Direction

## One-sentence thesis

Treat each tool-call pause as an interruptible optimization window: keep a
small hot KV cache on GPU, keep a larger cold backing store off the critical
path, and spend idle time repeatedly sampling stale context, rescoring it under
recent queries, and committing top-k cache repairs until the tool returns.

## Why this should be the main idea

The current repo already has the right ingredients:

- `idlekv/core/phase1_rescore.py`: fast local repair from a small shadow buffer
- `idlekv/core/phase2_refresh.py`: slower shallow-first refresh from full KV
- `idlekv/core/query_buffer.py`: recent per-layer hidden states used as the
  query context for repair
- `idlekv/core/scheduler.py`: interruptible idle-time execution

The sharper project idea is to frame these not as two disconnected phases, but
as points on one spectrum:

- fast repair uses a tiny candidate set already on GPU
- deeper repair uses a larger candidate set from a cold store
- both are instances of the same anytime loop:
  sample -> score -> merge -> keep top-k -> commit -> continue until interrupted

That story is cleaner, more general, and easier to defend as a research
contribution than "we have Phase 1 and also a slower optional Phase 2."

## Core intuition

KV compression makes irrevocable decisions before the final effective query is
known. Tool calls create a period where generation is paused but the next query
is becoming better-defined. The system should use that pause to revisit
previously evicted context.

The key constraint is that the pause length is unknown in advance. That means
the repair algorithm must be:

- interruptible
- valid at every partial step
- able to trade memory for compute
- able to spend 20 ms, 200 ms, or 2 s usefully

This is exactly what an anytime stochastic repair loop gives you.

## Fundamental objects

- Layer: one transformer block in the stack. Each layer has its own KV cache.
- KV head: one KV attention group inside a layer.
- Layer-local scoring: compare candidates only within the same layer, because
  layer 3 and layer 20 live in different representation spaces.
- Query set: recent per-layer hidden states projected through that layer's
  query matrix.
- Candidate pool: the current retained tokens plus the sampled evicted tokens
  being considered in one repair round.
- Cold store: evicted information kept off the hot GPU path. In the simplest
  version this is full KV on CPU.

## Refined algorithm

### Proposed main mechanism

1. Keep a bounded hot cache on GPU for normal decoding.
2. Keep evicted spans in a cold store.
3. When a tool call begins, start an idle-time repair worker.
4. Repeatedly choose a layer to work on, preferably shallow-first.
5. Sample candidate spans from the cold store for that layer.
6. Build one shared candidate pool: current retained tokens plus sampled spans.
7. Score the pool under the same recent query set.
8. Keep the merged top-k tokens for that layer and commit the update.
9. Stop immediately when the tool returns.

### Preferred choices

- Sample spans/chunks, not isolated tokens.
- Use merged top-k selection, not "candidate beats current minimum."
- Keep generated tokens pinned; only prefill context competes for repair.
- Work shallow-first by default, because early-layer changes propagate forward.
- Use small commit units so interruption always leaves a valid cache.

## Stress test

### Where the idea is strong

- Unknown idle duration: the algorithm naturally degrades to "do a little good
  work and stop."
- Delayed-query drift: rescoring under newer queries is exactly the regime where
  earlier eviction decisions can be wrong.
- Single-agent dedicated GPU: CPU-backed cold storage is viable and the repair
  work stays off the decode critical path.
- Span sampling: avoids scanning the entire evicted history every time.

### Where the idea breaks or weakens

- Very short tool calls: if the idle window is only a few milliseconds, even
  staging candidates may cost too much. The scheduler needs a minimum-work gate.
- No query drift: if the effective query has not changed, repair may do little
  beyond reshuffling.
- High concurrency: full CPU KV is reasonable for one agent, but it scales
  poorly across many simultaneous sessions.
- Large cold-store scans: if candidate retrieval is too broad, PCIe traffic
  erases the value of repair.
- Per-head exact scoring everywhere: this may be more expensive than the gain
  justifies. A coarser per-layer shortlist may be enough.

### Design responses to those failure modes

- Add a "do nothing below X ms" scheduler threshold.
- Add a drift gate based on query change magnitude before launching deep repair.
- Sample a small number of spans per round instead of loading whole layers.
- Prioritize spans by recency, prior importance, and semantic locality.
- Use a tiered policy:
  shadow buffer first, then sampled cold-store spans, then optional full-layer
  refresh only if the idle window is still open.

## Memory and compute tradeoff

There are three reasonable storage regimes:

- Token IDs only: smallest memory, largest recompute; not enough to recover
  exact KV without replaying the model.
- Hidden-state checkpoints: middle ground; cheaper than full per-layer KV, but
  you must project or partially recompute to rebuild candidates.
- Full KV in cold storage: largest memory, least recompute; best baseline for a
  first single-agent implementation.

Rough scale from `configs/models.yaml` with bf16 storage:

- Llama 3.1 8B full KV is about 128 KiB/token.
- Qwen2.5 7B full KV is about 56 KiB/token.
- One hidden-state checkpoint is about 8 KiB/token for Llama 3.1 8B.
- One hidden-state checkpoint is about 7 KiB/token for Qwen2.5 7B.

Implications:

- Full CPU KV is reasonable for one or a few agents.
- Hidden-state checkpoints become attractive when CPU RAM or concurrency is the
  real bottleneck.
- The strongest first system is still full CPU KV, because it minimizes
  implementation ambiguity and isolates the value of idle-time repair itself.

## Refined project claim

The main project should not be "full refresh from CPU is good." That is too
implementation-specific and too easy to lose on memory cost.

The main claim should be:

> Agentic tool-call pauses can be converted into an anytime cache-repair budget
> by sampling stale context from a cold store, rescoring it under updated
> per-layer queries, and committing interruptible top-k cache improvements
> without adding work to the decode critical path.

That claim is broader, cleaner, and still compatible with the existing repo.

## Most defensible system instantiation

For the actual project, the most defensible first system is:

- hot compressed KV on GPU
- recent-query buffer on GPU
- small on-GPU shadow buffer for near-miss evictions
- full evicted KV on CPU as the initial cold store
- shallow-first stochastic span repair during tool idle
- optional escalation to larger refresh only if the idle window persists

This preserves the current Phase 1 strength while giving a better long-window
story than a monolithic Phase 2 sweep.

## Concrete research hypotheses

- H1: Under delayed-query drift, sampled cold-store repair beats shadow-buffer
  only repair at the same idle budget.
- H2: Span sampling beats token-level random sampling at the same transferred
  bytes.
- H3: Shallow-first repair produces better quality per millisecond than uniform
  layer scheduling.
- H4: Hidden-state checkpoints recover most of the benefit of full CPU KV when
  idle windows are long enough to tolerate extra recompute.
- H5: Below a short idle threshold, the best policy is to skip deep repair and
  run only local shadow-buffer repair or nothing.

## Minimal implementation delta from the current repo

The current codebase is already close. The smallest meaningful extension is:

1. Add a cold-store sampler that returns span candidates per layer.
2. Generalize `phase1_rescore` from shadow-buffer-only candidates to arbitrary
   candidate pools.
3. Replace the hard Phase 1 / Phase 2 boundary in `scheduler.py` with a tiered
   anytime loop.
4. Add scheduling policies:
   `shadow_only`, `sampled_spans`, `full_refresh`.
5. Evaluate quality recovered per millisecond and per transferred byte.

## What to avoid claiming

- Do not claim this is a win for multi-tenant serving engines.
- Do not claim full CPU KV is universally cheap; it is a single-agent-friendly
  baseline, not the final scalability answer.
- Do not claim long idle windows are required; the point is graceful use of
  whatever idle budget exists.
- Do not oversell exact per-head repair as necessary until ablations prove it.

## Recommendation

Make the main research direction:

**Stochastic anytime cache repair from a cold store during tool-call idle
time**, with the current shadow-buffer Phase 1 reinterpreted as the smallest
candidate-pool instance of that general mechanism.

That gives the project a cleaner theory, a stronger systems story, and a more
natural path from the current implementation to a publishable next version.
