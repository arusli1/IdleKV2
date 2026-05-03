# Runtime Evidence Plan

Last updated: 2026-05-03.

## Claim

Idle-window repair should be presented as a systems-capacity claim:

> A plausible IdleKV-style repair primitive can score an offloaded evicted-KV
> store and restore selected rows within sub-second to few-second idle windows
> for the measured candidate-store sizes.

This does not claim end-to-end agent speedup, production readiness, or that the
current exact answer scorer is optimized.

## Reviewer Critique Of The Old Table

- It mixed exact answer-scoring probes, proxy answer-scoring probes, pure
  move/inject probes, and chunked offloaded-store scans in one table.
- The rows did not share the same denominator: some rows were task-specific
  quality runs, while others were synthetic Qwen-shaped systems probes.
- The earlier 1M candidate scan used a one-chunk source pool. That measured the
  arithmetic and transfer loop shape, but it was too easy to read as a full
  offloaded-store scan. The rigorous run must report source-pool coverage and
  use coverage 1.0 for headline values.
- Exact scorer latency is still useful, but only as a diagnostic upper bound
  for the current research implementation. It should not carry the feasibility
  claim.

## Main Experiment

Use one consistent runtime envelope:

- Hardware: same evaluation GPU.
- Model-shaped KV: Qwen-shaped BF16 KV by default.
- Active cache: 32K rows for the main repair-envelope figure.
- Candidate store: 32K, 64K, 128K, 256K, 512K, 1M, 2M, and 4M offloaded
  candidate rows. These powers-of-two checkpoints align with context-length
  reporting conventions and test beyond the common million-token headline
  scale without changing the runtime protocol.
  The main paper should stop at 4M: it reaches the representative 5s
  checkpoint while keeping the one-column heatmap readable. An 8M row is
  appendix-only stress evidence unless it answers a new reviewer-facing
  question.
- Query length: 64 rows for the main run. Query-length sensitivity is useful
  appendix material only after the main envelope is stable.
- Restore budgets: `K in {96, 512, 1024, 5000}`. The main figure shows the
  full grid; the middle budgets check monotonicity and show that restored-row
  budget is not the dominant latency axis.
- Measurement: p50/p95/p99 over robust trials after warmup.
- Source pool: enough distinct pinned host chunks to cover every measured
  candidate row for one KV layer (`host_pool_coverage = 1.0`).
- Runtime reported for the main figure: a conservative component-summed p95
  from separate measured stages: chunked candidate scoring, top-K selection,
  selected-KV host-to-GPU movement, and reinsertion. Do not describe this as a
  joined end-to-end latency distribution unless a joined run produced the row.

The full 1M offloaded KV state is about 57 GB for this shape; 4M rows are about
240 GB. The scan
materializes a full one-layer key pool and streams it once per layer; it does
not materialize the full key+value store on GPU.

## Ideal Paper Artifacts

Main text:

- One runtime-capacity grid: restore budget `K` by offloaded candidate rows,
  with annotated component-summed p95 latency and representative idle-budget
  colorbar checkpoints. Add a compact component-share strip below it.
- Prefer prose anchors over a main-text table. If exact values are needed,
  move a compact p50/p95/p99 table to the appendix and do not include a
  categorical "idle" column in the main text.

Appendix:

- Query-length sensitivity for 16/64/128 query rows, ideally as an appendix
  heatmap or compact table, because the main grid fixes query length at 64.
- Move/inject-only active-cache scaling for 32K/100K/500K active rows.
- Exact/proxy scorer diagnostics with quality numbers.

## Idle-Window Distribution

If we have an empirical tool-call trace, plot a survival curve
`P(idle window >= t)` and overlay vertical measured repair latencies. If we do
not have a trace, do not claim a fraction of tool calls covered. Idle time is a
continuous budget; use checkpoint language only for readability. For example,
"this operating point fits within a 1s representative idle-budget checkpoint
with 10% slack" is acceptable, while "the method covers 1s tool calls" is not.
