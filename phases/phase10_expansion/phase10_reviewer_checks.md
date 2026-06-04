# Phase 10 Reviewer Checks

Last updated: 2026-05-03 01:50:00 UTC.

This file condenses the senior-ML-systems and AdaptFM critique pass. It is
operational guidance for Phase 10 decisions, not paper-facing prose.

## Claim Boundaries

- Use **matched active KV budget**, not "matched memory," unless CPU buffer
  storage is explicitly counted.
- Use **Gold-K reference** or **hindsight span reference**, not undefined
  "oracle" language.
- Treat agent workflows as motivation. The evidence should be phrased as
  controlled topic-shift or relevance-shift diagnostics, not agent benchmark
  performance.
- The novelty is not query-aware KV selection in general. The narrower claim
  is post-compression, post-query, pre-resume repair from a buffered evicted
  cache without full-prefix recomputation.
- If Refresh-buffered reaches the ceiling, frame RepairKV as an incremental
  low-recompute repair primitive, not the strongest possible Q2-time
  reselection policy.

## Main-Figure Priority

1. Method schematic showing compression, evicted CPU buffer, idle window,
   query-time scoring, restore, and resumed generation.
2. 4Q/6Q matched-active-budget frontier with matched no-repair, RepairKV,
   Random-K, Oldest-K, and Gold-K.
3. Locked specificity contrast showing stale-query and donor-query controls.
4. Multi-turn relevance-shift trajectory only if the smoke and locked run pass.

Operating-regime heatmap, query-count breadth, accumulated-attention retention
breadth, selector variants, model transfer, proxy/runtime, and quantization
belong in the appendix unless one is unusually clean and main-text space
remains.

## Query-Count Breadth Gate

Endpoint `K={48,96}`, exact-Q, Gold-K, Random-K, Oldest-K, `n=24` is enough for
appendix breadth. It is main-text material only if 8Q is very clean.

Promotion checks:

- Full-cache score `A >= 0.90`.
- Full-cache versus matched no-repair gap `A - B_match >= 0.20`.
- RepairKV gain over matched no-repair `>= 0.15`.
- RepairKV beats `max(Random-K, Oldest-K)` by a meaningful margin.
- Gold-K is at least as high as RepairKV.
- No severe split heterogeneity or negative split-level lift.

2Q is a sanity endpoint even when positive because it saturates easily. 8Q is
the only query-count endpoint with realistic main-text upside.

## GPU Priority

1. Finish active 2Q/8Q endpoint run.
2. Run multi-turn relevance-shift smoke. If positive, scale to locked `n=12`.
3. Run the accumulated-attention retention smoke as the most useful
   non-SnapKV portability branch.
4. Run selector-variant smoke only if the GPU queue opens with a short gap or
   after the higher-priority branches.
5. Do model transfer only after selecting a model that passes a full-cache
   ability smoke.

## Failure Handling

- Do not salvage a run into the main paper if Random-K/Oldest-K catch RepairKV.
- Do not promote a task where full-cache accuracy is weak.
- Do not expand query-count into a full sweep unless 8Q endpoint evidence is
  clean and visually useful.
- Keep the row-store quantization sweep as negative evidence/future work; it
  does not support the main claim.
