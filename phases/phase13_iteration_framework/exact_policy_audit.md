# Exact Prior-Policy Audit

Last updated: 2026-05-03.

This note separates faithful prior-method reproductions from
protocol-matched retention probes. It is operational context for future runs,
not paper-facing prose.

Primary sources: H2O (NeurIPS 2023),
https://papers.nips.cc/paper_files/paper/2023/hash/6ceefa7b15572587b78ecfcebb2827f8-Abstract-Conference.html;
StreamingLLM (ICLR 2024),
https://proceedings.iclr.cc/paper_files/paper/2024/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html;
Scissorhands (NeurIPS 2023),
https://papers.nips.cc/paper_files/paper/2023/hash/a452a7c6c463e4ae8fbdc614c6e983e6-Abstract-Conference.html;
FastGen (ICLR 2024),
https://proceedings.iclr.cc/paper_files/paper/2024/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html;
SnapKV (NeurIPS 2024),
https://proceedings.neurips.cc/paper_files/paper/2024/hash/28ab418242603e0f7323e54185d19bde-Abstract-Conference.html.

## Current Paper Rows

- `SnapKV`: primary first-stage retention rule. In the paper, define the
  implementation as a context-only SnapKV-style rule inside the matched
  two-turn protocol.
- `Accumulated-attention retention inspired by H2O`: useful robustness probe,
  not canonical H2O. The current implementation scores a frozen post-turn
  cache from recent rows rather than running H2O's decode-time heavy-hitter
  cache manager.
- `Sink-plus-recent retention inspired by StreamingLLM`: useful structural
  robustness probe, not canonical StreamingLLM. The current implementation
  keeps sink positions and a recent window under the two-turn matched-budget
  protocol rather than reproducing rolling streaming inference and position
  handling.

## Exact Candidates

1. **Scissorhands.** Best next faithful baseline if a new branch is worth
   opening. It is fixed-budget and attention-history based, so it matches the
   current global-position repair abstraction better than rolling-streaming or
   layer-varying methods.
2. **Exact H2O.** Strong canonical heavy-hitter baseline, but only exact if
   the run logs actual generation-time attention and applies the published
   recent-plus-heavy-hitter update rule under audited budget accounting.
3. **FastGen.** Important adjacent baseline, but a faithful run needs
   attention-head profiling and head-specific retention patterns. It does not
   fit the current single retained-position-set repair path without new
   machinery.
4. **PyramidKV/Ada-KV-like methods.** Defer until layer-wise retained-position
   sets and layer-wise matched-budget accounting are implemented.
5. **QUEST-like paging.** Defer to future page-level repair; it is query-time
   page loading rather than a clean first-stage global-token eviction policy.

## Required Before Claiming Exact

- Log the runtime signal the original algorithm uses.
- Implement the original update timing and budget accounting, not just a
  top-K approximation with a similar intuition.
- Unit-test toy traces for budget invariants, deterministic tie handling,
  mandatory sink/recency rows when applicable, and evicted-store compatibility.
- Run a CPU selector smoke before a GPU smoke.
- Run a minimal MQ-NIAH-4Q smoke before any full grid.

## Current Decision

Do not start an exact prior-policy full run while the locked multi-turn branch
is unresolved. If GPU time opens and the paper needs a new named-algorithm
baseline, open Scissorhands first; otherwise keep the current policy-breadth
figure as mechanism-level robustness evidence.
