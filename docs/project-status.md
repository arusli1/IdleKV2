# Project Status

IdleKV is now paper-first. Early phases established the KV-cache access,
eviction, and offloaded-store machinery; the current work centers on
matched-budget repair evidence, paper figures, and robustness experiments for
dynamic between-turn cache repair.

## Current Readout

| Phase | Status | Notes |
|---|---|---|
| P0 | Done | Baseline RULER validation exists. |
| P1 | Done enough for support | Degradation harness and task generators exist; tests now collect in repo-wide pytest. |
| P2 | Done | KV round-trip, injection, transfer profiling, and heatmap generation are in place. |
| P3 | Done enough for support | Eviction policies, tests, live smoke, and pilot/full degradation runs exist. |
| P4 | Done enough for support | Host-memory eviction buffer and first profiling pass are implemented. |
| P6 | Active foundation | Matched-budget repair runner, selectors, reporting, first-stage retention hooks, and paper exporters are implemented and tested. |
| P7 | Main evidence | Locked 4Q/6Q matched-budget frontier data are integrated into the paper. |
| P8 | Appendix support | Strict active-cache streaming/spill diagnostics are appendix-only coverage evidence. |
| P9 | Main/appendix support | Operating-regime, proxy-scorer, and paper figure validation are implemented; renderer tests pass. |
| P10 | Active expansion | Specificity controls, 2Q/8Q full frontiers, and the locked five-turn relevance-shift diagnostic are integrated; selector variants remain staged. |
| P11 | Done | Accumulated-attention retention inspired by H2O on the 4Q full K-grid `n=24` and Llama-3.1-8B 4Q full K-grid `n=24` both passed the main-candidate gate. |
| P12 | Done | Sink-plus-recent policy-breadth `n=24` follow-up passed the gate and is integrated as a main figure with SnapKV and accumulated-attention rows. |
| P13 | Done for current paper | Iterative gates, live-branch audit script, and paired uncertainty checks are implemented. The locked `n=24`, `K=80` five-turn run passed the main gate and is integrated as a main diagnostic with stale-query caveats. |
| P14 | Active closure | AdaptFM/test-time-adaptation framing, controlled proxy-scorer validation, Llama portability cleanup, and real-agentic diagnostic planning are underway. |

## Paper State

- Draft: `paper/main.tex`.
- Built PDF: `paper/main.pdf`.
- Figure renderer: `paper/scripts/render_paper_figures.py`.
- Paper rules and terminology: `paper_guide.md`.
- Active closure status:
  `phases/phase14_critical_flaw_closure/phase14_status.md`.
- Historical Phase 10 queue: `phases/phase10_expansion/run_state.md`.
- High-signal experiment map: `phases/phase10_expansion/phase10_high_signal_map.md`.
- Main-candidate robustness plan: `phases/phase11_main_robustness/phase11_plan.md`.
- Policy-breadth plan: `phases/phase12_policy_breadth/phase12_plan.md`.
- Iterative closure framework: `phases/phase13_iteration_framework/phase13_plan.md`.
- Critical-flaw closure plan:
  `phases/phase14_critical_flaw_closure/phase14_plan.md`.
- Exact prior-policy audit:
  `phases/phase13_iteration_framework/exact_policy_audit.md`.

The main paper currently uses a compact one-column matched-budget raw-score
frontier for 2Q/4Q/6Q/8Q, a specificity-control figure, a locked five-turn
relevance-shift figure, and a first-stage policy-breadth figure covering
SnapKV, accumulated-attention, and sink-plus-recent rows. The 2Q and 8Q curves
both come from full K-grid runs, not endpoint-only breadth evidence. The
appendix contains graph-first robustness views for query-count breadth, the
operating-regime heatmap, selection diagnostics, partition endpoints, scorer
latency, and the plotted Llama portability check that is summarized in main
prose. The Qwen2.5-3B
same-family transfer result remains fallback appendix evidence rather than
the preferred paper-facing transfer check.

Phase 11 found two positive robustness checks. The accumulated-attention
retention K-grid inspired by H2O and the Llama-3.1-8B full K-grid both pass the
numerical promotion gate and are now reflected in main prose plus appendix
frontiers. The default placement remains appendix unless a compact robustness
figure can be added without overclaiming broad retention-rule or model-family
generalization.

Phase 12 tested a second non-SnapKV policy because it earned the space. The
sink-plus-recent `n=24` K-grid at `B_base=16384` passed the policy-curve gate:
best gain `0.431` at `K=128`, three eligible adjacent positive points,
controls near matched no-repair, and the Gold-K reference covering IdleKV.
The main figure now shows SnapKV, accumulated-attention, and sink-plus-recent
policy rows as a robustness check rather than a broad canonical reproduction
claim.

Phase 13 makes the remaining work explicit rather than ad hoc. Each live idea
must state a reviewer question, diagnose failures, pass unit tests, pass a
minimal smoke, and then pass a written promotion gate before it can enter the
main paper. The locked multi-turn run passed: non-initial IdleKV gain is
`0.542` with paired interval `[0.458,0.620]`, and current-query-only repair
beats stale-query-only repair by `0.307` with paired interval `[0.240,0.370]`.

Phase 14 reframes the paper as test-time adaptation of active KV state and
keeps the remaining closure work promotion-gated. The canonical live-status
file is `phases/phase14_critical_flaw_closure/phase14_status.md`. The active
GPU run is the controlled proxy-scorer frontier; the real-agentic diagnostic is
currently a design artifact, not a completed experiment.

## Validation

From the repo root:

```bash
.venv/bin/python -m pytest -q
```

Most recent recorded repo-wide CPU result:
`241 passed, 16 warnings, 304 subtests passed` on 2026-05-03. Re-run before a
release snapshot because active Phase 14 edits may change tests.

After any paper or figure edit:

```bash
.venv/bin/python paper/scripts/render_paper_figures.py
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Canonical Saved Summaries

- `saved_results/phase0/baseline_ruler.json`
- `saved_results/phase1/vt4hop_permute_aggregate_summary.json`
- `saved_results/phase1/mq_niah_4q_smoke_summary.json`
- `saved_results/phase2/phase2_summary.json`
- `saved_results/phase3/phase3_suite_summary.json`
- `saved_results/phase3/phase3_pilot10_summary.json`
- `saved_results/phase3/phase3_full100_summary.json`
- `saved_results/phase4/feasibility_frontier.json`
- `saved_results/phase4/transfer_latency.json`
- `saved_results/phase4/scoring_latency.json`
- `saved_results/phase4/attention_overhead.json`
- `saved_results/phase4/end_to_end_repair.json`
- `saved_results/phase4/selection_quality.json`
- `saved_results/phase4/run_metadata.json`

## Notes

- The tracked `saved_results/` directory is the lightweight memory layer for
  early phases. Later phase result trees are mostly ignored and regenerated or
  exported into paper CSVs/figures as needed.
- `models/` is local-only and ignored by git.
- Keep generated LaTeX byproducts out of the repo root. The ICML build
  byproducts under `paper/` are ignored and can be regenerated.
