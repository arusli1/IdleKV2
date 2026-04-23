# Project Status

This repo has been kept to a minimum-viable standard through Phase 4. The first
Phase 4 profiling pass has now been executed on the local GPU.

## Current Readout

| Phase | Status | Notes |
|---|---|---|
| P0 | Done | Baseline RULER run exists and acceptance checks pass. |
| P1 | Done enough for smoke | The harness and key summaries exist, but the phase note is still partly a scaffold rather than a final paper table. |
| P2 | Done | KV round-trip, injection, transfer profiling, and heatmap generation are in place. |
| P3 | Done enough for smoke | Eviction policies, tests, live smoke, and pilot/full degradation runs exist. |
| P4 | Done enough for MVP | Buffer/profiling code exists and the first live profiling pass wrote the feasibility frontier plus transfer/scoring/attention/end-to-end artifacts. |

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
  the project.
- The original generated outputs are still kept in the local phase `results/`
  directories, but those directories are ignored by git.
- `saved_results/phase3/` also keeps the small Phase 3 launcher log/status pair
  so there is no separate top-level `run_logs/` directory.
- The first P4 run used the real local geometry (`32K` live fixture,
  `k_budget=4096`) and found that the current transfer sweep saturates at the
  measured ceiling of `K=2000`, so the next P4 refinement should extend the
  transfer grid upward if the paper needs a wider frontier.
- `docs/phases/` contains the human-readable phase notes in order from P0 to
  P4.
