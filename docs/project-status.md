# Project Status

This repo is currently being kept to a smoke-run standard for phases before
Phase 4. That means the goal so far is to prove the path works, preserve the
important outputs, and keep the repo understandable before the rigorous
experiment pass.

## Current Readout

| Phase | Status | Notes |
|---|---|---|
| P0 | Done | Baseline RULER run exists and acceptance checks pass. |
| P1 | Done enough for smoke | The harness and key summaries exist, but the phase note is still partly a scaffold rather than a final paper table. |
| P2 | Done | KV round-trip, injection, transfer profiling, and heatmap generation are in place. |
| P3 | Done enough for smoke | Eviction policies, tests, live smoke, and pilot/full degradation runs exist. |
| P4 | Next | Build CPU eviction buffer and profile feasible repair budget. |

## Canonical Saved Summaries

- `saved_results/phase0/baseline_ruler.json`
- `saved_results/phase1/vt4hop_permute_aggregate_summary.json`
- `saved_results/phase1/mq_niah_4q_smoke_summary.json`
- `saved_results/phase2/phase2_summary.json`
- `saved_results/phase3/phase3_suite_summary.json`
- `saved_results/phase3/phase3_pilot10_summary.json`
- `saved_results/phase3/phase3_full100_summary.json`

## Notes

- The tracked `saved_results/` directory is the lightweight memory layer for
  the project.
- The original generated outputs are still kept in the local phase `results/`
  directories, but those directories are ignored by git.
- `saved_results/phase3/` also keeps the small Phase 3 launcher log/status pair
  so there is no separate top-level `run_logs/` directory.
- `docs/phases/` contains the human-readable phase notes in order from P0 to
  P3.
