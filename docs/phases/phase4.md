# Phase 4 Note

Phase 4 has now been implemented and run once.

## Goal

Build the CPU eviction buffer and the profiling harness that determines the
feasible repair budget `K` for each tool-call duration.

## Current State

- `phases/phase4_eviction_buffer/src/buffer/` now contains:
  - `eviction_buffer.py`
  - `profiling.py`
  - `runtime.py`
  - `quality.py`
  - `feasibility.py`
- the CLI runner in `scripts/run_phase4_profiling.py` writes:
  - `transfer_latency.json`
  - `scoring_latency.json`
  - `attention_overhead.json`
  - `end_to_end_repair.json`
  - `feasibility_frontier.json`
  - `selection_quality.json`
  - `run_metadata.json`
- a focused `unittest` suite exists under `tests/`

## First Run

The first full run used:

- a log-backed profiling buffer built from Phase 3 SnapKV artifacts
- a live `32K` SnapKV fixture at `k_budget=4096`
- a `20` token resumed query for attention-overhead measurement

Key results from `phases/phase4_eviction_buffer/results/phase4_profiling/`:

- CPU→GPU transfer stayed monotone and fast:
  - `K=1000` transfer `p50 ≈ 25.9 ms`
  - `K=2000` transfer `p50 ≈ 55.2 ms`, `p90 ≈ 59.0 ms`
- CPU scoring stayed small at the key scale:
  - `l2_norm`, `N=1000`: `p50 ≈ 2.77 ms`
  - `dot_product`, `N=1000`: `p50 ≈ 4.00 ms`
- end-to-end repair stayed far below the `2s` tool-call budget:
  - `K=100`: `p90 ≈ 35.8 ms`
  - `K=500`: `p90 ≈ 49.6 ms`
- post-injection attention overhead was modest through `K=1000`:
  - `K=1000`: `≈ 1.81 ms` p50 overhead on the next resumed forward pass

## Caveats

- The current feasibility frontier saturates at `K=2000` because the transfer
  sweep only went up to `2000` tokens and all measured tool-call budgets still
  fit that point.
- The current VT-4hop smoke slice has `0/50` incorrect examples at
  `snapkv / k512`, so `selection_quality.json` is a diagnostic artifact, not a
  useful selector-quality comparison yet.

## Next Step

If Phase 4 needs one more refinement before Phase 5, extend the transfer sweep
above `K=2000` so the frontier stops saturating at the current measurement
ceiling. Otherwise the MVP Phase 4 gate is satisfied and Phase 5 can start.
