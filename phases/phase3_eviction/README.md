# Phase 3: Eviction Algorithm Validation

This folder contains the Phase 3 implementation for the IdleKV plan: a shared
eviction interface, SnapKV and query-aware SnapKV, a StreamingLLM baseline,
logging utilities, and a test harness that stress-tests the selection logic
without tying anything to the Phase 1 task variants.

## What Is Here

- `instructions.md`
  The phase-specific spec that this implementation follows.
- `src/eviction/`
  The eviction contract, policy implementations, and log writer.
- `src/runtime.py`
  Shared helpers for the live smoke test and the suite runner.
- `tests/`
  Synthetic stress coverage plus a live-Qwen smoke test.
- `scripts/run_phase3_suite.py`
  End-to-end runner for the Phase 3 test suite and smoke artifact generation.
- `results/`
  Generated smoke summaries and eviction logs.

## Design Notes

- Phase 3 reuses the Phase 2 `PositionTrackedCache` and KV helpers instead of
  copying another cache representation.
- The query-aware policy consumes actual query token ids and runs them against
  the full cache before scoring kept slots.
- The synthetic tests target budget handling, ordering, CPU offload, and
  deterministic query-routing behavior. They do not compare against Phase 1.

## How To Run

From the repo root:

```bash
source .venv/bin/activate
python phases/phase3_eviction/scripts/run_phase3_suite.py
```

The runner executes:

1. Synthetic stress tests for SnapKV, query-aware SnapKV, and StreamingLLM
2. A live Qwen smoke test when CUDA and the local model are available
3. Result writing under `phases/phase3_eviction/results/`
