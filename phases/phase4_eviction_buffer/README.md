# Phase 4: CPU Eviction Buffer

This folder contains the Phase 4 setup for IdleKV's CPU eviction buffer and
profiling harness.

The purpose of this phase is to turn Phase 3's evicted-token artifacts into a
buffer that can:

- store evicted token KV pairs on CPU pinned memory,
- score buffered candidates against recent active-cache context,
- move selected winners back to GPU,
- and profile the repair pipeline before any Phase 6 claims are made.

## Status

This phase is scaffolded only.

- The instructions/spec live in `instructions.md`.
- The buffer and profiling modules live under `src/buffer/`.
- No profiling run has been executed yet.
- No Phase 4 results have been written yet.

## What Is Here

- `instructions.md`
  - cleaned Phase 4 implementation spec
- `src/buffer/eviction_buffer.py`
  - `BufferEntry`, `EvictionBuffer`, and `extract_recent_q_vecs`
- `src/buffer/profiling.py`
  - profiling helpers for transfer, scoring, attention overhead, and end-to-end repair
- `src/buffer/feasibility.py`
  - feasibility-frontier computation from profiling outputs
- `scripts/run_phase4_profiling.py`
  - CLI scaffold for the log-based profiling path

## Design Notes

- Phase 4 reuses `PositionTrackedCache` from Phase 2 and `EvictionResult` from
  Phase 3 rather than introducing a second cache representation.
- The synthetic profiling defaults match the local Phase 2 environment in this
  repo: `28` layers, `4` KV heads, `head_dim = 128`.
- The research-plan text still contains an older `32`-layer / `32`-KV-head
  assumption. The code scaffold here follows the repo's actual local geometry.
- The log-ingestion helpers recurse through nested Phase 3 eviction-log trees,
  so they can point at the benchmark log root or one task-specific leaf
  directory.

## Intended Outputs

When runs begin, generated profiling artifacts should be written under:

- `results/phase4_profiling/transfer_latency.json`
- `results/phase4_profiling/scoring_latency.json`
- `results/phase4_profiling/attention_overhead.json`
- `results/phase4_profiling/end_to_end_repair.json`
- `results/phase4_profiling/feasibility_frontier.json`
- `results/phase4_profiling/selection_quality.json`
