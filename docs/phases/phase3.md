# Phase 3 Smoke Note

This note captures the current Phase 3 state at the smoke-run standard being
used before Phase 4.

## What Exists

- Eviction policy code under `phases/phase3_eviction/src/`
- Synthetic tests plus a live Qwen smoke test
- A top-level suite summary
- A benchmark pilot (`pilot10`)
- A full smoke benchmark (`full100`)

## Smoke Outcome

- Test suite summary: `9` tests run, `0` failures, `0` errors
- Live smoke artifact written successfully
- Pilot benchmark completed
- Full benchmark completed with `rc=0`

## Key Artifacts

- `saved_results/phase3/phase3_suite_summary.json`
- `saved_results/phase3/phase3_live_smoke.json`
- `saved_results/phase3/phase3_pilot10_summary.json`
- `saved_results/phase3/phase3_full100_summary.json`
- `saved_results/phase3/phase3_pilot_then_full.log`
- `saved_results/phase3/phase3_pilot_then_full.status`

## High-Level Readout

- `MQ-NIAH-4q`: strong degradation under compression relative to full KV
- `S-NIAH`: full KV solves the task, compressed runs collapse in the current
  smoke setup
- `VT-4hop`: remains too easy in the current Phase 3 smoke benchmark

## Decision

Under the current smoke-only standard, Phase 3 is complete enough to start
Phase 4.
