# Phase 0 Baseline Note

This note captures the current Phase 0 status for the baseline RULER run in
`phases/phase0_baseline`.

## Purpose

Phase 0 is the control layer for the rest of the project. It establishes that
the local Qwen2.5 setup behaves sanely on the long-context baseline before any
eviction or repair logic is introduced.

## Saved Summary

- `saved_results/phase0/baseline_ruler.json`
- `saved_results/phase0/degradation_curve.csv`

## Current Readout

- S-NIAH passes the Phase 0 minimum at 4K.
- S-NIAH passes the Phase 0 minimum at 32K.
- VT-2hop and FWE baseline outputs are present in the saved summary artifact.

## Outcome

Under the current smoke-first workflow, Phase 0 is complete and provides the
baseline reference point for all later phases.
