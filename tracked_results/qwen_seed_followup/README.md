# Qwen Seed Follow-up Snapshot

Purpose:
- Preserve the finished Qwen-only follow-up scout in git so the current method
  read does not depend on untracked `results/` state.

Run:
- config: `configs/qwen_seed_followup.yaml`
- scope:
  - model: `qwen7b`
  - seeds: `123`, `456`
  - benchmark: `RULER 4K`
  - ratio: `r=0.7`
  - conditions:
    - `0ms, phase=1`
    - `100ms, phase=1`
    - `1000ms, phase=1`
    - `1000ms, phase=1+2`

Headline results:
- mean across seeds:
  - `0ms, phase=1`: `0.877`
  - `100ms, phase=1`: `0.903`
  - `1000ms, phase=1`: `0.903`
  - `1000ms, phase=1+2`: `0.653`
- hardest subtask mean across seeds:
  - `0ms, phase=1`: `0.63`
  - `100ms, phase=1`: `0.71`
  - `1000ms, phase=1`: `0.71`
  - `1000ms, phase=1+2`: `0.01`

Interpretation:
- the short-idle `0ms -> 100ms` Phase 1 gain is robust across both seeds
- giving Phase 1 more than `100ms` does not help on this slice
- long-budget `1+2` is not a mild regression; it collapses on the informative
  subtask in both seeds

Decision read:
- `Phase 1 @ 100ms` is the current workshop-core operating point
- `Phase 2` should not be part of the `SCALE 2026` core claim unless a later
  debugging pass rescues it
- the next critical small runs should be:
  - a throughput / wall-clock spot-check for the shortlisted Phase 1 setting
  - one harder Llama probe to decide whether Llama belongs in the workshop
    matrix or should be deferred to the larger-GPU expansion
