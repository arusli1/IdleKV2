# Preliminary IdleKV Snapshot

Purpose:
- Preserve the first finished reduced IdleKV scouting suite in git so later
  planning does not depend on untracked `results/` state.

Run:
- config: `configs/preliminary_idlekv.yaml`
- scope:
  - models: `llama8b`, `qwen7b`
  - seed: `42`
  - benchmark: `RULER 4K`
  - ratio: `r=0.7`
  - budgets: `0`, `100`, `1000 ms`
  - phases: `1`, `1+2`

Headline results:
- `llama8b`
  - all 6 conditions were at `100%`
  - interpretation: this slice is ceiling for Llama and is not useful for
    tuning the method
- `qwen7b`
  - `0ms`: `0.86`
  - `100ms`: `0.90`
  - `1000ms phase=1`: `0.90`
  - `1000ms phase=1+2`: `0.667`
  - interpretation:
    - short idle (`100ms`) helps on the informative model
    - Phase 1 is carrying the signal
    - the heavier `1+2` path may be unstable or harmful on Qwen at `1000ms`

Decision read:
- keep `qwen7b` as the main tuning model for the next small scout
- keep `100ms` as the key short-idle operating point
- do not assume Phase 2 belongs in the final large run without further evidence
- treat `llama8b` on `RULER 4K` as a ceiling slice, not the main discriminative one
