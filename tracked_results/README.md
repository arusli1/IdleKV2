# Tracked Result Snapshots

Purpose:
- Preserve the small number of results that have already materially shaped the
  project so a future operator does not need access to the full untracked
  `results/` tree to understand the current state.

Snapshots currently tracked:

- `preliminary_baselines/`
  - four preserved baseline files from the aborted long A10G queue
  - useful as an early baseline read, not as the final workshop baseline matrix

- `preliminary_idlekv/`
  - first reduced two-model scout
  - established that `llama8b` on `RULER 4K` is ceiling
  - established that `qwen7b` at `r=0.7` is informative

- `qwen_seed_followup/`
  - multi-seed confirmation on the informative Qwen slice
  - established that `Phase 1 @ 100ms` is robust
  - established that long-budget `P1+P2` currently fails badly enough that it
    should be removed from the workshop-core story

How to use this folder:
- read the README in each snapshot subdirectory first
- treat these files as preserved milestones, not the complete result store
- use `STATUS.md` for the live next-step plan and active submission scope
