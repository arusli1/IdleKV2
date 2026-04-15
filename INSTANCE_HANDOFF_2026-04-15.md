# Instance Handoff — 2026-04-15

This file captures the concrete state from the last Codex instance so a fresh
instance can continue immediately after `git pull`.

## What Changed In This Session

Two important commits are already on `main`:

- `c055571` — `Implement stochastic anytime cache repair`
- `935daaf` — `Save stochastic proposal and paper updates`

Those commits added:

- policy-driven stochastic repair in the runtime
- retained-position tracking and cold-span sampling
- the shared repair module: `idlekv/core/anytime_repair.py`
- the stochastic experiment config: `configs/stochastic_core.yaml`
- the proposal / paper rewrite note:
  `STOCHASTIC_ANYTIME_REPAIR_REFINED.md`

## Main Method Direction

Treat **stochastic anytime repair** as the main method.

Operationally:

- `sampled_spans` is the main method
- `shadow_only` is the fast local baseline
- `full_refresh` is the explicit long-budget control

Do **not** center the project on the old “Phase 1 plus optional Phase 2”
framing anymore. The code still supports the legacy path, but the intended
paper/runtime direction is:

1. compress on the latency-critical path
2. keep a small shadow pool on GPU
3. keep a cold backing store off-path
4. spend idle time sampling cold spans, rescoring, and committing valid
   partial repairs until interrupted

## Validation State

Completed in this session:

- `/home/ubuntu/IdleKV/.venv/bin/pytest -q`
  - result: `47 passed`
- official runner dry-run for the stochastic config succeeded:
  - `configs/stochastic_core.yaml`

Not completed in this session:

- no fresh measured 7B stochastic result
- a real `qwen7b` stochastic smoke run was started and intentionally stopped
  when the user requested that no long-running jobs continue

Interpretation:

- the codebase is structurally ready for the stochastic experiment
- the next instance still needs to run the first real measured stochastic-core
  experiment

## Recommended First Steps On A Fresh Instance

1. Read:
   - `STATUS.md`
   - `STOCHASTIC_ANYTIME_REPAIR_REFINED.md`
   - `configs/stochastic_core.yaml`
2. Reconfirm:
   - `pytest -q`
3. Preview the main run:
   - `python scripts/run_experiments.py --config configs/stochastic_core.yaml --dry-run --model qwen7b`
4. If the environment is ready, run the first real stochastic-core experiment
   in `tmux`

## Experiment To Run Next

Main next experiment:

- config: `configs/stochastic_core.yaml`
- model: `qwen7b`
- benchmark: `RULER 4K`
- policy: `sampled_spans`
- budgets: `0, 50, 100, 200, 500, 1000 ms`

Purpose:

- produce the first end-to-end stochastic anytime curve on the informative
  Qwen slice
- compare against compressed baselines on the current A10G machine

Secondary next run:

- `scripts/throughput_spotcheck.py`

## Paper Direction

The paper should be re-centered around stochastic anytime repair as the main
contribution.

The important rewrite is:

- old framing:
  - short-idle `Phase 1`
  - exploratory `Phase 2`
- new framing:
  - stochastic anytime repair from a cold store during tool-call idle time

Use `STOCHASTIC_ANYTIME_REPAIR_REFINED.md` as the main source of truth for:

- method framing
- memory/algorithm/sampling variants
- stress tests
- paper restructure

## Hardware / Environment Facts

- machine target: single `NVIDIA A10G`
- VRAM: `23028 MiB`
- current method/config choices were made to stay conservative on this card

## Worktree Notes

At the end of this session, two local LaTeX style files existed but were not
committed:

- `paper/algorithm.sty`
- `paper/algorithmic.sty`

If a fresh instance needs to compile the paper and those files are absent after
`git pull`, either install the corresponding LaTeX packages in the environment
or restore local copies before treating it as a repo bug.

## Summary

The repo now has the stochastic runtime path, the stochastic experiment config,
the proposal note, passing tests, and a clear next experiment.

The missing piece is not implementation anymore; it is the first real measured
stochastic-core run and then updating the paper/results around that run.
