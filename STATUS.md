# IdleKV Status

Read this first if you are taking over the repo on a fresh SSH/Codex session.

## Current State

- Core runtime is implemented and the current test suite passes.
- The delayed-query pilot is validated.
- The experiment runner is real and resumable; it is no longer a mock scaffold.
- The repo is intentionally scoped to a stable single-A10G workshop-scale matrix.

Current validated test status:
- `41 passed`

## What Was Validated

Mechanism-level pilot:
- `scripts/go_no_go.py`
- delayed-query stress gate at `r=0.7`
- verified result on the A10G path:
  - full cache: `100.0%`
  - compressed baseline: `91.7%`
  - IdleKV + Phase 1: `100.0%`
  - delta: `+8.3%`
  - mean Phase 1 time: `27.4ms`

Interpretation:
- The hypothesis looks real when the query arrives after compression and the task is hard enough to create recoverable damage.
- `r=0.5` is mostly a ceiling sanity check on this synthetic pilot.

## Stable Default Run Tonight

This is the scoped default matrix for a single `A10G 24GB`.

Baselines:
- benchmarks: `RULER 4K` + `LongBench`
- methods:
  - `full_cache`
  - `snapkv_0.7`
  - `snapkv_0.5`
  - `snapkv_0.3`
  - `h2o_0.5`
  - `streaminglm`
  - `snapkv_0.3_isocompute`

IdleKV main sweep:
- benchmark: `RULER 4K`
- ratio: `r=0.7`
- budgets: `0, 50, 100, 500, 1000, 2000, 5000 ms`
- phases: `1`, `2`, `1+2`

Shadow-buffer ablations:
- benchmark: `RULER 4K`
- ratio: `r=0.7`
- buffer sizes: `0, 64, 128, 256, 512`

Models:
- `llama8b`
- `qwen7b`

Seeds:
- `42`
- `123`
- `456`

## Not In The Default A10G Matrix

These are intentionally excluded from the default overnight run on this machine:
- `sync_refresh`
- `LongBench + IdleKV`
- `8K RULER`

Why:
- the remaining hard memory limit is long decode/cache growth on a `24GB` card
- the default matrix is the set we believe is worth running tonight without pretending broader scope is already stable

## If Better Hardware Becomes Available

If you later get an `A100 40GB` or `80GB`, extend the matrix in this order:

1. `LongBench + IdleKV`
2. `8K RULER`
3. `sync_refresh`
4. broader cross-model / seed coverage if needed

Scale-up config:
- `configs/a100_scaleup.yaml`
- intended for A100-class GPUs
- broadens the default matrix to:
  - `RULER 4K + 8K`
  - `LongBench + IdleKV`
  - `sync_refresh`
- keeps the main A10G config untouched and honest

Do not automatically discard the A10G results:
- keep them as the constrained-hardware study
- add larger-GPU runs as the expanded scale-up tier

## Why 24GB Is Tight

The issue is mostly VRAM peak during long decode, not basic setup mistakes.

Already fixed:
- prefill no longer stores full hidden-state stacks just to get the SnapKV window
- online eviction no longer rebuilds the cache on every token by default

Remaining limit:
- model weights take most of the card
- long decode still pays for cache growth plus temporary cache copies
- this is why the stable A10G matrix is narrower than the aspirational full matrix

## Recommended Run Order

On one A10G, run in this order:

1. baselines
2. IdleKV
3. ablations

Commands:

```bash
python scripts/run_experiments.py --config configs/main.yaml --only-baselines
python scripts/run_experiments.py --config configs/main.yaml --only-idlekv
python scripts/run_experiments.py --config configs/main.yaml --only-ablations
```

Use `tmux` and consider `--skip-existing` if resuming after interruption.

## Source-Of-Truth Files

For a deeper handoff, read these next:
- `README.md`
- `SETUP.md`
- `TASKS.md`
- `configs/main.yaml`
- `paper/idlekv_scale.tex`
