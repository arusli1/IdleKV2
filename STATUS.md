# IdleKV Status

Read this first if you are taking over the repo on a fresh SSH/Codex session.

## Current State

- Core runtime is implemented and the current test suite passes.
- The delayed-query pilot is validated.
- The experiment runner is real and resumable; it is no longer a mock scaffold.
- The repo is intentionally scoped to a stable single-A10G matrix with a
  sharper `SCALE 2026` workshop-core path on top.
- A reduced preliminary IdleKV scout is now complete and is more informative
  than the original long baseline-first queue for choosing the next runs.
- The Qwen seed follow-up scout is also complete and has materially narrowed
  the workshop-core claim.

## Submission Target

- Primary near-term target: `ICML SCALE 2026` workshop submission.
- Preferred form: `7` content pages plus references.
- Fallback if the broader matrix is not ready in time: `3`-page late-breaker.
- Longer-term expansion target: `NeurIPS 2026` main-track version after a
  broader larger-GPU matrix is measured.

Critical read on evidence bar:
- A credible `SCALE 2026` paper does not need the full aspirational matrix.
- It does need one clean mechanism claim, one informative benchmark slice, and
  an honest scope boundary.
- `NeurIPS 2026` is a different bar: broader benchmarks, more models/seeds,
  stronger systems metrics, and a cleaner final story on whether Phase 2
  belongs at all.

Current validated test status:
- `44 passed`

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

## Preliminary IdleKV Scout

Finished reduced scout:
- config: `configs/preliminary_idlekv.yaml`
- scope:
  - `llama8b`, `qwen7b`
  - seed `42`
  - `RULER 4K`
  - `r=0.7`
  - budgets `0`, `100`, `1000 ms`
  - phases `1`, `1+2`

Headline read:
- `llama8b`
  - all conditions at `100%`
  - this slice is ceiling and not useful for tuning
- `qwen7b`
  - `0ms`: `0.86`
  - `100ms`: `0.90`
  - `1000ms phase=1`: `0.90`
  - `1000ms phase=1+2`: `0.667`

Interpretation:
- `qwen7b` is the informative tuning model on the current `RULER 4K` slice
- `100ms` is a credible short-idle operating point
- Phase 1 is carrying the useful signal
- Phase 2 needs more scrutiny before it earns a place in the large run
- the current `llama8b` `RULER 4K` slice should not be treated as the main
  discriminative evaluation setting

Current critical read:
- The strongest workshop-grade claim is now ``Phase 1 at 100ms helps on an
  informative delayed-query compressed-cache slice.''
- That is enough to anchor a `7`-page `SCALE` paper if we support it with a
  small but coherent matrix.
- It is not enough for a strong `NeurIPS` paper by itself, and it is not yet a
  reason to center the story on Phase 2.

Tracked snapshot:
- `tracked_results/preliminary_idlekv/`

## Qwen Seed Follow-up Scout

Finished follow-up:
- config: `configs/qwen_seed_followup.yaml`
- scope:
  - `qwen7b`
  - seeds `123`, `456`
  - `RULER 4K`
  - `r=0.7`
  - conditions:
    - `0ms, phase=1`
    - `100ms, phase=1`
    - `1000ms, phase=1`
    - `1000ms, phase=1+2`

Headline read:
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
- the `0ms -> 100ms` Phase 1 gain is robust across both follow-up seeds
- there is no evidence that giving Phase 1 more than `100ms` helps on this
  slice
- `1000ms phase=1+2` is not merely noisy; it fails badly on the informative
  subtask in both seeds

Decision read:
- `Phase 1 @ 100ms` is now the best-supported workshop-core operating point
- Phase 2 should be treated as a separate debugging / scale-up topic, not part
  of the `SCALE 2026` core claim
- `llama8b` on `RULER 4K` remains a ceiling slice and should not drive method
  selection

Tracked snapshot:
- `tracked_results/qwen_seed_followup/`

## Stable Default A10G Matrix

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

These are intentionally excluded from the default scoped run on this machine:
- `sync_refresh`
- `LongBench + IdleKV`
- `8K RULER`

Why:
- the remaining hard memory limit is long decode/cache growth on a `24GB` card
- the default matrix is the set we believe is worth running on this hardware
  without pretending broader scope is already stable

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

## What Is Enough For Which Paper

Likely enough for a strong `SCALE 2026` `7`-page submission:
- delayed-query pilot at `r=0.7`
- informative Qwen scouts showing a robust `0ms -> 100ms` gain
- one sharper follow-up matrix that locks the main setting
- honest positioning: Phase 1 is the durable claim, Phase 2 is explicitly not
  part of the core story yet

Possible `3`-page late-breaker package if time is tight:
- delayed-query pilot
- Qwen scout / follow-up only
- one concise figure showing the short-idle gain
- no attempt to claim the full multi-benchmark story yet

Not enough for `NeurIPS 2026` main track:
- only one informative benchmark slice
- no broad larger-GPU IdleKV matrix yet
- no final throughput/latency Pareto package
- Phase 2 still unstable on the informative Qwen slice

## Why 24GB Is Tight

The issue is mostly VRAM peak during long decode, not basic setup mistakes.

Already fixed:
- prefill no longer stores full hidden-state stacks just to get the SnapKV window
- online eviction no longer rebuilds the cache on every token by default
- manager prefill now requests only the final token logits instead of the full
  `[seq_len, vocab]` slab, removing an avoidable 4K-8K prefill peak

Remaining limit:
- model weights take most of the card
- long decode still pays for cache growth plus temporary cache copies
- this is why the stable A10G matrix is narrower than the aspirational full matrix

## Recommended Run Order

On one A10G, run in this order:

1. preliminary scouts that choose the final matrix
2. scoped baselines / ablations only after the final matrix is sharper
3. larger expansion on A100-class hardware

Commands:

```bash
python scripts/run_experiments.py --config configs/preliminary_idlekv.yaml --only-idlekv
python scripts/run_experiments.py --config configs/qwen_seed_followup.yaml --only-idlekv
python scripts/throughput_spotcheck.py --model Qwen/Qwen2.5-7B-Instruct --context-length 4096 --num-trials 3 --num-measure-tokens 128
python scripts/run_experiments.py --config configs/llama_hardness_probe.yaml --num-samples 10
python scripts/run_experiments.py --config configs/scale_core.yaml
python scripts/run_experiments.py --config configs/scale_mechanism_ablation.yaml --only-ablations
```

## Next Critical Small Runs

Do these before committing to a broader `SCALE` matrix:

1. Throughput / wall-clock spot-check on the shortlisted method
   - script: `scripts/throughput_spotcheck.py`
   - compare compressed no-idle `r=0.7` against IdleKV `r=0.7, 100ms, phase=1`
   - goal: confirm the workshop claim that quality improves without materially
     changing the practical decode operating point
2. One harder Llama probe
   - config: `configs/llama_hardness_probe.yaml`
   - run with `--num-samples 10`
   - do not run more `Llama RULER 4K` ceiling checks
   - use a tiny harder slice to decide whether Llama belongs in the workshop
     matrix or should be deferred to the larger-GPU expansion

If those confirm the current read, the workshop-core matrix is:
- config: `configs/scale_core.yaml`
- optional minimal mechanism check: `configs/scale_mechanism_ablation.yaml`

Exact commands:

```bash
python scripts/throughput_spotcheck.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --context-length 4096 \
  --num-trials 3 \
  --num-measure-tokens 128

python scripts/run_experiments.py \
  --config configs/llama_hardness_probe.yaml \
  --num-samples 10

python scripts/run_experiments.py \
  --config configs/scale_core.yaml

python scripts/run_experiments.py \
  --config configs/scale_mechanism_ablation.yaml \
  --only-ablations
```

Use `tmux` and consider `--skip-existing` if resuming after interruption.

## Source-Of-Truth Files

For a deeper handoff, read these next:
- `README.md`
- `SETUP.md`
- `TASKS.md`
- `configs/main.yaml`
- `paper/idlekv_scale.tex`
