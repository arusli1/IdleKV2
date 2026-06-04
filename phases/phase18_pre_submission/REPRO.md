# Phase 18 reproducibility recipe

Round-8 attack 2 defuse: a one-liner script + frozen configs so a
reviewer can rerun the full Phase 18 pipeline at the head of the
amendment chain.

## One-liner: full pipeline

```bash
git checkout 8041a13   # latest Phase 18 commit
bash phases/phase18_pre_submission/scripts/queue_post_tight_v5.sh
```

Total wall-clock: ~3-4 hours on a single 96GB Blackwell GPU.

## Pre-registration commit chain

| Commit | Description |
|---|---|
| `601d807` | v5 plan |
| `55e8bda` | v5.1 scope amendment (n=24→12, K=9→5) |
| `af2fd93` | gate-logic correction (post K-sweep) |
| `c1f08a7` | PageSummary fusion bug fix |
| `853dfb1` | chunk-size CLI |
| `e437c19` | outcome bands BEFORE rerun data |
| `934fb8b` | RepairKV-chunked condition |
| `e706b32` | runtime paragraph reframe |
| `25cc542` | tight K-sweep at mult 0.10 |
| `f1f8813` | bands for tight K-sweep |
| `29ac393` | nulls vs wins in pre-reg |
| `8041a13` | abstract reframe to "matches/dominates" |

## Expected sanity checks

1. K-sweep redo K=96: RepairKV ≈ 0.917, B_match ≈ 0.21, PageSummary ≈ 0.19 (post-fix), Oracle-K ≈ 1.0.
2. Tight sweep at multiplier 0.05: RepairKV 0.917 vs Refresh-K-budgeted 0.389 (Δ=+0.528, Holm p<1e-4).
3. Tight sweep at multiplier 1.05: RepairKV 0.917 vs Refresh-K-budgeted 1.000 (Δ=-0.083, matches clause).
4. GPU verify (PHASE18_SCORE_ON_GPU=1): RepairKV mean within ±0.02 of CPU K-sweep.
5. Llama K=32: RepairKV > B_match by ≥0.1 (cross-model robustness).

## Outputs after full pipeline

`phases/phase18_pre_submission/results/` contains:
- `w1/` main K-sweep redo + analysis CSVs
- `w1_tight/` 4 per-multiplier artifacts + tight_sweep_summary.csv
- `w1_tight_ksweep/` tight K-sweep at 150ms abs
- `chunk_size_sens/` per-chunk_size artifacts
- `llama_lowk/` Llama K=32, 48
- `gpu_verify/` n=12 GPU-scoring artifact
- `w2/` W2 runtime probe CSV + full_prefill.json
- `figures/` frontier figure, walltime bar, tight sweep figure
- `recency/` 12→34 partition appendix
- `RESULTS_FINAL.md` aggregate
- `RESULTS_AUDIT.md` initial audit + addenda
- `PRE_REG_BANDS.md` outcome bands
- `paper_edits_draft.md` green-marked paper edits (not applied)
