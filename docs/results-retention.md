# Results Retention

This repo now separates durable project memory from bulky generated output.

## Keep In The Repo

These files are the minimum lightweight record of what has been done so far:

- `saved_results/`
- `docs/phases/`
- `docs/project-status.md`

If you only need a compact memory of P0-P3, this is the set to preserve.

## Keep Locally For Deeper Analysis

These are the original phase outputs that are useful for rerunning analysis or
answering detailed questions later:

- `phases/phase0_baseline/results/`
- `phases/phase1_degradation/results/`
- `phases/phase2_kv_cache/results/`
- `phases/phase3_eviction/results/`

These directories are intentionally ignored by git because they are generated
and can grow large.

## Large Outputs That Can Be Archived Elsewhere

If space becomes a problem, archive these outside git before deleting them:

- `phases/phase0_baseline/artifacts/`
- `phases/phase0_baseline/logs/`
- `phases/phase1_degradation/artifacts/`
- `phases/phase1_degradation/logs/`
- `phases/phase1_degradation/results/`
- `phases/phase3_eviction/results/phase3_eviction_logs/`
- `phases/phase3_eviction/results/phase3_raw_examples/`

## Current Canonical Originals

The main originals behind the saved summaries are:

- P0: `phases/phase0_baseline/results/baseline_ruler.json`
- P1: `phases/phase1_degradation/results/vt4hop_permute_avg5_1024_32768_20260420_aggregate_summary.json`
- P1: `phases/phase1_degradation/results/smoke_postfixfix_lowk64_128_parallel_mq_niah_4q_summary.json`
- P2: `phases/phase2_kv_cache/results/phase2_summary.json`
- P2: `phases/phase2_kv_cache/results/phase2_run_report.md`
- P3: `phases/phase3_eviction/results/phase3_summary.json`
- P3: `phases/phase3_eviction/results/phase3_live_smoke.json`
- P3: `phases/phase3_eviction/results/phase3_degradation/pilot10/phase3_summary.json`
- P3: `phases/phase3_eviction/results/phase3_degradation/full100/phase3_summary.json`
