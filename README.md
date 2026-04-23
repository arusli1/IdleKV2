# IdleKV

IdleKV explores a simple idea for batch-1 LLM agents: use GPU idle time during
tool calls to repair KV-cache damage caused by eviction.

## Repo Layout

- `phases/phase0_baseline/`: Phase 0 baseline RULER validation.
- `phases/phase1_degradation/`: Phase 1 degradation harness and task variants.
- `phases/phase2_kv_cache/`: Phase 2 KV save/load/slice/inject validation.
- `phases/phase3_eviction/`: Phase 3 eviction policies, smoke suite, and degradation benchmark.
- `ruler/`: vendored upstream RULER checkout.
- `paper/`: paper draft and figures.
- `saved_results/`: tracked memory layer with the canonical P0-P3 summaries and the small Phase 3 launcher log/status pair.
- `docs/`: phase notes, current status, and result-retention guidance.
- `models/`: local model weights only, ignored by git.

## Clean Repo Guide

- Keep runnable project code under `phases/`.
- Keep bulky generated outputs inside each phase under `results/`, `artifacts/`, or `logs/`.
- Copy only the small summaries worth remembering into `saved_results/`.
- Keep human-readable project memory in `docs/`.
- Treat `ruler/` as vendored code, not project source.
- Do not track local model weights or full generated output trees.
- If you need to prune local outputs later, read `docs/results-retention.md` first.

## Current State

- P0 baseline: done.
- P1 degradation harness: done enough for smoke, not yet polished into final
  experiment tables.
- P2 KV access layer: validated.
- P3 eviction validation: done enough for smoke.
- P4 CPU eviction buffer: next.

See `docs/project-status.md` for the current phase-by-phase readout.

## Phase Notes

- `docs/phases/phase0.md`
- `docs/phases/phase1.md`
- `docs/phases/phase2.md`
- `docs/phases/phase3.md`

## Key Docs

- `plan.md`: full development plan.
- `instructions.md`: current working spec.
- `docs/project-status.md`: concise current repo state.
- `docs/results-retention.md`: what to keep vs what can be regenerated.
