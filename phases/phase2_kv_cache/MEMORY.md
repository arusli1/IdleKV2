# Phase 2 Memory

## Goal

Build the Phase 2 KV-cache access layer, run the acceptance checks from
`instructions.md`, stress-test the logic, and record the results.

## Verified Environment Facts

- Repo: `/home/ubuntu/IdleKV`
- Model path: `/home/ubuntu/IdleKV/models/Qwen2.5-7B-Instruct`
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, ~101.97 GB VRAM
- `torch == 2.10.0+cu128`
- `transformers == 5.2.0`
- Live Qwen config:
  - 28 layers
  - 28 attention heads
  - 4 KV heads
  - head dim 128
- Live cache format:
  - `DynamicCache`
  - layer tensors available as `DynamicLayer.keys` / `DynamicLayer.values`

## Execution Checklist

- [x] Re-read the repo-level plan and the Phase 2 instructions
- [x] Inspect the local model and verify the real cache shape assumptions
- [x] Create the Phase 2 package structure and basic files
- [x] Implement `save_kv`, `load_kv`, `slice_kv`, `merge_kv`, `inject_kv`
- [x] Add `PositionTrackedCache` and cache conversion helpers
- [x] Add synthetic stress tests for the core logic
- [x] Run round-trip identity on live Qwen
- [x] Run selective injection on live Qwen
- [x] Run CPU<->GPU transfer latency profiling and save JSON
- [x] Generate 4K / 8K attention heatmaps and a hop-marked figure
- [x] Interpret the results and summarize follow-up risks

## Notes To Preserve During Execution

- Use the phase instructions as the source of truth for required artifacts.
- Keep the API pure except for disk I/O in `save_kv` / `load_kv`.
- Resume generation from modified caches with explicit `position_ids` and
  `cache_position`; the local stack is `transformers 5.2`, so the model wants
  a cache object on resume rather than a raw legacy tuple.
- Heatmap extraction must be layer-restricted and chunked to stay memory-safe
  at 4K and 8K context lengths.

## Final Outcome

- The Phase 2 acceptance gate passed: save/load round-trip resumed with
  `max_abs_logit_diff = 0.0`.
- The selective injection check passed: removing the fact span changed the
  answer (`99273` -> `1234567`), and reinjection restored the exact reference
  logits and text.
- The stress suite passed all 64 randomized recovery trials plus the direct
  save/load and cache-conversion checks.
- Restore latency on this machine is well below the phase warning threshold:
  p50 was about `1.15 ms` for 1K tokens and `5.31 ms` for 5K tokens, so
  transfer bandwidth is not the limiting factor for later repair experiments on
  this GPU/PCIe setup.
- Required JSON summaries and the 4K/8K heatmap PNGs are present under
  `results/`.
