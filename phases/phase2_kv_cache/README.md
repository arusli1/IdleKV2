# Phase 2: KV Cache Load Verification

This folder contains the Phase 2 implementation for the IdleKV plan: a clean
KV-cache access layer, a repeatable test/measurement harness, and the
artifacts requested by `instructions.md`.

## What Is Here

- `MEMORY.md`
  External checklist and run notes for this phase.
- `instructions.md`
  The phase-specific spec that this implementation follows.
- `src/kv_utils.py`
  The five-function KV API plus cache conversion helpers and position tracking.
- `src/runtime.py`
  Shared runtime helpers for loading Qwen, building test prompts, running
  resume-from-cache checks, measuring transfer latency, and producing
  attention heatmaps.
- `tests/`
  `unittest` coverage for synthetic stress tests and live-Qwen integration
  checks.
- `scripts/run_phase2_suite.py`
  End-to-end runner for the Phase 2 test suite and artifact generation.
- `results/`
  Generated JSON summaries and PNG heatmaps.

## Local Environment Notes

The local `Qwen2.5-7B-Instruct` install in this repo does not match the older
shape assumptions written in the top-level plan. The live checks for this
environment are:

- `num_hidden_layers = 28`
- `num_attention_heads = 28`
- `num_key_value_heads = 4`
- `head_dim = 128`
- `past_key_values` is returned as `transformers.cache_utils.DynamicCache`
  with per-layer tensors stored as `DynamicLayer.keys` and `DynamicLayer.values`

The implementation in `src/kv_utils.py` normalizes those runtime-specific
details so downstream code can still work against a stable tuple-of-tuples
representation.

## How To Run

From the repo root:

```bash
source .venv/bin/activate
python phases/phase2_kv_cache/scripts/run_phase2_suite.py
```

The runner executes:

1. Synthetic stress tests for the KV API
2. Live round-trip identity and selective injection tests on local Qwen
3. CPU<->GPU KV transfer latency profiling
4. Attention heatmap generation at 4K and 8K context lengths

Outputs are written under `phases/phase2_kv_cache/results/`.
