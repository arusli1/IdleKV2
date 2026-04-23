# Phase 2 Validation Note

This document is the top-level Phase 2 validation note for the main
`phases/phase2_kv_cache` harness.

It captures the four checks we currently treat as the Phase 2 set:

1. `Round-trip identity`
2. `Selective injection`
3. `CPU<->GPU transfer latency`
4. `Attention heatmap generation`

This file is intentionally written as a stable Phase 2 readout for the KV-cache
access layer. Unlike the Phase 1 file, this phase is not a budget curve search.
It is an infrastructure gate: if these checks fail, nothing in later repair
phases can be trusted.

## Common Setup

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Local model dir: `models/Qwen2.5-7B-Instruct`
- Runtime stack: `torch 2.10.0+cu128`, `transformers 5.2.0`
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`, `~101.97 GB VRAM`
- Live cache format: `DynamicCache`
- Live model geometry on this machine:
  - `28` layers
  - `28` attention heads
  - `4` KV heads
  - `head_dim = 128`
- Phase 2 package root: `phases/phase2_kv_cache/`
- Result artifacts:
  - `phases/phase2_kv_cache/results/phase2_summary.json`
  - `phases/phase2_kv_cache/results/phase2_run_report.md`

## Validation 1: Round-trip Identity

### Task Definition

Prefill a context, save the KV cache to disk layer-by-layer, load it back, then
resume generation from the loaded cache.

The acceptance criterion is numeric identity of the resumed next-token logits,
up to the Phase 2 tolerance. In practice, this should be exactly equal on this
setup.

### Why We Keep It

This is the hard gate for the entire phase:

- it validates serialization and deserialization
- it validates cache-format normalization between tuple form and
  `DynamicCache`
- it validates explicit resume positioning via `position_ids` and
  `cache_position`

### Primary Metrics

- `max_abs_logit_diff`
- `pass`

### What A Failure Means

- nonzero drift means the save/load path is corrupting the cache, or
- the resumed forward pass is not aligned to the original logical positions, or
- the cache conversion layer is not reconstructing the model-facing cache
  correctly

### Current Result

- context length: `1024`
- dtype: `torch.bfloat16`
- `max_abs_logit_diff = 0.0`
- `pass = true`

This means the Phase 2 acceptance gate passed exactly on the local Qwen setup.

## Validation 2: Selective Injection

### Task Definition

Construct a fact-retrieval context with a known numeric answer, remove the fact
span from the KV cache, query the model, then inject the removed span back at
its original absolute positions and query again.

The current fact string is built around the access code:

- `99273`

### Why We Keep It

This is the semantic correctness check for the cache-editing API:

- `slice_kv` must remove the intended positions
- `inject_kv` must restore them in correct causal order
- `PositionTrackedCache` must preserve the original absolute positions of
  compressed cache slots

### Primary Metrics

- `reference_text`
- `degraded_text`
- `restored_text`
- `reference_vs_degraded_max_abs_logit_diff`
- `reference_vs_restored_max_abs_logit_diff`
- `recovered_text_match`

### What A Failure Means

- unchanged degraded output suggests the removed span was not actually
  important to the answer, or
- restored output failing to recover suggests the injection path is misordered,
  mispositioned, or otherwise corrupting the cache

### Current Result

- context length: `1536`
- query length: `18`
- removed fact token count: `21`
- reference text: `99273`
- degraded text: `1234567`
- restored text: `99273`
- `reference_vs_degraded_max_abs_logit_diff = 19.875`
- `reference_vs_restored_max_abs_logit_diff = 0.0`
- `recovered_text_match = true`

Interpretation: removing the fact span materially damages the answer path, and
injecting it back restores the exact reference behavior.

## Validation 3: CPU<->GPU Transfer Latency

### Task Definition

Build a full `32K` cache, slice out repaired-token fragments of several sizes,
and profile CPU<->GPU movement of those fragments.

This is not a correctness test. It is the feasibility ceiling for later repair
phases.

### Why We Keep It

Later phases will need to move selected cached tokens between CPU and GPU during
idle-time repair. If this transfer step is too slow, the later repair algorithm
will be budget-constrained before attention-recompute even begins.

### Primary Metrics

- `restore p50`
- `restore p90`
- `evict p50`
- `evict p90`
- implied throughput at `p50`

### What A Failure Means

- high restore times would shrink the feasible repair budget `K`
- if `restore p50` for `1000` tokens exceeded `1 second`, later phases would
  need to redraw the feasibility frontier around much smaller repair budgets

### Current Result

| Tokens | Restore p50 (ms) | Restore p90 (ms) | Evict p50 (ms) |
|---|---:|---:|---:|
| 100 | 0.292 | 0.306 | 1.172 |
| 500 | 0.648 | 0.655 | 2.966 |
| 1000 | 1.145 | 1.220 | 19.105 |
| 2000 | 2.184 | 2.302 | 52.494 |
| 5000 | 5.311 | 5.417 | 136.509 |

The main Phase 2 interpretation is favorable:

- restore latency is far below the warning threshold
- transfer bandwidth is not the bottleneck for later repair experiments on this
  machine
- the later feasibility ceiling is more likely to be shaped by attention
  recompute and selection logic than by raw PCIe transfer cost

## Validation 4: Attention Heatmap Generation

### Task Definition

Generate layer-averaged attention heatmaps at `4K` and `8K` context lengths,
plus a VT-like figure with marked hop positions.

The heatmaps are computed from layers:

- `0`
- `14`
- `27`

### Why We Keep It

This is the Phase 2 research artifact:

- it verifies that the harness can extract interpretable attention structure
- it gives a figure path for showing sinks, recency bias, and middle-context
  weakness
- it supports later attribution work on hop survival and eviction dead zones

### Primary Artifacts

- `results/phase2_attention_heatmaps/layer_avg_4k.png`
- `results/phase2_attention_heatmaps/layer_avg_8k.png`
- `results/phase2_attention_heatmaps/vt4hop_hop_positions_marked.png`

### What A Failure Means

- missing heatmaps means the phase still lacks one of its required paper-facing
  outputs
- if the exporter only works at tiny sequence lengths, later attention analysis
  is not ready for serious use

### Current Result

The required `4K` and `8K` heatmap artifacts were generated successfully.

The marked `8K` VT-like figure uses the following positions:

- `hop1 = 983`
- `hop2 = 3031`
- `hop3 = 5079`
- `hop4 = 6881`
- `query = 8175`

## Reporting Table

| Validation | Metric | Result |
|---|---:|---:|
| Round-trip identity | max_abs_logit_diff | `0.0` |
| Round-trip identity | pass | `true` |
| Selective injection | reference_vs_degraded_max_abs_logit_diff | `19.875` |
| Selective injection | reference_vs_restored_max_abs_logit_diff | `0.0` |
| Selective injection | recovered_text_match | `true` |
| Transfer latency | restore p50 at 1000 tokens | `1.145 ms` |
| Transfer latency | restore p50 at 5000 tokens | `5.311 ms` |
| Attention artifacts | required files written | `true` |

## Final Readout

The intended Phase 2 story is now clear:

- the KV access layer is trustworthy on the local setup
- save/load round-trip identity is exact
- semantic cache editing via slice plus reinjection is functioning correctly
- transfer latency is comfortably below the threshold that would threaten later
  repair feasibility
- the phase now has the required paper-facing attention artifacts

This is enough to treat Phase 2 as complete and unblock later phases that
depend on direct KV-cache manipulation.

## Current Run Notes

At the time this file was written:

- the full Phase 2 suite passed with `7/7` tests
- the main summary artifact is
  `phases/phase2_kv_cache/results/phase2_summary.json`
- the human-readable companion report is
  `phases/phase2_kv_cache/results/phase2_run_report.md`
- the top-level Phase 2 implementation lives under
  `phases/phase2_kv_cache/`
