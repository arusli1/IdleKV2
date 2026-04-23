# Phase 2 Run Report

## Environment

- Model dir: `/home/ubuntu/IdleKV/models/Qwen2.5-7B-Instruct`
- Layers: `28`
- KV heads: `4`
- Cache runtime type: `DynamicCache`

## Round-trip identity

- Pass: `True`
- Max abs logit diff: `0`

## Selective injection

- Reference text: `99273\n\`\`\``
- Degraded text: `1234567`
- Restored text: `99273\n\`\`\``
- Restored text matches reference: `True`
- Reference vs degraded max logit diff: `19.875`
- Reference vs restored max logit diff: `0`

## Transfer latency

- Sizes profiled: `100, 500, 1000, 2000, 5000`

## Heatmaps

- `phase2_attention_heatmaps/layer_avg_4k.png`
- `phase2_attention_heatmaps/layer_avg_8k.png`
- `phase2_attention_heatmaps/vt4hop_hop_positions_marked.png`
