# Preliminary Baseline Snapshot

Purpose:
- Preserve the first completed baseline outputs from the long A10G overnight run
  before pivoting to a smaller, higher-value preliminary IdleKV suite.

Context:
- Original queue order was `baselines -> idlekv -> ablations`.
- The full queue was progressing too slowly for the immediate research question.
- We stopped after four completed baseline files and pivoted to a reduced
  preliminary IdleKV matrix.

Included files:
- `full_cache_llama8b_seed42.json`
- `snapkv_0.7_llama8b_seed42.json`
- `snapkv_0.5_llama8b_seed42.json`
- `snapkv_0.3_llama8b_seed42.json`

Headline read:
- All four are at ceiling on `RULER 4K` (`avg_accuracy = 1.0`).
- LongBench average score degrades monotonically as compression gets more
  aggressive:
  - `full_cache`: `29.417`
  - `snapkv_0.3`: `29.251`
  - `snapkv_0.5`: `28.830`
  - `snapkv_0.7`: `28.460`

Interpretation:
- These files are useful as a preserved scouting baseline snapshot.
- They are not the final baseline matrix for the paper.
