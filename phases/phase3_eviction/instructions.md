# Phase 3: Eviction Algorithm Implementation and Validation

**Goal:** Implement SnapKV (standard and query-aware variants) and StreamingLLM under a unified interface that integrates with `PositionTrackedCache` from P2. Generate clean degradation curves across k_budget values and validate against published benchmark numbers from the KV eviction literature.

**Hardware context:** 96GB VRAM, ~20GB typically in use. Two-pass approach throughout — build the full KV cache, score, slice. No hooks, no monkey-patching.

---

## Architecture Overview

Every eviction algorithm returns the same output structure. This is the contract that P4, P5, and P6 all depend on.

```python
@dataclass
class EvictionResult:
    compressed: PositionTrackedCache    # k_budget tokens kept on GPU
    evicted: PositionTrackedCache       # evicted tokens moved to CPU immediately
    importance_scores: dict[int, float] # original_position → score for ALL tokens
    obs_window_q_vecs: torch.Tensor     # shape: [n_layers, obs_len, head_dim]
                                        # used by P4 buffer scoring
```

```python
class EvictionPolicy:
    def evict(
        self,
        full_cache: PositionTrackedCache,
        k_budget: int,
        obs_window: torch.Tensor | None = None,
    ) -> EvictionResult:
        raise NotImplementedError
```

Evicted tokens are moved to CPU-pinned memory inside `evict()`. The caller never manages the CPU/GPU split manually.

---

## Two-Pass Approach

1. Run the full context through the model with `use_cache=True` — produces the complete KV cache
2. Score tokens using key vectors from the observation window (or query tokens for the query-aware variant)
3. Slice the cache using `slice_kv` from P2

The full KV cache at 32K context is approximately 16GB for Qwen-7B. With 96GB VRAM and ~20GB in use, this fits comfortably. No memory tricks needed.

**Critical: do not use `output_attentions=True` at 32K.** The full `[batch, n_heads, seq, seq]` attention matrix at 32K is ~120GB per layer across all layers — it will OOM. Instead, recompute attention scores for the observation window only inside `_score_tokens()`, as shown below. This is cheaper and gives you exactly the scores you need.

```python
# The two-pass call: get full KV cache, then score it
with torch.no_grad():
    outputs = model(
        context_input_ids,
        use_cache=True,
        output_attentions=False      # do NOT set True at long contexts
    )

full_kv = to_tuple_cache(outputs.past_key_values)
```

---

## SnapKV Implementation

### Standard SnapKV

Observation window = last `obs_window_size` tokens of the context prompt.

```python
class SnapKV(EvictionPolicy):
    def __init__(
        self,
        obs_window_size: int = 32,
        sink_size: int = 4,
        recency_window: int = 1024,
        pooling: str = "max",
    ):
        self.obs_window_size = obs_window_size
        self.sink_size = sink_size
        self.recency_window = recency_window
        self.pooling = pooling

    def evict(
        self,
        full_cache: PositionTrackedCache,
        k_budget: int,
        obs_window: torch.Tensor | None = None,
    ) -> EvictionResult:
        full_kv = full_cache.kv
        all_positions = full_cache.positions
        seq_len = len(all_positions)

        importance = self._score_tokens(full_kv, seq_len)

        sink_positions = set(all_positions[:self.sink_size])
        recency_positions = set(all_positions[-self.recency_window:])
        mandatory = sink_positions | recency_positions

        mandatory_indices = [i for i, p in enumerate(all_positions) if p in mandatory]
        n_mandatory = len(set(mandatory_indices))
        n_selectable = max(0, k_budget - n_mandatory)

        candidate_indices = [i for i, p in enumerate(all_positions)
                             if p not in mandatory]
        candidate_scores = importance[candidate_indices]

        if n_selectable > 0 and len(candidate_indices) > 0:
            topk = min(n_selectable, len(candidate_indices))
            _, top_indices = candidate_scores.topk(topk)
            selected_indices = [candidate_indices[i.item()] for i in top_indices]
        else:
            selected_indices = []

        keep_indices = sorted(set(mandatory_indices) | set(selected_indices))
        evict_indices = [i for i in range(seq_len) if i not in set(keep_indices)]

        keep_positions = [all_positions[i] for i in keep_indices]
        evict_positions = [all_positions[i] for i in evict_indices]

        compressed_kv = slice_kv(full_kv, keep_indices)
        evicted_kv = slice_kv(full_kv, evict_indices)

        # Move evicted tokens to CPU-pinned memory immediately
        evicted_kv_cpu = tuple(
            (k.cpu().pin_memory(), v.cpu().pin_memory())
            for k, v in evicted_kv
        )

        scores_dict = {
            all_positions[i]: importance[i].item()
            for i in range(seq_len)
        }

        obs_q_vecs = self._extract_obs_q_vecs(full_kv, seq_len)

        return EvictionResult(
            compressed=PositionTrackedCache(kv=compressed_kv, positions=keep_positions),
            evicted=PositionTrackedCache(kv=evicted_kv_cpu, positions=evict_positions),
            importance_scores=scores_dict,
            obs_window_q_vecs=obs_q_vecs,
        )

    def _score_tokens(self, full_kv: tuple, seq_len: int) -> torch.Tensor:
        """
        Compute per-token importance by running attention from the observation
        window rows against all key vectors. Avoids materializing the full
        seq × seq attention matrix.
        """
        obs_start = max(0, seq_len - self.obs_window_size)

        layer_scores = []
        for k, v in full_kv:
            # k: [1, n_kv_heads, seq_len, head_dim]
            k_queries = k[:, :, obs_start:, :]    # [1, n_kv_heads, obs_len, head_dim]
            head_dim = k.shape[-1]

            # Scores: [1, n_kv_heads, obs_len, seq_len]
            scores = torch.matmul(k_queries, k.transpose(-2, -1)) / (head_dim ** 0.5)
            scores = torch.softmax(scores, dim=-1)

            if self.pooling == "max":
                token_scores = scores.max(dim=2).values   # [1, n_kv_heads, seq_len]
            else:
                token_scores = scores.mean(dim=2)

            token_scores = token_scores.mean(dim=[0, 1])  # [seq_len]
            layer_scores.append(token_scores)

        return torch.stack(layer_scores, dim=0).mean(dim=0)  # [seq_len]

    def _extract_obs_q_vecs(self, full_kv: tuple, seq_len: int) -> torch.Tensor:
        """
        Extract key vectors from the observation window as a proxy for query vectors.
        Shape: [n_layers, obs_len, head_dim], averaged across KV heads, on CPU.
        """
        obs_start = max(0, seq_len - self.obs_window_size)
        obs_vecs = []
        for k, v in full_kv:
            obs_k = k[0, :, obs_start:, :].mean(dim=0)  # [obs_len, head_dim]
            obs_vecs.append(obs_k.cpu())
        return torch.stack(obs_vecs, dim=0)  # [n_layers, obs_len, head_dim]
```

### Query-Aware SnapKV

Uses the actual query tokens as the observation window instead of the end of the context. In an agentic workload, the model has already generated the tool call and knows what question will be asked after the tool returns. Scoring evicted tokens against the actual query gives strictly better importance estimates than scoring against the context tail.

```python
class QueryAwareSnapKV(SnapKV):
    """
    SnapKV variant using actual query tokens as the observation window.
    Requires query token ids passed as obs_window at eviction time.
    """
    def evict(
        self,
        full_cache: PositionTrackedCache,
        k_budget: int,
        obs_window: torch.Tensor | None = None,
    ) -> EvictionResult:
        if obs_window is None:
            raise ValueError(
                "QueryAwareSnapKV requires obs_window (query token ids). "
                "Use standard SnapKV if the query is unavailable at eviction time."
            )

        # Run query tokens through the model attending to the full context cache.
        # This gives attention weights from query positions to all context positions.
        with torch.no_grad():
            query_out = model(
                obs_window,
                past_key_values=full_cache.kv,
                use_cache=True,
                output_attentions=False
            )

        extended_kv = to_tuple_cache(query_out.past_key_values)
        context_len = len(full_cache.positions)

        # Score context tokens using query key vectors as the observation window
        layer_scores = []
        for k, v in extended_kv:
            k_query_rows = k[:, :, context_len:, :]         # [1, heads, query_len, head_dim]
            k_context = k[:, :, :context_len, :]             # [1, heads, context_len, head_dim]
            head_dim = k.shape[-1]

            scores = torch.matmul(k_query_rows, k_context.transpose(-2, -1))
            scores = scores / (head_dim ** 0.5)
            scores = torch.softmax(scores, dim=-1)

            if self.pooling == "max":
                token_scores = scores.max(dim=2).values.mean(dim=[0, 1])
            else:
                token_scores = scores.mean(dim=[0, 1, 2])

            layer_scores.append(token_scores)

        importance = torch.stack(layer_scores, dim=0).mean(dim=0)  # [context_len]

        # Reuse parent keep/evict/slice logic with pre-computed importance
        return self._evict_with_scores(full_cache, k_budget, importance, extended_kv)
```

Refactor the parent class to accept a pre-computed importance tensor in `_evict_with_scores()` so the keep/evict/slice logic is not duplicated.

---

## StreamingLLM Implementation

StreamingLLM is the structural lower bound — sinks plus recency window, everything in the middle discarded. Any importance-scoring method should substantially outperform it on VT and MQ-NIAH.

```python
class StreamingLLM(EvictionPolicy):
    def __init__(self, sink_size: int = 4, recency_window: int = 1024):
        self.sink_size = sink_size
        self.recency_window = recency_window

    def evict(
        self,
        full_cache: PositionTrackedCache,
        k_budget: int,
        obs_window: torch.Tensor | None = None,
    ) -> EvictionResult:
        all_positions = full_cache.positions
        seq_len = len(all_positions)

        sink_indices = list(range(min(self.sink_size, seq_len)))
        recency_start = max(self.sink_size, seq_len - self.recency_window)
        recency_indices = list(range(recency_start, seq_len))

        keep_indices = sorted(set(sink_indices) | set(recency_indices))
        evict_indices = [i for i in range(seq_len) if i not in set(keep_indices)]

        keep_positions = [all_positions[i] for i in keep_indices]
        evict_positions = [all_positions[i] for i in evict_indices]

        compressed_kv = slice_kv(full_cache.kv, keep_indices)
        evicted_kv = slice_kv(full_cache.kv, evict_indices)
        evicted_kv_cpu = tuple(
            (k.cpu().pin_memory(), v.cpu().pin_memory())
            for k, v in evicted_kv
        )

        scores_dict = {p: 1.0 for p in keep_positions}
        scores_dict.update({p: 0.0 for p in evict_positions})

        return EvictionResult(
            compressed=PositionTrackedCache(kv=compressed_kv, positions=keep_positions),
            evicted=PositionTrackedCache(kv=evicted_kv_cpu, positions=evict_positions),
            importance_scores=scores_dict,
            obs_window_q_vecs=torch.zeros(
                len(full_cache.kv), 1, full_cache.kv[0][0].shape[-1]
            ),
        )
```

---

## Eviction Log Format

Write a log entry for every inference call before running the model on the compressed cache.

```python
def log_eviction(
    result: EvictionResult,
    example_id: str,
    task: str,
    task_relevant_positions: list[int],
    log_dir: str
):
    survived = [p in set(result.compressed.positions)
                for p in task_relevant_positions]

    entry = {
        "example_id": example_id,
        "task": task,
        "k_budget": len(result.compressed.positions),
        "seq_len": len(result.compressed.positions) + len(result.evicted.positions),
        "kept_positions": result.compressed.positions,
        "evicted_positions": result.evicted.positions,
        "task_relevant_positions": task_relevant_positions,
        "task_relevant_survived": survived,
        "importance_scores": {str(k): v for k, v in result.importance_scores.items()},
    }

    with open(os.path.join(log_dir, f"{example_id}.json"), "w") as f:
        json.dump(entry, f)

    torch.save(
        result.obs_window_q_vecs,
        os.path.join(log_dir, f"{example_id}_qvecs.pt")
    )
```

---

## Degradation Curves: What to Run and What to Expect

### Experimental Configuration

Generate fresh degradation curves using standard RULER-KVR task definitions. Do not compare directly to Phase 1 numbers — Phase 1 used modified task variants (permuted VT chain, distractor variable) that are not directly comparable.

| Parameter | Values |
|-----------|--------|
| Context length | 32K |
| Tasks | VT-4hop, MQ-NIAH-4q, S-NIAH |
| Eviction methods | SnapKV, QueryAwareSnapKV, StreamingLLM |
| k_budget | 128, 256, 512, 1024, 2048, FullKV |
| Examples per cell | 100 |

Total: 3 tasks × 3 methods × 6 budget values × 100 examples = 5,400 calls. At 5–8 seconds per 32K call, approximately 8–12 hours. The three tasks are independent — parallelize if you have multiple GPUs or stagger runs.

If runtime is the constraint, drop k_budget=128 and k_budget=1024 and run 4 budget points. The curve shape is still clear with FullKV, 2048, 512, 256 as the four points.

### Expected Results and Literature Anchors

Use these ranges as sanity checks against published numbers. Numbers significantly outside these ranges indicate an implementation issue.

**VT-4hop at 32K:**

| Method | k_budget | Expected accuracy | Source |
|--------|----------|-------------------|--------|
| FullKV | — | 85–95% | RULER paper, Qwen-class 7B models at 32K |
| SnapKV | 2048 | 65–80% | Moderate degradation; sinks and recency partially cover early hops |
| SnapKV | 512 | 25–50% | Severe; middle hops evicted consistently |
| SnapKV | 256 | 5–25% | Near floor; only hop 4 (recency window) reliably retrieved |
| StreamingLLM | any | 0–15% | RefreshKV (2025): eviction-based methods achieve < 20% on chain tasks with > 2 hops |
| QueryAwareSnapKV | 512 | 40–65% | Should outperform standard SnapKV by 10–20pp at this budget |

**MQ-NIAH-4q at 32K (mean recall / 4):**

| Method | k_budget | Expected mean recall | Notes |
|--------|----------|---------------------|-------|
| FullKV | — | 3.7–3.9 | Near-perfect |
| SnapKV | 512 | 2.0–2.5 | Depths 10%, 37% likely evicted |
| SnapKV | 256 | 1.0–1.5 | Only depth-90% needle in recency window |
| StreamingLLM | any | 0.8–1.2 | Only recency-window needle survives |
| QueryAwareSnapKV | 512 | 2.5–3.2 | Query explicitly names keys; larger advantage than VT |

**S-NIAH at 32K:**

| Method | k_budget | Expected accuracy |
|--------|----------|-------------------|
| FullKV | — | 95–99% |
| SnapKV | 512 | 70–85% |
| SnapKV | 256 | 50–70% |
| StreamingLLM | any | 30–50% |

**The query-aware gap:**

Ada-KV (2024) reports that at budget=2048 on Llama, SnapKV drops from 49.09 to 42.86 (about 6 points) when moving from question-aware to question-agnostic settings. At smaller budgets (512, 256) the gap should be larger because the selection decision matters more when you can keep only a small fraction of tokens. If your QueryAwareSnapKV vs SnapKV gap is smaller than expected, check: (1) that query tokens are being processed correctly against the full context cache, and (2) that the query text is informative — "What is the value of VAR_A?" should attend to the hop-1 link but not necessarily hops 2–4, so the query-aware advantage may be smaller for VT than for MQ-NIAH.

### Plotting

**Plot 1 — Accuracy vs k_budget (the main degradation curve):**
X-axis: k_budget on a log scale (128, 256, 512, 1024, 2048, FullKV). Y-axis: accuracy (or mean recall for MQ-NIAH). Three lines: SnapKV, QueryAwareSnapKV, StreamingLLM. FullKV as horizontal dashed reference line. One plot per task. This is your paper's Figure 2.

**Plot 2 — Eviction survival rate by hop number (mechanistic attribution):**
For VT-4hop only. X-axis: k_budget. Y-axis: fraction of examples where each hop link survived eviction. Four lines, one per hop. The crossing pattern — hop 4 surviving at all budgets, hops 2–3 dropping out first, hop 1 intermediate — is the mechanistic explanation for why VT fails. This goes in the motivation section alongside the P2 attention heatmaps.

---

## Validation Checks

### Check 1: FullKV Matches P0

Run FullKV (no eviction) for all three tasks at 32K. These scores must match P0 baseline scores within ±2%. If they diverge, the inference pipeline has changed between phases. This is different from comparing to Phase 1 — Phase 1 used modified tasks. Matching P0 uses the same standard task definitions.

### Check 2: StreamingLLM Floor

StreamingLLM on VT-4hop must score below 15% at any k_budget. If it scores above 30%, the hop links are accidentally falling in the recency window — adjust hop depth placement so hops 2 and 3 land at depths 35–65%, well outside the recency window.

### Check 3: Eviction Log Attribution

For VT-4hop failures at k_budget=512 under SnapKV, pull the eviction logs and check which hop numbers appear in `evicted_positions`:

```python
failures = [e for e in logs if not e["correct"]]
for hop_idx in range(4):
    evicted = sum(1 for e in failures if not e["task_relevant_survived"][hop_idx])
    print(f"Hop {hop_idx+1} evicted in {evicted}/{len(failures)} failures")
```

Expected: hops 2 and 3 account for > 60% of failures. If hop 1 or hop 4 dominate, the depth configuration is off — hop 1 should be at ~12% (close enough to observation window to sometimes survive) and hop 4 at ~87% (in or near recency window, usually kept).

### Check 4: Query-Aware vs Standard

At k_budget=512, QueryAwareSnapKV must outperform standard SnapKV on both VT-4hop and MQ-NIAH-4q. A smaller gap on VT than on MQ-NIAH is expected and interpretable. Zero gap on both tasks means the query processing is broken — check that `obs_window` is being passed and that `extended_kv` is indexed correctly at `context_len:`.

---

## Hyperparameter Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `obs_window_size` | 32 | Standard SnapKV default |
| `sink_size` | 4 | Standard across all eviction literature |
| `recency_window` | min(1024, k_budget - sink_size) | Cap at k_budget minus sinks to avoid budget overrun at small budgets |
| `pooling` | `"max"` | Max over observation window rows; RefreshKV Table 8 confirms max > mean for GQA models |
| StreamingLLM `recency_window` | Match k_budget - sink_size | For fair comparison at each budget level |

**Recency window capping:** At k_budget=256 with recency_window=1024, the mandatory keepers already exceed the budget. Cap recency_window at `k_budget - sink_size` and apply this cap consistently across both SnapKV and StreamingLLM.

---

## File Structure

```
src/
  eviction/
    __init__.py
    base.py               # EvictionPolicy, EvictionResult, PositionTrackedCache
    snapkv.py             # SnapKV, QueryAwareSnapKV
    streaming_llm.py      # StreamingLLM
    logging.py            # log_eviction()

results/
  phase3_degradation/
    VT4hop_degradation.json
    MQNIAH_degradation.json
    SNIAH_degradation.json
    figures/
      VT4hop_accuracy_vs_budget.png
      VT4hop_survival_by_hop.png
      MQNIAH_recall_vs_budget.png
      query_aware_comparison.png
  phase3_eviction_logs/
    VT4hop_snapkv_k512/
      ex001.json
      ex001_qvecs.pt
      ...
```

---

## Deliverable

`src/eviction/` with SnapKV, QueryAwareSnapKV, and StreamingLLM. Degradation curves for VT-4hop and MQ-NIAH across all budget values with FullKV confirmed against P0. Hop-level survival breakdown for VT. Query-aware vs standard comparison. All `_qvecs.pt` files saved and indexed — P4 needs these immediately.

Before moving to P4: confirm `results/phase3_degradation/VT4hop_degradation.json` shows SnapKV at k_budget=512 producing accuracy in the 25–60% range. Outside this range indicates either broken eviction (too high) or task configuration problems (too low — check hop depths against expected literature values).
