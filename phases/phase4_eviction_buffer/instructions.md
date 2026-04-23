# Phase 4: CPU Eviction Buffer

**Storage, Selection, and Profiling**

> **Goal:** Build the `EvictionBuffer` class that stores evicted token KV pairs on CPU, implements four candidate selection strategies, and profiles every operation that will run during the tool call idle window. The profiling table produced here determines the maximum repair budget K for every tool call duration in P6 — it is load-bearing infrastructure, not optional characterization.

---

## Table of Contents

1. [What Phase 4 Produces and Why It Matters](#what-phase-4-produces-and-why-it-matters)
2. [Memory Layout](#memory-layout)
3. [EvictionBuffer Class](#evictionbuffer-class)
4. [Extracting Recent Q Vectors from the Active Cache](#extracting-recent-q-vectors-from-the-active-cache)
5. [Profiling Suite](#profiling-suite)
   - Op 1: CPU → GPU Transfer
   - Op 2: CPU Scoring (Buffer Query)
   - Op 3: Attention Recompute Cost (Post-Injection)
   - Op 4: End-to-End Repair Pipeline
6. [The Feasibility Frontier](#the-feasibility-frontier)
7. [Scoring Strategy Comparison](#scoring-strategy-comparison)
8. [Connecting to Phase 3 Logs](#connecting-to-phase-3-logs)
9. [Validation Checks](#validation-checks)
10. [File Structure](#file-structure)
11. [Deliverable](#deliverable)

---

## What Phase 4 Produces and Why It Matters

The repair algorithm in P6 works like this: during a tool call, take evicted tokens from the CPU buffer, score them against recent query vectors to identify which ones are now relevant, move the top-K back to GPU, and inject them into the active cache before the next LLM turn.

Every step of that pipeline has a cost: scoring N candidates on CPU, transferring K winners to GPU, and the downstream attention cost of having K extra tokens in the cache. Phase 4 measures all of these costs on your actual hardware so that P6 does not make infeasible claims about repair budget.

The key output is the **feasibility frontier**: given a tool call idle window of T seconds, what is the maximum K (tokens repaired) you can afford? This table appears directly in the paper's system design section and sets the scope of every ablation in P6.

---

## Memory Layout

Each evicted token's KV pairs cost:

```
n_layers × 2 tensors × n_kv_heads × 1 token × head_dim × bytes_per_element
```

For Qwen-7B at fp16:

```
32 × 2 × 32 × 1 × 128 × 2 = 524,288 bytes ≈ 512KB per token
```

**Storing N evicted tokens:**

| N tokens | CPU RAM |
|---------:|--------:|
| 100      | ~50MB   |
| 500      | ~250MB  |
| 1,000    | ~512MB  |
| 5,000    | ~2.5GB  |
| 10,000   | ~5GB    |

All of these are feasible — a modern workstation has 64–256GB of RAM. The constraint is not storage but transfer bandwidth and scoring latency during the idle window.

> **Always allocate eviction buffer storage as pinned (page-locked) memory.** Pinned memory uses DMA for GPU transfers, bypassing the CPU and roughly doubling PCIe throughput compared to pageable memory. This is the single biggest latency optimization available without any algorithmic changes.

```python
# Pinned allocation (done inside EvictionBuffer.push)
k_pinned = k_tensor.cpu().pin_memory()
v_pinned = v_tensor.cpu().pin_memory()
```

---

## EvictionBuffer Class

```python
from dataclasses import dataclass, field
from typing import Literal
import torch
import numpy as np

SelectionStrategy = Literal["l2_norm", "dot_product", "random", "recency_inverse"]


@dataclass
class BufferEntry:
    position: int                        # original absolute sequence position
    kv: tuple                            # (key, value) per layer, on CPU pinned memory
    importance_score: float              # score assigned at eviction time
    q_vec: torch.Tensor                  # observation window Q vector, on CPU
                                         # shape: [n_layers, head_dim], averaged across obs_len


class EvictionBuffer:
    def __init__(
        self,
        max_tokens: int = 10_000,
        selection_strategy: SelectionStrategy = "l2_norm",
    ):
        self.max_tokens = max_tokens
        self.selection_strategy = selection_strategy
        self._entries: dict[int, BufferEntry] = {}  # position → entry

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def push_from_result(self, eviction_result, obs_q_vecs_path: str | None = None):
        """
        Populate the buffer from an EvictionResult produced by Phase 3.

        eviction_result.evicted is a PositionTrackedCache on CPU (pinned).
        eviction_result.obs_window_q_vecs has shape [n_layers, obs_len, head_dim].
        We average across obs_len to get a single vector per layer per position.
        """
        evicted = eviction_result.evicted
        positions = evicted.positions
        kv_cpu = evicted.kv
        scores = eviction_result.importance_scores

        # Average obs window Q vectors across obs_len dimension
        # q_vecs shape: [n_layers, obs_len, head_dim] → [n_layers, head_dim]
        q_vecs_mean = eviction_result.obs_window_q_vecs.mean(dim=1)  # [n_layers, head_dim]

        # Per-token KV pairs: slice each layer's KV for each token position
        n_layers = len(kv_cpu)
        for token_idx, pos in enumerate(positions):
            per_layer_kv = tuple(
                (
                    k[:, :, token_idx:token_idx+1, :].contiguous().cpu().pin_memory(),
                    v[:, :, token_idx:token_idx+1, :].contiguous().cpu().pin_memory(),
                )
                for k, v in kv_cpu
            )
            entry = BufferEntry(
                position=pos,
                kv=per_layer_kv,
                importance_score=scores.get(pos, 0.0),
                q_vec=q_vecs_mean,  # same observation context for all tokens in this eviction
            )
            self._entries[pos] = entry

            # Evict from buffer if over capacity (drop lowest importance score)
            if len(self._entries) > self.max_tokens:
                min_pos = min(self._entries, key=lambda p: self._entries[p].importance_score)
                del self._entries[min_pos]

    def push(self, entry: BufferEntry):
        """Add a single entry directly."""
        self._entries[entry.position] = entry
        if len(self._entries) > self.max_tokens:
            min_pos = min(self._entries, key=lambda p: self._entries[p].importance_score)
            del self._entries[min_pos]

    def clear(self):
        self._entries.clear()

    def __len__(self):
        return len(self._entries)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def query(
        self,
        recent_q_vecs: torch.Tensor,   # [n_layers, M, head_dim] — last M tokens' K vectors
        top_k: int,
    ) -> list[BufferEntry]:
        """
        Score all buffer entries against recent_q_vecs and return the top_k
        most relevant entries.

        recent_q_vecs should be the key vectors from the last M tokens of the
        active (compressed) KV cache — a proxy for the model's current query
        state. Extracted from PositionTrackedCache before calling query().
        """
        if len(self._entries) == 0:
            return []

        top_k = min(top_k, len(self._entries))
        entries = list(self._entries.values())

        if self.selection_strategy == "l2_norm":
            scores = self._score_l2_norm(entries)
        elif self.selection_strategy == "dot_product":
            scores = self._score_dot_product(entries, recent_q_vecs)
        elif self.selection_strategy == "random":
            scores = self._score_random(entries)
        elif self.selection_strategy == "recency_inverse":
            scores = self._score_recency_inverse(entries)
        else:
            raise ValueError(f"Unknown strategy: {self.selection_strategy}")

        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [entries[i] for i in top_indices]

    def _score_l2_norm(self, entries: list[BufferEntry]) -> np.ndarray:
        """
        Score by L2 norm of the stored Q vectors.
        High norm = was strongly queried at eviction time.
        Fully CPU-side — no GPU operations.
        """
        scores = np.array([
            entry.q_vec.norm(p=2).item()
            for entry in entries
        ])
        return scores

    def _score_dot_product(
        self,
        entries: list[BufferEntry],
        recent_q_vecs: torch.Tensor,   # [n_layers, M, head_dim]
    ) -> np.ndarray:
        """
        Score by dot product between stored Q vectors and current Q vectors.
        Measures how similar each evicted token's "query context" is to
        the current generation context.

        Computation stays on CPU using torch — avoids GPU round-trips.
        """
        # recent_q_vecs averaged across M and layers → [head_dim]
        query_mean = recent_q_vecs.mean(dim=[0, 1])  # [head_dim]

        scores = np.array([
            torch.dot(entry.q_vec.mean(dim=0), query_mean).item()
            for entry in entries
        ])
        return scores

    def _score_random(self, entries: list[BufferEntry]) -> np.ndarray:
        """Ablation baseline — random selection, no information used."""
        return np.random.rand(len(entries))

    def _score_recency_inverse(self, entries: list[BufferEntry]) -> np.ndarray:
        """
        Prefer oldest evicted tokens (lowest position index).
        Targets attention sink victims — early context tokens most likely
        to be in the eviction dead zone.
        """
        positions = np.array([entry.position for entry in entries])
        # Invert: lower position → higher score
        scores = 1.0 / (positions + 1)
        return scores

    # ------------------------------------------------------------------
    # GPU restoration
    # ------------------------------------------------------------------

    def to_gpu(
        self,
        entries: list[BufferEntry],
        device: str = "cuda",
    ) -> "PositionTrackedCache":
        """
        Move selected entries' KV pairs to GPU and return as a PositionTrackedCache.

        Transfers layer-by-layer to allow Python GC to free CPU tensors
        incrementally rather than holding the full buffer in both CPU and GPU RAM.
        """
        if not entries:
            return PositionTrackedCache(kv=tuple(), positions=[])

        positions = [e.position for e in entries]
        n_layers = len(entries[0].kv)

        merged_layers = []
        for layer_idx in range(n_layers):
            keys = torch.cat(
                [e.kv[layer_idx][0] for e in entries], dim=2
            ).to(device, non_blocking=True)
            vals = torch.cat(
                [e.kv[layer_idx][1] for e in entries], dim=2
            ).to(device, non_blocking=True)
            merged_layers.append((keys, vals))

        torch.cuda.synchronize()
        return PositionTrackedCache(kv=tuple(merged_layers), positions=positions)
```

---

## Extracting Recent Q Vectors from the Active Cache

The `query()` method needs `recent_q_vecs` — the key vectors from the last M tokens of the current active (compressed) KV cache. These serve as the proxy for "what the model is currently thinking about" and are used to score which evicted tokens are newly relevant.

```python
def extract_recent_q_vecs(
    active_cache: PositionTrackedCache,
    m: int = 64,                    # number of recent tokens to use
) -> torch.Tensor:
    """
    Extract the last M key vectors from the active cache as a proxy for
    current query vectors. Returns shape [n_layers, M, head_dim] on CPU.
    """
    vecs = []
    for k, v in active_cache.kv:
        # k: [1, n_kv_heads, seq_len, head_dim]
        seq_len = k.shape[2]
        start = max(0, seq_len - m)
        recent_k = k[0, :, start:, :].mean(dim=0)  # [actual_m, head_dim]
        vecs.append(recent_k.cpu())
    return torch.stack(vecs, dim=0)  # [n_layers, actual_m, head_dim]
```

Call this immediately before `buffer.query()`. The cost is negligible — it is just indexing into tensors already on GPU and moving a small slice to CPU.

---

## Profiling Suite

This is the critical output of Phase 4. Run all four operations, report p50/p90/p99 across 50 trials each, and save results to disk. The numbers determine the feasibility frontier for P6.

> Use `torch.cuda.synchronize()` before every GPU timing operation. Use `time.perf_counter()` for wall-clock measurement, not `time.time()`.

### Operation 1: CPU → GPU Transfer

```python
def profile_cpu_to_gpu_transfer(
    buffer: EvictionBuffer,
    n_tokens_list: list[int] = [50, 100, 250, 500, 1000, 2000],
    n_trials: int = 50,
    device: str = "cuda",
) -> dict:
    """
    Profile the time to transfer N evicted token KV pairs from CPU to GPU.
    This is the dominant cost for small K values.
    """
    results = {}

    for n_tokens in n_tokens_list:
        # Get N entries from the buffer (or use synthetic if buffer is smaller)
        entries = list(buffer._entries.values())[:n_tokens]
        if len(entries) < n_tokens:
            print(f"Warning: buffer has only {len(entries)} entries, requested {n_tokens}")
            continue

        times = []
        for _ in range(n_trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # This is exactly what to_gpu() does
            n_layers = len(entries[0].kv)
            for layer_idx in range(n_layers):
                keys = torch.cat([e.kv[layer_idx][0] for e in entries], dim=2)
                vals = torch.cat([e.kv[layer_idx][1] for e in entries], dim=2)
                _ = keys.to(device, non_blocking=True)
                _ = vals.to(device, non_blocking=True)

            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        results[n_tokens] = {
            "p50_ms": np.percentile(times, 50) * 1000,
            "p90_ms": np.percentile(times, 90) * 1000,
            "p99_ms": np.percentile(times, 99) * 1000,
            "mean_ms": np.mean(times) * 1000,
        }
        print(f"Transfer {n_tokens} tokens: p50={results[n_tokens]['p50_ms']:.1f}ms "
              f"p90={results[n_tokens]['p90_ms']:.1f}ms")

    return results
```

### Operation 2: CPU Scoring (Buffer Query)

```python
def profile_buffer_scoring(
    buffer: EvictionBuffer,
    recent_q_vecs: torch.Tensor,   # [n_layers, M, head_dim]
    top_k: int = 250,
    strategies: list[SelectionStrategy] = ["l2_norm", "dot_product", "random", "recency_inverse"],
    buffer_sizes: list[int] = [500, 1000, 2000, 5000],
    n_trials: int = 50,
) -> dict:
    """
    Profile time to score N buffer entries and select top_k.
    This runs entirely on CPU — measures Python + torch CPU overhead.
    """
    results = {}

    for strategy in strategies:
        buffer.selection_strategy = strategy
        results[strategy] = {}

        for n_entries in buffer_sizes:
            # Subsample or use full buffer
            entries = list(buffer._entries.values())[:n_entries]

            times = []
            for _ in range(n_trials):
                t0 = time.perf_counter()
                if strategy == "dot_product":
                    scores = buffer._score_dot_product(entries, recent_q_vecs)
                elif strategy == "l2_norm":
                    scores = buffer._score_l2_norm(entries)
                elif strategy == "random":
                    scores = buffer._score_random(entries)
                elif strategy == "recency_inverse":
                    scores = buffer._score_recency_inverse(entries)

                # Top-k selection
                top_k_actual = min(top_k, len(entries))
                _ = np.argpartition(scores, -top_k_actual)[-top_k_actual:]
                times.append(time.perf_counter() - t0)

            results[strategy][n_entries] = {
                "p50_ms": np.percentile(times, 50) * 1000,
                "p90_ms": np.percentile(times, 90) * 1000,
            }
            print(f"Score {n_entries} entries ({strategy}): "
                  f"p50={results[strategy][n_entries]['p50_ms']:.2f}ms")

    return results
```

### Operation 3: Attention Recompute Cost (Post-Injection)

After injecting K tokens back into the active cache, the next forward pass must attend over `k_budget + K` tokens instead of `k_budget` tokens. This additional attention cost is borne by the model, not by the repair pipeline — but it adds latency to the generation step that follows the tool call.

Profile by running a short generation step with varying cache sizes:

```python
def profile_injection_attention_overhead(
    model,
    base_cache: PositionTrackedCache,        # k_budget tokens — the compressed cache
    extra_token_counts: list[int] = [50, 100, 250, 500, 1000],
    query_len: int = 20,                     # typical query length
    n_trials: int = 20,
) -> dict:
    """
    Profile the additional generation latency from having K extra tokens
    in the KV cache due to repair injection.

    Measures: time to run one forward pass of query_len tokens against
    a cache of size (base_size + K) vs base_size. Reports the delta.
    """
    import time

    base_size = len(base_cache.positions)
    query_ids = torch.ones(1, query_len, dtype=torch.long, device="cuda")
    results = {}

    # Baseline: generate against base cache only
    baseline_times = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(query_ids, past_key_values=base_cache.kv, use_cache=False)
        torch.cuda.synchronize()
        baseline_times.append(time.perf_counter() - t0)
    baseline_p50 = np.percentile(baseline_times, 50) * 1000

    for k_extra in extra_token_counts:
        # Extend cache by k_extra synthetic tokens (same shape as real entries)
        n_layers = len(base_cache.kv)
        extended_kv = tuple(
            (
                torch.cat([k, torch.zeros(1, k.shape[1], k_extra, k.shape[3], device="cuda")], dim=2),
                torch.cat([v, torch.zeros(1, v.shape[1], k_extra, v.shape[3], device="cuda")], dim=2),
            )
            for k, v in base_cache.kv
        )

        times = []
        for _ in range(n_trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(query_ids, past_key_values=extended_kv, use_cache=False)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        p50_ms = np.percentile(times, 50) * 1000
        results[k_extra] = {
            "total_p50_ms": p50_ms,
            "overhead_vs_base_ms": p50_ms - baseline_p50,
            "overhead_pct": (p50_ms - baseline_p50) / baseline_p50 * 100,
        }
        print(f"K={k_extra} extra tokens: total={p50_ms:.1f}ms "
              f"overhead={results[k_extra]['overhead_vs_base_ms']:.1f}ms "
              f"({results[k_extra]['overhead_pct']:.1f}%)")

    return {"baseline_p50_ms": baseline_p50, "by_k": results}
```

### Operation 4: End-to-End Repair Pipeline

Profile the full sequence as it will actually run during a tool call:

```python
def profile_end_to_end_repair(
    buffer: EvictionBuffer,
    active_cache: PositionTrackedCache,
    top_k_values: list[int] = [50, 100, 250, 500],
    n_trials: int = 20,
) -> dict:
    """
    Profile the full repair pipeline:
    1. Extract recent Q vectors from active cache
    2. Score buffer entries
    3. Transfer top-K to GPU
    4. Inject into active cache

    This is the actual wall-clock budget consumed during the idle window.
    """
    results = {}

    for top_k in top_k_values:
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()

            # Step 1: extract recent Q vectors
            recent_q = extract_recent_q_vecs(active_cache, m=64)

            # Step 2: score and select
            selected = buffer.query(recent_q, top_k=top_k)

            # Step 3: transfer to GPU
            restored = buffer.to_gpu(selected, device="cuda")

            # Step 4: inject (merge + sort by position)
            repaired = inject_kv_with_positions(
                active_cache=active_cache,
                active_positions=active_cache.positions,
                new_pairs=restored,
                new_positions=restored.positions,
            )

            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        results[top_k] = {
            "p50_ms": np.percentile(times, 50) * 1000,
            "p90_ms": np.percentile(times, 90) * 1000,
            "p99_ms": np.percentile(times, 99) * 1000,
        }
        print(f"End-to-end repair K={top_k}: "
              f"p50={results[top_k]['p50_ms']:.1f}ms "
              f"p90={results[top_k]['p90_ms']:.1f}ms")

    return results
```

---

## The Feasibility Frontier

After running all four profiling operations, compute the feasibility frontier table. This is what P6 uses to determine the repair budget K for each tool call duration.

```python
def compute_feasibility_frontier(
    transfer_results: dict,
    scoring_results: dict,
    strategy: str = "l2_norm",
    overhead_budget_pct: float = 0.90,   # use 90% of idle window for repair
) -> dict:
    """
    For each tool call duration T (seconds), compute the maximum K such that
    the full repair pipeline (score + transfer) fits within T * overhead_budget_pct.

    Returns: {tool_call_duration_s: max_K}
    """
    tool_call_durations = [0.1, 0.5, 1.0, 2.0, 5.0, 8.0, 15.0, 30.0, 60.0]

    frontier = {}
    for T in tool_call_durations:
        budget_ms = T * 1000 * overhead_budget_pct

        # Binary search for max K such that transfer cost ≤ budget
        # (scoring cost is small relative to transfer for all strategies)
        # Use scoring p50 for strategy at max buffer size as a constant overhead
        scoring_overhead_ms = max(
            scoring_results.get(strategy, {}).get(max(scoring_results.get(strategy, {}).keys(), default=1000), {}).get("p50_ms", 5.0),
            1.0
        )
        available_for_transfer_ms = budget_ms - scoring_overhead_ms

        max_k = 0
        for n_tokens, timing in sorted(transfer_results.items()):
            if timing["p90_ms"] <= available_for_transfer_ms:
                max_k = n_tokens

        frontier[T] = {
            "budget_ms": budget_ms,
            "scoring_overhead_ms": scoring_overhead_ms,
            "max_K": max_k,
        }

    return frontier
```

**Print the frontier table clearly:**

| Tool call duration | Repair budget | Max K (tokens restored)        |
|--------------------|---------------|---------------------------------|
| 0.1s (cat/ls)      | 90ms          | ~0 (transfer only, no scoring) |
| 0.5s (grep)        | 450ms         | ~[measure]                      |
| 1.0s               | 900ms         | ~[measure]                      |
| 2.0s (git)         | 1.8s          | ~[measure]                      |
| 5.0s               | 4.5s          | ~[measure]                      |
| 8.0s               | 7.2s          | ~[measure]                      |
| 15s (pytest)       | 13.5s         | ~[measure]                      |
| 30s                | 27s           | ~[measure]                      |
| 60s (pip)          | 54s           | oracle feasible                 |

This table goes directly into the paper. The shape of it — zero benefit at sub-0.5s tool calls, significant benefit at pytest/pip durations — contextualizes the repair algorithm within the real SWE-bench tool call distribution discussed earlier.

---

## Scoring Strategy Comparison

Run the buffer scoring profiler across all four strategies at `buffer_size=1000` and `top_k=250`. Report latency and also run a correctness comparison: for 50 VT-4hop examples at `k_budget=512`, score the evicted buffer with each strategy and check what fraction of the time the strategy correctly ranks a surviving-if-restored hop link in the top-K.

```python
def evaluate_selection_quality(
    eviction_logs_dir: str,  # Phase 3 logs for VT-4hop at k_budget=512
    buffer: EvictionBuffer,
    top_k: int = 250,
    strategies: list = ["l2_norm", "dot_product", "random", "recency_inverse"],
) -> dict:
    """
    For each strategy, compute what fraction of the time the strategy
    correctly selects a broken hop link (if that link is in the buffer)
    into the top-K.

    This is the selection precision for the repair task — a direct proxy
    for whether the strategy will help VT-4hop accuracy in P6.
    """
    results = {s: {"correct_selections": 0, "total_reparable": 0} for s in strategies}

    for log_file in os.listdir(eviction_logs_dir):
        if not log_file.endswith(".json"):
            continue
        with open(os.path.join(eviction_logs_dir, log_file)) as f:
            log = json.load(f)

        if log.get("correct", True):
            continue  # skip correct examples

        # Find broken hop links that are in the buffer
        broken_positions = [
            pos for pos, survived in zip(log["task_relevant_positions"],
                                          log["task_relevant_survived"])
            if not survived and pos in buffer._entries
        ]

        if not broken_positions:
            continue  # broken link not in buffer — can't evaluate

        # Load Q vectors for this example
        q_vec_path = os.path.join(
            eviction_logs_dir,
            log_file.replace(".json", "_qvecs.pt")
        )
        recent_q = torch.load(q_vec_path).unsqueeze(1)  # [n_layers, 1, head_dim]

        for strategy in strategies:
            buffer.selection_strategy = strategy
            selected = buffer.query(recent_q, top_k=top_k)
            selected_positions = {e.position for e in selected}

            if any(p in selected_positions for p in broken_positions):
                results[strategy]["correct_selections"] += 1
            results[strategy]["total_reparable"] += 1

    for strategy in strategies:
        total = results[strategy]["total_reparable"]
        if total > 0:
            prec = results[strategy]["correct_selections"] / total
            results[strategy]["selection_precision"] = prec
            print(f"{strategy}: precision={prec:.3f} ({results[strategy]['correct_selections']}/{total})")

    return results
```

**Expected ordering:** `dot_product` or `l2_norm` should outperform `random` and `recency_inverse` on selection precision. If random beats l2_norm, the Q vector proxy is not informative for this task — investigate whether the Q vectors are being computed correctly in Phase 3's `_extract_obs_q_vecs`.

---

## Connecting to Phase 3 Logs

Phase 4's buffer should be populated from the Phase 3 eviction logs, not from a fresh run. This ensures the profiling reflects the actual distribution of evicted tokens from your target tasks.

```python
def build_buffer_from_logs(
    log_dir: str,
    strategy: SelectionStrategy = "l2_norm",
    max_tokens: int = 10_000,
) -> EvictionBuffer:
    """
    Reconstruct an EvictionBuffer from Phase 3 eviction logs.
    Used for profiling — gives realistic token distributions.
    """
    buffer = EvictionBuffer(max_tokens=max_tokens, selection_strategy=strategy)

    for log_file in sorted(os.listdir(log_dir)):
        if not log_file.endswith(".json") or "_qvecs" in log_file:
            continue

        with open(os.path.join(log_dir, log_file)) as f:
            log = json.load(f)

        q_vec_path = os.path.join(log_dir, log_file.replace(".json", "_qvecs.pt"))
        if not os.path.exists(q_vec_path):
            continue
        q_vecs = torch.load(q_vec_path)  # [n_layers, obs_len, head_dim]
        q_vec_mean = q_vecs.mean(dim=1)  # [n_layers, head_dim]

        # We don't have the actual KV tensors from the log (too large to store in JSON).
        # For profiling purposes, use synthetic tensors of the correct shape.
        # For correctness evaluation, re-run eviction from Phase 3 and capture results.
        for pos in log["evicted_positions"]:
            score = float(log["importance_scores"].get(str(pos), 0.0))
            synthetic_kv = tuple(
                (
                    torch.zeros(1, 32, 1, 128, dtype=torch.float16).pin_memory(),
                    torch.zeros(1, 32, 1, 128, dtype=torch.float16).pin_memory(),
                )
                for _ in range(32)  # n_layers
            )
            entry = BufferEntry(
                position=pos,
                kv=synthetic_kv,
                importance_score=score,
                q_vec=q_vec_mean,
            )
            buffer.push(entry)

    print(f"Built buffer with {len(buffer)} entries from {log_dir}")
    return buffer
```

> **Note on synthetic KV for profiling:** The profiling operations (transfer latency, scoring latency) only need correctly shaped tensors — the values do not matter. For the selection quality evaluation in `evaluate_selection_quality()`, you need real KV tensors. Run a fresh Phase 3 pass on 50 VT-4hop examples and capture `EvictionResult` objects directly instead of reconstructing from logs.

---

## Validation Checks

### Check 1: Transfer Latency Is Monotone

CPU→GPU transfer time must increase monotonically with N tokens. If it does not, you have a caching or batching artifact. The latency curve should be roughly linear with N — transfer time is bandwidth-limited, so doubling N should approximately double transfer time.

### Check 2: Scoring Latency Is Negligible vs Transfer

For all strategies at `buffer_size=1000`, scoring p50 should be < 10ms. If `dot_product` scoring takes > 50ms at 1000 entries, consider a vectorized implementation:

```python
# Vectorized dot_product scoring (replace the loop in _score_dot_product)
query_vec = recent_q_vecs.mean(dim=[0, 1])  # [head_dim]
stored_vecs = torch.stack([e.q_vec.mean(dim=0) for e in entries], dim=0)  # [N, head_dim]
scores = torch.mv(stored_vecs, query_vec).numpy()
```

### Check 3: End-to-End Budget at 2 Seconds

At T=2s (git tool call, representative mid-range), the end-to-end repair pipeline at `top_k=100` must complete in < 1.8 seconds p90. If it does not, K=100 is not feasible at 2-second tool calls and the P6 ablation must reflect this.

### Check 4: Selection Precision Above Random

L2-norm and dot-product selection precision must both exceed random (expected `~top_k/buffer_size ≈ 0.25` for `top_k=250` from buffer of 1000). If they do not, the Q vector proxy is uninformative. Possible causes: Q vectors are being averaged over too many layers (losing specificity), or the observation window in Phase 3 is too small to capture enough signal.

---

## File Structure

```
src/
  buffer/
    __init__.py
    eviction_buffer.py      # EvictionBuffer, BufferEntry, extract_recent_q_vecs
    profiling.py            # all four profile_* functions
    feasibility.py          # compute_feasibility_frontier

results/
  phase4_profiling/
    transfer_latency.json       # operation 1 results
    scoring_latency.json        # operation 2 results by strategy
    attention_overhead.json     # operation 3 results
    end_to_end_repair.json      # operation 4 results
    feasibility_frontier.json   # derived table: tool_call_duration → max_K
    selection_quality.json      # precision by strategy on VT-4hop failures
    figures/
      transfer_latency_curve.png
      feasibility_frontier.png  # the paper figure
      selection_precision_bar.png
```

---

## Deliverable

`src/buffer/` with `EvictionBuffer`, all four selection strategies, and `extract_recent_q_vecs`. All four profiling operations run and saved to `results/phase4_profiling/`. Feasibility frontier table computed and plotted. Selection quality evaluation showing L2-norm and dot-product outperforming random on VT-4hop failures.

> **Before moving to P5:** review the feasibility frontier table. If `max_K` at 2 seconds is below 50, the repair algorithm has very limited scope at common tool call durations — you may want to profile whether a FAISS-based CPU index for dot-product scoring can reduce scoring overhead and recover budget. If `max_K` at 15 seconds (pytest) is above 1000, the oracle experiment in P5 is straightforwardly runnable within the idle window without any approximation.