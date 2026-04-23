# Phase 2: KV Cache Access Layer

**Goal:** Build a clean, tested API for reading, writing, slicing, and modifying KV cache tensors. Every phase from P3 onward calls into this API. Bugs here propagate silently and produce mysterious score fluctuations — treat this like library code, not a script.

The acceptance gate for this phase is a single test: round-trip identity. Save the KV cache after context prefill, reload it, and continue generation. The output must be numerically identical to a run that never saved or loaded anything. Until this test passes, nothing downstream is valid.

---

## Before You Write Any Code: Understand the KV Cache Structure

### What `past_key_values` Actually Is

When you call `model(input_ids, use_cache=True)`, HuggingFace returns an object where `outputs.past_key_values` is a Python tuple of length `n_layers`. Each element of that tuple is itself a tuple of two tensors: `(key_states, value_states)`.

For **Qwen2.5-7B-Instruct**, inspect the model config before assuming anything:

```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
print(config.num_hidden_layers)          # number of layers → length of past_key_values tuple
print(config.num_attention_heads)        # query heads
print(config.num_key_value_heads)        # KV heads (less than query heads with GQA)
print(config.hidden_size // config.num_attention_heads)  # head_dim
```

Qwen2.5-7B uses **Grouped Query Attention (GQA)**. This means there are fewer KV heads than query heads — multiple query heads share a single KV head. The key consequence: `key_states` and `value_states` have shape:

```
[batch_size, num_key_value_heads, seq_len, head_dim]
```

NOT `num_attention_heads`. If you assume the KV tensors have the same head dimension as the query tensors, your slice operations will be wrong.

Print the shape of an actual KV tensor from a short test run before writing any of the API functions:

```python
test_input = tokenizer("Hello world", return_tensors="pt").input_ids.to("cuda")
with torch.no_grad():
    out = model(test_input, use_cache=True)
kv = out.past_key_values
print(f"n_layers: {len(kv)}")
print(f"layer 0 key shape: {kv[0][0].shape}")   # [batch, n_kv_heads, seq_len, head_dim]
print(f"layer 0 val shape: {kv[0][1].shape}")
```

Write down these shapes. They are the ground truth for all your tensor operations.

### Memory Footprint

At 32K tokens (fp16):

```
n_layers × 2 × n_kv_heads × seq_len × head_dim × 2 bytes
```

Compute this with your actual config values. For a rough estimate: it will be somewhere between 8GB and 20GB. This will not fit in GPU VRAM alongside the model weights. CPU offloading is required for any full-cache operation.

Check free GPU memory before and after a 32K prefill to confirm you understand where the cache lives:

```python
torch.cuda.reset_peak_memory_stats()
# run prefill
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak GPU memory: {peak:.1f} GB")
```

### The `DynamicCache` vs Tuple Format

Recent versions of `transformers` (≥ 4.38) may return `past_key_values` as a `DynamicCache` object rather than a raw tuple, depending on model version. Check what you actually get:

```python
print(type(out.past_key_values))
```

If it is a `DynamicCache`, it has `.key_cache` and `.value_cache` attributes (lists of tensors per layer). Convert it to a plain tuple of tuples for your API — all downstream code assumes the tuple format and working with a consistent representation is worth the one-time conversion cost:

```python
def to_tuple_cache(past_key_values) -> tuple:
    if isinstance(past_key_values, tuple):
        return past_key_values
    # DynamicCache format
    return tuple(
        (past_key_values.key_cache[i], past_key_values.value_cache[i])
        for i in range(len(past_key_values.key_cache))
    )
```

Add this conversion at the top of every API function so callers do not have to think about it.

---

## The Five-Function API

File: `src/kv_utils.py`

All functions operate on the tuple-of-tuples format described above. All functions are pure (no side effects outside the returned value) except `save_kv` and `load_kv` which touch disk.

### Function 1: `save_kv`

```python
def save_kv(past_key_values: tuple, path: str) -> None:
    """
    Serialize KV cache to disk.
    
    Saves each layer as a separate file to avoid loading the entire cache
    into CPU RAM at once during large-context operations. Creates a directory
    at `path` containing one file per layer plus a metadata file.
    """
    import os
    os.makedirs(path, exist_ok=True)
    
    past_key_values = to_tuple_cache(past_key_values)
    n_layers = len(past_key_values)
    
    # Move to CPU before saving — avoids GPU memory being held during disk write
    for layer_idx, (k, v) in enumerate(past_key_values):
        torch.save({
            "key": k.cpu(),
            "val": v.cpu()
        }, os.path.join(path, f"layer_{layer_idx:03d}.pt"))
    
    # Metadata: shapes and dtype for verification on load
    meta = {
        "n_layers": n_layers,
        "key_shape": list(past_key_values[0][0].shape),
        "val_shape": list(past_key_values[0][1].shape),
        "dtype": str(past_key_values[0][0].dtype),
    }
    torch.save(meta, os.path.join(path, "meta.pt"))
```

**Implementation notes:**
- Save layer-by-layer, not as one giant tensor. A 32K cache at 16GB cannot be `torch.stack`-ed into a single tensor and saved without OOM on most systems.
- Always move to CPU before saving. If you save GPU tensors, `torch.load` without a `map_location` argument will try to restore them to the original GPU device, which may fail if the device index changed between runs.
- The metadata file is used by `load_kv` to verify the cache shape matches expectations before attempting to use it.

### Function 2: `load_kv`

```python
def load_kv(path: str, device: str = "cuda") -> tuple:
    """
    Load KV cache from disk and move to specified device.
    
    Verifies metadata before loading. Returns tuple-of-tuples format.
    """
    import os
    
    meta = torch.load(os.path.join(path, "meta.pt"))
    n_layers = meta["n_layers"]
    
    layers = []
    for layer_idx in range(n_layers):
        data = torch.load(
            os.path.join(path, f"layer_{layer_idx:03d}.pt"),
            map_location="cpu"  # always load to CPU first
        )
        k = data["key"].to(device)
        v = data["val"].to(device)
        layers.append((k, v))
    
    return tuple(layers)
```

**Implementation notes:**
- Always load to CPU first, then move to device. Loading directly to GPU with `map_location="cuda"` can cause fragmentation issues with large tensors.
- The `to(device)` call on each layer individually allows Python's garbage collector to free CPU memory incrementally rather than holding the entire cache in both CPU and GPU RAM simultaneously during the transfer.

### Function 3: `slice_kv`

```python
def slice_kv(past_key_values: tuple, token_indices) -> tuple:
    """
    Extract a subset of sequence positions from the KV cache.
    
    `token_indices` can be a list of ints (arbitrary positions) or a slice.
    Returns a new KV cache containing only the specified positions,
    preserving their original order.
    
    This is the core operation for eviction: call with the indices of tokens
    to KEEP, not tokens to evict.
    """
    past_key_values = to_tuple_cache(past_key_values)
    
    if isinstance(token_indices, list):
        # Sort to preserve causal order — attention is position-sensitive
        token_indices = sorted(token_indices)
        idx_tensor = torch.tensor(token_indices, dtype=torch.long,
                                  device=past_key_values[0][0].device)
        return tuple(
            (k[:, :, idx_tensor, :], v[:, :, idx_tensor, :])
            for k, v in past_key_values
        )
    else:
        # Slice object — faster path for contiguous ranges
        return tuple(
            (k[:, :, token_indices, :], v[:, :, token_indices, :])
            for k, v in past_key_values
        )
```

**Implementation notes:**
- The sequence dimension is dim=2 (index 2 of the 4D tensor `[batch, heads, seq, head_dim]`). This is easy to get wrong — double-check against your printed shapes from the inspection step.
- Sort the indices. Attention is causally ordered — if you pass indices out of order, the positional encoding will be misaligned and you will get garbage outputs without an obvious error message.
- For large index lists, `torch.index_select` is faster than fancy indexing with a tensor. Profile both if slicing becomes a bottleneck.

### Function 4: `merge_kv`

```python
def merge_kv(cache_a: tuple, cache_b: tuple) -> tuple:
    """
    Concatenate two KV caches along the sequence dimension.
    
    cache_a tokens will appear before cache_b tokens in the merged cache.
    Both caches must have the same batch size, number of heads, and head dim.
    The sequence lengths can differ.
    """
    cache_a = to_tuple_cache(cache_a)
    cache_b = to_tuple_cache(cache_b)
    
    assert len(cache_a) == len(cache_b), \
        f"Layer count mismatch: {len(cache_a)} vs {len(cache_b)}"
    
    return tuple(
        (
            torch.cat([ka, kb], dim=2),
            torch.cat([va, vb], dim=2)
        )
        for (ka, va), (kb, vb) in zip(cache_a, cache_b)
    )
```

**Implementation notes:**
- Order matters. `merge_kv(a, b)` puts `a` tokens before `b` tokens. For the repair use case, you will call `merge_kv(repaired_fragment, active_cache_tail)` or similar — think carefully about ordering before calling.
- Both caches must be on the same device. Add a device check assertion if you are frequently moving caches between CPU and GPU.

### Function 5: `inject_kv`

```python
def inject_kv(
    active_cache: tuple,
    new_pairs: tuple,
    positions: list[int]
) -> tuple:
    """
    Insert repaired KV pairs into the active cache at their original positions.
    
    `active_cache` is the current (eviction-compressed) cache, with seq_len = k_budget.
    `new_pairs` is a cache fragment containing the tokens to restore, with
    seq_len = len(positions).
    `positions` are the original absolute sequence positions of the tokens in
    `new_pairs`, before eviction. These are used to determine where to insert
    them in the merged result.
    
    The returned cache is sorted by original sequence position, so the model
    sees tokens in causal order.
    """
    active_cache = to_tuple_cache(active_cache)
    new_pairs = to_tuple_cache(new_pairs)
    
    # Build the merged sequence: active cache positions + restored positions
    # We need the original positions of the active cache tokens to merge correctly.
    # These must be tracked externally — inject_kv cannot infer them.
    # Caller is responsible for passing position metadata alongside the cache.
    
    # Simple implementation: concatenate and sort by position
    # Requires caller to also pass active_cache_positions: list[int]
    # See inject_kv_with_positions() below for the full signature.
    raise NotImplementedError(
        "Use inject_kv_with_positions() which requires explicit position tracking."
    )


def inject_kv_with_positions(
    active_cache: tuple,
    active_positions: list[int],
    new_pairs: tuple,
    new_positions: list[int]
) -> tuple:
    """
    Full signature: requires knowing the original positions of BOTH
    the active cache tokens and the new tokens being injected.
    
    Returns a merged cache sorted by original sequence position.
    """
    active_cache = to_tuple_cache(active_cache)
    new_pairs = to_tuple_cache(new_pairs)
    
    # Merge position lists and sort
    all_positions = active_positions + new_positions
    sort_order = sorted(range(len(all_positions)), key=lambda i: all_positions[i])
    sort_tensor = torch.tensor(sort_order, dtype=torch.long,
                               device=active_cache[0][0].device)
    
    merged = merge_kv(active_cache, new_pairs)
    
    return tuple(
        (k[:, :, sort_tensor, :], v[:, :, sort_tensor, :])
        for k, v in merged
    )
```

**Why `inject_kv` requires position tracking:** After eviction, the active cache contains `k_budget` tokens but does not remember which original positions they came from — the eviction algorithm knows, but that information is in the eviction log, not in the KV tensors themselves. Your eviction implementation (P3) must store a `kept_positions: list[int]` alongside every compressed cache it produces. Pass this to `inject_kv_with_positions`. If you do not track positions, you cannot merge correctly — the tokens will be in wrong causal order and attention will be misaligned.

Add a `PositionTrackedCache` dataclass to make this explicit:

```python
from dataclasses import dataclass

@dataclass
class PositionTrackedCache:
    kv: tuple               # the actual KV cache tuple
    positions: list[int]    # original absolute positions of each token in kv
    
    def __len__(self):
        return len(self.positions)
    
    def to_device(self, device: str) -> "PositionTrackedCache":
        return PositionTrackedCache(
            kv=tuple((k.to(device), v.to(device)) for k, v in self.kv),
            positions=self.positions
        )
```

Use `PositionTrackedCache` everywhere a KV cache is stored after eviction. The plain tuple format is only valid for the full uncompressed cache (condition A) or immediately after a fresh prefill before any eviction.

---

## Required Tests

Run all three tests before proceeding to P3. They must all pass. Do not skip the latency test — the numbers it produces are the feasibility ceiling for P4.

### Test 1: Round-Trip Identity

This is the acceptance gate. Fail here → nothing downstream is valid.

```python
def test_round_trip_identity():
    """
    Save KV cache after prefill, reload it, continue generation.
    Output must match a run with no save/load.
    """
    test_text = "The quick brown fox jumps over the lazy dog. " * 200  # ~1K tokens
    inputs = tokenizer(test_text, return_tensors="pt").to("cuda")
    
    # Reference run: single continuous generation
    with torch.no_grad():
        ref_out = model(**inputs, use_cache=True)
        ref_kv = ref_out.past_key_values
        ref_next_logits = model(
            torch.tensor([[tokenizer.eos_token_id]], device="cuda"),
            past_key_values=ref_kv
        ).logits
    
    # Save/load run
    save_kv(ref_kv, "/tmp/test_kv_cache")
    loaded_kv = load_kv("/tmp/test_kv_cache", device="cuda")
    
    with torch.no_grad():
        loaded_next_logits = model(
            torch.tensor([[tokenizer.eos_token_id]], device="cuda"),
            past_key_values=loaded_kv
        ).logits
    
    # Check: logits must be numerically identical within fp16 tolerance
    max_diff = (ref_next_logits - loaded_next_logits).abs().max().item()
    assert max_diff < 1e-3, f"Round-trip identity failed: max logit diff = {max_diff:.6f}"
    print(f"PASS: round-trip identity (max diff = {max_diff:.2e})")
```

**What to check if this fails:**
- Are you accidentally re-running the forward pass instead of using the loaded cache? Add a print statement inside the model's attention forward to confirm it is not being called during the "loaded" generation step.
- Is the dtype preserved? `torch.save` / `torch.load` preserves dtype, but check that the loaded tensors are still fp16, not promoted to fp32 somewhere.
- Is the device correct? If loaded tensors are on CPU when the model expects CUDA, you will get a device mismatch error (not a silent correctness failure), so this is usually obvious.

### Test 2: Selective Injection Round-Trip

Verifies that `slice_kv` and `inject_kv_with_positions` are semantically correct, not just dimensionally correct.

```python
def test_selective_injection():
    """
    Remove one token from the middle of the cache, generate a wrong answer,
    then inject it back and verify the answer recovers.
    
    Use a simple factual needle: inject a number at a known position,
    remove it, verify model cannot retrieve it, restore it, verify it can.
    """
    # Build a context with a needle at a known position
    prefix = "The answer to the special question is: 99273. " + ("filler text " * 300)
    inputs = tokenizer(prefix, return_tensors="pt").to("cuda")
    query = tokenizer("What is the answer to the special question?",
                      return_tensors="pt").input_ids.to("cuda")
    
    with torch.no_grad():
        ctx_out = model(**inputs, use_cache=True)
        full_kv = to_tuple_cache(ctx_out.past_key_values)
    
    # Find approximately where "99273" appears in the token sequence
    # (inspect tokenizer output to get the exact position)
    needle_tokens = tokenizer("99273", add_special_tokens=False).input_ids
    all_tokens = inputs.input_ids[0].tolist()
    needle_pos = next(i for i in range(len(all_tokens))
                      if all_tokens[i:i+len(needle_tokens)] == needle_tokens)
    needle_positions = list(range(needle_pos, needle_pos + len(needle_tokens)))
    
    # All positions
    all_positions = list(range(full_kv[0][0].shape[2]))
    
    # Evict the needle positions
    keep_positions = [p for p in all_positions if p not in needle_positions]
    evicted_kv = slice_kv(full_kv, keep_positions)
    evicted_fragment = slice_kv(full_kv, needle_positions)
    
    # Generate with needle evicted — should fail to retrieve 99273
    with torch.no_grad():
        wrong_out = model(query, past_key_values=evicted_kv)
    wrong_text = tokenizer.decode(wrong_out.logits[0, -1].argmax().unsqueeze(0))
    
    # Restore the needle
    restored_kv = inject_kv_with_positions(
        active_cache=evicted_kv,
        active_positions=keep_positions,
        new_pairs=evicted_fragment,
        new_positions=needle_positions
    ).kv  # unwrap PositionTrackedCache if needed
    
    # Generate with needle restored — should retrieve 99273
    with torch.no_grad():
        correct_out = model(query, past_key_values=restored_kv)
    correct_text = tokenizer.decode(correct_out.logits[0, -1].argmax().unsqueeze(0))
    
    print(f"Without needle: '{wrong_text}'")
    print(f"With needle restored: '{correct_text}'")
    # The exact assertion depends on tokenization — inspect outputs
    # manually first, then write the specific assertion
```

**Note on this test:** The exact behavior depends on how aggressively the model attends to the needle position. You may need to place the needle in a more prominent position (e.g. immediately before the query) to get a clean pass/fail signal. The goal is not a perfect test — it is a sanity check that `inject_kv_with_positions` does not corrupt the cache structure.

### Test 3: CPU → GPU Transfer Latency

Not a correctness test — a measurement. The numbers it produces determine the maximum feasible repair budget in P4 and P6.

```python
def test_transfer_latency():
    """
    Profile CPU → GPU transfer time for KV caches of different sizes.
    Run 20 trials per size, report median and p90.
    """
    import time
    import numpy as np
    
    # Build caches of different seq lengths by slicing a full 32K cache
    # (Assumes you have already run a 32K prefill and have full_kv available)
    
    test_sizes = [100, 500, 1000, 2000, 5000]
    results = {}
    
    for n_tokens in test_sizes:
        positions = list(range(n_tokens))
        fragment_gpu = slice_kv(full_kv_32k, positions)  # already on GPU
        
        # Move to CPU to simulate the eviction buffer
        fragment_cpu = tuple(
            (k.cpu().pin_memory(), v.cpu().pin_memory())  # pin_memory for faster transfer
            for k, v in fragment_gpu
        )
        
        # Profile GPU → CPU (eviction)
        evict_times = []
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = tuple((k.cpu(), v.cpu()) for k, v in fragment_gpu)
            torch.cuda.synchronize()
            evict_times.append(time.perf_counter() - t0)
        
        # Profile CPU → GPU (repair)
        restore_times = []
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = tuple((k.cuda(), v.cuda()) for k, v in fragment_cpu)
            torch.cuda.synchronize()
            restore_times.append(time.perf_counter() - t0)
        
        results[n_tokens] = {
            "evict_p50_ms": np.median(evict_times) * 1000,
            "evict_p90_ms": np.percentile(evict_times, 90) * 1000,
            "restore_p50_ms": np.median(restore_times) * 1000,
            "restore_p90_ms": np.percentile(restore_times, 90) * 1000,
        }
        print(f"n_tokens={n_tokens}: "
              f"evict p50={results[n_tokens]['evict_p50_ms']:.1f}ms "
              f"restore p50={results[n_tokens]['restore_p50_ms']:.1f}ms")
    
    return results
```

**What to do with these numbers:** The restore p50 for your target N is your per-repair-operation latency floor. Combined with P4's attention recompute timing, this determines the feasible K (tokens repaired) for a given tool call budget. Record the full results table to a file — you will reference it repeatedly in P4 and P6.

**Pinned memory:** The `pin_memory()` call on CPU tensors is important. Pinned (page-locked) memory enables DMA transfers that bypass the CPU, roughly doubling transfer speed on PCIe. Always use pinned memory for tensors in the eviction buffer that will be frequently moved to GPU.

---

## Attention Score Visualization

This is not a test — it is a research artifact that goes into the paper's motivation section. Run it once on a VT-4hop example at 32K context using the condition B pipeline from P1.

```python
def visualize_attention(
    model,
    input_ids: torch.Tensor,
    layer_indices: list[int] = None,   # which layers to visualize; None = all
    head_indices: list[int] = None,    # which heads; None = average across all
    save_path: str = None
):
    """
    Run a forward pass with output_attentions=True and save attention heatmaps.
    
    WARNING: output_attentions=True at 32K context produces attention tensors
    of shape [batch, heads, seq_len, seq_len] per layer. At 32K this is
    32K × 32K = 1 billion elements per layer. Do NOT run this at 32K context
    without restricting to a small number of layers and averaging across heads.
    
    For the paper's motivation figure, use a 4K or 8K context instead.
    The attention sink and recency bias patterns are clearly visible at 4K.
    """
    if input_ids.shape[1] > 8192:
        raise ValueError(
            "Attention visualization at > 8K context requires ~100GB+ memory. "
            "Use a shorter context or implement block-sparse attention extraction."
        )
    
    with torch.no_grad():
        outputs = model(
            input_ids,
            use_cache=False,                # don't need the cache here
            output_attentions=True
        )
    
    attn_weights = outputs.attentions      # tuple of [batch, heads, seq, seq] per layer
    
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    
    layers_to_plot = layer_indices or list(range(len(attn_weights)))
    n_cols = min(4, len(layers_to_plot))
    n_rows = (len(layers_to_plot) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)
    
    for plot_idx, layer_idx in enumerate(layers_to_plot):
        attn = attn_weights[layer_idx][0]  # [heads, seq, seq]
        
        if head_indices is not None:
            attn = attn[head_indices].mean(0)
        else:
            attn = attn.mean(0)            # average across all heads
        
        attn_np = attn.float().cpu().numpy()
        
        im = axes[plot_idx].imshow(
            attn_np,
            aspect="auto",
            cmap="hot",
            norm=mcolors.PowerNorm(gamma=0.3)  # enhance low-value visibility
        )
        axes[plot_idx].set_title(f"Layer {layer_idx}", fontsize=9)
        axes[plot_idx].set_xlabel("Key position")
        axes[plot_idx].set_ylabel("Query position")
        plt.colorbar(im, ax=axes[plot_idx])
    
    for ax in axes[len(layers_to_plot):]:
        ax.set_visible(False)
    
    plt.suptitle("Attention weights — averaged across heads", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
```

**What to look for in the heatmaps:**

The attention sink pattern: columns 0–4 should be bright (high attention weight) in almost every row, for almost every layer. This is the structural artifact that eviction algorithms preserve by design.

The recency bias pattern: the last ~500–1000 columns should be bright in the last ~500 rows (recent queries attending to recent keys). This is the recency window that eviction preserves.

The eviction dead zone: the middle columns (roughly positions 10%–85% of sequence length) should be dim. Needle tokens and hop link tokens that happen to fall here are candidates for eviction — and the ones that do not receive high attention from the observation window will be dropped by SnapKV.

For the VT-4hop task, mark the positions of the four hop links on the x-axis (key positions) of the heatmap. You should see relatively low attention weight on hop links 2 and 3 compared to hop links 1 and 4, which explains why those links are evicted.

Run at 4K context (not 32K) for the visualization. The patterns are identical but the memory cost is 64× lower.

---

## Common Failure Modes

**`past_key_values` shape mismatch after slicing**

Symptom: Model raises a shape error when you pass a sliced cache as `past_key_values`.

Cause: HuggingFace models check that the sequence length in `past_key_values` is consistent with `input_ids` length and `position_ids`. If your slice changes seq_len in a way the model does not expect, it will error.

Fix: Pass `position_ids` explicitly when calling the model with a modified cache. The `position_ids` should correspond to the tokens you are generating, offset by the number of tokens in the cache:

```python
cache_len = active_cache[0][0].shape[2]  # seq dim of the cache
position_ids = torch.arange(cache_len, cache_len + query_len,
                             device="cuda").unsqueeze(0)
model(query_tokens, past_key_values=active_cache, position_ids=position_ids)
```

**Silent numeric drift after save/load**

Symptom: Round-trip test max diff is 1e-4 to 1e-2 — larger than fp16 rounding but not catastrophically wrong.

Cause: Usually a dtype promotion issue. If anything in your save/load path promotes tensors to fp32 (e.g. numpy conversion, certain `torch.cat` operations with mixed dtypes), the values will differ slightly.

Fix: Add dtype assertions at the end of `load_kv`:

```python
expected_dtype = torch.float16  # or bfloat16 — check your model config
for k, v in loaded_cache:
    assert k.dtype == expected_dtype, f"Key dtype mismatch: {k.dtype}"
    assert v.dtype == expected_dtype, f"Val dtype mismatch: {v.dtype}"
```

**OOM during load**

Symptom: `torch.cuda.OutOfMemoryError` during `load_kv`.

Cause: Loading all layers to GPU simultaneously when GPU memory is already partially occupied by the model.

Fix: Load layers one at a time and keep them on CPU until you need them, or use the layer-by-layer approach in the provided implementation. For P4's eviction buffer, you will never load the full cache to GPU at once anyway — only the selected K tokens.

**`inject_kv_with_positions` produces wrong answers despite passing the test**

Symptom: The injection test passes but downstream VT accuracy does not recover under the oracle (P5).

Cause: Position tracking mismatch. The eviction algorithm's `kept_positions` list and the `new_positions` list passed to inject are misaligned by one due to off-by-one indexing.

Fix: Add a consistency check: after injection, verify that the merged cache's sequence length equals `len(active_positions) + len(new_positions)`, and that there are no duplicate positions in the merged sorted list.

---

## File Structure

```
src/
  kv_utils.py           # all five functions + PositionTrackedCache + to_tuple_cache
  tests/
    test_round_trip.py  # Test 1
    test_injection.py   # Test 2
    test_latency.py     # Test 3 (writes results to disk, not a pass/fail test)

results/
  phase2_transfer_latency.json   # output of Test 3
  phase2_attention_heatmaps/
    layer_avg_4k.png
    layer_avg_8k.png
    vt4hop_hop_positions_marked.png
```

---

## Deliverable

`src/kv_utils.py` with all five functions plus `PositionTrackedCache` and `to_tuple_cache`. All three tests passing. Transfer latency table saved to `results/phase2_transfer_latency.json`. Attention heatmaps at 4K and 8K context saved to `results/phase2_attention_heatmaps/`.

The latency table must be reviewed before starting P4. If restore p50 for N=1000 tokens exceeds 1 second, the repair algorithm is constrained to smaller K values than the plan assumes and P4's feasibility frontier needs to be redrawn accordingly.