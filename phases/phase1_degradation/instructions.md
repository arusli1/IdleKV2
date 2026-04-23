# Phase 1: RULER-KVR — Eviction Degradation Measurement

**Goal:** Measure where SnapKV begins to hurt quality on the reduced 32K harness, using budgets that are still large enough to matter for the later repair experiment. This produces the condition B baseline for the repair-vs-no-repair comparison.

---

## What You Are Measuring

**Condition A** is already the P0 baseline: one monolithic pass, no eviction.

**Condition B** is the reduced Phase 1 run here: prefill the 32K context with SnapKV active, let SnapKV compress the live KV cache during prefill, then answer the query from that compressed cache without reprefilling the context.

Do not interpret Phase 1 as a contest to make the model fail as hard as possible. The purpose is to find the **onset-of-regression regime** that later makes condition C informative. If `k_budget` is so small that the cache is obviously unusable, the later repair experiment is poorly posed.

---

## Reduced-Harness Notes

This reduced harness has a few implementation details that the analysis must respect:

- SnapKV's actually protected recent tail in this harness is **64 tokens**.
- Late control spans are pinned into that real protected tail, not just described as "late by percentage."
- `phase1/generators/common.py` supports `tail_inserts` so tail-anchored spans land at the end of the rendered 32K context while still matching the target length.
- `VT-4hop` keeps hops 1/2/3 at fixed depths and pins hop 4 plus the terminal value into the protected tail.
- `MQ-NIAH-4q` keeps needles 1/2/3 at fixed depths and pins needle 4 into the protected tail.
- Span survival is based on the actual per-layer/per-head `kept_mask`; the old indexing bug in `compute_span_survival()` is fixed.
- `eviction_survival_rate` in the summary is computed from the recorded `survival_fraction` values, not a coarse survived/dead flag.

---

## Exact Run Configuration

**Model:** `Qwen/Qwen2.5-7B-Instruct`
**Context length:** 32K tokens only
**Eviction algorithm:** SnapKV only
**Main k_budget sweep:** `16384`, `8192`, `4096`
**Examples per main cell:** 100
**Default main tasks:** `VT-4hop`, `MQ-NIAH-4q`

Why these budgets:

- `k=16384` is the non-pathological sanity budget. If this collapses, the compressed-cache continuation path is not trustworthy.
- `k=8192` is the upper onset budget.
- `k=4096` is the lower onset budget and should show clear degradation without collapsing the setup into an obviously impossible retrieval problem.

`k=2048` and below are **stress budgets only**. They are not part of the default Phase 1 table and should only be used after the higher-budget path already behaves sensibly.

With the default sweep, the main run is:

- `2 tasks × 3 budgets × 100 examples = 600` compressed runs at 32K context

---

## Pilot Gate Before Any Full Run

Run a short pilot first. The point is to reject a broken setup quickly.

1. Spot-check condition A on 2–3 examples each for `VT-4hop` and `MQ-NIAH-4q`.
2. Run 2–3 examples per task at `k=16384`, `k=8192`, and `k=4096`.
3. Inspect both scores and eviction logs before launching the 100-example sweep.

Continue only if the pilot passes common-sense checks:

- Condition A is correct or nearly correct on the sampled examples.
- `k=16384` is close to condition A and definitely not near-all-zero.
- `k=8192` and/or `k=4096` show degradation relative to `k=16384`.
- Tail-pinned spans have nonzero `survival_fraction` in the detailed logs.
- The curve is not obviously pathological, such as `k=16384` collapsing harder than `k=4096` or all task-relevant spans being reported as zero-survival again.

If the pilot fails these checks, stop. Do not run the full sweep until the compressed-cache semantics are fixed.

---

## Tasks

### Task 1: VT-4hop (primary)

Four variable-assignment hops are hidden in the 32K context. The query asks for the final numeric value reachable from the first variable.

Fixed placement:

- Hop 1: depth `12%`
- Hop 2: depth `37%`
- Hop 3: depth `62%`
- Hop 4: tail-pinned inside the protected recent tail
- Terminal value: tail-pinned after hop 4

This task has single-point-of-failure structure. If an earlier hop disappears, the chain breaks completely. That makes it the cleanest attribution task for Phase 1.

Expected behavior in the corrected setup:

- `k=16384`: little or no degradation
- `k=8192`: mild degradation may begin
- `k=4096`: clear degradation should appear, usually through lost middle hops

### Task 2: MQ-NIAH-4q (secondary)

Four different key-value needles are hidden in the 32K context. The query asks for all four values.

Fixed placement:

- Needle 1: depth `10%`
- Needle 2: depth `37%`
- Needle 3: depth `63%`
- Needle 4: tail-pinned inside the protected recent tail

This task yields graded recall rather than binary success. It is useful for seeing whether degradation begins gradually rather than only as a hard failure.

Expected behavior in the corrected setup:

- `k=16384`: near-full recall
- `k=8192`: some reduction in mean recall may begin
- `k=4096`: partial recall should become common, with the tail-pinned needle most robust

### Optional Spot Check: S-NIAH

Single needle at depth `15%`.

Use this only as an auxiliary spot check after the pilot, not as the main Phase 1 gate and not at `k=256`.

If you want a single-needle check:

- use `k=16384` as the non-pathological sanity budget
- optionally compare to `k=4096` as a lighter stress point

Do not use `S-NIAH at k=256` as the default bug test for this reduced harness.

---

## Logging Requirements

The detailed eviction log is mandatory. It is what makes the final analysis attributable rather than just "score went down."

For every compressed run, save:

```python
{
    "example_id": int,
    "task": str,
    "k_budget": int,
    "context_length": 32768,
    "eviction_mask": {
        "kept_positions": list[int],
        "evicted_positions": list[int]
    },
    "token_importance_scores": dict[int, float],
    "task_relevant_positions": list[int],
    "task_relevant_survived": list[bool],
    "task_relevant_spans": [
        {
            "name": str,
            "kind": str,
            "depth_fraction": float,
            "survived": bool,
            "survival_fraction": float
        }
    ],
    "raw_model_output": str,
    "gold_answer": str,
    "correct": bool,      # or fractional score for MQ-NIAH
    "error_type": str
}
```

Important reporting rule:

- `task_relevant_survived` is a convenient boolean view
- the summary metric `eviction_survival_rate` must be computed from the recorded `survival_fraction` values

Also save the Q vectors for the last 64 tokens at eviction time. Later repair phases need them.

---

## Error Taxonomy

Classify each failed example into exactly one type:

| Error type | Definition |
|------------|------------|
| `eviction_miss` | Relevant span(s) were dropped by eviction |
| `chain_break` | VT only: the first missing hop breaks the chain |
| `partial_recall` | MQ-NIAH only: some but not all values were returned |
| `hallucination` | Answer contains values not present in context |
| `other` | Failure does not fit the categories above |

For VT failures, record the first broken hop and its depth. In the corrected setup, chain breaks should mainly concentrate on the middle hops, not the tail-pinned final control spans.

---

## Output Files

```text
results/
  phase1_condition_b.json
  phase1_summary.json
  phase1_eviction_logs/
    VT4hop_k16384_ex001.json
    MQNIAH4q_k4096_ex042.json
    ...
  phase1_q_vectors/
    VT4hop_k16384_ex001_qvecs.pt
    ...
```

Example summary schema:

```json
{
  "VT-4hop": {
    "k16384": {"accuracy": 0.96, "error_breakdown": {...}, "eviction_survival_rate": 0.94},
    "k8192":  {"accuracy": 0.88, "error_breakdown": {...}, "eviction_survival_rate": 0.82},
    "k4096":  {"accuracy": 0.69, "error_breakdown": {...}, "eviction_survival_rate": 0.61}
  },
  "MQ-NIAH-4q": {
    "k16384": {"mean_recall": 3.92, "full_recall_rate": 0.91, ...},
    "k8192":  {"mean_recall": 3.31, "full_recall_rate": 0.58, ...},
    "k4096":  {"mean_recall": 2.37, "full_recall_rate": 0.19, ...}
  }
}
```

---

## Go/No-Go Criteria

Proceed to the next phases only if the corrected sweep supports the intended repair story:

1. `k=16384` stays close to condition A for both main tasks.
2. `k=4096` shows a clear drop relative to `k=16384` on at least one main task.
3. The degradation is attributable in the logs, with lost middle spans and nonzero survival on the tail-pinned controls.

If `k=16384` is already near-zero, or if all budgets behave almost identically, stop and debug the compressed-cache continuation path before doing any more analysis.

---

## What Not to Run

- Do not use `256/512` as the main Phase 1 sweep.
- Do not treat the most pathological budget as the main repair setting.
- Do not reprefill the context after eviction.
- Do not start the 100-example sweep before the short pilot passes.
- Do not use S-NIAH as the primary Phase 1 result.
