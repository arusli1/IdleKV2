# Development Plan

Day-by-day implementation plan mapped to the codebase. Each day lists what to build, what to test, and the exit criteria before moving on.

## Day 1 (Apr 13): Infrastructure

**Goal:** Models load, eval pipelines run, baseline numbers match published results.

| Task | File(s) | Exit criteria |
|------|---------|---------------|
| Install deps on GPU instance | `Makefile` → `make install` | `nvidia-smi` shows GPU, `import kvpress` works |
| Download models | — | `Llama-3.1-8B-Instruct` and `Qwen2.5-7B-Instruct` cached locally |
| Verify kvpress RULER eval | `idlekv/eval/ruler.py` | kvpress SnapKV r=0.5 on RULER NIAH 4K produces a number (don't need to match exactly yet) |
| Verify LongBench eval | `idlekv/eval/longbench.py` | Full-cache LongBench on 3 subtasks runs end-to-end |

**Implementation notes:**
- kvpress has a CLI for RULER. Try `python -m kvpress.eval` first. If it works, wrap it. If not, build a minimal RULER runner using their dataset format.
- For LongBench, use the HuggingFace `datasets` loader. The eval scripts from the LongBench repo are simple F1/ROUGE scorers.

---

## Day 2 (Apr 14): Simulation harness + shadow buffer

**Goal:** Can generate tokens with SnapKV compression, pause at intervals, and capture evicted tokens.

| Task | File(s) | Exit criteria |
|------|---------|---------------|
| Build simulation harness | `idlekv/simulation/harness.py` | `run_simulation()` generates 200 tokens with 2 tool-call pauses, returns a `SimulationTrace` |
| Build shadow buffer | `idlekv/core/shadow_buffer.py` | `make test` passes all `test_shadow_buffer.py` tests |
| Build query buffer | `idlekv/core/query_buffer.py` | `make test` passes all `test_query_buffer.py` tests |
| Hook shadow buffer into compression | `idlekv/core/compression.py` | After `prefill()`, shadow buffer has >0 entries per layer |

**Implementation notes:**
- Start with `CompressedKVManager._compress()`. The SnapKV scoring is ~40 lines. Get the indices right before worrying about performance.
- The harness doesn't need to be fancy yet — just a loop calling `model()` one token at a time with `past_key_values`.
- **Critical debug check:** print `shadow_buffer.layers[0].count` after prefill. If 0, eviction isn't hooked up.

---

## Day 3 (Apr 15): Phase 1 + go/no-go

**Goal:** Phase 1 re-scoring works. Go/no-go decision made.

| Task | File(s) | Exit criteria |
|------|---------|---------------|
| Implement Phase 1 re-scoring | `idlekv/core/phase1_rescore.py` | Given a compressed cache + shadow buffer + query buffer, produces a new cache with swapped tokens |
| Wire Phase 1 into scheduler | `idlekv/core/scheduler.py` | `IdleScheduler.run(max_time_ms=100)` runs Phase 1 and returns |
| Run go/no-go | `scripts/go_no_go.py` | Prints GO/MARGINAL/NO-GO with delta percentage |

**Implementation notes:**
- The hardest part of Phase 1 is `_project_queries()` — getting the GQA head mapping right. Llama-3.1-8B has 32 Q heads and 8 KV heads (group size 4). Qwen2.5-7B has 28 Q heads and 4 KV heads (group size 7). **Test both.**
- For the go/no-go, you need `output_hidden_states=True` on the model forward pass to populate the query buffer. This adds ~5% overhead to generation but is only needed during the 32-token "warm-up" before the tool call.
- **If the go/no-go fails (<1% delta):** Before panicking, check:
  1. Is the shadow buffer actually populated? (Print counts per layer)
  2. Are the re-scored importance scores different from the original? (Print top-5 indices before/after)
  3. Does the needle token appear in the shadow buffer? (Check if the eviction is capturing it)

---

## Day 4 (Apr 16): Phase 2

**Goal:** Full-attention refresh from CPU works end-to-end.

| Task | File(s) | Exit criteria |
|------|---------|---------------|
| Implement CPU KV store | `idlekv/core/phase2_refresh.py` → `CPUKVStore` | After prefill, `cpu_kv_store.memory_bytes` is ~512MB |
| Implement Phase 2 refresh | `idlekv/core/phase2_refresh.py` → `phase2_refresh()` | Refreshing 8 layers changes the compressed cache (different token indices) |
| Wire into scheduler | `idlekv/core/scheduler.py` | `IdleScheduler.run(max_time_ms=2000)` runs Phase 1 then Phase 2 |
| Measure per-layer timing | — | Print time per layer. Should be ~15-40ms on A100. |

**Implementation notes:**
- CPU→GPU transfer: use `tensor.to(device, non_blocking=True)` followed by `torch.cuda.synchronize()`. The `non_blocking` overlaps transfer with any remaining GPU work.
- **Memory management is critical.** After processing each layer, `del full_k, full_v` and let PyTorch reclaim the GPU memory. Only one layer's full KV should be on GPU at a time (~16MB).
- The anytime property is free if you process layers in a simple for-loop with an interrupt check. No special rollback needed — each layer's update is independent.

---

## Days 5–6 (Apr 17–18): Main experiments

**Goal:** Full results table populated.

| Task | File(s) | Exit criteria |
|------|---------|---------------|
| Run 7 baselines × 2 models × RULER + LongBench × 3 seeds | `scripts/run_experiments.py` | JSON results for all configs |
| Run idle-budget sweep (7 budgets) | `scripts/run_experiments.py` | Data for Figure 1 |
| Measure throughput (tokens/sec) | `idlekv/eval/metrics.py` → `measure_throughput()` | IdleKV tok/s = SnapKV tok/s (within 2%) |
| Measure wall-clock on 50-turn trace | `idlekv/simulation/harness.py` | IdleKV wall-clock ≤ SnapKV wall-clock |

**Implementation notes:**
- Use `tmux` or `nohup` for long runs. SSH drops will kill the process otherwise.
- Checkpoint after each (model, baseline, seed) triple. Save intermediate results to `results/partial/`.
- For Baseline 6 (sync refresh), you need to implement RefreshKV-style periodic full attention during generation. Simplest version: every 15 decode steps, recompute full attention for all layers. This adds ~200ms every 15 tokens.
- **Baseline 7 (iso-compute):** Run SnapKV r=0.3 and measure its decode tok/s. The comparison is: does IdleKV at r=0.5 achieve similar quality with higher tok/s?

---

## Day 7 (Apr 19): Ablations + figures

**Goal:** All ablation data collected. All figures generated.

| Task | File(s) | Exit criteria |
|------|---------|---------------|
| Shadow buffer size ablation | — | 5 sizes × RULER at r=0.5 |
| Compression ratio sweep | — | r=0.3/0.5/0.7 × IdleKV × RULER |
| Phase 1 vs Phase 2 vs both | — | 3 configs × 2 models × RULER |
| Generate all figures | `scripts/plot_figures.py` | Fig 1-4 as PDF in `figures/` |
| Per-subtask RULER breakdown | — | 13-subtask heatmap showing where gains concentrate |

**Implementation notes:**
- The per-subtask breakdown is the most diagnostic figure. Expect gains to concentrate on retrieval subtasks (NIAH variants, multi-key, multi-value, multi-query). If gains are uniform across subtasks, that's a stronger result than expected.

---

## Days 8–12 (Apr 20–24): Writing + submission

See the workshop plan PDF for paper structure. Key writing tasks:

| Day | Task |
|-----|------|
| 8 | Method section with architecture diagram. Algorithm box for Phase 1 + Phase 2. |
| 9 | Experiments: tables, figures, per-subtask analysis. |
| 10 | Introduction + motivation. Related work table. |
| 11 | Mechanistic analysis (TIR + free compute). Revise everything. |
| 12 | ICML format check. Anonymize. Submit to SCALE via OpenReview. |

---

## Code quality checklist (before submission)

- [ ] `make test` passes
- [ ] `make lint` passes
- [ ] All results are reproducible from `configs/main.yaml`
- [ ] Results JSON files include: model, method, seed, all metric values, timing
- [ ] Figures are generated from results (not hand-made)
- [ ] README explains how to reproduce
- [ ] No model names or institution names in anonymized paper
