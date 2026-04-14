# IdleKV TODO List

**✅ Core Implementation Complete** - Core runtime is implemented and the current suite passes locally (`41 passed`).

**📋 Prerequisites**: Complete `SETUP.md` first - A10G environment ready, models downloaded, HF authenticated.

---

## 🧪 Validation (~20 min)


| Task                         | Files                 | Runtime | Exit Criteria                                 |
| ---------------------------- | --------------------- | ------- | --------------------------------------------- |
| **Go/No-Go decision**        | `scripts/go_no_go.py` | ~10 min | Delayed-query stress gate at default `r=0.7` prints GO/MARGINAL/NO-GO |
| **Performance verification** | —                     | ~10 min | Phase 1: target <100ms, Phase 2: verify locally on A10G |


## 🏃‍♂️ Main Experiments (~12-30 hours total - multi-day runs)

**⚠️ Use `tmux` for long runs - SSH disconnects will kill processes**

**Verified A10G nightly default**
- baselines on `RULER 4K` + `LongBench`
- IdleKV and ablations on `RULER 4K` at `r=0.7`
- `sync_refresh` excluded from the default matrix because its clean isolated `4K` path still OOMs
- `8K` RULER kept as explicit follow-up work, not default
- `LongBench + IdleKV` kept as explicit follow-up work until decode-cache growth is optimized

| Task                       | Files                          | Runtime      | Exit Criteria                                               |
| -------------------------- | ------------------------------ | ------------ | ----------------------------------------------------------- |
| **Full baseline suite**    | `scripts/run_experiments.py`   | ~10-18 hours | 7 default baselines × 2 models × 2 benchmarks × 3 seeds |
| **IdleKV budget sweep**    | `scripts/run_experiments.py`   | ~6-12 hours  | 7 budgets × 3 phases × core configs on `RULER 4K`             |
| **Throughput measurement** | `idlekv/eval/metrics.py`       | ~1 hour      | IdleKV tok/s ≈ SnapKV tok/s (±2%)                           |
| **Wall-clock timing**      | `idlekv/simulation/harness.py` | ~1 hour      | IdleKV ≤ SnapKV on 50-turn traces                           |


## 📊 Ablations & Analysis (~4-8 hours)


| Task                        | Files                     | Runtime    | Exit Criteria                                   |
| --------------------------- | ------------------------- | ---------- | ----------------------------------------------- |
| **Shadow buffer ablation**  | —                         | ~2-3 hours | 5 buffer sizes × RULER r=0.5 × multiple seeds   |
| **Compression ratio sweep** | —                         | ~2-3 hours | r=0.3/0.5/0.7 × IdleKV × RULER × multiple seeds |
| **Phase comparison**        | —                         | ~2-3 hours | Phase 1 vs 2 vs both × 2 models × benchmarks    |
| **Figure generation**       | `scripts/plot_figures.py` | ~15 min    | All PDFs in `figures/` directory                |
| **Per-subtask analysis**    | —                         | ~30 min    | 13-subtask RULER breakdown heatmap              |


## 📝 Paper Tasks (~3-5 days)


| Task                            | Runtime  | Exit Criteria                           |
| ------------------------------- | -------- | --------------------------------------- |
| **Method section**              | ~4 hours | Architecture diagram + Algorithm boxes  |
| **Experiment results**          | ~4 hours | Tables, figures, per-subtask analysis   |
| **Introduction + related work** | ~4 hours | Motivation + related work table         |
| **Mechanistic analysis**        | ~3 hours | TIR explanation + free compute analysis |
| **Final polish**                | ~2 hours | ICML format + anonymization             |


---

## 💡 Implementation Notes

### **Critical Debug Checks**

- **Shadow buffer populated?** `print(shadow_buffer.layers[0].count)` after prefill
- **Re-scoring working?** Compare top-5 token indices before/after Phase 1
- **GQA head mapping correct?** Test both Llama (32→8) and Qwen (28→4)
- **Memory management?** `del full_k, full_v` after each Phase 2 layer

### **Performance Targets (A10G)**

- **Phase 1**: Target sub-100ms for Llama-3.1-8B at 4K context
- **Phase 2**: Tens of ms per layer plus CPU offload overhead; verify on-host
- **Throughput**: Match SnapKV decode speed within a few percent
- **Wall-clock**: ≤ SnapKV on realistic agentic traces
- **Macro note**: workshop-scale evidence should center on the stable A10G
  matrix first; LongBench+IdleKV and 8K runs are scale-up follow-up work

### **Long-Running Tasks**

- Use `tmux` or `nohup` for multi-hour experiments
- Checkpoint after each (model, baseline, seed) completion
- Save partial results to `results/partial/` for recovery

### **Quality Checklist**

- `make test` passes (`41 passed`) ✅
- `make lint` passes (clean code)
- All results reproducible from `configs/main.yaml`
- JSON results include: model, method, seed, metrics, timing
- Figures generated from code (not hand-made)
- Paper anonymized (no model/institution names)

---

## ✅ Already Completed

**Core Implementation (100% done):**

- Simulation harness (`idlekv/simulation/harness.py`) - 327 lines
- Shadow buffer (`idlekv/core/shadow_buffer.py`) - FIFO ring buffer
- Query buffer (`idlekv/core/query_buffer.py`) - Rolling buffer  
- Compression integration (`idlekv/core/compression.py`) - SnapKV + shadow + full KV
- Phase 1 re-scoring (`idlekv/core/phase1_rescore.py`) - TIR importance ranking
- Phase 2 refresh (`idlekv/core/phase2_refresh.py`) - Progressive layer refresh
- Scheduler integration (`idlekv/core/scheduler.py`) - Tiered execution with thread-safe interrupts

**Bug Fixes Applied:**

- RoPE projection in Phase 1
- O(n) KV tracking instead of O(n²)
- Thread-safe interruption
- A10G memory management / CPU full-KV offload
- GQA head mapping for Llama/Qwen

**Ready to run the workshop-scale A10G sweep tonight.** Baselines cover `RULER 4K` + `LongBench`; IdleKV/ablations cover `RULER 4K`. `8K`, `sync_refresh`, and `LongBench + IdleKV` stay follow-up targets until additional decode-memory work lands.
