# IdleKV TODO List

**✅ Core Implementation Complete** - All shadow buffer, Phase 1/2, scheduler, and compression components are implemented and tested (21/21 tests passing).

Remaining tasks organized by priority. Complete in order within each section.

## 🔧 A100 Setup & Verification (~30 min)

| Task | Files | Runtime | Exit Criteria |
|------|-------|---------|---------------|
| **Environment setup** | `SETUP.md` | ~5 min | `nvidia-smi` shows A100, `make test` passes |
| **Model download + HF auth** | — | ~15 min | Llama-3.1-8B + Qwen2.5-7B cached via HF |
| **Baseline verification** | `idlekv/eval/` | ~10 min | kvpress SnapKV + LongBench run end-to-end |

## 🧪 Pre-Experiment Validation (~20 min)

| Task | Files | Runtime | Exit Criteria |
|------|-------|---------|---------------|
| **Go/No-Go decision** | `scripts/go_no_go.py` | ~10 min | Prints GO/MARGINAL/NO-GO with delta % |
| **Performance verification** | — | ~10 min | Phase 1: ~15-70ms, Phase 2: ~500ms-1.3s total |

## 🏃‍♂️ Main Experiments (~20-40 hours total - multi-day runs)

**⚠️ Use `tmux` for long runs - SSH disconnects will kill processes**

| Task | Files | Runtime | Exit Criteria |
|------|-------|---------|---------------|
| **Full baseline suite** | `scripts/run_experiments.py` | ~15-25 hours | 7 baselines × 2 models × 2 benchmarks × 3 seeds (~168 runs) |
| **IdleKV budget sweep** | `scripts/run_experiments.py` | ~8-15 hours | 7 budgets × 3 phases × core configs (~504 runs) |
| **Throughput measurement** | `idlekv/eval/metrics.py` | ~1 hour | IdleKV tok/s ≈ SnapKV tok/s (±2%) |
| **Wall-clock timing** | `idlekv/simulation/harness.py` | ~1 hour | IdleKV ≤ SnapKV on 50-turn traces |

## 📊 Ablations & Analysis (~4-8 hours)

| Task | Files | Runtime | Exit Criteria |
|------|-------|---------|---------------|
| **Shadow buffer ablation** | — | ~2-3 hours | 5 buffer sizes × RULER r=0.5 × multiple seeds |
| **Compression ratio sweep** | — | ~2-3 hours | r=0.3/0.5/0.7 × IdleKV × RULER × multiple seeds |
| **Phase comparison** | — | ~2-3 hours | Phase 1 vs 2 vs both × 2 models × benchmarks |
| **Figure generation** | `scripts/plot_figures.py` | ~15 min | All PDFs in `figures/` directory |
| **Per-subtask analysis** | — | ~30 min | 13-subtask RULER breakdown heatmap |

## 📝 Paper Tasks (~3-5 days)

| Task | Runtime | Exit Criteria |
|------|---------|---------------|
| **Method section** | ~4 hours | Architecture diagram + Algorithm boxes |
| **Experiment results** | ~4 hours | Tables, figures, per-subtask analysis |
| **Introduction + related work** | ~4 hours | Motivation + related work table |
| **Mechanistic analysis** | ~3 hours | TIR explanation + free compute analysis |
| **Final polish** | ~2 hours | ICML format + anonymization |

---

## 💡 Implementation Notes

### **Critical Debug Checks**
- **Shadow buffer populated?** `print(shadow_buffer.layers[0].count)` after prefill
- **Re-scoring working?** Compare top-5 token indices before/after Phase 1
- **GQA head mapping correct?** Test both Llama (32→8) and Qwen (28→4)
- **Memory management?** `del full_k, full_v` after each Phase 2 layer

### **Performance Targets (A100)**
- **Phase 1**: 15-70ms for Llama-3.1-8B at 4K context
- **Phase 2**: 15-40ms per layer, 500ms-1.3s total (32 layers)
- **Throughput**: Match SnapKV decode speed (±2%)
- **Wall-clock**: ≤ SnapKV on realistic agentic traces

### **Long-Running Tasks**
- Use `tmux` or `nohup` for multi-hour experiments
- Checkpoint after each (model, baseline, seed) completion
- Save partial results to `results/partial/` for recovery

### **Quality Checklist**
- [x] `make test` passes (21/21 tests) ✅
- [ ] `make lint` passes (clean code)
- [ ] All results reproducible from `configs/main.yaml`
- [ ] JSON results include: model, method, seed, metrics, timing
- [ ] Figures generated from code (not hand-made)
- [ ] Paper anonymized (no model/institution names)

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
- A100 memory management
- GQA head mapping for Llama/Qwen

**Ready to deploy on A100 and start experiments immediately!** 🚀
