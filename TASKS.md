# IdleKV TODO List

**✅ Core Implementation Complete** - Core runtime is implemented and the current suite passes locally (`44 passed`).

**📋 Prerequisites**: Complete `SETUP.md` first - A10G environment ready, models downloaded, HF authenticated.

Fresh handoff:
- read `STATUS.md` first for the current scoped run plan, validated pilot, and scale-up path

## 🎯 Submission Strategy

- Primary target: `ICML SCALE 2026` workshop paper.
- Preferred package: `7` content pages plus references.
- Fallback if the broader matrix is not ready in time: `3`-page late-breaker.
- Longer-term target: expand the same project into a `NeurIPS 2026` main-track
  paper after larger-GPU runs and stronger systems evidence.

Critical scope rule:
- the `SCALE` version should center on the durable claim that the current data
  actually supports
- do not force Phase 2 or the full aspirational matrix into the workshop story
  unless later runs clearly rescue them

---

## 🧪 Validation (~20 min)


| Task                         | Files                 | Runtime | Exit Criteria                                 |
| ---------------------------- | --------------------- | ------- | --------------------------------------------- |
| **Go/No-Go decision**        | `scripts/go_no_go.py` | ~10 min | Delayed-query stress gate at default `r=0.7` prints GO/MARGINAL/NO-GO |
| **Performance verification** | —                     | ~10 min | Phase 1: target <100ms, Phase 2: verify locally on A10G |

## 🔬 Current Confirmation Priority (~1-2 hours)

- The Qwen follow-up scout is finished.
- Key result:
  - `0ms, phase=1`: `0.877`
  - `100ms, phase=1`: `0.903`
  - `1000ms, phase=1`: `0.903`
  - `1000ms, phase=1+2`: `0.653`
- Read:
  - the `0ms -> 100ms` Phase 1 gain is robust across seeds
  - there is no evidence that `>100ms` helps Phase 1 on this slice
  - Phase 2 is currently a liability, not part of the workshop-core story

Next tiny runs:
- Throughput / wall-clock spot-check
  - use `scripts/throughput_spotcheck.py`
  - compare compressed no-idle `r=0.7` vs IdleKV `r=0.7, 100ms, phase=1`
  - goal: verify that the shortlisted workshop method keeps the same practical
    decode operating point
- Llama hardness probe
  - use `configs/llama_hardness_probe.yaml` with `--num-samples 10`
  - goal: decide whether Llama belongs in the `SCALE` matrix or only in the
    later A100 expansion

If those confirm the current read:
- main workshop matrix: `configs/scale_core.yaml`
- minimal mechanism ablation: `configs/scale_mechanism_ablation.yaml`

## 🏃‍♂️ Main Experiments (~12-30 hours total - multi-day runs)

**⚠️ Use `tmux` for long runs - SSH disconnects will kill processes**

**Verified A10G nightly default**
- baselines on `RULER 4K` + `LongBench`
- IdleKV and ablations on `RULER 4K` at `r=0.7`
- `sync_refresh` excluded from the default matrix because its clean isolated `4K` path still OOMs
- `8K` RULER kept as explicit follow-up work, not default
- `LongBench + IdleKV` kept as explicit follow-up work until decode-cache growth is optimized

Interpretation:
- this A10G matrix is useful for scouting and constrained-hardware evidence
- it is not automatically the final `SCALE` paper matrix, and it is not the
  final `NeurIPS` matrix
- after the Qwen scout, the default assumption should be a sharper
  Phase-1-centered `SCALE` matrix rather than brute-forcing every remaining
  condition

| Task                       | Files                          | Runtime      | Exit Criteria                                               |
| -------------------------- | ------------------------------ | ------------ | ----------------------------------------------------------- |
| **Full baseline suite**    | `scripts/run_experiments.py`   | ~10-18 hours | 7 default baselines × 2 models × 2 benchmarks × 3 seeds |
| **IdleKV budget sweep**    | `scripts/run_experiments.py`   | ~6-12 hours  | 7 budgets × 3 phases × core configs on `RULER 4K`             |
| **Throughput measurement** | `idlekv/eval/metrics.py`       | ~1 hour      | IdleKV tok/s ≈ SnapKV tok/s (±2%)                           |
| **Wall-clock timing**      | `idlekv/simulation/harness.py` | ~1 hour      | IdleKV ≤ SnapKV on 50-turn traces                           |


## 📊 Ablations & Analysis (~4-8 hours)


| Task                        | Files                     | Runtime    | Exit Criteria                                   |
| --------------------------- | ------------------------- | ---------- | ----------------------------------------------- |
| **Shadow buffer ablation**  | —                         | ~2-3 hours | 5 buffer sizes × RULER `r=0.7` × multiple seeds |
| **Compression ratio sweep** | —                         | ~2-3 hours | r=0.3/0.5/0.7 × IdleKV × RULER × multiple seeds |
| **Phase comparison**        | —                         | ~2-3 hours | Phase 1 vs 2 vs both on `RULER 4K` first; expand to LongBench on A100 scale-up |
| **Figure generation**       | `scripts/plot_figures.py` | ~15 min    | All PDFs in `figures/` directory                |
| **Per-subtask analysis**    | —                         | ~30 min    | 13-subtask RULER breakdown heatmap              |


## 📝 Paper Tasks (~3-5 days)


| Task                                | Runtime  | Exit Criteria                                                         |
| ----------------------------------- | -------- | --------------------------------------------------------------------- |
| **SCALE 7-page draft**              | ~1 day   | one clean workshop story centered on delayed-query + Phase 1 / 100ms |
| **Late-breaker fallback**           | ~3 hours | compress the same evidence into 3 pages if broader runs slip         |
| **Experiment results**              | ~4 hours | only measured tables/figures for the active submission scope         |
| **Introduction + related work**     | ~4 hours | motivation + prior KV-cache work positioned honestly                 |
| **Mechanistic analysis**            | ~3 hours | TIR explanation + idle-compute framing                              |
| **NeurIPS expansion plan (later)**  | ~1 day   | broader A100 matrix and systems package defined, not necessarily run |
| **Final polish**                    | ~2 hours | SCALE format + anonymization                                         |


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
- **Macro note**: the `SCALE` version should center on the best-supported
  workshop claim first; LongBench+IdleKV, 8K, and any Phase 2-heavy story are
  scale-up follow-up work unless later runs clearly support them

### **Long-Running Tasks**

- Use `tmux` or `nohup` for multi-hour experiments
- Checkpoint after each (model, baseline, seed) completion
- Save partial results to `results/partial/` for recovery

### **Quality Checklist**

- `make test` passes (`44 passed`) ✅
- `make lint` passes (clean code)
- All results reproducible from `configs/main.yaml`
- Larger-memory expansion is reproducible from `configs/a100_scaleup.yaml`
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

**Ready for the final tiny confirmation runs and then a sharper larger run.**
The informative Qwen slice is now locked enough to demote Phase 2 from the core
story. Choose the smallest matrix that is strong enough for `SCALE 2026`, then
save the broader `NeurIPS 2026` evidence package for the larger-GPU phase.
