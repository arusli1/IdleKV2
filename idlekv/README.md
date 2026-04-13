# IdleKV

**Treating agentic idle time as a first-class compute resource for KV cache quality recovery.**

IdleKV recovers KV cache quality lost during compression by running refinement operations during GPU idle time in agentic LLM inference (tool-call pauses). It adds no measurable overhead to generation throughput.

## Quick Start

```bash
pip install -e ".[dev]"

# Run the Day 3 go/no-go check
python scripts/go_no_go.py --model meta-llama/Llama-3.1-8B-Instruct --ratio 0.5

# Run full experiment suite
python scripts/run_experiments.py --config configs/main.yaml
```

## Repo Structure

```
idlekv/
├── idlekv/
│   ├── core/
│   │   ├── shadow_buffer.py      # FIFO ring buffer for evicted KV pairs
│   │   ├── phase1_rescore.py     # TIR-informed re-scoring during idle time
│   │   ├── phase2_refresh.py     # Progressive full-attention refresh
│   │   ├── query_buffer.py       # Rolling buffer of recent hidden states
│   │   ├── scheduler.py          # Tiered idle-time scheduler
│   │   └── compression.py        # SnapKV wrapper with shadow buffer hooks
│   ├── eval/
│   │   ├── ruler.py              # RULER benchmark runner
│   │   ├── longbench.py          # LongBench benchmark runner
│   │   ├── metrics.py            # KL divergence, recovery delta, throughput
│   │   └── runner.py             # Unified experiment runner
│   ├── simulation/
│   │   ├── harness.py            # Agentic simulation (pause/resume generation)
│   │   └── tool_distributions.py # Tool-call duration distributions
│   ├── baselines/
│   │   └── kvpress_baselines.py  # Wrappers for SnapKV, H2O, StreamingLLM, sync refresh
│   └── utils/
│       ├── logging.py            # Structured JSON logging
│       ├── timing.py             # GPU timing utilities (CUDA events)
│       └── memory.py             # Memory tracking
├── scripts/
│   ├── go_no_go.py               # Day 3 decision gate
│   ├── run_experiments.py        # Full experiment suite
│   ├── run_ablations.py          # Ablation studies
│   └── plot_figures.py           # Generate paper figures
├── configs/
│   ├── main.yaml                 # Main experiment config
│   ├── ablations.yaml            # Ablation configs
│   └── models.yaml               # Model paths and settings
├── tests/
│   ├── test_shadow_buffer.py
│   ├── test_phase1.py
│   ├── test_phase2.py
│   └── test_harness.py
└── notebooks/
    └── explore_results.ipynb     # Interactive results analysis
```

## Key Design Decisions

1. **kvpress for baselines only.** IdleKV's refinement operates directly on `past_key_values` tensors, not through kvpress hooks. This avoids fighting kvpress's architecture.

2. **Atomic cache updates.** Phase 1 and Phase 2 build a new cache in a separate buffer, then swap atomically at completion. No partially-modified cache is ever visible to generation.

3. **Anytime interruption.** Both phases process layers independently. If the tool returns mid-refinement, already-processed layers keep their improvements; unprocessed layers keep their original state.

4. **Reproducibility.** All experiments use 3 seeds. Configs are YAML. Results are structured JSON. Figures are generated from results, not hand-made.
