# IdleKV

Treating agentic idle time as a first-class compute resource for KV cache quality recovery. IdleKV runs refinement operations during tool-call pauses in agentic LLM workflows to improve compressed KV cache quality without affecting user-perceived latency.

## Setup

### Mac Development (CPU)
```bash
git clone https://github.com/user/IdleKV.git
cd IdleKV
pip install -e ".[dev]"
pytest tests/ -v
```

### GPU Experiments (A100 80GB)
```bash
# On GPU machine
git pull
pip install -e ".[dev]"
pip install flash-attn --no-build-isolation
pytest tests/ -v  # Should pass on GPU too
```

## Running Experiments

### Step 1: Go/No-Go Check (< 10 min)
Test if Phase 1 refinement provides meaningful accuracy gains:
```bash
python scripts/go_no_go.py --model meta-llama/Llama-3.1-8B-Instruct --num-trials 5
```

### Step 2: Full Experiment Suite
Run incremental subsets:
```bash
# Quick baseline comparison
python scripts/run_experiments.py --config configs/main.yaml --only-baselines --model llama8b

# IdleKV ablation study  
python scripts/run_experiments.py --config configs/main.yaml --only-idlekv --model llama8b

# Full suite (all baselines + IdleKV + ablations)
python scripts/run_experiments.py --config configs/main.yaml --model llama8b --seed 42
```

### Step 3: Generate Figures
```bash
python scripts/plot_figures.py --results-dir results/
```

## Repository Structure
```
IdleKV/
├── idlekv/
│   ├── core/                    # Core IdleKV components
│   │   ├── compression.py       # CompressedKVManager (main interface)
│   │   ├── phase1_rescore.py   # Fast re-scoring with shadow buffer
│   │   ├── phase2_refresh.py   # Progressive full-attention refresh
│   │   ├── scheduler.py        # Idle-time scheduler
│   │   ├── shadow_buffer.py    # Recently evicted KV storage
│   │   └── query_buffer.py     # Recent query tracking
│   ├── eval/                   # Benchmark implementations
│   │   ├── ruler.py            # RULER needle-in-a-haystack
│   │   ├── longbench.py        # LongBench multi-task
│   │   └── metrics.py          # Evaluation utilities
│   ├── simulation/             # Agentic workload simulation
│   │   ├── harness.py          # Tool-call pause simulator
│   │   └── tool_distributions.py
│   ├── baselines/              # Baseline implementations
│   │   └── kvpress_baselines.py
│   └── utils/                  # Utilities
│       ├── kv_cache.py         # DynamicCache/tuple compatibility
│       ├── memory.py           # Memory monitoring
│       └── timing.py           # Performance measurement
├── configs/
│   ├── main.yaml              # Main experiment configuration
│   └── models.yaml            # Model specifications
├── scripts/
│   ├── go_no_go.py            # Day 3 decision gate
│   ├── run_experiments.py     # Full experiment runner
│   └── plot_figures.py        # Result visualization
└── tests/                     # Test suite (CPU compatible)
```

## Hardware Note

**Target hardware:** Single NVIDIA A100 (80GB HBM2e, 1935 GB/s bandwidth)

With 80GB VRAM, full uncompressed KV caches (4K-8K context) easily fit on GPU alongside model weights. IdleKV keeps full prefill KV on GPU by default, eliminating CPU-GPU transfers during Phase 2 refresh.

For smaller GPUs, set `offload_full_kv: true` in config to fall back to CPU storage.
