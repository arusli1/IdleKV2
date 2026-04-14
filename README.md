# IdleKV

Treating agentic idle time as a first-class compute resource for KV cache quality recovery. IdleKV runs refinement operations during tool-call pauses in agentic LLM workflows to improve compressed KV cache quality without affecting user-perceived latency.

## Start Here

If you are taking over this repo on a fresh SSH/Codex session, read these in order:

1. [STATUS.md](STATUS.md)
2. [SETUP.md](SETUP.md)
3. [TASKS.md](TASKS.md)
4. [`configs/main.yaml`](configs/main.yaml)

## Setup

### Mac Development (CPU)
```bash
git clone https://github.com/user/IdleKV.git
cd IdleKV
uv venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

### GPU Experiments (A10G 24GB)
```bash
# On GPU machine
git pull
uv venv .venv
source .venv/bin/activate
make install
pytest tests/ -v  # Should pass on GPU too
```

## Running Experiments

### Step 1: Go/No-Go Check (< 10 min)
Run the delayed-query stress gate. It compresses the ledger first, then feeds a
post-compression query suffix and gives Phase 1 a strict 100ms idle window.
This is a real TIR check; the old "question already in prefill" toy setup was a
ceiling task on Llama-8B.
```bash
python scripts/go_no_go.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-trials 12
```

The script defaults to `--ratio 0.7` because this synthetic gate typically
stays at ceiling for `r=0.5`. Main experiments should still sweep the planned
compression ratios.
Treat this gate as a mechanism check for delayed-query recovery, not as a
substitute for the full RULER/LongBench experiment suite.

### Step 2: Small Scouts First
Before launching the broader matrix, run the reduced scouts that choose the
final settings:
```bash
# Reduced two-model scout that already completed on this machine
python scripts/run_experiments.py --config configs/preliminary_idlekv.yaml --only-idlekv

# Qwen seed follow-up scout
python scripts/run_experiments.py --config configs/qwen_seed_followup.yaml --only-idlekv
```

Current critical read after those scouts:
- `Phase 1 @ 100ms` is the best-supported operating point
- `Phase 2` is not part of the core story until it is debugged and rescued
- `llama8b` on `RULER 4K` is ceiling, so one harder Llama probe is more useful
  than more 4K ceiling runs
- for the live next-step plan, read `STATUS.md`

Focused next steps:
```bash
# Decode operating-point check
python scripts/throughput_spotcheck.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --context-length 4096 \
  --num-trials 3 \
  --num-measure-tokens 128

# Tiny harder Llama support probe
python scripts/run_experiments.py \
  --config configs/llama_hardness_probe.yaml \
  --num-samples 10

# SCALE workshop-core matrix
python scripts/run_experiments.py --config configs/scale_core.yaml

# Minimal shadow-buffer mechanism ablation
python scripts/run_experiments.py \
  --config configs/scale_mechanism_ablation.yaml \
  --only-ablations
```

### Step 3: Broader Experiment Suite
Once the scouts have locked the settings, expand to the broader matrix. On a
single A10G, the stable default path is:
- baselines run on `RULER 4K` plus `LongBench`
- IdleKV and ablations default to `RULER 4K` at `r=0.7`
- `sync_refresh` excluded from the default matrix because its clean isolated
  `4K` path still OOMs on this hardware
- `LongBench + IdleKV` stays follow-up work on this box until decode-time cache
  growth is optimized beyond the current HF `DynamicCache` concat path

Use explicit follow-up overrides for `8K` RULER, `sync_refresh`, or
`LongBench + IdleKV` once decode/Phase 2 memory work improves or you move to a
larger-memory GPU.

For a larger-memory scale-up run, start from:
```bash
python scripts/run_experiments.py --config configs/a100_scaleup.yaml --dry-run --model llama8b
```

```bash
# Preview the default nightly matrix
python scripts/run_experiments.py --config configs/main.yaml --dry-run --model llama8b
```

Then run the actual subsets:
```bash
# Quick baseline comparison
python scripts/run_experiments.py --config configs/main.yaml --only-baselines --model llama8b

# IdleKV study on the stable A10G path
python scripts/run_experiments.py --config configs/main.yaml --only-idlekv --model llama8b

# Full workshop-scale suite (stable default matrix)
python scripts/run_experiments.py --config configs/main.yaml --model llama8b --seed 42
```

### Step 4: Generate Figures
```bash
python scripts/plot_figures.py --results-dir results/
```

## Repository Structure
```
IdleKV/
├── STATUS.md                # Current scope, handoff, and run plan
├── README.md                # High-level project entrypoint
├── SETUP.md                 # Environment and run instructions
├── TASKS.md                 # Execution checklist and experiment plan
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
│   ├── preliminary_idlekv.yaml # Reduced scout used to choose main settings
│   ├── qwen_seed_followup.yaml # Qwen-only seed robustness scout
│   ├── scale_core.yaml        # SCALE workshop-core matrix
│   ├── scale_mechanism_ablation.yaml # Minimal workshop mechanism ablation
│   ├── llama_hardness_probe.yaml # Tiny harder Llama support probe
│   ├── a100_scaleup.yaml      # Larger-memory expansion config
│   └── models.yaml            # Model specifications
├── scripts/
│   ├── go_no_go.py            # Day 3 decision gate
│   ├── run_experiments.py     # Full experiment runner
│   ├── throughput_spotcheck.py # Decode operating-point confirmation
│   └── plot_figures.py        # Result visualization
└── tests/                     # Test suite (CPU compatible)
```

## Hardware Note

**Target hardware:** Single NVIDIA A10G (~24GB VRAM)

On A10G-class GPUs, the default path is to keep the compressed working cache on
GPU and offload the full prefill KV backup to CPU (`offload_full_kv: true`).
This avoids exhausting 24GB VRAM while still enabling Phase 2 refresh during
idle windows.

Verified current operating point on this box:
- baseline matrix supports `RULER 4K` plus `LongBench`
- IdleKV/ablation matrix defaults to `RULER 4K` at `r=0.7`
- `8K` IdleKV remains a follow-up target and is not the default nightly setting
- `sync_refresh` is implemented but not part of the default A10G sweep because
  its clean isolated `4K` path still OOMs
- `LongBench + IdleKV` remains follow-up work until the decode cache path is
  made less allocation-heavy on long generations

On larger-memory GPUs, set `offload_full_kv: false` if you prefer to keep the full backup on-device and avoid CPU->GPU transfer during Phase 2 refresh.
