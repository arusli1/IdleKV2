# IdleKV Setup Guide

Complete setup instructions for running IdleKV experiments on AWS A10G instances.

Fresh handoff on this machine:
- read [STATUS.md](STATUS.md) first for current scope, validated results, the recommended run order, and the larger-GPU follow-up path

## Prerequisites

- AWS A10G instance with CUDA drivers installed
- Python 3.9+ 
- Git
- HuggingFace account with access to Llama models

## Step 1: Hugging Face Authentication

### 1.1 Accept Model Licenses
**Critical:** You must accept licenses before downloading models.

1. Visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Click **"Agree and access repository"**
3. Fill out the required form and submit

### 1.2 Generate Access Token
1. Go to https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Give it a name (e.g., "idlekv-experiments")
4. Select **"Read"** permissions
5. Copy the token (starts with `hf_...`)

### 1.3 Login on Server (after Step 2)
```bash
# After Step 2 installs the repo virtualenv and dependencies

# Login with your token
.venv/bin/hf auth login
# Paste your token when prompted: hf_xxxxxxxxxxxxxxxxxxxxxxx

# Verify login
.venv/bin/hf auth whoami
```

## Step 2: Environment Setup

### 2.1 Clone Repository
```bash
git clone https://github.com/your-username/IdleKV.git
cd IdleKV
```

### 2.2 Create Virtual Environment
```bash
uv venv .venv  # or: python3 -m venv .venv
source .venv/bin/activate
```

### 2.3 Verify GPU
```bash
nvidia-smi
# Should show NVIDIA A10G with ~24GB memory
```

### 2.4 Install Dependencies
```bash
# Option A: Use Makefile inside the virtualenv (recommended)
make install

# Option B: Manual installation
pip install -e ".[dev]"
pip install flash-attn --no-build-isolation  # Optional; requires nvcc/CUDA toolkit
pip install kvpress  # Optional, for baseline comparisons
```

## Step 3: Verification

### 3.1 Test Installation
```bash
# Run test suite
make test
# Expected output: tests should pass; current repo state is `44 passed`
```

### 3.2 Test Model Access
```bash
# Test Llama model loading (downloads ~16GB first time)
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Loading Llama tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
print('✓ Llama tokenizer loaded successfully')
print('Testing model loading...')
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.1-8B-Instruct', 
    torch_dtype='auto', 
    device_map='auto'
)
print('✓ Llama model loaded successfully')
print(f'Model device: {next(model.parameters()).device}')
"
```

### 3.3 Test Qwen Model (Optional)
```bash
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
print('✓ Qwen tokenizer accessible')
"
```

## Step 4: Quick Validation

### 4.1 Go/No-Go Check
Run the delayed-query stress gate. This compresses the context before the
retrieval query arrives, appends a longer post-compression query/tool-style
suffix, then gives Phase 1 a strict `100ms` idle window. That setup is
deliberate: it creates a real query shift and online evictions, which is what
Phase 1 is supposed to recover from.

```bash
# Quick stress test (12 trials, ~5-10 minutes)
python scripts/go_no_go.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-trials 12

# Optional: confirm that the old easier operating point is a ceiling task
python scripts/go_no_go.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-trials 12 \
    --ratio 0.5

# More thorough stress test (20 trials, ~20-30 minutes)
make go-no-go
```

**Expected Output:**
```
=== RESULTS ===
Full cache reference:              100.0%
Compressed baseline (r=0.7):      91.7%
IdleKV + Phase 1:                 100.0%
Delta:                            +8.3%
Mean Phase 1 time:                27.4ms

✅ GO: Phase 1 shows a meaningful recovery gain on the delayed-query gate.
```

**Decision Criteria:**
- **GO** (>=3% absolute improvement): Proceed with full experiments
- **MARGINAL** (1-3% improvement): Proceed carefully; gains are real but modest
- **NO-GO** (<1% improvement): Debug before proceeding

This gate is a mechanism-level check for delayed-query recovery. It is not a
replacement for the full RULER/LongBench sweeps used to support broader paper
claims.

## Step 5: Full Experiments

### 5.1 Configuration
The main configuration is in `configs/main.yaml`. Key settings for A10G:

```yaml
hardware: a10g_24gb
compression:
  offload_full_kv: true  # Store full KV on CPU for 24GB setup
evaluation:
  ruler:
    context_lengths: [4096]  # A10G default path
models:
  - meta-llama/Llama-3.1-8B-Instruct  
  - Qwen/Qwen2.5-7B-Instruct
```

The broader A10G matrix intentionally excludes the `sync_refresh` baseline on
single-A10G runs. The baseline remains implemented, but its clean isolated `4K`
path still OOMs on this hardware, so it should be treated as follow-up work
after Phase 2 memory optimization or on a larger-memory GPU.

The default single-A10G matrix also separates the broad baseline readout from
the heavier IdleKV readout:
- baselines run on `RULER 4K` plus `LongBench`
- IdleKV and ablations default to `RULER 4K` at `r=0.7`
- `LongBench + IdleKV` is follow-up work until decode-time cache growth is
  optimized beyond the current HF `DynamicCache` concat path

If you move to an A100-class GPU later, use `configs/a100_scaleup.yaml` as the
starting point rather than mutating the single-A10G default in place.

### 5.2 Run Experiments
Start with the reduced scouts that choose the larger matrix:
```bash
# Two-model reduced scout
python scripts/run_experiments.py \
    --config configs/preliminary_idlekv.yaml \
    --only-idlekv

# Qwen follow-up scout: seeds 123 and 456
python scripts/run_experiments.py \
    --config configs/qwen_seed_followup.yaml \
    --only-idlekv
```

Current read after those scouts:
- `Phase 1 @ 100ms` is the best-supported operating point
- `1000ms phase=1+2` is not part of the workshop-core story on current
  evidence
- before scaling up, prefer one throughput / wall-clock spot-check and one
  harder Llama probe over a large additional exploratory sweep
- read `STATUS.md` for the live next-step plan on this machine

Focused next commands:
```bash
python scripts/throughput_spotcheck.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --context-length 4096 \
    --num-trials 3 \
    --num-measure-tokens 128

python scripts/run_experiments.py \
    --config configs/llama_hardness_probe.yaml \
    --num-samples 10

python scripts/run_experiments.py \
    --config configs/scale_core.yaml

python scripts/run_experiments.py \
    --config configs/scale_mechanism_ablation.yaml \
    --only-ablations
```

Then, if you want the broader constrained-hardware expansion matrix rather than
the workshop-core package, run:
```bash
# Broader A10G exploratory suite (multi-day on one A10G)
make run-main

# Or with specific options
python scripts/run_experiments.py \
    --config configs/main.yaml \
    --model llama8b \
    --seed 42

# Dry run (see what will be executed)
make dry-run

# Explicit 8K follow-up run (not part of the default A10G path)
python scripts/run_experiments.py \
    --config configs/main.yaml \
    --model llama8b \
    --seed 42 \
    --benchmarks ruler \
    --ruler-context-lengths 4096,8192

# Explicit LongBench + IdleKV follow-up probe after decode-memory work lands
python scripts/run_experiments.py \
    --config configs/main.yaml \
    --only-idlekv \
    --model llama8b \
    --benchmarks longbench \
    --longbench-max-input-length 3840

# A100-class scale-up dry run
python scripts/run_experiments.py \
    --config configs/a100_scaleup.yaml \
    --dry-run \
    --model llama8b
```

### 5.3 Monitor Progress
Experiments save results incrementally to `results/`:
```bash
# Check progress
ls -la results/
tail -f results/experiment_log.txt  # If logging is enabled

# Check GPU utilization
watch -n 1 nvidia-smi
```

## Step 6: Results & Analysis

### 6.1 Generate Figures
```bash
# After experiments complete
make figures

# Check outputs
ls figures/
# Should contain: accuracy_vs_budget.pdf, throughput_comparison.pdf, etc.
```

### 6.2 Results Structure
```
results/
├── baselines/
│   ├── full_cache_llama8b_ruler_seed42.json
│   ├── snapkv_0.5_llama8b_ruler_seed42.json
│   └── ...
├── idlekv/
│   ├── idlekv_budget100_llama8b_ruler_seed42.json
│   └── ...
└── summary/
    └── aggregated_results.csv
```

## Troubleshooting

### Model Loading Issues
```bash
# Check HF login status
hf auth whoami

# Re-login if needed
hf auth logout
hf auth login

# Test direct model access
python -c "from huggingface_hub import hf_hub_download; print('Testing download...'); hf_hub_download('meta-llama/Llama-3.1-8B-Instruct', 'config.json')"
```

### CUDA/GPU Issues
```bash
# Check CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"

# Check GPU memory
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
```

### Memory Issues
If you get CUDA OOM errors:
1. Set `offload_full_kv: true` in `configs/main.yaml`
2. Reduce batch size in experiments
3. Prefer the scoped workshop configs (`scale_core.yaml`, `scale_mechanism_ablation.yaml`) over the broader `main.yaml` matrix on a single A10G

### Flash Attention Installation
```bash
# If flash-attn fails to install
pip install ninja  # Required build tool
pip install flash-attn --no-build-isolation --no-cache-dir

# Check installation
python -c "import flash_attn; print('Flash attention installed successfully')"
```

### Permission Issues
```bash
# If you get permission errors during model download
chmod -R 755 ~/.cache/huggingface/
```

## Performance Expectations

### A10G 24GB Expectations
- **Model loading**: ~2-3 minutes (first time)
- **Go/No-Go (12 trials)**: ~5-15 minutes
- **Single scout experiment**: typically minutes, not hours, on `RULER 4K`
- **Workshop-core matrix (`scale_core.yaml`)**: same-day run on one A10G
- **Broader A10G matrix (`main.yaml`)**: multi-day if run end-to-end
- **Phase 1 refinement**: target sub-100ms
- **Phase 2 per layer**: hardware-dependent; verify on your local prompt mix
- **Default A10G scope**: baselines on `RULER 4K` + `LongBench`; IdleKV and
  ablations on `RULER 4K` at `r=0.7`
- **Default A10G RULER context**: `4K`
- **8K status**: clean isolated IdleKV still OOMs on this A10G path; treat as
  follow-up, not default
- **LongBench + IdleKV status**: follow-up on this A10G path until decode
  cache growth is optimized

### Throughput Targets
- **Absolute tok/s**: depends on model, prompt length, and PCIe overhead on A10G
- **Relative target**: IdleKV should stay close to SnapKV throughput (within a few percent)

## Experiment Configurations

### Quick Testing
```bash
# Test single model, single seed
python scripts/run_experiments.py --config configs/main.yaml --model llama8b --seed 42 --only-baselines

# Test only IdleKV variations
python scripts/run_experiments.py --config configs/main.yaml --model llama8b --only-idlekv

# Test explicit 8K follow-up path
python scripts/run_experiments.py --config configs/main.yaml --model llama8b --only-idlekv --benchmarks ruler --ruler-context-lengths 8192

# Test explicit LongBench + IdleKV follow-up path
python scripts/run_experiments.py --config configs/main.yaml --model llama8b --only-idlekv --benchmarks longbench --longbench-max-input-length 3840
```

### Production Runs
```bash
# All models, all seeds, all baselines + IdleKV
python scripts/run_experiments.py --config configs/main.yaml

# With specific output directory
python scripts/run_experiments.py --config configs/main.yaml --output-dir /data/idlekv_results/
```

## Support

### Debug Mode
```bash
# Run with verbose logging
python scripts/run_experiments.py --config configs/main.yaml --verbose

# Python debug mode
python -u scripts/go_no_go.py --model meta-llama/Llama-3.1-8B-Instruct --num-trials 5
```

### Common Error Messages

**"Repository not found"**
- Check HuggingFace login: `hf auth whoami`
- Verify model access permissions

**"CUDA out of memory"**  
- Set `offload_full_kv: true` in config
- Check `nvidia-smi` for memory usage
- On a single A10G, keep the default `4K` RULER setting unless you are
  explicitly probing the 8K or `LongBench + IdleKV` follow-up paths

**"No module named 'flash_attn'"**
- This is optional - experiments will run without it
- Install with: `pip install flash-attn --no-build-isolation`

**"Tests failing"**
- Check Python version: `python --version` (need >=3.9)
- Reinstall: `pip install -e ".[dev]" --force-reinstall`

For additional support, check the issue tracker or refer to the test suite in `tests/` for working examples.
