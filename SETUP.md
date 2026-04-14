# IdleKV Setup Guide

Complete setup instructions for running IdleKV experiments on AWS A10G instances.

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
# Expected output: ====================== 23 passed ======================
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
Test if Phase 1 refinement provides meaningful accuracy gains:

```bash
# Quick test (5 trials, ~5-10 minutes)
python scripts/go_no_go.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-trials 5 \
    --ratio 0.5

# More thorough test (20 trials, ~20-30 minutes)
make go-no-go
```

**Expected Output:**
```
Phase 1 Refinement Results:
========================
Baseline accuracy: 0.756
IdleKV accuracy:   0.782
Delta: +2.6%

Result: GO - Phase 1 shows meaningful improvement
```

**Decision Criteria:**
- **GO** (>1% improvement): Proceed with full experiments
- **MARGINAL** (0.5-1% improvement): Consider proceeding but expect modest gains
- **NO-GO** (<0.5% improvement): Debug before proceeding

## Step 5: Full Experiments

### 5.1 Configuration
The main configuration is in `configs/main.yaml`. Key settings for A10G:

```yaml
hardware: a10g_24gb
compression:
  offload_full_kv: true  # Store full KV on CPU for 24GB setup
models:
  - meta-llama/Llama-3.1-8B-Instruct  
  - Qwen/Qwen2.5-7B-Instruct
```

### 5.2 Run Experiments
```bash
# Full experiment suite (several hours)
make run-main

# Or with specific options
python scripts/run_experiments.py \
    --config configs/main.yaml \
    --model llama8b \
    --seed 42

# Dry run (see what will be executed)
make dry-run
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
3. Use gradient checkpointing:
   ```python
   model.gradient_checkpointing_enable()
   ```

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
- **Go/No-Go (5 trials)**: ~5-15 minutes
- **Single experiment run**: ~30-90 minutes
- **Full experiment suite**: 4-12 hours
- **Phase 1 refinement**: target sub-100ms
- **Phase 2 per layer**: hardware-dependent; verify on your local prompt mix

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

**"No module named 'flash_attn'"**
- This is optional - experiments will run without it
- Install with: `pip install flash-attn --no-build-isolation`

**"Tests failing"**
- Check Python version: `python --version` (need >=3.9)
- Reinstall: `pip install -e ".[dev]" --force-reinstall`

For additional support, check the issue tracker or refer to the test suite in `tests/` for working examples.
