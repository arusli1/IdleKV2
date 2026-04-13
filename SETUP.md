# IdleKV Setup Guide

Complete setup instructions for running IdleKV experiments on AWS A100 instances.

## Prerequisites

- AWS A100 instance with CUDA drivers installed
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

### 1.3 Login on Server
```bash
# Install HF CLI
pip install huggingface_hub

# Login with your token
huggingface-cli login
# Paste your token when prompted: hf_xxxxxxxxxxxxxxxxxxxxxxx

# Verify login
huggingface-cli whoami
```

## Step 2: Environment Setup

### 2.1 Clone Repository
```bash
git clone https://github.com/your-username/IdleKV.git
cd IdleKV
```

### 2.2 Verify GPU
```bash
nvidia-smi
# Should show A100 with ~80GB memory
```

### 2.3 Install Dependencies
```bash
# Option A: Use Makefile (recommended)
make install

# Option B: Manual installation
pip install -e ".[dev]"
pip install flash-attn --no-build-isolation  # Optional but recommended
pip install kvpress  # Optional, for baseline comparisons
```

## Step 3: Verification

### 3.1 Test Installation
```bash
# Run test suite
make test
# Expected output: ====================== 21 passed ======================
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
The main configuration is in `configs/main.yaml`. Key settings for A100:

```yaml
hardware: a100_80gb
compression:
  offload_full_kv: false  # Keep full KV on GPU with 80GB
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
huggingface-cli whoami

# Re-login if needed
huggingface-cli logout
huggingface-cli login

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

### A100 80GB Benchmarks
- **Model loading**: ~2-3 minutes (first time)
- **Go/No-Go (5 trials)**: ~5-10 minutes
- **Single experiment run**: ~30-60 minutes
- **Full experiment suite**: 4-8 hours
- **Phase 1 refinement**: ~15-70ms
- **Phase 2 per layer**: ~15-40ms

### Throughput Targets
- **Baseline (no compression)**: ~50-80 tokens/sec
- **SnapKV**: ~60-90 tokens/sec  
- **IdleKV**: ~60-90 tokens/sec (should match SnapKV)

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
- Check HuggingFace login: `huggingface-cli whoami`
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