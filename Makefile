.PHONY: install test lint go-no-go run-main run-ablations run-scale-core run-scale-ablation run-llama-probe run-throughput-qwen dry-run dry-run-scale figures clean

install:
	pip install -e ".[dev]"
	pip install flash-attn --no-build-isolation || true

test:
	pytest tests/ -v --tb=short

lint:
	ruff check idlekv/ scripts/ tests/
	black --check idlekv/ scripts/ tests/

format:
	black idlekv/ scripts/ tests/

# Day 3 decision gate
go-no-go:
	python scripts/go_no_go.py \
		--model meta-llama/Llama-3.1-8B-Instruct \
		--ratio 0.7 \
		--num-trials 20

# Broader A10G exploratory matrix
run-main:
	python scripts/run_experiments.py --config configs/main.yaml

# Broader A10G ablations only
run-ablations:
	python scripts/run_experiments.py --config configs/main.yaml --only-ablations

# Broader A10G dry run
dry-run:
	python scripts/run_experiments.py --config configs/main.yaml --dry-run

# SCALE workshop-core matrix
run-scale-core:
	python scripts/run_experiments.py --config configs/scale_core.yaml

# SCALE minimal mechanism ablation
run-scale-ablation:
	python scripts/run_experiments.py --config configs/scale_mechanism_ablation.yaml --only-ablations

# Tiny harder Llama support probe
run-llama-probe:
	python scripts/run_experiments.py --config configs/llama_hardness_probe.yaml --num-samples 10

# Decode operating-point confirmation on the informative Qwen slice
run-throughput-qwen:
	python scripts/throughput_spotcheck.py --model Qwen/Qwen2.5-7B-Instruct --context-length 4096 --num-trials 3 --num-measure-tokens 128

# SCALE dry run
dry-run-scale:
	python scripts/run_experiments.py --config configs/scale_core.yaml --dry-run

# Generate paper figures
figures:
	python scripts/plot_figures.py --results-dir results/ --output-dir figures/

clean:
	rm -rf results/ figures/ __pycache__ .pytest_cache *.egg-info
