.PHONY: install test lint go-no-go run-main run-ablations figures clean

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

# Full experiment suite
run-main:
	python scripts/run_experiments.py --config configs/main.yaml

# Ablations only
run-ablations:
	python scripts/run_experiments.py --config configs/main.yaml --only-ablations

# Dry run (print experiment matrix)
dry-run:
	python scripts/run_experiments.py --config configs/main.yaml --dry-run

# Generate paper figures
figures:
	python scripts/plot_figures.py --results-dir results/ --output-dir figures/

clean:
	rm -rf results/ figures/ __pycache__ .pytest_cache *.egg-info
