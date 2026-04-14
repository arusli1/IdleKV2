"""Tests for experiment runner matrix construction."""

from argparse import Namespace

from scripts.run_experiments import (
    benchmarks_for_experiment,
    build_experiment_matrix,
    experiment_slug,
    longbench_max_input_length,
    phases_label,
    release_model,
    required_attn_implementation,
)


def _args(**overrides):
    defaults = {
        "only_baselines": False,
        "only_idlekv": False,
        "only_ablations": False,
        "model": None,
        "ratio": None,
        "seed": None,
        "max_experiments": None,
        "benchmarks": None,
        "num_samples": None,
        "skip_existing": False,
        "config": "configs/main.yaml",
        "output_dir": None,
        "dry_run": False,
        "verbose": False,
        "ruler_context_lengths": None,
        "longbench_max_input_length": None,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def _config():
    return {
        "models": [
            {"name": "model-a", "short": "a"},
            {"name": "model-b", "short": "b"},
        ],
        "evaluation": {
            "seeds": [1, 2],
            "benchmarks": ["ruler", "longbench"],
            "benchmarks_by_type": {
                "baseline": ["ruler", "longbench"],
                "idlekv": ["ruler"],
                "ablation_buffer": ["ruler"],
            },
            "ruler": {"context_lengths": [4096]},
            "longbench": {"max_input_length": 4096},
        },
        "compression": {
            "primary_ratio": 0.5,
            "shadow_buffer_size": 256,
            "query_buffer_size": 32,
            "offload_full_kv": True,
        },
        "baselines": [
            {"name": "full_cache", "ratio": None},
            {"name": "snapkv_0.5", "ratio": 0.5, "method": "snapkv"},
        ],
        "idlekv": {
            "idle_budgets_ms": [0, 100],
            "phases": [1, "1+2"],
        },
        "ablations": {
            "shadow_buffer_sizes": [0, 64],
        },
    }


def test_build_experiment_matrix_counts_all_modes():
    experiments = build_experiment_matrix(_config(), _args())
    assert len(experiments) == 32


def test_build_experiment_matrix_filters_model_and_seed():
    experiments = build_experiment_matrix(_config(), _args(model="b", seed=7))
    assert all(exp["model"]["short"] == "b" for exp in experiments)
    assert all(exp["seed"] == 7 for exp in experiments)
    assert len(experiments) == 8


def test_build_experiment_matrix_supports_explicit_idlekv_conditions():
    config = _config()
    config["idlekv"] = {
        "conditions": [
            {"idle_budget_ms": 0, "phases": 1},
            {"idle_budget_ms": 1000, "phases": "1+2"},
        ]
    }
    experiments = build_experiment_matrix(config, _args(only_idlekv=True, model="a", seed=7))
    assert len(experiments) == 2
    assert experiments[0]["idle_budget_ms"] == 0
    assert experiments[0]["phases"] == 1
    assert experiments[1]["idle_budget_ms"] == 1000
    assert experiments[1]["phases"] == "1+2"


def test_experiment_slug_encodes_phase_selection():
    assert phases_label(1) == "1"
    assert phases_label("1+2") == "1+2"
    exp = {
        "type": "idlekv",
        "model": {"short": "llama8b"},
        "seed": 42,
        "ratio": 0.7,
        "idle_budget_ms": 100,
        "phases": "1+2",
    }
    assert experiment_slug(exp) == "idlekv_r0.7_budget100_phases12_llama8b_seed42"


def test_required_attn_implementation_uses_eager_for_h2o():
    h2o_exp = {
        "type": "baseline",
        "baseline": {"name": "h2o_0.5", "method": "h2o"},
    }
    idlekv_exp = {
        "type": "idlekv",
        "phases": "1+2",
    }
    assert required_attn_implementation(h2o_exp) == "eager"
    assert required_attn_implementation(idlekv_exp) == "sdpa"


def test_release_model_returns_cleared_bindings():
    model, tokenizer = release_model(object(), object())
    assert model is None
    assert tokenizer is None


def test_longbench_max_input_length_prefers_cli_override():
    assert longbench_max_input_length(_config(), _args()) == 4096
    assert longbench_max_input_length(_config(), _args(longbench_max_input_length=8192)) == 8192


def test_benchmarks_for_experiment_respects_type_default_and_cli_filter():
    baseline_exp = {
        "type": "baseline",
        "baseline": {"name": "full_cache", "ratio": None},
    }
    idlekv_exp = {
        "type": "idlekv",
        "phases": 1,
    }

    assert benchmarks_for_experiment(baseline_exp, _config(), _args()) == ["ruler", "longbench"]
    assert benchmarks_for_experiment(idlekv_exp, _config(), _args()) == ["ruler"]
    assert benchmarks_for_experiment(
        baseline_exp,
        _config(),
        _args(benchmarks="longbench"),
    ) == ["longbench"]
