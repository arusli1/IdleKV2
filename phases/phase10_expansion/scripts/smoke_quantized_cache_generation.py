#!/usr/bin/env python3
"""Tiny generation smoke for Transformers QuantizedCache backends."""

from __future__ import annotations

import argparse
import json
import os
import sys
import sysconfig
import time
from pathlib import Path


def _ensure_python_script_dir_on_path() -> None:
    """Expose the active venv's scripts to extension builders.

    Some quantized-cache backends JIT-compile CUDA/C++ extensions through
    PyTorch, which shells out to the ``ninja`` executable. The Python package
    can be installed while its executable is still absent from PATH when the
    venv was not activated by the caller.
    """
    script_dirs = [
        Path(sys.executable).parent,
        Path(sysconfig.get_path("scripts") or ""),
    ]
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    for script_dir in reversed([str(path) for path in script_dirs if str(path)]):
        if script_dir not in path_parts:
            path_parts.insert(0, script_dir)
    os.environ["PATH"] = os.pathsep.join(path_parts)


_ensure_python_script_dir_on_path()

import torch

PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_ROOT.parents[1]
for root in (PHASE_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from phases.phase2_kv_cache.src.runtime import MODEL_DIR as DEFAULT_MODEL_DIR  # noqa: E402
from phases.phase2_kv_cache.src.runtime import load_model, load_tokenizer  # noqa: E402


def _decode_new_text(tokenizer, output_ids: torch.Tensor, input_len: int) -> str:
    return tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)


def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    model_dir = Path(args.model_dir).expanduser()
    model = load_model(model_dir)
    tokenizer = load_tokenizer(model_dir)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    input_len = int(inputs["input_ids"].shape[1])

    started = time.perf_counter()
    with torch.no_grad():
        baseline_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=int(args.max_new_tokens),
            use_cache=True,
        )
    baseline_s = time.perf_counter() - started

    quant_started = time.perf_counter()
    with torch.no_grad():
        quantized_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=int(args.max_new_tokens),
            cache_implementation="quantized",
            cache_config={
                "backend": str(args.backend),
                "nbits": int(args.nbits),
                "axis_key": int(args.axis_key),
                "axis_value": int(args.axis_value),
                "q_group_size": int(args.q_group_size),
                "residual_length": int(args.residual_length),
            },
        )
    quantized_s = time.perf_counter() - quant_started

    payload = {
        "model_dir": str(model_dir),
        "backend": str(args.backend),
        "nbits": int(args.nbits),
        "axis_key": int(args.axis_key),
        "axis_value": int(args.axis_value),
        "q_group_size": int(args.q_group_size),
        "residual_length": int(args.residual_length),
        "prompt": str(args.prompt),
        "max_new_tokens": int(args.max_new_tokens),
        "baseline_text": _decode_new_text(tokenizer, baseline_ids, input_len),
        "quantized_text": _decode_new_text(tokenizer, quantized_ids, input_len),
        "baseline_token_count": int(baseline_ids.shape[1] - input_len),
        "quantized_token_count": int(quantized_ids.shape[1] - input_len),
        "baseline_s": round(baseline_s, 6),
        "quantized_s": round(quantized_s, 6),
        "passed": bool(quantized_ids.shape[1] > input_len),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--backend", choices=("quanto", "hqq"), default="quanto")
    parser.add_argument("--nbits", type=int, choices=(2, 4), default=4)
    parser.add_argument("--axis-key", type=int, default=0)
    parser.add_argument("--axis-value", type=int, default=0)
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument("--residual-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PHASE_ROOT / "results" / "quantized_cache_generation_smoke.json",
    )
    return parser.parse_args()


def main() -> int:
    payload = run_smoke(parse_args())
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
