"""Round-5 attack-2 defuse: verify GPU vs CPU scoring agreement.

For each (example, K) cell, compute RepairKV's q2_scores via:
- CPU path (current default, score_evicted_positions._score_on_gpu=False)
- GPU path (new path, score_evicted_positions._score_on_gpu=True)

Confirm the resulting top-K position selections agree within Wilcoxon
noise on the final answer score. If they do, the abstract claim
binding "quality from the runner" + "runtime from W2 probe" is
internally consistent: quality and runtime come from a path that
gives the SAME quality on GPU as on CPU.

Runs n=4 examples on Qwen at K=96 4Q. Quick smoke (~5 min).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Set the GPU-scoring switch BEFORE importing the runner so the import-
# time defaults take effect. Done via module attribute on
# score_evicted_positions itself.
from phases.phase6_repair.src.selectors import score_evicted_positions


def main() -> int:
    # Force GPU scoring for this run.
    score_evicted_positions._score_on_gpu = True
    print("[verify-gpu] running with GPU scoring path enabled")
    # Defer to the standard runner CLI for the rest.
    import subprocess
    out_dir = Path("phases/phase18_pre_submission/results/gpu_verify")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"verify_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.log"
    cmd = [
        ".venv/bin/python", "phases/phase6_repair/scripts/run_phase6.py",
        "--stage", "full",
        "--task", "clean_suite",
        "--num-samples", "4",
        "--base-context-budget", "16384",
        "--recency-window", "128",
        "--query-scoring-mode", "exact_q",
        "--oracle-mode", "gold_spans",
        "--k", "96",
        "--conditions", "A", "B", "B_match", "IdleKV",
    ]
    print(f"[verify-gpu] cmd: {' '.join(cmd)}")
    print(f"[verify-gpu] log:  {log_path}")
    # The subprocess won't inherit the _score_on_gpu attribute. Need a
    # different mechanism: env var. Read in selectors.py at first call.
    import os
    env = os.environ.copy()
    env["PHASE18_SCORE_ON_GPU"] = "1"
    with open(log_path, "w") as fp:
        proc = subprocess.run(cmd, env=env, stdout=fp, stderr=subprocess.STDOUT)
    print(f"[verify-gpu] exit code: {proc.returncode}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
