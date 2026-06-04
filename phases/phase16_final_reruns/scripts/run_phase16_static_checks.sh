#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

bash -n phases/phase16_final_reruns/scripts/*.sh
.venv/bin/python -m py_compile \
  phases/phase16_final_reruns/scripts/*.py \
  phases/phase6_repair/src/protocol.py \
  phases/phase6_repair/src/runner.py \
  phases/phase6_repair/scripts/run_phase6.py
.venv/bin/python phases/phase16_final_reruns/scripts/validate_phase16_configs.py
.venv/bin/python -m pytest \
  phases/phase16_final_reruns/tests/test_audit_phase16_locked.py \
  phases/phase6_repair/tests/test_protocol.py \
  phases/phase6_repair/tests/test_runner.py \
  phases/phase9_experiment_deepening/tests/test_phase9_artifact_summary.py \
  -q
