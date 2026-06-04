#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

MODEL_DIR="${1:-models/Qwen2.5-3B-Instruct}"
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[phase10-model-transfer-3b-n12] model dir not found: $MODEL_DIR"
  exit 1
fi

MODEL_TRANSFER_MODEL_DIR="$MODEL_DIR" \
  bash phases/phase10_expansion/scripts/run_model_transfer_locked_n12.sh
