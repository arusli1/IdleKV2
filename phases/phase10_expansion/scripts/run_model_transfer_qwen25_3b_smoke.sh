#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

MODEL_DIR="${1:-models/Qwen2.5-3B-Instruct}"
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[phase10-model-transfer-3b] model dir not found: $MODEL_DIR"
  exit 1
fi
MODEL_LABEL="$(basename "$MODEL_DIR" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_')"
ABILITY_SAMPLES="${MODEL_TRANSFER_ABILITY_NUM_SAMPLES:-4}"
ABILITY_CSV="${MODEL_TRANSFER_ABILITY_CSV:-phases/phase10_expansion/results/model_transfer_ability_${MODEL_LABEL}_n${ABILITY_SAMPLES}.csv}"
if [[ ! -f "$ABILITY_CSV" ]]; then
  echo "[phase10-model-transfer-3b] missing ability gate: $ABILITY_CSV"
  echo "[phase10-model-transfer-3b] run phases/phase10_expansion/scripts/run_model_transfer_ability_smoke.sh first"
  exit 1
fi

MODEL_TRANSFER_MODEL_DIR="$MODEL_DIR" \
MODEL_TRANSFER_ABILITY_CSV="$ABILITY_CSV" \
MODEL_TRANSFER_NUM_SAMPLES="${MODEL_TRANSFER_NUM_SAMPLES:-2}" \
MODEL_TRANSFER_K_VALUES="${MODEL_TRANSFER_K_VALUES:-48 96}" \
  bash phases/phase10_expansion/scripts/run_model_transfer_smoke.sh
