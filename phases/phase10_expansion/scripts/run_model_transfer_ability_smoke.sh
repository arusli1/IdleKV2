#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

MODEL_DIR="${1:-models/Qwen2.5-3B-Instruct}"
MODEL_LABEL="$(basename "$MODEL_DIR" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_')"
NUM_SAMPLES="${MODEL_TRANSFER_ABILITY_NUM_SAMPLES:-4}"
LOG_DIR="phases/phase10_expansion/results/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/model_transfer_ability_${MODEL_LABEL}_$(date -u +%Y%m%dT%H%M%SZ).log"
OUT_CSV="phases/phase10_expansion/results/model_transfer_ability_${MODEL_LABEL}_n${NUM_SAMPLES}.csv"

echo "[phase10-model-ability] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "$LOG_PATH"
echo "[phase10-model-ability] model=${MODEL_DIR} task=clean_suite B=16384 n=${NUM_SAMPLES} K=48 full-cache/matched sanity" | tee -a "$LOG_PATH"

tmp="$(mktemp)"
.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
  --stage smoke \
  --task clean_suite \
  --num-samples "$NUM_SAMPLES" \
  --base-context-budget 16384 \
  --recency-window 128 \
  --k 48 \
  --conditions A B B_match \
  --query-scoring-mode proxy \
  --model-dir "$MODEL_DIR" \
  2>&1 | tee -a "$LOG_PATH" | tee "$tmp"

artifact="$(grep -E '(^/.*\.json$|^phases/.*\.json$)' "$tmp" | tail -n 1 || true)"
rm -f "$tmp"
if [[ -z "$artifact" ]]; then
  echo "[phase10-model-ability] could not locate artifact" | tee -a "$LOG_PATH"
  exit 1
fi

.venv/bin/python phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py \
  --artifact "$artifact" \
  --out-csv "$OUT_CSV" \
  2>&1 | tee -a "$LOG_PATH"

echo "[phase10-model-ability] wrote ${OUT_CSV}" | tee -a "$LOG_PATH"
echo "[phase10-model-ability] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOG_PATH"
