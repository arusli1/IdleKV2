#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

MODEL_DIR="${MODEL_TRANSFER_MODEL_DIR:-models/Llama-3.1-8B-Instruct}"
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[phase13-llama-6q-locked] model dir not found: ${MODEL_DIR}" >&2
  exit 1
fi

NUM_SAMPLES="${PHASE13_LLAMA_6Q_LOCKED_N:-12}"
BASE_CONTEXT_BUDGET="${PHASE13_LLAMA_6Q_LOCKED_B:-18432}"
K_VALUES=(${PHASE13_LLAMA_6Q_LOCKED_K_VALUES:-32 64 96 128})
K_LABEL="$(IFS=-; echo "${K_VALUES[*]}")"
LOG_DIR="phases/phase13_iteration_framework/results/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/llama31_8b_6q_locked_n${NUM_SAMPLES}_b${BASE_CONTEXT_BUDGET}_k${K_LABEL}_$(date -u +%Y%m%dT%H%M%SZ).log"
OUT_CSV="phases/phase13_iteration_framework/results/llama31_8b_6q_locked_n${NUM_SAMPLES}_b${BASE_CONTEXT_BUDGET}_k${K_LABEL}.csv"

echo "[phase13-llama-6q-locked] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "$LOG_PATH"
echo "[phase13-llama-6q-locked] model=${MODEL_DIR} task=mq_niah_6q_clean_suite B=${BASE_CONTEXT_BUDGET} n=${NUM_SAMPLES} K=${K_VALUES[*]} exact_q gold_spans" | tee -a "$LOG_PATH"

tmp="$(mktemp)"
.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
  --stage full \
  --task mq_niah_6q_clean_suite \
  --num-samples "$NUM_SAMPLES" \
  --model-dir "$MODEL_DIR" \
  --base-context-budget "$BASE_CONTEXT_BUDGET" \
  --recency-window 128 \
  --k "${K_VALUES[@]}" \
  --conditions A B B_match Random-K Oldest-K IdleKV Oracle-K \
  --query-scoring-mode exact_q \
  --oracle-mode gold_spans \
  2>&1 | tee -a "$LOG_PATH" | tee "$tmp"

artifact="$(grep -E '(^/.*\.json$|^phases/.*\.json$)' "$tmp" | tail -n 1 || true)"
rm -f "$tmp"
if [[ -z "$artifact" ]]; then
  echo "[phase13-llama-6q-locked] could not locate run artifact" | tee -a "$LOG_PATH"
  exit 1
fi

.venv/bin/python phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py \
  --artifact "$artifact" \
  --out-csv "$OUT_CSV" \
  2>&1 | tee -a "$LOG_PATH"

.venv/bin/python phases/phase13_iteration_framework/scripts/audit_live_branches.py \
  --json \
  2>&1 | tee -a "$LOG_PATH"

echo "[phase13-llama-6q-locked] wrote ${OUT_CSV}" | tee -a "$LOG_PATH"
echo "[phase13-llama-6q-locked] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOG_PATH"
