#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

NUM_SAMPLES="${PHASE13_MULTITURN_KCAL_N:-2}"
BASE_CONTEXT_BUDGET="${PHASE13_MULTITURN_B:-18432}"
K_VALUES=(${PHASE13_MULTITURN_K_VALUES:-64 80 96})
CONDITIONS=(${PHASE13_MULTITURN_CONDITIONS:-Full Matched IdleKV CurrentQOnly-K Random-K Oldest-K StaleQ-K StaleQOnly-K Gold-K})
LOG_DIR="phases/phase13_iteration_framework/results/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/multiturn_hard_kcal_smoke_n${NUM_SAMPLES}_$(date -u +%Y%m%dT%H%M%SZ).log"
OUT_ROWS="phases/phase13_iteration_framework/results/multiturn_hard_kcal_smoke_rows_n${NUM_SAMPLES}.csv"
OUT_SUMMARY="phases/phase13_iteration_framework/results/multiturn_hard_kcal_smoke_summary_n${NUM_SAMPLES}.csv"
OUT_RAW="phases/phase13_iteration_framework/results/multiturn_hard_kcal_smoke_n${NUM_SAMPLES}_raw.json"

echo "[phase13-multiturn-kcal] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "$LOG_PATH"
echo "[phase13-multiturn-kcal] schedule=mq_niah_8q_hard_revisit B=${BASE_CONTEXT_BUDGET} n=${NUM_SAMPLES} K=${K_VALUES[*]} conditions=${CONDITIONS[*]} exact_q" | tee -a "$LOG_PATH"

.venv/bin/python phases/phase10_expansion/scripts/run_multiturn_smoke.py \
  --schedule mq_niah_8q_hard_revisit \
  --num-samples "$NUM_SAMPLES" \
  --base-context-budget "$BASE_CONTEXT_BUDGET" \
  --k "${K_VALUES[@]}" \
  --conditions "${CONDITIONS[@]}" \
  --query-scoring-mode exact_q \
  --output-csv "$OUT_ROWS" \
  --summary-csv "$OUT_SUMMARY" \
  --raw-json "$OUT_RAW" \
  2>&1 | tee -a "$LOG_PATH"

.venv/bin/python phases/phase13_iteration_framework/scripts/audit_live_branches.py \
  --json \
  2>&1 | tee -a "$LOG_PATH"

echo "[phase13-multiturn-kcal] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOG_PATH"
