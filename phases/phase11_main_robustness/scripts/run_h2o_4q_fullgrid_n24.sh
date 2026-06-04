#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

LOG_DIR="phases/phase11_main_robustness/results/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/h2o_4q_fullgrid_n24_$(date -u +%Y%m%dT%H%M%SZ).log"
OUT_CSV="phases/phase11_main_robustness/results/h2o_4q_fullgrid_n24.csv"

echo "[phase11-h2o-4q-n24] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "$LOG_PATH"
echo "[phase11-h2o-4q-n24] task=clean_suite B=16384 n=24 K=8 16 24 32 48 64 80 96 128 exact_q gold_spans" | tee -a "$LOG_PATH"

tmp="$(mktemp)"
.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
  --stage full \
  --task clean_suite \
  --num-samples 24 \
  --base-context-budget 16384 \
  --recency-window 128 \
  --k 8 16 24 32 48 64 80 96 128 \
  --conditions A B B_match Random-K Oldest-K IdleKV Oracle-K \
  --query-scoring-mode exact_q \
  --oracle-mode gold_spans \
  --initial-compressor h2o \
  2>&1 | tee -a "$LOG_PATH" | tee "$tmp"

artifact="$(grep -E '(^/.*\.json$|^phases/.*\.json$)' "$tmp" | tail -n 1 || true)"
rm -f "$tmp"
if [[ -z "$artifact" ]]; then
  echo "[phase11-h2o-4q-n24] could not locate run artifact" | tee -a "$LOG_PATH"
  exit 1
fi

.venv/bin/python phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py \
  --artifact "$artifact" \
  --out-csv "$OUT_CSV" \
  2>&1 | tee -a "$LOG_PATH"

echo "[phase11-h2o-4q-n24] wrote ${OUT_CSV}" | tee -a "$LOG_PATH"
echo "[phase11-h2o-4q-n24] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOG_PATH"

