#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="phases/phase10_expansion/results/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/h2o_compressor_locked_n12_$(date -u +%Y%m%dT%H%M%SZ).log"
OUT_CSV="phases/phase10_expansion/results/h2o_compressor_locked_n12.csv"

echo "[phase10-h2o-n12] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "${LOG_PATH}"
echo "[phase10-h2o-n12] task=clean_suite B=16384 n=12 K=48 96 exact_q gold_spans" | tee -a "${LOG_PATH}"

tmp="$(mktemp)"
.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
  --stage full \
  --task clean_suite \
  --num-samples 12 \
  --base-context-budget 16384 \
  --recency-window 128 \
  --k 48 96 \
  --conditions A B B_match Random-K Oldest-K IdleKV Oracle-K \
  --query-scoring-mode exact_q \
  --oracle-mode gold_spans \
  --initial-compressor h2o \
  2>&1 | tee -a "${LOG_PATH}" | tee "$tmp"

artifact="$(grep -E '(^/.*\.json$|^phases/.*\.json$)' "$tmp" | tail -n 1 || true)"
rm -f "$tmp"
if [[ -z "$artifact" ]]; then
  echo "[phase10-h2o-n12] could not locate artifact" | tee -a "${LOG_PATH}"
  exit 1
fi

.venv/bin/python phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py \
  --artifact "$artifact" \
  --out-csv "$OUT_CSV" \
  2>&1 | tee -a "${LOG_PATH}"

echo "[phase10-h2o-n12] wrote ${OUT_CSV}" | tee -a "${LOG_PATH}"
echo "[phase10-h2o-n12] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "${LOG_PATH}"
