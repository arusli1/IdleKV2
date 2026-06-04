#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

LOG_DIR="phases/phase10_expansion/results/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/8q_full_frontier_n24_$(date -u +%Y%m%dT%H%M%SZ).log"
OUT_CSV="phases/phase10_expansion/results/mq_niah_8q_frontier_n24.csv"

echo "[phase10-8q-frontier-n24] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "$LOG_PATH"
echo "[phase10-8q-frontier-n24] task=mq_niah_8q_clean_suite B=18432 n=24 K=8 16 24 32 48 64 80 96 128" | tee -a "$LOG_PATH"

tmp="$(mktemp)"
.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
  --stage full \
  --task mq_niah_8q_clean_suite \
  --num-samples 24 \
  --base-context-budget 18432 \
  --recency-window 128 \
  --k 8 16 24 32 48 64 80 96 128 \
  --conditions A B B_match Random-K Oldest-K IdleKV Oracle-K \
  --query-scoring-mode exact_q \
  --oracle-mode gold_spans \
  2>&1 | tee -a "$LOG_PATH" | tee "$tmp"

artifact="$(grep -E '(^/.*\.json$|^phases/.*\.json$)' "$tmp" | tail -n 1 || true)"
rm -f "$tmp"
if [[ -z "$artifact" ]]; then
  echo "[phase10-8q-frontier-n24] could not locate run artifact" | tee -a "$LOG_PATH"
  exit 1
fi

.venv/bin/python phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py \
  --artifact "$artifact" \
  --out-csv "$OUT_CSV" \
  2>&1 | tee -a "$LOG_PATH"

echo "[phase10-8q-frontier-n24] wrote ${OUT_CSV}" | tee -a "$LOG_PATH"
echo "[phase10-8q-frontier-n24] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOG_PATH"
