#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

LOG_DIR="phases/phase10_expansion/results/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/multiturn_hard_locked_n12_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "[phase10-multiturn-hard-n12] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "$LOG_PATH"
echo "[phase10-multiturn-hard-n12] schedule=mq_niah_8q_hard_revisit B=18432 n=12 K=48 96 exact_q" | tee -a "$LOG_PATH"

.venv/bin/python phases/phase10_expansion/scripts/run_multiturn_smoke.py \
  --schedule mq_niah_8q_hard_revisit \
  --num-samples 12 \
  --base-context-budget 18432 \
  --k 48 96 \
  --conditions Full Matched IdleKV Random-K Oldest-K StaleQ-K Gold-K \
  --query-scoring-mode exact_q \
  --output-csv phases/phase10_expansion/results/multiturn_hard_locked_rows_n12.csv \
  --summary-csv phases/phase10_expansion/results/multiturn_hard_locked_summary_n12.csv \
  --raw-json phases/phase10_expansion/results/multiturn_hard_locked_n12_raw.json \
  2>&1 | tee -a "$LOG_PATH"

echo "[phase10-multiturn-hard-n12] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOG_PATH"
