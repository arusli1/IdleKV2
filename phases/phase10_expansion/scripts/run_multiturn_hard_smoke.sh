#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

LOG_DIR="phases/phase10_expansion/results/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/multiturn_hard_smoke_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "[phase10-multiturn-hard-smoke] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "$LOG_PATH"
echo "[phase10-multiturn-hard-smoke] schedule=mq_niah_8q_hard_revisit B=18432 n=1 K=48 96 exact_q" | tee -a "$LOG_PATH"

.venv/bin/python phases/phase10_expansion/scripts/run_multiturn_smoke.py \
  --schedule mq_niah_8q_hard_revisit \
  --num-samples 1 \
  --base-context-budget 18432 \
  --k 48 96 \
  --conditions Full Matched IdleKV Random-K Oldest-K StaleQ-K Gold-K \
  --query-scoring-mode exact_q \
  --output-csv phases/phase10_expansion/results/multiturn_hard_smoke_rows_n1.csv \
  --summary-csv phases/phase10_expansion/results/multiturn_hard_smoke_summary_n1.csv \
  --raw-json phases/phase10_expansion/results/multiturn_hard_smoke_n1_raw.json \
  2>&1 | tee -a "$LOG_PATH"

echo "[phase10-multiturn-hard-smoke] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOG_PATH"
