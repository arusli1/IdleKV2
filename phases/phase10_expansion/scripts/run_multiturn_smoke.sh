#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

LOG_DIR="phases/phase10_expansion/results/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/multiturn_smoke_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "[phase10-multiturn-smoke] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "$LOG_PATH"
echo "[phase10-multiturn-smoke] schedule=mq_niah_8q_shift_revisit B=18432 n=1 K=96 exact_q" | tee -a "$LOG_PATH"

.venv/bin/python phases/phase10_expansion/scripts/run_multiturn_smoke.py \
  --schedule mq_niah_8q_shift_revisit \
  --num-samples 1 \
  --base-context-budget 18432 \
  --k 96 \
  --conditions Full Matched IdleKV Random-K Oldest-K StaleQ-K Gold-K \
  --query-scoring-mode exact_q \
  2>&1 | tee -a "$LOG_PATH"

echo "[phase10-multiturn-smoke] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOG_PATH"
