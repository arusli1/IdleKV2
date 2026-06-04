#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

LOG_DIR="phases/phase10_expansion/results/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/specificity_smoke_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "[phase10-specificity] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "$LOG"
echo "[phase10-specificity] task=clean_suite B=16384 n=1 K=48 96" | tee -a "$LOG"

.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
  --stage smoke \
  --task clean_suite \
  --num-samples 1 \
  --base-context-budget 16384 \
  --recency-window 128 \
  --query-scoring-mode exact_q \
  --oracle-mode gold_spans \
  --wrong-query-mode donor_q2 \
  --k 48 96 \
  --conditions A B B_match StaleQ-K WrongQ-K Refresh-K IdleKV Oracle-K \
  2>&1 | tee -a "$LOG"

artifact="$(
  find phases/phase6_repair/results/smoke -type f \
    -name 'clean_suite_b16384_r128_qexact_q_ogold_spans_wqdonor_q2_n1_k48-96_ca-b-bmatch-staleqk-wrongqk-refreshk-idlekv-oraclek.json' \
    | sort | tail -n 1
)"

if [[ -z "${artifact}" ]]; then
  echo "[phase10-specificity] could not locate specificity artifact" | tee -a "$LOG"
  exit 1
fi

.venv/bin/python phases/phase10_expansion/scripts/summarize_specificity.py \
  --artifact "$artifact" \
  --out-csv phases/phase10_expansion/results/specificity_smoke_n1.csv \
  2>&1 | tee -a "$LOG"

.venv/bin/python phases/phase10_expansion/scripts/recommend_specificity_next.py \
  --summary-csv phases/phase10_expansion/results/specificity_smoke_n1.csv \
  2>&1 | tee -a "$LOG"

echo "[phase10-specificity] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOG"
