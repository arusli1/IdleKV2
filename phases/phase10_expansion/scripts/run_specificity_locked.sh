#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

LOG_DIR="phases/phase10_expansion/results/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/specificity_locked_$(date -u +%Y%m%dT%H%M%SZ).log"

SPECIFICITY_NUM_SAMPLES="${SPECIFICITY_NUM_SAMPLES:-24}"
SPECIFICITY_BASE_BUDGET="${SPECIFICITY_BASE_BUDGET:-16384}"
SPECIFICITY_K_VALUES="${SPECIFICITY_K_VALUES:-48}"

read -r -a k_values <<< "$SPECIFICITY_K_VALUES"

echo "[phase10-specificity-locked] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "$LOG"
echo "[phase10-specificity-locked] task=clean_suite B=${SPECIFICITY_BASE_BUDGET} n=${SPECIFICITY_NUM_SAMPLES} K=${SPECIFICITY_K_VALUES}" | tee -a "$LOG"

.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
  --stage full \
  --task clean_suite \
  --num-samples "$SPECIFICITY_NUM_SAMPLES" \
  --base-context-budget "$SPECIFICITY_BASE_BUDGET" \
  --recency-window 128 \
  --query-scoring-mode exact_q \
  --oracle-mode gold_spans \
  --wrong-query-mode donor_q2 \
  --k "${k_values[@]}" \
  --conditions A B B_match StaleQ-K WrongQ-K Refresh-K IdleKV Oracle-K \
  2>&1 | tee -a "$LOG"

k_label="$(IFS=-; echo "${k_values[*]}")"
artifact="$(
  find phases/phase6_repair/results/full -type f \
    -name "clean_suite_b${SPECIFICITY_BASE_BUDGET}_r128_qexact_q_ogold_spans_wqdonor_q2_n${SPECIFICITY_NUM_SAMPLES}_k${k_label}_ca-b-bmatch-staleqk-wrongqk-refreshk-idlekv-oraclek.json" \
    | sort | tail -n 1
)"

if [[ -z "${artifact}" ]]; then
  echo "[phase10-specificity-locked] could not locate specificity artifact" | tee -a "$LOG"
  exit 1
fi

out_csv="phases/phase10_expansion/results/specificity_locked_n${SPECIFICITY_NUM_SAMPLES}_k${k_label}.csv"
.venv/bin/python phases/phase10_expansion/scripts/summarize_specificity.py \
  --artifact "$artifact" \
  --out-csv "$out_csv" \
  2>&1 | tee -a "$LOG"

.venv/bin/python phases/phase10_expansion/scripts/recommend_specificity_next.py \
  --summary-csv "$out_csv" \
  2>&1 | tee -a "$LOG"

echo "[phase10-specificity-locked] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOG"
