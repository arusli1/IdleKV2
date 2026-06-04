#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

LOG_DIR="phases/phase10_expansion/results/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/query_count_locked_$(date -u +%Y%m%dT%H%M%SZ).log"
OUT_CSV="phases/phase10_expansion/results/query_count_locked_n12.csv"

echo "[phase10-query-count-locked] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "$LOG_PATH"
echo "[phase10-query-count-locked] tasks=2Q/3Q/8Q n=12 K=48 96 controls=random/oldest" | tee -a "$LOG_PATH"

artifacts=()

run_suite() {
  local task="$1"
  local budget="$2"
  local tmp
  tmp="$(mktemp)"
  echo "[phase10-query-count-locked] task=${task} B=${budget}" | tee -a "$LOG_PATH"
  .venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
    --stage full \
    --task "$task" \
    --num-samples 12 \
    --base-context-budget "$budget" \
    --recency-window 128 \
    --k 48 96 \
    --conditions A B B_match Random-K Oldest-K IdleKV Oracle-K \
    --query-scoring-mode exact_q \
    --oracle-mode gold_spans \
    2>&1 | tee -a "$LOG_PATH" | tee "$tmp"
  local artifact
  artifact="$(grep -E '(^/.*\.json$|^phases/.*\.json$)' "$tmp" | tail -n 1 || true)"
  rm -f "$tmp"
  if [[ -z "$artifact" ]]; then
    echo "[phase10-query-count-locked] could not locate artifact for task=${task} budget=${budget}" | tee -a "$LOG_PATH"
    exit 1
  fi
  artifacts+=("$artifact")
}

run_suite mq_niah_2q_clean_suite 8192
run_suite mq_niah_3q_clean_suite 14336
run_suite mq_niah_8q_clean_suite 18432

summary_args=()
for artifact in "${artifacts[@]}"; do
  summary_args+=(--artifact "$artifact")
done

.venv/bin/python phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py \
  "${summary_args[@]}" \
  --out-csv "$OUT_CSV" \
  2>&1 | tee -a "$LOG_PATH"

echo "[phase10-query-count-locked] wrote ${OUT_CSV}" | tee -a "$LOG_PATH"
echo "[phase10-query-count-locked] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOG_PATH"
