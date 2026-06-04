#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

LOG="phases/phase10_expansion/results/logs/query_count_smokes.log"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

artifacts=()

run_suite() {
  local task="$1"
  local budget="$2"
  local samples="$3"
  local tmp
  tmp="$(mktemp)"
  echo "[phase10] task=$task B=$budget n=$samples K=16 48 96 128" | tee -a "$LOG"
  .venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
    --stage smoke \
    --task "$task" \
    --num-samples "$samples" \
    --base-context-budget "$budget" \
    --recency-window 128 \
    --query-scoring-mode exact_q \
    --oracle-mode gold_spans \
    --k 16 48 96 128 \
    --conditions A B B_match IdleKV Oracle-K \
    2>&1 | tee -a "$LOG" | tee "$tmp"
  local artifact
  artifact="$(grep -E '(^/.*\.json$|^phases/.*\.json$)' "$tmp" | tail -n 1 || true)"
  rm -f "$tmp"
  if [[ -z "$artifact" ]]; then
    echo "[phase10] could not locate artifact for task=$task budget=$budget" | tee -a "$LOG"
    exit 1
  fi
  artifacts+=("$artifact")
}

run_suite mq_niah_2q_clean_suite 8192 2
run_suite mq_niah_2q_clean_suite 12288 2
run_suite mq_niah_2q_clean_suite 14336 2

run_suite mq_niah_3q_clean_suite 12288 2
run_suite mq_niah_3q_clean_suite 14336 2
run_suite mq_niah_3q_clean_suite 16384 2

run_suite mq_niah_8q_clean_suite 18432 2
run_suite mq_niah_8q_clean_suite 24576 2
run_suite mq_niah_8q_clean_suite 28672 2

summary_args=()
for artifact in "${artifacts[@]}"; do
  summary_args+=(--artifact "$artifact")
done

.venv/bin/python phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py \
  "${summary_args[@]}" \
  --out-csv phases/phase10_expansion/results/query_count_smoke_n2.csv \
  2>&1 | tee -a "$LOG"

echo "[phase10] query-count smokes complete" | tee -a "$LOG"
