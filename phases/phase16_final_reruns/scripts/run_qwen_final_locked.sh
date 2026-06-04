#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

RESULTS_DIR="phases/phase16_final_reruns/results"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

NUM_SAMPLES="${PHASE16_QWEN_FINAL_N:-24}"
LOG_PATH="${LOG_DIR}/qwen_final_locked_$(date -u +%Y%m%dT%H%M%SZ).log"

run_one() {
  local name="$1"
  local task="$2"
  local base_budget="$3"
  shift 3
  local k_values=("$@")
  local out_csv="${RESULTS_DIR}/${name}_n${NUM_SAMPLES}_b${base_budget}.csv"

  echo "[phase16-qwen-final] ${name} task=${task} B=${base_budget} n=${NUM_SAMPLES} K=${k_values[*]}" | tee -a "${LOG_PATH}"
  tmp="$(mktemp)"
  .venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
    --stage full \
    --task "${task}" \
    --num-samples "${NUM_SAMPLES}" \
    --base-context-budget "${base_budget}" \
    --recency-window 128 \
    --query-scoring-mode exact_q \
    --oracle-mode gold_spans \
    --k "${k_values[@]}" \
    --conditions A B B_match Random-K Oldest-K IdleKV Oracle-K \
    2>&1 | tee -a "${LOG_PATH}" | tee "${tmp}"

  artifact="$(grep -E '(^/.*\.json$|^phases/.*\.json$)' "${tmp}" | tail -n 1 || true)"
  rm -f "${tmp}"
  if [[ -z "${artifact}" ]]; then
    echo "[phase16-qwen-final] could not locate artifact for ${name}" | tee -a "${LOG_PATH}"
    exit 1
  fi

  .venv/bin/python phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py \
    --artifact "${artifact}" \
    --out-csv "${out_csv}" \
    2>&1 | tee -a "${LOG_PATH}"
  echo "[phase16-qwen-final] wrote ${out_csv}" | tee -a "${LOG_PATH}"
}

echo "[phase16-qwen-final] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "${LOG_PATH}"
run_one qwen_4q_final clean_suite 16384 8 16 24 32 48 64 80 96 128
run_one qwen_6q_final mq_niah_6q_clean_suite 18432 16 24 32 48 64 80 96 128
run_one qwen_8q_final mq_niah_8q_clean_suite 18432 16 24 32 48 64 80 96 128
echo "[phase16-qwen-final] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "${LOG_PATH}"

