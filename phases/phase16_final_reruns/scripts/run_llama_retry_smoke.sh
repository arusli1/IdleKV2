#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

RESULTS_DIR="phases/phase16_final_reruns/results"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

MODEL_DIR="${PHASE16_LLAMA_MODEL_DIR:-models/Llama-3.1-8B-Instruct}"
NUM_SAMPLES="${PHASE16_LLAMA_RETRY_SMOKE_N:-2}"
BASE_CONTEXT_BUDGET="${PHASE16_LLAMA_BASE_BUDGET:-16384}"
K_VALUES_TEXT="${PHASE16_LLAMA_K_VALUES:-24 48 96}"
read -r -a K_VALUES <<< "${K_VALUES_TEXT}"
LOG_PATH="${LOG_DIR}/llama_retry_smoke_$(date -u +%Y%m%dT%H%M%SZ).log"
OUT_CSV="${RESULTS_DIR}/llama_retry_smoke_n${NUM_SAMPLES}_b${BASE_CONTEXT_BUDGET}.csv"

echo "[phase16-llama-retry-smoke] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "${LOG_PATH}"
echo "[phase16-llama-retry-smoke] model=${MODEL_DIR} task=mq_niah_6q_clean_suite B=${BASE_CONTEXT_BUDGET} n=${NUM_SAMPLES} K=${K_VALUES[*]}" | tee -a "${LOG_PATH}"

tmp="$(mktemp)"
.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
  --stage smoke \
  --task mq_niah_6q_clean_suite \
  --model-dir "${MODEL_DIR}" \
  --num-samples "${NUM_SAMPLES}" \
  --base-context-budget "${BASE_CONTEXT_BUDGET}" \
  --recency-window 128 \
  --query-scoring-mode exact_q \
  --oracle-mode gold_spans \
  --k "${K_VALUES[@]}" \
  --conditions A B B_match Random-K Oldest-K IdleKV Oracle-K \
  2>&1 | tee -a "${LOG_PATH}" | tee "${tmp}"

artifact="$(grep -E '(^/.*\.json$|^phases/.*\.json$)' "${tmp}" | tail -n 1 || true)"
rm -f "${tmp}"
if [[ -z "${artifact}" ]]; then
  echo "[phase16-llama-retry-smoke] could not locate artifact" | tee -a "${LOG_PATH}"
  exit 1
fi

.venv/bin/python phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py \
  --artifact "${artifact}" \
  --out-csv "${OUT_CSV}" \
  2>&1 | tee -a "${LOG_PATH}"

.venv/bin/python phases/phase16_final_reruns/scripts/evaluate_phase16_smokes.py \
  --summary-csv "${OUT_CSV}" \
  2>&1 | tee -a "${LOG_PATH}" || true

echo "[phase16-llama-retry-smoke] wrote ${OUT_CSV}" | tee -a "${LOG_PATH}"
echo "[phase16-llama-retry-smoke] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "${LOG_PATH}"

