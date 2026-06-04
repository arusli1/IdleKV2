#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

BASE_CONTEXT_BUDGET="${BASE_CONTEXT_BUDGET:-16384}"
LOG_DIR="phases/phase12_policy_breadth/results/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/streamingllm_4q_fullgrid_n24_b${BASE_CONTEXT_BUDGET}_$(date -u +%Y%m%dT%H%M%SZ).log"
OUT_CSV="phases/phase12_policy_breadth/results/streamingllm_4q_fullgrid_n24_b${BASE_CONTEXT_BUDGET}.csv"
K_VALUES=(8 16 24 32 48 64 80 96 128)

echo "[phase12-streamingllm-fullgrid] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "${LOG_PATH}"
echo "[phase12-streamingllm-fullgrid] task=clean_suite B=${BASE_CONTEXT_BUDGET} n=24 K=${K_VALUES[*]} exact_q gold_spans" | tee -a "${LOG_PATH}"

tmp="$(mktemp)"
.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
  --stage full \
  --task clean_suite \
  --num-samples 24 \
  --base-context-budget "${BASE_CONTEXT_BUDGET}" \
  --recency-window 128 \
  --k "${K_VALUES[@]}" \
  --conditions A B B_match Random-K Oldest-K IdleKV Oracle-K \
  --query-scoring-mode exact_q \
  --oracle-mode gold_spans \
  --initial-compressor streaming_llm \
  2>&1 | tee -a "${LOG_PATH}" | tee "$tmp"

artifact="$(grep -E '(^/.*\.json$|^phases/.*\.json$)' "$tmp" | tail -n 1 || true)"
rm -f "$tmp"
if [[ -z "${artifact}" ]]; then
  echo "[phase12-streamingllm-fullgrid] could not locate artifact" | tee -a "${LOG_PATH}"
  exit 1
fi

.venv/bin/python phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py \
  --artifact "${artifact}" \
  --out-csv "${OUT_CSV}" \
  2>&1 | tee -a "${LOG_PATH}"

echo "[phase12-streamingllm-fullgrid] wrote ${OUT_CSV}" | tee -a "${LOG_PATH}"
.venv/bin/python phases/phase11_main_robustness/scripts/recommend_main_candidate.py \
  --summary-csv "${OUT_CSV}" \
  2>&1 | tee -a "${LOG_PATH}"
echo "[phase12-streamingllm-fullgrid] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "${LOG_PATH}"
