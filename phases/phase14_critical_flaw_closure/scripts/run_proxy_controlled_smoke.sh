#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

RESULTS_DIR="phases/phase14_critical_flaw_closure/results"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

NUM_SAMPLES="${PHASE14_PROXY_SMOKE_N:-4}"
K_VALUES_TEXT="${PHASE14_PROXY_K_VALUES:-48 96 128}"
read -r -a K_VALUES <<< "${K_VALUES_TEXT}"
LOG_PATH="${LOG_DIR}/proxy_controlled_smoke_$(date -u +%Y%m%dT%H%M%SZ).log"
OUT_CSV="${RESULTS_DIR}/proxy_controlled_smoke_n${NUM_SAMPLES}.csv"

echo "[phase14-proxy-smoke] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "${LOG_PATH}"
echo "[phase14-proxy-smoke] n=${NUM_SAMPLES} K=${K_VALUES[*]} conditions=A/B/B_match/Random-K/Oldest-K/IdleKV/Oracle-K" | tee -a "${LOG_PATH}"

artifacts=()

run_cell() {
  local label="$1"
  local task="$2"
  local base_budget="$3"
  local tmp
  tmp="$(mktemp)"
  echo "[phase14-proxy-smoke] ${label} task=${task} B=${base_budget}" | tee -a "${LOG_PATH}"
  .venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
    --stage smoke \
    --task "${task}" \
    --num-samples "${NUM_SAMPLES}" \
    --base-context-budget "${base_budget}" \
    --recency-window 128 \
    --query-scoring-mode proxy \
    --oracle-mode gold_spans \
    --k "${K_VALUES[@]}" \
    --conditions A B B_match Random-K Oldest-K IdleKV Oracle-K \
    2>&1 | tee -a "${LOG_PATH}" | tee "${tmp}"

  local artifact
  artifact="$(grep -E '(^/.*\.json$|^phases/.*\.json$)' "${tmp}" | tail -n 1 || true)"
  rm -f "${tmp}"
  if [[ -z "${artifact}" ]]; then
    echo "[phase14-proxy-smoke] could not locate artifact for ${label}" | tee -a "${LOG_PATH}"
    exit 1
  fi
  artifacts+=("${artifact}")
}

run_cell "4q" "clean_suite" "16384"
run_cell "6q" "mq_niah_6q_clean_suite" "18432"

summary_args=()
for artifact in "${artifacts[@]}"; do
  summary_args+=(--artifact "${artifact}")
done

.venv/bin/python phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py \
  "${summary_args[@]}" \
  --out-csv "${OUT_CSV}" \
  2>&1 | tee -a "${LOG_PATH}"

.venv/bin/python phases/phase14_critical_flaw_closure/scripts/evaluate_proxy_controlled_smoke.py \
  --summary-csv "${OUT_CSV}" \
  2>&1 | tee -a "${LOG_PATH}" || true

echo "[phase14-proxy-smoke] wrote ${OUT_CSV}" | tee -a "${LOG_PATH}"
echo "[phase14-proxy-smoke] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "${LOG_PATH}"
