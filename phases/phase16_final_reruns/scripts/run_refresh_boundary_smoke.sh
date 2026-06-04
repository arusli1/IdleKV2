#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

RESULTS_DIR="phases/phase16_final_reruns/results"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

NUM_SAMPLES="${PHASE16_REFRESH_SMOKE_N:-2}"
K_VALUES_TEXT="${PHASE16_REFRESH_K_VALUES:-24 48 80 96}"
read -r -a K_VALUES <<< "${K_VALUES_TEXT}"
LOG_PATH="${LOG_DIR}/refresh_boundary_smoke_$(date -u +%Y%m%dT%H%M%SZ).log"
OUT_CSV="${RESULTS_DIR}/refresh_boundary_smoke_n${NUM_SAMPLES}.csv"

echo "[phase16-refresh-smoke] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "${LOG_PATH}"
echo "[phase16-refresh-smoke] task=clean_suite B=16384 n=${NUM_SAMPLES} K=${K_VALUES[*]}" | tee -a "${LOG_PATH}"

tmp="$(mktemp)"
.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
  --stage smoke \
  --task clean_suite \
  --num-samples "${NUM_SAMPLES}" \
  --base-context-budget 16384 \
  --recency-window 128 \
  --query-scoring-mode exact_q \
  --oracle-mode gold_spans \
  --wrong-query-mode donor_q2 \
  --k "${K_VALUES[@]}" \
  --conditions A B B_match StaleQ-K WrongQ-K Refresh-K IdleKV Oracle-K \
  2>&1 | tee -a "${LOG_PATH}" | tee "${tmp}"

artifact="$(grep -E '(^/.*\.json$|^phases/.*\.json$)' "${tmp}" | tail -n 1 || true)"
rm -f "${tmp}"
if [[ -z "${artifact}" ]]; then
  echo "[phase16-refresh-smoke] could not locate artifact" | tee -a "${LOG_PATH}"
  exit 1
fi

.venv/bin/python phases/phase10_expansion/scripts/summarize_specificity.py \
  --artifact "${artifact}" \
  --out-csv "${OUT_CSV}" \
  2>&1 | tee -a "${LOG_PATH}"

.venv/bin/python phases/phase14_critical_flaw_closure/scripts/evaluate_phase14_smokes.py \
  --kind refresh \
  --summary-csv "${OUT_CSV}" \
  2>&1 | tee -a "${LOG_PATH}" || true

echo "[phase16-refresh-smoke] wrote ${OUT_CSV}" | tee -a "${LOG_PATH}"
echo "[phase16-refresh-smoke] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "${LOG_PATH}"

