#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

RESULTS_DIR="phases/phase9_experiment_deepening/results"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

K_VALUES=("48" "96")

artifact_from_summary() {
  local summary_csv="$1"
  .venv/bin/python - "$summary_csv" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(newline="", encoding="utf-8") as handle:
    first = next(csv.DictReader(handle), None)
if first is None or not first.get("artifact"):
    raise SystemExit(f"summary has no artifact value: {path}")
print(first["artifact"])
PY
}

summarize_exact_reference() {
  local label="$1"
  local artifact="$2"
  .venv/bin/python phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py \
    --artifact "${artifact}" \
    --out-csv "${RESULTS_DIR}/phase9_proxy_exact_${label}_reference.csv"
}

run_proxy_cell() {
  local label="$1"
  local task="$2"
  local base_budget="$3"
  local num_samples="$4"
  local exact_artifact="$5"
  local proxy_csv="${RESULTS_DIR}/phase9_proxy_${label}_full_n${num_samples}.csv"
  local paired_csv="${RESULTS_DIR}/phase9_proxy_${label}_full_n${num_samples}_paired.csv"
  local log_path="${LOG_DIR}/phase9_proxy_${label}_full_n${num_samples}.log"

  echo "[run] ${label} proxy full n=${num_samples} K=${K_VALUES[*]}"
  .venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
    --stage full \
    --task "${task}" \
    --num-samples "${num_samples}" \
    --base-context-budget "${base_budget}" \
    --recency-window 128 \
    --query-scoring-mode proxy \
    --oracle-mode gold_spans \
    --k "${K_VALUES[@]}" \
    --conditions A B B_match IdleKV | tee "${log_path}"

  local proxy_artifact
  proxy_artifact="$(grep -E '^/.+\.json$' "${log_path}" | tail -n 1)"
  if [[ -z "${proxy_artifact}" || ! -f "${proxy_artifact}" ]]; then
    echo "Could not find proxy artifact in ${log_path}" >&2
    exit 1
  fi

  .venv/bin/python phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py \
    --artifact "${proxy_artifact}" \
    --gate proxy \
    --out-csv "${proxy_csv}"

  .venv/bin/python phases/phase9_experiment_deepening/scripts/proxy_paired_bootstrap.py \
    --exact-artifact "${exact_artifact}" \
    --proxy-artifact "${proxy_artifact}" \
    --k "${K_VALUES[@]}" \
    --out-csv "${paired_csv}"
}

EXACT_2Q="$(artifact_from_summary phases/phase10_expansion/results/mq_niah_2q_frontier_n100.csv)"
EXACT_8Q="$(artifact_from_summary phases/phase10_expansion/results/mq_niah_8q_frontier_n24.csv)"

summarize_exact_reference "2q" "${EXACT_2Q}"
summarize_exact_reference "8q" "${EXACT_8Q}"
run_proxy_cell "2q" "mq_niah_2q_clean_suite" "8192" "100" "${EXACT_2Q}"
run_proxy_cell "8q" "mq_niah_8q_clean_suite" "18432" "24" "${EXACT_8Q}"

echo "[done] proxy breadth quality-latency suite"
