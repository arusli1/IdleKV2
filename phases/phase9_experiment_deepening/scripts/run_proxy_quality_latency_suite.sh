#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

STAGE="full"
NUM_SAMPLES="100"
K_VALUES=("48" "96")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --num-samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --k)
      shift
      K_VALUES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        K_VALUES+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ${#K_VALUES[@]} -eq 0 ]]; then
  echo "At least one --k value is required." >&2
  exit 2
fi

RESULTS_DIR="phases/phase9_experiment_deepening/results"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

EXACT_4Q="phases/phase6_repair/results/full/clean_suite_b16384_r128_qexact_q_ogold_spans_n100_k8-16-24-32-48-64-80-96-128_ca-b-bmatch-idlekv-randomk-oldestk-oraclek.json"
EXACT_6Q="phases/phase6_repair/results/full/mq_niah_6q_clean_suite_b18432_r128_qexact_q_ogold_spans_n100_k8-16-24-32-48-64-80-96-128_ca-b-bmatch-idlekv-randomk-oldestk-oraclek.json"

summarize_exact_reference() {
  local label="$1"
  local artifact="$2"
  local out_csv="${RESULTS_DIR}/phase9_proxy_exact_${label}_reference.csv"
  .venv/bin/python phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py \
    --artifact "${artifact}" \
    --out-csv "${out_csv}"
}

run_proxy_cell() {
  local label="$1"
  local task="$2"
  local base_budget="$3"
  local exact_artifact="$4"
  local exact_csv="${RESULTS_DIR}/phase9_proxy_exact_${label}_reference.csv"
  local proxy_csv="${RESULTS_DIR}/phase9_proxy_${label}_${STAGE}_n${NUM_SAMPLES}.csv"
  local paired_csv="${RESULTS_DIR}/phase9_proxy_${label}_${STAGE}_n${NUM_SAMPLES}_paired.csv"
  local svg_path="${RESULTS_DIR}/phase9_quality_latency_${label}_${STAGE}_n${NUM_SAMPLES}.svg"
  local log_path="${LOG_DIR}/phase9_proxy_${label}_${STAGE}_n${NUM_SAMPLES}.log"

  echo "[run] ${label} proxy ${STAGE} n=${NUM_SAMPLES} K=${K_VALUES[*]}"
  .venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
    --stage "${STAGE}" \
    --task "${task}" \
    --num-samples "${NUM_SAMPLES}" \
    --base-context-budget "${base_budget}" \
    --recency-window 128 \
    --query-scoring-mode proxy \
    --oracle-mode gold_spans \
    --k "${K_VALUES[@]}" \
    --conditions A B B_match IdleKV | tee "${log_path}"

  local artifact_path
  artifact_path="$(grep -E '^/.+\.json$' "${log_path}" | tail -n 1)"
  if [[ -z "${artifact_path}" ]]; then
    echo "Could not find artifact path in ${log_path}" >&2
    exit 1
  fi

  echo "[summarize] ${artifact_path}"
  .venv/bin/python phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py \
    --artifact "${artifact_path}" \
    --gate proxy \
    --out-csv "${proxy_csv}"

  .venv/bin/python phases/phase9_experiment_deepening/scripts/proxy_paired_bootstrap.py \
    --exact-artifact "${exact_artifact}" \
    --proxy-artifact "${artifact_path}" \
    --k "${K_VALUES[@]}" \
    --out-csv "${paired_csv}"

  if [[ -f "${exact_csv}" ]]; then
    .venv/bin/python phases/phase9_experiment_deepening/scripts/plot_quality_latency_svg.py \
      --exact-csv "${exact_csv}" \
      --proxy-csv "${proxy_csv}" \
      --k 96 \
      --out "${svg_path}" \
      --title "${label^^} proxy quality-latency"
    set +e
    .venv/bin/python phases/phase9_experiment_deepening/scripts/check_proxy_quality_latency.py \
      --exact-csv "${exact_csv}" \
      --proxy-csv "${proxy_csv}" \
      --headline-k 96 \
      --guardrail-k 48
    local gate_status=$?
    set -e
    if [[ ${gate_status} -ne 0 ]]; then
      echo "[warn] ${label} proxy quality-latency promotion gate failed; inspect ${proxy_csv}" >&2
    fi
  fi
}

summarize_exact_reference "4q" "${EXACT_4Q}"
summarize_exact_reference "6q" "${EXACT_6Q}"
run_proxy_cell "4q" "clean_suite" "16384" "${EXACT_4Q}"
run_proxy_cell "6q" "mq_niah_6q_clean_suite" "18432" "${EXACT_6Q}"

echo "[done] proxy quality-latency suite"
