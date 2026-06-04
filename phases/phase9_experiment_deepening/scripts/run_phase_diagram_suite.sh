#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODE="${1:-smoke}"
RESULTS_DIR="phases/phase9_experiment_deepening/results"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${LOG_DIR}"

K_VALUES=(16 48 96 128)
CONDITIONS=(A B B_match IdleKV Oracle-K)
RECENCY_WINDOW=128

case "${MODE}" in
  smoke)
    STAGE="smoke"
    NUM_SAMPLES="${PHASE9_PHASE_DIAGRAM_SMOKE_N:-1}"
    OUT_TAG="smoke_n${NUM_SAMPLES}"
    ;;
  final)
    STAGE="full"
    NUM_SAMPLES="${PHASE9_PHASE_DIAGRAM_FINAL_N:-12}"
    OUT_TAG="final_n${NUM_SAMPLES}"
    ;;
  *)
    echo "usage: $0 [smoke|final]" >&2
    exit 2
    ;;
esac

run_cell() {
  local label="$1"
  local task="$2"
  local budget="$3"
  local log_path="${LOG_DIR}/phase9_phase_diagram_${OUT_TAG}_${label}_b${budget}.log"

  echo "[phase-diagram] ${label} task=${task} B=${budget} n=${NUM_SAMPLES} K=${K_VALUES[*]}" >&2
  "${PYTHON_BIN}" phases/phase6_repair/scripts/run_phase6.py \
    --stage "${STAGE}" \
    --task "${task}" \
    --num-samples "${NUM_SAMPLES}" \
    --base-context-budget "${budget}" \
    --recency-window "${RECENCY_WINDOW}" \
    --query-scoring-mode exact_q \
    --oracle-mode gold_spans \
    --k "${K_VALUES[@]}" \
    --conditions "${CONDITIONS[@]}" | tee "${log_path}" >&2

  tail -n 1 "${log_path}"
}

summarize_task() {
  local label="$1"
  shift
  local artifacts=("$@")
  local csv_path="${RESULTS_DIR}/phase9_phase_diagram_${label}_${OUT_TAG}.csv"
  local svg_path="${RESULTS_DIR}/phase9_phase_diagram_${label}_${OUT_TAG}.svg"

  local args=()
  for artifact in "${artifacts[@]}"; do
    args+=(--artifact "${artifact}")
  done
  "${PYTHON_BIN}" phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py \
    "${args[@]}" \
    --out-csv "${csv_path}"
  "${PYTHON_BIN}" phases/phase9_experiment_deepening/scripts/plot_phase_diagram_svg.py \
    --csv "${csv_path}" \
    --out "${svg_path}" \
    --title "IdleKV ${label^^}: score gain over matched no-repair"
  echo "${csv_path}"
  echo "${svg_path}"
}

artifacts_4q=()
for budget in 14336 16384 18432; do
  artifacts_4q+=("$(run_cell "4q" "clean_suite" "${budget}")")
done

artifacts_6q=()
for budget in 12288 18432 24576; do
  artifacts_6q+=("$(run_cell "6q" "mq_niah_6q_clean_suite" "${budget}")")
done

summarize_task "4q" "${artifacts_4q[@]}"
summarize_task "6q" "${artifacts_6q[@]}"
