#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
RUNNER="$ROOT/phases/phase6_repair/scripts/run_phase6.py"

STAGE="${STAGE:-full}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
RECENCY_WINDOW="${RECENCY_WINDOW:-128}"
DATASET_SEED_OFFSET="${DATASET_SEED_OFFSET:-0}"
QUERY_SCORING_MODE="${QUERY_SCORING_MODE:-exact_q}"
ORACLE_MODE="${ORACLE_MODE:-gold_spans}"

BASE_CONTEXT_BUDGET_3Q="${BASE_CONTEXT_BUDGET_3Q:-14336}"
BASE_CONTEXT_BUDGET_4Q="${BASE_CONTEXT_BUDGET_4Q:-16384}"
BASE_CONTEXT_BUDGET_6Q="${BASE_CONTEXT_BUDGET_6Q:-18432}"

RUN_LIGHT_TASK="${RUN_LIGHT_TASK:-0}"
RUN_4Q="${RUN_4Q:-0}"
RUN_6Q="${RUN_6Q:-1}"

K_VALUES_3Q=(${K_VALUES_3Q:-8 16 32 48 64})
K_VALUES_4Q=(${K_VALUES_4Q:-8 16 24 32 48 64 80 96 128})
K_VALUES_6Q=(${K_VALUES_6Q:-8 16 24 32 48 64 80 96 128})

COMMON_ARGS=(
  --stage "$STAGE"
  --num-samples "$NUM_SAMPLES"
  --conditions A B B_match IdleKV Random-K Oldest-K Oracle-K
  --recency-window "$RECENCY_WINDOW"
  --dataset-seed-offset "$DATASET_SEED_OFFSET"
  --query-scoring-mode "$QUERY_SCORING_MODE"
  --oracle-mode "$ORACLE_MODE"
)

run_task() {
  local task="$1"
  local budget="$2"
  shift 2
  local k_values=("$@")
  echo "Running $task with B_base=$budget and K={${k_values[*]}}"
  "$PYTHON_BIN" "$RUNNER" --task "$task" --base-context-budget "$budget" --k "${k_values[@]}" "${COMMON_ARGS[@]}"
}

echo "Running Phase 7 suite with:"
echo "  stage=$STAGE"
echo "  num_samples=$NUM_SAMPLES"
echo "  recency_window=$RECENCY_WINDOW"
echo "  dataset_seed_offset=$DATASET_SEED_OFFSET"
echo "  query_scoring_mode=$QUERY_SCORING_MODE"
echo "  oracle_mode=$ORACLE_MODE"
echo "  B_3q=$BASE_CONTEXT_BUDGET_3Q"
echo "  B_4q=$BASE_CONTEXT_BUDGET_4Q"
echo "  B_6q=$BASE_CONTEXT_BUDGET_6Q"
echo "  run_light_task=$RUN_LIGHT_TASK"
echo "  run_4q=$RUN_4Q"
echo "  run_6q=$RUN_6Q"
echo "  K_3q=${K_VALUES_3Q[*]}"
echo "  K_4q=${K_VALUES_4Q[*]}"
echo "  K_6q=${K_VALUES_6Q[*]}"

if [[ "$RUN_LIGHT_TASK" == "1" ]]; then
  run_task mq_niah_3q_split_3_to_12 "$BASE_CONTEXT_BUDGET_3Q" "${K_VALUES_3Q[@]}"
fi

if [[ "$RUN_4Q" == "1" ]]; then
  run_task clean_suite "$BASE_CONTEXT_BUDGET_4Q" "${K_VALUES_4Q[@]}"
fi

if [[ "$RUN_6Q" == "1" ]]; then
  run_task mq_niah_6q_clean_suite "$BASE_CONTEXT_BUDGET_6Q" "${K_VALUES_6Q[@]}"
fi
