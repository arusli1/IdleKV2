#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="phases/phase10_expansion/results/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/streamingllm_smoke_$(date -u +%Y%m%dT%H%M%SZ).log"
OUT_CSV="phases/phase10_expansion/results/streamingllm_smoke_n1.csv"

echo "[phase10-streamingllm] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "${LOG_PATH}"
echo "[phase10-streamingllm] task=clean_suite n=1 budgets=8192 12288 16384 K=48 96" | tee -a "${LOG_PATH}"

artifacts=()

for budget in 8192 12288 16384; do
  tmp="$(mktemp)"
  echo "[phase10-streamingllm] budget=${budget}" | tee -a "${LOG_PATH}"
  .venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
    --stage smoke \
    --task clean_suite \
    --num-samples 1 \
    --base-context-budget "${budget}" \
    --recency-window 128 \
    --k 48 96 \
    --conditions A B B_match Random-K Oldest-K IdleKV Oracle-K \
    --query-scoring-mode exact_q \
    --oracle-mode gold_spans \
    --initial-compressor streaming_llm 2>&1 | tee -a "${LOG_PATH}" | tee "$tmp"
  artifact="$(grep -E '(^/.*\.json$|^phases/.*\.json$)' "$tmp" | tail -n 1 || true)"
  rm -f "$tmp"
  if [[ -z "$artifact" ]]; then
    echo "[phase10-streamingllm] could not locate artifact for budget=${budget}" | tee -a "${LOG_PATH}"
    exit 1
  fi
  artifacts+=("$artifact")
done

summary_args=()
for artifact in "${artifacts[@]}"; do
  summary_args+=(--artifact "$artifact")
done

.venv/bin/python phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py \
  "${summary_args[@]}" \
  --out-csv "$OUT_CSV" \
  2>&1 | tee -a "${LOG_PATH}"

echo "[phase10-streamingllm] wrote ${OUT_CSV}" | tee -a "${LOG_PATH}"
echo "[phase10-streamingllm] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "${LOG_PATH}"
