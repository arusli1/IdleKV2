#!/usr/bin/env bash
# Phase 18 Step 6: Llama-3.1-8B-Instruct appendix run.
#
# 4Q at K=96 only, n=12, with the binding contrast set:
# A, B_match, RepairKV, Refresh-K-budgeted, PageSummary-Quest-inspired,
# RepairKV-no-burst.
# Descriptive, no gate per v5.1 plan -- just preliminary cross-model
# evidence for the Limitations paragraph.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
RESULTS_DIR="phases/phase18_pre_submission/results/llama"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/llama_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "[llama] $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "${LOG_PATH}"
.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
    --stage full \
    --task clean_suite \
    --num-samples 12 \
    --base-context-budget 16384 \
    --recency-window 128 \
    --query-scoring-mode exact_q \
    --oracle-mode gold_spans \
    --k 96 \
    --conditions A B B_match IdleKV Refresh-K-budgeted PageSummary-Quest-inspired RepairKV-no-burst \
    --model-dir /home/ubuntu/IdleKV/models/Llama-3.1-8B-Instruct \
    2>&1 | tee -a "${LOG_PATH}"
echo "[llama] done $(date -u)"
