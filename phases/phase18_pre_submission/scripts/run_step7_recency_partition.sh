#!/usr/bin/env bash
# Phase 18 Step 7: recency-favorable partition appendix run.
#
# Tests the v3 hostile-reviewer attack: "the pooled 4Q partitions
# explicitly EXCLUDE tail-anchored needles from Q2, which is exactly
# what SnapKV evicts first." If RepairKV's gain shrinks proportionally
# to B_match's improvement on a recency-favorable partition (12->34),
# the operator is robust. If RepairKV underperforms B_match, the
# headline is partition-dependent.
#
# Single recency-favorable partition (12->34), K=96, n=12.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
RESULTS_DIR="phases/phase18_pre_submission/results/recency"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/recency_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "[recency] $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "${LOG_PATH}"
.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
    --stage full \
    --task mq_niah_4q_split_12_to_34 \
    --num-samples 12 \
    --base-context-budget 16384 \
    --recency-window 128 \
    --query-scoring-mode exact_q \
    --oracle-mode gold_spans \
    --k 96 \
    --conditions A B B_match IdleKV Refresh-K Refresh-K-budgeted PageSummary-Quest-inspired RepairKV-no-burst \
    2>&1 | tee -a "${LOG_PATH}"
echo "[recency] done $(date -u)"
