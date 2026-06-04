#!/usr/bin/env bash
# Phase 18 W1: 4Q K-sweep paper-quality run.
#
# Runs n=24 across 9 K's at B_base=16384 with the 6-condition MVP:
#   A, B, B_match, IdleKV (RepairKV), Refresh-K (unbudgeted ceiling),
#   Refresh-K-budgeted, PageSummary-Quest-inspired, RepairKV-no-burst
# (TM-Recompute-BM25 deferred to Step 5.6 single-K=96 add-on.)
#
# Pre-launch sanity:
#   - Step 1 smoke must have completed (artifact JSON exists).
#   - σ(T_repair)/μ(T_repair) from the smoke must be < 0.10 to keep the
#     1.05 multiplier; otherwise pass --tm-budget-multiplier 1.20.
#
# Outputs land under phases/phase18_pre_submission/results/w1/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

RESULTS_DIR="phases/phase18_pre_submission/results/w1"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

NUM_SAMPLES="${PHASE18_W1_N:-12}"
TASK="${PHASE18_W1_TASK:-clean_suite}"
BASE_BUDGET="${PHASE18_W1_BASE:-16384}"
TM_MULT="${PHASE18_W1_TM_MULT:-1.05}"
LOG_PATH="${LOG_DIR}/w1_${TASK}_b${BASE_BUDGET}_n${NUM_SAMPLES}_$(date -u +%Y%m%dT%H%M%SZ).log"

# Reduced from 9 K's to 5 K's for the 2-day workshop deadline. 5 K's
# still gives a clean frontier shape including the headline K=96 cell
# and the saturating K=128 cell. Smaller K's drop because RepairKV's
# smoke shows score < 0.6 at K<64 -- diminishing returns for the
# binding-contrast question.
K_VALUES=(32 64 80 96 128)

echo "[w1-ksweep] $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "${LOG_PATH}"
echo "[w1-ksweep] task=${TASK} base=${BASE_BUDGET} n=${NUM_SAMPLES} K=${K_VALUES[*]}" | tee -a "${LOG_PATH}"
echo "[w1-ksweep] tm_budget_multiplier=${TM_MULT}" | tee -a "${LOG_PATH}"
echo "[w1-ksweep] pre-registration commit: $(git rev-parse HEAD)" | tee -a "${LOG_PATH}"

# Conditions: A, B, B_match (matched no-repair floor), IdleKV (RepairKV),
# Refresh-K (unbudgeted ceiling reference), Refresh-K-budgeted (TM
# matched-time reselection), PageSummary-Quest-inspired (TM matched-time
# Quest-style), RepairKV-no-burst (burst-expansion ablation).
# Random-K + Oldest-K are content-agnostic floors -- include them too.
.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
    --stage full \
    --task "${TASK}" \
    --num-samples "${NUM_SAMPLES}" \
    --base-context-budget "${BASE_BUDGET}" \
    --recency-window 128 \
    --query-scoring-mode exact_q \
    --oracle-mode gold_spans \
    --k "${K_VALUES[@]}" \
    --conditions A B B_match Random-K Oldest-K IdleKV Refresh-K Oracle-K \
                 Refresh-K-budgeted PageSummary-Quest-inspired RepairKV-no-burst \
    2>&1 | tee -a "${LOG_PATH}"

echo "[w1-ksweep] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "${LOG_PATH}"
