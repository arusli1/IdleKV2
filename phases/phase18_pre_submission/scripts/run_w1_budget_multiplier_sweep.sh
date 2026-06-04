#!/usr/bin/env bash
# Phase 18 v5.2: tight-budget multiplier sweep at K=96, n=12, 4Q.
#
# Senior ML researcher + devil's advocate critique flagged the load-bearing
# flaw of the current Phase 18: Refresh-K-budgeted's 1.5s per-K T_repair
# budget rarely fires the cap, so it's effectively unbudgeted Refresh-K.
# The "approaches budgeted Q2-aware reselection quality" abstract clause
# is technically supported but the comparator is degenerate.
#
# This sweep varies tm_budget_multiplier across {0.05, 0.10, 0.30, 1.05}
# at K=96 only. At multiplier=0.10 the budget is ~150ms (matches the
# W2-probe-measured deployment runtime cost). At 0.05 it's ~75ms (sub-
# deployment cost). At 1.05 it's the original loose budget used in the
# K-sweep.
#
# Outputs go to phases/phase18_pre_submission/results/w1_tight/.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
RESULTS_DIR="phases/phase18_pre_submission/results/w1_tight"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

NUM_SAMPLES="${PHASE18_W1T_N:-12}"
MULTIPLIERS=(0.05 0.10 0.30 1.05)

for MULT in "${MULTIPLIERS[@]}"; do
    LOG_PATH="${LOG_DIR}/tight_mult${MULT}_$(date -u +%Y%m%dT%H%M%SZ).log"
    echo "[tight] mult=${MULT} $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "${LOG_PATH}"
    .venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
        --stage full \
        --task clean_suite \
        --num-samples "${NUM_SAMPLES}" \
        --base-context-budget 16384 \
        --recency-window 128 \
        --query-scoring-mode exact_q \
        --oracle-mode gold_spans \
        --k 96 \
        --conditions A B B_match IdleKV Refresh-K Refresh-K-budgeted PageSummary-Quest-inspired RepairKV-no-burst \
        --tm-budget-multiplier "${MULT}" \
        2>&1 | tee -a "${LOG_PATH}"
    # Preserve the artifact under a multiplier-tagged name so successive
    # multipliers don't overwrite each other (the runner's artifact path
    # is deterministic on the input parameters, but doesn't include
    # tm_budget_multiplier).
    LATEST_ART="$(ls -t phases/phase6_repair/results/full/clean_suite_b16384_r128_qexact_q_ogold_spans_n${NUM_SAMPLES}_k96_*.json 2>/dev/null | head -1)"
    if [[ -n "${LATEST_ART}" ]]; then
        DEST="${RESULTS_DIR}/tight_mult${MULT}_$(basename "${LATEST_ART}")"
        cp "${LATEST_ART}" "${DEST}"
        echo "[tight] preserved -> ${DEST}" | tee -a "${LOG_PATH}"
    fi
    echo "[tight] mult=${MULT} done $(date -u)" | tee -a "${LOG_PATH}"
done

echo "[tight] all multipliers done $(date -u)"
