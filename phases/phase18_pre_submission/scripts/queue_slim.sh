#!/usr/bin/env bash
# Slim queue: tight K-sweep at 150ms abs + GPU verify + final analyses.
# Skips chunk-size sensitivity (4.4hr/cs × 4 = too long) and Llama low-K.
set -euo pipefail
cd /home/ubuntu/IdleKV
LOGD=/tmp/queue_slim.log
echo "[slim] $(date -u): start" | tee "${LOGD}"

# Tight K-sweep at 150ms ABSOLUTE budget across K
echo "[slim] $(date -u): tight K-sweep at 150ms abs (Qwen, K∈{32,64,96,128}, n=12)..." | tee -a "${LOGD}"
mkdir -p phases/phase18_pre_submission/results/w1_tight_ksweep/logs
LOGT="phases/phase18_pre_submission/results/w1_tight_ksweep/logs/tight_ksweep_$(date -u +%Y%m%dT%H%M%SZ).log"
.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
    --stage full --task clean_suite --num-samples 12 --base-context-budget 16384 \
    --recency-window 128 --query-scoring-mode exact_q --oracle-mode gold_spans \
    --k 32 64 96 128 \
    --conditions A B B_match IdleKV Refresh-K Refresh-K-budgeted PageSummary-Quest-inspired RepairKV-no-burst \
    --tm-budget-absolute-s 0.150 \
    2>&1 | tee "${LOGT}"
LATEST_TKSW="$(find phases/phase6_repair/results/full -maxdepth 1 -name 'clean_suite_b16384_*n12_k32-64-96-128*' -newer "${LOGT}" 2>/dev/null | head -1)"
if [[ -n "${LATEST_TKSW}" ]]; then
    cp "${LATEST_TKSW}" "phases/phase18_pre_submission/results/w1_tight_ksweep/$(basename "${LATEST_TKSW}")"
    echo "[slim] $(date -u): tight K-sweep artifact: $(basename "${LATEST_TKSW}")" | tee -a "${LOGD}"
else
    echo "[slim] $(date -u): WARNING: tight K-sweep artifact not found" | tee -a "${LOGD}"
fi

# GPU verify smoke (n=2, validates GPU scoring path)
echo "[slim] $(date -u): GPU smoke n=2..." | tee -a "${LOGD}"
mkdir -p phases/phase18_pre_submission/results/gpu_verify/logs
LOG_GPU_S="phases/phase18_pre_submission/results/gpu_verify/logs/smoke_n2_$(date -u +%Y%m%dT%H%M%SZ).log"
PHASE18_SCORE_ON_GPU=1 .venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
    --stage full --task clean_suite --num-samples 2 --base-context-budget 16384 \
    --recency-window 128 --query-scoring-mode exact_q --oracle-mode gold_spans \
    --k 96 --conditions A B B_match IdleKV \
    2>&1 | tee "${LOG_GPU_S}"

# GPU verify full (n=12 K=96)
echo "[slim] $(date -u): GPU verify n=12 K=96..." | tee -a "${LOGD}"
LOG_GPU="phases/phase18_pre_submission/results/gpu_verify/logs/verify_n12_$(date -u +%Y%m%dT%H%M%SZ).log"
PHASE18_SCORE_ON_GPU=1 .venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
    --stage full --task clean_suite --num-samples 12 --base-context-budget 16384 \
    --recency-window 128 --query-scoring-mode exact_q --oracle-mode gold_spans \
    --k 96 --conditions A B B_match IdleKV \
    2>&1 | tee "${LOG_GPU}"
LATEST_GPU="$(find phases/phase6_repair/results/full -maxdepth 1 -name 'clean_suite_b16384_*n12_k96_*' -newer "${LOG_GPU}" 2>/dev/null | head -1)"
if [[ -n "${LATEST_GPU}" ]]; then
    cp "${LATEST_GPU}" "phases/phase18_pre_submission/results/gpu_verify/gpu_verify_$(basename "${LATEST_GPU}")"
    echo "[slim] $(date -u): GPU verify artifact: $(basename "${LATEST_GPU}")" | tee -a "${LOGD}"
fi

# Final analyses
echo "[slim] $(date -u): final analyses..." | tee -a "${LOGD}"
.venv/bin/python phases/phase18_pre_submission/scripts/aggregate_phase18_results.py 2>&1 | tee -a "${LOGD}"

echo "[slim] $(date -u): DONE" | tee -a "${LOGD}"
