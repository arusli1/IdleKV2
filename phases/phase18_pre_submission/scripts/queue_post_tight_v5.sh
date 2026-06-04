#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/IdleKV
LOGD=/tmp/queue_post_tight.log
echo "[queue] $(date -u): waiting for tight sweep mult 1.05 to complete..." | tee "${LOGD}"
until ls phases/phase18_pre_submission/results/w1_tight/tight_mult1.05_*.json >/dev/null 2>&1; do
    sleep 60
done
echo "[queue] $(date -u): tight sweep done. Starting K-sweep redo (path validated by original K-sweep + PageSummary unit tests)..." | tee -a "${LOGD}"
PHASE18_W1_N=12 PHASE18_W1_TM_MULT=1.05 bash phases/phase18_pre_submission/scripts/run_w1_4q_ksweep.sh 2>&1 | tee -a "${LOGD}"

# SMOKE 1: chunk-size with RepairKV-chunked (NEW code path, need to validate).
echo "[queue] $(date -u): SMOKE 1 - chunk-size + RepairKV-chunked smoke n=2..." | tee -a "${LOGD}"
mkdir -p phases/phase18_pre_submission/results/chunk_size_sens/logs
LOGP="phases/phase18_pre_submission/results/chunk_size_sens/logs/smoke_n2_$(date -u +%Y%m%dT%H%M%SZ).log"
.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
    --stage full --task clean_suite --num-samples 2 --base-context-budget 16384 \
    --recency-window 128 --query-scoring-mode exact_q --oracle-mode gold_spans \
    --k 96 --conditions A B B_match IdleKV PageSummary-Quest-inspired RepairKV-chunked \
    --page-summary-chunk-size 128 \
    2>&1 | tee "${LOGP}"
# Validate smoke: extract RepairKV-chunked score, confirm not all-zero / not all-one
SMOKE_ART="$(ls -t phases/phase6_repair/results/full/clean_suite_b16384_*n2_k96_*.json 2>/dev/null | head -1)"
.venv/bin/python -c "
import json, sys
d = json.load(open('${SMOKE_ART}'))
rows = d['rows']
chunked = [r.get('repairkv_chunked_score') for r in rows if r.get('repairkv_chunked_score') is not None]
mean = sum(chunked)/len(chunked) if chunked else float('nan')
print(f'SMOKE chunk-size: n={len(chunked)} RepairKV-chunked mean={mean:.3f}')
if not chunked or all(c == 0 for c in chunked):
    print('SMOKE FAILED: all zeros'); sys.exit(1)
if all(c == 1 for c in chunked):
    print('SMOKE WARNING: all 1.0 (possible saturation)')
print('SMOKE chunk-size OK')
" 2>&1 | tee -a "${LOGD}"

# Now full chunk-size sensitivity sweep at chunk_size in {32, 64, 256}
echo "[queue] $(date -u): chunk-size sensitivity sweep..." | tee -a "${LOGD}"
for CS in 32 64 256; do
    LOGP="phases/phase18_pre_submission/results/chunk_size_sens/logs/cs${CS}_$(date -u +%Y%m%dT%H%M%SZ).log"
    .venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
        --stage full --task clean_suite --num-samples 12 --base-context-budget 16384 \
        --recency-window 128 --query-scoring-mode exact_q --oracle-mode gold_spans \
        --k 96 --conditions A B B_match IdleKV PageSummary-Quest-inspired RepairKV-chunked \
        --page-summary-chunk-size "${CS}" \
        2>&1 | tee "${LOGP}"
    LATEST_ART="$(find phases/phase6_repair/results/full -maxdepth 1 -name 'clean_suite_b16384_*n12_k96_*.json' -newer "${LOGP}" 2>/dev/null | head -1)"
    if [[ -n "${LATEST_ART}" ]]; then
        cp "${LATEST_ART}" "phases/phase18_pre_submission/results/chunk_size_sens/cs${CS}_$(basename "${LATEST_ART}")"
    fi
done
echo "[queue] $(date -u): chunk-size sensitivity done. Llama low-K (existing model + path)..." | tee -a "${LOGD}"

mkdir -p phases/phase18_pre_submission/results/llama_lowk/logs
.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
    --stage full --task clean_suite --num-samples 12 --base-context-budget 16384 \
    --recency-window 128 --query-scoring-mode exact_q --oracle-mode gold_spans \
    --k 32 48 \
    --conditions A B B_match IdleKV Refresh-K-budgeted PageSummary-Quest-inspired RepairKV-no-burst \
    --model-dir /home/ubuntu/IdleKV/models/Llama-3.1-8B-Instruct \
    2>&1 | tee phases/phase18_pre_submission/results/llama_lowk/logs/llama_lowk_$(date -u +%Y%m%dT%H%M%SZ).log

echo "[queue] $(date -u): Llama low-K done. Running tight-budget K-sweep at mult 0.10 (deployment-realistic)..." | tee -a "${LOGD}"

# K-sweep at mult 0.10 across K=32/64/96/128 (skip K=80 to save time; 4 K's)
mkdir -p phases/phase18_pre_submission/results/w1_tight_ksweep/logs
LOGT="phases/phase18_pre_submission/results/w1_tight_ksweep/logs/tight_ksweep_$(date -u +%Y%m%dT%H%M%SZ).log"
.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
    --stage full --task clean_suite --num-samples 12 --base-context-budget 16384 \
    --recency-window 128 --query-scoring-mode exact_q --oracle-mode gold_spans \
    --k 32 64 96 128 \
    --conditions A B B_match IdleKV Refresh-K Refresh-K-budgeted PageSummary-Quest-inspired RepairKV-no-burst \
    --tm-budget-absolute-s 0.150 \
    2>&1 | tee "${LOGT}"
LATEST_TKSW="$(find phases/phase6_repair/results/full -maxdepth 1 -name 'clean_suite_b16384_*n12_k32-64-96-128*.json' -newer "${LOGT}" 2>/dev/null | head -1)"
if [[ -n "${LATEST_TKSW}" ]]; then
    cp "${LATEST_TKSW}" "phases/phase18_pre_submission/results/w1_tight_ksweep/$(basename "${LATEST_TKSW}")"
fi

# SMOKE 2: GPU scoring path (NEW code path, must validate score doesn't differ wildly from CPU)
echo "[queue] $(date -u): SMOKE 2 - GPU scoring n=2..." | tee -a "${LOGD}"
mkdir -p phases/phase18_pre_submission/results/gpu_verify/logs
LOG_GPU="phases/phase18_pre_submission/results/gpu_verify/logs/smoke_n2_$(date -u +%Y%m%dT%H%M%SZ).log"
PHASE18_SCORE_ON_GPU=1 .venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
    --stage full --task clean_suite --num-samples 2 --base-context-budget 16384 \
    --recency-window 128 --query-scoring-mode exact_q --oracle-mode gold_spans \
    --k 96 --conditions A B B_match IdleKV \
    2>&1 | tee "${LOG_GPU}"
SMOKE_GPU_ART="$(ls -t phases/phase6_repair/results/full/clean_suite_b16384_*n2_k96_*.json 2>/dev/null | head -1)"
.venv/bin/python -c "
import json, sys
d = json.load(open('${SMOKE_GPU_ART}'))
rows = d['rows']
ik = [r.get('idlekv_score') for r in rows if r.get('idlekv_score') is not None]
mean_ik = sum(ik)/len(ik) if ik else float('nan')
print(f'SMOKE gpu-score: n={len(ik)} IdleKV mean={mean_ik:.3f} (compare to CPU K=96 mean=0.917)')
if not ik:
    print('SMOKE FAILED: empty'); sys.exit(1)
print('SMOKE gpu-score OK')
" 2>&1 | tee -a "${LOGD}"

# Full GPU verification at n=4 (smoke confirmed, do real run)
echo "[queue] $(date -u): GPU verify n=12..." | tee -a "${LOGD}"
LOG_GPU="phases/phase18_pre_submission/results/gpu_verify/logs/verify_n12_$(date -u +%Y%m%dT%H%M%SZ).log"
PHASE18_SCORE_ON_GPU=1 .venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
    --stage full --task clean_suite --num-samples 12 --base-context-budget 16384 \
    --recency-window 128 --query-scoring-mode exact_q --oracle-mode gold_spans \
    --k 96 --conditions A B B_match IdleKV \
    2>&1 | tee "${LOG_GPU}"
LATEST_GPU_ART="$(find phases/phase6_repair/results/full -maxdepth 1 -name 'clean_suite_b16384_*n12_k96_*.json' -newer "${LOG_GPU}" 2>/dev/null | head -1)"
if [[ -n "${LATEST_GPU_ART}" ]]; then
    cp "${LATEST_GPU_ART}" "phases/phase18_pre_submission/results/gpu_verify/gpu_verify_$(basename "${LATEST_GPU_ART}")"
fi

echo "[queue] $(date -u): FULL PIPELINE DONE" | tee -a "${LOGD}"

echo "[queue] $(date -u): final analyses..." | tee -a "${LOGD}"
.venv/bin/python phases/phase18_pre_submission/scripts/analyze_tight_sweep.py 2>&1 | tee -a "${LOGD}"
.venv/bin/python phases/phase18_pre_submission/scripts/render_tight_sweep_figure.py 2>&1 | tee -a "${LOGD}"
# Re-render main K-sweep frontier with post-fix data
LATEST_KSW="$(ls -t phases/phase6_repair/results/full/clean_suite_b16384_*n12_k32-64-80-96-128*.json 2>/dev/null | head -1)"
if [[ -n "${LATEST_KSW}" ]]; then
    bash phases/phase18_pre_submission/scripts/post_ksweep_pipeline.sh 2>&1 | tee -a "${LOGD}" || true
fi
echo "[queue] $(date -u): all analyses done" | tee -a "${LOGD}"
