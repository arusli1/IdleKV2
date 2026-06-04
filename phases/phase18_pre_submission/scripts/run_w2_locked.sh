#!/usr/bin/env bash
# Phase 18 Step 2: W2 paper-quality runtime probe, post-bugfix.
#
# Stage decomposition for the runtime claim. Now uses full-pool warmup
# (no host_pool_coverage<1.0 silent failure), BF16-async H2D
# (no dtype-upcast block), and 3 warmup trials (no first-fault bias).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
RESULTS_DIR="phases/phase18_pre_submission/results/w2"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/w2_locked_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "[w2] $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "${LOG_PATH}"
echo "[w2] post-bugfix: dtype-upcast + full-pool-warmup fixes from Step 0a" | tee -a "${LOG_PATH}"

# Phase 18 W2 cells: candidate sizes 32K / 256K / 1M (skip 4M for time).
# K = 96 (headline quality cell) and K = 5000 (envelope/scaling reference).
# Pool chunks set to ensure full-pool coverage (32768/16384=2, 262144/16384=16, 1048576/16384=64).
.venv/bin/python phases/phase4_eviction_buffer/scripts/run_runtime_capacity_profile.py \
    --mode chunked_select_multi_k \
    --candidate-tokens 32768,262144,1048576 \
    --k 96,5000 \
    --query-len 64 \
    --chunk-tokens 16384 \
    --source-pool-chunks 64 \
    --trials 80 \
    --warmup-trials 3 \
    --device cuda --dtype bfloat16 \
    --n-layers 28 --n-query-heads 28 --n-kv-heads 4 --head-dim 128 \
    --out-prefix "${RESULTS_DIR}/w2_chunked_select" \
    2>&1 | tee -a "${LOG_PATH}"

echo "[w2] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "${LOG_PATH}"
