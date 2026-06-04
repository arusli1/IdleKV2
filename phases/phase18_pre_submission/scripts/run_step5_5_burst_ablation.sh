#!/usr/bin/env bash
# Phase 18 Step 5.5: burst-expansion ablation.
#
# RepairKV-no-burst (L=R=0) at K=96 only on 4Q, n=24.
# Determines whether the abstract says "lifecycle slot" (if no-burst
# stays meaningfully above PageSummary-Quest-inspired) or "burst
# expansion at the lifecycle slot" (if no-burst collapses).
#
# Already wired into runner.py as condition "RepairKV-no-burst", so
# this is a single-K, single-task invocation.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
RESULTS_DIR="phases/phase18_pre_submission/results/burst_ablation"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/burst_ablation_$(date -u +%Y%m%dT%H%M%SZ).log"

# Note: the K-sweep already includes RepairKV-no-burst at every K. So
# Step 5.5 can be SATISFIED by reading the K=96 row from the K-sweep
# artifact -- no additional GPU run is needed unless we want a
# separate audit cell.
echo "[burst-ablation] $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "${LOG_PATH}"
echo "[burst-ablation] If the K-sweep artifact already has the K=96 row" | tee -a "${LOG_PATH}"
echo "[burst-ablation] for RepairKV-no-burst, this script is a no-op." | tee -a "${LOG_PATH}"

K_SWEEP_ART_GLOB="phases/phase6_repair/results/full/clean_suite_b16384_*n24*ca-*-repairkvnoburst.json"
LATEST_ART="$(ls -t ${K_SWEEP_ART_GLOB} 2>/dev/null | head -1)"

if [[ -n "${LATEST_ART}" ]]; then
    echo "[burst-ablation] Found K-sweep artifact: ${LATEST_ART}" | tee -a "${LOG_PATH}"
    .venv/bin/python <<PY 2>&1 | tee -a "${LOG_PATH}"
import json
art = "${LATEST_ART}"
d = json.load(open(art))
rows = d['rows']
k96 = [r for r in rows if int(r.get('k', 0)) == 96]
if k96:
    print(f"[burst-ablation] K=96 rows already present in K-sweep artifact ({len(k96)} examples)")
    print(f"[burst-ablation] RepairKV-no-burst @ K=96: mean={sum(r['repairkv_no_burst_score'] for r in k96)/len(k96):.3f}")
    print(f"[burst-ablation] RepairKV @ K=96:         mean={sum(r['idlekv_score'] for r in k96)/len(k96):.3f}")
    print(f"[burst-ablation] No additional run needed.")
else:
    print(f"[burst-ablation] No K=96 rows found; need to re-run.")
PY
    echo "[burst-ablation] done $(date -u)"
    exit 0
fi

# Fallback: run K=96 only with RepairKV-no-burst (and minimal contrast set).
.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
    --stage full \
    --task clean_suite \
    --num-samples 24 \
    --base-context-budget 16384 \
    --recency-window 128 \
    --query-scoring-mode exact_q \
    --oracle-mode gold_spans \
    --k 96 \
    --conditions A B B_match IdleKV Refresh-K-budgeted PageSummary-Quest-inspired RepairKV-no-burst \
    2>&1 | tee -a "${LOG_PATH}"

echo "[burst-ablation] done $(date -u)"
