#!/usr/bin/env bash
# Phase 18 post-K-sweep one-shot pipeline:
#   1. Find latest K-sweep artifact.
#   2. Run analyze_w1_ksweep -> contrasts/tost/frontier CSVs.
#   3. Run decide_gate -> verdict JSON + console output.
#   4. Render figures from the frontier CSV + artifact.
#   5. Print headline numbers ready to slot into the abstract.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

# Find latest K-sweep artifact
ARTIFACT="$(ls -t phases/phase6_repair/results/full/clean_suite_b16384_*n12_k32-64-80-96-128_*.json 2>/dev/null | head -1)"
if [[ -z "${ARTIFACT}" ]]; then
    echo "[post] no K-sweep artifact found"
    exit 1
fi
echo "[post] using artifact: ${ARTIFACT}"

OUT_DIR="phases/phase18_pre_submission/results/w1"
FIG_DIR="phases/phase18_pre_submission/results/figures"
mkdir -p "${OUT_DIR}" "${FIG_DIR}"

# Step 1: analyze
echo "[post] analyze_w1_ksweep ..."
.venv/bin/python phases/phase18_pre_submission/scripts/analyze_w1_ksweep.py \
    --artifact "${ARTIFACT}" \
    --out-dir "${OUT_DIR}"

ART_STEM="$(basename "${ARTIFACT}" .json)"
CONTRASTS_CSV="${OUT_DIR}/${ART_STEM}_w1_contrasts.csv"
FRONTIER_CSV="${OUT_DIR}/${ART_STEM}_w1_frontier.csv"

# Step 2: decide gate
echo "[post] decide_gate ..."
.venv/bin/python phases/phase18_pre_submission/scripts/decide_gate.py \
    --contrasts-csv "${CONTRASTS_CSV}" \
    --frontier-csv "${FRONTIER_CSV}" \
    --k-target 96 \
    --out-json "${OUT_DIR}/${ART_STEM}_gate_decision.json"

# Step 3: render figures
echo "[post] render_phase18_figures ..."
.venv/bin/python phases/phase18_pre_submission/scripts/render_phase18_figures.py \
    --frontier-csv "${FRONTIER_CSV}" \
    --artifact "${ARTIFACT}" \
    --out-dir "${FIG_DIR}"

echo "[post] done -- inspect ${OUT_DIR} and ${FIG_DIR}"
