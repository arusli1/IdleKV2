#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

LOG_DIR="phases/phase10_expansion/results/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/selector_variant_smoke_$(date -u +%Y%m%dT%H%M%SZ).log"
OUT_CSV="phases/phase10_expansion/results/selector_variant_smoke_n1.csv"

echo "[phase10-selector-smoke] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "$LOG_PATH"
echo "[phase10-selector-smoke] task=clean_suite B=16384 n=1 K=24 48 96 exact_q gold_spans" | tee -a "$LOG_PATH"

tmp="$(mktemp)"
.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
  --stage smoke \
  --task clean_suite \
  --num-samples 1 \
  --base-context-budget 16384 \
  --recency-window 128 \
  --k 24 48 96 \
  --conditions A B B_match IdleKV IdleKV-Coverage IdleKV-MMR Oracle-K \
  --query-scoring-mode exact_q \
  --oracle-mode gold_spans \
  2>&1 | tee -a "$LOG_PATH" | tee "$tmp"

artifact="$(grep -E '(^/.*\.json$|^phases/.*\.json$)' "$tmp" | tail -n 1 || true)"
rm -f "$tmp"
if [[ -z "$artifact" ]]; then
  echo "[phase10-selector-smoke] could not locate artifact" | tee -a "$LOG_PATH"
  exit 1
fi

.venv/bin/python phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py \
  --artifact "$artifact" \
  --out-csv "$OUT_CSV" \
  2>&1 | tee -a "$LOG_PATH"

echo "[phase10-selector-smoke] wrote ${OUT_CSV}" | tee -a "$LOG_PATH"
echo "[phase10-selector-smoke] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOG_PATH"
