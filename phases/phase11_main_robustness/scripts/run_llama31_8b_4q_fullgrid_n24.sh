#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

MODEL_DIR="${MODEL_TRANSFER_MODEL_DIR:-models/Llama-3.1-8B-Instruct}"
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[phase11-llama-4q-n24] model dir not found: ${MODEL_DIR}" >&2
  exit 1
fi

MODEL_LABEL="$(basename "$MODEL_DIR" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_')"
ABILITY_CSV="${MODEL_TRANSFER_ABILITY_CSV:-phases/phase10_expansion/results/model_transfer_ability_${MODEL_LABEL}_n4.csv}"
MIN_FULL_SCORE="${MODEL_TRANSFER_MIN_FULL_SCORE:-0.75}"
LOG_DIR="phases/phase11_main_robustness/results/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/llama31_8b_4q_fullgrid_n24_$(date -u +%Y%m%dT%H%M%SZ).log"
OUT_CSV="phases/phase11_main_robustness/results/llama31_8b_4q_fullgrid_n24.csv"

if [[ ! -f "$ABILITY_CSV" ]]; then
  echo "[phase11-llama-4q-n24] missing ability gate: ${ABILITY_CSV}" | tee "$LOG_PATH"
  exit 1
fi

.venv/bin/python - "$ABILITY_CSV" "$MIN_FULL_SCORE" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
threshold = float(sys.argv[2])
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit(f"ability CSV is empty: {path}")
best = max(float(row.get("condition_a", 0.0) or 0.0) for row in rows)
if best < threshold:
    raise SystemExit(f"ability gate failed: best full-cache score {best:.3f} < {threshold:.3f}")
print(f"[phase11-llama-4q-n24] ability gate passed: best full-cache score {best:.3f}")
PY

echo "[phase11-llama-4q-n24] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "$LOG_PATH"
echo "[phase11-llama-4q-n24] model=${MODEL_DIR} task=clean_suite B=16384 n=24 K=8 16 24 32 48 64 80 96 128 exact_q gold_spans" | tee -a "$LOG_PATH"

tmp="$(mktemp)"
.venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
  --stage full \
  --task clean_suite \
  --num-samples 24 \
  --model-dir "$MODEL_DIR" \
  --base-context-budget 16384 \
  --recency-window 128 \
  --k 8 16 24 32 48 64 80 96 128 \
  --conditions A B B_match Random-K Oldest-K IdleKV Oracle-K \
  --query-scoring-mode exact_q \
  --oracle-mode gold_spans \
  2>&1 | tee -a "$LOG_PATH" | tee "$tmp"

artifact="$(grep -E '(^/.*\.json$|^phases/.*\.json$)' "$tmp" | tail -n 1 || true)"
rm -f "$tmp"
if [[ -z "$artifact" ]]; then
  echo "[phase11-llama-4q-n24] could not locate run artifact" | tee -a "$LOG_PATH"
  exit 1
fi

.venv/bin/python phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py \
  --artifact "$artifact" \
  --out-csv "$OUT_CSV" \
  2>&1 | tee -a "$LOG_PATH"

echo "[phase11-llama-4q-n24] wrote ${OUT_CSV}" | tee -a "$LOG_PATH"
echo "[phase11-llama-4q-n24] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOG_PATH"

