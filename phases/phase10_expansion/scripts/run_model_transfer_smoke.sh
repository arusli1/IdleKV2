#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

if [[ -z "${MODEL_TRANSFER_MODEL_DIR:-}" ]]; then
  echo "[phase10-model-transfer] set MODEL_TRANSFER_MODEL_DIR to an ability-gated model directory" >&2
  exit 1
fi

MODEL_DIR="$MODEL_TRANSFER_MODEL_DIR"
MODEL_LABEL="$(basename "$MODEL_DIR" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_')"
NUM_SAMPLES="${MODEL_TRANSFER_NUM_SAMPLES:-1}"
K_VALUES="${MODEL_TRANSFER_K_VALUES:-48 96}"
ABILITY_SAMPLES="${MODEL_TRANSFER_ABILITY_NUM_SAMPLES:-4}"
ABILITY_CSV="${MODEL_TRANSFER_ABILITY_CSV:-phases/phase10_expansion/results/model_transfer_ability_${MODEL_LABEL}_n${ABILITY_SAMPLES}.csv}"
MIN_FULL_SCORE="${MODEL_TRANSFER_MIN_FULL_SCORE:-0.75}"
LOG_DIR="phases/phase10_expansion/results/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/model_transfer_smoke_${MODEL_LABEL}_$(date -u +%Y%m%dT%H%M%SZ).log"
OUT_CSV="phases/phase10_expansion/results/model_transfer_${MODEL_LABEL}_smoke_n${NUM_SAMPLES}.csv"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[phase10-model-transfer] model dir not found: $MODEL_DIR" | tee "$LOG"
  exit 1
fi
if [[ "${MODEL_TRANSFER_SKIP_ABILITY_GATE:-0}" != "1" ]]; then
  if [[ ! -f "$ABILITY_CSV" ]]; then
    echo "[phase10-model-transfer] missing ability gate: $ABILITY_CSV" | tee "$LOG"
    echo "[phase10-model-transfer] run scripts/run_model_transfer_ability_smoke.sh first, or set MODEL_TRANSFER_SKIP_ABILITY_GATE=1 for an explicit exploratory run" | tee -a "$LOG"
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
scores = [float(row.get("condition_a", 0.0) or 0.0) for row in rows]
best = max(scores)
if best < threshold:
    raise SystemExit(f"ability gate failed: best full-cache score {best:.3f} < {threshold:.3f}")
print(f"[phase10-model-transfer] ability gate passed: best full-cache score {best:.3f}")
PY
fi

echo "[phase10-model-transfer] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "$LOG"
echo "[phase10-model-transfer] model=$MODEL_DIR task=clean_suite n=$NUM_SAMPLES K=$K_VALUES ability_csv=$ABILITY_CSV" | tee -a "$LOG"

artifacts=()
run_suite() {
  local budget="$1"
  local tmp
  tmp="$(mktemp)"
  echo "[phase10-model-transfer] budget=$budget" | tee -a "$LOG"
  .venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
    --stage smoke \
    --task clean_suite \
    --num-samples "$NUM_SAMPLES" \
    --model-dir "$MODEL_DIR" \
    --base-context-budget "$budget" \
    --recency-window 128 \
    --k $K_VALUES \
    --conditions A B B_match Random-K Oldest-K IdleKV Oracle-K \
    --query-scoring-mode exact_q \
    --oracle-mode gold_spans \
    2>&1 | tee -a "$LOG" | tee "$tmp"
  local artifact
  artifact="$(grep -E '(^/.*\.json$|^phases/.*\.json$)' "$tmp" | tail -n 1 || true)"
  rm -f "$tmp"
  if [[ -z "$artifact" ]]; then
    echo "[phase10-model-transfer] could not locate artifact for budget=$budget" | tee -a "$LOG"
    exit 1
  fi
  artifacts+=("$artifact")
}

run_suite 8192
run_suite 16384

summary_args=()
for artifact in "${artifacts[@]}"; do
  summary_args+=(--artifact "$artifact")
done

.venv/bin/python phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py \
  "${summary_args[@]}" \
  --out-csv "$OUT_CSV" \
  2>&1 | tee -a "$LOG"

echo "[phase10-model-transfer] wrote $OUT_CSV" | tee -a "$LOG"
echo "[phase10-model-transfer] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOG"
