#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

if [[ -z "${MODEL_TRANSFER_MODEL_DIR:-}" ]]; then
  echo "[phase10-model-transfer-locked] set MODEL_TRANSFER_MODEL_DIR to an ability-gated model directory" >&2
  exit 1
fi

MODEL_DIR="$MODEL_TRANSFER_MODEL_DIR"
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[phase10-model-transfer-locked] model dir not found: $MODEL_DIR" >&2
  exit 1
fi

MODEL_LABEL="$(basename "$MODEL_DIR" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_')"
NUM_SAMPLES="${MODEL_TRANSFER_LOCKED_NUM_SAMPLES:-12}"
K_VALUES="${MODEL_TRANSFER_LOCKED_K_VALUES:-48 96}"
BUDGETS="${MODEL_TRANSFER_LOCKED_BUDGETS:-8192 16384}"
ABILITY_SAMPLES="${MODEL_TRANSFER_ABILITY_NUM_SAMPLES:-4}"
ABILITY_CSV="${MODEL_TRANSFER_ABILITY_CSV:-phases/phase10_expansion/results/model_transfer_ability_${MODEL_LABEL}_n${ABILITY_SAMPLES}.csv}"
MIN_FULL_SCORE="${MODEL_TRANSFER_MIN_FULL_SCORE:-0.75}"
OUT_CSV="phases/phase10_expansion/results/model_transfer_${MODEL_LABEL}_locked_n${NUM_SAMPLES}.csv"
LOG_DIR="phases/phase10_expansion/results/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/model_transfer_locked_${MODEL_LABEL}_n${NUM_SAMPLES}_$(date -u +%Y%m%dT%H%M%SZ).log"

if [[ ! -f "$ABILITY_CSV" ]]; then
  echo "[phase10-model-transfer-locked] missing ability gate: $ABILITY_CSV" | tee "$LOG"
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
print(f"[phase10-model-transfer-locked] ability gate passed: best full-cache score {best:.3f}")
PY

echo "[phase10-model-transfer-locked] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "$LOG"
echo "[phase10-model-transfer-locked] model=$MODEL_DIR task=clean_suite budgets=$BUDGETS n=$NUM_SAMPLES K=$K_VALUES exact_q gold_spans" | tee -a "$LOG"

artifacts=()
for budget in $BUDGETS; do
  tmp="$(mktemp)"
  echo "[phase10-model-transfer-locked] budget=$budget" | tee -a "$LOG"
  .venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
    --stage full \
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
  artifact="$(grep -E '(^/.*\.json$|^phases/.*\.json$)' "$tmp" | tail -n 1 || true)"
  rm -f "$tmp"
  if [[ -z "$artifact" ]]; then
    echo "[phase10-model-transfer-locked] could not locate artifact for budget=$budget" | tee -a "$LOG"
    exit 1
  fi
  artifacts+=("$artifact")
done

summary_args=()
for artifact in "${artifacts[@]}"; do
  summary_args+=(--artifact "$artifact")
done

.venv/bin/python phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py \
  "${summary_args[@]}" \
  --out-csv "$OUT_CSV" \
  2>&1 | tee -a "$LOG"

echo "[phase10-model-transfer-locked] wrote $OUT_CSV" | tee -a "$LOG"
echo "[phase10-model-transfer-locked] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOG"
