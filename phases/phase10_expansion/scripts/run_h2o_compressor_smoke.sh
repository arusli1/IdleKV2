#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="phases/phase10_expansion/results/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/h2o_compressor_smoke_$(date -u +%Y%m%dT%H%M%SZ).log"
OUT_CSV="phases/phase10_expansion/results/h2o_compressor_smoke_n2.csv"
BUDGETS=(14336 16384 18432)
K_VALUES=(48 96 128)

echo "[phase10-h2o-smoke] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee "${LOG_PATH}"
echo "[phase10-h2o-smoke] task=clean_suite B=${BUDGETS[*]} n=2 K=${K_VALUES[*]} exact_q gold_spans" | tee -a "${LOG_PATH}"

summary_paths=()
for budget in "${BUDGETS[@]}"; do
  tmp="$(mktemp)"
  echo "[phase10-h2o-smoke] running B=${budget}" | tee -a "${LOG_PATH}"
  .venv/bin/python phases/phase6_repair/scripts/run_phase6.py \
    --stage full \
    --task clean_suite \
    --num-samples 2 \
    --base-context-budget "${budget}" \
    --recency-window 128 \
    --k "${K_VALUES[@]}" \
    --conditions A B B_match Random-K Oldest-K IdleKV Oracle-K \
    --query-scoring-mode exact_q \
    --oracle-mode gold_spans \
    --initial-compressor h2o \
    2>&1 | tee -a "${LOG_PATH}" | tee "$tmp"

  artifact="$(grep -E '(^/.*\.json$|^phases/.*\.json$)' "$tmp" | tail -n 1 || true)"
  rm -f "$tmp"
  if [[ -z "$artifact" ]]; then
    echo "[phase10-h2o-smoke] could not locate artifact for B=${budget}" | tee -a "${LOG_PATH}"
    exit 1
  fi

  budget_csv="phases/phase10_expansion/results/h2o_compressor_smoke_b${budget}_n2.csv"
  .venv/bin/python phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py \
    --artifact "$artifact" \
    --out-csv "$budget_csv" \
    2>&1 | tee -a "${LOG_PATH}"
  summary_paths+=("$budget_csv")
done

.venv/bin/python - "${OUT_CSV}" "${summary_paths[@]}" <<'PY'
import csv
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
inputs = [Path(path) for path in sys.argv[2:]]
rows = []
fieldnames = None
for path in inputs:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if fieldnames is None:
            fieldnames = reader.fieldnames
        rows.extend(reader)
if fieldnames is None:
    raise SystemExit("no H2O smoke summaries were produced")
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY

echo "[phase10-h2o-smoke] wrote ${OUT_CSV}" | tee -a "${LOG_PATH}"
.venv/bin/python phases/phase10_expansion/scripts/recommend_compressor_smoke.py \
  --summary-csv "${OUT_CSV}" \
  2>&1 | tee -a "${LOG_PATH}"
echo "[phase10-h2o-smoke] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "${LOG_PATH}"
