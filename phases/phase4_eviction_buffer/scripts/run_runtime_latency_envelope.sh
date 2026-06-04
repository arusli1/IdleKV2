#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

STAMP="${IDLEKV_RUNTIME_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULTS_DIR="phases/phase4_eviction_buffer/results/runtime_capacity"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

TRIALS="${TRIALS:-80}"
WARMUP_TRIALS="${WARMUP_TRIALS:-5}"
CANDIDATE_TOKENS="${CANDIDATE_TOKENS:-32768,65536,131072,262144,524288,1048576}"
K_VALUES="${K_VALUES:-96,512,1024,5000}"
QUERY_LENGTHS="${QUERY_LENGTHS:-16,64,128}"
ACTIVE_TOKENS="${ACTIVE_TOKENS:-32768,131072,524288}"
CHUNK_TOKENS="${CHUNK_TOKENS:-16384}"
SOURCE_POOL_CHUNKS="${SOURCE_POOL_CHUNKS:-64}"
DEVICE="${DEVICE:-cuda}"

echo "[runtime-envelope] stamp=${STAMP}"
echo "[runtime-envelope] trials=${TRIALS} warmup=${WARMUP_TRIALS}"
echo "[runtime-envelope] candidates=${CANDIDATE_TOKENS}"
echo "[runtime-envelope] k=${K_VALUES} q=${QUERY_LENGTHS} active=${ACTIVE_TOKENS}"

IFS=',' read -r -a QUERY_LENGTH_ARRAY <<< "${QUERY_LENGTHS}"
SELECT_FILES=()
for query_len in "${QUERY_LENGTH_ARRAY[@]}"; do
  query_len="$(echo "${query_len}" | xargs)"
  prefix="${RESULTS_DIR}/runtime_latency_envelope_${STAMP}_q${query_len}"
  SELECT_FILES+=("${prefix}.csv")
  .venv/bin/python phases/phase4_eviction_buffer/scripts/run_runtime_capacity_profile.py \
    --mode chunked_select_multi_k \
    --device "${DEVICE}" \
    --candidate-tokens "${CANDIDATE_TOKENS}" \
    --k "${K_VALUES}" \
    --query-len "${query_len}" \
    --chunk-tokens "${CHUNK_TOKENS}" \
    --source-pool-chunks "${SOURCE_POOL_CHUNKS}" \
    --trials "${TRIALS}" \
    --warmup-trials "${WARMUP_TRIALS}" \
    --out-prefix "${prefix}" \
    2>&1 | tee "${LOG_DIR}/runtime_latency_envelope_${STAMP}_q${query_len}.log"
done

move_prefix="${RESULTS_DIR}/runtime_latency_envelope_${STAMP}_move"
.venv/bin/python phases/phase4_eviction_buffer/scripts/run_runtime_capacity_profile.py \
  --mode move_inject \
  --device "${DEVICE}" \
  --active-tokens "${ACTIVE_TOKENS}" \
  --k "${K_VALUES}" \
  --trials "${TRIALS}" \
  --warmup-trials "${WARMUP_TRIALS}" \
  --out-prefix "${move_prefix}" \
  2>&1 | tee "${LOG_DIR}/runtime_latency_envelope_${STAMP}_move.log"

.venv/bin/python - "$STAMP" "${SELECT_FILES[@]}" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

stamp = sys.argv[1]
paths = [Path(arg) for arg in sys.argv[2:]]
rows: list[dict[str, str]] = []
fieldnames: list[str] = []

for path in paths:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"missing header: {path}")
        for name in reader.fieldnames:
            if name not in fieldnames:
                fieldnames.append(name)
        rows.extend(reader)

out_path = Path("phases/phase4_eviction_buffer/results/runtime_capacity") / (
    f"runtime_latency_envelope_{stamp}_select.csv"
)
with out_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"[runtime-envelope] wrote {out_path}")
PY

cat > "${RESULTS_DIR}/runtime_latency_envelope_${STAMP}_manifest.json" <<EOF
{
  "stamp": "${STAMP}",
  "candidate_tokens": "${CANDIDATE_TOKENS}",
  "k_values": "${K_VALUES}",
  "query_lengths": "${QUERY_LENGTHS}",
  "active_tokens": "${ACTIVE_TOKENS}",
  "chunk_tokens": "${CHUNK_TOKENS}",
  "source_pool_chunks": "${SOURCE_POOL_CHUNKS}",
  "trials": ${TRIALS},
  "warmup_trials": ${WARMUP_TRIALS},
  "device": "${DEVICE}",
  "select_csv": "${RESULTS_DIR}/runtime_latency_envelope_${STAMP}_select.csv",
  "move_csv": "${move_prefix}.csv"
}
EOF
echo "[runtime-envelope] wrote ${RESULTS_DIR}/runtime_latency_envelope_${STAMP}_manifest.json"
