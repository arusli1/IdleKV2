#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

mkdir -p phases/phase8_streaming_strict_cap/results/two_tier_snapkv/logs

TASK="${TASK:-clean_suite}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-65536 98304 131072 196608 327680}"
CPU_STORE_FRACTIONS="${CPU_STORE_FRACTIONS:-0.50 0.625 0.75 0.875 1.0}"

for context_length in $CONTEXT_LENGTHS; do
  .venv/bin/python phases/phase8_streaming_strict_cap/scripts/sweep_two_tier_snapkv_spill.py \
    --task "$TASK" \
    --num-samples "$NUM_SAMPLES" \
    --total-context-length "$context_length" \
    --cpu-store-fractions $CPU_STORE_FRACTIONS
done
