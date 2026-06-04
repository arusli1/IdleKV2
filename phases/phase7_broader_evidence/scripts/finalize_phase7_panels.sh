#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
POSTPROCESS="$ROOT/phases/phase7_broader_evidence/scripts/postprocess_phase7_artifact.sh"
PLOTTER="$ROOT/phases/phase7_broader_evidence/scripts/plot_frontier_svg.py"

FOURQ_ARTIFACT="${FOURQ_ARTIFACT:-$ROOT/phases/phase6_repair/results/full/clean_suite_b16384_r128_qexact_q_ogold_spans_n100_k8-16-24-32-48-64-80-96-128_ca-b-bmatch-idlekv-randomk-oldestk-oraclek.json}"
SIXQ_ARTIFACT="${1:-${SIXQ_ARTIFACT:-}}"

if [[ ! -f "$FOURQ_ARTIFACT" ]]; then
  echo "Missing frozen 4q artifact: $FOURQ_ARTIFACT" >&2
  exit 1
fi

if [[ -z "$SIXQ_ARTIFACT" ]]; then
  echo "Usage: $0 <6q_artifact.json>" >&2
  echo "  or set SIXQ_ARTIFACT=/abs/path/to/final_6q.json" >&2
  exit 1
fi

if [[ ! -f "$SIXQ_ARTIFACT" ]]; then
  echo "Missing 6q artifact: $SIXQ_ARTIFACT" >&2
  exit 1
fi

SIXQ_BASENAME="$(basename "$SIXQ_ARTIFACT" .json)"
SIXQ_PREFIX_DEFAULT="$(printf '%s\n' "$SIXQ_BASENAME" | sed -E 's/_r[0-9]+_qexact_q_ogold_spans_.*$//')"
SIXQ_PREFIX="${SIXQ_PREFIX:-phase7_${SIXQ_PREFIX_DEFAULT}_exact}"

cd "$ROOT"

bash "$POSTPROCESS" \
  "$FOURQ_ARTIFACT" \
  phase7_clean_suite_b16384_exact \
  "MQ-NIAH-4Q exact Q2 score vs. restore budget"

bash "$POSTPROCESS" \
  "$SIXQ_ARTIFACT" \
  "$SIXQ_PREFIX" \
  "MQ-NIAH-6Q exact Q2 score vs. restore budget"

"$PYTHON_BIN" "$PLOTTER" \
  --csv "$ROOT/paper/figures/phase7_clean_suite_b16384_exact_overall.csv" \
  --out "$ROOT/paper/figures/phase7_clean_suite_b16384_exact_main.svg" \
  --title "MQ-NIAH-4Q exact Q2 score vs. restore budget" \
  --series b_match,idlekv,random_k,oracle_k

"$PYTHON_BIN" "$PLOTTER" \
  --csv "$ROOT/paper/figures/${SIXQ_PREFIX}_overall.csv" \
  --out "$ROOT/paper/figures/${SIXQ_PREFIX}_main.svg" \
  --title "MQ-NIAH-6Q exact Q2 score vs. restore budget" \
  --series b_match,idlekv,random_k,oracle_k

echo "Phase 7 panels finalized:"
echo "  4q main svg: $ROOT/paper/figures/phase7_clean_suite_b16384_exact_main.svg"
echo "  6q main svg: $ROOT/paper/figures/${SIXQ_PREFIX}_main.svg"
