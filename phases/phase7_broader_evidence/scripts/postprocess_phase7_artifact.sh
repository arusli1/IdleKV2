#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

ARTIFACT="${1:?usage: postprocess_phase7_artifact.sh <artifact.json> <prefix> <title>}"
PREFIX="${2:?usage: postprocess_phase7_artifact.sh <artifact.json> <prefix> <title>}"
TITLE="${3:?usage: postprocess_phase7_artifact.sh <artifact.json> <prefix> <title>}"

OUTDIR="$ROOT/paper/figures"
mkdir -p "$OUTDIR"

echo "[export]"
"$PYTHON_BIN" "$ROOT/phases/phase6_repair/scripts/export_phase6_frontier.py" \
  --artifact "$ARTIFACT" \
  --outdir "$OUTDIR" \
  --prefix "$PREFIX"

OVERALL_CSV="$OUTDIR/${PREFIX}_overall.csv"
OVERALL_SVG="$OUTDIR/${PREFIX}_overall.svg"
RUNTIME_CSV="$OUTDIR/${PREFIX}_runtime_overall.csv"
RUNTIME_SVG="$OUTDIR/${PREFIX}_runtime_overall.svg"
OVERLAP_CSV="$OUTDIR/${PREFIX}_overlap_overall.csv"
OVERLAP_SVG="$OUTDIR/${PREFIX}_overlap_overall.svg"

echo
echo "[audit]"
"$PYTHON_BIN" "$ROOT/phases/phase7_broader_evidence/scripts/audit_phase7_artifact.py" \
  --artifact "$ARTIFACT" \
  --overall-csv "$OVERALL_CSV"

if [[ -f "$OVERALL_CSV" ]]; then
  echo
  echo "[plot]"
  "$PYTHON_BIN" "$ROOT/phases/phase7_broader_evidence/scripts/plot_frontier_svg.py" \
    --csv "$OVERALL_CSV" \
    --out "$OVERALL_SVG" \
    --title "$TITLE"
fi
if [[ -f "$RUNTIME_CSV" ]]; then
  echo
  echo "[runtime plot]"
  "$PYTHON_BIN" "$ROOT/phases/phase7_broader_evidence/scripts/plot_runtime_svg.py" \
    --csv "$RUNTIME_CSV" \
    --out "$RUNTIME_SVG" \
    --title "${TITLE} Runtime"
fi
if [[ -f "$OVERLAP_CSV" ]]; then
  echo
  echo "[overlap plot]"
  "$PYTHON_BIN" "$ROOT/phases/phase7_broader_evidence/scripts/plot_frontier_svg.py" \
    --csv "$OVERLAP_CSV" \
    --out "$OVERLAP_SVG" \
    --title "${TITLE} Overlap" \
    --suffix "_overlap"
fi

echo
echo "done:"
echo "  artifact: $ARTIFACT"
echo "  csv prefix: $OUTDIR/${PREFIX}_*"
if [[ -f "$OVERALL_SVG" ]]; then
  echo "  svg: $OVERALL_SVG"
fi
if [[ -f "$RUNTIME_SVG" ]]; then
  echo "  runtime svg: $RUNTIME_SVG"
fi
if [[ -f "$OVERLAP_SVG" ]]; then
  echo "  overlap svg: $OVERLAP_SVG"
fi
