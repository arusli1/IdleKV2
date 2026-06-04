#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

SUMMARY_CSV="phases/phase10_expansion/results/mq_niah_8q_frontier_n24.csv"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "missing ${SUMMARY_CSV}; wait for the 8Q frontier run to finish" >&2
  exit 1
fi

ARTIFACT="$(
  .venv/bin/python - <<'PY'
import csv
from pathlib import Path

summary = Path("phases/phase10_expansion/results/mq_niah_8q_frontier_n24.csv")
with summary.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    first = next(reader, None)
if first is None or not first.get("artifact"):
    raise SystemExit("summary CSV has no artifact column/value")
print(first["artifact"])
PY
)"

if [[ ! -f "${ARTIFACT}" ]]; then
  echo "artifact not found: ${ARTIFACT}" >&2
  exit 1
fi

.venv/bin/python phases/phase6_repair/scripts/export_phase6_frontier.py \
  --artifact "${ARTIFACT}" \
  --outdir paper/figures \
  --prefix phase10_mq_niah_8q_frontier_n24

.venv/bin/python paper/scripts/render_paper_figures.py

(
  cd paper
  pdflatex -interaction=nonstopmode -halt-on-error main.tex >/tmp/idlekv_pdflatex1.log
  pdflatex -interaction=nonstopmode -halt-on-error main.tex >/tmp/idlekv_pdflatex2.log
  rm -f main.aux main.log main.out main.bbl main.blg main.fdb_latexmk main.fls main.synctex.gz
  rm -f aux/main.aux aux/main.log aux/main.out aux/main.bbl aux/main.blg aux/main.fdb_latexmk aux/main.fls aux/main.synctex.gz
)
