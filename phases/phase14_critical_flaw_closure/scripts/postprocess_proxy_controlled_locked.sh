#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

CSV_PATH="${1:-phases/phase14_critical_flaw_closure/results/proxy_controlled_locked_n100.csv}"

if [[ ! -f "${CSV_PATH}" ]]; then
  echo "[phase14-proxy-postprocess] missing CSV: ${CSV_PATH}" >&2
  exit 1
fi

echo "[phase14-proxy-postprocess] evaluating ${CSV_PATH}"
.venv/bin/python phases/phase14_critical_flaw_closure/scripts/evaluate_proxy_controlled_smoke.py \
  --summary-csv "${CSV_PATH}"

echo "[phase14-proxy-postprocess] running readiness audit"
.venv/bin/python phases/phase14_critical_flaw_closure/scripts/audit_phase14_readiness.py

echo "[phase14-proxy-postprocess] exporting controlled proxy CSV to paper/figures"
cp "${CSV_PATH}" paper/figures/proxy_controlled_locked_n100.csv

echo "[phase14-proxy-postprocess] rendering paper figures"
.venv/bin/python paper/scripts/render_paper_figures.py

echo "[phase14-proxy-postprocess] rebuilding paper/main.pdf"
(
  cd paper
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
)

echo "[phase14-proxy-postprocess] checking LaTeX log"
if rg -n "undefined|Citation.*undefined|Overfull|LaTeX Warning: Reference|Package natbib Warning" paper/aux/main.log; then
  echo "[phase14-proxy-postprocess] found blocking LaTeX warnings" >&2
  exit 1
fi

echo "[phase14-proxy-postprocess] done"
