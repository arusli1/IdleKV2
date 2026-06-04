#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

RESULTS_DIR="phases/phase14_critical_flaw_closure/results"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

QUEUE_LOG="${LOG_DIR}/phase14_queued_closure_$(date -u +%Y%m%dT%H%M%SZ).log"

log() {
  echo "[phase14-queue] $(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" | tee -a "${QUEUE_LOG}"
}

run_logged() {
  log "run: $*"
  "$@" 2>&1 | tee -a "${QUEUE_LOG}"
}

wait_for_tmux_session() {
  local session="$1"
  if tmux has-session -t "${session}" 2>/dev/null; then
    log "waiting for tmux session ${session}"
    while tmux has-session -t "${session}" 2>/dev/null; do
      sleep 60
    done
    log "tmux session ${session} finished"
  else
    log "tmux session ${session} not present; continuing"
  fi
}

eval_smoke() {
  local kind="$1"
  local csv_path="$2"
  run_logged .venv/bin/python phases/phase14_critical_flaw_closure/scripts/evaluate_phase14_smokes.py \
    --kind "${kind}" \
    --summary-csv "${csv_path}"
}

main() {
  log "start queued closure at commit $(git rev-parse --short HEAD)"

  wait_for_tmux_session "phase14_proxy_locked"

  if [[ -f "${RESULTS_DIR}/proxy_controlled_locked_n100.csv" ]]; then
    run_logged bash phases/phase14_critical_flaw_closure/scripts/postprocess_proxy_controlled_locked.sh
  else
    log "missing controlled proxy CSV; stopping queue before downstream GPU work"
    exit 1
  fi

  if [[ -d "${PHASE14_LLAMA_MODEL_DIR:-models/Llama-3.1-8B-Instruct}" ]]; then
    run_logged bash phases/phase14_critical_flaw_closure/scripts/run_llama_calibrated_smoke.sh
    if eval_smoke llama "${RESULTS_DIR}/llama_calibrated_smoke_n${PHASE14_LLAMA_SMOKE_N:-2}_b${PHASE14_LLAMA_BASE_BUDGET:-16384}.csv"; then
      log "Llama smoke passed; launching locked calibrated Llama run"
      run_logged bash phases/phase14_critical_flaw_closure/scripts/run_llama_calibrated_locked.sh
      eval_smoke llama "${RESULTS_DIR}/llama_calibrated_locked_n${PHASE14_LLAMA_LOCKED_N:-24}_b${PHASE14_LLAMA_BASE_BUDGET:-16384}.csv" || true
    else
      log "Llama smoke did not pass; skipping locked Llama run"
    fi
  else
    log "Llama model directory not found; skipping Llama branch"
  fi

  run_logged bash phases/phase14_critical_flaw_closure/scripts/run_selector_variant_smoke.sh
  if eval_smoke selector "${RESULTS_DIR}/selector_variant_smoke_n${PHASE14_SELECTOR_SMOKE_N:-1}.csv"; then
    log "selector smoke passed; launching locked selector-variant run"
    run_logged bash phases/phase14_critical_flaw_closure/scripts/run_selector_variant_locked.sh
    eval_smoke selector "${RESULTS_DIR}/selector_variant_locked_n${PHASE14_SELECTOR_LOCKED_N:-24}.csv" || true
  else
    log "selector smoke did not pass; skipping locked selector-variant run"
  fi

  run_logged .venv/bin/python phases/phase14_critical_flaw_closure/scripts/audit_phase14_readiness.py
  run_logged .venv/bin/python -m pytest phases/phase14_critical_flaw_closure/tests/test_audit_phase14_readiness.py \
    phases/phase13_iteration_framework/tests/test_paper_language.py \
    phases/phase9_experiment_deepening/tests/test_paper_figure_renderer.py -q

  log "queued closure done"
}

main "$@"
