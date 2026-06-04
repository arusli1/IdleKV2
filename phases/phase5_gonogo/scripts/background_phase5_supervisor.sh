#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/IdleKV"
PHASE_DIR="$ROOT/phases/phase5_gonogo"
RESULTS_DIR="$PHASE_DIR/results/phase5_oracle"
WATCH_DIR="$RESULTS_DIR/watchdog"
ATTEMPT_DIR="$WATCH_DIR/attempts"
PROMPT_FILE="$PHASE_DIR/scripts/background_phase5_codex_prompt.md"
RESUME_PROMPT_FILE="$PHASE_DIR/scripts/background_phase5_codex_resume.md"
THREAD_FILE="$WATCH_DIR/codex_thread_id.txt"
LAST_MESSAGE_FILE="$WATCH_DIR/last_message.txt"
SUPERVISOR_LOG="$WATCH_DIR/supervisor.log"
DONE_FILE="$RESULTS_DIR/watchdog_done.json"

mkdir -p "$ATTEMPT_DIR"
cd "$ROOT"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf "[%s] %s\n" "$(timestamp)" "$*" | tee -a "$SUPERVISOR_LOG"
}

extract_thread_id() {
  local attempt_file="$1"
  grep -m1 '"type":"thread.started"' "$attempt_file" | sed -E 's/.*"thread_id":"([^"]+)".*/\1/' || true
}

run_new_worker() {
  local attempt_file="$1"
  local exit_code

  log "Starting new Codex background worker."
  set +e
  codex exec \
    --json \
    --dangerously-bypass-approvals-and-sandbox \
    -C "$ROOT" \
    -o "$LAST_MESSAGE_FILE" \
    - < "$PROMPT_FILE" 2>&1 | tee "$attempt_file" >> "$SUPERVISOR_LOG"
  exit_code=${PIPESTATUS[0]}
  set -e

  if [[ ! -s "$THREAD_FILE" ]]; then
    local thread_id
    thread_id="$(extract_thread_id "$attempt_file")"
    if [[ -n "$thread_id" ]]; then
      printf "%s\n" "$thread_id" > "$THREAD_FILE"
      log "Stored Codex thread id $thread_id."
    else
      log "Warning: failed to extract Codex thread id from $attempt_file."
    fi
  fi

  return "$exit_code"
}

run_resume_worker() {
  local attempt_file="$1"
  local thread_id
  local exit_code

  thread_id="$(<"$THREAD_FILE")"
  log "Resuming Codex background worker thread $thread_id."
  set +e
  codex exec resume \
    "$thread_id" \
    --json \
    --dangerously-bypass-approvals-and-sandbox \
    -o "$LAST_MESSAGE_FILE" \
    - < "$RESUME_PROMPT_FILE" 2>&1 | tee "$attempt_file" >> "$SUPERVISOR_LOG"
  exit_code=${PIPESTATUS[0]}
  set -e

  if [[ "$exit_code" -ne 0 ]] && grep -Eqi 'not found|no matching session|unknown session|unknown thread' "$attempt_file"; then
    log "Stored thread id looks invalid now; clearing it so the next cycle starts a fresh worker."
    rm -f "$THREAD_FILE"
  fi

  return "$exit_code"
}

log "Phase 5 background supervisor booted."
log "Results dir: $RESULTS_DIR"
log "Watchdog dir: $WATCH_DIR"

while true; do
  if [[ -f "$DONE_FILE" ]]; then
    log "Done marker found at $DONE_FILE. Exiting supervisor."
    break
  fi

  stamp="$(date -u +"%Y%m%dT%H%M%SZ")"
  attempt_file="$ATTEMPT_DIR/$stamp.jsonl"

  if [[ -s "$THREAD_FILE" ]]; then
    if run_resume_worker "$attempt_file"; then
      log "Codex worker exited with code 0."
    else
      rc=$?
      log "Codex worker exited with code $rc."
    fi
  else
    if run_new_worker "$attempt_file"; then
      log "Codex worker exited with code 0."
    else
      rc=$?
      log "Codex worker exited with code $rc."
    fi
  fi

  if [[ -f "$DONE_FILE" ]]; then
    log "Done marker detected after worker exit. Exiting supervisor."
    break
  fi

  log "Sleeping 300 seconds before the next supervisor cycle."
  sleep 300
done
