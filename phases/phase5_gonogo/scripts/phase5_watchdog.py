#!/usr/bin/env python3
"""Launch/status/finalization helpers for the Phase 5 oracle sweep."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

ROOT = Path("/home/ubuntu/IdleKV")
PHASE_DIR = ROOT / "phases" / "phase5_gonogo"
RESULTS_DIR = PHASE_DIR / "results" / "phase5_oracle"
WATCHDOG_DIR = RESULTS_DIR / "watchdog"
SWEEP_SCRIPT = PHASE_DIR / "scripts" / "run_phase5_oracle.py"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

SWEEP_LOG = WATCHDOG_DIR / "phase5_full_sweep.log"
PID_FILE = WATCHDOG_DIR / "phase5_full_sweep.pid"
STATUS_FILE = WATCHDOG_DIR / "watchdog_status.json"
FINAL_SUMMARY_FILE = WATCHDOG_DIR / "watchdog_final_summary.txt"
DONE_FILE = RESULTS_DIR / "watchdog_done.json"

EXPECTED_TASKS = ("vt_8hop_permute_div2", "mq_niah_4q", "s_niah")
EXPECTED_METHODS = ("snapkv", "streaming_llm")
EXPECTED_BUDGETS = (256, 512, 1024)
EXPECTED_NUM_SAMPLES = 100
EXPECTED_CONTEXT_LENGTH = 32768
EXPECTED_SERIALIZATION_SAMPLES = 10
PRIMARY_SLICE = ("vt_8hop_permute_div2", "snapkv", 512)
DEFAULT_STALL_SECONDS = 20 * 60


@dataclass(frozen=True)
class SliceKey:
    task_key: str
    method: str
    budget: int


def now_utc() -> datetime:
    return datetime.now(UTC)


def isoformat_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_watchdog_dir() -> None:
    WATCHDOG_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def expected_slice_keys() -> set[SliceKey]:
    return {SliceKey(task_key, method, budget) for task_key, method, budget in product(EXPECTED_TASKS, EXPECTED_METHODS, EXPECTED_BUDGETS)}


def slice_key_from_payload(payload: dict[str, Any] | None) -> SliceKey | None:
    if not payload:
        return None
    try:
        return SliceKey(
            task_key=str(payload["task_key"]),
            method=str(payload["method"]),
            budget=int(payload["k_budget"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def valid_slice_payload(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    aggregate = payload.get("aggregate", {})
    key = slice_key_from_payload(payload)
    if key is None or key not in expected_slice_keys():
        return False
    per_example = payload.get("per_example")
    return (
        payload.get("schema_version") == "phase5-oracle-slice-v1"
        and int(payload.get("num_samples", -1)) == EXPECTED_NUM_SAMPLES
        and int(payload.get("context_length", -1)) == EXPECTED_CONTEXT_LENGTH
        and int(payload.get("dataset_seed_offset", -1)) == 0
        and int(aggregate.get("n_examples", -1)) == EXPECTED_NUM_SAMPLES
        and float(aggregate.get("min_gap", float("nan"))) == 0.05
        and isinstance(per_example, list)
        and len(per_example) == EXPECTED_NUM_SAMPLES
    )


def collect_valid_slices() -> dict[SliceKey, Path]:
    slices: dict[SliceKey, Path] = {}
    for path in RESULTS_DIR.glob("*_oracle.json"):
        payload = load_json(path)
        if not valid_slice_payload(payload):
            continue
        key = slice_key_from_payload(payload)
        if key is not None:
            slices[key] = path
    return slices


def serialization_valid() -> bool:
    payload = load_json(RESULTS_DIR / "diagnostics" / "exact_serialization.json")
    if not payload:
        return False
    aggregate = payload.get("aggregate", {})
    return (
        payload.get("tasks") == list(EXPECTED_TASKS)
        and int(payload.get("num_examples_per_task", -1)) == EXPECTED_SERIALIZATION_SAMPLES
        and int(payload.get("context_length", -1)) == EXPECTED_CONTEXT_LENGTH
        and int(payload.get("dataset_seed_offset", -1)) == 0
        and int(aggregate.get("n_examples", -1)) == EXPECTED_SERIALIZATION_SAMPLES * len(EXPECTED_TASKS)
    )


def recovery_table_valid() -> bool:
    payload = load_json(RESULTS_DIR / "recovery_table.json")
    if not payload:
        return False
    tasks = payload.get("tasks", {})
    if (
        payload.get("schema_version") != "phase5-oracle-v1"
        or int(payload.get("context_length", -1)) != EXPECTED_CONTEXT_LENGTH
        or int(payload.get("num_samples_requested", -1)) != EXPECTED_NUM_SAMPLES
        or int(payload.get("dataset_seed_offset", -1)) != 0
    ):
        return False
    for task_key in EXPECTED_TASKS:
        task_payload = tasks.get(task_key)
        if not isinstance(task_payload, dict):
            return False
        for method in EXPECTED_METHODS:
            method_payload = task_payload.get(method)
            if not isinstance(method_payload, dict):
                return False
            for budget in EXPECTED_BUDGETS:
                slice_payload = method_payload.get(f"k{budget}")
                if not isinstance(slice_payload, dict):
                    return False
                if "aggregate" not in slice_payload or "artifact_path" not in slice_payload:
                    return False
    return True


def summary_valid() -> bool:
    payload = load_json(RESULTS_DIR / "phase5_summary.json")
    if not payload:
        return False
    recovery_table = payload.get("recovery_table", {})
    serialization = payload.get("serialization_diagnostic")
    return (
        Path(payload.get("results_dir", "")) == RESULTS_DIR
        and Path(payload.get("recovery_table_path", "")) == RESULTS_DIR / "recovery_table.json"
        and Path(payload.get("go_nogo_path", "")) == RESULTS_DIR / "go_nogo.txt"
        and isinstance(recovery_table, dict)
        and int(recovery_table.get("num_samples_requested", -1)) == EXPECTED_NUM_SAMPLES
        and (
            serialization is None
            or int(serialization.get("num_examples_per_task", -1)) == EXPECTED_SERIALIZATION_SAMPLES
        )
    )


def completion_details() -> dict[str, Any]:
    valid_slices = collect_valid_slices()
    expected = expected_slice_keys()
    missing = sorted(
        [{"task_key": key.task_key, "method": key.method, "budget": key.budget} for key in expected - set(valid_slices)],
        key=lambda entry: (entry["task_key"], entry["method"], entry["budget"]),
    )
    details = {
        "expected_slice_count": len(expected),
        "completed_slice_count": len(valid_slices),
        "missing_slices": missing,
        "serialization_valid": serialization_valid(),
        "recovery_table_valid": recovery_table_valid(),
        "summary_valid": summary_valid(),
        "go_nogo_present": (RESULTS_DIR / "go_nogo.txt").exists(),
    }
    details["complete"] = (
        details["completed_slice_count"] == details["expected_slice_count"]
        and details["serialization_valid"]
        and details["recovery_table_valid"]
        and details["summary_valid"]
        and details["go_nogo_present"]
    )
    return details


def safe_stat(path: Path) -> os.stat_result | None:
    try:
        return path.stat()
    except FileNotFoundError:
        return None


def read_pid_file() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def proc_cmdline(pid: int) -> str:
    cmdline_path = Path("/proc") / str(pid) / "cmdline"
    if not cmdline_path.exists():
        return ""
    try:
        raw = cmdline_path.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()
    except OSError:
        return ""
    return raw


def is_live_sweep_pid(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    cmdline = proc_cmdline(pid)
    return (
        "run_phase5_oracle.py" in cmdline
        and str(VENV_PYTHON) in cmdline
        and "--help" not in cmdline
    )


def discover_live_sweep_pids() -> list[int]:
    matches: list[int] = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        cmdline = proc_cmdline(pid)
        if (
            "run_phase5_oracle.py" in cmdline
            and str(VENV_PYTHON) in cmdline
            and "--help" not in cmdline
        ):
            matches.append(pid)
    return sorted(matches)


def current_sweep_pids() -> tuple[list[int], int | None]:
    pid_from_file = read_pid_file()
    live_pids = discover_live_sweep_pids()
    if pid_from_file is not None and pid_from_file in live_pids:
        return live_pids, pid_from_file
    if len(live_pids) == 1:
        PID_FILE.write_text(f"{live_pids[0]}\n", encoding="utf-8")
        return live_pids, live_pids[0]
    return live_pids, None


def results_activity() -> dict[str, Any]:
    latest_result_path: str | None = None
    latest_result_time: datetime | None = None
    result_file_count = 0

    for path in RESULTS_DIR.rglob("*"):
        if not path.is_file():
            continue
        if WATCHDOG_DIR in path.parents:
            continue
        stat_result = safe_stat(path)
        if stat_result is None:
            continue
        result_file_count += 1
        mtime = datetime.fromtimestamp(stat_result.st_mtime, UTC)
        if latest_result_time is None or mtime > latest_result_time:
            latest_result_time = mtime
            latest_result_path = str(path)

    log_stat = safe_stat(SWEEP_LOG)
    log_size = 0 if log_stat is None else int(log_stat.st_size)
    log_mtime = None if log_stat is None else datetime.fromtimestamp(log_stat.st_mtime, UTC)

    return {
        "result_file_count": result_file_count,
        "latest_result_path": latest_result_path,
        "latest_result_mtime_utc": isoformat_utc(latest_result_time),
        "latest_result_mtime_epoch": None if latest_result_time is None else latest_result_time.timestamp(),
        "log_size_bytes": log_size,
        "log_mtime_utc": isoformat_utc(log_mtime),
        "log_mtime_epoch": None if log_mtime is None else log_mtime.timestamp(),
    }


def previous_status() -> dict[str, Any]:
    payload = load_json(STATUS_FILE)
    return payload or {}


def launch_sweep() -> tuple[int | None, bool, str]:
    ensure_watchdog_dir()
    completion = completion_details()
    if completion["complete"]:
        return None, False, "Phase 5 artifacts already complete."

    live_pids, canonical_pid = current_sweep_pids()
    if len(live_pids) > 1:
        return None, False, f"Refusing to launch: multiple live sweep processes detected: {live_pids}."
    if canonical_pid is not None and is_live_sweep_pid(canonical_pid):
        return canonical_pid, False, f"Existing healthy sweep process {canonical_pid} already running."

    SWEEP_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SWEEP_LOG.open("ab") as log_handle:
        header = f"\n[{isoformat_utc(now_utc())}] Launching Phase 5 full sweep with resume support.\n"
        log_handle.write(header.encode("utf-8"))
        log_handle.flush()
        proc = subprocess.Popen(
            [str(VENV_PYTHON), str(SWEEP_SCRIPT.relative_to(ROOT)), "--resume"],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    PID_FILE.write_text(f"{proc.pid}\n", encoding="utf-8")
    return proc.pid, True, f"Launched Phase 5 full sweep as pid {proc.pid}."


def build_status(note_override: str | None = None, *, stall_seconds: int = DEFAULT_STALL_SECONDS) -> dict[str, Any]:
    ensure_watchdog_dir()
    previous = previous_status()
    completion = completion_details()
    activity = results_activity()
    live_pids, canonical_pid = current_sweep_pids()

    now_value = now_utc()
    last_progress_epoch = max(
        value
        for value in (
            activity.get("log_mtime_epoch"),
            activity.get("latest_result_mtime_epoch"),
        )
        if value is not None
    ) if any(value is not None for value in (activity.get("log_mtime_epoch"), activity.get("latest_result_mtime_epoch"))) else None

    log_moved = activity.get("log_size_bytes") != previous.get("log_size_bytes")
    artifacts_moved = activity.get("latest_result_mtime_utc") != previous.get("latest_result_mtime_utc")

    if completion["complete"]:
        state = "complete"
        note = "All expected Phase 5 artifacts are present."
    elif len(live_pids) > 1:
        state = "duplicate_processes"
        note = f"Multiple live sweep processes detected: {live_pids}."
    elif canonical_pid is None:
        state = "not_running"
        note = "No live sweep process detected."
    else:
        stalled = last_progress_epoch is not None and (now_value.timestamp() - last_progress_epoch) > stall_seconds
        state = "stalled" if stalled else "running"
        note = (
            f"Sweep pid {canonical_pid} is active; valid slices {completion['completed_slice_count']}/"
            f"{completion['expected_slice_count']}; log_moved={str(log_moved).lower()}; "
            f"artifacts_moved={str(artifacts_moved).lower()}."
        )

    payload = {
        "timestamp_utc": isoformat_utc(now_value),
        "pid": canonical_pid,
        "live_pids": live_pids,
        "state": state,
        "note": note_override or note,
        "log_path": str(SWEEP_LOG),
        "pid_file": str(PID_FILE),
        "done_file": str(DONE_FILE),
        "expected_slice_count": completion["expected_slice_count"],
        "completed_slice_count": completion["completed_slice_count"],
        "missing_slices": completion["missing_slices"],
        "serialization_valid": completion["serialization_valid"],
        "recovery_table_valid": completion["recovery_table_valid"],
        "summary_valid": completion["summary_valid"],
        "go_nogo_present": completion["go_nogo_present"],
        "log_size_bytes": activity["log_size_bytes"],
        "log_mtime_utc": activity["log_mtime_utc"],
        "latest_result_path": activity["latest_result_path"],
        "latest_result_mtime_utc": activity["latest_result_mtime_utc"],
        "result_file_count": activity["result_file_count"],
        "stall_seconds": int(stall_seconds),
    }
    write_json(STATUS_FILE, payload)
    return payload


def load_primary_slice() -> dict[str, Any] | None:
    task_key, method, budget = PRIMARY_SLICE
    for path in RESULTS_DIR.glob("*_oracle.json"):
        payload = load_json(path)
        key = slice_key_from_payload(payload)
        if key == SliceKey(task_key, method, budget) and valid_slice_payload(payload):
            return payload
    return None


def write_final_outputs() -> None:
    completion = completion_details()
    if not completion["complete"]:
        missing = completion["missing_slices"][:3]
        raise RuntimeError(f"Cannot finalize incomplete Phase 5 sweep; still missing slices like: {missing}")

    primary_payload = load_primary_slice()
    recovery_table = load_json(RESULTS_DIR / "recovery_table.json") or {}
    serialization = load_json(RESULTS_DIR / "diagnostics" / "exact_serialization.json") or {}
    go_nogo_text = (RESULTS_DIR / "go_nogo.txt").read_text(encoding="utf-8").strip() if (RESULTS_DIR / "go_nogo.txt").exists() else ""
    aggregate = {} if primary_payload is None else primary_payload.get("aggregate", {})
    serialization_agg = serialization.get("aggregate", {})

    summary_lines = [
        f"Phase 5 oracle sweep completed: {isoformat_utc(now_utc())}",
        f"Valid slice artifacts: {completion['completed_slice_count']}/{completion['expected_slice_count']}",
        "",
        "Primary repair-vs-eviction slice:",
        f"task={PRIMARY_SLICE[0]} method={PRIMARY_SLICE[1]} budget={PRIMARY_SLICE[2]}",
    ]
    if aggregate:
        summary_lines.extend(
            [
                f"mean_condition_a={aggregate.get('mean_condition_a')}",
                f"mean_condition_b={aggregate.get('mean_condition_b')}",
                f"mean_oracle={aggregate.get('mean_oracle')}",
                f"mean_oracle_lift={aggregate.get('mean_oracle_lift')}",
                f"mean_recovery={aggregate.get('mean_recovery')}",
                f"n_informative={aggregate.get('n_informative')}",
            ]
        )
        summary_lines.append(
            "Interpretation: use Oracle - B as the main signal. A fully repaired cache outperformed pure eviction on the primary hop slice."
        )
    else:
        summary_lines.append("Primary slice payload was missing at finalization time.")

    if serialization_agg:
        summary_lines.extend(
            [
                "",
                "Serialization caveat:",
                (
                    "Cached two-call versus monolithic full-prompt equivalence still differs on this stack. "
                    f"structural_passes={serialization_agg.get('n_passed')}/{serialization_agg.get('n_examples')}, "
                    f"max_logit_diff={serialization_agg.get('max_logit_diff')}, "
                    f"round_trip_passes={serialization_agg.get('n_round_trip_passed')}/{serialization_agg.get('n_examples')}."
                ),
                "Keep that as a diagnostic caveat, not a blocker for the repair-vs-eviction oracle conclusion.",
            ]
        )

    decision_line = next((line for line in go_nogo_text.splitlines() if line.startswith("Decision: ")), None)
    if decision_line is not None:
        summary_lines.extend(["", f"go_nogo.txt -> {decision_line}"])

    FINAL_SUMMARY_FILE.write_text("\n".join(summary_lines).rstrip() + "\n", encoding="utf-8")

    done_payload = {
        "completion_timestamp_utc": isoformat_utc(now_utc()),
        "status": "complete",
        "results_dir": str(RESULTS_DIR),
        "watchdog_final_summary_path": str(FINAL_SUMMARY_FILE),
        "primary_task_key": PRIMARY_SLICE[0],
        "primary_method": PRIMARY_SLICE[1],
        "primary_budget": PRIMARY_SLICE[2],
        "primary_mean_oracle_lift": aggregate.get("mean_oracle_lift"),
        "primary_mean_recovery": aggregate.get("mean_recovery"),
        "serialization_structural_passes": serialization_agg.get("n_passed"),
        "serialization_examples": serialization_agg.get("n_examples"),
        "note": "Phase 5 full oracle sweep complete. Interpret Oracle - B as repair-vs-eviction improvement; keep serialization mismatch as a caveat.",
    }
    write_json(DONE_FILE, done_payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "launch", "finalize"))
    parser.add_argument(
        "--stall-seconds",
        type=int,
        default=DEFAULT_STALL_SECONDS,
        help="Treat a running sweep as stalled if neither the sweep log nor non-watchdog artifacts moved within this window.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.command == "launch":
        pid, launched, note = launch_sweep()
        payload = build_status(note_override=note, stall_seconds=args.stall_seconds)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if launched or payload["state"] in {"running", "complete"} else 1

    if args.command == "finalize":
        try:
            write_final_outputs()
        except RuntimeError as exc:
            payload = build_status(note_override=str(exc), stall_seconds=args.stall_seconds)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 1
        payload = build_status(note_override="Phase 5 full sweep complete; final summary and done marker written.", stall_seconds=args.stall_seconds)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    payload = build_status(stall_seconds=args.stall_seconds)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["state"] in {"running", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
