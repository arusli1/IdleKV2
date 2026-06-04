Continue the same background Phase 5 task.

Work root: `/home/ubuntu/RepairKV`

First inspect these paths:
- `/home/ubuntu/RepairKV/phases/phase5_gonogo/results/phase5_oracle/watchdog_done.json`
- `/home/ubuntu/RepairKV/phases/phase5_gonogo/results/phase5_oracle/watchdog/watchdog_status.json`
- `/home/ubuntu/RepairKV/phases/phase5_gonogo/results/phase5_oracle/watchdog/phase5_full_sweep.pid`
- `/home/ubuntu/RepairKV/phases/phase5_gonogo/results/phase5_oracle/watchdog/phase5_full_sweep.log`
- `/home/ubuntu/RepairKV/phases/phase5_gonogo/results/phase5_oracle`

Prefer using the watchdog helper for routine state refreshes and completion:
- `/home/ubuntu/RepairKV/.venv/bin/python phases/phase5_gonogo/scripts/phase5_watchdog.py status`
- `/home/ubuntu/RepairKV/.venv/bin/python phases/phase5_gonogo/scripts/phase5_watchdog.py finalize`

Resume policy:
- If the done marker already exists and the final artifacts look complete, verify that state, refresh the status file if needed, and exit.
- If the sweep process is healthy and still making progress, do not interrupt it. Update `watchdog_status.json`, wait 300 seconds, and check again.
- If the sweep stopped, failed, or stalled, diagnose from the log, fix the problem in the repo, validate the fix, and relaunch.
- Keep the user’s actual goal in mind: show improvement from pure eviction to repaired/oracle after eviction happens. Do not block completion solely on the monolithic-equivalence caveat if the sweep itself can run and the repair-vs-eviction results are valid.
- Only exit after writing `/home/ubuntu/RepairKV/phases/phase5_gonogo/results/phase5_oracle/watchdog_done.json`.
