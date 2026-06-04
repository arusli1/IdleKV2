#!/usr/bin/env python3
"""Summarize live Phase 14 controlled-proxy progress from a log file."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re


EVENT_RE = re.compile(r"\[(mq_niah_[^:]+):ex(\d+)\].*?(?=(?:\n\[mq_niah_)|\Z)", re.S)
START_RE = re.compile(r"\[phase14-proxy-locked\] start (?P<stamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC")
CONFIG_RE = re.compile(r"\[phase14-proxy-locked\] n=(?P<n>\d+) K=(?P<ks>[0-9 ]+)")
K_RE = re.compile(
    r"k=(?P<k>\d+):Bm=(?P<b_match>[0-9.]+)/I=(?P<idlekv>[0-9.]+)/"
    r"R=(?P<random_k>[0-9.]+)/O=(?P<oldest_k>[0-9.]+)/Or=(?P<gold_k>[0-9.]+)"
)
EXPECTED_SPLITS = {
    "mq_niah_4q": 3,
    "mq_niah_6q": 4,
}


@dataclass
class KSummary:
    count: int = 0
    b_match: float = 0.0
    idlekv: float = 0.0
    random_k: float = 0.0
    oldest_k: float = 0.0
    gold_k: float = 0.0

    def add(self, *, b_match: float, idlekv: float, random_k: float, oldest_k: float, gold_k: float) -> None:
        self.count += 1
        self.b_match += float(b_match)
        self.idlekv += float(idlekv)
        self.random_k += float(random_k)
        self.oldest_k += float(oldest_k)
        self.gold_k += float(gold_k)

    def mean(self, field: str) -> float:
        if self.count == 0:
            return 0.0
        return float(getattr(self, field)) / float(self.count)


def task_family(task: str) -> str:
    return str(task).split("_split_", 1)[0]


def parse_proxy_log(text: str) -> tuple[dict[str, int], dict[tuple[str, int], KSummary]]:
    progress: dict[str, int] = {}
    summaries: dict[tuple[str, int], KSummary] = {}
    for event in EVENT_RE.finditer(text):
        task = event.group(1)
        family = task_family(task)
        progress[family] = max(progress.get(family, 0), int(event.group(2)))
        body = event.group(0)
        for match in K_RE.finditer(body):
            k = int(match.group("k"))
            key = (family, k)
            summaries.setdefault(key, KSummary()).add(
                b_match=float(match.group("b_match")),
                idlekv=float(match.group("idlekv")),
                random_k=float(match.group("random_k")),
                oldest_k=float(match.group("oldest_k")),
                gold_k=float(match.group("gold_k")),
            )
    return progress, summaries


def parse_run_metadata(text: str) -> tuple[datetime | None, int | None]:
    start: datetime | None = None
    num_samples: int | None = None
    if start_match := START_RE.search(text):
        start = datetime.strptime(start_match.group("stamp"), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    if config_match := CONFIG_RE.search(text):
        num_samples = int(config_match.group("n"))
    return start, num_samples


def latest_log(results_dir: Path) -> Path:
    logs = sorted((results_dir / "logs").glob("proxy_controlled_locked_*.log"))
    if not logs:
        raise FileNotFoundError(f"No proxy_controlled_locked_*.log under {results_dir / 'logs'}")
    return logs[-1]


def render_summary(progress: dict[str, int], summaries: dict[tuple[str, int], KSummary]) -> str:
    lines: list[str] = []
    for family in sorted(progress):
        lines.append(f"{family}: max_ex={progress[family]}")
        for family_key, k in sorted(summaries):
            if family_key != family:
                continue
            row = summaries[(family_key, k)]
            b_match = row.mean("b_match")
            idlekv = row.mean("idlekv")
            random_k = row.mean("random_k")
            oldest_k = row.mean("oldest_k")
            gold_k = row.mean("gold_k")
            max_control_lift = max(random_k - b_match, oldest_k - b_match)
            lines.append(
                "  "
                f"K={k}: n={row.count} "
                f"Bm={b_match:.3f} IdleKV={idlekv:.3f} "
                f"Rand={random_k:.3f} Old={oldest_k:.3f} Gold={gold_k:.3f} "
                f"lift={idlekv - b_match:.3f} max_control_lift={max_control_lift:.3f}"
            )
    return "\n".join(lines)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def render_eta(
    *,
    start: datetime | None,
    num_samples: int | None,
    summaries: dict[tuple[str, int], KSummary],
    now: datetime | None = None,
) -> str:
    if start is None or num_samples is None:
        return "eta: unavailable; missing run start or sample count"

    now = now or datetime.now(timezone.utc)
    elapsed = max(0.0, (now - start).total_seconds())
    completed = 0
    expected = 0
    for family, splits in EXPECTED_SPLITS.items():
        family_counts = [summary.count for (summary_family, _), summary in summaries.items() if summary_family == family]
        completed += max(family_counts, default=0)
        expected += splits * num_samples

    if completed <= 0 or expected <= 0:
        return "eta: warming up"
    remaining = max(0, expected - completed)
    seconds_per_case = elapsed / float(completed)
    eta = remaining * seconds_per_case
    return (
        f"eta: completed {completed}/{expected} example-split rows "
        f"({completed / expected:.0%}); elapsed {format_duration(elapsed)}, "
        f"rough remaining {format_duration(eta)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("phases/phase14_critical_flaw_closure/results"),
    )
    parser.add_argument("--log", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = args.log if args.log is not None else latest_log(args.results_dir)
    text = log_path.read_text(errors="ignore")
    progress, summaries = parse_proxy_log(text)
    start, num_samples = parse_run_metadata(text)
    print(log_path)
    print(render_summary(progress, summaries))
    print(render_eta(start=start, num_samples=num_samples, summaries=summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
