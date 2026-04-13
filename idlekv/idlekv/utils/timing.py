"""
GPU timing utilities using CUDA events for accurate measurement.

CUDA events are the correct way to time GPU operations — time.perf_counter()
measures CPU time and misses async GPU work.
"""

import torch
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class TimingRecord:
    name: str
    elapsed_ms: float


class GPUTimer:
    """
    Accumulates GPU-timed measurements using CUDA events.

    Usage:
        timer = GPUTimer()
        with timer.measure("phase1"):
            phase1_rescore(...)
        with timer.measure("phase2_layer_0"):
            ...
        timer.summary()
    """

    def __init__(self):
        self.records: list[TimingRecord] = []

    @contextmanager
    def measure(self, name: str):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)  # milliseconds
        self.records.append(TimingRecord(name=name, elapsed_ms=elapsed))

    def summary(self) -> dict:
        result = {}
        for r in self.records:
            if r.name not in result:
                result[r.name] = {"count": 0, "total_ms": 0, "measurements": []}
            result[r.name]["count"] += 1
            result[r.name]["total_ms"] += r.elapsed_ms
            result[r.name]["measurements"].append(r.elapsed_ms)
        for v in result.values():
            v["mean_ms"] = v["total_ms"] / v["count"]
        return result

    def reset(self):
        self.records = []
