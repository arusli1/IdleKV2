"""Structured logging for experiment results."""

import json
import sys
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    """
    Logs experiment results as structured JSON.

    Usage:
        logger = ExperimentLogger("results/")
        logger.log_result(
            experiment="idlekv_r0.5_budget1000",
            model="llama8b",
            seed=42,
            ruler_accuracy=0.92,
            longbench_score=40.1,
            kl_divergence=0.05,
            tokens_per_sec=45.2,
        )
        logger.flush()
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def log_result(self, **kwargs):
        entry = {
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.results.append(entry)
        # Also print to stdout for real-time monitoring
        print(json.dumps(entry, indent=None, default=str), file=sys.stderr)

    def flush(self, filename: str = None):
        if not filename:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to {path}")
        return path
