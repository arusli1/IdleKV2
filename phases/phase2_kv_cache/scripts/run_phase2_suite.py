#!/usr/bin/env python3
"""End-to-end Phase 2 runner."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_ROOT))

from src.runtime import (  # noqa: E402
    RESULTS_DIR,
    ensure_results_dirs,
    generate_attention_heatmaps,
    inspect_environment,
    load_model,
    load_tokenizer,
    write_json,
    write_markdown_report,
)


def run_unittests() -> dict[str, object]:
    """Run the Phase 2 unittest suite and return a small summary."""
    suite = unittest.defaultTestLoader.discover(str(PHASE_ROOT / "tests"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "successful": result.wasSuccessful(),
    }


def load_json_artifact(path: Path) -> dict[str, object]:
    """Read one JSON artifact that should already exist after the tests."""
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ensure_results_dirs()
    test_summary = run_unittests()
    summary: dict[str, object] = {"tests": test_summary}

    if test_summary["successful"]:
        tokenizer = load_tokenizer()
        model = load_model()
        try:
            summary["environment"] = inspect_environment(model, tokenizer)
            summary["heatmaps"] = generate_attention_heatmaps(model, tokenizer)
        finally:
            del model

        summary["round_trip"] = load_json_artifact(RESULTS_DIR / "phase2_round_trip.json")
        summary["injection"] = load_json_artifact(RESULTS_DIR / "phase2_injection.json")
        summary["transfer_latency"] = load_json_artifact(RESULTS_DIR / "phase2_transfer_latency.json")
        write_json(RESULTS_DIR / "phase2_summary.json", summary)
        write_markdown_report(summary)
        return 0

    write_json(RESULTS_DIR / "phase2_summary.json", summary)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
