#!/usr/bin/env python3
"""End-to-end Phase 3 runner."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch

PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_ROOT.parents[1]
for root in (PHASE_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from src.runtime import MODEL_DIR, RESULTS_DIR, ensure_results_dirs, load_model, load_tokenizer, run_live_smoke, write_json  # noqa: E402


def run_unittests() -> dict[str, object]:
    """Run the Phase 3 unittest suite and collect a compact summary."""
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
    """Read one JSON artifact from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ensure_results_dirs()
    test_summary = run_unittests()
    summary: dict[str, object] = {"tests": test_summary}

    if test_summary["successful"]:
        smoke_path = RESULTS_DIR / "phase3_live_smoke.json"
        if smoke_path.exists():
            summary["live_smoke"] = load_json_artifact(smoke_path)
        elif torch.cuda.is_available() and MODEL_DIR.exists():
            tokenizer = load_tokenizer()
            model = load_model()
            try:
                summary["live_smoke"] = run_live_smoke(model, tokenizer)
            finally:
                del model
                torch.cuda.empty_cache()

    write_json(RESULTS_DIR / "phase3_summary.json", summary)
    return 0 if test_summary["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
