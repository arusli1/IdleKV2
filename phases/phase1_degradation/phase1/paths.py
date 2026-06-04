"""Canonical filesystem locations used by the Phase 1 package."""

from __future__ import annotations

from pathlib import Path


# Base roots used to anchor all other locations.
PHASE1_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE1_ROOT.parents[1]

# External assets that live outside the Phase 1 package directory.
MODEL_DIR = REPO_ROOT / "models" / "Qwen2.5-7B-Instruct"
RULER_JSON_DIR = REPO_ROOT / "ruler" / "scripts" / "data" / "synthetic" / "json"

# Phase 1 runtime output locations (created/used by this package).
ARTIFACTS_DIR = PHASE1_ROOT / "artifacts"
RESULTS_DIR = PHASE1_ROOT / "results"
LOGS_DIR = PHASE1_ROOT / "logs"
