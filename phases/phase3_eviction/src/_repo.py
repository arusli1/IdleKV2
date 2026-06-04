"""Shared path helpers for Phase 3 modules."""

from __future__ import annotations

import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_ROOT.parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

__all__ = ["PHASE_ROOT", "REPO_ROOT"]
