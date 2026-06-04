"""Pytest import setup for phase-local Phase 1 tests."""

from __future__ import annotations

import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_ROOT))
