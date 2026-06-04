"""Pytest import setup for phase-local Phase 3 tests."""

from __future__ import annotations

import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_ROOT.parents[1]

for module_name in list(sys.modules):
    if module_name == "src" or module_name.startswith("src."):
        del sys.modules[module_name]

for root in (REPO_ROOT, PHASE_ROOT):
    if str(root) in sys.path:
        sys.path.remove(str(root))
for root in (REPO_ROOT, PHASE_ROOT):
    sys.path.insert(0, str(root))
