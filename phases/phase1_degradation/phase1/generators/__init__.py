"""Synthetic-task builders used by the Phase 1 benchmark package.

This module is intentionally small and only re-exports the public factory
helpers for each synthetic task type so call sites have a single import
location that stays stable even if internal modules move around.
"""

# Import the concrete builders to expose them at the package level.
from .fwe import build_fwe_example
from .mk_niah import build_cross_turn_mk_niah_example
from .niah import build_niah_example
from .vt import build_vt_example

# Public re-export list for the package namespace.
__all__ = [
    "build_cross_turn_mk_niah_example",
    "build_fwe_example",
    "build_niah_example",
    "build_vt_example",
]
