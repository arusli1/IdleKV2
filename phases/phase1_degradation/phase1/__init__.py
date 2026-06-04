"""Small public surface for the Phase 1 package."""

# Re-export the canonical defaults so callers can import from the package root.
from .config import DEFAULT_ALGORITHMS, DEFAULT_BUDGETS, DEFAULT_CONTEXT_LENGTHS, DEFAULT_TASKS

# Explicit export list keeps the package API narrow and predictable.
__all__ = [
    # Algorithms the phase-1 runner should consider by default.
    "DEFAULT_ALGORITHMS",
    # Budget tiers used by default across evaluations.
    "DEFAULT_BUDGETS",
    # Context window sizes that define the evaluation regimes.
    "DEFAULT_CONTEXT_LENGTHS",
    # Task identifiers included in the phase-1 suite.
    "DEFAULT_TASKS",
]
