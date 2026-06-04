"""Public eviction helpers exposed to the rest of the Phase 1 package."""

# Re-export the factory used to assemble eviction policies for external callers.
from .policies import build_press
# Re-export the trace recorder so other modules can subscribe to eviction events.
from .tracing import EvictionTraceRecorder

# Limit the public surface of this package to stable, documented helpers.
__all__ = ["EvictionTraceRecorder", "build_press"]
