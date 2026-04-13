"""
Tool-call duration distributions calibrated to real agentic traces.

Source: Continuum (Li et al., Nov 2025) SWE-Bench measurements.
- Simple tools (cat, sed, grep): 50-500ms
- Medium tools (python scripts): 500ms-5s
- Heavy tools (pytest, builds): 5-30s
- Overall: log-normal, median ~1s
"""

import numpy as np


def sample_tool_duration(
    rng: np.random.RandomState,
    median_ms: float = 1000,
    min_ms: float = 100,
    max_ms: float = 10000,
    sigma: float = 1.0,
) -> float:
    """
    Sample a tool-call duration from a log-normal distribution.

    Args:
        rng: numpy random state for reproducibility
        median_ms: median duration in ms (log-normal mu = log(median))
        min_ms: clamp minimum
        max_ms: clamp maximum
        sigma: log-normal sigma (spread)

    Returns:
        Duration in milliseconds
    """
    mu = np.log(median_ms)
    duration = rng.lognormal(mean=mu, sigma=sigma)
    return float(np.clip(duration, min_ms, max_ms))
