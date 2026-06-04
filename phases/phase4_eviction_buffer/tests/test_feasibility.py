"""Unit tests for feasibility-frontier helpers."""

from __future__ import annotations

import unittest

from phases.phase4_eviction_buffer.src.buffer.feasibility import compute_feasibility_frontier


class FeasibilityFrontierTests(unittest.TestCase):
    def test_frontier_is_monotone_in_tool_call_duration(self) -> None:
        transfer = {
            50: {"p90_ms": 40.0},
            100: {"p90_ms": 80.0},
            250: {"p90_ms": 220.0},
        }
        scoring = {"l2_norm": {1000: {"p50_ms": 10.0}}}

        frontier = compute_feasibility_frontier(
            transfer,
            scoring,
            strategy="l2_norm",
            tool_call_durations_s=(0.1, 0.5, 1.0),
        )

        self.assertEqual(frontier[0.1]["max_K"], 100)
        self.assertEqual(frontier[0.5]["max_K"], 250)
        self.assertLessEqual(frontier[0.1]["max_K"], frontier[0.5]["max_K"])
        self.assertLessEqual(frontier[0.5]["max_K"], frontier[1.0]["max_K"])


if __name__ == "__main__":
    unittest.main()
