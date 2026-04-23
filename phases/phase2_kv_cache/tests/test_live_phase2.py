"""Live-Qwen integration tests for the Phase 2 acceptance checks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PHASE_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_ROOT))

from src.runtime import MODEL_DIR, load_model, load_tokenizer, run_round_trip_identity, run_selective_injection, run_transfer_latency


@unittest.skipUnless(torch.cuda.is_available() and MODEL_DIR.exists(), "Live Phase 2 tests require local Qwen on CUDA.")
class LivePhase2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = load_tokenizer()
        cls.model = load_model()

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.model
        torch.cuda.empty_cache()

    def test_round_trip_identity(self) -> None:
        result = run_round_trip_identity(self.model, self.tokenizer)
        self.assertTrue(result["pass"])
        self.assertLess(result["max_abs_logit_diff"], 1e-3)

    def test_selective_injection(self) -> None:
        result = run_selective_injection(self.model, self.tokenizer)
        self.assertLess(result["reference_vs_restored_max_abs_logit_diff"], 1e-3)
        self.assertTrue(result["recovered_text_match"])
        self.assertTrue(result["degraded_text_differs"] or result["reference_vs_degraded_max_abs_logit_diff"] > 1e-4)

    def test_transfer_latency(self) -> None:
        result = run_transfer_latency(self.model, self.tokenizer)
        self.assertIn("1000", result["test_sizes"])
        self.assertGreater(result["test_sizes"]["1000"]["restore"]["p50_ms"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
