"""Live-Qwen smoke test for the Phase 3 eviction stack."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_ROOT.parents[1]
for root in (PHASE_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from src.runtime import MODEL_DIR, load_model, load_tokenizer, run_live_smoke


@unittest.skipUnless(torch.cuda.is_available() and MODEL_DIR.exists(), "Live Phase 3 tests require local Qwen on CUDA.")
class LivePhase3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = load_tokenizer()
        cls.model = load_model()

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.model
        torch.cuda.empty_cache()

    def test_live_smoke(self) -> None:
        result = run_live_smoke(self.model, self.tokenizer, context_tokens=384, k_budget=96)

        self.assertEqual(result["context_tokens"], 384)
        self.assertIn("snapkv", result["policies"])
        self.assertIn("query_aware_snapkv", result["policies"])
        self.assertIn("streaming_llm", result["policies"])

        for policy_summary in result["policies"].values():
            self.assertEqual(policy_summary["kept_count"], 96)
            self.assertEqual(policy_summary["evicted_count"], 384 - 96)
            self.assertEqual(policy_summary["evicted_device"], "cpu")
            self.assertGreater(policy_summary["obs_window_q_vecs_shape"][0], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
