from __future__ import annotations

import sys
import unittest
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
for root in (PHASE_ROOT,):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from phase1.modeling import load_tokenizer
from phase1.paths import MODEL_DIR
from phase1.task_registry import build_task_example


class NIAHGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = load_tokenizer(str(MODEL_DIR))

    def test_mq_niah_6q_builds_six_needles_with_tail_anchored_last_needle(self) -> None:
        example = build_task_example("mq_niah_6q", 0, 1024, self.tokenizer)

        self.assertEqual(example.task_name, "mq_niah_6q")
        self.assertEqual(example.task_family, "niah")
        self.assertEqual(len(example.outputs), 6)
        self.assertEqual(len(example.metadata["query_keys"]), 6)
        self.assertEqual(example.metadata["num_needles"], 6)
        self.assertEqual(example.metadata["tail_anchored_needles"], 1)
        self.assertEqual(len(example.relevant_spans), 6)
        self.assertEqual([span.name for span in example.relevant_spans], [f"needle_{i}" for i in range(1, 7)])
        self.assertEqual(example.relevant_spans[-1].depth_fraction, 1.0)
        self.assertTrue(all("key" in span.metadata and "value" in span.metadata for span in example.relevant_spans))

    def test_dataset_seed_offset_changes_mq_niah_6q_keys_and_outputs(self) -> None:
        base_example = build_task_example("mq_niah_6q", 0, 1024, self.tokenizer, dataset_seed_offset=0)
        same_seed_example = build_task_example("mq_niah_6q", 0, 1024, self.tokenizer, dataset_seed_offset=0)
        shifted_example = build_task_example("mq_niah_6q", 0, 1024, self.tokenizer, dataset_seed_offset=1)

        self.assertEqual(base_example.outputs, same_seed_example.outputs)
        self.assertEqual(base_example.metadata["query_keys"], same_seed_example.metadata["query_keys"])
        self.assertNotEqual(base_example.outputs, shifted_example.outputs)
        self.assertNotEqual(base_example.metadata["query_keys"], shifted_example.metadata["query_keys"])


if __name__ == "__main__":
    unittest.main()
