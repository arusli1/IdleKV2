from __future__ import annotations

import unittest

from phase1.modeling import load_tokenizer
from phase1.paths import MODEL_DIR
from phase1.task_registry import build_task_example


class VTGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = load_tokenizer(str(MODEL_DIR))

    def test_vt_8hop_permute_div2_builds_a_permuted_chain_with_two_distractors(self) -> None:
        example = build_task_example("vt_8hop_permute_div2", 0, 1024, self.tokenizer)

        self.assertEqual(example.metadata["num_hops"], 8)
        self.assertTrue(example.metadata["permute"])
        self.assertTrue(example.metadata["random_permute"])
        self.assertEqual(example.metadata["num_divergences"], 2)
        self.assertEqual(example.metadata["filler_kind"], "prose")
        self.assertEqual(example.metadata["query_var"], example.metadata["variables"][-1])
        self.assertEqual(
            example.question,
            f"What is the final numeric value of {example.metadata['query_var']}?",
        )

        relevant_names = [span.name for span in example.relevant_spans]
        hop_spans = [span for span in example.relevant_spans if span.kind == "hop"]
        distractor_spans = [span for span in example.relevant_spans if span.kind == "distractor"]

        self.assertEqual(relevant_names, example.metadata["permute_order"])
        self.assertEqual(len(hop_spans), 8)
        self.assertEqual(len(distractor_spans), 2)
        self.assertEqual(set(relevant_names), {f"hop_{index}" for index in range(1, 9)} | {"distractor_1", "distractor_2"})
        self.assertTrue(all(span.depth_fraction < 1.0 for span in example.relevant_spans))
        self.assertEqual(
            set(example.metadata["divergence_sources"]),
            {span.metadata["source_var_index"] for span in distractor_spans},
        )

    def test_dataset_seed_offset_changes_generated_vt_example(self) -> None:
        base_example = build_task_example("vt_4hop_permute", 0, 1024, self.tokenizer, dataset_seed_offset=0)
        same_seed_example = build_task_example("vt_4hop_permute", 0, 1024, self.tokenizer, dataset_seed_offset=0)
        shifted_example = build_task_example("vt_4hop_permute", 0, 1024, self.tokenizer, dataset_seed_offset=1)

        self.assertEqual(base_example.outputs, same_seed_example.outputs)
        self.assertEqual(base_example.metadata["variables"], same_seed_example.metadata["variables"])
        self.assertNotEqual(base_example.outputs, shifted_example.outputs)
        self.assertNotEqual(base_example.metadata["variables"], shifted_example.metadata["variables"])

    def test_vt_tasks_use_sanitized_prose_filler(self) -> None:
        example = build_task_example("vt_4hop_permute", 0, 1024, self.tokenizer)

        self.assertEqual(example.metadata["filler_kind"], "prose")
        spans = sorted((span.char_start, span.char_end) for span in example.relevant_spans)
        filler_parts = []
        cursor = 0
        for start, end in spans:
            filler_parts.append(example.context[cursor:start])
            cursor = end
        filler_parts.append(example.context[cursor:])
        filler_only = " ".join(filler_parts)

        self.assertNotRegex(filler_only, r"\b63%?\b")
        self.assertNotRegex(filler_only, r"\b\d{2,}\b")
        self.assertNotIn("The grass is green.", filler_only)


if __name__ == "__main__":
    unittest.main()
