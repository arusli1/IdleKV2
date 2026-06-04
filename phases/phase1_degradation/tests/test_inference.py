from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import phase1.inference as inference
from phase1.inference import PreparedExample, generate_answer, run_example
from phase1.models import TaskExample


class RecordingTokenizer:
    def decode(self, token_ids, skip_special_tokens=True) -> str:
        del skip_special_tokens
        return " ".join(str(int(token_id)) for token_id in token_ids.tolist())


class RecordingModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.generation_config = SimpleNamespace(eos_token_id=99)
        self.calls: list[dict[str, torch.Tensor]] = []

    def __call__(self, **kwargs):
        self.calls.append(
            {
                "input_ids": kwargs["input_ids"].detach().clone(),
                "position_ids": kwargs["position_ids"].detach().clone(),
                "cache_position": kwargs["cache_position"].detach().clone(),
            }
        )
        seq_len = kwargs["input_ids"].shape[1]
        logits = torch.zeros((1, seq_len, 4), dtype=torch.float32)
        logits[0, -1, len(self.calls)] = 1.0
        return SimpleNamespace(logits=logits)


class InferenceTests(unittest.TestCase):
    def test_generate_answer_separates_position_ids_from_cache_positions(self) -> None:
        model = RecordingModel()
        tokenizer = RecordingTokenizer()

        generate_answer(
            model=model,
            tokenizer=tokenizer,
            question_ids=torch.tensor([[10, 11, 12]], dtype=torch.long),
            cache=object(),
            position_base=32_768,
            cache_position_base=4_096,
            max_new_tokens=2,
        )

        self.assertEqual(model.calls[0]["position_ids"].tolist(), [[32_768, 32_769, 32_770]])
        self.assertEqual(model.calls[0]["cache_position"].tolist(), [4_096, 4_097, 4_098])
        self.assertEqual(model.calls[1]["position_ids"].tolist(), [[32_771]])
        self.assertEqual(model.calls[1]["cache_position"].tolist(), [4_099])

    def test_run_example_uses_compressed_cache_length_for_question_resume(self) -> None:
        class FakeCache:
            def get_seq_length(self) -> int:
                return 4

        class PrefillOnlyModel:
            def __init__(self) -> None:
                self.device = torch.device("cpu")
                self.prefill_calls: list[dict[str, torch.Tensor]] = []
                self.model = self

            def __call__(self, **kwargs):
                self.prefill_calls.append(
                    {
                        "position_ids": kwargs["position_ids"].detach().clone(),
                        "cache_position": kwargs["cache_position"].detach().clone(),
                    }
                )
                return None

        captured_generate_args: dict[str, int] = {}

        def fake_generate_answer(*, position_base: int, cache_position_base: int, **kwargs) -> str:
            del kwargs
            captured_generate_args["position_base"] = position_base
            captured_generate_args["cache_position_base"] = cache_position_base
            return "answer"

        example = TaskExample(
            index=0,
            task_name="task",
            task_family="family",
            context="context",
            question="question",
            answer_prefix="",
            outputs=["answer"],
            max_new_tokens=1,
            target_context_length=10,
            relevant_spans=[],
            prefill_segments=[],
            metadata={},
        )
        prepared = PreparedExample(
            example=example,
            rendered_context="context",
            context_ids=torch.arange(10, dtype=torch.long).unsqueeze(0),
            question_ids=torch.tensor([[1, 2]], dtype=torch.long),
            span_token_positions={},
            segment_token_ranges=[("context", 0, 10)],
        )

        with patch.object(inference, "DynamicCache", FakeCache), patch.object(
            inference, "generate_answer", side_effect=fake_generate_answer
        ):
            record = run_example(
                model=PrefillOnlyModel(),
                tokenizer=RecordingTokenizer(),
                prepared=prepared,
                algorithm="snapkv",
                budget=None,
                condition="condition_a",
                trace_path=None,
                query_log_tokens=64,
            )

        self.assertEqual(captured_generate_args, {"position_base": 10, "cache_position_base": 4})
        self.assertEqual(record.compressed_context_length, 4)


if __name__ == "__main__":
    unittest.main()
