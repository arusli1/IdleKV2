"""Fail-fast checks for benchmark wrappers."""

import pytest
import torch

from idlekv.eval.longbench import evaluate_longbench
from idlekv.eval.ruler import evaluate_ruler_niah


def test_ruler_raises_when_all_samples_fail(monkeypatch):
    monkeypatch.setattr(
        "idlekv.eval.ruler.create_niah_test",
        lambda *args, **kwargs: (torch.tensor([[1, 2, 3]]), ["1234"]),
    )

    def always_fail(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("idlekv.eval.ruler.generate_text", always_fail)

    with pytest.raises(RuntimeError, match="failed for all 1 samples"):
        evaluate_ruler_niah(
            model=None,
            tokenizer=object(),
            num_samples=1,
            device="cpu",
        )


def test_longbench_raises_when_all_samples_fail(monkeypatch):
    monkeypatch.setattr(
        "idlekv.eval.longbench.select_longbench_records",
        lambda *args, **kwargs: [{
            "input": "Question?",
            "context": "Context",
            "answers": ["Answer"],
        }],
    )

    class FakeBatch(dict):
        def to(self, device):
            return self

    class FakeTokenizer:
        def __call__(self, *args, **kwargs):
            return FakeBatch({"input_ids": torch.tensor([[1, 2, 3]])})

    def always_fail(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("idlekv.eval.longbench.generate_text", always_fail)

    with pytest.raises(RuntimeError, match="failed for all 1 samples"):
        evaluate_longbench(
            model=None,
            tokenizer=FakeTokenizer(),
            subtasks=["narrativeqa"],
            num_samples=1,
            device="cpu",
        )
