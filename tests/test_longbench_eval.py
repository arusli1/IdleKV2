"""LongBench loader and scoring tests."""

import json
import zipfile

import torch

from idlekv.eval.longbench import evaluate_longbench, load_longbench_records


class FakeBatch(dict):
    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, device):
        return self


class FakeTokenizer:
    def __call__(self, text, *args, **kwargs):
        length = min(max(len(text.split()), 1), 8)
        return FakeBatch({"input_ids": torch.arange(length).unsqueeze(0)})

    def decode(self, token_ids, skip_special_tokens=True):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return " ".join(str(token) for token in token_ids)


def test_load_longbench_records_reads_zip_archive(tmp_path, monkeypatch):
    archive_path = tmp_path / "data.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "data/narrativeqa.jsonl",
            json.dumps({
                "input": "Who?",
                "context": "Context",
                "answers": ["Alice"],
            }) + "\n",
        )

    monkeypatch.setattr(
        "idlekv.eval.longbench.longbench_archive_path",
        lambda: str(archive_path),
    )
    load_longbench_records.cache_clear()

    records = load_longbench_records("narrativeqa")
    assert len(records) == 1
    assert records[0]["answers"] == ["Alice"]


def test_evaluate_longbench_reports_official_scale(monkeypatch):
    monkeypatch.setattr(
        "idlekv.eval.longbench.select_longbench_records",
        lambda *args, **kwargs: [{
            "input": "",
            "context": "Paragraph A. Paragraph A. Paragraph B.",
            "answers": ["2"],
            "all_classes": None,
        }],
    )
    monkeypatch.setattr(
        "idlekv.eval.longbench.generate_text",
        lambda **kwargs: {"text": "2"},
    )

    results = evaluate_longbench(
        model=None,
        tokenizer=FakeTokenizer(),
        subtasks=["passage_count"],
        num_samples=1,
        device="cpu",
    )

    assert results[0].subtask == "passage_count"
    assert results[0].score == 100.0
    assert results[0].num_samples == 1


def test_evaluate_longbench_uses_single_line_classification_scoring(monkeypatch):
    monkeypatch.setattr(
        "idlekv.eval.longbench.select_longbench_records",
        lambda *args, **kwargs: [{
            "input": "Question: Where?",
            "context": "Examples...",
            "answers": ["Food"],
            "all_classes": ["Food", "Date"],
        }],
    )
    monkeypatch.setattr(
        "idlekv.eval.longbench.generate_text",
        lambda **kwargs: {"text": "Food\nextra analysis"},
    )

    results = evaluate_longbench(
        model=None,
        tokenizer=FakeTokenizer(),
        subtasks=["trec"],
        num_samples=1,
        device="cpu",
    )

    assert results[0].score == 100.0
