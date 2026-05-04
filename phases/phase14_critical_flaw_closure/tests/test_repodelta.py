from __future__ import annotations

from pathlib import Path

import torch

from phases.phase14_critical_flaw_closure.src.repodelta import (
    build_repodelta_base_example,
    build_repodelta_prepared_example,
    extract_candidates,
    split_repodelta_for_turn,
)


class CharTokenizer:
    """Minimal tokenizer with chat-template and offset support for span tests."""

    def apply_chat_template(self, messages, *, add_generation_prompt: bool, tokenize: bool):
        assert not tokenize
        assert add_generation_prompt
        content = messages[0]["content"]
        return f"<user>\n{content}\n<assistant>\n"

    def encode(self, text: str, *, return_tensors: str, add_special_tokens: bool):
        assert return_tensors == "pt"
        assert not add_special_tokens
        return torch.arange(len(text), dtype=torch.long).unsqueeze(0)

    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        assert not add_special_tokens
        assert return_offsets_mapping
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


def _write_repo(root: Path) -> None:
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "alpha.py").write_text(
        "\n".join(
            [
                "ALPHA_TIMEOUT_LIMIT = 37",
                "",
                "def compute_alpha_window(config):",
                "    return config.window + ALPHA_TIMEOUT_LIMIT",
            ]
        ),
        encoding="utf-8",
    )
    (root / "src" / "beta.py").write_text(
        "\n".join(
            [
                "class BetaRepairScheduler:",
                "    pass",
                "",
                "def load_beta_manifest(path):",
                "    return path",
            ]
        ),
        encoding="utf-8",
    )
    (root / "tests" / "test_alpha.py").write_text(
        "def test_alpha_window_contract():\n    assert True\n",
        encoding="utf-8",
    )


def test_extract_candidates_finds_code_identifiers() -> None:
    text = "class CachePolicy:\n    pass\n\ndef repair_window():\n    pass\nMAX_ROWS = 8\n"

    candidates = extract_candidates("src/cache.py", text)

    assert [candidate.answer for candidate in candidates] == [
        "CachePolicy",
        "repair_window",
        "MAX_ROWS",
    ]
    assert {candidate.kind for candidate in candidates} == {"class", "function", "constant"}


def test_repodelta_base_has_distinct_turns_without_tool_answer_leak(tmp_path: Path) -> None:
    _write_repo(tmp_path)

    example = build_repodelta_base_example(
        repo_root=tmp_path,
        index=0,
        target_context_length=1000,
        tokenizer=CharTokenizer(),
    )

    q1 = example.metadata["q1"]
    q2 = example.metadata["q2"]
    assert q1["path"] != q2["path"]
    assert example.outputs == [q1["answer"], q2["answer"]]
    assert example.context.count(str(q2["answer"])) == 1
    assert f"{int(q2['line_no']):04d}:" in example.context
    assert str(q2["answer"]) not in str(example.metadata["tool_event"])
    assert {span.name for span in example.relevant_spans} == {"repo_fact_1", "repo_fact_2"}
    for span in example.relevant_spans:
        sliced = example.context[span.char_start : span.char_end]
        assert str(span.metadata["answer"]) in sliced
        assert str(span.metadata["line_text"]) in sliced


def test_repodelta_turn_projection_uses_event_localized_q2(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    base = build_repodelta_base_example(
        repo_root=tmp_path,
        index=1,
        target_context_length=1000,
        tokenizer=CharTokenizer(),
    )

    q1 = split_repodelta_for_turn(base, turn="q1", split_name="repodelta:q1")
    q2 = split_repodelta_for_turn(base, turn="q2", split_name="repodelta:q2")

    assert "Tool event:" not in q1.question
    assert "Tool event:" in q2.question
    assert "reported failure location" in q2.question
    assert f"In `{base.metadata['q2']['path']}`" not in q2.question
    assert f"line {base.metadata['q2']['line_no']}?" not in q2.question
    assert q1.outputs == [base.metadata["q1"]["answer"]]
    assert q2.outputs == [base.metadata["q2"]["answer"]]
    assert str(base.metadata["q2"]["answer"]) not in q2.question
    assert q2.metadata["repair_cue"] in q2.question
    assert q2.metadata["repair_signal_mode"] == "event_only"
    assert q2.metadata["decode_prompt_mode"] == "event_plus_q2"


def test_repodelta_prepared_maps_q2_span_to_tokens(tmp_path: Path) -> None:
    _write_repo(tmp_path)

    prepared = build_repodelta_prepared_example(
        repo_root=tmp_path,
        index=2,
        target_context_length=1000,
        tokenizer=CharTokenizer(),
    )

    assert prepared.q1_span_names == ("repo_fact_1",)
    assert prepared.q2_span_names == ("repo_fact_2",)
    assert prepared.q2_prepared.span_token_positions["repo_fact_2"]
    assert prepared.q2_prepared.segment_token_ranges
    assert prepared.q1_prepared.example.metadata["turn"] == "q1"
    assert prepared.q2_prepared.example.metadata["turn"] == "q2"
