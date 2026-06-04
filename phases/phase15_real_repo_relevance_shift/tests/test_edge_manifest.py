from __future__ import annotations

from pathlib import Path

import torch

from phases.phase15_real_repo_relevance_shift.src.edge import (
    DeclarationCandidate,
    EdgeCandidate,
    boundary_occurrences,
    extract_callsite_edges,
    extract_class_base_edges,
    extract_declarations,
    extract_exception_edges,
    first_pair_by_files,
    iter_python_files,
)
from phases.phase15_real_repo_relevance_shift.src.manifest import (
    RepoSource,
    build_phase15_prepared_example,
    encode_repair_signal,
    manifest_row_from_dict,
    split_prepared_from_manifest_row,
    stable_manifest_hash,
)
from phases.phase15_real_repo_relevance_shift.src.protocol import (
    Phase15Protocol,
    protocol_hash,
)
from phases.phase15_real_repo_relevance_shift.src.runner import (
    build_repair_signal,
    tool_file_positions,
    with_wrong_event,
)


class CharTokenizer:
    """Minimal tokenizer with chat-template and offset support for CPU audits."""

    def apply_chat_template(self, messages, *, add_generation_prompt: bool, tokenize: bool):
        assert not tokenize
        assert add_generation_prompt
        content = messages[0]["content"]
        return f"<user>\n{content}\n<assistant>\n"

    def encode(self, text: str, *, return_tensors: str, add_special_tokens: bool):
        assert return_tensors == "pt"
        assert not add_special_tokens
        return torch.tensor([[ord(char) for char in text]], dtype=torch.long)

    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        assert not add_special_tokens
        assert return_offsets_mapping
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


def _write_repo(root: Path) -> None:
    (root / "src").mkdir()
    (root / "src" / "alpha.py").write_text(
        "\n".join(
            [
                "class AlphaWindow:",
                "    pass",
                "# OtherBuildRowSuffix exercises subword collision auditing.",
                "",
                "class BuildRow:",
                "    pass",
                "",
                "def alpha_probe():",
                "    return 1",
            ]
        ),
        encoding="utf-8",
    )
    (root / "src" / "beta.py").write_text(
        "\n".join(
            [
                "def run_beta(value):",
                "    return BuildRow(value)",
            ]
        ),
        encoding="utf-8",
    )
    (root / "tests").mkdir()
    (root / "tests" / "test_beta.py").write_text(
        "def test_helper():\n    return HiddenTestHelper()\n",
        encoding="utf-8",
    )
    (root / "examples").mkdir()
    (root / "examples" / "demo.py").write_text(
        "def demo_helper():\n    return DemoHelper()\n",
        encoding="utf-8",
    )


def test_extract_callsite_edges_keeps_single_leaf_callee() -> None:
    text = "def run_beta(value):\n    return BuildRow(value)\n"

    edges = extract_callsite_edges("src/beta.py", text)

    assert len(edges) == 1
    assert edges[0].answer == "BuildRow"
    assert edges[0].anchor_name == "run_beta"
    assert edges[0].edge_type == "callsite_leaf_callee"


def test_extract_class_base_edges_keeps_leaf_base_identifier() -> None:
    text = "class SpecialView(BaseView):\n    pass\n"

    edges = extract_class_base_edges("src/views.py", text)

    assert len(edges) == 1
    assert edges[0].anchor_name == "SpecialView"
    assert edges[0].answer == "BaseView"
    assert edges[0].edge_type == "class_base_identifier"


def test_extract_exception_edges_keeps_assert_raises_target() -> None:
    text = "\n".join(
            [
                "def test_failure():",
                "    with self.assertRaisesMessage(CustomFailure, msg):",
                "        run_check()",
            ]
        )

    edges = extract_exception_edges("tests/test_errors.py", text)

    assert len(edges) == 1
    assert edges[0].answer == "CustomFailure"
    assert edges[0].answer_kind == "expected_exception_identifier"
    assert edges[0].anchor_name == "test_failure"


def test_extract_declarations_finds_q1_candidate() -> None:
    text = "class AlphaWindow:\n    pass\n\ndef alpha_probe():\n    return 1\n"

    declarations = extract_declarations("src/alpha.py", text)

    assert [candidate.answer for candidate in declarations] == ["AlphaWindow", "alpha_probe"]


def test_iter_python_files_ignores_tests_and_examples(tmp_path: Path) -> None:
    _write_repo(tmp_path)

    paths = [path.relative_to(tmp_path).as_posix() for path in iter_python_files(tmp_path)]

    assert "src/alpha.py" in paths
    assert "src/beta.py" in paths
    assert "tests/test_beta.py" not in paths
    assert "examples/demo.py" not in paths


def test_first_pair_prioritizes_callsite_edges_before_exception_reserve() -> None:
    q1 = DeclarationCandidate(
        path="src/alpha.py",
        kind="function",
        answer="alpha_probe",
        line_no=1,
        line_text="def alpha_probe():",
        answer_start_in_line=4,
        answer_end_in_line=15,
    )
    exception = EdgeCandidate(
        path="src/beta.py",
        edge_type="exception_identifier",
        answer_kind="raised_exception_identifier",
        answer="CustomError",
        line_no=3,
        line_text="raise CustomError()",
        answer_start_in_line=6,
        answer_end_in_line=17,
        anchor_name="run_beta",
        anchor_line_no=1,
    )
    callsite = EdgeCandidate(
        path="src/beta.py",
        edge_type="callsite_leaf_callee",
        answer_kind="callee_identifier",
        answer="BuildRow",
        line_no=2,
        line_text="return BuildRow(value)",
        answer_start_in_line=7,
        answer_end_in_line=15,
        anchor_name="run_beta",
        anchor_line_no=1,
    )

    _q1, q2 = first_pair_by_files([q1], [exception, callsite])

    assert q2.answer == "BuildRow"


def test_build_phase15_prepared_example_has_event_only_signal_and_audit(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    repo = RepoSource(
        repo_id="toyrepo",
        repo_root=str(tmp_path),
        repo_url="https://example.invalid/toyrepo",
        commit_sha="abc123",
        license_spdx="MIT",
        archive_sha256="deadbeef",
    )

    prepared = build_phase15_prepared_example(
        repo=repo,
        index=0,
        tokenizer=CharTokenizer(),
        target_context_length=1000,
        max_context_tokens=10_000,
        recency_window=0,
        k_max=0,
    )

    row = prepared.row
    assert row.source_task == "repodelta_edge"
    assert row.audit.passed
    assert row.audit.warnings == ()
    assert row.audit.answer_token_start is not None
    assert row.audit.q2_span_token_start <= row.audit.answer_token_start
    assert row.answer == "BuildRow"
    assert row.answer not in row.tool_event
    assert row.answer not in row.q2_question
    assert str(row.q1["answer"]) not in row.q1_question
    assert "<identifier>" in row.q1_question
    assert "<identifier>" in row.tool_event
    assert row.metadata["redacted_edge_line"] == "return <identifier>(value)"
    assert "line 2" not in row.tool_event
    assert row.tool_event in row.final_q2_prompt
    assert row.q2_question in row.final_q2_prompt
    assert row.metadata["repair_signal_mode"] == "event_only"
    assert prepared.repair_cue_ids.shape[1] < prepared.q2_prepared.question_ids.shape[1]
    assert prepared.q2_prepared.span_token_positions["repo_fact_2"]
    assert len(boundary_occurrences(prepared.q2_prepared.rendered_context, row.answer)) >= 1


def test_repair_signal_modes_are_explicit(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    repo = RepoSource(repo_id="toyrepo", repo_root=str(tmp_path), commit_sha="abc123")
    prepared = build_phase15_prepared_example(
        repo=repo,
        index=0,
        tokenizer=CharTokenizer(),
        target_context_length=1000,
        max_context_tokens=10_000,
        recency_window=0,
        k_max=0,
    )

    event_only = build_repair_signal(CharTokenizer(), prepared.row, mode="event_only")
    event_plus_q2 = encode_repair_signal(CharTokenizer(), prepared.row, mode="event_plus_q2")

    assert event_only.mode == "event_only"
    assert event_only.decode_prompt_mode == "event_plus_q2"
    assert event_only.ids.shape[1] < event_plus_q2.shape[1]


def test_wrong_event_and_tool_file_control_helpers(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    repo = RepoSource(repo_id="toyrepo", repo_root=str(tmp_path), commit_sha="abc123")
    prepared = build_phase15_prepared_example(
        repo=repo,
        index=0,
        tokenizer=CharTokenizer(),
        target_context_length=1000,
        max_context_tokens=10_000,
        recency_window=0,
        k_max=0,
    )

    wrong = with_wrong_event(prepared.row, prepared.row)
    selected = tool_file_positions(
        evicted_positions=tuple(range(int(prepared.q2_prepared.context_ids.shape[1]))),
        segment_token_ranges=prepared.q2_prepared.segment_token_ranges,
        q2_path=str(prepared.row.q2["path"]),
        k=5,
    )

    assert wrong.metadata["wrong_event_source_example_id"] == prepared.row.example_id
    assert len(selected) == 5
    assert selected == sorted(selected)


def test_manifest_hash_is_stable(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    repo = RepoSource(repo_id="toyrepo", repo_root=str(tmp_path), commit_sha="abc123")
    prepared = build_phase15_prepared_example(
        repo=repo,
        index=0,
        tokenizer=CharTokenizer(),
        target_context_length=1000,
        max_context_tokens=10_000,
        recency_window=0,
        k_max=0,
    )

    assert stable_manifest_hash([prepared.row]) == stable_manifest_hash([prepared.row])


def test_manifest_row_roundtrip_rebuilds_prepared_split(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    repo = RepoSource(repo_id="toyrepo", repo_root=str(tmp_path), commit_sha="abc123")
    prepared = build_phase15_prepared_example(
        repo=repo,
        index=0,
        tokenizer=CharTokenizer(),
        target_context_length=1000,
        max_context_tokens=10_000,
        recency_window=0,
        k_max=0,
    )

    row = manifest_row_from_dict(prepared.row.to_dict())
    split = split_prepared_from_manifest_row(row, CharTokenizer())

    assert split.split_spec.name == "repodelta_edge"
    assert split.q1_prepared.example.outputs == [prepared.row.q1["answer"]]
    assert split.q2_prepared.example.outputs == [prepared.row.answer]
    assert split.q2_prepared.span_token_positions["repo_fact_2"]
    assert row.tool_event in split.q2_prepared.example.question


def test_protocol_hash_is_stable() -> None:
    protocol = Phase15Protocol(k_grid=(32, 96), conditions=("A", "IdleKV-EventOnly-K"))

    assert protocol_hash(protocol) == protocol_hash(protocol)
    assert protocol.to_dict()["k_grid"] == [32, 96]


def test_audit_flags_tail_leakage(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    repo = RepoSource(repo_id="toyrepo", repo_root=str(tmp_path), commit_sha="abc123")

    prepared = build_phase15_prepared_example(
        repo=repo,
        index=0,
        tokenizer=CharTokenizer(),
        target_context_length=1000,
        max_context_tokens=10_000,
        recency_window=10_000,
        k_max=128,
    )

    assert not prepared.row.audit.passed
    assert "q2_span_in_tail_reserve" in prepared.row.audit.flags


def test_audit_flags_context_under_repair_budget(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    repo = RepoSource(repo_id="toyrepo", repo_root=str(tmp_path), commit_sha="abc123")

    prepared = build_phase15_prepared_example(
        repo=repo,
        index=0,
        tokenizer=CharTokenizer(),
        target_context_length=1000,
        max_context_tokens=10_000,
        recency_window=0,
        k_max=0,
        min_context_tokens=10_000,
    )

    assert not prepared.row.audit.passed
    assert "rendered_context_under_repair_budget" in prepared.row.audit.flags
