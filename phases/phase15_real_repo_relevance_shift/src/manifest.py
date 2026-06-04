"""Frozen-manifest helpers for Phase 15 RepoDelta-Edge."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Sequence

import torch

from phases.phase1_degradation.phase1.inference import PreparedExample, prepare_example_for_model
from phases.phase1_degradation.phase1.models import PrefillSegment, RelevantSpan, TaskExample
from phases.phase1_degradation.phase1.prompting import char_span_to_token_positions
from phases.phase6_repair.src.protocol import SplitPreparedExample, SplitTaskSpec

from .edge import (
    DeclarationCandidate,
    EdgeCandidate,
    boundary_occurrences,
    extract_callsite_edges,
    extract_class_base_edges,
    extract_declarations,
    extract_exception_edges,
    first_pair_by_files,
    iter_python_files,
    line_char_bounds,
    number_file_card,
    read_text,
)
from .scorer import validate_gold_identifier

ANSWER_PREFIX = " Answer:"
MAX_NEW_TOKENS = 16
PHASE15_SPLIT_SPEC = SplitTaskSpec(
    name="repodelta_edge",
    base_task_key="repodelta_edge",
    q1_indices=(0,),
    q2_indices=(1,),
    answer_prefix=ANSWER_PREFIX,
    max_new_tokens=MAX_NEW_TOKENS,
)


@dataclass(frozen=True)
class RepoSource:
    """One pinned source repository for Phase 15 manifests."""

    repo_id: str
    repo_root: str
    repo_url: str = ""
    commit_sha: str = ""
    license_spdx: str = ""
    archive_sha256: str = ""
    split: str = "dev"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ManifestAudit:
    """CPU audit result for one prepared Phase 15 example."""

    passed: bool
    flags: tuple[str, ...]
    warnings: tuple[str, ...]
    rendered_context_tokens: int
    q2_span_token_start: int | None
    q2_span_token_end: int | None
    answer_token_count: int
    answer_boundary_occurrences: int
    answer_isolated_token_sequence_occurrences: int
    answer_token_start: int | None
    answer_token_end: int | None
    depth_bin: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase15ManifestRow:
    """Serializable manifest row with the prompt and audit metadata needed to run."""

    example_id: str
    source_task: str
    repo: RepoSource
    index: int
    context: str
    q1_question: str
    tool_event: str
    q2_question: str
    final_q2_prompt: str
    answer: str
    q1: dict[str, Any]
    q2: dict[str, Any]
    audit: ManifestAudit
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["repo"] = self.repo.to_dict()
        payload["audit"] = self.audit.to_dict()
        return payload


@dataclass(frozen=True)
class Phase15PreparedExample:
    """Prepared Q1/Q2 views plus event-only repair cue ids."""

    row: Phase15ManifestRow
    base_example: TaskExample
    q1_prepared: PreparedExample
    q2_prepared: PreparedExample
    repair_cue_ids: torch.Tensor
    q1_span_names: tuple[str, ...] = ("repo_fact_1",)
    q2_span_names: tuple[str, ...] = ("repo_fact_2",)


def stable_manifest_hash(rows: Sequence[Phase15ManifestRow]) -> str:
    """Return a deterministic hash over manifest rows."""
    payload = json.dumps([row.to_dict() for row in rows], sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_rng(repo: RepoSource, index: int, seed_offset: int) -> random.Random:
    payload = f"{repo.repo_id}:{repo.commit_sha}:{index}:{seed_offset}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return random.Random(seed)


def _candidate_metadata(candidate: DeclarationCandidate | EdgeCandidate) -> dict[str, Any]:
    payload = asdict(candidate)
    payload["line_text"] = str(payload["line_text"]).strip()
    return payload


def _is_test_path(path: str) -> bool:
    parts = Path(path).parts
    name = Path(path).name
    return (
        any(part in {"doc", "docs", "example", "examples", "test", "tests", "testing"} for part in parts)
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _is_project_local_edge(candidate: EdgeCandidate, declaration_names: set[str]) -> bool:
    """Prefer source-file targets that resolve to declarations in the repo sample."""
    if _is_test_path(candidate.path):
        return False
    if candidate.answer not in declaration_names:
        return False
    if candidate.anchor_name in {"module_scope", candidate.answer}:
        return False
    return True


def _redacted_edge_line(candidate: EdgeCandidate) -> str:
    """Return the source line with the gold identifier removed from the cue."""
    line = str(candidate.line_text).strip()
    start = int(candidate.answer_start_in_line)
    end = int(candidate.answer_end_in_line)
    if start < 0 or end <= start or end > len(candidate.line_text):
        return line.replace(candidate.answer, "<identifier>", 1)
    return (candidate.line_text[:start] + "<identifier>" + candidate.line_text[end:]).strip()


def _redacted_declaration_line(candidate: DeclarationCandidate) -> str:
    """Return the Q1 declaration line with its identifier removed."""
    line = str(candidate.line_text).strip()
    start = int(candidate.answer_start_in_line)
    end = int(candidate.answer_end_in_line)
    if start < 0 or end <= start or end > len(candidate.line_text):
        return line.replace(candidate.answer, "<identifier>", 1)
    return (candidate.line_text[:start] + "<identifier>" + candidate.line_text[end:]).strip()


def _q1_question(candidate: DeclarationCandidate) -> str:
    redacted = _redacted_declaration_line(candidate)
    return (
        f"Earlier repository context included `{candidate.path}`. "
        f"The relevant declaration line is `{redacted}`. "
        f"What {candidate.kind} identifier is hidden by `<identifier>`? "
        "Respond with the identifier only."
    )


def _edge_tool_event(candidate: EdgeCandidate) -> str:
    redacted = _redacted_edge_line(candidate)
    if candidate.edge_type == "exception_identifier":
        return (
            "Tool event: a repository check failed while exercising error handling "
            f"in `{candidate.anchor_name}` from `{candidate.path}`. "
            "The report includes this answer-redacted statement: "
            f"`{redacted}`. Use the repository context to recover the missing identifier."
        )
    if candidate.edge_type == "class_base_identifier":
        return (
            "Tool event: a repository check failed while constructing "
            f"`{candidate.anchor_name}` from `{candidate.path}`. "
            "The report includes this answer-redacted declaration: "
            f"`{redacted}`. Use the repository context to recover the missing identifier."
        )
    return (
        "Tool event: a repository check failed while executing "
        f"`{candidate.path}` inside `{candidate.anchor_name}`. "
        "The report includes this answer-redacted statement: "
        f"`{redacted}`. Use the repository context to recover the missing identifier."
    )


def _edge_q2_question(candidate: EdgeCandidate) -> str:
    if candidate.edge_type == "exception_identifier":
        return (
            "What exception identifier is hidden by `<identifier>`? "
            "Respond with the identifier only."
        )
    if candidate.edge_type == "class_base_identifier":
        return (
            "What base-class identifier is hidden by `<identifier>`? "
            "Respond with the identifier only."
        )
    return (
        "What callable identifier is hidden by `<identifier>`? "
        "Respond with the identifier only."
    )


def _line_span_name(turn: str) -> str:
    return "repo_fact_1" if turn == "q1" else "repo_fact_2"


def _build_context_from_files(
    repo_root: Path,
    files: Sequence[Path],
    *,
    target_context_length: int,
) -> tuple[str, list[tuple[str, str, int, int]], dict[str, str]]:
    context_parts: list[str] = []
    cards: list[tuple[str, str, int, int]] = []
    text_by_rel: dict[str, str] = {}
    cursor = 0
    for path in files:
        text = read_text(path)
        if not text:
            continue
        rel = path.relative_to(repo_root).as_posix()
        card = number_file_card(rel, text)
        if context_parts:
            context_parts.append("\n")
            cursor += 1
        start = cursor
        end = start + len(card)
        context_parts.append(card)
        cards.append((rel, card, start, end))
        text_by_rel[rel] = text
        cursor = end
        if cursor >= int(target_context_length) and len(cards) >= 2:
            break
    return "".join(context_parts), cards, text_by_rel


def build_edge_base_example(
    *,
    repo: RepoSource,
    index: int,
    target_context_length: int,
    seed_offset: int = 0,
    max_files: int = 24,
    allowed_edge_types: set[str] | None = None,
    require_project_local_targets: bool = True,
    max_answer_boundary_occurrences: int = 12,
    min_q2_depth_fraction: float = 0.0,
    max_q2_depth_fraction: float = 1.0,
) -> TaskExample:
    """Build one RepoDelta-Edge base example from a pinned repository snapshot."""
    repo_root = Path(repo.repo_root)
    files = iter_python_files(repo_root)
    if len(files) < 2:
        raise ValueError(f"Need at least two Python files under {repo_root}.")
    rng = _stable_rng(repo, index, seed_offset)
    sampled = files[:]
    rng.shuffle(sampled)
    selected_files = sampled[: max(2, int(max_files))]
    context, cards, text_by_rel = _build_context_from_files(
        repo_root,
        selected_files,
        target_context_length=target_context_length,
    )
    if len(cards) < 2:
        raise ValueError("Need at least two readable file cards.")

    declarations: list[DeclarationCandidate] = []
    edges: list[EdgeCandidate] = []
    for rel, text in text_by_rel.items():
        declarations.extend(extract_declarations(rel, text))
        edges.extend(extract_exception_edges(rel, text))
        edges.extend(extract_class_base_edges(rel, text))
        edges.extend(extract_callsite_edges(rel, text))

    source_declarations = [candidate for candidate in declarations if not _is_test_path(candidate.path)]
    declaration_names = {candidate.answer for candidate in source_declarations}
    declarations = [
        candidate
        for candidate in source_declarations
        if len(boundary_occurrences(context, candidate.answer)) == 1
    ]
    card_by_path = {rel: (card, start, end) for rel, card, start, end in cards}

    def _candidate_depth(candidate: EdgeCandidate) -> float:
        card, card_start, _ = card_by_path[candidate.path]
        line_start, _line_end = line_char_bounds(card_start, card, candidate.line_no)
        return line_start / max(len(context), 1)

    allowed = set(allowed_edge_types) if allowed_edge_types is not None else None
    edges = [
        candidate
        for candidate in edges
        if (allowed is None or candidate.edge_type in allowed)
        and (not require_project_local_targets or _is_project_local_edge(candidate, declaration_names))
        and not _is_test_path(candidate.path)
        and (
            candidate.edge_type != "callsite_leaf_callee"
            or "_" in candidate.answer
            or any(char.isupper() for char in candidate.answer[1:])
        )
        and 1 <= len(boundary_occurrences(context, candidate.answer)) <= int(max_answer_boundary_occurrences)
        and float(min_q2_depth_fraction) <= _candidate_depth(candidate) <= float(max_q2_depth_fraction)
        and candidate.answer not in _edge_tool_event(candidate)
        and candidate.answer not in _edge_q2_question(candidate)
    ]
    anchor_counts: dict[tuple[str, str], int] = {}
    for candidate in edges:
        key = (candidate.path, candidate.anchor_name)
        anchor_counts[key] = anchor_counts.get(key, 0) + 1
    edges = [
        candidate
        for candidate in edges
        if anchor_counts.get((candidate.path, candidate.anchor_name), 0) == 1
    ]
    q1, q2 = first_pair_by_files(declarations, edges)
    validate_gold_identifier(q1.answer)
    validate_gold_identifier(q2.answer)

    spans: list[RelevantSpan] = []
    for turn, candidate in (("q1", q1), ("q2", q2)):
        card, card_start, _ = card_by_path[candidate.path]
        line_start, line_end = line_char_bounds(card_start, card, candidate.line_no)
        spans.append(
            RelevantSpan(
                name=_line_span_name(turn),
                kind="repo_symbol_line" if turn == "q1" else "repo_edge_callsite_line",
                char_start=line_start,
                char_end=line_end,
                depth_fraction=line_start / max(len(context), 1),
                metadata=_candidate_metadata(candidate),
            )
        )

    segments = [
        PrefillSegment(name=f"file:{rel}", char_start=start, char_end=end)
        for rel, _card, start, end in cards
    ]
    tool_event = _edge_tool_event(q2)
    q2_question = _edge_q2_question(q2)
    metadata = {
        "source_task": "repodelta_edge",
        "repo": repo.to_dict(),
        "q1": _candidate_metadata(q1),
        "q2": _candidate_metadata(q2),
        "tool_event": tool_event,
        "q2_question": q2_question,
        "redacted_edge_line": _redacted_edge_line(q2),
        "repair_cue": tool_event,
        "repair_signal_mode": "event_only",
        "decode_prompt_mode": "event_plus_q2",
        "query_keys": [f"{q1.path}:{q1.line_no}", f"{q2.path}:{q2.line_no}"],
    }
    return TaskExample(
        index=int(index),
        task_name="repodelta_edge",
        task_family="repo_delta",
        context=context,
        question=_q1_question(q1),
        answer_prefix=ANSWER_PREFIX,
        outputs=[q1.answer, q2.answer],
        max_new_tokens=MAX_NEW_TOKENS,
        target_context_length=int(target_context_length),
        relevant_spans=spans,
        prefill_segments=segments,
        metadata=metadata,
    )


def _project_turn(base_example: TaskExample, *, turn: str) -> TaskExample:
    if turn == "q1":
        metadata = dict(base_example.metadata)
        metadata["turn"] = "q1"
        q1 = metadata["q1"]
        return TaskExample(
            index=base_example.index,
            task_name=base_example.task_name,
            task_family=base_example.task_family,
            context=base_example.context,
            question=base_example.question,
            answer_prefix=ANSWER_PREFIX,
            outputs=[str(q1["answer"])],
            max_new_tokens=MAX_NEW_TOKENS,
            target_context_length=base_example.target_context_length,
            relevant_spans=base_example.relevant_spans,
            prefill_segments=base_example.prefill_segments,
            metadata=metadata,
        )
    if turn == "q2":
        metadata = dict(base_example.metadata)
        metadata["turn"] = "q2"
        q2 = metadata["q2"]
        question = str(metadata["tool_event"]) + "\n\n" + str(metadata["q2_question"])
        return TaskExample(
            index=base_example.index,
            task_name=base_example.task_name,
            task_family=base_example.task_family,
            context=base_example.context,
            question=question,
            answer_prefix=ANSWER_PREFIX,
            outputs=[str(q2["answer"])],
            max_new_tokens=MAX_NEW_TOKENS,
            target_context_length=base_example.target_context_length,
            relevant_spans=base_example.relevant_spans,
            prefill_segments=base_example.prefill_segments,
            metadata=metadata,
        )
    raise ValueError(f"Unsupported turn: {turn!r}")


def _subsequence_count(haystack: list[int], needle: list[int]) -> int:
    if not needle:
        return 0
    count = 0
    width = len(needle)
    for start in range(0, len(haystack) - width + 1):
        if haystack[start : start + width] == needle:
            count += 1
    return count


def _depth_bin(start: int | None, total: int) -> str:
    if start is None or total <= 0:
        return "missing"
    frac = start / total
    if frac < 0.33:
        return "early"
    if frac < 0.67:
        return "middle"
    return "late"


def audit_prepared_example(
    *,
    prepared: PreparedExample,
    tokenizer,
    recency_window: int,
    k_max: int,
    max_context_tokens: int,
    min_context_tokens: int = 0,
    max_answer_tokens: int = 12,
    max_answer_boundary_occurrences: int = 12,
) -> ManifestAudit:
    """Run strict CPU manifest audits on a prepared Q2 example."""
    flags: list[str] = []
    warnings: list[str] = []
    context_tokens = int(prepared.context_ids.shape[1])
    if context_tokens > int(max_context_tokens):
        flags.append("rendered_context_over_budget")
    if int(min_context_tokens) > 0 and context_tokens <= int(min_context_tokens):
        flags.append("rendered_context_under_repair_budget")

    positions = prepared.span_token_positions.get("repo_fact_2", [])
    span_start = min(positions) if positions else None
    span_end = max(positions) + 1 if positions else None
    if not positions:
        flags.append("missing_q2_token_span")
    elif span_end is not None and span_end > context_tokens - (int(recency_window) + int(k_max)):
        flags.append("q2_span_in_tail_reserve")

    answer = str(prepared.example.outputs[0])
    answer_ids = tokenizer.encode(answer, return_tensors="pt", add_special_tokens=False)
    answer_token_list = [int(token) for token in answer_ids.flatten().tolist()]
    if len(answer_token_list) > int(max_answer_tokens):
        flags.append("answer_too_many_tokens")
    context_id_list = [int(token) for token in prepared.context_ids.flatten().tolist()]
    isolated_token_occurrences = _subsequence_count(context_id_list, answer_token_list)

    rendered_occurrences = boundary_occurrences(prepared.rendered_context, answer)
    if not rendered_occurrences:
        flags.append("answer_text_missing")
    if len(rendered_occurrences) > int(max_answer_boundary_occurrences):
        flags.append("answer_text_too_many_boundary_occurrences")
    answer_token_start = None
    answer_token_end = None
    answer_in_q2_line = False
    for char_start, char_end in rendered_occurrences:
        answer_token_positions = char_span_to_token_positions(
            tokenizer,
            prepared.rendered_context,
            char_start,
            char_end,
        )
        q2_line_positions = set(positions)
        if answer_token_positions and set(answer_token_positions) <= q2_line_positions:
            answer_in_q2_line = True
            answer_token_start = min(answer_token_positions)
            answer_token_end = max(answer_token_positions) + 1
            break
    if not answer_in_q2_line:
        flags.append("answer_token_span_missing_from_q2_line")

    cue = str(prepared.example.metadata.get("repair_cue", ""))
    q2_question = str(prepared.example.metadata.get("q2_question", ""))
    q2_meta = prepared.example.metadata.get("q2", {})
    if answer in cue:
        flags.append("answer_leaks_in_tool_cue")
    if answer in q2_question:
        flags.append("answer_leaks_in_q2_question")
    if isinstance(q2_meta, dict):
        if answer in str(q2_meta.get("path", "")):
            flags.append("answer_leaks_in_q2_path")
        if answer in str(q2_meta.get("anchor_name", "")):
            flags.append("answer_leaks_in_q2_anchor")

    return ManifestAudit(
        passed=not flags,
        flags=tuple(flags),
        warnings=tuple(warnings),
        rendered_context_tokens=context_tokens,
        q2_span_token_start=span_start,
        q2_span_token_end=span_end,
        answer_token_count=len(answer_token_list),
        answer_boundary_occurrences=len(rendered_occurrences),
        answer_isolated_token_sequence_occurrences=isolated_token_occurrences,
        answer_token_start=answer_token_start,
        answer_token_end=answer_token_end,
        depth_bin=_depth_bin(span_start, context_tokens),
    )


def build_phase15_prepared_example(
    *,
    repo: RepoSource,
    index: int,
    tokenizer,
    target_context_length: int,
    max_context_tokens: int,
    recency_window: int,
    k_max: int,
    min_context_tokens: int = 0,
    seed_offset: int = 0,
    max_files: int = 24,
    allowed_edge_types: set[str] | None = None,
    require_project_local_targets: bool = True,
    max_answer_boundary_occurrences: int = 12,
    min_q2_depth_fraction: float = 0.0,
    max_q2_depth_fraction: float = 1.0,
) -> Phase15PreparedExample:
    """Build and audit one prepared Phase 15 Edge example."""
    base = build_edge_base_example(
        repo=repo,
        index=index,
        target_context_length=target_context_length,
        seed_offset=seed_offset,
        max_files=max_files,
        allowed_edge_types=allowed_edge_types,
        require_project_local_targets=require_project_local_targets,
        max_answer_boundary_occurrences=max_answer_boundary_occurrences,
        min_q2_depth_fraction=min_q2_depth_fraction,
        max_q2_depth_fraction=max_q2_depth_fraction,
    )
    q1_example = _project_turn(base, turn="q1")
    q2_example = _project_turn(base, turn="q2")
    q1_prepared = prepare_example_for_model(q1_example, tokenizer)
    q2_prepared = prepare_example_for_model(q2_example, tokenizer)
    audit = audit_prepared_example(
        prepared=q2_prepared,
        tokenizer=tokenizer,
        recency_window=recency_window,
        k_max=k_max,
        max_context_tokens=max_context_tokens,
        min_context_tokens=min_context_tokens,
        max_answer_boundary_occurrences=max_answer_boundary_occurrences,
    )
    q2_meta = q2_example.metadata["q2"]
    row = Phase15ManifestRow(
        example_id=_stable_example_id(repo, q2_meta, index),
        source_task="repodelta_edge",
        repo=repo,
        index=int(index),
        context=base.context,
        q1_question=q1_example.question,
        tool_event=str(q2_example.metadata["tool_event"]),
        q2_question=str(q2_example.metadata["q2_question"]),
        final_q2_prompt=q2_example.question,
        answer=str(q2_example.outputs[0]),
        q1=dict(q2_example.metadata["q1"]),
        q2=dict(q2_meta),
        audit=audit,
        metadata={
            "repair_signal_mode": "event_only",
            "decode_prompt_mode": "event_plus_q2",
            "relevant_spans": [span.to_dict() for span in base.relevant_spans],
            "prefill_segments": [segment.to_dict() for segment in base.prefill_segments],
            "target_context_length": int(base.target_context_length),
            "redacted_edge_line": str(q2_example.metadata.get("redacted_edge_line", "")),
        },
    )
    repair_cue_ids = encode_repair_signal(tokenizer, row, mode="event_only")
    return Phase15PreparedExample(
        row=row,
        base_example=base,
        q1_prepared=q1_prepared,
        q2_prepared=q2_prepared,
        repair_cue_ids=repair_cue_ids,
    )


def manifest_row_from_dict(payload: dict[str, Any]) -> Phase15ManifestRow:
    """Rehydrate one JSONL manifest row."""
    audit_payload = dict(payload["audit"])
    audit_payload.setdefault("warnings", ())
    audit_payload.setdefault("answer_boundary_occurrences", 1)
    audit_payload.setdefault(
        "answer_isolated_token_sequence_occurrences",
        audit_payload.pop("answer_token_sequence_occurrences", 0),
    )
    audit_payload.setdefault("answer_token_start", None)
    audit_payload.setdefault("answer_token_end", None)
    return Phase15ManifestRow(
        example_id=str(payload["example_id"]),
        source_task=str(payload["source_task"]),
        repo=RepoSource(**dict(payload["repo"])),
        index=int(payload["index"]),
        context=str(payload["context"]),
        q1_question=str(payload["q1_question"]),
        tool_event=str(payload["tool_event"]),
        q2_question=str(payload["q2_question"]),
        final_q2_prompt=str(payload["final_q2_prompt"]),
        answer=str(payload["answer"]),
        q1=dict(payload["q1"]),
        q2=dict(payload["q2"]),
        audit=ManifestAudit(**audit_payload),
        metadata=dict(payload.get("metadata", {})),
    )


def split_prepared_from_manifest_row(row: Phase15ManifestRow, tokenizer) -> SplitPreparedExample:
    """Prepare a frozen manifest row for the Phase 6 cache-repair runner."""
    span_payloads = row.metadata.get("relevant_spans")
    segment_payloads = row.metadata.get("prefill_segments")
    if not isinstance(span_payloads, list) or not isinstance(segment_payloads, list):
        raise ValueError("Manifest row is missing persisted span/segment metadata.")
    spans = [RelevantSpan(**dict(span)) for span in span_payloads]
    segments = [PrefillSegment(**dict(segment)) for segment in segment_payloads]
    target_context_length = int(row.metadata.get("target_context_length", len(row.context)))
    base_metadata = {
        **dict(row.metadata),
        "source_task": row.source_task,
        "repo": row.repo.to_dict(),
        "q1": dict(row.q1),
        "q2": dict(row.q2),
        "tool_event": row.tool_event,
        "q2_question": row.q2_question,
        "repair_cue": row.tool_event,
    }
    q1_example = TaskExample(
        index=int(row.index),
        task_name="repodelta_edge",
        task_family="repo_delta",
        context=row.context,
        question=row.q1_question,
        answer_prefix=ANSWER_PREFIX,
        outputs=[str(row.q1["answer"])],
        max_new_tokens=MAX_NEW_TOKENS,
        target_context_length=target_context_length,
        relevant_spans=spans,
        prefill_segments=segments,
        metadata={**base_metadata, "turn": "q1"},
    )
    q2_example = TaskExample(
        index=int(row.index),
        task_name="repodelta_edge",
        task_family="repo_delta",
        context=row.context,
        question=row.final_q2_prompt,
        answer_prefix=ANSWER_PREFIX,
        outputs=[row.answer],
        max_new_tokens=MAX_NEW_TOKENS,
        target_context_length=target_context_length,
        relevant_spans=spans,
        prefill_segments=segments,
        metadata={**base_metadata, "turn": "q2"},
    )
    return SplitPreparedExample(
        split_spec=PHASE15_SPLIT_SPEC,
        base_example=q1_example,
        q1_prepared=prepare_example_for_model(q1_example, tokenizer),
        q2_prepared=prepare_example_for_model(q2_example, tokenizer),
        q1_span_names=("repo_fact_1",),
        q2_span_names=("repo_fact_2",),
    )


def encode_repair_signal(tokenizer, row: Phase15ManifestRow, *, mode: str) -> torch.Tensor:
    """Encode the cue used for Phase 15 row scoring."""
    if mode == "event_only":
        text = "\n\n" + row.tool_event
    elif mode == "event_plus_q2":
        text = "\n\n" + row.final_q2_prompt + ANSWER_PREFIX
    elif mode == "stale_event":
        text = "\n\n" + row.q1_question
    elif mode == "wrong_event":
        wrong = str(row.metadata.get("wrong_event", "Tool event: unrelated repository check failed."))
        text = "\n\n" + wrong
    else:
        raise ValueError(f"Unsupported repair signal mode: {mode!r}")
    return tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)


def _stable_example_id(repo: RepoSource, q2_meta: dict[str, Any], index: int) -> str:
    payload = "|".join(
        [
            repo.repo_id,
            repo.commit_sha,
            str(q2_meta.get("path", "")),
            str(q2_meta.get("line_no", "")),
            str(q2_meta.get("answer", "")),
            str(index),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"phase15:{repo.repo_id}:{digest}"
