"""Real-repository relevance-shift examples for Phase 14.

The RepoDelta diagnostic is intentionally narrower than end-to-end coding-agent
success. It builds two-turn retrieval examples from real repository files so a
future GPU smoke can test whether a tool-like event shifts relevance to a
different file under the same matched-budget repair protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import random
import re
from typing import Iterable, Sequence

from phases.phase1_degradation.phase1.inference import prepare_example_for_model
from phases.phase1_degradation.phase1.models import PrefillSegment, RelevantSpan, TaskExample
from phases.phase6_repair.src.protocol import SplitPreparedExample, SplitTaskSpec

ANSWER_PREFIX = " Answer:"
MAX_NEW_TOKENS = 16
DEFAULT_EXTENSIONS = (
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
)
IGNORED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "aux",
    "models",
    "node_modules",
    "results",
    "saved_results",
}


@dataclass(frozen=True)
class RepoDeltaCandidate:
    """One answerable exact-span fact from a repository file."""

    path: str
    kind: str
    answer: str
    line_no: int
    line_text: str
    answer_start_in_line: int
    answer_end_in_line: int


@dataclass(frozen=True)
class RepoDeltaSplitSpec:
    """Fixed Q1/Q2 candidate pair for a RepoDelta example."""

    name: str = "repodelta_q1_to_q2"
    q1_index: int = 0
    q2_index: int = 1


def _stable_seed(index: int, seed_offset: int) -> int:
    payload = f"repodelta:{index}:{seed_offset}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _is_ignored(path: Path) -> bool:
    return any(part in IGNORED_DIRS or part.startswith(".") and part != "." for part in path.parts)


def iter_repo_files(
    repo_root: Path | str,
    *,
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
    max_bytes: int = 80_000,
) -> list[Path]:
    """Return deterministic candidate files from a repository tree."""
    root = Path(repo_root)
    paths: list[Path] = []
    allowed = {ext.lower() for ext in extensions}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _is_ignored(path.relative_to(root)):
            continue
        if path.suffix.lower() not in allowed:
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
        except OSError:
            continue
        paths.append(path)
    return paths


_CANDIDATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("function", re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")),
    ("class", re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b")),
    ("function", re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")),
    ("constant", re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=")),
    ("constant", re.compile(r"^\s*([A-Z][A-Z0-9_]{3,})\s*=")),
)


def extract_candidates(path: str, text: str) -> list[RepoDeltaCandidate]:
    """Extract short identifier answers that can be asked about by file+line."""
    candidates: list[RepoDeltaCandidate] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for kind, pattern in _CANDIDATE_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            answer = match.group(1)
            if len(answer) < 4 or len(answer) > 64:
                continue
            candidates.append(
                RepoDeltaCandidate(
                    path=path,
                    kind=kind,
                    answer=answer,
                    line_no=line_no,
                    line_text=line,
                    answer_start_in_line=match.start(1),
                    answer_end_in_line=match.end(1),
                )
            )
            break
    return candidates


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except UnicodeDecodeError:
            return None
    except OSError:
        return None


def _file_card(rel_path: str, text: str) -> str:
    numbered = "\n".join(
        f"{line_no:04d}: {line}" for line_no, line in enumerate(text.rstrip().splitlines(), start=1)
    )
    return f"--- FILE: {rel_path} ---\n{numbered}\n"


def _line_char_bounds(card_start: int, card: str, line_no: int) -> tuple[int, int]:
    offset = 0
    lines = card.splitlines(keepends=True)
    for current, line in enumerate(lines, start=1):
        next_offset = offset + len(line)
        if current == line_no + 1:
            return card_start + offset, card_start + next_offset
        offset = next_offset
    raise ValueError(f"Line {line_no} is outside generated card.")


def _choose_pair(candidates: Sequence[RepoDeltaCandidate], rng: random.Random) -> tuple[RepoDeltaCandidate, RepoDeltaCandidate]:
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    for q1 in shuffled:
        for q2 in shuffled:
            if q1.path != q2.path and q1.answer != q2.answer:
                return q1, q2
    raise ValueError("Need at least two answer candidates from different files.")


def _unique_answer_candidates(candidates: Iterable[RepoDeltaCandidate], context: str) -> list[RepoDeltaCandidate]:
    unique: list[RepoDeltaCandidate] = []
    for candidate in candidates:
        if context.count(candidate.answer) == 1:
            unique.append(candidate)
    return unique


def build_repodelta_base_example(
    *,
    repo_root: Path | str,
    index: int,
    target_context_length: int,
    tokenizer=None,
    dataset_seed_offset: int = 0,
    max_files: int = 24,
) -> TaskExample:
    """Build one real-repository two-turn relevance-shift example.

    `target_context_length` is a soft raw-character target for this generator.
    The later GPU runner should still verify the rendered token length under the
    model tokenizer before launching a locked run.
    """
    del tokenizer
    root = Path(repo_root)
    files = iter_repo_files(root)
    if not files:
        raise ValueError(f"No candidate source files found under {root}.")

    rng = random.Random(_stable_seed(index, dataset_seed_offset))
    sampled = files[:]
    rng.shuffle(sampled)
    selected_paths = sampled[: max(2, int(max_files))]

    cards: list[tuple[str, str, int, int]] = []
    context_parts: list[str] = []
    cursor = 0
    all_candidates: list[RepoDeltaCandidate] = []
    for path in selected_paths:
        text = _read_text(path)
        if not text:
            continue
        rel = path.relative_to(root).as_posix()
        card = _file_card(rel, text)
        if context_parts:
            context_parts.append("\n")
            cursor += 1
        start = cursor
        end = start + len(card)
        context_parts.append(card)
        cards.append((rel, card, start, end))
        all_candidates.extend(extract_candidates(rel, text))
        cursor = end
        if cursor >= target_context_length and len({candidate.path for candidate in all_candidates}) >= 2:
            break

    context = "".join(context_parts)
    unique_candidates = _unique_answer_candidates(all_candidates, context)
    q1, q2 = _choose_pair(unique_candidates, rng)

    spans: list[RelevantSpan] = []
    segments: list[PrefillSegment] = []
    card_by_path = {rel: (card, start, end) for rel, card, start, end in cards}
    for rel, _card, start, end in cards:
        segments.append(PrefillSegment(name=f"file:{rel}", char_start=start, char_end=end))
    for name, candidate in (("repo_fact_1", q1), ("repo_fact_2", q2)):
        card, card_start, _ = card_by_path[candidate.path]
        line_start, line_end = _line_char_bounds(card_start, card, candidate.line_no)
        spans.append(
            RelevantSpan(
                name=name,
                kind="repo_symbol_line",
                char_start=line_start,
                char_end=line_end,
                depth_fraction=line_start / max(len(context), 1),
                metadata={
                    "path": candidate.path,
                    "kind": candidate.kind,
                    "answer": candidate.answer,
                    "line_no": candidate.line_no,
                    "line_text": candidate.line_text.strip(),
                },
            )
        )

    question = _repo_question(q1, turn="q1")
    metadata = {
        "query_keys": [f"{q1.path}:{q1.line_no}", f"{q2.path}:{q2.line_no}"],
        "repo_root": str(root),
        "q1": _candidate_metadata(q1),
        "q2": _candidate_metadata(q2),
        "tool_event": _tool_event(q2),
        "task_design": "repodelta_retrieval",
    }
    return TaskExample(
        index=index,
        task_name="repodelta_retrieval",
        task_family="repo_delta",
        context=context,
        question=question,
        answer_prefix=ANSWER_PREFIX,
        outputs=[q1.answer, q2.answer],
        max_new_tokens=MAX_NEW_TOKENS,
        target_context_length=target_context_length,
        relevant_spans=spans,
        prefill_segments=segments,
        metadata=metadata,
    )


def _candidate_metadata(candidate: RepoDeltaCandidate) -> dict[str, object]:
    return {
        "path": candidate.path,
        "kind": candidate.kind,
        "answer": candidate.answer,
        "line_no": candidate.line_no,
        "line_text": candidate.line_text.strip(),
    }


def _tool_event(candidate: RepoDeltaCandidate) -> str:
    return (
        "Tool event: a follow-up check failed while executing repository code in "
        f"`{candidate.path}` at line {candidate.line_no}. Inspect that location before answering."
    )


def _repo_question(candidate: RepoDeltaCandidate, *, turn: str) -> str:
    if turn == "q2":
        return (
            _tool_event(candidate)
            + "\n\n"
            + f"What is the {candidate.kind} name at the reported failure location? "
            + "Respond with the identifier only."
        )
    return (
        f"In `{candidate.path}`, what is the {candidate.kind} name on line "
        f"{candidate.line_no}? Respond with the identifier only."
    )


def split_repodelta_for_turn(
    base_example: TaskExample,
    *,
    turn: str,
    split_name: str,
) -> TaskExample:
    """Project a RepoDelta base example into its Q1 or Q2 turn."""
    if turn not in {"q1", "q2"}:
        raise ValueError(f"Unsupported RepoDelta turn: {turn}")
    candidate = base_example.metadata[turn]
    answer = str(candidate["answer"])
    repo_candidate = RepoDeltaCandidate(
        path=str(candidate["path"]),
        kind=str(candidate["kind"]),
        answer=answer,
        line_no=int(candidate["line_no"]),
        line_text=str(candidate["line_text"]),
        answer_start_in_line=0,
        answer_end_in_line=len(answer),
    )
    question = _repo_question(repo_candidate, turn=turn)
    metadata = dict(base_example.metadata)
    metadata["split_name"] = split_name
    metadata["turn"] = turn
    metadata["response_format"] = "identifier_only"
    if turn == "q2":
        metadata["repair_cue"] = _tool_event(repo_candidate)
        metadata["repair_signal_mode"] = "event_only"
        metadata["decode_prompt_mode"] = "event_plus_q2"
    return replace(
        base_example,
        question=question,
        answer_prefix=ANSWER_PREFIX,
        outputs=[answer],
        max_new_tokens=MAX_NEW_TOKENS,
        metadata=metadata,
    )


def build_repodelta_prepared_example(
    *,
    repo_root: Path | str,
    index: int,
    target_context_length: int,
    tokenizer,
    dataset_seed_offset: int = 0,
) -> SplitPreparedExample:
    """Build a Phase-6-compatible prepared Q1/Q2 RepoDelta split."""
    split_spec = SplitTaskSpec(
        name="repodelta_q1_to_q2",
        base_task_key="repodelta_retrieval",
        q1_indices=(0,),
        q2_indices=(1,),
        max_new_tokens=MAX_NEW_TOKENS,
    )
    base = build_repodelta_base_example(
        repo_root=repo_root,
        index=index,
        target_context_length=target_context_length,
        tokenizer=tokenizer,
        dataset_seed_offset=dataset_seed_offset,
    )
    q1 = split_repodelta_for_turn(base, turn="q1", split_name=f"{split_spec.name}:q1")
    q2 = split_repodelta_for_turn(base, turn="q2", split_name=f"{split_spec.name}:q2")
    return SplitPreparedExample(
        split_spec=split_spec,
        base_example=base,
        q1_prepared=prepare_example_for_model(q1, tokenizer),
        q2_prepared=prepare_example_for_model(q2, tokenizer),
        q1_span_names=("repo_fact_1",),
        q2_span_names=("repo_fact_2",),
    )
