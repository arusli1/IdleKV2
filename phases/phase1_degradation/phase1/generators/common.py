"""Shared context-construction helpers for the synthetic task generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..helpers import binary_search_word_count, filler_words, render_inserted_text
from ..prompting import count_rendered_context_tokens


@dataclass
class BuiltContext:
    """Rendered context text plus the character spans of inserted target strings."""

    context: str
    insert_spans: list[tuple[int, int]]


def search_context(
    tokenizer,
    target_context_length: int,
    prefix: str,
    base_filler_kind: str,
    inserts: list[str],
    depths: list[float],
    tail_inserts: list[str] | None = None,
) -> BuiltContext:
    """Fit a synthetic context to a token budget while keeping inserts near target depths."""
    # High-level flow:
    # 1) Select filler words based on the requested base filler kind.
    # 2) Build a context candidate with inserts placed at target depths.
    # 3) Use a binary search on filler word count to satisfy the token budget.
    base_words = filler_words(base_filler_kind)
    tail_inserts = tail_inserts or []

    def build(count: int) -> BuiltContext:
        # Build the main filler body with inserts placed at depth targets.
        body, spans = render_inserted_text(base_words, count, inserts, depths)
        # Assemble the full context by applying the prefix before the filler body.
        context = prefix + body
        # Adjust span offsets to account for the prefix length.
        offset = len(prefix)
        shifted = [(start + offset, end + offset) for start, end in spans]
        if tail_inserts:
            # Tail inserts are appended after the main filler so they land
            # near the end of the context regardless of the binary search.
            # This keeps late-context probes stable while the filler size changes.
            if context and not context.endswith(" "):
                context += " "
            cursor = len(context)
            for insert_index, insert in enumerate(tail_inserts):
                start = cursor
                # Append each tail insert and record its span.
                context += insert
                cursor += len(insert)
                shifted.append((start, cursor))
                if insert_index != len(tail_inserts) - 1:
                    context += " "
                    cursor += 1
        return BuiltContext(context=context, insert_spans=shifted)

    def token_counter(count: int) -> int:
        # Section: render full context so token counting reflects actual template cost.
        return count_rendered_context_tokens(tokenizer, build(count).context)

    # Section: binary search on filler word count to fit the token budget.
    # The raw word count is just a proxy; the binary search keeps adjusting it
    # until the rendered chat-template token length lands just under budget.
    best_count = binary_search_word_count(target_context_length, token_counter)
    return build(best_count)


def build_fixed_parts_context(parts: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate already-prepared text fragments and record each fragment span."""
    # This helper preserves exact fragment boundaries so callers can map back
    # to the original pieces without re-tokenizing the composed text.
    text_parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for part in parts:
        start = cursor
        text_parts.append(part)
        cursor += len(part)
        spans.append((start, cursor))
    return "".join(text_parts), spans
