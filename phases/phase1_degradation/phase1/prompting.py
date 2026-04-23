"""Prompt-rendering helpers that expose the exact token layout the model sees."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RenderedExample:
    """Tokenized split between the prefed context and the later question suffix."""

    # Strings are kept for tracing/debugging; tensors are what actually feed the model.
    rendered_context: str
    template_prefix: str
    question_suffix: str
    context_ids: torch.Tensor
    question_ids: torch.Tensor


def render_context_plan(tokenizer, context: str, question: str, answer_prefix: str) -> RenderedExample:
    """Render the chat template once, then split it into prefill and resume pieces."""
    separator_token = "<|phase1_kvr_separator|>"
    # We insert a synthetic marker into the user message so we can ask the
    # tokenizer to render the final chat template once and then split it back
    # into "context already seen" and "question arriving after the gap".
    while separator_token in context:
        separator_token += "_x"
    separator = "\n\n" + separator_token

    # Let the tokenizer add any chat-template wrappers the model expects,
    # because those wrappers count toward the true KV cache length.
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": context + separator}],
        add_generation_prompt=True,
        tokenize=False,
    )
    # Split point: everything before the synthetic marker is the prefillable
    # context; everything after becomes the question suffix when we resume.
    rendered_context, question_suffix = rendered.split(separator)

    # `template_prefix` is the chat-template text that appears before the raw
    # user context. We need its length later when mapping raw prompt spans to
    # the rendered token positions the model actually saw.
    template_prefix = rendered_context[: len(rendered_context) - len(context)]
    # Assemble the resume-time payload: question + post-marker template tail +
    # answer prefix, in the exact order the model will see it.
    question_text = "\n\n" + question + question_suffix + answer_prefix
    # Encode each side separately because Phase 1 prefills the context first
    # and only later appends the question and answer stub.
    context_ids = tokenizer.encode(rendered_context, return_tensors="pt", add_special_tokens=False)
    question_ids = tokenizer.encode(question_text, return_tensors="pt", add_special_tokens=False)
    return RenderedExample(
        rendered_context=rendered_context,
        template_prefix=template_prefix,
        question_suffix=question_suffix,
        context_ids=context_ids,
        question_ids=question_ids,
    )


def count_rendered_context_tokens(tokenizer, context: str) -> int:
    """Measure context length after the chat template has expanded around it."""
    # This gives the true KV length for a prefill-only pass, not just raw text tokens.
    rendered = render_context_plan(tokenizer, context=context, question="", answer_prefix="")
    return int(rendered.context_ids.shape[1])


def char_span_to_token_positions(tokenizer, rendered_context: str, char_start: int, char_end: int) -> list[int]:
    """Translate a character span in the rendered prompt into token positions."""
    # Offsets are computed on the fully rendered template so positions align
    # with the model's actual token indices, not the raw user text.
    encoded = tokenizer(rendered_context, add_special_tokens=False, return_offsets_mapping=True)
    if "offset_mapping" not in encoded:
        raise RuntimeError("Tokenizer does not support offset_mapping; cannot map spans to token positions.")
    positions: list[int] = []
    for token_index, (token_start, token_end) in enumerate(encoded["offset_mapping"]):
        # Offset mapping gives us a direct char-to-token bridge; scan until we
        # leave the span, and collect every overlapping token index.
        # Skip tokens before the span and stop once we have moved past it; that
        # keeps the scan cheap even for 32K-token contexts.
        if token_end <= char_start:
            continue
        if token_start >= char_end:
            break
        positions.append(token_index)
    # Returning every overlapping token lets the trace code reason about
    # partial survival rather than collapsing a span to one single index.
    return positions
