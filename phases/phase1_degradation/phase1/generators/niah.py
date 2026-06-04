"""Needle-in-a-haystack task generation for single- and multi-query variants.

This module synthesizes a long context with hidden key/value "needles" and
packages the resulting prompt, answer, and span metadata into TaskExample.
"""

from __future__ import annotations

from ..helpers import join_natural, make_rng, random_key, random_number
from ..models import PrefillSegment, RelevantSpan, TaskExample
from .common import search_context


def build_niah_example(
    task_name: str,
    index: int,
    target_context_length: int,
    tokenizer,
    *,
    mode: str,
    num_needles: int,
    depths: list[float],
    dataset_seed_offset: int = 0,
) -> TaskExample:
    """Build a NIAH example with deterministic needle placement and answer metadata."""
    if len(depths) > num_needles:
        raise ValueError(f"Expected at most {num_needles} fixed depths for {task_name}; got {len(depths)}.")

    # --- Key/value generation: deterministic needles per task shape + index ---
    # Seed from task shape and sample index so reruns regenerate the exact same
    # hidden keys, values, and placement pattern.
    rng = make_rng((target_context_length * 1000) + index + len(task_name) + int(dataset_seed_offset))
    key_values = []
    if mode == "multi_value":
        # Multi-value mode hides several answers behind the same key so recall,
        # not just lookup, becomes the bottleneck.
        shared_key = random_key(rng, prefix="artifact")
        for _ in range(num_needles):
            key_values.append((shared_key, random_number(rng)))
    else:
        for _ in range(num_needles):
            key_values.append((random_key(rng, prefix="artifact"), random_number(rng)))

    # --- Placement strategy: build insert strings and assign body vs tail ---
    inserts = [f"One of the special magic values for {key} is: {value}." for key, value in key_values]
    body_specs: list[tuple[str, str, str, float, dict]] = []
    tail_specs: list[tuple[str, str, str, float, dict]] = []
    for needle_index, ((key, value), insert) in enumerate(zip(key_values, inserts), start=1):
        # Depth-controlled needles go into the main body; any extras are forced
        # to the end of the prompt to create a harsher eviction target.
        target_specs = body_specs if needle_index <= len(depths) else tail_specs
        target_specs.append(
            (
                f"needle_{needle_index}",
                "needle",
                insert,
                depths[needle_index - 1] if needle_index <= len(depths) else 1.0,
                {"key": key, "value": value},
            )
        )

    prefix = "Some special magic values are hidden within the following text. Memorize them carefully.\n\n"
    built = search_context(
        tokenizer=tokenizer,
        target_context_length=target_context_length,
        prefix=prefix,
        base_filler_kind="essay",
        inserts=[spec[2] for spec in body_specs],
        depths=[spec[3] for spec in body_specs],
        tail_inserts=[spec[2] for spec in tail_specs],
    )

    # --- Span bookkeeping: map inserted text back to character ranges ---
    # Convert the inserted strings into explicit span objects so evaluation can
    # later ask "which exact needle was dropped?" instead of just "was answer wrong?".
    relevant_spans = []
    for (name, kind, _, depth_fraction, metadata), (char_start, char_end) in zip(body_specs + tail_specs, built.insert_spans):
        # Keep the exact character span for each needle so later we can map it
        # to token positions and measure whether compression kept it alive.
        relevant_spans.append(
            RelevantSpan(
                name=name,
                kind=kind,
                char_start=char_start,
                char_end=char_end,
                depth_fraction=depth_fraction,
                metadata=metadata,
            )
        )

    # --- Question/answer setup: choose keys to ask about and expected values ---
    if mode == "single":
        # Single-query is the cleanest "did the one needle survive?" setting.
        query_keys = [key_values[0][0]]
        outputs = [key_values[0][1]]
    elif mode == "multi_query":
        # Multi-query asks for several independent keys in one answer.
        query_keys = [key for key, _ in key_values]
        outputs = [value for _, value in key_values]
    elif mode == "multi_value":
        # Multi-value asks for several values hidden behind one shared key.
        query_keys = [key_values[0][0]]
        outputs = [value for _, value in key_values]
    else:
        raise ValueError(f"Unsupported NIAH mode: {mode}")

    # The question phrasing stays simple on purpose; the benchmark should be
    # about cache survival, not about prompt interpretation quirks.
    query = join_natural(query_keys)
    question = f"What are the special magic values for {query} mentioned in the provided text?"
    answer_prefix = f" The special magic values for {query} mentioned in the provided text are"
    return TaskExample(
        index=index,
        task_name=task_name,
        task_family="niah",
        context=built.context,
        question=question,
        answer_prefix=answer_prefix,
        outputs=outputs,
        max_new_tokens=max(32, 16 * len(outputs)),
        target_context_length=target_context_length,
        relevant_spans=relevant_spans,
        prefill_segments=[PrefillSegment(name="context", char_start=0, char_end=len(built.context))],
        metadata={
            "mode": mode,
            "query_keys": query_keys,
            "depths": depths,
            "num_needles": num_needles,
            "tail_anchored_needles": max(num_needles - len(depths), 0),
        },
    )
