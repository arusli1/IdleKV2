"""Cross-turn memory-killer NIAH variant with a target and a late distractor."""

from __future__ import annotations

from ..helpers import make_rng, random_key, random_number, repeat_to_length, filler_words
from ..models import PrefillSegment, RelevantSpan, TaskExample
from ..prompting import count_rendered_context_tokens


def build_cross_turn_mk_niah_example(task_name: str, index: int, target_context_length: int, tokenizer) -> TaskExample:
    """Build a two-turn context where the model must keep the original value, not the distractor."""
    # Deterministically derive a unique key/value pair for this example.
    rng = make_rng((target_context_length * 3000) + index)
    key = random_key(rng, prefix="artifact")
    target_value = random_number(rng)
    distractor_value = random_number(rng)
    while distractor_value == target_value:
        distractor_value = random_number(rng)

    # Target/distractor setup: the first turn states the true value, the second
    # turn injects a conflicting distractor later in the context.
    turn_1_prefix = (
        "The original mapping appears in the first section. Keep the original value even if later text contains a distractor.\n\n"
    )
    turn_1_insert = f"The original special magic value for {key} is: {target_value}."
    turn_2_insert = f"A later distractor claims the special magic value for {key} is: {distractor_value}."
    base_words = filler_words("essay")

    def build_with_word_count(total_words: int) -> tuple[str, list[tuple[int, int]], int]:
        # Context assembly: split filler into two "turns" and inject the target
        # in the first turn, then the distractor late in the second turn.
        # Split the filler into an "original statement" region and a later
        # update region so the benchmark can mimic cross-turn interference.
        turn_1_words = max(int(total_words * 0.55), 64)
        turn_2_words = max(total_words - turn_1_words, 64)
        part_1_words = repeat_to_length(base_words, turn_1_words)
        part_2_words = repeat_to_length(base_words[::-1], turn_2_words)
        part_1_body = " ".join(part_1_words[: int(turn_1_words * 0.25)]) + " " + turn_1_insert + " " + " ".join(part_1_words[int(turn_1_words * 0.25) :])
        part_2_body = " ".join(part_2_words[: int(turn_2_words * 0.8)]) + " " + turn_2_insert + " " + " ".join(part_2_words[int(turn_2_words * 0.8) :])
        part_1 = turn_1_prefix + part_1_body.strip()
        separator = "\n\nAdditional update:\n\n"
        full_context = part_1 + separator + part_2_body.strip()
        # Record both the true target and the distractor so later analysis can
        # tell whether eviction preserved the right fact but not the wrong one.
        target_start = full_context.index(turn_1_insert)
        target_end = target_start + len(turn_1_insert)
        distractor_start = full_context.index(turn_2_insert)
        distractor_end = distractor_start + len(turn_2_insert)
        turn_split = len(part_1)
        return full_context, [(target_start, target_end), (distractor_start, distractor_end)], turn_split

    def token_count(total_words: int) -> int:
        # Count rendered tokens for the composed context at a given word budget.
        context, _, _ = build_with_word_count(total_words)
        return count_rendered_context_tokens(tokenizer, context)

    low = 64
    high = max(target_context_length * 8, 256)
    while token_count(high) <= target_context_length:
        high *= 2
    best = low
    while low <= high:
        # Use the same token-budget fitting pattern as the other generators so
        # all tasks land near the requested rendered length.
        mid = (low + high) // 2
        if token_count(mid) <= target_context_length:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    # Final example construction: build the sized context, then wrap spans and
    # prefill boundaries so downstream evaluation can score target vs distractor.
    context, spans, turn_split = build_with_word_count(best)
    question = f"What is the original special magic value for {key}?"
    answer_prefix = f" Answer: The original special magic value for {key} is "
    return TaskExample(
        index=index,
        task_name=task_name,
        task_family="cross_turn_niah",
        context=context,
        question=question,
        answer_prefix=answer_prefix,
        outputs=[target_value],
        max_new_tokens=32,
        target_context_length=target_context_length,
        relevant_spans=[
            # The first span is the answer-bearing target; the second is a late
            # distractor we expect the model to ignore if memory survives.
            RelevantSpan(
                name="target_needle",
                kind="needle",
                char_start=spans[0][0],
                char_end=spans[0][1],
                depth_fraction=0.25,
                metadata={"key": key, "value": target_value},
            ),
            RelevantSpan(
                name="distractor_needle",
                kind="distractor",
                char_start=spans[1][0],
                char_end=spans[1][1],
                depth_fraction=0.9,
                metadata={"key": key, "value": distractor_value},
            ),
        ],
        prefill_segments=[
            # The explicit split lets inference prefill the two "turns" in order,
            # which mirrors the later gap/eviction analysis.
            PrefillSegment(name="turn_1", char_start=0, char_end=turn_split),
            PrefillSegment(name="turn_2", char_start=turn_split, char_end=len(context)),
        ],
        metadata={
            "key": key,
            "target_value": target_value,
            "distractor_value": distractor_value,
        },
    )
