"""Frequent-word extraction task generation."""

from __future__ import annotations

import numpy as np

from ..helpers import make_rng, np_rng
from ..models import PrefillSegment, TaskExample
from ..prompting import count_rendered_context_tokens


def build_fwe_example(task_name: str, index: int, target_context_length: int, tokenizer, *, alpha: float = 2.0, vocab_size: int = 64) -> TaskExample:
    """Build a Zipfian coded-text task where the model must recover the top words."""
    # Seed both RNGs deterministically so prompts and answers are reproducible
    # for a given (target_context_length, index) pair.
    rng = make_rng((target_context_length * 4000) + index)
    nrng = np_rng((target_context_length * 4000) + index)
    # Define a fixed coded vocabulary to sample from.
    vocab = [f"cw{token_index:03d}" for token_index in range(vocab_size)]
    # Build a Zipf distribution that heavily favors a few codes.
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    weights = ranks ** (-alpha)
    probabilities = weights / weights.sum()

    # Human-readable instruction prefix; the sampled words are appended below.
    prefix = "Read the following coded text and track the frequency of each coded word.\n\n"

    def build_with_word_count(word_count: int) -> str:
        # Sampling stage: draw coded words from the Zipfian distribution.
        # The Zipfian sampler makes a few coded words dominate frequency, which
        # gives the task a stable "top 3" answer even in long random contexts.
        sampled = nrng.choice(vocab, size=word_count, p=probabilities, replace=True)
        return prefix + " ".join(sampled.tolist())

    def token_count(word_count: int) -> int:
        # Sizing stage: estimate rendered token count for a candidate word count.
        return count_rendered_context_tokens(tokenizer, build_with_word_count(word_count))

    # Sizing stage: binary search for a word count that fits the target context length.
    low = 128
    high = max(target_context_length * 8, 512)
    while token_count(high) <= target_context_length:
        high *= 2
    best = low
    while low <= high:
        # Tune the number of sampled coded words until the rendered prompt is
        # as close as possible to the requested token budget.
        mid = (low + high) // 2
        if token_count(mid) <= target_context_length:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    # Sampling stage: finalize the sampled context at the chosen size.
    context = build_with_word_count(best)
    # Counting stage: tally word frequencies from the rendered context.
    words = context[len(prefix) :].split()
    counts: dict[str, int] = {}
    for word in words:
        # Count frequencies directly from the rendered text so the gold answer
        # matches the exact prompt the model receives.
        counts[word] = counts.get(word, 0) + 1
    # Answer construction stage: select the top 3 words with deterministic tie-breaking.
    # Ties are broken lexicographically so the answer is deterministic.
    top_words = [word for word, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]]
    question = "What are the three most frequently appeared coded words in the text above?"
    answer_prefix = " Answer: According to the coded text above, the three most frequently appeared words are:"
    return TaskExample(
        index=index,
        task_name=task_name,
        task_family="fwe",
        context=context,
        question=question,
        answer_prefix=answer_prefix,
        outputs=top_words,
        max_new_tokens=50,
        target_context_length=target_context_length,
        relevant_spans=[],
        prefill_segments=[PrefillSegment(name="context", char_start=0, char_end=len(context))],
        metadata={"alpha": alpha, "vocab_size": vocab_size},
    )
