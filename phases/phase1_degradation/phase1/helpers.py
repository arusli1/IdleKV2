"""Small shared utilities for I/O, random synthetic data, and context sizing."""

from __future__ import annotations

import json
import math
import random
import re
import string
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .paths import RULER_JSON_DIR


def ensure_parent(path: Path) -> None:
    """Create the parent directory for an output file if it does not exist yet."""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of JSON rows with one object per line."""
    # Ensure the output directory exists before writing.
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=True)
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON rows while ignoring blank lines."""
    # Skip empty lines so files can be safely concatenated or edited.
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_essay_words() -> list[str]:
    """Load the long natural-language filler corpus vendored from RULER assets."""
    # Load and split the essay corpus into words for filler generation.
    essay_path = RULER_JSON_DIR / "PaulGrahamEssays.json"
    with open(essay_path, "r", encoding="utf-8") as handle:
        essay_text = json.load(handle)["text"]
    return essay_text.replace("\n", " ").split()


def load_prose_words() -> list[str]:
    """Load long-form prose while removing dense numeric tokens and footnote markers."""
    essay_path = RULER_JSON_DIR / "PaulGrahamEssays.json"
    with open(essay_path, "r", encoding="utf-8") as handle:
        text = json.load(handle)["text"]

    text = text.replace("\n", " ")
    text = re.sub(r"\[\d+\]", " ", text)
    words: list[str] = []
    for token in text.split():
        if any(char.isdigit() for char in token) or "%" in token or "$" in token:
            continue
        if token.lower().startswith(("http://", "https://")):
            continue
        if not any(char.isalpha() for char in token):
            continue
        words.append(token)
    if not words:
        raise ValueError("Sanitized prose filler is empty.")
    return words


def filler_words(kind: str) -> list[str]:
    """Choose the word source used to bulk out a synthetic long context."""
    # Map the requested filler type to its word source.
    if kind == "essay":
        return load_essay_words()
    if kind == "prose":
        return load_prose_words()
    if kind == "noise":
        return "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.".split()
    raise ValueError(f"Unknown filler kind: {kind}")


def repeat_to_length(words: list[str], total_words: int) -> list[str]:
    """Repeat a word list until it reaches the requested length."""
    # Repeat the list to reach the required length, then trim.
    if total_words <= len(words):
        return words[:total_words]
    repeats = math.ceil(total_words / len(words))
    return (words * repeats)[:total_words]


def render_inserted_text(
    base_words: list[str],
    haystack_word_count: int,
    inserts: list[str],
    depths: list[float],
) -> tuple[str, list[tuple[int, int]]]:
    """Insert target strings into filler text at approximately fixed depth fractions."""
    # Build a full haystack, then splice in the inserts while recording their spans.
    words = repeat_to_length(base_words, haystack_word_count)
    cut_points = sorted(int(haystack_word_count * depth) for depth in depths)
    cursor = 0
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    for insert, cut_point in zip(inserts, cut_points):
        # Section 1: materialize filler up to the next insertion point.
        chunk = " ".join(words[cursor:cut_point]).strip()
        if chunk:
            parts.append(chunk)
        # Section 2: append the insert and record its character span.
        prefix = " ".join(parts)
        text = f"{prefix} {insert}".strip() if prefix else insert
        char_start = len(text) - len(insert)
        char_end = len(text)
        parts.append(insert)
        spans.append((char_start, char_end))
        cursor = cut_point
    # Section 3: append the remaining filler to finish the haystack.
    tail = " ".join(words[cursor:]).strip()
    if tail:
        parts.append(tail)
    return " ".join(part for part in parts if part).strip(), spans


def make_rng(seed: int) -> random.Random:
    """Create a deterministic Python RNG for reproducible synthetic samples."""
    # Standardized RNG constructor for repeatability across helpers.
    return random.Random(seed)


def random_number(rng: random.Random, digits: int = 5) -> str:
    """Generate a fixed-width random decimal string."""
    # Generate a number in the requested digit range and stringify it.
    low = 10 ** (digits - 1)
    high = (10**digits) - 1
    return str(rng.randint(low, high))


def random_uuidish(rng: random.Random) -> str:
    """Generate a UUID-like lowercase identifier without depending on uuid4()."""
    # Produce 5 groups of lowercase letters/digits to mimic UUID formatting.
    alphabet = string.ascii_lowercase + string.digits
    groups = [8, 4, 4, 4, 12]
    return "-".join("".join(rng.choice(alphabet) for _ in range(size)) for size in groups)


def random_key(rng: random.Random, prefix: str = "key") -> str:
    """Generate a human-readable synthetic key token."""
    # Emit a readable key prefix plus a random numeric suffix.
    return f"{prefix}-{random_number(rng, digits=5)}"


def random_var(rng: random.Random) -> str:
    """Generate a synthetic variable name for variable-tracking tasks."""
    # Use uppercase letters to make variable names stand out in text.
    letters = "".join(rng.choice(string.ascii_uppercase) for _ in range(4))
    return f"VAR_{letters}"


def join_natural(items: list[str]) -> str:
    """Join a short list in a way that reads naturally inside task prompts."""
    # Handle the 1-2 item cases explicitly for natural phrasing.
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def binary_search_word_count(
    target_tokens: int,
    token_counter: Callable[[int], int],
    minimum: int = 32,
    maximum: int | None = None,
) -> int:
    """Find the largest word count whose rendered token length stays on budget."""
    # Expand the search window, then binary search for the largest safe word count.
    if maximum is None:
        maximum = max(target_tokens * 8, minimum * 2)
    # Grow the search window until it definitely brackets the target token
    # count, then binary-search the largest feasible filler length.
    while token_counter(maximum) <= target_tokens:
        maximum *= 2
    best = minimum
    low = minimum
    high = maximum
    while low <= high:
        mid = (low + high) // 2
        tokens = token_counter(mid)
        # We keep the largest count that still fits because denser filler makes
        # the task closer to a true long-context benchmark.
        if tokens <= target_tokens:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def np_rng(seed: int) -> np.random.Generator:
    """Create a deterministic NumPy RNG for tasks that sample arrays directly."""
    # NumPy-specific RNG for array-based tasks.
    return np.random.default_rng(seed)
