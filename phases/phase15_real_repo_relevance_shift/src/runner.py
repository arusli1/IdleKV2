"""Phase 15 runner-facing helpers.

This module deliberately avoids launching GPU experiments. It defines the
event-only repair signal contract that a later GPU wrapper must use instead of
feeding the full Q2 prompt into Phase 6 scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .manifest import Phase15ManifestRow, encode_repair_signal


@dataclass(frozen=True)
class RepairSignal:
    """Tokenized signal used to score evicted rows before Q2 decoding."""

    mode: str
    ids: torch.Tensor
    decode_prompt_mode: str


def build_repair_signal(tokenizer, row: Phase15ManifestRow, *, mode: str = "event_only") -> RepairSignal:
    """Build a tokenized Phase 15 repair signal.

    Main paper runs should use `mode='event_only'`. `event_plus_q2` is reserved
    for a boundary ablation and should not be described as tool-event repair.
    """
    ids = encode_repair_signal(tokenizer, row, mode=mode)
    return RepairSignal(mode=mode, ids=ids, decode_prompt_mode="event_plus_q2")


def with_wrong_event(row: Phase15ManifestRow, donor: Phase15ManifestRow) -> Phase15ManifestRow:
    """Attach a donor event for WrongEvent-K without mutating the original row."""
    metadata = dict(row.metadata)
    metadata["wrong_event"] = donor.tool_event
    metadata["wrong_event_source_example_id"] = donor.example_id
    metadata["wrong_event_source_repo_id"] = donor.repo.repo_id
    return replace(row, metadata=metadata)


def tool_file_positions(
    *,
    evicted_positions: list[int] | tuple[int, ...],
    segment_token_ranges: list[tuple[str, int, int]],
    q2_path: str,
    k: int,
) -> list[int]:
    """Select up to K evicted positions from the Q2 file segment.

    This supports the planned ToolFile-K heuristic control: if naming the file
    alone is enough, this selector should approach IdleKV.
    """
    target_segment = f"file:{q2_path}"
    ranges = [(int(start), int(end)) for name, start, end in segment_token_ranges if name == target_segment]
    if not ranges:
        return []
    available = sorted(int(position) for position in evicted_positions)
    selected: list[int] = []
    for start, end in ranges:
        for position in available:
            if start <= position < end:
                selected.append(position)
                if len(selected) >= int(k):
                    return selected
    return selected
