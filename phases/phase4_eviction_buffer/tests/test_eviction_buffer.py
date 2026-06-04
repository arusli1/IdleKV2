"""Unit tests for the Phase 4 buffer core."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from phases.phase4_eviction_buffer.src.buffer.eviction_buffer import BufferEntry, EvictionBuffer
from phases.phase4_eviction_buffer.src.buffer.profiling import SyntheticKVSpec, build_buffer_from_log_artifact


def _make_entry(position: int, importance: float, q_scale: float) -> BufferEntry:
    kv = (
        (
            torch.zeros((1, 1, 1, 2), dtype=torch.float16),
            torch.zeros((1, 1, 1, 2), dtype=torch.float16),
        ),
        (
            torch.zeros((1, 1, 1, 2), dtype=torch.float16),
            torch.zeros((1, 1, 1, 2), dtype=torch.float16),
        ),
    )
    q_vec = torch.full((2, 2), float(q_scale), dtype=torch.float32)
    return BufferEntry(position=position, kv=kv, importance_score=importance, q_vec=q_vec)


class EvictionBufferTests(unittest.TestCase):
    def test_trim_keeps_highest_importance_entries(self) -> None:
        buffer = EvictionBuffer(max_tokens=2, selection_strategy="l2_norm")
        buffer.push(_make_entry(10, 0.2, 1.0))
        buffer.push(_make_entry(20, 0.8, 1.0))
        buffer.push(_make_entry(30, 0.5, 1.0))

        self.assertEqual(len(buffer), 2)
        self.assertEqual(sorted(buffer._entries), [20, 30])

    def test_dot_product_query_prefers_matching_q_vector(self) -> None:
        buffer = EvictionBuffer(max_tokens=4, selection_strategy="dot_product")
        buffer.push(_make_entry(10, 0.3, 0.0))
        buffer.push(_make_entry(20, 0.4, 3.0))
        recent_q = torch.full((2, 1, 2), 2.0, dtype=torch.float32)

        selected = buffer.query(recent_q, top_k=1)
        self.assertEqual([entry.position for entry in selected], [20])

    def test_build_buffer_from_single_log_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "ex001.json"
            qvec_path = Path(tmpdir) / "ex001_qvecs.pt"
            log_path.write_text(
                "{\n"
                '  "evicted_positions": [11, 12, 13],\n'
                '  "importance_scores": {"11": 0.1, "12": 0.4, "13": 0.2}\n'
                "}\n",
                encoding="utf-8",
            )
            torch.save(torch.ones((2, 3, 4), dtype=torch.float32), qvec_path)

            buffer = build_buffer_from_log_artifact(
                log_path,
                strategy="l2_norm",
                max_tokens=10,
                synthetic_spec=SyntheticKVSpec(n_layers=2, n_kv_heads=1, head_dim=4),
            )

            self.assertEqual(len(buffer), 3)
            self.assertEqual(sorted(buffer._entries), [11, 12, 13])


if __name__ == "__main__":
    unittest.main()
