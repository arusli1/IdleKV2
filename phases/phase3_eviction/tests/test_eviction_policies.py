"""Deterministic unit tests for the Phase 3 eviction policies."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_ROOT.parents[1]
for root in (PHASE_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache, to_tuple_cache
from src.eviction import QueryAwareSnapKV, SnapKV, StreamingLLM, log_eviction


def make_position_cache(rows: list[list[float]], *, positions: list[int] | None = None) -> PositionTrackedCache:
    """Create a one-layer one-head cache from explicit key/value rows."""
    seq_len = len(rows)
    head_dim = len(rows[0])
    tensor = torch.tensor(rows, dtype=torch.float32).reshape(1, 1, seq_len, head_dim)
    cache = ((tensor.clone(), tensor.clone()),)
    return PositionTrackedCache(cache, positions or list(range(seq_len)))


class ToyQueryAwareModel(nn.Module):
    """Small model stub that appends deterministic query rows to the cache."""

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = None
        self.head_dim = head_dim

    def forward(
        self,
        input_ids,
        past_key_values,
        use_cache: bool = True,
        position_ids=None,
        cache_position=None,
        logits_to_keep=None,
        **_: object,
    ):
        del use_cache, position_ids, cache_position, logits_to_keep
        query_ids = input_ids if input_ids.ndim == 2 else input_ids.unsqueeze(0)
        cache = to_tuple_cache(past_key_values)
        layers = []

        for key, value in cache:
            rows = []
            for token_id in query_ids[0].tolist():
                if token_id == 99:
                    row = torch.tensor([0.0, 1.0], device=key.device, dtype=key.dtype)
                elif token_id == 77:
                    row = torch.tensor([1.0, 0.0], device=key.device, dtype=key.dtype)
                else:
                    row = torch.zeros((self.head_dim,), device=key.device, dtype=key.dtype)
                    row[min(int(token_id) % self.head_dim, self.head_dim - 1)] = 1.0
                rows.append(row)

            query_block = torch.stack(rows, dim=0).reshape(1, 1, query_ids.shape[1], self.head_dim)
            query_block = query_block.expand(key.shape[0], key.shape[1], query_ids.shape[1], self.head_dim).contiguous()
            layers.append(
                (
                    torch.cat([key, query_block], dim=2),
                    torch.cat([value, query_block.clone()], dim=2),
                )
            )

        return SimpleNamespace(past_key_values=tuple(layers))


class EvictionPolicyTests(unittest.TestCase):
    def test_snapkv_respects_budget_and_moves_evicted_cache_to_cpu(self) -> None:
        cache = make_position_cache(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.5],
                [0.0, 0.4],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            positions=[10, 20, 30, 40, 50, 60],
        )
        policy = SnapKV(obs_window_size=2, sink_size=1, recency_window=1, pooling="max")
        result = policy.evict(cache, k_budget=3)

        self.assertEqual(result.compressed.positions, [10, 50, 60])
        self.assertEqual(result.evicted.positions, [20, 30, 40])
        self.assertEqual(len(result.compressed), 3)
        self.assertEqual(len(result.evicted), 3)
        self.assertEqual(set(result.importance_scores), {10, 20, 30, 40, 50, 60})
        self.assertEqual(tuple(result.obs_window_q_vecs.shape), (1, 2, 2))
        self.assertEqual(result.evicted.device.type, "cpu")

    def test_streamingllm_keeps_sink_then_recency(self) -> None:
        cache = make_position_cache([[float(index), 0.0] for index in range(8)], positions=[2, 4, 6, 8, 10, 12, 14, 16])
        policy = StreamingLLM(sink_size=3, recency_window=1)
        result = policy.evict(cache, k_budget=4)

        self.assertEqual(result.compressed.positions, [2, 4, 6, 16])
        self.assertEqual(result.evicted.positions, [8, 10, 12, 14])
        self.assertEqual(result.importance_scores[2], 1.0)
        self.assertEqual(result.importance_scores[10], 0.0)
        self.assertEqual(tuple(result.obs_window_q_vecs.shape), (1, 1, 2))

    def test_query_aware_snapkv_requires_query_tokens(self) -> None:
        cache = make_position_cache([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        policy = QueryAwareSnapKV(ToyQueryAwareModel(head_dim=2), obs_window_size=2, sink_size=1, recency_window=1)
        with self.assertRaises(ValueError):
            policy.evict(cache, k_budget=2)

    def test_query_aware_snapkv_changes_selection_when_query_signal_changes(self) -> None:
        cache = make_position_cache(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.5],
                [0.0, 0.4],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            positions=[10, 20, 30, 40, 50, 60],
        )
        standard = SnapKV(obs_window_size=2, sink_size=1, recency_window=1, pooling="max").evict(cache, k_budget=3)
        query_aware = QueryAwareSnapKV(
            ToyQueryAwareModel(head_dim=2),
            obs_window_size=4,
            sink_size=1,
            recency_window=1,
            pooling="max",
        ).evict(cache, k_budget=3, obs_window=torch.tensor([[99]], dtype=torch.long))

        self.assertEqual(standard.compressed.positions, [10, 50, 60])
        self.assertEqual(query_aware.compressed.positions, [10, 20, 60])
        self.assertEqual(tuple(query_aware.obs_window_q_vecs.shape), (1, 1, 2))

    def test_log_eviction_writes_json_and_qvec_artifacts(self) -> None:
        cache = make_position_cache(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.5],
                [0.0, 0.4],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            positions=[10, 20, 30, 40, 50, 60],
        )
        result = SnapKV(obs_window_size=2, sink_size=1, recency_window=1).evict(cache, k_budget=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            entry = log_eviction(
                result,
                example_id="ex001",
                task="VT-4hop",
                task_relevant_positions=[10, 20, 60],
                log_dir=tmpdir,
                metadata={"method": "snapkv"},
            )
            json_path = Path(tmpdir) / "ex001.json"
            qvec_path = Path(tmpdir) / "ex001_qvecs.pt"

            self.assertTrue(json_path.exists())
            self.assertTrue(qvec_path.exists())
            self.assertEqual(entry["task_relevant_survived"], [True, False, True])

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["method"], "snapkv")
            self.assertEqual(payload["k_budget"], 3)
            self.assertEqual(payload["task_relevant_survived"], [True, False, True])
            self.assertEqual(tuple(torch.load(qvec_path).shape), (1, 2, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
