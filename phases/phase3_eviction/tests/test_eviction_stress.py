"""Randomized stress tests for the Phase 3 eviction policies."""

from __future__ import annotations

import random
import sys
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
from src.eviction import QueryAwareSnapKV, SnapKV, StreamingLLM

STRESS_TRIAL_COUNT = 96
QUERY_AWARE_TRIAL_COUNT = 48


def make_random_cache(
    *,
    seed: int,
    n_layers: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Build a random tuple-style cache on CPU."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    layers = []
    for _ in range(n_layers):
        key = torch.randn((1, n_heads, seq_len, head_dim), generator=generator, dtype=torch.float32)
        value = torch.randn((1, n_heads, seq_len, head_dim), generator=generator, dtype=torch.float32)
        layers.append((key, value))
    return tuple(layers)


def make_basis_cache(
    *,
    n_layers: int,
    n_heads: int,
    seq_len: int,
) -> PositionTrackedCache:
    """Create a cache whose keys are one-hot basis rows by dense position."""
    basis = torch.eye(seq_len, dtype=torch.float32).reshape(1, 1, seq_len, seq_len)
    key = basis.expand(1, n_heads, seq_len, seq_len).contiguous()
    layers = tuple((key.clone(), key.clone()) for _ in range(n_layers))
    positions = [3 + 5 * index for index in range(seq_len)]
    return PositionTrackedCache(layers, positions)


def make_monotonic_positions(rng: random.Random, seq_len: int, *, start: int = 7) -> list[int]:
    """Create strictly increasing absolute positions."""
    current = start
    positions: list[int] = []
    for _ in range(seq_len):
        positions.append(current)
        current += rng.randint(1, 5)
    return positions


def expected_structural_indices(seq_len: int, budget: int, sink_size: int, recency_window: int) -> list[int]:
    """Independent structural baseline for StreamingLLM expectations."""
    budget = min(seq_len, max(0, budget))
    sink_count = min(seq_len, budget, sink_size)
    remaining = budget - sink_count
    recency_count = min(recency_window, max(0, seq_len - sink_count), remaining)
    recency_start = max(sink_count, seq_len - recency_count)
    return sorted(set(range(sink_count)) | set(range(recency_start, seq_len)))


class RoutingToyModel(nn.Module):
    """Route query token ids to matching one-hot query rows."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = None

    def forward(
        self,
        input_ids,
        past_key_values,
        use_cache: bool = True,
        position_ids=None,
        cache_position=None,
        num_logits_to_keep=None,
        **_: object,
    ):
        del use_cache, position_ids, cache_position, num_logits_to_keep
        query_ids = input_ids if input_ids.ndim == 2 else input_ids.unsqueeze(0)
        cache = to_tuple_cache(past_key_values)
        layers = []

        for key, value in cache:
            head_dim = int(key.shape[-1])
            rows = []
            for token_id in query_ids[0].tolist():
                row = torch.zeros((head_dim,), device=key.device, dtype=key.dtype)
                row[int(token_id) % head_dim] = 1.0
                rows.append(row)

            query_block = torch.stack(rows, dim=0).reshape(1, 1, query_ids.shape[1], head_dim)
            query_block = query_block.expand(key.shape[0], key.shape[1], query_ids.shape[1], head_dim).contiguous()
            layers.append(
                (
                    torch.cat([key, query_block], dim=2),
                    torch.cat([value, query_block.clone()], dim=2),
                )
            )

        return SimpleNamespace(past_key_values=tuple(layers))


class EvictionStressTests(unittest.TestCase):
    def test_randomized_snapkv_partition_invariants(self) -> None:
        rng = random.Random(0)
        for trial in range(STRESS_TRIAL_COUNT):
            seq_len = rng.randint(8, 40)
            n_layers = rng.randint(1, 4)
            n_heads = rng.randint(1, 3)
            head_dim = rng.choice([4, 8, 16, 32])
            positions = make_monotonic_positions(rng, seq_len, start=7)
            cache = PositionTrackedCache(
                make_random_cache(
                    seed=trial,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    seq_len=seq_len,
                    head_dim=head_dim,
                ),
                positions,
            )
            budget = rng.randint(0, seq_len)
            policy = SnapKV(
                obs_window_size=rng.randint(1, min(8, seq_len)),
                sink_size=rng.randint(0, 4),
                recency_window=rng.randint(0, 12),
                pooling=rng.choice(["max", "mean"]),
            )

            with self.subTest(trial=trial, seq_len=seq_len, budget=budget):
                result = policy.evict(cache, k_budget=budget)
                self.assertEqual(len(result.compressed) + len(result.evicted), seq_len)
                self.assertEqual(len(result.compressed), budget)
                self.assertEqual(set(result.compressed.positions).union(result.evicted.positions), set(positions))
                self.assertFalse(set(result.compressed.positions).intersection(result.evicted.positions))
                self.assertEqual(set(result.importance_scores), set(positions))
                self.assertEqual(tuple(result.obs_window_q_vecs.shape[:1]), (n_layers,))
                self.assertEqual(int(result.obs_window_q_vecs.shape[2]), head_dim)
                self.assertEqual(result.evicted.device.type, "cpu")
                self.assertEqual(result.compressed.positions, sorted(result.compressed.positions))
                self.assertEqual(result.evicted.positions, sorted(result.evicted.positions))

    def test_randomized_streamingllm_matches_structural_expectation(self) -> None:
        rng = random.Random(1)
        for trial in range(STRESS_TRIAL_COUNT):
            seq_len = rng.randint(6, 32)
            n_layers = rng.randint(1, 3)
            n_heads = rng.randint(1, 2)
            head_dim = rng.choice([8, 16])
            budget = rng.randint(0, seq_len)
            sink_size = rng.randint(0, 4)
            recency_window = rng.randint(0, 10)
            positions = [2 + 3 * index for index in range(seq_len)]
            cache = PositionTrackedCache(
                make_random_cache(
                    seed=1000 + trial,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    seq_len=seq_len,
                    head_dim=head_dim,
                ),
                positions,
            )
            policy = StreamingLLM(sink_size=sink_size, recency_window=recency_window)

            with self.subTest(trial=trial, seq_len=seq_len, budget=budget):
                result = policy.evict(cache, k_budget=budget)
                expected = expected_structural_indices(seq_len, budget, sink_size, recency_window)
                expected_positions = [positions[index] for index in expected]
                self.assertEqual(result.compressed.positions, expected_positions)
                self.assertEqual(len(result.compressed) + len(result.evicted), seq_len)
                self.assertEqual(tuple(result.obs_window_q_vecs.shape), (n_layers, 1, head_dim))

    def test_query_aware_randomized_target_selection(self) -> None:
        rng = random.Random(2)
        model = RoutingToyModel()
        policy = QueryAwareSnapKV(model, obs_window_size=1, sink_size=1, recency_window=1, pooling="max")

        for trial in range(QUERY_AWARE_TRIAL_COUNT):
            seq_len = rng.randint(6, 16)
            cache = make_basis_cache(n_layers=2, n_heads=2, seq_len=seq_len)
            target_dense_index = rng.randint(1, seq_len - 2)
            result = policy.evict(
                cache,
                k_budget=3,
                obs_window=torch.tensor([[target_dense_index]], dtype=torch.long),
            )

            with self.subTest(trial=trial, seq_len=seq_len, target=target_dense_index):
                self.assertIn(cache.positions[target_dense_index], result.compressed.positions)
                self.assertEqual(len(result.compressed), 3)
                self.assertEqual(tuple(result.obs_window_q_vecs.shape), (2, 1, seq_len))
                self.assertEqual(result.evicted.device.type, "cpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
