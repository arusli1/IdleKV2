"""Synthetic stress tests for the Phase 2 KV utilities."""

from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

import torch

PHASE_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_ROOT))

from src.kv_utils import PositionTrackedCache, inject_kv, load_kv, save_kv, sequence_length, slice_kv, to_dynamic_cache, to_tuple_cache

STRESS_TRIAL_COUNT = 64


def make_random_cache(
    *,
    seed: int,
    n_layers: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Create one synthetic tuple-style KV cache on CPU."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    layers = []
    for _ in range(n_layers):
        key = torch.randn((1, n_heads, seq_len, head_dim), generator=generator, dtype=dtype)
        value = torch.randn((1, n_heads, seq_len, head_dim), generator=generator, dtype=dtype)
        layers.append((key, value))
    return tuple(layers)


def assert_cache_close(testcase: unittest.TestCase, left, right) -> None:
    """Assert exact tensor equality layer by layer."""
    left_tuple = to_tuple_cache(left)
    right_tuple = to_tuple_cache(right)
    testcase.assertEqual(len(left_tuple), len(right_tuple))
    for layer_idx, ((left_key, left_value), (right_key, right_value)) in enumerate(zip(left_tuple, right_tuple)):
        testcase.assertTrue(torch.equal(left_key, right_key), msg=f"Layer {layer_idx} key mismatch.")
        testcase.assertTrue(torch.equal(left_value, right_value), msg=f"Layer {layer_idx} value mismatch.")


class KVUtilsStressTests(unittest.TestCase):
    def test_save_and_load_preserve_plain_and_tracked_caches(self) -> None:
        cache = make_random_cache(seed=7, n_layers=3, n_heads=2, seq_len=11, head_dim=8)
        tracked = PositionTrackedCache(cache, list(range(sequence_length(cache))))

        with tempfile.TemporaryDirectory() as tmpdir:
            save_kv(cache, f"{tmpdir}/plain")
            restored_plain = load_kv(f"{tmpdir}/plain", device="cpu")
            assert_cache_close(self, cache, restored_plain)

            save_kv(tracked, f"{tmpdir}/tracked")
            restored_tracked = load_kv(f"{tmpdir}/tracked", device="cpu")
            self.assertIsInstance(restored_tracked, PositionTrackedCache)
            assert_cache_close(self, tracked, restored_tracked)
            self.assertEqual(restored_tracked.positions, tracked.positions)

    def test_to_dynamic_cache_round_trip_preserves_tensor_values(self) -> None:
        cache = make_random_cache(seed=11, n_layers=2, n_heads=3, seq_len=9, head_dim=16)
        dynamic = to_dynamic_cache(cache)
        restored = to_tuple_cache(dynamic)
        assert_cache_close(self, cache, restored)

    def test_randomized_slice_and_inject_recover_original_cache(self) -> None:
        rng = random.Random(0)
        for trial in range(STRESS_TRIAL_COUNT):
            seq_len = rng.randint(8, 28)
            n_layers = rng.randint(1, 4)
            n_heads = rng.randint(1, 3)
            head_dim = rng.choice([8, 16, 32])
            cache = make_random_cache(
                seed=trial,
                n_layers=n_layers,
                n_heads=n_heads,
                seq_len=seq_len,
                head_dim=head_dim,
            )
            tracked = PositionTrackedCache(cache, list(range(seq_len)))
            removed_count = rng.randint(1, max(1, seq_len // 3))
            removed_positions = sorted(rng.sample(range(seq_len), removed_count))
            kept_positions = [position for position in range(seq_len) if position not in removed_positions]

            with self.subTest(trial=trial, seq_len=seq_len, removed=removed_count):
                active = slice_kv(tracked, kept_positions)
                fragment = slice_kv(tracked, removed_positions)
                self.assertIsInstance(active, PositionTrackedCache)
                self.assertIsInstance(fragment, PositionTrackedCache)
                restored = inject_kv(active, fragment, removed_positions)
                self.assertEqual(restored.positions, list(range(seq_len)))
                assert_cache_close(self, restored, cache)

    def test_slice_on_tracked_cache_preserves_original_absolute_positions(self) -> None:
        cache = make_random_cache(seed=19, n_layers=1, n_heads=1, seq_len=6, head_dim=8)
        tracked = PositionTrackedCache(cache, [0, 4, 7, 12, 15, 21])
        sliced = slice_kv(tracked, [5, 1, 3])
        self.assertIsInstance(sliced, PositionTrackedCache)
        self.assertEqual(sliced.positions, [4, 12, 21])


if __name__ == "__main__":
    unittest.main(verbosity=2)
