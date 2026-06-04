from __future__ import annotations

import unittest

import torch

from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache
from phases.phase10_expansion.src.precision_promotion import (
    LowBitLayerRows,
    LowBitRowStore,
    PrecisionBudget,
    effective_kv_bytes,
    evaluate_precision_promotion_rows,
    fake_quantize_cache,
    fake_quantize_positions,
    fake_quantize_tensor,
    lowbit_row_store_bytes,
    materialize_lowbit_cache,
    mixed_precision_kv_bytes,
    promote_high_precision_rows,
    quantize_position_rows,
)


def _toy_cache(*, positions: list[int] | None = None, seq_len: int = 4) -> PositionTrackedCache:
    positions = positions if positions is not None else list(range(seq_len))
    base = torch.arange(1, 1 + 1 * 2 * seq_len * 3, dtype=torch.float32).reshape(1, 2, seq_len, 3)
    return PositionTrackedCache(((base.clone(), (base + 100).clone()),), list(positions))


class PrecisionPromotionTests(unittest.TestCase):
    def test_fake_quantize_tensor_preserves_shape_dtype_and_improves_with_more_bits(self) -> None:
        tensor = torch.tensor([[[[-1.0, -0.3, 0.25, 0.9], [0.0, 0.0, 0.0, 0.0]]]], dtype=torch.float32)

        q2 = fake_quantize_tensor(tensor, nbits=2)
        q4 = fake_quantize_tensor(tensor, nbits=4)

        self.assertEqual(q2.shape, tensor.shape)
        self.assertEqual(q4.dtype, tensor.dtype)
        self.assertTrue(torch.equal(q2[:, :, 1:2, :], torch.zeros_like(q2[:, :, 1:2, :])))
        err2 = torch.mean((q2 - tensor).abs()).item()
        err4 = torch.mean((q4 - tensor).abs()).item()
        self.assertLess(err4, err2)

    def test_fake_quantize_cache_preserves_position_metadata(self) -> None:
        cache = _toy_cache(positions=[10, 20, 30, 40])

        quantized = fake_quantize_cache(cache, nbits=4)

        self.assertIsInstance(quantized, PositionTrackedCache)
        self.assertEqual(quantized.positions, [10, 20, 30, 40])
        self.assertEqual(quantized.kv[0][0].shape, cache.kv[0][0].shape)

    def test_fake_quantize_positions_changes_only_selected_absolute_rows(self) -> None:
        cache = _toy_cache(positions=[10, 20, 30, 40])

        quantized = fake_quantize_positions(cache, quantize_positions=[20, 40], nbits=2)

        for position in [10, 30]:
            dense = cache.positions.index(position)
            self.assertTrue(torch.equal(quantized.kv[0][0][:, :, dense, :], cache.kv[0][0][:, :, dense, :]))
            self.assertTrue(torch.equal(quantized.kv[0][1][:, :, dense, :], cache.kv[0][1][:, :, dense, :]))
        for position in [20, 40]:
            dense = cache.positions.index(position)
            self.assertFalse(torch.equal(quantized.kv[0][0][:, :, dense, :], cache.kv[0][0][:, :, dense, :]))
            self.assertFalse(torch.equal(quantized.kv[0][1][:, :, dense, :], cache.kv[0][1][:, :, dense, :]))
        self.assertEqual(quantized.positions, cache.positions)

    def test_fake_quantize_positions_rejects_missing_positions(self) -> None:
        cache = _toy_cache(positions=[10, 20, 30, 40])

        with self.assertRaisesRegex(ValueError, "missing"):
            fake_quantize_positions(cache, quantize_positions=[999], nbits=2)

    def test_lowbit_row_store_materializes_like_position_fake_quant(self) -> None:
        cache = _toy_cache(positions=[10, 20, 30, 40])
        store = quantize_position_rows(cache, quantize_positions=[20, 40], nbits=2)

        materialized = materialize_lowbit_cache(cache, store)
        fake = fake_quantize_positions(cache, quantize_positions=[20, 40], nbits=2)

        self.assertEqual(store.positions, (20, 40))
        self.assertEqual(store.layers[0].key_codes.dtype, torch.int8)
        self.assertTrue(torch.equal(materialized.kv[0][0], fake.kv[0][0]))
        self.assertTrue(torch.equal(materialized.kv[0][1], fake.kv[0][1]))

    def test_lowbit_row_store_promotion_keeps_selected_rows_high_precision(self) -> None:
        cache = _toy_cache(positions=[10, 20, 30, 40])
        store = quantize_position_rows(cache, quantize_positions=[20, 40], nbits=2)

        materialized = materialize_lowbit_cache(cache, store, promoted_positions=[40])

        promoted_dense = cache.positions.index(40)
        low_dense = cache.positions.index(20)
        self.assertTrue(torch.equal(materialized.kv[0][0][:, :, promoted_dense, :], cache.kv[0][0][:, :, promoted_dense, :]))
        self.assertFalse(torch.equal(materialized.kv[0][0][:, :, low_dense, :], cache.kv[0][0][:, :, low_dense, :]))

    def test_lowbit_row_store_bytes_counts_packed_codes_and_scales(self) -> None:
        cache = _toy_cache(positions=[10, 20, 30, 40])
        store = quantize_position_rows(cache, quantize_positions=[20, 40], nbits=2)

        with_scales = lowbit_row_store_bytes(store)
        without_scales = lowbit_row_store_bytes(store, include_scales=False)

        # Two rows, one layer, K+V, batch=1, heads=2, dim=3 => 24 code values.
        self.assertEqual(without_scales, 24 * 2 / 8.0)
        self.assertGreater(with_scales, without_scales)

    def test_lowbit_row_store_bytes_counts_hqq_packed_tensors_and_metadata(self) -> None:
        layer = LowBitLayerRows(
            key_codes=torch.ones(8, dtype=torch.uint8),
            key_scales=torch.tensor([], dtype=torch.float32),
            value_codes=torch.ones(4, dtype=torch.uint8),
            value_scales=torch.tensor([], dtype=torch.float32),
            key_meta={"scale": torch.ones(2, dtype=torch.float16), "axis": 0},
            value_meta={"zero": torch.ones(3, dtype=torch.float32), "shape": (1, 1, 1, 1)},
        )
        store = LowBitRowStore(
            layers=(layer,),
            positions=(10,),
            nbits=4,
            source_dtype=torch.float16,
            backend="hqq",
            group_size=64,
        )

        without_scales = lowbit_row_store_bytes(store, include_scales=False)
        with_scales = lowbit_row_store_bytes(store, include_scales=True)

        self.assertEqual(without_scales, 12.0)
        self.assertEqual(with_scales, 12.0 + 2 * 2 + 3 * 4)

    def test_promote_high_precision_rows_replaces_only_selected_positions(self) -> None:
        high = _toy_cache(positions=[10, 20, 30, 40])
        low = fake_quantize_cache(high, nbits=2)
        assert isinstance(low, PositionTrackedCache)

        promoted = promote_high_precision_rows(low, high, [30, 10])

        for position in [10, 30]:
            dense = promoted.positions.index(position)
            source = high.positions.index(position)
            self.assertTrue(torch.equal(promoted.kv[0][0][:, :, dense, :], high.kv[0][0][:, :, source, :]))
            self.assertTrue(torch.equal(promoted.kv[0][1][:, :, dense, :], high.kv[0][1][:, :, source, :]))
        for position in [20, 40]:
            dense = promoted.positions.index(position)
            self.assertTrue(torch.equal(promoted.kv[0][0][:, :, dense, :], low.kv[0][0][:, :, dense, :]))
            self.assertTrue(torch.equal(promoted.kv[0][1][:, :, dense, :], low.kv[0][1][:, :, dense, :]))

    def test_promote_high_precision_rows_rejects_missing_positions(self) -> None:
        high = _toy_cache(positions=[10, 20, 30, 40])
        low = fake_quantize_cache(high, nbits=2)
        assert isinstance(low, PositionTrackedCache)

        with self.assertRaisesRegex(ValueError, "missing"):
            promote_high_precision_rows(low, high, [999])

    def test_effective_kv_bytes_counts_low_bits_promoted_rows_and_side_buffer(self) -> None:
        cache = _toy_cache(seq_len=4)
        # One layer, K+V, batch=1, heads=2, seq=4, dim=3 => 48 scalars.
        # One promoted row has K+V, batch=1, heads=2, dim=3 => 12 scalars.
        budget = PrecisionBudget(low_precision_bits=2, high_precision_bits=16)

        active_bytes = effective_kv_bytes(cache, budget=budget, promoted_positions=[1])
        with_side = effective_kv_bytes(
            cache,
            budget=budget,
            promoted_positions=[1],
            include_high_precision_side_buffer=True,
        )

        expected_active_bits = 48 * 2 + 12 * (16 - 2)
        self.assertEqual(active_bytes, expected_active_bits / 8.0)
        self.assertEqual(with_side, active_bytes + (48 * 16 / 8.0))

    def test_mixed_precision_kv_bytes_counts_only_low_precision_rows(self) -> None:
        cache = _toy_cache(seq_len=4)
        budget = PrecisionBudget(low_precision_bits=2, high_precision_bits=16)

        active_bytes = mixed_precision_kv_bytes(
            cache,
            budget=budget,
            low_precision_positions=[1, 2],
            promoted_positions=[2],
        )
        with_side = mixed_precision_kv_bytes(
            cache,
            budget=budget,
            low_precision_positions=[1, 2],
            promoted_positions=[2],
            include_high_precision_side_buffer=True,
        )

        # One row has K+V, batch=1, heads=2, dim=3 => 12 scalars.
        # Active rows: positions 0,2,3 at 16-bit and position 1 at 2-bit.
        expected_active_bits = (3 * 12 * 16) + (1 * 12 * 2)
        self.assertEqual(active_bytes, expected_active_bits / 8.0)
        self.assertEqual(with_side, active_bytes + (2 * 12 * 16 / 8.0))

    def test_mixed_precision_kv_bytes_rejects_missing_rows(self) -> None:
        cache = _toy_cache(positions=[10, 20, 30, 40])
        budget = PrecisionBudget(low_precision_bits=2)

        with self.assertRaisesRegex(ValueError, "missing"):
            mixed_precision_kv_bytes(cache, budget=budget, low_precision_positions=[999])

    def test_precision_budget_rejects_invalid_precision_ordering(self) -> None:
        with self.assertRaisesRegex(ValueError, "larger"):
            PrecisionBudget(low_precision_bits=4, high_precision_bits=4)

    def test_evaluate_precision_promotion_rows_marks_fake_quant_as_appendix_only(self) -> None:
        rows = [
            {
                "nbits": 2,
                "k": 96,
                "full_fp16": 1.0,
                "lowbit_all": 0.55,
                "static_mixed": 0.58,
                "random_precision": 0.60,
                "oldest_precision": 0.57,
                "idlekv_precision": 0.80,
                "gold_precision": 0.95,
                "active_bytes": 1024,
                "side_buffer_bytes": 8192,
                "real_quantized_cache": False,
            }
        ]

        recommendation = evaluate_precision_promotion_rows(rows)[0]

        self.assertTrue(recommendation["appendix_ok"])
        self.assertFalse(recommendation["main_ok"])
        self.assertEqual(recommendation["action"], "appendix_quality_only")
        self.assertEqual(recommendation["lowbit_drop"], 0.45)

    def test_evaluate_precision_promotion_rows_rejects_when_lowbit_does_not_degrade(self) -> None:
        rows = [
            {
                "nbits": 4,
                "k": 96,
                "full_fp16": 1.0,
                "lowbit_all": 0.92,
                "static_mixed": 0.92,
                "random_precision": 0.92,
                "idlekv_precision": 1.0,
                "gold_precision": 1.0,
                "active_bytes": 1024,
            }
        ]

        recommendation = evaluate_precision_promotion_rows(rows)[0]

        self.assertFalse(recommendation["appendix_ok"])
        self.assertEqual(recommendation["action"], "do_not_promote_lowbit_not_degraded")

    def test_evaluate_precision_promotion_rows_rejects_when_control_matches_idlekv(self) -> None:
        rows = [
            {
                "nbits": 2,
                "k": 48,
                "full_fp16": 1.0,
                "lowbit_all": 0.40,
                "static_mixed": 0.78,
                "random_precision": 0.55,
                "idlekv_precision": 0.80,
                "gold_precision": 1.0,
                "active_bytes": 1024,
            }
        ]

        recommendation = evaluate_precision_promotion_rows(rows)[0]

        self.assertFalse(recommendation["appendix_ok"])
        self.assertEqual(recommendation["action"], "do_not_promote_controls_match_or_beat_idlekv")


if __name__ == "__main__":
    unittest.main()
