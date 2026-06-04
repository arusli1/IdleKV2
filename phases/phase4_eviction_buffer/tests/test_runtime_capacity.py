"""Unit tests for synthetic runtime-capacity profiling helpers."""

from __future__ import annotations

import unittest

import torch

from phases.phase4_eviction_buffer.src.buffer.runtime_capacity import (
    KVRuntimeSpec,
    feasibility_rows,
    interleaved_active_positions,
    interleaved_restore_positions,
    make_synthetic_cache,
    parse_dtype,
    percentile_summary,
    profile_chunked_selection_capacity,
    profile_chunked_selection_capacity_multi_k,
    profile_end_to_end_repair_capacity,
    profile_end_to_end_repair_capacity_multi_k,
)


class RuntimeCapacityTests(unittest.TestCase):
    def test_spec_bytes_per_token_matches_qwen_shape(self) -> None:
        spec = KVRuntimeSpec(n_layers=28, n_kv_heads=4, head_dim=128, dtype=torch.bfloat16)
        self.assertEqual(spec.bytes_per_token, 57_344)

    def test_parse_dtype_accepts_common_names(self) -> None:
        self.assertIs(parse_dtype("bf16"), torch.bfloat16)
        self.assertIs(parse_dtype("float16"), torch.float16)
        self.assertIs(parse_dtype("torch.float32"), torch.float32)

    def test_interleaved_positions_do_not_overlap(self) -> None:
        active = set(interleaved_active_positions(4))
        restore = set(interleaved_restore_positions(4))
        self.assertEqual(active, {0, 2, 4, 6})
        self.assertEqual(restore, {1, 3, 5, 7})
        self.assertFalse(active & restore)

    def test_make_synthetic_cache_preserves_shape_and_positions(self) -> None:
        spec = KVRuntimeSpec(n_layers=2, n_kv_heads=1, head_dim=4, dtype=torch.float16)
        cache = make_synthetic_cache(
            seq_len=3,
            positions=[0, 2, 4],
            spec=spec,
            device="cpu",
        )
        self.assertEqual(cache.positions, [0, 2, 4])
        self.assertEqual(len(cache.kv), 2)
        self.assertEqual(tuple(cache.kv[0][0].shape), (1, 1, 3, 4))

    def test_percentile_summary_reports_named_percentiles(self) -> None:
        summary = percentile_summary([1.0, 2.0, 3.0])
        self.assertEqual(summary["p50_ms"], 2.0)
        self.assertGreater(summary["p95_ms"], summary["p50_ms"])

    def test_feasibility_rows_reports_largest_measured_k(self) -> None:
        rows = [
            {"active_tokens": 8, "k": 96, "p95_total_ms": 30.0},
            {"active_tokens": 8, "k": 500, "p95_total_ms": 80.0},
            {"active_tokens": 8, "k": 1000, "p95_total_ms": 120.0},
        ]
        fit = feasibility_rows(rows, idle_windows_s=(0.1, 0.2), budget_fraction=0.9)
        self.assertEqual(fit[0]["max_measured_k"], 500)
        self.assertEqual(fit[1]["max_measured_k"], 1000)

    def test_feasibility_rows_can_group_by_candidate_store(self) -> None:
        rows = [
            {"active_tokens": 8, "candidate_tokens": 12, "k": 3, "p95_total_ms": 30.0},
            {"active_tokens": 8, "candidate_tokens": 12, "k": 6, "p95_total_ms": 120.0},
            {"active_tokens": 8, "candidate_tokens": 24, "k": 3, "p95_total_ms": 120.0},
        ]
        fit = feasibility_rows(
            rows,
            idle_windows_s=(0.1,),
            budget_fraction=1.0,
            group_fields=("active_tokens", "candidate_tokens"),
        )
        by_candidates = {row["candidate_tokens"]: row["max_measured_k"] for row in fit}
        self.assertEqual(by_candidates, {12: 3, 24: 0})

    def test_chunked_selection_capacity_cpu_smoke(self) -> None:
        spec = KVRuntimeSpec(n_layers=2, n_query_heads=2, n_kv_heads=1, head_dim=4, dtype=torch.float16)
        row = profile_chunked_selection_capacity(
            candidate_tokens=12,
            k_tokens=3,
            spec=spec,
            query_len=2,
            chunk_tokens=5,
            source_pool_chunks=3,
            device="cpu",
            trials=1,
            warmup_trials=0,
            pin_memory=False,
        )
        self.assertEqual(row["candidate_tokens"], 12)
        self.assertEqual(row["k"], 3)
        self.assertEqual(row["candidate_chunks"], 3)
        self.assertEqual(row["source_pool_chunks"], 3)
        self.assertEqual(row["host_pool_tokens"], 12)
        self.assertEqual(row["host_pool_coverage"], 1.0)
        self.assertEqual(row["offloaded_kv_bytes"], spec.bytes_per_token * 12)
        self.assertGreater(float(row["p50_total_ms"]), 0.0)

    def test_chunked_selection_clamps_source_pool_to_chunk_count(self) -> None:
        spec = KVRuntimeSpec(n_layers=1, n_query_heads=1, n_kv_heads=1, head_dim=4, dtype=torch.float16)
        row = profile_chunked_selection_capacity(
            candidate_tokens=12,
            k_tokens=3,
            spec=spec,
            query_len=2,
            chunk_tokens=5,
            source_pool_chunks=99,
            device="cpu",
            trials=1,
            warmup_trials=0,
            pin_memory=False,
        )
        self.assertEqual(row["source_pool_chunks"], 3)

    def test_chunked_selection_multi_k_reports_one_row_per_k(self) -> None:
        spec = KVRuntimeSpec(n_layers=1, n_query_heads=1, n_kv_heads=1, head_dim=4, dtype=torch.float16)
        rows = profile_chunked_selection_capacity_multi_k(
            candidate_tokens=12,
            k_tokens_values=(3, 6),
            spec=spec,
            query_len=2,
            chunk_tokens=5,
            source_pool_chunks=3,
            device="cpu",
            trials=1,
            warmup_trials=0,
            pin_memory=False,
        )
        self.assertEqual([row["k"] for row in rows], [3, 6])
        for row in rows:
            self.assertEqual(row["candidate_tokens"], 12)
            self.assertEqual(row["candidate_chunks"], 3)
            self.assertEqual(row["source_pool_chunks"], 3)
            self.assertEqual(row["host_pool_coverage"], 1.0)
            self.assertIn("p95_scan_ms", row)
            self.assertIn("p95_topk_ms", row)
            self.assertGreater(float(row["p50_total_ms"]), 0.0)

    def test_chunked_selection_reports_partial_source_pool_coverage(self) -> None:
        spec = KVRuntimeSpec(n_layers=1, n_query_heads=1, n_kv_heads=1, head_dim=4, dtype=torch.float16)
        row = profile_chunked_selection_capacity(
            candidate_tokens=12,
            k_tokens=3,
            spec=spec,
            query_len=2,
            chunk_tokens=5,
            source_pool_chunks=1,
            device="cpu",
            trials=1,
            warmup_trials=0,
            pin_memory=False,
        )
        self.assertEqual(row["candidate_chunks"], 3)
        self.assertEqual(row["source_pool_chunks"], 1)
        self.assertEqual(row["host_pool_tokens"], 5)
        self.assertLess(float(row["host_pool_coverage"]), 0.5)

    def test_end_to_end_repair_capacity_cpu_smoke(self) -> None:
        spec = KVRuntimeSpec(n_layers=1, n_query_heads=1, n_kv_heads=1, head_dim=4, dtype=torch.float16)
        row = profile_end_to_end_repair_capacity(
            active_tokens=4,
            candidate_tokens=12,
            k_tokens=3,
            spec=spec,
            query_len=2,
            chunk_tokens=5,
            source_pool_chunks=2,
            device="cpu",
            trials=1,
            warmup_trials=0,
            pin_memory=False,
        )
        self.assertEqual(row["active_tokens"], 4)
        self.assertEqual(row["candidate_tokens"], 12)
        self.assertEqual(row["k"], 3)
        self.assertGreater(float(row["p50_select_ms"]), 0.0)
        self.assertGreater(float(row["p50_move_inject_ms"]), 0.0)
        self.assertGreater(float(row["p50_total_ms"]), 0.0)

    def test_end_to_end_repair_capacity_multi_k_cpu_smoke(self) -> None:
        spec = KVRuntimeSpec(n_layers=1, n_query_heads=1, n_kv_heads=1, head_dim=4, dtype=torch.float16)
        rows = profile_end_to_end_repair_capacity_multi_k(
            active_tokens=4,
            candidate_tokens=12,
            k_tokens_values=(3, 6),
            spec=spec,
            query_len=2,
            chunk_tokens=5,
            source_pool_chunks=3,
            device="cpu",
            trials=1,
            warmup_trials=0,
            pin_memory=False,
        )
        self.assertEqual([row["k"] for row in rows], [3, 6])
        for row in rows:
            self.assertEqual(row["active_tokens"], 4)
            self.assertEqual(row["candidate_tokens"], 12)
            self.assertIn("p95_scan_ms", row)
            self.assertIn("p95_topk_ms", row)
            self.assertIn("p95_move_inject_ms", row)
            self.assertGreater(float(row["p50_total_ms"]), 0.0)


if __name__ == "__main__":
    unittest.main()
