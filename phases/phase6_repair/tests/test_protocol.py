"""Unit tests for the Phase 6 protocol helpers."""

from __future__ import annotations

import unittest

import torch

from phases.phase1_degradation.phase1.models import PrefillSegment, RelevantSpan, TaskExample
from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache
from phases.phase6_repair.src.protocol import (
    ALL_SPLIT_SPECS,
    CLEAN_SPLIT_SPECS,
    DEFAULT_SPLIT_SPEC,
    MQ_NIAH_2Q_CLEAN_SPLIT_SPEC,
    MQ_NIAH_3Q_CLEAN_SPLIT_SPEC,
    MQ_NIAH_6Q_CLEAN_SPLIT_SPEC,
    MQ_NIAH_6Q_CLEAN_SPLIT_SPECS,
    MQ_NIAH_8Q_CLEAN_SPLIT_SPECS,
    TAIL_LEAKY_SPLIT_SPECS,
    _apply_rotary_pos_emb_for_model,
    build_turn_n_keep_plan,
    materialize_context_partition,
    split_example_for_turn,
)


def _make_cache(seq_len: int) -> PositionTrackedCache:
    layer = (
        torch.zeros((1, 1, seq_len, 2), dtype=torch.float16),
        torch.zeros((1, 1, seq_len, 2), dtype=torch.float16),
    )
    return PositionTrackedCache((layer,), list(range(seq_len)))


def _make_scalar_key_cache(values: list[float]) -> PositionTrackedCache:
    key = torch.tensor(values, dtype=torch.float16).reshape(1, 1, len(values), 1)
    value = torch.zeros_like(key)
    return PositionTrackedCache(((key, value),), list(range(len(values))))


class _FakeModel:
    def __init__(self, model_type: str) -> None:
        self.config = type("Config", (), {"model_type": model_type})()


class Phase6ProtocolTests(unittest.TestCase):
    def test_split_registry_covers_all_balanced_two_two_partitions(self) -> None:
        self.assertEqual(len(ALL_SPLIT_SPECS), 6)
        self.assertEqual(len(CLEAN_SPLIT_SPECS), 3)
        self.assertEqual(len(TAIL_LEAKY_SPLIT_SPECS), 3)
        self.assertEqual(len({spec.name for spec in ALL_SPLIT_SPECS}), 6)
        for spec in ALL_SPLIT_SPECS:
            self.assertEqual(set(spec.q1_indices) | set(spec.q2_indices), {0, 1, 2, 3})
            self.assertEqual(set(spec.q1_indices) & set(spec.q2_indices), set())
            self.assertEqual(len(spec.q1_indices), 2)
            self.assertEqual(len(spec.q2_indices), 2)

    def test_split_example_uses_balanced_indices_and_values_only_format(self) -> None:
        example = TaskExample(
            index=0,
            task_name="mq_niah_4q",
            task_family="niah",
            context="ctx",
            question="orig",
            answer_prefix="orig",
            outputs=["11", "22", "33", "44"],
            max_new_tokens=64,
            target_context_length=128,
            relevant_spans=[
                RelevantSpan(name=f"needle_{i}", kind="needle", char_start=i, char_end=i + 1, depth_fraction=0.1 * i)
                for i in range(1, 5)
            ],
            prefill_segments=[PrefillSegment(name="context", char_start=0, char_end=3)],
            metadata={"query_keys": ["k1", "k2", "k3", "k4"]},
        )

        q1 = split_example_for_turn(example, query_indices=DEFAULT_SPLIT_SPEC.q1_indices, split_name="q1")
        q2 = split_example_for_turn(example, query_indices=DEFAULT_SPLIT_SPEC.q2_indices, split_name="q2")

        self.assertEqual(q1.outputs, ["11", "44"])
        self.assertEqual(q2.outputs, ["22", "33"])
        self.assertEqual(q1.answer_prefix, " Answer:")
        self.assertIn("comma-separated", q1.question)
        self.assertEqual(q1.metadata["query_keys"], ["k1", "k4"])
        self.assertEqual(q2.metadata["query_keys"], ["k2", "k3"])

    def test_mq_niah_2q_clean_split_uses_tail_first_turn_and_one_value_q2(self) -> None:
        example = TaskExample(
            index=0,
            task_name="mq_niah_2q",
            task_family="niah",
            context="ctx",
            question="orig",
            answer_prefix="orig",
            outputs=["11", "22"],
            max_new_tokens=32,
            target_context_length=128,
            relevant_spans=[
                RelevantSpan(name=f"needle_{i}", kind="needle", char_start=i, char_end=i + 1, depth_fraction=0.1 * i)
                for i in range(1, 3)
            ],
            prefill_segments=[PrefillSegment(name="context", char_start=0, char_end=3)],
            metadata={"query_keys": ["k1", "k2"]},
        )

        q1 = split_example_for_turn(example, query_indices=MQ_NIAH_2Q_CLEAN_SPLIT_SPEC.q1_indices, split_name="q1")
        q2 = split_example_for_turn(example, query_indices=MQ_NIAH_2Q_CLEAN_SPLIT_SPEC.q2_indices, split_name="q2")

        self.assertEqual(q1.outputs, ["22"])
        self.assertEqual(q2.outputs, ["11"])
        self.assertEqual(q1.metadata["query_keys"], ["k2"])
        self.assertEqual(q2.metadata["query_keys"], ["k1"])

    def test_mq_niah_3q_clean_split_uses_tail_first_turn_and_two_value_q2(self) -> None:
        example = TaskExample(
            index=0,
            task_name="mq_niah_3q",
            task_family="niah",
            context="ctx",
            question="orig",
            answer_prefix="orig",
            outputs=["11", "22", "33"],
            max_new_tokens=64,
            target_context_length=128,
            relevant_spans=[
                RelevantSpan(name=f"needle_{i}", kind="needle", char_start=i, char_end=i + 1, depth_fraction=0.1 * i)
                for i in range(1, 4)
            ],
            prefill_segments=[PrefillSegment(name="context", char_start=0, char_end=3)],
            metadata={"query_keys": ["k1", "k2", "k3"]},
        )

        q1 = split_example_for_turn(example, query_indices=MQ_NIAH_3Q_CLEAN_SPLIT_SPEC.q1_indices, split_name="q1")
        q2 = split_example_for_turn(example, query_indices=MQ_NIAH_3Q_CLEAN_SPLIT_SPEC.q2_indices, split_name="q2")

        self.assertEqual(q1.outputs, ["33"])
        self.assertEqual(q2.outputs, ["11", "22"])
        self.assertEqual(q1.metadata["query_keys"], ["k3"])
        self.assertEqual(q2.metadata["query_keys"], ["k1", "k2"])
        self.assertIn("comma-separated", q2.question)

    def test_mq_niah_6q_clean_split_uses_late_three_needles_in_turn1(self) -> None:
        example = TaskExample(
            index=0,
            task_name="mq_niah_6q",
            task_family="niah",
            context="ctx",
            question="orig",
            answer_prefix="orig",
            outputs=["11", "22", "33", "44", "55", "66"],
            max_new_tokens=96,
            target_context_length=128,
            relevant_spans=[
                RelevantSpan(name=f"needle_{i}", kind="needle", char_start=i, char_end=i + 1, depth_fraction=0.1 * i)
                for i in range(1, 7)
            ],
            prefill_segments=[PrefillSegment(name="context", char_start=0, char_end=3)],
            metadata={"query_keys": ["k1", "k2", "k3", "k4", "k5", "k6"]},
        )

        q1 = split_example_for_turn(
            example,
            query_indices=MQ_NIAH_6Q_CLEAN_SPLIT_SPEC.q1_indices,
            split_name="q1",
            max_new_tokens=MQ_NIAH_6Q_CLEAN_SPLIT_SPEC.max_new_tokens,
        )
        q2 = split_example_for_turn(
            example,
            query_indices=MQ_NIAH_6Q_CLEAN_SPLIT_SPEC.q2_indices,
            split_name="q2",
            max_new_tokens=MQ_NIAH_6Q_CLEAN_SPLIT_SPEC.max_new_tokens,
        )

        self.assertEqual(q1.outputs, ["44", "55", "66"])
        self.assertEqual(q2.outputs, ["11", "22", "33"])
        self.assertEqual(q1.metadata["query_keys"], ["k4", "k5", "k6"])
        self.assertEqual(q2.metadata["query_keys"], ["k1", "k2", "k3"])
        self.assertEqual(q1.max_new_tokens, 48)

    def test_mq_niah_6q_clean_suite_keeps_last_two_needles_in_turn1(self) -> None:
        self.assertEqual(len(MQ_NIAH_6Q_CLEAN_SPLIT_SPECS), 4)
        for spec in MQ_NIAH_6Q_CLEAN_SPLIT_SPECS:
            self.assertEqual(set(spec.q1_indices) | set(spec.q2_indices), {0, 1, 2, 3, 4, 5})
            self.assertEqual(set(spec.q1_indices) & set(spec.q2_indices), set())
            self.assertIn(4, spec.q1_indices)
            self.assertIn(5, spec.q1_indices)
            self.assertNotIn(4, spec.q2_indices)
            self.assertNotIn(5, spec.q2_indices)
            self.assertEqual(len(spec.q1_indices), 3)
            self.assertEqual(len(spec.q2_indices), 3)

    def test_mq_niah_8q_clean_suite_keeps_late_needles_in_turn1(self) -> None:
        self.assertEqual(len(MQ_NIAH_8Q_CLEAN_SPLIT_SPECS), 5)
        for spec in MQ_NIAH_8Q_CLEAN_SPLIT_SPECS:
            self.assertEqual(set(spec.q1_indices) | set(spec.q2_indices), set(range(8)))
            self.assertEqual(set(spec.q1_indices) & set(spec.q2_indices), set())
            self.assertIn(6, spec.q1_indices)
            self.assertIn(7, spec.q1_indices)
            self.assertNotIn(6, spec.q2_indices)
            self.assertNotIn(7, spec.q2_indices)
            self.assertEqual(len(spec.q1_indices), 4)
            self.assertEqual(len(spec.q2_indices), 4)
            self.assertEqual(spec.max_new_tokens, 64)

    def test_context_partition_keeps_context_budget_and_full_tail(self) -> None:
        cache = _make_cache(seq_len=7)
        q1_answer_ids = torch.tensor([1, 2], dtype=torch.long)
        keep_plan = build_turn_n_keep_plan(
            post_q1_cache=cache,
            q1_answer_ids=q1_answer_ids,
            context_len=5,
            sink_size=1,
            recency_window=1,
            pooling="max",
        )
        # Make the keep order deterministic for the unit test rather than relying on all-zero scores.
        keep_plan = keep_plan.__class__(
            context_len=keep_plan.context_len,
            tail_positions=keep_plan.tail_positions,
            mandatory_context_positions=keep_plan.mandatory_context_positions,
            ranked_candidate_positions=(2, 3, 1),
            importance_scores={0: 0.0, 1: 0.1, 2: 0.9, 3: 0.5, 4: 0.0},
        )

        partition = materialize_context_partition(
            full_post_q1_cache=cache,
            keep_plan=keep_plan,
            context_budget=3,
        )

        self.assertEqual(partition.kept_context_positions, (0, 2, 4))
        self.assertEqual(partition.evicted_context_positions, (1, 3))
        self.assertEqual(partition.compressed.positions, [0, 2, 4, 5, 6])
        self.assertEqual(partition.evicted.positions, [1, 3])

    def test_streaming_llm_keep_plan_uses_sink_then_recency_order(self) -> None:
        cache = _make_cache(seq_len=8)
        keep_plan = build_turn_n_keep_plan(
            post_q1_cache=cache,
            q1_answer_ids=torch.empty((0,), dtype=torch.long),
            context_len=6,
            sink_size=2,
            recency_window=1,
            pooling="max",
            initial_compressor="streaming_llm",
        )

        self.assertEqual(keep_plan.mandatory_context_positions, (0, 1))
        self.assertEqual(keep_plan.ranked_candidate_positions, (5, 4, 3, 2))
        self.assertEqual(keep_plan.tail_positions, (6, 7))

        partition = materialize_context_partition(
            full_post_q1_cache=cache,
            keep_plan=keep_plan,
            context_budget=4,
        )
        self.assertEqual(partition.kept_context_positions, (0, 1, 4, 5))
        self.assertEqual(partition.evicted_context_positions, (2, 3))
        self.assertEqual(partition.compressed.positions, [0, 1, 4, 5, 6, 7])

    def test_h2o_keep_plan_uses_accumulated_attention_order(self) -> None:
        cache = _make_scalar_key_cache([0.0, 1.0, 4.0, 2.0, 3.0, 5.0, 0.0, 4.0])
        keep_plan = build_turn_n_keep_plan(
            post_q1_cache=cache,
            q1_answer_ids=torch.tensor([7], dtype=torch.long),
            context_len=6,
            sink_size=1,
            recency_window=1,
            pooling="max",
            initial_compressor="h2o",
        )

        self.assertEqual(keep_plan.mandatory_context_positions, (0, 5))
        self.assertEqual(keep_plan.ranked_candidate_positions, (2, 4, 3, 1))

        partition = materialize_context_partition(
            full_post_q1_cache=cache,
            keep_plan=keep_plan,
            context_budget=4,
        )
        self.assertEqual(partition.kept_context_positions, (0, 2, 4, 5))

    def test_scissorhands_keep_plan_uses_persistent_attention_order(self) -> None:
        cache = _make_scalar_key_cache([0.0, 1.0, 4.0, 2.0, 3.0, 5.0, 0.0, 4.0])
        keep_plan = build_turn_n_keep_plan(
            post_q1_cache=cache,
            q1_answer_ids=torch.tensor([7], dtype=torch.long),
            context_len=6,
            sink_size=1,
            recency_window=1,
            pooling="max",
            initial_compressor="scissorhands",
        )

        self.assertEqual(keep_plan.mandatory_context_positions, (0, 5))
        self.assertEqual(keep_plan.ranked_candidate_positions, (2, 4, 3, 1))

        partition = materialize_context_partition(
            full_post_q1_cache=cache,
            keep_plan=keep_plan,
            context_budget=4,
        )
        self.assertEqual(partition.kept_context_positions, (0, 2, 4, 5))

    def test_rotary_helper_supports_qwen_llama_and_mistral_model_families(self) -> None:
        query = torch.ones((1, 2, 3, 4), dtype=torch.float32)
        key = torch.ones((1, 2, 3, 4), dtype=torch.float32)
        cos = torch.ones((1, 3, 4), dtype=torch.float32)
        sin = torch.zeros((1, 3, 4), dtype=torch.float32)

        for model_type in ("qwen2", "llama", "mistral"):
            rotated_query, rotated_key = _apply_rotary_pos_emb_for_model(
                _FakeModel(model_type),
                query,
                key,
                cos,
                sin,
            )

            self.assertEqual(rotated_query.shape, query.shape)
            self.assertEqual(rotated_key.shape, key.shape)
            torch.testing.assert_close(rotated_query, query)
            torch.testing.assert_close(rotated_key, key)


if __name__ == "__main__":
    unittest.main()
