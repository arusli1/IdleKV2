"""Protocol helpers for the Phase 6 two-turn matched-footprint experiment."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

import torch

from phases.phase1_degradation.phase1.helpers import join_natural
from phases.phase1_degradation.phase1.inference import PreparedExample, prepare_example_for_model
from phases.phase1_degradation.phase1.models import TaskExample
from phases.phase1_degradation.phase1.task_registry import build_task_example
from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache, slice_kv, to_dynamic_cache, to_tuple_cache
from phases.phase2_kv_cache.src.runtime import model_device, resume_forward
from phases.phase3_eviction.src.eviction import SnapKV

VALUES_ONLY_SUFFIX = "Respond with the values only, comma-separated."
VALUES_ONLY_PREFIX = " Answer:"
VALUES_ONLY_MAX_NEW_TOKENS = 24
MISMATCHED_KEY_STEM = "phantom_key"


def _apply_rotary_pos_emb_for_model(
    model,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the RoPE helper that matches the checkpoint family."""
    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    if model_type == "llama":
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    elif model_type == "mistral":
        from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
    else:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    return apply_rotary_pos_emb(query, key, cos, sin)


@dataclass(frozen=True)
class SplitTaskSpec:
    """One fixed Q1/Q2 split over an existing task family."""

    name: str
    base_task_key: str
    q1_indices: tuple[int, ...]
    q2_indices: tuple[int, ...]
    question_suffix: str = VALUES_ONLY_SUFFIX
    answer_prefix: str = VALUES_ONLY_PREFIX
    max_new_tokens: int = VALUES_ONLY_MAX_NEW_TOKENS


DEFAULT_SPLIT_SPEC = SplitTaskSpec(
    name="mq_niah_4q_split_14_to_23",
    base_task_key="mq_niah_4q",
    q1_indices=(0, 3),
    q2_indices=(1, 2),
)

MQ_NIAH_2Q_CLEAN_SPLIT_SPEC = SplitTaskSpec(
    name="mq_niah_2q_split_2_to_1",
    base_task_key="mq_niah_2q",
    q1_indices=(1,),
    q2_indices=(0,),
)

MQ_NIAH_3Q_CLEAN_SPLIT_SPEC = SplitTaskSpec(
    name="mq_niah_3q_split_3_to_12",
    base_task_key="mq_niah_3q",
    q1_indices=(2,),
    q2_indices=(0, 1),
)

MQ_NIAH_6Q_CLEAN_SPLIT_SPEC = SplitTaskSpec(
    name="mq_niah_6q_split_456_to_123",
    base_task_key="mq_niah_6q",
    q1_indices=(3, 4, 5),
    q2_indices=(0, 1, 2),
    max_new_tokens=48,
)

MQ_NIAH_6Q_CLEAN_SPLIT_SPECS = (
    SplitTaskSpec(
        name="mq_niah_6q_split_156_to_234",
        base_task_key="mq_niah_6q",
        q1_indices=(0, 4, 5),
        q2_indices=(1, 2, 3),
        max_new_tokens=48,
    ),
    SplitTaskSpec(
        name="mq_niah_6q_split_256_to_134",
        base_task_key="mq_niah_6q",
        q1_indices=(1, 4, 5),
        q2_indices=(0, 2, 3),
        max_new_tokens=48,
    ),
    SplitTaskSpec(
        name="mq_niah_6q_split_356_to_124",
        base_task_key="mq_niah_6q",
        q1_indices=(2, 4, 5),
        q2_indices=(0, 1, 3),
        max_new_tokens=48,
    ),
    MQ_NIAH_6Q_CLEAN_SPLIT_SPEC,
)

MQ_NIAH_8Q_CLEAN_SPLIT_SPECS = (
    SplitTaskSpec(
        name="mq_niah_8q_split_5678_to_1234",
        base_task_key="mq_niah_8q",
        q1_indices=(4, 5, 6, 7),
        q2_indices=(0, 1, 2, 3),
        max_new_tokens=64,
    ),
    SplitTaskSpec(
        name="mq_niah_8q_split_1678_to_2345",
        base_task_key="mq_niah_8q",
        q1_indices=(0, 5, 6, 7),
        q2_indices=(1, 2, 3, 4),
        max_new_tokens=64,
    ),
    SplitTaskSpec(
        name="mq_niah_8q_split_2678_to_1345",
        base_task_key="mq_niah_8q",
        q1_indices=(1, 5, 6, 7),
        q2_indices=(0, 2, 3, 4),
        max_new_tokens=64,
    ),
    SplitTaskSpec(
        name="mq_niah_8q_split_3678_to_1245",
        base_task_key="mq_niah_8q",
        q1_indices=(2, 5, 6, 7),
        q2_indices=(0, 1, 3, 4),
        max_new_tokens=64,
    ),
    SplitTaskSpec(
        name="mq_niah_8q_split_4678_to_1235",
        base_task_key="mq_niah_8q",
        q1_indices=(3, 5, 6, 7),
        q2_indices=(0, 1, 2, 4),
        max_new_tokens=64,
    ),
)

CLEAN_SPLIT_SPECS = (
    DEFAULT_SPLIT_SPEC,
    SplitTaskSpec(
        name="mq_niah_4q_split_24_to_13",
        base_task_key="mq_niah_4q",
        q1_indices=(1, 3),
        q2_indices=(0, 2),
    ),
    SplitTaskSpec(
        name="mq_niah_4q_split_34_to_12",
        base_task_key="mq_niah_4q",
        q1_indices=(2, 3),
        q2_indices=(0, 1),
    ),
)

TAIL_LEAKY_SPLIT_SPECS = (
    SplitTaskSpec(
        name="mq_niah_4q_split_12_to_34",
        base_task_key="mq_niah_4q",
        q1_indices=(0, 1),
        q2_indices=(2, 3),
    ),
    SplitTaskSpec(
        name="mq_niah_4q_split_13_to_24",
        base_task_key="mq_niah_4q",
        q1_indices=(0, 2),
        q2_indices=(1, 3),
    ),
    SplitTaskSpec(
        name="mq_niah_4q_split_23_to_14",
        base_task_key="mq_niah_4q",
        q1_indices=(1, 2),
        q2_indices=(0, 3),
    ),
)

ALL_SPLIT_SPECS = CLEAN_SPLIT_SPECS + TAIL_LEAKY_SPLIT_SPECS
EXTRA_SPLIT_SPECS = (
    MQ_NIAH_2Q_CLEAN_SPLIT_SPEC,
    MQ_NIAH_3Q_CLEAN_SPLIT_SPEC,
    *MQ_NIAH_6Q_CLEAN_SPLIT_SPECS,
    *MQ_NIAH_8Q_CLEAN_SPLIT_SPECS,
)
SPLIT_SPECS_BY_NAME = {spec.name: spec for spec in ALL_SPLIT_SPECS + EXTRA_SPLIT_SPECS}


@dataclass(frozen=True)
class SplitPreparedExample:
    """Base example plus Q1/Q2 prepared views and span names for the split."""

    split_spec: SplitTaskSpec
    base_example: TaskExample
    q1_prepared: PreparedExample
    q2_prepared: PreparedExample
    q1_span_names: tuple[str, ...]
    q2_span_names: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedTurn:
    """Decoded text, emitted token ids, and the updated tracked cache."""

    text: str
    token_ids: torch.Tensor
    cache: PositionTrackedCache


@dataclass(frozen=True)
class ContextKeepPlan:
    """Reusable context-only keep order for one fixed post-Q1 cache."""

    context_len: int
    tail_positions: tuple[int, ...]
    mandatory_context_positions: tuple[int, ...]
    ranked_candidate_positions: tuple[int, ...]
    importance_scores: dict[int, float]


@dataclass(frozen=True)
class ContextPartition:
    """One materialized context-only partition at a specific context budget."""

    compressed: PositionTrackedCache
    evicted: PositionTrackedCache
    kept_context_positions: tuple[int, ...]
    evicted_context_positions: tuple[int, ...]


def split_example_for_turn(
    example: TaskExample,
    *,
    query_indices: Sequence[int],
    split_name: str,
    question_suffix: str = VALUES_ONLY_SUFFIX,
    answer_prefix: str = VALUES_ONLY_PREFIX,
    max_new_tokens: int = VALUES_ONLY_MAX_NEW_TOKENS,
) -> TaskExample:
    """Project one multi-query example into one values-only split turn."""
    query_keys = list(example.metadata.get("query_keys", []))
    if not query_keys:
        raise ValueError("Split turns require TaskExample.metadata['query_keys'].")

    selected_indices = tuple(int(index) for index in query_indices)
    selected_keys = [query_keys[index] for index in selected_indices]
    selected_outputs = [example.outputs[index] for index in selected_indices]
    query = join_natural(selected_keys)
    question = (
        f"What are the special magic values for {query} mentioned in the provided text? "
        f"{question_suffix}"
    )
    metadata = dict(example.metadata)
    metadata["query_keys"] = selected_keys
    metadata["split_name"] = split_name
    metadata["split_indices"] = list(selected_indices)
    metadata["response_format"] = "values_only_csv"
    return replace(
        example,
        question=question,
        answer_prefix=answer_prefix,
        outputs=selected_outputs,
        max_new_tokens=int(max_new_tokens),
        metadata=metadata,
    )


def build_mismatched_question_ids(
    *,
    base_example: TaskExample,
    split_spec: SplitTaskSpec,
    tokenizer,
) -> torch.Tensor:
    """Build a task-matched decoy query whose keys do not exist in the context."""
    key_count = max(1, len(split_spec.q2_indices))
    wrong_keys = [f"{MISMATCHED_KEY_STEM}_{slot + 1}" for slot in range(key_count)]
    query = join_natural(wrong_keys)
    question = (
        f"What are the special magic values for {query} mentioned in the provided text? "
        f"{split_spec.question_suffix}"
    )
    decoy_example = replace(
        base_example,
        question=question,
        answer_prefix=split_spec.answer_prefix,
        outputs=[],
        max_new_tokens=int(split_spec.max_new_tokens),
    )
    prepared = prepare_example_for_model(decoy_example, tokenizer)
    return prepared.question_ids


def build_base_example(
    *,
    split_spec: SplitTaskSpec,
    index: int,
    context_length: int,
    tokenizer,
    dataset_seed_offset: int = 0,
) -> TaskExample:
    """Build the base long-context example shared across Q1/Q2 split views."""
    return build_task_example(
        split_spec.base_task_key,
        index,
        context_length,
        tokenizer,
        dataset_seed_offset=dataset_seed_offset,
    )


def build_split_prepared_from_base_example(
    *,
    base_example: TaskExample,
    split_spec: SplitTaskSpec,
    tokenizer,
) -> SplitPreparedExample:
    """Build both split-turn prepared views from one already-built base example."""
    q1_example = split_example_for_turn(
        base_example,
        query_indices=split_spec.q1_indices,
        split_name=f"{split_spec.name}:q1",
        question_suffix=split_spec.question_suffix,
        answer_prefix=split_spec.answer_prefix,
        max_new_tokens=split_spec.max_new_tokens,
    )
    q2_example = split_example_for_turn(
        base_example,
        query_indices=split_spec.q2_indices,
        split_name=f"{split_spec.name}:q2",
        question_suffix=split_spec.question_suffix,
        answer_prefix=split_spec.answer_prefix,
        max_new_tokens=split_spec.max_new_tokens,
    )
    q1_span_names = tuple(f"needle_{needle_index + 1}" for needle_index in split_spec.q1_indices)
    q2_span_names = tuple(f"needle_{needle_index + 1}" for needle_index in split_spec.q2_indices)
    return SplitPreparedExample(
        split_spec=split_spec,
        base_example=base_example,
        q1_prepared=prepare_example_for_model(q1_example, tokenizer),
        q2_prepared=prepare_example_for_model(q2_example, tokenizer),
        q1_span_names=q1_span_names,
        q2_span_names=q2_span_names,
    )


def build_split_prepared_example(
    *,
    split_spec: SplitTaskSpec,
    index: int,
    context_length: int,
    tokenizer,
    dataset_seed_offset: int = 0,
) -> SplitPreparedExample:
    """Build one base example and both split-turn prepared views."""
    base_example = build_base_example(
        split_spec=split_spec,
        index=index,
        context_length=context_length,
        tokenizer=tokenizer,
        dataset_seed_offset=dataset_seed_offset,
    )
    return build_split_prepared_from_base_example(
        base_example=base_example,
        split_spec=split_spec,
        tokenizer=tokenizer,
    )


def _move_tensor_to_cpu_pinned(tensor: torch.Tensor) -> torch.Tensor:
    cpu_tensor = tensor.detach().to("cpu").contiguous()
    try:
        return cpu_tensor.pin_memory()
    except RuntimeError:
        return cpu_tensor


def _cache_to_cpu_pinned(cache: PositionTrackedCache) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (
            _move_tensor_to_cpu_pinned(key),
            _move_tensor_to_cpu_pinned(value),
        )
        for key, value in cache.kv
    )


def next_logical_position(cache: PositionTrackedCache) -> int:
    """Return the next logical absolute position after a tracked cache."""
    if not cache.positions:
        return 0
    return int(max(cache.positions) + 1)


def generate_turn(model, tokenizer, prepared: PreparedExample, cache: PositionTrackedCache) -> GeneratedTurn:
    """Greedy-decode one turn and return the cache with all emitted tokens included."""
    device = model_device(model)
    question_ids = prepared.question_ids.to(device)
    if question_ids.ndim == 1:
        question_ids = question_ids.unsqueeze(0)

    logical_base = next_logical_position(cache)
    dense_base = len(cache)
    query_len = int(question_ids.shape[1])
    question_positions = list(range(logical_base, logical_base + query_len))

    live_cache = to_dynamic_cache(cache, config=model.config)
    position_ids = torch.arange(logical_base, logical_base + query_len, device=device).unsqueeze(0)
    cache_position = torch.arange(dense_base, dense_base + query_len, device=device)
    stop_ids = model.generation_config.eos_token_id
    if not isinstance(stop_ids, list):
        stop_ids = [stop_ids]
    stop_ids = [int(stop_id) for stop_id in stop_ids if stop_id is not None]

    with torch.no_grad():
        outputs = model(
            input_ids=question_ids,
            past_key_values=live_cache,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
            logits_to_keep=1,
        )
        live_cache = outputs.past_key_values
        next_token = outputs.logits[0, -1].argmax().reshape(1, 1)
        generated_ids: list[torch.Tensor] = []
        generated_positions: list[int] = []
        next_position = logical_base + query_len
        next_cache_position = dense_base + query_len

        for _ in range(int(prepared.example.max_new_tokens)):
            token_scalar = next_token[0, 0].detach().cpu()
            generated_ids.append(token_scalar)
            generated_positions.append(int(next_position))
            outputs = model(
                input_ids=next_token,
                past_key_values=live_cache,
                position_ids=torch.tensor([[next_position]], device=device),
                cache_position=torch.tensor([next_cache_position], device=device),
                use_cache=True,
                logits_to_keep=1,
            )
            live_cache = outputs.past_key_values
            if int(token_scalar.item()) in stop_ids:
                break
            next_token = outputs.logits[0, -1].argmax().reshape(1, 1)
            next_position += 1
            next_cache_position += 1

    token_ids = torch.stack(generated_ids) if generated_ids else torch.empty((0,), dtype=torch.long)
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    full_positions = list(cache.positions) + question_positions + generated_positions
    tracked_cache = PositionTrackedCache(to_tuple_cache(live_cache), full_positions)
    return GeneratedTurn(
        text=text,
        token_ids=token_ids,
        cache=tracked_cache,
    )


def build_turn_n_keep_plan(
    *,
    post_q1_cache: PositionTrackedCache,
    q1_answer_ids: torch.Tensor,
    context_len: int,
    sink_size: int,
    recency_window: int,
    pooling: str = "max",
    initial_compressor: str = "snapkv",
) -> ContextKeepPlan:
    """Score the full post-Q1 cache from the generated answer tail and freeze one keep order."""
    if context_len <= 0 or context_len > len(post_q1_cache):
        raise ValueError(f"context_len must lie in [1, {len(post_q1_cache)}], got {context_len}.")
    compressor = str(initial_compressor).strip().lower()

    if compressor == "snapkv":
        if int(q1_answer_ids.numel()) <= 0:
            raise ValueError("Q1 answer ids must be non-empty so the turn-N observation window is well defined.")
        policy = SnapKV(
            obs_window_size=int(q1_answer_ids.numel()),
            sink_size=0,
            recency_window=0,
            pooling=pooling,
        )
        _, importance, _ = policy.prepare_eviction_inputs(post_q1_cache)

        sink_count = min(int(context_len), int(sink_size))
        recency_count = min(int(recency_window), max(0, int(context_len) - sink_count))
        recency_start = max(sink_count, int(context_len) - recency_count)
        mandatory_context_positions = tuple(sorted(set(range(sink_count)) | set(range(recency_start, int(context_len)))))
        mandatory_set = set(mandatory_context_positions)
        ranked_candidate_positions = tuple(
            sorted(
                (position for position in range(int(context_len)) if position not in mandatory_set),
                key=lambda position: (-float(importance[position].item()), position),
            )
        )
        importance_scores = {
            int(position): float(importance[position].item())
            for position in range(int(context_len))
        }
    elif compressor == "h2o":
        if int(q1_answer_ids.numel()) <= 0:
            raise ValueError("H2O-inspired compression requires non-empty recent observation ids.")
        importance = _score_h2o_tokens(
            post_q1_cache,
            obs_window_size=int(q1_answer_ids.numel()),
        )
        sink_count = min(int(context_len), int(sink_size))
        recency_count = min(int(recency_window), max(0, int(context_len) - sink_count))
        recency_start = max(sink_count, int(context_len) - recency_count)
        mandatory_context_positions = tuple(sorted(set(range(sink_count)) | set(range(recency_start, int(context_len)))))
        mandatory_set = set(mandatory_context_positions)
        ranked_candidate_positions = tuple(
            sorted(
                (position for position in range(int(context_len)) if position not in mandatory_set),
                key=lambda position: (-float(importance[position].item()), position),
            )
        )
        importance_scores = {
            int(position): float(importance[position].item())
            for position in range(int(context_len))
        }
    elif compressor == "scissorhands":
        if int(q1_answer_ids.numel()) <= 0:
            raise ValueError("Scissorhands-style compression requires non-empty recent observation ids.")
        importance = _score_scissorhands_tokens(
            post_q1_cache,
            obs_window_size=int(q1_answer_ids.numel()),
        )
        sink_count = min(int(context_len), int(sink_size))
        recency_count = min(int(recency_window), max(0, int(context_len) - sink_count))
        recency_start = max(sink_count, int(context_len) - recency_count)
        mandatory_context_positions = tuple(sorted(set(range(sink_count)) | set(range(recency_start, int(context_len)))))
        mandatory_set = set(mandatory_context_positions)
        ranked_candidate_positions = tuple(
            sorted(
                (position for position in range(int(context_len)) if position not in mandatory_set),
                key=lambda position: (-float(importance[position].item()), position),
            )
        )
        importance_scores = {
            int(position): float(importance[position].item())
            for position in range(int(context_len))
        }
    elif compressor == "streaming_llm":
        sink_count = min(int(context_len), int(sink_size))
        mandatory_context_positions = tuple(range(sink_count))
        ranked_candidate_positions = tuple(range(int(context_len) - 1, sink_count - 1, -1))
        denominator = max(1, int(context_len) - 1)
        importance_scores = {
            int(position): float(position / denominator)
            for position in range(int(context_len))
        }
    else:
        raise ValueError(f"Unsupported initial_compressor: {initial_compressor!r}.")

    tail_positions = tuple(range(int(context_len), len(post_q1_cache)))
    return ContextKeepPlan(
        context_len=int(context_len),
        tail_positions=tail_positions,
        mandatory_context_positions=mandatory_context_positions,
        ranked_candidate_positions=ranked_candidate_positions,
        importance_scores=importance_scores,
    )


def _score_h2o_tokens(
    full_cache: PositionTrackedCache,
    *,
    obs_window_size: int,
) -> torch.Tensor:
    """Approximate H2O-inspired accumulated attention scores over the latest rows."""
    cache = to_tuple_cache(full_cache)
    seq_len = len(full_cache)
    obs_len = min(max(1, int(obs_window_size)), seq_len)
    obs_start = seq_len - obs_len
    layer_scores: list[torch.Tensor] = []

    for key, _ in cache:
        key_float = key.detach().to(dtype=torch.float32)
        obs_rows = key_float[:, :, obs_start:, :]
        scores = torch.matmul(obs_rows, key_float.transpose(-2, -1)) / math.sqrt(key_float.shape[-1])
        scores = torch.softmax(scores, dim=-1)
        layer_scores.append(scores.sum(dim=2).mean(dim=(0, 1)))

    return torch.stack(layer_scores, dim=0).mean(dim=0)


def _score_scissorhands_tokens(
    full_cache: PositionTrackedCache,
    *,
    obs_window_size: int,
) -> torch.Tensor:
    """Score tokens by Scissorhands-style persistence over the latest rows.

    The original Scissorhands compressor increments a drop counter for tokens
    that receive below-average attention within a history window, while recent
    tokens are protected separately by the caller's recency reserve. This
    one-shot Phase 6 variant ranks tokens by the fraction of layer/head/history
    observations for which their attention is at least the uniform threshold.
    """
    cache = to_tuple_cache(full_cache)
    seq_len = len(full_cache)
    obs_len = min(max(1, int(obs_window_size)), seq_len)
    obs_start = seq_len - obs_len
    layer_scores: list[torch.Tensor] = []

    for key, _ in cache:
        key_float = key.detach().to(dtype=torch.float32)
        obs_rows = key_float[:, :, obs_start:, :]
        scores = torch.matmul(obs_rows, key_float.transpose(-2, -1)) / math.sqrt(key_float.shape[-1])
        scores = torch.softmax(scores, dim=-1)
        threshold = 1.0 / max(1, int(seq_len))
        pivotal_fraction = (scores >= threshold).to(dtype=torch.float32).mean(dim=(0, 1, 2))
        mean_attention = scores.mean(dim=(0, 1, 2))
        layer_scores.append(pivotal_fraction + 1e-3 * mean_attention)

    return torch.stack(layer_scores, dim=0).mean(dim=0)


def materialize_context_partition(
    *,
    full_post_q1_cache: PositionTrackedCache,
    keep_plan: ContextKeepPlan,
    context_budget: int,
) -> ContextPartition:
    """Keep exactly one context budget plus all of the Q1 tail."""
    context_budget = min(max(int(context_budget), 0), int(keep_plan.context_len))
    mandatory = list(keep_plan.mandatory_context_positions[: min(context_budget, len(keep_plan.mandatory_context_positions))])
    remaining_slots = max(0, context_budget - len(mandatory))
    selected = list(keep_plan.ranked_candidate_positions[:remaining_slots])
    kept_context_positions = tuple(sorted(set(mandatory + selected)))
    kept_context_set = set(kept_context_positions)
    evicted_context_positions = tuple(
        position
        for position in range(int(keep_plan.context_len))
        if position not in kept_context_set
    )

    keep_dense_indices = list(kept_context_positions) + list(keep_plan.tail_positions)
    compressed = slice_kv(full_post_q1_cache, keep_dense_indices)
    evicted = slice_kv(full_post_q1_cache, evicted_context_positions)
    if not isinstance(compressed, PositionTrackedCache) or not isinstance(evicted, PositionTrackedCache):
        raise RuntimeError("Phase 2 slice_kv did not preserve position tracking for the Phase 6 partition.")
    evicted_cpu = PositionTrackedCache(_cache_to_cpu_pinned(evicted), list(evicted.positions))
    return ContextPartition(
        compressed=compressed,
        evicted=evicted_cpu,
        kept_context_positions=kept_context_positions,
        evicted_context_positions=evicted_context_positions,
    )


def compute_q2_query_rows(
    model,
    *,
    active_cache: PositionTrackedCache,
    question_ids: torch.Tensor,
) -> torch.Tensor:
    """Encode Q2 and return appended attention-state rows used for repair scoring.

    The current runner scores repair candidates with the appended per-token rows
    available after the forward pass, which act as a Q2-conditioned proxy rather
    than exact extracted Q projections.
    """
    outputs = resume_forward(
        model,
        question_ids,
        active_cache,
        logical_position_base=next_logical_position(active_cache),
        dense_cache_position_base=len(active_cache),
        logits_to_keep=1,
    )
    extended_kv = to_tuple_cache(outputs.past_key_values)
    context_len = len(active_cache)
    query_rows: list[torch.Tensor] = []
    for key, _ in extended_kv:
        key_float = key.detach().to("cpu", dtype=torch.float32)
        query_rows.append(key_float[0, :, context_len:, :].contiguous())
    return torch.stack(query_rows, dim=0)


def compute_q2_exact_query_rows(
    model,
    *,
    active_cache: PositionTrackedCache,
    question_ids: torch.Tensor,
) -> torch.Tensor:
    """Encode Q2 and return exact per-layer query projections after RoPE."""
    device = model_device(model)
    question_ids = question_ids.to(device)
    if question_ids.ndim == 1:
        question_ids = question_ids.unsqueeze(0)

    logical_base = next_logical_position(active_cache)
    dense_base = len(active_cache)
    seq_len = int(question_ids.shape[1])
    position_ids = torch.arange(logical_base, logical_base + seq_len, device=device).unsqueeze(0)
    cache_position = torch.arange(dense_base, dense_base + seq_len, device=device)
    model_cache = to_dynamic_cache(active_cache, config=model.config)

    kwargs = {
        "input_ids": question_ids,
        "past_key_values": model_cache,
        "position_ids": position_ids,
        "cache_position": cache_position,
        "use_cache": True,
        "output_hidden_states": True,
        "logits_to_keep": 1,
    }
    with torch.no_grad():
        try:
            outputs = model(**kwargs)
        except TypeError:
            kwargs.pop("logits_to_keep", None)
            try:
                outputs = model(**kwargs)
            except TypeError:
                kwargs.pop("cache_position", None)
                outputs = model(**kwargs)

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states for exact Q-row extraction.")

    query_rows: list[torch.Tensor] = []
    for layer_index, layer in enumerate(model.model.layers):
        layer_input = hidden_states[layer_index]
        normalized = layer.input_layernorm(layer_input)
        attention = layer.self_attn
        query = attention.q_proj(normalized).view(*normalized.shape[:-1], -1, attention.head_dim).transpose(1, 2)
        key = attention.k_proj(normalized).view(*normalized.shape[:-1], -1, attention.head_dim).transpose(1, 2)
        cos, sin = model.model.rotary_emb(normalized, position_ids)
        query, _ = _apply_rotary_pos_emb_for_model(model, query, key, cos, sin)
        query_rows.append(query[0].detach().to("cpu", dtype=torch.float32).contiguous())
    return torch.stack(query_rows, dim=0)


def relevant_positions_for_spans(prepared: PreparedExample, span_names: Sequence[str]) -> tuple[int, ...]:
    """Return the sorted token positions for a fixed list of relevant span names."""
    positions: set[int] = set()
    for span_name in span_names:
        positions.update(int(position) for position in prepared.span_token_positions.get(span_name, []))
    return tuple(sorted(positions))


def relevant_position_groups_for_spans(prepared: PreparedExample, span_names: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    """Return one sorted token-position tuple per relevant span name."""
    groups: list[tuple[int, ...]] = []
    for span_name in span_names:
        groups.append(
            tuple(
                sorted(
                    int(position)
                    for position in prepared.span_token_positions.get(span_name, [])
                )
            )
        )
    return tuple(groups)
