"""Inference helpers for preparing prompts, running compression, and scoring outputs."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import DynamicCache

from .evaluation import classify_error, matched_outputs, sample_score
from .models import PredictionRecord, SpanSurvival, TaskExample
from .prompting import char_span_to_token_positions, render_context_plan


@dataclass
class PreparedExample:
    """Task example enriched with rendered tokens and span-to-token mappings."""

    example: TaskExample
    rendered_context: str
    context_ids: torch.Tensor
    question_ids: torch.Tensor
    span_token_positions: dict[str, list[int]]
    segment_token_ranges: list[tuple[str, int, int]]


def _char_range_to_token_range(tokenizer, rendered_context: str, char_start: int, char_end: int) -> tuple[int, int]:
    """Collapse a character span into an inclusive-exclusive token slice."""
    positions = char_span_to_token_positions(tokenizer, rendered_context, char_start, char_end)
    if not positions:
        raise RuntimeError(f"No token positions found for char range [{char_start}, {char_end}).")
    return positions[0], positions[-1] + 1


def prepare_example_for_model(example: TaskExample, tokenizer) -> PreparedExample:
    """Render one example into the exact token layout expected by the model and trace code."""
    # Render the prompt with the chat template so token offsets match runtime.
    rendered = render_context_plan(tokenizer, example.context, example.question, example.answer_prefix)
    prefix_offset = len(rendered.template_prefix)
    span_token_positions: dict[str, list[int]] = {}
    # Map task-relevant character spans into token positions within the rendered context.
    for span in example.relevant_spans:
        # Each task span is stored in raw-context character coordinates, so we
        # shift by the chat-template prefix before converting to token indices.
        positions = char_span_to_token_positions(
            tokenizer,
            rendered.rendered_context,
            prefix_offset + span.char_start,
            prefix_offset + span.char_end,
        )
        span_token_positions[span.name] = positions
    # Build segment ranges that simulate staged prompt arrival during prefill.
    segment_ranges = []
    for segment_index, segment in enumerate(example.prefill_segments):
        # Segment boundaries let Phase 1 mimic a staged prompt arrival pattern,
        # even though everything ultimately comes from one rendered message.
        token_end_start, token_end = _char_range_to_token_range(
            tokenizer,
            rendered.rendered_context,
            prefix_offset + segment.char_start,
            prefix_offset + segment.char_end,
        )
        start = 0 if segment_index == 0 else token_end_start
        end = token_end
        segment_ranges.append((segment.name, start, end))
    return PreparedExample(
        example=example,
        rendered_context=rendered.rendered_context,
        context_ids=rendered.context_ids,
        question_ids=rendered.question_ids,
        span_token_positions=span_token_positions,
        segment_token_ranges=segment_ranges,
    )


def generate_answer(
    model,
    tokenizer,
    question_ids: torch.Tensor,
    cache: DynamicCache,
    *,
    position_base: int,
    cache_position_base: int,
    max_new_tokens: int,
) -> str:
    """Greedy-generate an answer continuation on top of an already-built KV cache."""
    device = model.device
    question_ids = question_ids.to(device)
    # Keep the logical RoPE timeline aligned to the original prompt even when
    # the live cache has been compacted into a shorter dense prefix.
    position_ids = torch.arange(position_base, position_base + question_ids.shape[1], device=device).unsqueeze(0)
    cache_position = torch.arange(
        cache_position_base,
        cache_position_base + question_ids.shape[1],
        device=device,
    )
    # Run a single forward pass for the prompt+question to seed generation.
    outputs = model(
        input_ids=question_ids,
        past_key_values=cache,
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=True,
        logits_to_keep=1,
    )
    position_ids = position_ids[:, -1:] + 1
    cache_position = cache_position[-1:] + 1
    # Decode greedily one token at a time, updating positions into the cache.
    # Seed decoding with the first argmax from the prompt+question pass, then
    # continue one token at a time from the updated cache.
    generated_ids = [outputs.logits[0, -1].argmax()]
    stop_ids = model.generation_config.eos_token_id
    if not isinstance(stop_ids, list):
        stop_ids = [stop_ids]
    for step in range(max_new_tokens - 1):
        outputs = model(
            input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
            past_key_values=cache,
            position_ids=position_ids + step,
            cache_position=cache_position + step,
            use_cache=True,
        )
        new_id = outputs.logits[0, -1].argmax()
        generated_ids.append(new_id)
        # Stop early on EOS so short correct answers do not get polluted by
        # unnecessary continuation text.
        if new_id.item() in stop_ids:
            break
    return tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True)


def compute_span_survival(
    example: TaskExample,
    prepared: PreparedExample,
    recorder: EvictionTraceRecorder | None,
) -> list[SpanSurvival]:
    """Estimate how much of each task-relevant span survived compression."""
    # Condition A (no trace) treats any mapped span as fully preserved.
    if recorder is None or not recorder.layers:
        # Condition A/no-trace runs are treated as fully preserved for any span
        # that was successfully mapped into tokens.
        spans = []
        for span in example.relevant_spans:
            token_positions = prepared.span_token_positions.get(span.name, [])
            spans.append(
                SpanSurvival(
                    name=span.name,
                    kind=span.kind,
                    depth_fraction=span.depth_fraction,
                    survival_fraction=1.0 if token_positions else 0.0,
                    kept_token_count=len(token_positions),
                    total_token_count=len(token_positions),
                    metadata=span.metadata,
                )
            )
        return spans

    spans: list[SpanSurvival] = []
    # Aggregate survival by checking each span's tokens against trace masks.
    for span in example.relevant_spans:
        token_positions = prepared.span_token_positions.get(span.name, [])
        if not token_positions:
            # If the tokenizer never produced positions for this span, it cannot
            # survive by definition.
            spans.append(
                SpanSurvival(
                    name=span.name,
                    kind=span.kind,
                    depth_fraction=span.depth_fraction,
                    survival_fraction=0.0,
                    kept_token_count=0,
                    total_token_count=0,
                    metadata=span.metadata,
                )
            )
            continue
        # Track how often tokens are kept across layers/heads.
        total_checks = 0
        kept_checks = 0
        kept_token_count = 0
        for trace in recorder.layers.values():
            kept_mask = trace.kept_mask
            if kept_mask.ndim == 3:
                kept_mask = kept_mask.squeeze(0)
            elif kept_mask.ndim == 1:
                kept_mask = kept_mask.unsqueeze(0)
            seq_len = kept_mask.shape[-1]
            head_count = kept_mask.shape[0]
            valid_positions = [position for position in token_positions if position < seq_len]
            # `survival_fraction` averages over every layer/head membership test,
            # while `kept_token_count` asks whether any head kept each token.
            total_checks += len(token_positions) * head_count
            if valid_positions:
                kept_checks += int(kept_mask[:, valid_positions].sum().item())
                kept_token_count += int(kept_mask[:, valid_positions].any(dim=0).sum().item())
        # Summarize the span with both fractional survival and kept-token counts.
        spans.append(
            SpanSurvival(
                name=span.name,
                kind=span.kind,
                depth_fraction=span.depth_fraction,
                survival_fraction=(kept_checks / total_checks) if total_checks else 0.0,
                kept_token_count=kept_token_count,
                total_token_count=len(token_positions) * max(len(recorder.layers), 1),
                metadata=span.metadata,
            )
        )
    return spans


@torch.inference_mode()
def run_example(
    *,
    model,
    tokenizer,
    prepared: PreparedExample,
    algorithm: str,
    budget: int | None,
    condition: str,
    trace_path: Path | None,
    query_log_tokens: int,
) -> PredictionRecord:
    """Run one example under the requested condition and package the scored result."""
    example = prepared.example
    cache = DynamicCache()
    recorder = None
    press = None
    # Configure compression instrumentation for condition B.
    if condition == "condition_b":
        # Condition B is the compressed path: attach a recorder and a kvpress
        # compression policy so we can later explain which tokens were kept.
        from .eviction import EvictionTraceRecorder, build_press

        assert budget is not None
        assert trace_path is not None
        recorder = EvictionTraceRecorder(
            trace_path=trace_path,
            algorithm=algorithm,
            budget=budget,
            context_length=example.target_context_length,
            sample_index=example.index,
            metadata={"task_name": example.task_name},
        )
        press = build_press(algorithm, budget=budget, recorder=recorder, query_log_tokens=query_log_tokens)

    # Prefill the context into the cache, optionally under a compression policy.
    ctx = press(model) if press is not None else contextlib.nullcontext()
    with ctx:
        for _, start, end in prepared.segment_token_ranges:
            # Feed the context segments into the base model to build the KV cache.
            segment_ids = prepared.context_ids[:, start:end].to(model.device)
            segment_positions = torch.arange(start, end, device=model.device)
            model.model(
                input_ids=segment_ids,
                past_key_values=cache,
                position_ids=segment_positions.unsqueeze(0),
                cache_position=segment_positions,
                use_cache=True,
            )

    compressed_context_length = int(cache.get_seq_length())
    original_context_length = int(prepared.context_ids.shape[1])
    # Decode the answer from the cached context and appended question.
    # Once the cache is ready, append the question and decode the answer using
    # the post-compression cache as the starting point.
    prediction = generate_answer(
        model=model,
        tokenizer=tokenizer,
        question_ids=prepared.question_ids,
        cache=cache,
        position_base=original_context_length,
        cache_position_base=compressed_context_length,
        max_new_tokens=example.max_new_tokens,
    )

    if recorder is not None:
        recorder.save()

    # Attach both raw accuracy and attribution-friendly span survival data so
    # downstream analysis can distinguish recall failures from hallucinations.
    spans = compute_span_survival(example, prepared, recorder)
    matched = matched_outputs(prediction, example.outputs)
    error_type = classify_error(example, prediction, matched, spans)
    return PredictionRecord(
        index=example.index,
        task_name=example.task_name,
        task_family=example.task_family,
        context_length=example.target_context_length,
        algorithm=algorithm,
        budget=budget,
        condition=condition,
        outputs=example.outputs,
        prediction=prediction,
        sample_score=sample_score(prediction, example.outputs),
        error_type=error_type,
        matched_outputs=matched,
        span_survival=spans,
        compressed_context_length=compressed_context_length,
        trace_path=None if trace_path is None else str(trace_path),
        metadata=example.metadata,
    )
