"""Shared inference helpers for benchmark evaluation."""

from contextlib import nullcontext
from typing import Optional, Union

import torch

from idlekv.core.compression import CompressedKVManager
from idlekv.utils.generation import greedy_generation_config
from idlekv.utils.kv_cache import get_layer_kv


def _model_device(model) -> torch.device:
    """Return the device of the first model parameter."""
    return next(model.parameters()).device


def decode_with_manager(
    model,
    tokenizer,
    manager: CompressedKVManager,
    past_key_values,
    initial_logits: torch.Tensor,
    max_new_tokens: int = 64,
    sync_refresh_stride: Optional[int] = None,
    sync_refresh_phases: Union[str, int, None] = "2",
):
    """
    Greedy decode while preserving IdleKV semantic positions.

    When `sync_refresh_stride` is set, runs synchronous refinement every N
    generated tokens to model a RefreshKV-style baseline.
    """
    generated_tokens = []
    refinement_results = []
    current_past_kv = past_key_values
    next_token_id = None
    logits = initial_logits

    with torch.inference_mode():
        for step in range(max_new_tokens):
            if next_token_id is not None:
                outputs = model(
                    input_ids=next_token_id.unsqueeze(0),
                    past_key_values=current_past_kv,
                    position_ids=manager.next_position_ids(num_new_tokens=1),
                    use_cache=True,
                    output_hidden_states=True,
                )
                logits = outputs.logits[:, -1, :]
                current_past_kv = outputs.past_key_values

                query_state = manager.build_query_state(outputs.hidden_states)
                new_kv_per_layer = [
                    (
                        get_layer_kv(current_past_kv, layer_idx)[0][:, :, -1:, :],
                        get_layer_kv(current_past_kv, layer_idx)[1][:, :, -1:, :],
                    )
                    for layer_idx in range(manager.num_layers)
                ]
                manager.on_token_generated(query_state, new_kv_per_layer)
                current_past_kv = manager.maybe_evict_online(current_past_kv)

            next_token_id = logits.argmax(dim=-1)
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token_id.item())

            if (
                sync_refresh_stride is not None
                and sync_refresh_stride > 0
                and (step + 1) % sync_refresh_stride == 0
            ):
                refinement = manager.idle_refine(
                    current_past_kv,
                    max_time_ms=None,
                    phases=sync_refresh_phases,
                )
                current_past_kv = refinement.past_key_values
                refinement_results.append(refinement)

    return {
        "text": tokenizer.decode(generated_tokens, skip_special_tokens=True),
        "past_key_values": current_past_kv,
        "refinement_results": refinement_results,
    }


def generate_text(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    manager: Optional[CompressedKVManager] = None,
    press=None,
    idle_budget_ms: float = 0.0,
    phases: Union[str, int, None] = "1+2",
    sync_refresh_stride: Optional[int] = None,
):
    """
    Generate a continuation under one of the supported benchmark modes.

    Modes:
      - full cache baseline: `manager=None`, `press=None`
      - kvpress baseline: `press=...`
      - IdleKV benchmark: `manager=...`, `idle_budget_ms >= 0`
      - sync refresh baseline: `manager=...`, `sync_refresh_stride=N`
    """
    device = _model_device(model)
    input_ids = input_ids.to(device)

    if manager is None:
        press_ctx = press(model) if press is not None else nullcontext()
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        generation_config = greedy_generation_config(model)
        with torch.inference_mode(), press_ctx:
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        continuation = output_ids[0][input_ids.shape[1]:]
        return {
            "text": tokenizer.decode(continuation, skip_special_tokens=True),
            "refinement_results": [],
        }

    past_key_values = manager.prefill(input_ids)
    seeded_queries = manager.seed_query_buffer_from_prefill()
    refinement_results = []

    if sync_refresh_stride is None and idle_budget_ms > 0:
        refinement = manager.idle_refine(
            past_key_values,
            max_time_ms=idle_budget_ms,
            phases=phases,
        )
        past_key_values = refinement.past_key_values
        refinement_results.append(refinement)

    decoded = decode_with_manager(
        model=model,
        tokenizer=tokenizer,
        manager=manager,
        past_key_values=past_key_values,
        initial_logits=manager.last_prefill_logits,
        max_new_tokens=max_new_tokens,
        sync_refresh_stride=sync_refresh_stride,
    )
    refinement_results.extend(decoded["refinement_results"])
    return {
        "text": decoded["text"],
        "refinement_results": refinement_results,
        "seeded_queries": seeded_queries,
    }
