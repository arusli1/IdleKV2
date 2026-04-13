"""
Agentic simulation harness.

Wraps HuggingFace generate() to simulate tool-call pauses. Every N tokens,
pauses generation, runs IdleKV refinement for a sampled duration, then resumes.

This controlled setup isolates the effect of idle-time compute from confounds
in real agentic pipelines.
"""

import time
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from idlekv.core.compression import CompressedKVManager
from idlekv.core.scheduler import RefinementResult
from idlekv.simulation.tool_distributions import sample_tool_duration


@dataclass
class SimulationConfig:
    """Configuration for agentic simulation."""
    tool_call_interval: int = 100       # tokens between tool calls
    duration_median_ms: float = 1000    # median idle duration
    duration_min_ms: float = 100        # minimum idle duration
    duration_max_ms: float = 10000      # maximum idle duration
    max_gen_tokens: int = 512           # max tokens to generate total
    seed: int = 42


@dataclass
class SimulationTrace:
    """Full trace of a simulated agentic session."""
    generated_text: str = ""
    tool_calls: list = field(default_factory=list)  # list of dicts
    refinement_results: list = field(default_factory=list)
    total_gen_time_ms: float = 0
    total_idle_time_ms: float = 0
    tokens_per_sec: float = 0
    total_tokens: int = 0


def run_simulation(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    manager: CompressedKVManager,
    config: SimulationConfig,
    enable_refinement: bool = True,
) -> SimulationTrace:
    """
    Run a simulated agentic generation session.

    Generates tokens using the compressed cache, inserting tool-call pauses
    at regular intervals. During each pause, runs IdleKV refinement (if enabled).

    Args:
        model: HF causal LM
        tokenizer: HF tokenizer
        input_ids: [1, seq_len] prefill context
        manager: CompressedKVManager (already initialized)
        config: simulation parameters
        enable_refinement: if False, still pauses but doesn't refine (baseline)

    Returns:
        SimulationTrace with full session data
    """
    rng = np.random.RandomState(config.seed)
    trace = SimulationTrace()
    device = input_ids.device

    # Prefill
    past_kv = manager.prefill(input_ids)
    next_token_id = None
    gen_times = []
    tokens_generated = 0

    for step in range(config.max_gen_tokens):
        # Generate one token
        gen_start = time.perf_counter()

        if next_token_id is None:
            # First token: use prefill output
            with torch.no_grad():
                outputs = model(input_ids, past_key_values=past_kv, use_cache=True)
            logits = outputs.logits[:, -1, :]
            past_kv = outputs.past_key_values
        else:
            with torch.no_grad():
                outputs = model(
                    next_token_id.unsqueeze(0),
                    past_key_values=past_kv,
                    use_cache=True,
                )
            logits = outputs.logits[:, -1, :]
            past_kv = outputs.past_key_values

        # Sample next token (greedy for reproducibility)
        next_token_id = logits.argmax(dim=-1)
        gen_end = time.perf_counter()
        gen_times.append((gen_end - gen_start) * 1000)

        tokens_generated += 1

        # Track hidden state for query buffer
        # (In practice, hook into model's last layer output)
        # For now, use the last hidden state from the model
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            last_h = outputs.hidden_states[-1][:, -1, :]
        else:
            # Fallback: we need hidden states. Enable them.
            last_h = None  # Will need model output_hidden_states=True

        if last_h is not None:
            manager.on_token_generated(last_h, [
                (past_kv[l][0][:, :, -1:, :], past_kv[l][1][:, :, -1:, :])
                for l in range(manager.num_layers)
            ])

        # Check for EOS
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        # Tool-call pause every N tokens
        if tokens_generated % config.tool_call_interval == 0:
            duration_ms = sample_tool_duration(
                rng,
                median_ms=config.duration_median_ms,
                min_ms=config.duration_min_ms,
                max_ms=config.duration_max_ms,
            )

            tool_call_info = {
                "step": tokens_generated,
                "duration_ms": duration_ms,
            }

            if enable_refinement:
                result = manager.idle_refine(past_kv, max_time_ms=duration_ms)
                past_kv = result.past_key_values
                trace.refinement_results.append(result)
                tool_call_info["refinement"] = {
                    "phase1_time_ms": result.phase1_time_ms,
                    "phase2_time_ms": result.phase2_time_ms,
                    "total_time_ms": result.total_time_ms,
                }
            else:
                # Simulate idle time without refinement (just wait)
                time.sleep(duration_ms / 1000.0)

            trace.tool_calls.append(tool_call_info)
            trace.total_idle_time_ms += duration_ms

    # Finalize trace
    trace.total_tokens = tokens_generated
    trace.total_gen_time_ms = sum(gen_times)
    trace.tokens_per_sec = (tokens_generated / trace.total_gen_time_ms * 1000
                            if trace.total_gen_time_ms > 0 else 0)

    return trace
