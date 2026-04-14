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
from typing import Optional, Dict, List, Union

from idlekv.core.compression import CompressedKVManager
from idlekv.core.scheduler import RefinementResult
from idlekv.utils.kv_cache import cache_size, get_layer_kv


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
class SimulationResult:
    """Results from an agentic simulation."""
    total_tokens: int
    total_time_ms: float
    num_tool_calls: int
    total_idle_time_ms: float
    refinement_results: List[RefinementResult]
    throughput_tokens_per_sec: float = field(init=False)

    def __post_init__(self):
        self.throughput_tokens_per_sec = (
            self.total_tokens / (self.total_time_ms / 1000.0)
            if self.total_time_ms > 0 else 0
        )


def sample_tool_duration(
    median_ms: float,
    min_ms: float,
    max_ms: float,
    rng: np.random.Generator
) -> float:
    """Sample tool call duration from log-normal distribution."""
    # Log-normal distribution clipped to [min_ms, max_ms]
    log_median = np.log(median_ms)
    sigma = 0.5  # Standard deviation in log space

    duration = rng.lognormal(log_median, sigma)
    return np.clip(duration, min_ms, max_ms)


def simulate_agentic_workload(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    manager: Optional[CompressedKVManager] = None,
    config: Optional[SimulationConfig] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> SimulationResult:
    """
    Simulate an agentic workload with tool-call pauses and IdleKV refinement.

    Args:
        model: HF model
        tokenizer: HF tokenizer
        input_ids: Initial context
        manager: Optional CompressedKVManager for IdleKV
        config: Simulation configuration
        device: Device to use

    Returns:
        SimulationResult with metrics
    """
    if config is None:
        config = SimulationConfig()

    rng = np.random.default_rng(config.seed)
    start_time = time.perf_counter()

    # Initialize
    if manager is not None:
        past_key_values = manager.prefill(input_ids)
    else:
        past_key_values = None

    current_input = input_ids
    generated_tokens = 0
    tool_calls = 0
    total_idle_time = 0.0
    refinement_results = []

    print(f"    Simulating agentic workload (max {config.max_gen_tokens} tokens)...")

    with torch.no_grad():
        while generated_tokens < config.max_gen_tokens:
            # Generate next batch of tokens until tool call
            tokens_to_generate = min(
                config.tool_call_interval,
                config.max_gen_tokens - generated_tokens
            )

            if tokens_to_generate <= 0:
                break

            # Generate tokens
            if past_key_values is not None:
                # Use compressed cache
                current_seq_len = cache_size(past_key_values, layer_idx=0)
                position_ids = torch.arange(
                    current_seq_len,
                    current_seq_len + tokens_to_generate,
                    device=device
                ).unsqueeze(0)

                output = model.generate(
                    current_input[:, -1:] if generated_tokens > 0 else current_input,
                    past_key_values=past_key_values,
                    max_new_tokens=tokens_to_generate,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
            else:
                # Full cache baseline
                output = model.generate(
                    current_input,
                    max_new_tokens=tokens_to_generate,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )

            # Update state
            generated_tokens += tokens_to_generate
            current_input = output

            # Simulate tool call pause
            if generated_tokens < config.max_gen_tokens:
                tool_calls += 1

                # Sample idle duration
                idle_duration = sample_tool_duration(
                    config.duration_median_ms,
                    config.duration_min_ms,
                    config.duration_max_ms,
                    rng
                )
                total_idle_time += idle_duration

                # Run IdleKV refinement if available.
                # Query buffer is populated by real hidden states in run_simulation
                # via manager.on_token_generated; this harness path assumes the
                # caller has already done so (or will populate it before calling).
                if manager is not None:
                    if manager.query_buffer.count == 0:
                        raise RuntimeError(
                            "Query buffer is empty at idle_refine time — "
                            "hidden states must be captured during generation "
                            "via manager.on_token_generated(hidden_state, kv) "
                            "before calling idle_refine."
                        )
                    refinement = manager.idle_refine(
                        past_key_values,
                        max_time_ms=idle_duration
                    )
                    past_key_values = refinement.past_key_values
                    refinement_results.append(refinement)

    total_time = (time.perf_counter() - start_time) * 1000.0  # Convert to ms

    print(f"      Generated {generated_tokens} tokens, {tool_calls} tool calls")
    print(f"      Total idle time: {total_idle_time:.1f}ms")

    return SimulationResult(
        total_tokens=generated_tokens,
        total_time_ms=total_time,
        num_tool_calls=tool_calls,
        total_idle_time_ms=total_idle_time,
        refinement_results=refinement_results
    )


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

        # Explicit position_ids: after compression, the physical cache length
        # differs from the semantic sequence length. We must feed the true
        # absolute position so that RoPE on the new query is consistent with
        # the RoPE already baked into the retained keys.
        if next_token_id is None:
            # First post-prefill step: reuse the logits prefill already computed
            # for the last prefilled position, rather than re-running forward
            # with the entire input_ids (which would double-process the prompt).
            logits = manager.last_prefill_logits
            outputs = None  # no new forward; no hidden states to capture
        else:
            pos = manager.next_position_ids(num_new_tokens=1)
            with torch.no_grad():
                outputs = model(
                    next_token_id.unsqueeze(0),
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True,
                    position_ids=pos,
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
        if outputs is not None and getattr(outputs, 'hidden_states', None) is not None:
            last_h = outputs.hidden_states[-1][:, -1, :]
        else:
            # First-iter path reuses prefill logits without a new forward, so
            # no hidden state to capture here; prefill already populated the
            # KV cache, and the next iter will capture the next token's state.
            last_h = None

        if last_h is not None:
            new_kv = [
                (get_layer_kv(past_kv, l)[0][:, :, -1:, :],
                 get_layer_kv(past_kv, l)[1][:, :, -1:, :])
                for l in range(manager.num_layers)
            ]
            manager.on_token_generated(last_h, new_kv)
            # Online eviction keeps the cache bounded at budget_per_layer and
            # feeds the shadow buffer with recently-evicted middle tokens.
            past_kv = manager.maybe_evict_online(past_kv)

        # Check for EOS
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        # Tool-call pause every N tokens
        if tokens_generated % config.tool_call_interval == 0:
            duration_ms = sample_tool_duration(
                median_ms=config.duration_median_ms,
                min_ms=config.duration_min_ms,
                max_ms=config.duration_max_ms,
                rng=rng,
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
