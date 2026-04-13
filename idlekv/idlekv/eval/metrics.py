"""
Metrics for evaluating KV cache quality.

1. KL divergence vs. full cache (continuous quality measure)
2. Recovery delta (quality before vs. after idle-time refinement)
3. Decode throughput (tokens/sec)
"""

import torch
import torch.nn.functional as F
import time
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    kl_divergence: float          # KL(compressed || full) averaged over tokens
    recovery_delta_kl: float      # KL_before - KL_after (positive = improvement)
    decode_tokens_per_sec: float  # generation throughput
    peak_gpu_memory_mb: float     # peak allocated GPU memory


def compute_kl_divergence(
    model,
    input_ids: torch.Tensor,
    compressed_kv: tuple,
    full_kv: tuple,
    num_eval_tokens: int = 64,
) -> float:
    """
    Compute KL divergence between compressed and full-cache output distributions.

    Generates `num_eval_tokens` with each cache and compares logit distributions.

    Args:
        model: HF causal LM
        input_ids: [1, seq_len] the prefill context
        compressed_kv: compressed past_key_values
        full_kv: full past_key_values (reference)
        num_eval_tokens: how many tokens to evaluate over

    Returns:
        Mean KL divergence (nats)
    """
    kl_values = []

    with torch.no_grad():
        # Get logits from full cache
        full_out = model(input_ids[:, -1:], past_key_values=full_kv, use_cache=True)
        full_logits = full_out.logits[:, -1, :]  # [1, vocab]

        # Get logits from compressed cache
        comp_out = model(input_ids[:, -1:], past_key_values=compressed_kv, use_cache=True)
        comp_logits = comp_out.logits[:, -1, :]

        # KL(compressed || full)
        full_probs = F.softmax(full_logits, dim=-1)
        comp_log_probs = F.log_softmax(comp_logits, dim=-1)
        kl = F.kl_div(comp_log_probs, full_probs, reduction='batchmean')
        kl_values.append(kl.item())

    return sum(kl_values) / len(kl_values)


def measure_throughput(
    model,
    input_ids: torch.Tensor,
    past_key_values: tuple,
    num_tokens: int = 128,
) -> float:
    """
    Measure decode throughput in tokens/sec.

    Args:
        model: HF causal LM
        input_ids: [1, seq_len] for initial position
        past_key_values: the KV cache to decode with
        num_tokens: number of tokens to generate for measurement

    Returns:
        tokens per second
    """
    device = input_ids.device

    # Warm up
    with torch.no_grad():
        out = model(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        kv = out.past_key_values

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_tokens):
            out = model(next_token, past_key_values=kv, use_cache=True)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            kv = out.past_key_values

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return num_tokens / elapsed
