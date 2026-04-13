"""GPU and CPU memory tracking."""

import torch


def gpu_memory_summary(device: int = 0) -> dict:
    """Get current GPU memory usage."""
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / 1e6,
        "reserved_mb": torch.cuda.memory_reserved(device) / 1e6,
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1e6,
    }


def reset_peak_memory(device: int = 0):
    torch.cuda.reset_peak_memory_stats(device)


def kv_cache_size_mb(past_key_values: tuple) -> float:
    """Compute total size of a KV cache in MB."""
    total = 0
    for k, v in past_key_values:
        total += k.nelement() * k.element_size()
        total += v.nelement() * v.element_size()
    return total / 1e6
