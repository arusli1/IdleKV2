"""
Shadow buffer: per-layer FIFO ring buffer storing recently evicted KV pairs.

When SnapKV (or any scorer) evicts tokens, instead of discarding them, we push
them into this buffer. Phase 1 re-scoring pulls candidates from here.

Design: one buffer per layer. Each buffer stores up to `max_size` KV pairs.
When full, oldest entries are overwritten (FIFO). Keys and values are stored
as contiguous tensors for efficient batched attention scoring.
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class LayerBuffer:
    """Buffer for a single layer's evicted KV pairs."""
    keys: torch.Tensor      # [num_heads, current_size, head_dim]
    values: torch.Tensor    # [num_heads, current_size, head_dim]
    write_idx: int          # next write position (wraps around)
    count: int              # number of valid entries (up to max_size)
    max_size: int


class ShadowBuffer:
    """
    Per-layer FIFO ring buffer for evicted KV pairs.

    Usage:
        buf = ShadowBuffer(num_layers=32, max_size=256, num_kv_heads=8, head_dim=128, device="cuda")
        buf.push(layer_idx=0, keys=evicted_k, values=evicted_v)
        shadow_k, shadow_v = buf.get(layer_idx=0)
    """

    def __init__(
        self,
        num_layers: int,
        max_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.max_size = max_size
        self.layers: list[LayerBuffer] = []

        for _ in range(num_layers):
            self.layers.append(LayerBuffer(
                keys=torch.zeros(num_kv_heads, max_size, head_dim, device=device, dtype=dtype),
                values=torch.zeros(num_kv_heads, max_size, head_dim, device=device, dtype=dtype),
                write_idx=0,
                count=0,
                max_size=max_size,
            ))

    def push(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor):
        """
        Push evicted KV pairs into the buffer for a given layer.

        Args:
            layer_idx: which transformer layer
            keys: [num_kv_heads, num_evicted, head_dim]
            values: [num_kv_heads, num_evicted, head_dim]
        """
        buf = self.layers[layer_idx]
        num_evicted = keys.shape[1]

        if num_evicted == 0:
            return

        if num_evicted >= buf.max_size:
            # More evicted than buffer can hold — keep the most recent
            buf.keys.copy_(keys[:, -buf.max_size:, :])
            buf.values.copy_(values[:, -buf.max_size:, :])
            buf.write_idx = 0
            buf.count = buf.max_size
            return

        # Write into ring buffer, handling wrap-around
        space_before_wrap = buf.max_size - buf.write_idx
        if num_evicted <= space_before_wrap:
            buf.keys[:, buf.write_idx:buf.write_idx + num_evicted, :] = keys
            buf.values[:, buf.write_idx:buf.write_idx + num_evicted, :] = values
        else:
            # Split write across wrap boundary
            buf.keys[:, buf.write_idx:, :] = keys[:, :space_before_wrap, :]
            buf.values[:, buf.write_idx:, :] = values[:, :space_before_wrap, :]
            remainder = num_evicted - space_before_wrap
            buf.keys[:, :remainder, :] = keys[:, space_before_wrap:, :]
            buf.values[:, :remainder, :] = values[:, space_before_wrap:, :]

        buf.write_idx = (buf.write_idx + num_evicted) % buf.max_size
        buf.count = min(buf.count + num_evicted, buf.max_size)

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get all valid shadow KV pairs for a layer.

        Returns:
            keys: [num_kv_heads, count, head_dim]
            values: [num_kv_heads, count, head_dim]
        """
        buf = self.layers[layer_idx]
        if buf.count == 0:
            return buf.keys[:, :0, :], buf.values[:, :0, :]
        return buf.keys[:, :buf.count, :].clone(), buf.values[:, :buf.count, :].clone()

    def clear(self, layer_idx: Optional[int] = None):
        """Clear buffer for a specific layer or all layers."""
        layers = [layer_idx] if layer_idx is not None else range(self.num_layers)
        for idx in layers:
            buf = self.layers[idx]
            buf.keys.zero_()
            buf.values.zero_()
            buf.write_idx = 0
            buf.count = 0

    @property
    def memory_bytes(self) -> int:
        """Total GPU memory used by shadow buffers."""
        total = 0
        for buf in self.layers:
            total += buf.keys.nelement() * buf.keys.element_size()
            total += buf.values.nelement() * buf.values.element_size()
        return total
