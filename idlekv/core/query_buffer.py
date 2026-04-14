"""
Query buffer: stores the last N per-layer hidden states produced during
generation.

These are used as "queries" during Phase 1 re-scoring. We store hidden states
(not projected queries) because different layers have different Q projections.
Phase 1 projects them per-layer at re-scoring time.

Stored on GPU. ~8MB for buffer_size=32, num_layers=32, hidden_dim=4096, fp16.
"""

import torch
from typing import Optional


class QueryBuffer:
    """
    Rolling buffer of recent per-layer hidden states from generation.

    Usage:
        qbuf = QueryBuffer(
            buffer_size=32,
            num_layers=32,
            hidden_dim=4096,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        # During generation, after each token:
        qbuf.append(hidden_states)  # [num_layers, hidden_dim]
        # At idle time:
        recent_h = qbuf.get(layer_idx=0)  # [count, hidden_dim]
    """

    def __init__(
        self,
        buffer_size: int = 32,
        num_layers: int = 1,
        hidden_dim: int = 4096,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
    ):
        self.buffer_size = buffer_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.buffer = torch.zeros(
            buffer_size,
            num_layers,
            hidden_dim,
            device=device,
            dtype=dtype,
        )
        self.write_idx = 0
        self.count = 0

    def _normalize_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize inputs to [num_layers, hidden_dim]."""
        h = hidden_states

        if isinstance(h, (list, tuple)):
            h = torch.stack([
                state.squeeze(0) if state.dim() == 2 and state.shape[0] == 1 else state
                for state in h
            ], dim=0)
        elif h.dim() == 3 and h.shape[0] == 1:
            h = h.squeeze(0)
        elif h.dim() == 1 and self.num_layers == 1:
            h = h.unsqueeze(0)
        elif h.dim() == 2 and self.num_layers == 1 and h.shape[0] != 1:
            raise ValueError(
                "Single-layer QueryBuffer expects [hidden_dim] or [1, hidden_dim] inputs."
            )

        if h.dim() != 2 or h.shape[0] != self.num_layers or h.shape[1] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden states with shape [{self.num_layers}, {self.hidden_dim}], "
                f"got {tuple(h.shape)}"
            )

        return h

    def _ordered_buffer(self) -> torch.Tensor:
        """Return the valid buffer contents in oldest-to-newest order."""
        if self.count < self.buffer_size:
            return self.buffer[:self.count]
        return torch.cat([
            self.buffer[self.write_idx:],
            self.buffer[:self.write_idx],
        ], dim=0)

    def append(self, hidden_states: torch.Tensor):
        """
        Append per-layer hidden states to the buffer.

        Args:
            hidden_states: [num_layers, hidden_dim] or [1, num_layers, hidden_dim]
                containing the hidden state that feeds each layer for the most
                recently ingested/generated token. For a single-layer buffer,
                [hidden_dim] and [1, hidden_dim] are also accepted.
        """
        h = self._normalize_hidden_states(hidden_states)
        self.buffer[self.write_idx] = h
        self.write_idx = (self.write_idx + 1) % self.buffer_size
        self.count = min(self.count + 1, self.buffer_size)

    def get(self, layer_idx: Optional[int] = None) -> torch.Tensor:
        """
        Get all valid buffered hidden states.

        Returns:
            If `layer_idx` is provided, returns [count, hidden_dim] for that
            layer. Otherwise returns [count, num_layers, hidden_dim], ordered
            oldest-to-newest. For single-layer buffers, the default output is
            [count, hidden_dim] for backward compatibility.
        """
        ordered = self._ordered_buffer().clone()
        if layer_idx is not None:
            return ordered[:, layer_idx, :]
        if self.num_layers == 1:
            return ordered[:, 0, :]
        return ordered

    def clear(self):
        self.buffer.zero_()
        self.write_idx = 0
        self.count = 0
