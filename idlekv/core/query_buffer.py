"""
Query buffer: stores the last N hidden states produced during generation.

These are used as "queries" during Phase 1 re-scoring. We store hidden states
(not projected queries) because different layers have different Q projections.
Phase 1 projects them per-layer at re-scoring time.

Stored on GPU. ~16MB for buffer_size=32, hidden_dim=4096, fp16.
"""

import torch


class QueryBuffer:
    """
    Rolling buffer of recent hidden states from generation.

    Usage:
        qbuf = QueryBuffer(buffer_size=32, hidden_dim=4096, device="cuda" if torch.cuda.is_available() else "cpu")
        # During generation, after each token:
        qbuf.append(hidden_states)  # [1, hidden_dim]
        # At idle time:
        recent_h = qbuf.get()  # [count, hidden_dim]
    """

    def __init__(
        self,
        buffer_size: int = 32,
        hidden_dim: int = 4096,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
    ):
        self.buffer_size = buffer_size
        self.hidden_dim = hidden_dim
        self.buffer = torch.zeros(buffer_size, hidden_dim, device=device, dtype=dtype)
        self.write_idx = 0
        self.count = 0

    def append(self, hidden_state: torch.Tensor):
        """
        Append a hidden state to the buffer.

        Args:
            hidden_state: [1, hidden_dim] or [hidden_dim] — the last-layer
                          hidden state for the most recently generated token.
        """
        h = hidden_state.squeeze(0) if hidden_state.dim() == 2 else hidden_state
        self.buffer[self.write_idx] = h
        self.write_idx = (self.write_idx + 1) % self.buffer_size
        self.count = min(self.count + 1, self.buffer_size)

    def get(self) -> torch.Tensor:
        """
        Get all valid buffered hidden states.

        Returns:
            [count, hidden_dim] tensor of recent hidden states, ordered
            oldest-to-newest.
        """
        if self.count < self.buffer_size:
            return self.buffer[:self.count].clone()
        # Buffer is full and has wrapped — reorder to oldest-first
        return torch.cat([
            self.buffer[self.write_idx:],
            self.buffer[:self.write_idx],
        ], dim=0).clone()

    def clear(self):
        self.buffer.zero_()
        self.write_idx = 0
        self.count = 0
