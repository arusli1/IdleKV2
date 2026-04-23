"""Budgeted kvpress policies plus logging hooks used in Phase 1."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from kvpress import BasePress
from kvpress.presses.key_rerotation_press import KeyRerotationPress
from kvpress.utils import extract_keys_and_values, get_prerope_query_states
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from .tracing import EvictionTraceRecorder


@dataclass
class BudgetedLoggingPress(BasePress):
    """Common logging/compression wrapper around a kvpress policy."""

    target_size: int
    recorder: EvictionTraceRecorder | None = None
    query_log_tokens: int = 64
    rerotate_keys: bool = False

    def _compute_scores(self, module, hidden_states, keys, values, attentions, kwargs) -> torch.Tensor:
        """Return per-token keep scores for one attention layer."""
        raise NotImplementedError

    def _compute_query_vectors(self, module, hidden_states, kwargs) -> torch.Tensor:
        """Capture the most recent query vectors so trace files explain the eviction decision."""
        # Limit logging to a recent window so traces are compact and interpretable.
        q_window = min(hidden_states.shape[1], self.query_log_tokens)
        query_states = get_prerope_query_states(module, hidden_states[:, -q_window:])
        if "position_embeddings" in kwargs:
            # Apply RoPE if it was provided to the attention module.
            cos, sin = kwargs["position_embeddings"]
            cos, sin = cos[:, -q_window:], sin[:, -q_window:]
            query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))
        return query_states.squeeze(0)

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        """Score the current cache, keep the top tokens, and optionally log the decision."""
        # Early exit when the cache already fits the budget; still emit trace metadata.
        k_len = keys.shape[2]
        if self.target_size <= 0 or k_len <= self.target_size:
            # Nothing to compress yet, but still record a full-keep trace so the
            # attribution pipeline sees a uniform file structure.
            if self.recorder is not None:
                kept_mask = torch.ones_like(keys[..., 0], dtype=torch.bool)
                kept_indices = torch.arange(k_len, device=keys.device).view(1, 1, k_len).expand_as(kept_mask)
                scores = kept_mask.to(keys.dtype)
                self.recorder.record(
                    layer_idx=module.layer_idx,
                    scores=scores,
                    kept_mask=kept_mask,
                    kept_indices=kept_indices,
                    query_vectors=self._compute_query_vectors(module, hidden_states, kwargs),
                    input_kv_length=k_len,
                    kept_kv_length=k_len,
                )
            return keys, values

        # Ask the specific policy to score every cached token, then keep the top
        # `target_size` positions in original order so positions stay meaningful.
        scores = self._compute_scores(module, hidden_states, keys, values, attentions, kwargs)
        n_kept = min(self.target_size, k_len)
        kept_indices = scores.topk(n_kept, dim=-1).indices
        kept_indices = torch.sort(kept_indices, dim=2).values
        kept_mask = torch.zeros_like(scores, dtype=torch.bool)
        kept_mask.scatter_(2, kept_indices, True)

        # Persist scores and selection details for offline analysis.
        if self.recorder is not None:
            self.recorder.record(
                layer_idx=module.layer_idx,
                scores=scores,
                kept_mask=kept_mask,
                kept_indices=kept_indices,
                query_vectors=self._compute_query_vectors(module, hidden_states, kwargs),
                input_kv_length=k_len,
                kept_kv_length=n_kept,
            )

        # Apply the token selection to the cached keys/values.
        if self.rerotate_keys:
            # StreamingLLM-style policies need key rerotation because keeping a
            # non-contiguous subset changes the RoPE-relative geometry.
            keys = KeyRerotationPress.rerotate_keys(module, kept_indices, keys)
        else:
            gather_idx = kept_indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
            keys = keys.gather(2, gather_idx).contiguous()
        # Values are always gathered directly because they do not need rerotation.
        gather_idx = kept_indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        values = values.gather(2, gather_idx).contiguous()
        return keys, values

    def forward_hook(self, module, input, kwargs, output):
        """Intercept a layer forward pass, compress its cache, and write it back."""
        del input
        # Pull out the per-layer cache, apply compression, and write back in-place.
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        keys, values = extract_keys_and_values(cache, module.layer_idx)
        keys, values = self.compress(module, hidden_states, keys, values, output[1], kwargs)
        cache.layers[module.layer_idx].keys = keys
        cache.layers[module.layer_idx].values = values
        return output


@dataclass
class BudgetedSnapKVPress(BudgetedLoggingPress):
    """SnapKV-style policy that scores old tokens by recent attention demand."""

    window_size: int = 64
    kernel_size: int = 5

    def _compute_scores(self, module, hidden_states, keys, values, attentions, kwargs) -> torch.Tensor:
        """Approximate SnapKV's salience score for each cached token position."""
        bsz, num_key_value_heads, k_len, _ = keys.shape
        if hidden_states.shape[1] <= self.window_size:
            raise ValueError(
                f"SnapKV requires q_len > window_size; got q_len={hidden_states.shape[1]}, window={self.window_size}"
            )

        # Establish the head grouping used by this model configuration.
        num_heads = module.config.num_attention_heads
        num_key_value_groups = num_heads // num_key_value_heads

        if attentions is not None:
            # If the model already exposed attention weights, reuse them directly.
            attn_weights = attentions[..., -self.window_size :, : -self.window_size]
        else:
            # Otherwise rebuild the recent-window attention weights manually from
            # the cached keys and the current query states.
            # Build query/key representations for just the most recent window.
            query_states = get_prerope_query_states(module, hidden_states[:, -self.window_size :])
            cos, sin = kwargs["position_embeddings"]
            cos, sin = cos[:, -self.window_size :], sin[:, -self.window_size :]
            query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))
            key_states = repeat_kv(keys, num_key_value_groups)
            # Compute causal attention weights over the pre-window cache.
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(module.head_dim)
            attention_mask = torch.ones_like(attn_weights) * float("-inf")
            attention_mask = torch.triu(attention_mask, diagonal=k_len - self.window_size + 1)
            attn_weights += attention_mask
            # Normalize and drop the window tokens so scores align to older cache entries.
            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = attn_weights[..., :-self.window_size]

        # Average across the recent query window, then smooth locally so nearby
        # important tokens do not compete too harshly with one another.
        scores = attn_weights.mean(dim=-2)
        scores = F.avg_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, k_len - self.window_size).mean(2)
        # Pad the score array so all cache positions have a defined score.
        return F.pad(scores, (0, self.window_size), value=scores.max().item())


@dataclass
class BudgetedStreamingLLMPress(BudgetedLoggingPress):
    """StreamingLLM-style keep policy: preserve sink tokens plus a recent tail."""

    n_sink: int = 4

    def __post_init__(self) -> None:
        # StreamingLLM keeps a non-contiguous subset, so keys must be rerotated.
        self.rerotate_keys = True

    def _compute_scores(self, module, hidden_states, keys, values, attentions, kwargs) -> torch.Tensor:
        """Score sink tokens and the recent window as keepers, everything else as discardable."""
        del module, hidden_states, values, attentions, kwargs
        # Allocate a binary keep/discard mask in score form.
        k_len = keys.shape[2]
        scores = torch.ones_like(keys[..., 0])
        recent_budget = max(self.target_size - self.n_sink, 0)
        recent_start = max(k_len - recent_budget, self.n_sink)
        # Zero out the middle span so only sinks and the most recent tokens remain.
        if recent_start > self.n_sink:
            scores[:, :, self.n_sink:recent_start] = 0
        return scores


def build_press(
    algorithm: str,
    *,
    budget: int,
    recorder: EvictionTraceRecorder | None,
    query_log_tokens: int,
) -> BudgetedLoggingPress:
    """Instantiate the requested compression policy with Phase 1 logging attached."""
    # Normalize the selector and return the matching policy with shared logging behavior.
    algorithm = algorithm.lower()
    if algorithm == "snapkv":
        return BudgetedSnapKVPress(target_size=budget, recorder=recorder, query_log_tokens=query_log_tokens)
    if algorithm == "streamingllm":
        return BudgetedStreamingLLMPress(target_size=budget, recorder=recorder, query_log_tokens=query_log_tokens)
    raise ValueError(f"Unsupported algorithm: {algorithm}")
