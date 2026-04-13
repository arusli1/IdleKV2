"""
Wrappers for kvpress baselines.

Provides a unified interface for running SnapKV, H2O, StreamingLLM, and
sync refresh (RefreshKV-style) through kvpress.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import kvpress
    from kvpress import (
        SnapKVPress,
        ObservedAttentionPress,
        StreamingLLMPress,
        KnormPress,
        DecodingPress,
    )
    KVPRESS_AVAILABLE = True
except ImportError:
    KVPRESS_AVAILABLE = False


def get_press(method: str, compression_ratio: float = 0.5, **kwargs):
    """
    Get a kvpress Press object by method name.

    Args:
        method: one of "snapkv", "h2o", "streaminglm", "knorm"
        compression_ratio: fraction of tokens to EVICT (0.5 = keep 50%)

    Returns:
        kvpress Press object
    """
    if not KVPRESS_AVAILABLE:
        raise ImportError("kvpress not installed. Run: pip install kvpress")

    press_map = {
        "snapkv": SnapKVPress,
        "h2o": ObservedAttentionPress,
        "streaminglm": StreamingLLMPress,
        "knorm": KnormPress,
    }

    if method not in press_map:
        raise ValueError(f"Unknown method '{method}'. Available: {list(press_map.keys())}")

    return press_map[method](compression_ratio=compression_ratio)


def run_with_press(model, tokenizer, input_ids, press, max_new_tokens=128):
    """
    Run generation with a kvpress press applied.

    Returns:
        dict with 'output_ids', 'past_key_values', 'text'
    """
    with torch.no_grad(), press(model):
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return {"output_ids": outputs, "text": text}


class SyncRefreshBaseline:
    """
    RefreshKV-style synchronous refresh during generation.

    Every `stride` decode steps, recomputes full attention and rebuilds
    the compressed cache. This adds latency during generation (unlike IdleKV).

    Used as Baseline 6 to isolate IdleKV's scheduling contribution.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        compression_ratio: float = 0.5,
        stride: int = 15,
    ):
        self.model = model
        self.compression_ratio = compression_ratio
        self.stride = stride
        self.full_kv_cpu = None  # stored during prefill

    def prefill(self, input_ids: torch.Tensor) -> tuple:
        """Run prefill, store full KV, return compressed version."""
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)

        full_kv = outputs.past_key_values
        self.full_kv_cpu = [
            (k.cpu(), v.cpu()) for k, v in full_kv
        ]

        # Compress using SnapKV-style scoring
        # (Reuse IdleKV's compression for fair comparison)
        from idlekv.core.compression import CompressedKVManager
        manager = CompressedKVManager(self.model, self.compression_ratio, shadow_size=0)
        compressed = manager._compress(full_kv, input_ids,
                                        int(input_ids.shape[1] * (1 - self.compression_ratio)))
        return compressed

    def maybe_refresh(self, past_key_values: tuple, step: int) -> tuple:
        """
        If step is a multiple of stride, do a synchronous full-attention refresh.
        This is the operation that adds latency (unlike IdleKV).
        """
        if step % self.stride != 0 or step == 0:
            return past_key_values

        if self.full_kv_cpu is None:
            return past_key_values

        # Refresh: load full KV, re-score, re-select
        # (Same as IdleKV Phase 2 but synchronous)
        from idlekv.core.phase2_refresh import phase2_refresh
        from idlekv.core.query_buffer import QueryBuffer

        # Create a dummy query buffer from the last token
        # (In a real implementation, would track hidden states)
        # For now, this is a simplified version
        return past_key_values  # TODO: implement full sync refresh
