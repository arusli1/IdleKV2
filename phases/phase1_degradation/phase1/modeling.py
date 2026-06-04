"""Model and tokenizer loading helpers for the local HuggingFace backend."""

from __future__ import annotations

import importlib.util

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Keep loading logic in one place so callers don't have to repeat model-specific knobs.
def preferred_attention_implementation() -> str | None:
    """Prefer Flash Attention when it is installed, otherwise use the model default."""
    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return None


def load_tokenizer(model_dir: str):
    """Load the tokenizer and normalize padding so generation calls stay simple."""
    # Trust remote code so model repos with custom tokenizers load correctly.
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # Qwen can run without an explicit pad token, but setting one avoids
        # downstream edge cases in batched helper code.
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Use a large sentinel value to avoid premature truncation for long prompts.
    tokenizer.model_max_length = int(1e9)
    return tokenizer


def load_model(model_dir: str):
    """Load the causal LM with the dtype/device settings used in Phase 1."""
    # Centralize model defaults so experiment code doesn't have to remember them.
    kwargs = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "device_map": "auto",
    }
    attn_implementation = preferred_attention_implementation()
    if attn_implementation is not None:
        # Only opt in when the dependency is available; otherwise let
        # Transformers fall back to its standard attention path.
        kwargs["attn_implementation"] = attn_implementation
    # Load once, then keep the model in eval mode for consistent inference behavior.
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    model.eval()
    return model
