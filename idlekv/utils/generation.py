"""Helpers for deterministic generation configuration and efficient logits."""

from copy import deepcopy
from functools import lru_cache
import inspect


_SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "typical_p",
    "epsilon_cutoff",
    "eta_cutoff",
    "penalty_alpha",
)


def greedy_generation_config(model):
    """
    Return a generation config sanitized for greedy decoding.

    Some instruct-tuned checkpoints ship with sampling defaults in their saved
    generation config. Passing `do_sample=False` to `generate()` is correct, but
    Transformers will still warn about sampling-only fields such as
    `temperature` and `top_p`. Clearing those fields keeps benchmark logs clean
    without changing decoding semantics.
    """
    generation_config = deepcopy(model.generation_config)
    generation_config.do_sample = False
    for field in _SAMPLING_FIELDS:
        if hasattr(generation_config, field):
            setattr(generation_config, field, None)
    return generation_config


@lru_cache(maxsize=None)
def _supports_logits_to_keep(model_cls: type) -> bool:
    """Return whether this model class accepts `logits_to_keep` in forward()."""
    try:
        return "logits_to_keep" in inspect.signature(model_cls.forward).parameters
    except (AttributeError, TypeError, ValueError):
        return False


def last_token_logits_kwargs(model) -> dict:
    """
    Return kwargs that restrict LM-head logits to the final token when supported.

    On recent Transformers builds this avoids materializing the full
    `[batch, seq_len, vocab]` slab during long prefills when callers only need
    the next-token logits from the last position.
    """
    if _supports_logits_to_keep(type(model)):
        return {"logits_to_keep": 1}
    return {}
