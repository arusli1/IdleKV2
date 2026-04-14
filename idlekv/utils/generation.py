"""Helpers for deterministic generation configuration."""

from copy import deepcopy


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
