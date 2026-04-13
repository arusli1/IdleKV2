"""Utilities for handling KV cache formats across different Transformers versions."""

import torch
from torch import Tensor
from typing import Union, List, Tuple, Any


def get_layer_kv(past_key_values: Any, layer_idx: int) -> Tuple[Tensor, Tensor]:
    """
    Get (keys, values) for a specific layer from past_key_values.

    Works with both DynamicCache (HF Transformers 4.40+) and legacy tuple format.

    Args:
        past_key_values: Either DynamicCache object or tuple of (key, value) tuples
        layer_idx: Layer index to retrieve

    Returns:
        Tuple of (keys, values) tensors of shape [batch, num_kv_heads, seq_len, head_dim]
    """
    # Check if it's a DynamicCache (has layers attribute)
    if hasattr(past_key_values, 'layers'):
        layer = past_key_values.layers[layer_idx]
        return layer.keys, layer.values

    # Legacy tuple format
    elif isinstance(past_key_values, (list, tuple)):
        return past_key_values[layer_idx]

    else:
        raise ValueError(f"Unsupported past_key_values type: {type(past_key_values)}")


def set_layer_kv(past_key_values: Any, layer_idx: int, keys: Tensor, values: Tensor) -> Any:
    """
    Set (keys, values) for a specific layer in past_key_values.

    For DynamicCache: mutates in-place and returns the same object.
    For tuple format: returns a new tuple with updated layer.

    Args:
        past_key_values: Either DynamicCache object or tuple of (key, value) tuples
        layer_idx: Layer index to update
        keys: New keys tensor
        values: New values tensor

    Returns:
        Updated past_key_values (same object for DynamicCache, new tuple for legacy)
    """
    # DynamicCache: mutate in-place
    if hasattr(past_key_values, 'layers'):
        # Ensure layer exists
        while len(past_key_values.layers) <= layer_idx:
            past_key_values.update(
                torch.empty(0, 0, 0, 0, device=keys.device, dtype=keys.dtype),
                torch.empty(0, 0, 0, 0, device=keys.device, dtype=keys.dtype),
                layer_idx=len(past_key_values.layers)
            )
        # Update the layer
        layer = past_key_values.layers[layer_idx]
        layer.keys = keys
        layer.values = values
        return past_key_values

    # Legacy tuple format: create new tuple
    elif isinstance(past_key_values, (list, tuple)):
        new_cache = list(past_key_values)
        new_cache[layer_idx] = (keys, values)
        return tuple(new_cache) if isinstance(past_key_values, tuple) else new_cache

    else:
        raise ValueError(f"Unsupported past_key_values type: {type(past_key_values)}")


def build_cache(layer_kvs: List[Tuple[Tensor, Tensor]], cache_class: Any = None) -> Any:
    """
    Build a cache object from a list of (keys, values) per layer.

    Args:
        layer_kvs: List of (keys, values) tuples, one per layer
        cache_class: Optional cache class to use. If None, tries to import DynamicCache

    Returns:
        Cache object (DynamicCache if available, otherwise tuple)
    """
    if not layer_kvs:
        return ()

    # Try to use DynamicCache if available
    if cache_class is None:
        try:
            from transformers.cache_utils import DynamicCache
            cache_class = DynamicCache
        except ImportError:
            # Fall back to tuple format
            return tuple(layer_kvs)

    if cache_class is not None:
        # Build DynamicCache
        cache = cache_class()
        for layer_idx, (keys, values) in enumerate(layer_kvs):
            cache.update(keys, values, layer_idx=layer_idx)
        return cache
    else:
        return tuple(layer_kvs)


def num_layers(past_key_values: Any) -> int:
    """
    Get the number of layers from past_key_values.

    Args:
        past_key_values: Either DynamicCache object or tuple of (key, value) tuples

    Returns:
        Number of layers in the cache
    """
    if hasattr(past_key_values, 'layers'):
        return len(past_key_values.layers)
    elif isinstance(past_key_values, (list, tuple)):
        return len(past_key_values)
    else:
        raise ValueError(f"Unsupported past_key_values type: {type(past_key_values)}")


def clone_cache(past_key_values: Any) -> Any:
    """
    Create a deep copy of past_key_values.

    Args:
        past_key_values: Cache to clone

    Returns:
        Cloned cache
    """
    if hasattr(past_key_values, 'layers'):
        # DynamicCache: clone each tensor
        try:
            from transformers.cache_utils import DynamicCache
            new_cache = DynamicCache()
            for layer_idx, layer in enumerate(past_key_values.layers):
                new_cache.update(layer.keys.clone(), layer.values.clone(), layer_idx=layer_idx)
            return new_cache
        except ImportError:
            # Fallback: extract as tuples and clone
            layer_kvs = [(layer.keys.clone(), layer.values.clone()) for layer in past_key_values.layers]
            return tuple(layer_kvs)
    elif isinstance(past_key_values, (list, tuple)):
        return tuple((k.clone(), v.clone()) for k, v in past_key_values)
    else:
        raise ValueError(f"Unsupported past_key_values type: {type(past_key_values)}")


def cache_size(past_key_values: Any, layer_idx: int = 0) -> int:
    """
    Get the sequence length (number of tokens) in the cache.

    Args:
        past_key_values: Cache object
        layer_idx: Layer to check (default: 0)

    Returns:
        Sequence length
    """
    keys, _ = get_layer_kv(past_key_values, layer_idx)
    return keys.shape[2]  # [batch, num_heads, seq_len, head_dim]


def empty_cache_like(past_key_values: Any, seq_len: int = 0) -> Any:
    """
    Create an empty cache with the same format and device as the input.

    Args:
        past_key_values: Reference cache to match format
        seq_len: Initial sequence length (default: 0)

    Returns:
        Empty cache
    """
    if hasattr(past_key_values, 'layers'):
        try:
            from transformers.cache_utils import DynamicCache
            return DynamicCache()
        except ImportError:
            return ()
    else:
        return ()