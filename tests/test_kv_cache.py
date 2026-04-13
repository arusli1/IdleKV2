"""Tests for KV cache utility functions."""

import pytest
import torch
from idlekv.utils.kv_cache import (
    get_layer_kv,
    set_layer_kv,
    build_cache,
    num_layers,
    clone_cache,
    cache_size,
)


def test_tuple_format():
    """Test utilities with legacy tuple format."""
    # Create test KV cache as list of (K, V) tuples
    k1 = torch.randn(1, 4, 10, 32)  # [batch, heads, seq_len, head_dim]
    v1 = torch.randn(1, 4, 10, 32)
    k2 = torch.randn(1, 4, 10, 32)
    v2 = torch.randn(1, 4, 10, 32)

    kv_cache = [(k1, v1), (k2, v2)]

    # Test get_layer_kv
    k_out, v_out = get_layer_kv(kv_cache, 0)
    assert torch.equal(k_out, k1)
    assert torch.equal(v_out, v1)

    # Test num_layers
    assert num_layers(kv_cache) == 2

    # Test cache_size
    assert cache_size(kv_cache, 0) == 10

    # Test set_layer_kv
    k_new = torch.randn(1, 4, 5, 32)
    v_new = torch.randn(1, 4, 5, 32)
    updated_cache = set_layer_kv(kv_cache, 0, k_new, v_new)

    k_check, v_check = get_layer_kv(updated_cache, 0)
    assert torch.equal(k_check, k_new)
    assert torch.equal(v_check, v_new)

    # Original should be unchanged (new tuple returned)
    k_orig, v_orig = get_layer_kv(kv_cache, 0)
    assert torch.equal(k_orig, k1)
    assert torch.equal(v_orig, v1)


def test_dynamic_cache_format():
    """Test utilities with DynamicCache format (if available)."""
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        pytest.skip("DynamicCache not available in this transformers version")

    # Create DynamicCache
    cache = DynamicCache()
    k1 = torch.randn(1, 4, 10, 32)
    v1 = torch.randn(1, 4, 10, 32)
    k2 = torch.randn(1, 4, 8, 32)
    v2 = torch.randn(1, 4, 8, 32)

    cache.update(k1, v1, layer_idx=0)
    cache.update(k2, v2, layer_idx=1)

    # Test get_layer_kv
    k_out, v_out = get_layer_kv(cache, 0)
    assert torch.equal(k_out, k1)
    assert torch.equal(v_out, v1)

    # Test num_layers
    assert num_layers(cache) == 2

    # Test cache_size
    assert cache_size(cache, 0) == 10
    assert cache_size(cache, 1) == 8

    # Test set_layer_kv (should mutate in place)
    k_new = torch.randn(1, 4, 12, 32)
    v_new = torch.randn(1, 4, 12, 32)
    updated_cache = set_layer_kv(cache, 0, k_new, v_new)

    # Should return same object
    assert updated_cache is cache

    # Should be updated
    k_check, v_check = get_layer_kv(cache, 0)
    assert torch.equal(k_check, k_new)
    assert torch.equal(v_check, v_new)


def test_build_cache():
    """Test building cache from layer list."""
    k1 = torch.randn(1, 4, 10, 32)
    v1 = torch.randn(1, 4, 10, 32)
    k2 = torch.randn(1, 4, 8, 32)
    v2 = torch.randn(1, 4, 8, 32)

    layer_kvs = [(k1, v1), (k2, v2)]

    # Build cache
    cache = build_cache(layer_kvs)

    # Should work regardless of format
    assert num_layers(cache) == 2

    k_out, v_out = get_layer_kv(cache, 0)
    assert torch.equal(k_out, k1)
    assert torch.equal(v_out, v1)

    k_out, v_out = get_layer_kv(cache, 1)
    assert torch.equal(k_out, k2)
    assert torch.equal(v_out, v2)


def test_clone_cache():
    """Test cloning cache."""
    k1 = torch.randn(1, 4, 10, 32)
    v1 = torch.randn(1, 4, 10, 32)
    k2 = torch.randn(1, 4, 8, 32)
    v2 = torch.randn(1, 4, 8, 32)

    original = [(k1, v1), (k2, v2)]
    cloned = clone_cache(original)

    # Should have same data
    k_orig, v_orig = get_layer_kv(original, 0)
    k_clone, v_clone = get_layer_kv(cloned, 0)
    assert torch.equal(k_orig, k_clone)
    assert torch.equal(v_orig, v_clone)

    # But different tensors (should be cloned)
    assert k_orig is not k_clone
    assert v_orig is not v_clone


def test_empty_cache():
    """Test edge cases with empty caches."""
    empty_cache = []
    assert num_layers(empty_cache) == 0

    built = build_cache([])
    assert num_layers(built) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])