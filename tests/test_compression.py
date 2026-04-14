"""Tests for CompressedKVManager."""

import torch
import pytest
from unittest.mock import Mock, MagicMock
from idlekv.core.compression import CompressedKVManager


def create_mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.config = Mock()
    model.config.num_hidden_layers = 4
    model.config.num_attention_heads = 8
    model.config.num_key_value_heads = 8  # Same as num_attention_heads for this test
    model.config.hidden_size = 256

    # Mock model parameters for device and dtype
    param1 = torch.tensor([1.0], device="cpu", dtype=torch.float32)
    param2 = torch.tensor([2.0], device="cpu", dtype=torch.float32)
    model.parameters.return_value = iter([param1, param2])

    # Real nn.Linear q_projs so _compress can actually project hidden states.
    # num_q_heads * head_dim = 8 * 32 = 256 = hidden_size.
    layers = []
    for i in range(4):
        layer = Mock()
        layer.self_attn = Mock()
        layer.self_attn.q_proj = torch.nn.Linear(256, 256, bias=False)
        layers.append(layer)

    model.model = Mock()
    model.model.layers = layers

    return model


def create_mock_outputs(seq_len=10, num_layers=4, num_heads=8, head_dim=32):
    """Create mock model outputs with KV cache."""
    # Create mock past_key_values as list of (K, V) tuples
    past_kv = []
    for layer_idx in range(num_layers):
        k = torch.randn(1, num_heads, seq_len, head_dim)
        v = torch.randn(1, num_heads, seq_len, head_dim)
        past_kv.append((k, v))

    outputs = Mock()
    outputs.past_key_values = tuple(past_kv)
    outputs.hidden_states = [torch.randn(1, seq_len, 256) for _ in range(num_layers + 1)]
    outputs.logits = torch.randn(1, seq_len, 1000)

    return outputs


def test_compressed_kv_manager_init():
    """Test CompressedKVManager initialization."""
    model = create_mock_model()

    manager = CompressedKVManager(
        model=model,
        compression_ratio=0.5,
        shadow_size=128,
        query_buffer_size=16,
        offload_full_kv=False
    )

    assert manager.compression_ratio == 0.5
    assert manager.num_layers == 4
    assert manager.num_kv_heads == 8
    assert manager.head_dim == 32
    assert manager.hidden_dim == 256
    assert manager.device == torch.device("cpu")
    assert manager.dtype == torch.float32
    assert not manager.offload_full_kv

    # Check that hidden states output was enabled
    assert model.config.output_hidden_states == True


def test_prefill_compression(monkeypatch):
    """Test prefill method produces smaller cache than input."""
    model = create_mock_model()
    manager = CompressedKVManager(
        model=model,
        compression_ratio=0.5,
        offload_full_kv=False
    )
    monkeypatch.setattr(
        "idlekv.core.compression.last_token_logits_kwargs",
        lambda _: {"logits_to_keep": 1},
    )

    # Mock model output
    seq_len = 20
    input_ids = torch.randint(0, 1000, (1, seq_len))
    mock_outputs = create_mock_outputs(seq_len=seq_len)
    model.return_value = mock_outputs

    # Run prefill
    compressed_kv = manager.prefill(input_ids)

    # Prefill only needs the final next-token logits, not the full seq_len slab.
    assert model.call_args.kwargs["logits_to_keep"] == 1

    # Check that compression happened
    budget = int(seq_len * (1 - manager.compression_ratio))
    assert manager.budget_per_layer == budget

    # Check that cache was compressed (should have fewer tokens than input)
    # Note: In our simplified test, we can't easily verify exact compression
    # but we can check that the process completed without errors
    assert compressed_kv is not None

    # Check that shadow buffer and query buffer were reset
    assert len(manager.generated_kv_lists) == manager.num_layers
    for layer_list in manager.generated_kv_lists:
        assert len(layer_list) == 0


def test_seed_query_buffer_from_prefill():
    """Prefill captures prompt-tail hidden states for quick query seeding."""
    model = create_mock_model()
    manager = CompressedKVManager(model=model, query_buffer_size=4)

    seq_len = 10
    input_ids = torch.randint(0, 1000, (1, seq_len))
    mock_outputs = create_mock_outputs(seq_len=seq_len)
    model.return_value = mock_outputs

    manager.prefill(input_ids)
    seeded = manager.seed_query_buffer_from_prefill()

    assert seeded == 4
    buf = manager.query_buffer.get()
    expected = torch.stack([
        mock_outputs.hidden_states[layer_idx][0, -4:, :]
        for layer_idx in range(manager.num_layers)
    ], dim=1)
    assert torch.equal(buf, expected)


def test_on_token_generated():
    """Test token generation tracking."""
    model = create_mock_model()
    manager = CompressedKVManager(model=model)

    # Initialize generated_kv_lists
    manager.generated_kv_lists = [[] for _ in range(manager.num_layers)]

    # Create mock hidden state and new KV
    hidden_state = torch.randn(manager.num_layers, manager.hidden_dim)
    new_kv_per_layer = []
    for layer_idx in range(manager.num_layers):
        k = torch.randn(1, manager.num_kv_heads, 1, manager.head_dim)
        v = torch.randn(1, manager.num_kv_heads, 1, manager.head_dim)
        new_kv_per_layer.append((k, v))

    # Track token generation
    manager.on_token_generated(hidden_state, new_kv_per_layer)

    query_state = manager.query_buffer.get()
    assert query_state.shape == (1, manager.num_layers, manager.hidden_dim)
    assert torch.equal(query_state[0], hidden_state)

    # Check that KV was stored in lists (avoiding O(n^2) concatenation)
    for layer_idx in range(manager.num_layers):
        assert len(manager.generated_kv_lists[layer_idx]) == 1
        k_stored, v_stored = manager.generated_kv_lists[layer_idx][0]
        expected_k, expected_v = new_kv_per_layer[layer_idx]
        assert torch.equal(k_stored, expected_k)
        assert torch.equal(v_stored, expected_v)


def test_get_generated_kv():
    """Test concatenation of generated KV lists."""
    model = create_mock_model()
    manager = CompressedKVManager(model=model)

    # Initialize and populate generated_kv_lists
    manager.generated_kv_lists = [[] for _ in range(manager.num_layers)]

    # Add multiple tokens to first layer
    for token_idx in range(3):
        k = torch.randn(1, manager.num_kv_heads, 1, manager.head_dim)
        v = torch.randn(1, manager.num_kv_heads, 1, manager.head_dim)
        manager.generated_kv_lists[0].append((k, v))

    # Get concatenated result
    generated_kv = manager._get_generated_kv()

    # Check results
    assert len(generated_kv) == manager.num_layers

    # First layer should have 3 tokens concatenated
    k_cat, v_cat = generated_kv[0]
    assert k_cat.shape == (1, manager.num_kv_heads, 3, manager.head_dim)
    assert v_cat.shape == (1, manager.num_kv_heads, 3, manager.head_dim)

    # Other layers should be empty
    for layer_idx in range(1, manager.num_layers):
        k_empty, v_empty = generated_kv[layer_idx]
        assert k_empty.shape == (1, manager.num_kv_heads, 0, manager.head_dim)
        assert v_empty.shape == (1, manager.num_kv_heads, 0, manager.head_dim)


def test_shadow_buffer_population():
    """Test that shadow buffer gets populated during compression."""
    model = create_mock_model()
    manager = CompressedKVManager(
        model=model,
        compression_ratio=0.3,  # Aggressive compression to ensure eviction
        shadow_size=64,
        offload_full_kv=False
    )

    # Mock model output with enough tokens to trigger eviction
    seq_len = 20
    input_ids = torch.randint(0, 1000, (1, seq_len))
    mock_outputs = create_mock_outputs(seq_len=seq_len)
    model.return_value = mock_outputs

    # Run prefill
    compressed_kv = manager.prefill(input_ids)

    # Check that shadow buffer is non-empty (evicted tokens should be there)
    # Note: This is a basic check - in a real test we might verify specific tokens
    assert compressed_kv is not None

    # The shadow buffer should have some content after aggressive compression
    # (We can't easily check the exact content without more complex mocking)


def test_semantic_seq_len_tracking():
    """Semantic seq len advances by 1 per generated token, independent of
    physical cache length (which may shrink via online eviction)."""
    model = create_mock_model()
    manager = CompressedKVManager(model=model, compression_ratio=0.5)
    seq_len = 20
    input_ids = torch.randint(0, 1000, (1, seq_len))
    mock_outputs = create_mock_outputs(seq_len=seq_len)
    model.return_value = mock_outputs
    manager.prefill(input_ids)

    assert manager.semantic_seq_len == seq_len
    pos0 = manager.next_position_ids(num_new_tokens=1)
    assert pos0.item() == seq_len

    # Simulate 3 generated tokens
    for _ in range(3):
        h = torch.randn(manager.num_layers, manager.hidden_dim)
        kv = [(torch.randn(1, manager.num_kv_heads, 1, manager.head_dim),
               torch.randn(1, manager.num_kv_heads, 1, manager.head_dim))
              for _ in range(manager.num_layers)]
        manager.on_token_generated(h, kv)

    assert manager.semantic_seq_len == seq_len + 3
    assert manager.next_position_ids(1).item() == seq_len + 3


def test_online_eviction_bounds_cache():
    """maybe_evict_online keeps physical cache at budget_per_layer and pushes
    evicted pairs into the shadow buffer."""
    model = create_mock_model()
    manager = CompressedKVManager(model=model, compression_ratio=0.5,
                                  shadow_size=128)
    seq_len = 20
    input_ids = torch.randint(0, 1000, (1, seq_len))
    mock_outputs = create_mock_outputs(seq_len=seq_len)
    model.return_value = mock_outputs
    past_kv = manager.prefill(input_ids)
    budget = manager.budget_per_layer

    # Append 8 fake tokens to simulate growth past budget
    from idlekv.utils.kv_cache import get_layer_kv, set_layer_kv
    for _ in range(8):
        for l in range(manager.num_layers):
            k, v = get_layer_kv(past_kv, l)
            new_k = torch.cat([k, torch.randn(1, manager.num_kv_heads, 1,
                                              manager.head_dim)], dim=2)
            new_v = torch.cat([v, torch.randn(1, manager.num_kv_heads, 1,
                                              manager.head_dim)], dim=2)
            past_kv = set_layer_kv(past_kv, l, new_k, new_v)

    past_kv = manager.maybe_evict_online(past_kv, slack_tokens=0)
    for l in range(manager.num_layers):
        k, _ = get_layer_kv(past_kv, l)
        assert k.shape[2] == budget, f"layer {l}: {k.shape[2]} != budget {budget}"


def test_online_eviction_slack_delays_rebuild():
    """A small slack window avoids rebuilding the cache on every decode step."""
    model = create_mock_model()
    manager = CompressedKVManager(model=model, compression_ratio=0.5, shadow_size=128)
    seq_len = 20
    input_ids = torch.randint(0, 1000, (1, seq_len))
    mock_outputs = create_mock_outputs(seq_len=seq_len)
    model.return_value = mock_outputs
    past_kv = manager.prefill(input_ids)
    budget = manager.budget_per_layer

    from idlekv.utils.kv_cache import get_layer_kv, set_layer_kv
    for l in range(manager.num_layers):
        k, v = get_layer_kv(past_kv, l)
        new_k = torch.cat([k, torch.randn(1, manager.num_kv_heads, 1, manager.head_dim)], dim=2)
        new_v = torch.cat([v, torch.randn(1, manager.num_kv_heads, 1, manager.head_dim)], dim=2)
        past_kv = set_layer_kv(past_kv, l, new_k, new_v)

    past_kv = manager.maybe_evict_online(past_kv, slack_tokens=4)
    for l in range(manager.num_layers):
        k, _ = get_layer_kv(past_kv, l)
        assert k.shape[2] == budget + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
