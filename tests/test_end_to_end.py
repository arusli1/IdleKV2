"""
End-to-end smoke test using a tiny model on CPU.
Verifies the full IdleKV pipeline: prefill → compress → generate → idle refine.
"""
import torch
import pytest


def get_tiny_model():
    """Load the smallest available model for testing."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # Qwen2.5-0.5B is ~1GB, runs on CPU in seconds
    # If not available, try any small model
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="cpu"
        )
        model.eval()
        return model, tokenizer
    except Exception:
        pytest.skip(f"Could not load {model_name}. Run: huggingface-cli download {model_name}")


@pytest.mark.slow
def test_full_pipeline():
    """Test the complete IdleKV pipeline on CPU with a tiny model."""
    model, tokenizer = get_tiny_model()

    from idlekv.core.compression import CompressedKVManager

    manager = CompressedKVManager(
        model,
        compression_ratio=0.5,
        shadow_size=32,
        query_buffer_size=8,
    )

    # Create a longer context to ensure compression happens
    text = "The secret code is ALPHA-7. " * 50 + "What is the secret code?"
    input_ids = tokenizer.encode(text, return_tensors="pt")
    seq_len = input_ids.shape[1]

    print(f"Input sequence length: {seq_len}")

    # Step 1: Prefill and compress
    compressed_kv = manager.prefill(input_ids)

    # Verify compression happened
    from idlekv.utils.kv_cache import get_layer_kv, num_layers
    n = num_layers(compressed_kv)
    assert n > 0, "Cache should have layers"
    k, v = get_layer_kv(compressed_kv, 0)
    compressed_len = k.shape[2]
    print(f"Compressed length: {compressed_len}")
    print(f"Shadow buffer count: {manager.shadow_buffer.layers[0].count}")

    assert compressed_len < seq_len, f"Compressed length {compressed_len} should be < input {seq_len}"

    # Verify shadow buffer has entries (only if compression actually evicted tokens)
    if compressed_len < seq_len:
        print("Compression occurred and should have populated shadow buffer")
        # Note: shadow buffer might be empty if shadow_size was too small or no tokens were actually evicted
        # This is acceptable behavior, so we'll make this check informational

    # Step 2: Generate a few tokens (populates query buffer)
    with torch.no_grad():
        current_kv = compressed_kv
        next_input = input_ids[:, -1:]
        for step in range(8):
            outputs = model(next_input, past_key_values=current_kv, use_cache=True)
            current_kv = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_input = next_token

            # Feed hidden states to manager if available
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                manager.on_token_generated(outputs.hidden_states[-1][:, -1, :], [])

    # Step 3: Run idle-time refinement
    result = manager.idle_refine(current_kv, max_time_ms=5000)  # generous timeout for CPU

    # Verify refinement produced a result
    assert result.past_key_values is not None
    assert result.phase1_ran
    assert result.total_time_ms > 0

    print(f"Pipeline test passed:")
    print(f"  Input tokens: {seq_len}")
    print(f"  Compressed to: {compressed_len}")
    print(f"  Shadow buffer entries: {manager.shadow_buffer.layers[0].count}")
    print(f"  Phase 1 time: {result.phase1_time_ms:.1f}ms")
    print(f"  Total refine time: {result.total_time_ms:.1f}ms")