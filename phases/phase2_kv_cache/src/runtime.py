"""Shared runtime helpers for Phase 2 experiments and tests."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from .kv_utils import PositionTrackedCache, inject_kv, save_kv, load_kv, sequence_length, slice_kv, to_dynamic_cache, to_tuple_cache

PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_ROOT.parents[1]
MODEL_DIR = REPO_ROOT / "models" / "Qwen2.5-7B-Instruct"
RESULTS_DIR = PHASE_ROOT / "results"
HEATMAP_DIR = RESULTS_DIR / "phase2_attention_heatmaps"


def ensure_results_dirs() -> None:
    """Create the result directories expected by the phase instructions."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)


def model_device(model) -> torch.device:
    """Return the device the model uses for forward passes."""
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def load_tokenizer(model_dir: Path = MODEL_DIR):
    """Load the local Qwen tokenizer with stable padding settings."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = int(1e9)
    return tokenizer


def load_model(model_dir: Path = MODEL_DIR):
    """Load local Qwen on the single available GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 2 live tests require a CUDA device.")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    return model


def inspect_environment(model, tokenizer) -> dict[str, Any]:
    """Record the live model/cache layout that the Phase 2 code targets."""
    config = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    device = model_device(model)
    probe_ids = tokenizer("Hello world", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        probe_output = model(input_ids=probe_ids, use_cache=True)
    probe_cache = probe_output.past_key_values
    tuple_cache = to_tuple_cache(probe_cache)
    return {
        "model_dir": str(MODEL_DIR),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": int(config.hidden_size // config.num_attention_heads),
        "config_torch_dtype": str(getattr(config, "torch_dtype", "unknown")),
        "cache_runtime_type": type(probe_cache).__name__,
        "layer0_key_shape": list(tuple_cache[0][0].shape),
        "layer0_value_shape": list(tuple_cache[0][1].shape),
        "layer0_dtype": str(tuple_cache[0][0].dtype),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
    }


def _json_default(value: Any) -> Any:
    """Serialize numpy and path objects into JSON-friendly values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable.")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON artifact with consistent formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def make_exact_length_input_ids(
    tokenizer,
    *,
    target_tokens: int,
    device: torch.device | str,
    base_text: str | None = None,
) -> torch.Tensor:
    """Construct an exact token-length 1xN tensor by repeating one filler block."""
    source_text = base_text or (
        "Cache verification filler text about maintenance windows, diagnostics, telemetry, and routine operations. "
    )
    block = tokenizer(source_text, add_special_tokens=False).input_ids
    if not block:
        raise ValueError("Base text produced zero tokens.")
    repeats = (target_tokens // len(block)) + 2
    tokens = (block * repeats)[:target_tokens]
    return torch.tensor([tokens], dtype=torch.long, device=device)


def prefill_cache(model, input_ids: torch.Tensor):
    """Run a prefill pass and return a normalized tuple-style cache."""
    with torch.no_grad():
        output = model(input_ids=input_ids.to(model_device(model)), use_cache=True)
    return to_tuple_cache(output.past_key_values)


def _infer_logical_position_base(cache: object) -> int:
    """Infer the next absolute position id after a cache."""
    if isinstance(cache, PositionTrackedCache):
        if not cache.positions:
            return 0
        return int(max(cache.positions) + 1)
    return int(sequence_length(cache))


def _infer_dense_cache_base(cache: object) -> int:
    """Infer the next dense cache slot after a cache."""
    return int(len(cache)) if isinstance(cache, PositionTrackedCache) else int(sequence_length(cache))


def resume_forward(
    model,
    input_ids: torch.Tensor,
    cache: object,
    *,
    logical_position_base: int | None = None,
    dense_cache_position_base: int | None = None,
    logits_to_keep: int = 1,
):
    """Resume a forward pass from an existing cache with explicit positions."""
    device = model_device(model)
    tokens = input_ids.to(device)
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    logical_base = _infer_logical_position_base(cache) if logical_position_base is None else int(logical_position_base)
    dense_base = _infer_dense_cache_base(cache) if dense_cache_position_base is None else int(dense_cache_position_base)
    seq_len = int(tokens.shape[1])
    position_ids = torch.arange(logical_base, logical_base + seq_len, device=device).unsqueeze(0)
    cache_position = torch.arange(dense_base, dense_base + seq_len, device=device)
    model_cache = to_dynamic_cache(cache, config=model.config)
    with torch.no_grad():
        outputs = model(
            input_ids=tokens,
            past_key_values=model_cache,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
            logits_to_keep=logits_to_keep,
        )
    return outputs


def generate_from_cache(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    cache: object,
    *,
    logical_position_base: int,
    dense_cache_position_base: int,
    max_new_tokens: int,
) -> str:
    """Greedy-generate a continuation from an existing cache."""
    device = model_device(model)
    prompt_ids = prompt_ids.to(device)
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    position_ids = torch.arange(
        logical_position_base,
        logical_position_base + prompt_ids.shape[1],
        device=device,
    ).unsqueeze(0)
    cache_position = torch.arange(
        dense_cache_position_base,
        dense_cache_position_base + prompt_ids.shape[1],
        device=device,
    )
    model_cache = to_dynamic_cache(cache, config=model.config)
    stop_ids = model.generation_config.eos_token_id
    if not isinstance(stop_ids, list):
        stop_ids = [stop_ids]

    generated: list[torch.Tensor] = []
    with torch.no_grad():
        outputs = model(
            input_ids=prompt_ids,
            past_key_values=model_cache,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
            logits_to_keep=1,
        )
        next_token = outputs.logits[0, -1].argmax()
        generated.append(next_token)
        current_position = position_ids[:, -1:] + 1
        current_cache_position = cache_position[-1:] + 1

        for _ in range(max_new_tokens - 1):
            outputs = model(
                input_ids=generated[-1].reshape(1, 1),
                past_key_values=model_cache,
                position_ids=current_position,
                cache_position=current_cache_position,
                use_cache=True,
            )
            next_token = outputs.logits[0, -1].argmax()
            generated.append(next_token)
            if int(next_token.item()) in stop_ids:
                break
            current_position = current_position + 1
            current_cache_position = current_cache_position + 1

    return tokenizer.decode(torch.stack(generated), skip_special_tokens=True)


def max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    """Compute the maximum absolute logit difference between two tensors."""
    return float((left - right).abs().max().item())


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    """Find one subsequence inside a token list."""
    needle = list(needle)
    haystack = list(haystack)
    if not needle:
        raise ValueError("Needle sequence must be non-empty.")
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return start
    raise ValueError("Subsequence not found in token list.")


def build_fact_retrieval_case(
    tokenizer,
    *,
    device: torch.device | str,
    context_tokens: int = 1536,
    access_code: str = "99273",
) -> dict[str, Any]:
    """Build a context/query pair that depends on one recoverable fact span."""
    fact_text = f"The sealed vault access code is {access_code}. Memorize this exact five digit code.\n"
    filler_text = (
        "Archive note: routine maintenance, calibration logs, and equipment status updates with no access code.\n"
    )
    query_text = "Question: What is the sealed vault access code? Respond with only the digits.\nAnswer:"
    fact_tokens = tokenizer(fact_text, add_special_tokens=False).input_ids
    filler_tokens = tokenizer(filler_text, add_special_tokens=False).input_ids
    if context_tokens <= len(fact_tokens) + 16:
        raise ValueError("Context token budget is too small for the fact retrieval case.")

    prefix_len = (context_tokens - len(fact_tokens)) // 2
    suffix_len = context_tokens - prefix_len - len(fact_tokens)
    prefix_ids = (filler_tokens * ((prefix_len // len(filler_tokens)) + 2))[:prefix_len]
    suffix_ids = (filler_tokens * ((suffix_len // len(filler_tokens)) + 2))[:suffix_len]
    context_list = prefix_ids + fact_tokens + suffix_ids
    query_ids = tokenizer(query_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    fact_start = len(prefix_ids)
    fact_positions = list(range(fact_start, fact_start + len(fact_tokens)))
    code_tokens = tokenizer(access_code, add_special_tokens=False).input_ids
    code_offset = find_subsequence(fact_tokens, code_tokens)
    code_positions = list(range(fact_start + code_offset, fact_start + code_offset + len(code_tokens)))

    return {
        "access_code": access_code,
        "context_ids": torch.tensor([context_list], dtype=torch.long, device=device),
        "query_ids": query_ids,
        "fact_positions": fact_positions,
        "code_positions": code_positions,
    }


def run_round_trip_identity(model, tokenizer) -> dict[str, Any]:
    """Run the P2 acceptance test: save/load must preserve continuation logits."""
    ensure_results_dirs()
    device = model_device(model)
    context_ids = make_exact_length_input_ids(tokenizer, target_tokens=1024, device=device)
    full_cache = prefill_cache(model, context_ids)
    round_trip_dir = RESULTS_DIR / "tmp_round_trip_cache"
    continuation = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long, device=device)

    reference = resume_forward(
        model,
        continuation,
        full_cache,
        logical_position_base=context_ids.shape[1],
        dense_cache_position_base=context_ids.shape[1],
    )
    save_kv(full_cache, str(round_trip_dir))
    loaded_cache = load_kv(str(round_trip_dir), device=str(device))
    resumed = resume_forward(
        model,
        continuation,
        loaded_cache,
        logical_position_base=context_ids.shape[1],
        dense_cache_position_base=context_ids.shape[1],
    )

    result = {
        "context_tokens": int(context_ids.shape[1]),
        "continuation_token_id": int(tokenizer.eos_token_id),
        "max_abs_logit_diff": max_abs_diff(reference.logits, resumed.logits),
        "dtype": str(full_cache[0][0].dtype),
        "seq_len": int(sequence_length(full_cache)),
        "saved_cache_dir": round_trip_dir,
        "pass": max_abs_diff(reference.logits, resumed.logits) < 1e-3,
    }
    write_json(RESULTS_DIR / "phase2_round_trip.json", result)
    return result


def run_selective_injection(model, tokenizer) -> dict[str, Any]:
    """Evict one fact span, show the degraded continuation, then inject it back."""
    ensure_results_dirs()
    device = model_device(model)
    case = build_fact_retrieval_case(tokenizer, device=device)
    context_ids = case["context_ids"]
    query_ids = case["query_ids"]
    full_cache = prefill_cache(model, context_ids)
    tracked_full = PositionTrackedCache(full_cache, list(range(context_ids.shape[1])))

    keep_positions = [index for index in range(context_ids.shape[1]) if index not in case["fact_positions"]]
    evicted_cache = slice_kv(tracked_full, keep_positions)
    evicted_fragment = slice_kv(tracked_full, case["fact_positions"])
    if not isinstance(evicted_cache, PositionTrackedCache) or not isinstance(evicted_fragment, PositionTrackedCache):
        raise RuntimeError("Tracked cache slicing unexpectedly returned an untracked cache.")

    reference = resume_forward(
        model,
        query_ids,
        full_cache,
        logical_position_base=context_ids.shape[1],
        dense_cache_position_base=context_ids.shape[1],
    )
    degraded = resume_forward(
        model,
        query_ids,
        evicted_cache,
        logical_position_base=context_ids.shape[1],
        dense_cache_position_base=len(evicted_cache),
    )
    restored_cache = inject_kv(evicted_cache, evicted_fragment, case["fact_positions"])
    restored = resume_forward(
        model,
        query_ids,
        restored_cache,
        logical_position_base=context_ids.shape[1],
        dense_cache_position_base=len(restored_cache),
    )

    reference_text = generate_from_cache(
        model,
        tokenizer,
        query_ids,
        full_cache,
        logical_position_base=context_ids.shape[1],
        dense_cache_position_base=context_ids.shape[1],
        max_new_tokens=8,
    ).strip()
    degraded_text = generate_from_cache(
        model,
        tokenizer,
        query_ids,
        evicted_cache,
        logical_position_base=context_ids.shape[1],
        dense_cache_position_base=len(evicted_cache),
        max_new_tokens=8,
    ).strip()
    restored_text = generate_from_cache(
        model,
        tokenizer,
        query_ids,
        restored_cache,
        logical_position_base=context_ids.shape[1],
        dense_cache_position_base=len(restored_cache),
        max_new_tokens=8,
    ).strip()

    result = {
        "access_code": case["access_code"],
        "context_tokens": int(context_ids.shape[1]),
        "query_tokens": int(query_ids.shape[1]),
        "removed_fact_token_count": len(case["fact_positions"]),
        "reference_text": reference_text,
        "degraded_text": degraded_text,
        "restored_text": restored_text,
        "reference_vs_degraded_max_abs_logit_diff": max_abs_diff(reference.logits, degraded.logits),
        "reference_vs_restored_max_abs_logit_diff": max_abs_diff(reference.logits, restored.logits),
        "recovered_text_match": reference_text == restored_text,
        "degraded_text_differs": reference_text != degraded_text,
    }
    write_json(RESULTS_DIR / "phase2_injection.json", result)
    return result


def _profile_cuda_transfer(copy_fn, *, trials: int) -> list[float]:
    """Time a repeated CPU/GPU tensor transfer operation."""
    durations: list[float] = []
    for _ in range(trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        moved = copy_fn()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        del moved
    return durations


def _summarize_durations(durations: Sequence[float], *, bytes_copied: int) -> dict[str, float]:
    """Convert one timing vector into p50/p90 statistics."""
    arr = np.array(list(durations), dtype=np.float64)
    p50 = float(np.median(arr))
    p90 = float(np.percentile(arr, 90))
    gb = bytes_copied / 1e9
    return {
        "p50_ms": p50 * 1000.0,
        "p90_ms": p90 * 1000.0,
        "throughput_gb_per_s_at_p50": (gb / p50) if p50 > 0 else 0.0,
    }


def run_transfer_latency(model, tokenizer) -> dict[str, Any]:
    """Measure CPU<->GPU KV transfer time for several repaired-token budgets."""
    ensure_results_dirs()
    device = model_device(model)
    full_context = make_exact_length_input_ids(tokenizer, target_tokens=32768, device=device)
    full_cache = prefill_cache(model, full_context)

    results: dict[str, Any] = {
        "context_tokens": int(full_context.shape[1]),
        "test_sizes": {},
    }
    for n_tokens in [100, 500, 1000, 2000, 5000]:
        fragment_gpu = slice_kv(full_cache, slice(0, n_tokens))
        if isinstance(fragment_gpu, PositionTrackedCache):
            raise RuntimeError("Unexpected tracked cache during latency profiling.")
        fragment_cpu = tuple(
            (
                key.detach().to("cpu").pin_memory(),
                value.detach().to("cpu").pin_memory(),
            )
            for key, value in fragment_gpu
        )
        bytes_copied = sum((key.numel() + value.numel()) * key.element_size() for key, value in fragment_gpu)

        evict_times = _profile_cuda_transfer(
            lambda: tuple((key.to("cpu"), value.to("cpu")) for key, value in fragment_gpu),
            trials=20,
        )
        restore_times = _profile_cuda_transfer(
            lambda: tuple((key.to(device, non_blocking=True), value.to(device, non_blocking=True)) for key, value in fragment_cpu),
            trials=20,
        )

        results["test_sizes"][str(n_tokens)] = {
            "bytes_copied": bytes_copied,
            "evict": _summarize_durations(evict_times, bytes_copied=bytes_copied),
            "restore": _summarize_durations(restore_times, bytes_copied=bytes_copied),
        }

    write_json(RESULTS_DIR / "phase2_transfer_latency.json", results)
    return results


def _hot_colormap(values: np.ndarray) -> np.ndarray:
    """Simple hot-style color map for attention heatmaps."""
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(3.0 * values, 0.0, 1.0)
    green = np.clip(3.0 * values - 1.0, 0.0, 1.0)
    blue = np.clip(3.0 * values - 2.0, 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def save_attention_heatmap(
    matrix: np.ndarray,
    path: Path,
    *,
    title: str,
    markers: dict[str, int] | None = None,
    max_display_size: int = 1536,
) -> None:
    """Render one attention matrix as a PNG heatmap using PIL only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(matrix, dtype=np.float32)
    vmax = float(np.percentile(matrix, 99.5))
    if vmax <= 0:
        vmax = float(matrix.max()) if matrix.size else 1.0
    scaled = np.power(np.clip(matrix / max(vmax, 1e-12), 0.0, 1.0), 0.35)
    rgb = (_hot_colormap(scaled) * 255.0).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    seq_len = matrix.shape[0]
    display_size = min(max_display_size, seq_len)
    if image.size != (display_size, display_size):
        image = image.resize((display_size, display_size), resample=Image.Resampling.BILINEAR)

    top_margin = 54
    bottom_margin = 24
    canvas = Image.new("RGB", (display_size, display_size + top_margin + bottom_margin), color=(16, 16, 16))
    canvas.paste(image, (0, top_margin))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), title, fill=(240, 240, 240))
    draw.text((10, display_size + top_margin + 4), "x=key position, y=query position", fill=(180, 180, 180))

    if markers:
        for label, original_pos in markers.items():
            x = int(round((original_pos / max(seq_len - 1, 1)) * (display_size - 1)))
            draw.line((x, top_margin, x, top_margin + display_size), fill=(120, 220, 255), width=2)
            draw.text((min(x + 4, display_size - 120), top_margin + 4), label, fill=(120, 220, 255))

    canvas.save(path)


def build_vt4hop_like_sequence(
    tokenizer,
    *,
    total_tokens: int,
    device: torch.device | str,
) -> dict[str, Any]:
    """Construct a VT-like prompt with hop markers at fixed depths."""
    filler_tokens = tokenizer(
        "Background article text about archives, inventories, logistics, reports, and operations.\n",
        add_special_tokens=False,
    ).input_ids
    hop_segments = {
        "hop1": tokenizer("Hop 1: Orion points to Cedar.\n", add_special_tokens=False).input_ids,
        "hop2": tokenizer("Hop 2: Cedar points to Lattice.\n", add_special_tokens=False).input_ids,
        "hop3": tokenizer("Hop 3: Lattice points to Harbor.\n", add_special_tokens=False).input_ids,
        "hop4": tokenizer("Hop 4: Harbor points to 72814.\n", add_special_tokens=False).input_ids,
    }
    query_tokens = tokenizer(
        "Question: Starting from Orion, what final code do the hops end at?\nAnswer:",
        add_special_tokens=False,
    ).input_ids
    target_positions = {
        "hop1": int(total_tokens * 0.12),
        "hop2": int(total_tokens * 0.37),
        "hop3": int(total_tokens * 0.62),
        "hop4": int(total_tokens * 0.84),
    }

    tokens: list[int] = []
    markers: dict[str, int] = {}
    for label in ["hop1", "hop2", "hop3", "hop4"]:
        target = target_positions[label]
        if target < len(tokens):
            raise ValueError("Hop placement overran the available token budget.")
        filler_needed = target - len(tokens)
        tokens.extend((filler_tokens * ((filler_needed // len(filler_tokens)) + 2))[:filler_needed])
        markers[label] = len(tokens)
        tokens.extend(hop_segments[label])

    remaining = total_tokens - len(tokens) - len(query_tokens)
    if remaining < 0:
        raise ValueError("VT-like sequence exceeded the requested token budget.")
    tokens.extend((filler_tokens * ((remaining // len(filler_tokens)) + 2))[:remaining])
    markers["query"] = len(tokens)
    tokens.extend(query_tokens)
    return {
        "input_ids": torch.tensor([tokens], dtype=torch.long, device=device),
        "markers": markers,
    }


def compute_average_attention_matrix(
    model,
    input_ids: torch.Tensor,
    *,
    layer_indices: Sequence[int],
    query_chunk_size: int = 256,
) -> np.ndarray:
    """Compute a mean-over-heads, mean-over-layers attention heatmap."""
    device = model_device(model)
    tokens = input_ids.to(device)
    seq_len = int(tokens.shape[1])
    key_positions = torch.arange(seq_len, device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    with torch.no_grad():
        outputs = model.model(input_ids=tokens, use_cache=False, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states for attention extraction.")

    averaged = np.zeros((seq_len, seq_len), dtype=np.float32)
    for layer_idx in layer_indices:
        layer = model.model.layers[layer_idx]
        layer_input = hidden_states[layer_idx]
        normalized = layer.input_layernorm(layer_input)
        cos, sin = model.model.rotary_emb(normalized, position_ids)
        attention = layer.self_attn
        hidden_shape = (*normalized.shape[:-1], -1, attention.head_dim)
        query = attention.q_proj(normalized).view(hidden_shape).transpose(1, 2)
        key = attention.k_proj(normalized).view(hidden_shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        key = repeat_kv(key, attention.num_key_value_groups)

        for start in range(0, seq_len, query_chunk_size):
            end = min(seq_len, start + query_chunk_size)
            q_chunk = query[:, :, start:end, :]
            logits = torch.matmul(q_chunk, key.transpose(2, 3)) * attention.scaling
            query_positions = torch.arange(start, end, device=device)[:, None]
            causal_mask = key_positions[None, :] > query_positions
            logits = logits.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            weights = torch.softmax(logits.float(), dim=-1)
            averaged[start:end] += (
                weights.mean(dim=1).squeeze(0).detach().cpu().numpy() / len(layer_indices)
            )
            del q_chunk, logits, weights

        del normalized, query, key

    return averaged


def generate_attention_heatmaps(model, tokenizer) -> dict[str, Any]:
    """Produce the 4K/8K attention heatmap artifacts required by Phase 2."""
    ensure_results_dirs()
    device = model_device(model)
    layer_indices = [0, model.config.num_hidden_layers // 2, model.config.num_hidden_layers - 1]
    artifacts: dict[str, Any] = {
        "layer_indices": layer_indices,
        "files": {},
    }

    for total_tokens, filename in [(4096, "layer_avg_4k.png"), (8192, "layer_avg_8k.png")]:
        sequence = build_vt4hop_like_sequence(tokenizer, total_tokens=total_tokens, device=device)
        matrix = compute_average_attention_matrix(
            model,
            sequence["input_ids"],
            layer_indices=layer_indices,
            query_chunk_size=256,
        )
        output_path = HEATMAP_DIR / filename
        save_attention_heatmap(matrix, output_path, title=f"Average attention heatmap at {total_tokens} tokens")
        artifacts["files"][filename] = {
            "path": output_path,
            "markers": sequence["markers"],
            "tokens": total_tokens,
        }
        if total_tokens == 8192:
            marked_path = HEATMAP_DIR / "vt4hop_hop_positions_marked.png"
            save_attention_heatmap(
                matrix,
                marked_path,
                title="VT-like attention heatmap with hop markers at 8K",
                markers=sequence["markers"],
            )
            artifacts["files"]["vt4hop_hop_positions_marked.png"] = {
                "path": marked_path,
                "markers": sequence["markers"],
                "tokens": total_tokens,
            }

    write_json(RESULTS_DIR / "phase2_heatmaps.json", artifacts)
    return artifacts


def write_markdown_report(summary: dict[str, Any]) -> None:
    """Render a short human-readable result summary beside the JSON artifacts."""
    def sanitize(text: str) -> str:
        return text.replace("\\", "\\\\").replace("`", "\\`").replace("\n", "\\n")

    report = f"""# Phase 2 Run Report

## Environment

- Model dir: `{summary['environment']['model_dir']}`
- Layers: `{summary['environment']['num_hidden_layers']}`
- KV heads: `{summary['environment']['num_key_value_heads']}`
- Cache runtime type: `{summary['environment']['cache_runtime_type']}`

## Round-trip identity

- Pass: `{summary['round_trip']['pass']}`
- Max abs logit diff: `{summary['round_trip']['max_abs_logit_diff']:.6g}`

## Selective injection

- Reference text: `{sanitize(summary['injection']['reference_text'])}`
- Degraded text: `{sanitize(summary['injection']['degraded_text'])}`
- Restored text: `{sanitize(summary['injection']['restored_text'])}`
- Restored text matches reference: `{summary['injection']['recovered_text_match']}`
- Reference vs degraded max logit diff: `{summary['injection']['reference_vs_degraded_max_abs_logit_diff']:.6g}`
- Reference vs restored max logit diff: `{summary['injection']['reference_vs_restored_max_abs_logit_diff']:.6g}`

## Transfer latency

- Sizes profiled: `{", ".join(sorted(summary['transfer_latency']['test_sizes'].keys(), key=int))}`

## Heatmaps

- `phase2_attention_heatmaps/layer_avg_4k.png`
- `phase2_attention_heatmaps/layer_avg_8k.png`
- `phase2_attention_heatmaps/vt4hop_hop_positions_marked.png`
"""
    (RESULTS_DIR / "phase2_run_report.md").write_text(report, encoding="utf-8")


__all__ = [
    "HEATMAP_DIR",
    "MODEL_DIR",
    "RESULTS_DIR",
    "build_fact_retrieval_case",
    "build_vt4hop_like_sequence",
    "compute_average_attention_matrix",
    "ensure_results_dirs",
    "generate_attention_heatmaps",
    "generate_from_cache",
    "inspect_environment",
    "load_model",
    "load_tokenizer",
    "make_exact_length_input_ids",
    "max_abs_diff",
    "model_device",
    "prefill_cache",
    "resume_forward",
    "run_round_trip_identity",
    "run_selective_injection",
    "run_transfer_latency",
    "save_attention_heatmap",
    "write_json",
    "write_markdown_report",
]
