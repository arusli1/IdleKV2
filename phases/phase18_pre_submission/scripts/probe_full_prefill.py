"""W2 supplement: measure full 32K-token prefill wall-clock on the
evaluation GPU. This is V in the abstract claim 'RepairKV's repair
operation runs in a fraction of the wall-clock cost of full-prefix
recompute.'

Loads Qwen2.5-7B-Instruct and times one forward pass on a synthetic
32K-token sequence, with FlashAttention-2 (and FA-3 if available).
Reports p95 across N trials.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def probe(
    model_dir: Path,
    *,
    seq_lens: list[int] = (32_768,),
    n_trials: int = 5,
    n_warmup: int = 2,
    attn_impl: str = "flash_attention_2",
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[prefill-probe] loading model from {model_dir} (attn={attn_impl})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=attn_impl,
    )
    model.eval()

    rows = []
    for seq_len in seq_lens:
        # Synthetic input ids: random tokens. The model just runs prefill.
        torch.manual_seed(17)
        input_ids = torch.randint(
            0, tokenizer.vocab_size, (1, seq_len), device="cuda", dtype=torch.long
        )

        timings_ms = []
        for trial in range(n_warmup + n_trials):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(input_ids=input_ids, use_cache=True)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if trial >= n_warmup:
                timings_ms.append(elapsed_ms)
            print(f"[prefill-probe] seq_len={seq_len} trial={trial} elapsed={elapsed_ms:.1f}ms", flush=True)
        timings_ms_sorted = sorted(timings_ms)
        p50 = timings_ms_sorted[len(timings_ms_sorted) // 2]
        p95_idx = max(0, int(0.95 * (len(timings_ms_sorted) - 1)))
        p95 = timings_ms_sorted[p95_idx]
        rows.append({
            "seq_len": int(seq_len),
            "attn_impl": attn_impl,
            "p50_ms": float(p50),
            "p95_ms": float(p95),
            "all_ms": [float(t) for t in timings_ms],
        })
        print(f"[prefill-probe] seq_len={seq_len} p50={p50:.1f}ms p95={p95:.1f}ms", flush=True)
    return {"rows": rows, "model_dir": str(model_dir), "n_trials": n_trials}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("/home/ubuntu/IdleKV/models/Qwen2.5-7B-Instruct"))
    parser.add_argument("--seq-lens", type=str, default="16384,32768")
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--attn-impl", type=str, default="flash_attention_2")
    parser.add_argument("--out-json", type=Path, default=Path("phases/phase18_pre_submission/results/w2/full_prefill.json"))
    args = parser.parse_args()
    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    result = probe(args.model_dir, seq_lens=seq_lens, n_trials=args.n_trials, n_warmup=args.n_warmup, attn_impl=args.attn_impl)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as fp:
        json.dump(result, fp, indent=2)
    print(f"[prefill-probe] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
