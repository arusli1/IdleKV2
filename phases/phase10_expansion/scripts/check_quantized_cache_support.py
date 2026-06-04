#!/usr/bin/env python3
"""Check whether real Transformers quantized-cache backends are usable."""

from __future__ import annotations

import importlib
import inspect

import transformers


def main() -> int:
    print(f"transformers={transformers.__version__}")
    quantized_cache = getattr(transformers, "QuantizedCache", None)
    if quantized_cache is None:
        print("QuantizedCache=missing")
        return 1

    print(f"QuantizedCache={quantized_cache}")
    print(f"QuantizedCache_signature={inspect.signature(quantized_cache)}")
    usable_backend = False
    backend_imports = {
        "quanto": "optimum.quanto",
        "hqq": "hqq",
    }
    for backend, module_name in backend_imports.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            print(f"backend={backend} status=missing error={type(exc).__name__}: {exc}")
            continue
        usable_backend = True
        print(f"backend={backend} status=available version={getattr(module, '__version__', 'unknown')}")

    if not usable_backend:
        print("action=install_quantized_cache_backend_before_real_cache_smoke")
        return 2

    print("action=run_tiny_quantized_cache_generation_smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
