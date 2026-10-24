"""Measure the KV-cache decode speedup on a trained checkpoint (metric 3).

Times ``Ouroboros.generate`` with ``use_cache=True`` vs ``use_cache=False`` on
the same prompts. Without a cache every new token re-runs the whole sequence
through every layer and every loop, so per-token cost grows with length; with
the cache each step processes one token. In a looped model the cache holds one
entry per loop iteration, so the saving is paid for in memory that grows with
``n_loops`` (see ``ARCHITECTURE.md`` section 5).

Usage:
.. code-block:: bash

    python training/kv_speedup.py --ckpt checkpoints/final.pt
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from train import load_model


def sync(device: str) -> None:
    """Wait for queued GPU work, so the clock measures the work, not the launch."""
    if device == "cuda":
        torch.cuda.synchronize()


def time_generate(
    ckpt_path: str,
    use_cache: bool,
    prompt_len: int = 64,
    max_new_tokens: int = 256,
    n_loops: int = 8,
    repeats: int = 5,
) -> float:
    """Return decode throughput in generated tokens per second.

    Loads the model from the checkpoint and uses one fixed random prompt
    (batch 1: a single user decoding). One untimed warmup call, then ``repeats``
    timed calls to ``generate(..., use_cache=use_cache)``; on CUDA,
    ``torch.cuda.synchronize()`` runs before reading the clock on both ends.

    Args:
        ckpt_path: Checkpoint from ``training/train.py``.
        use_cache: Whether ``generate`` uses the KV cache.
        prompt_len: Prompt length in tokens.
        max_new_tokens: Tokens to generate per call.
        n_loops: Loop count for every step.
        repeats: Timed calls; the median is returned.

    Returns:
        Median tokens per second.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(ckpt_path, device)
    # content doesn't matter for timing; the work per token is the same
    gen = torch.Generator().manual_seed(0)
    prompt = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), generator=gen)
    prompt = prompt.to(device)

    def run() -> None:
        model.generate(prompt, max_new_tokens, n_loops=n_loops, use_cache=use_cache)

    run()  # warmup: CUDA kernel selection and allocator growth happen here
    rates = []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        run()
        sync(device)
        rates.append(max_new_tokens / (time.perf_counter() - t0))
    # median: one slow call (a GC pause, another process) doesn't skew it
    return statistics.median(rates)


def main() -> None:
    """Parse ``--ckpt``, time both settings, print both rates and the speedup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n-loops", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    kw = dict(
        n_loops=args.n_loops, max_new_tokens=args.max_new_tokens, repeats=args.repeats
    )
    cached = time_generate(args.ckpt, use_cache=True, **kw)
    print(f"with cache:    {cached:.1f} tokens/s", flush=True)
    uncached = time_generate(args.ckpt, use_cache=False, **kw)
    print(f"without cache: {uncached:.1f} tokens/s")
    print(f"speedup: {cached / uncached:.2f}x ({args.n_loops} loops)")


if __name__ == "__main__":
    main()
