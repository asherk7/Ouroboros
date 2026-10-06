"""Measure the KV-cache decode speedup on a trained checkpoint (metric 3).

Times ``Ouroboros.generate`` with ``use_cache=True`` vs ``use_cache=False`` on
the same prompts. Without a cache every new token re-runs the whole sequence
through every layer and every loop, so per-token cost grows with length; with
the cache each step processes one token. In a looped model the cache holds one
entry per loop iteration, so the saving is paid for in memory that grows with
``n_loops`` (see ``ARCHITECTURE.md`` section 5).

A stub: bodies raise ``NotImplementedError``.

Usage:
.. code-block:: bash

    python training/kv_speedup.py --ckpt checkpoints/final.pt
"""

from __future__ import annotations

import argparse


def time_generate(
    ckpt_path: str,
    use_cache: bool,
    prompt_len: int = 64,
    max_new_tokens: int = 256,
    n_loops: int = 8,
    repeats: int = 5,
) -> float:
    """Return decode throughput in generated tokens per second.

    Load the model from the checkpoint (configs are stored as dicts; rebuild
    with ``OuroborosConfig(**ckpt["model_cfg"])``). Use a fixed batch of
    prompts. Do one untimed warmup call, then time ``repeats`` calls to
    ``generate(..., use_cache=use_cache)``. On CUDA, call
    ``torch.cuda.synchronize()`` before reading the clock on both ends. Report
    the median.

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
    raise NotImplementedError


def main() -> None:
    """Parse ``--ckpt``, time both settings, print both rates and the speedup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    raise NotImplementedError


if __name__ == "__main__":
    main()
