"""FlashAttention's core algorithm, written in plain PyTorch to learn it.

This is a learning exercise, not a fast path: the model trains with
``F.scaled_dot_product_attention``, which already dispatches to a fused kernel.
The goal is to reproduce what that kernel does so the FlashAttention papers
(Dao et al., 2022; Dao, 2023) make concrete sense.

The idea: standard attention materializes the full ``(T, S)`` score matrix in
GPU main memory. FlashAttention never does. It walks over K/V in blocks and
keeps, for each query row, a running max ``m``, a running softmax denominator
``l``, and a running output accumulator ``o``. When a new block of scores
arrives with a larger max, the old ``l`` and ``o`` are rescaled by
``exp(m_old - m_new)`` (the "online softmax" trick), so the final result equals
the exact softmax without ever holding all scores at once. On a GPU those blocks
live in fast on-chip SRAM; here they are just slices, but the math is identical.
"""

from __future__ import annotations

import torch

__all__ = ["tiled_attention"]


def tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    block_size: int = 64,
) -> torch.Tensor:
    """Exact attention computed block by block with an online softmax.

    Algorithm, for each block of queries (outer loop) and each block of keys and
    values (inner loop):

    1. ``s = q_blk @ k_blk^T * head_dim**-0.5``; apply the causal mask inside the
       block if needed (and skip key blocks entirely in the future).
    2. ``m_new = max(m, s.max(-1))``.
    3. ``p = exp(s - m_new)``; ``l = l * exp(m - m_new) + p.sum(-1)``.
    4. ``o = o * exp(m - m_new) + p @ v_blk``; ``m = m_new``.

    After the inner loop, the block's output is ``o / l``. Keep ``m``, ``l`` and
    ``o`` in float32.

    Args:
        q: Queries of shape ``(B, H, T, head_dim)``.
        k: Keys of shape ``(B, H, S, head_dim)`` (KV heads already expanded to
            ``H`` for GQA).
        v: Values of shape ``(B, H, S, head_dim)``.
        causal: Mask so query ``i`` attends only to keys ``j <= i``; assumes
            ``T == S``.
        block_size: Rows per query block and per key/value block.

    Returns:
        Attention output of shape ``(B, H, T, head_dim)`` in ``q.dtype``. Must
        match ``F.scaled_dot_product_attention(q, k, v, is_causal=causal)``
        within floating-point tolerance.
    """
    raise NotImplementedError
