"""Attention for Ouroboros.

:class:`GQAttention` (Grouped Query Attention, Ainslie et al., 2023) is the one
attention mechanism used in the Prelude, Recurrent, and Coda stages. It has two
interchangeable paths that must agree numerically:

1. **SDPA path (used for training):** ``F.scaled_dot_product_attention``.
   PyTorch picks the fastest fused kernel available on the GPU (FlashAttention
   where supported; on a T4 usually the memory-efficient kernel, which uses the
   same tiling idea). Nothing to configure.
2. **Manual path (the reference):** plain ``softmax(Q K^T / sqrt(d)) V`` with an
   explicit causal mask. Slow, but every step is visible. Selected with
   ``use_sdpa=False``; the tests use it to check the SDPA path and
   :func:`~ouroboros.tiled_attention.tiled_attention`.

The ``forward`` signature ``(x, freqs_cis, causal=True, kv_cache=None,
cache_key="default")`` is the attention contract consumed by
:class:`~ouroboros.block.TransformerBlock`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import OuroborosConfig


class GQAttention(nn.Module):
    """Grouped Query Attention with QK-norm and a per-layer KV cache.

    GQA (https://arxiv.org/abs/2305.13245) uses fewer key/value heads than query
    heads: each KV head is shared by ``groups = n_heads // n_kv_heads`` query
    heads. That shrinks the KV cache (and the K/V projections) by ``groups``. It
    sits between multi-head attention (``n_kv_heads == n_heads``) and multi-query
    attention (``n_kv_heads == 1``). In a looped model every loop iteration
    stores its own K/V, so the cache grows with the loop count and the saving
    matters more than in a plain transformer.

    Projections (all ``bias=False``), with ``head_dim = dim // n_heads``:

    - ``wq``: ``dim -> n_heads * head_dim``
    - ``wk``: ``dim -> n_kv_heads * head_dim``
    - ``wv``: ``dim -> n_kv_heads * head_dim``
    - ``wo``: ``n_heads * head_dim -> dim``

    Order of operations in ``forward``:

    1. Project to q, k, v and reshape to heads ``(B, T, H, head_dim)``.
    2. **QK-norm:** apply ``q_norm`` and ``k_norm`` (two
       :class:`~ouroboros.norm.RMSNorm` of size ``head_dim``) to q and k. This
       bounds the attention logits, which matters when the same attention
       weights run many times inside the loop. Standard in Qwen3, OLMo 2 and
       Gemma 3.
    3. Apply RoPE to q and k.
    4. **KV cache (after RoPE):** if ``kv_cache`` is given, concatenate the new
       k, v onto ``kv_cache[cache_key]`` along the sequence dim and store the
       result back. Cached keys are already rotated, so they never need
       re-rotation. Layout: ``{cache_key: {"k": ..., "v": ...}}`` with ``k`` and
       ``v`` of shape ``(B, S, n_kv_heads, head_dim)``.
    5. Attention via the SDPA path or the manual path, then ``wo``.

    Gotchas:

    - **GQA head expansion.** Before attention, K/V heads must line up with
      query heads: ``repeat_interleave(groups)`` on the head dim (or pass
      ``enable_gqa=True`` to SDPA on PyTorch 2.5+).
    - **Causal flag.** ``causal=True`` assumes the cache was empty before this
      call (training, or prefill), so query and key lengths match. A
      single-token decode step passes ``causal=False``: one query attending to
      the whole cache needs no mask. Multi-token calls on a non-empty cache are
      not supported (SDPA's ``is_causal`` aligns the mask to the top-left, which
      would be wrong there).
    - **Manual-path mask.** Build the mask as a boolean ``triu`` and use
      ``masked_fill(-inf)`` on the logits, so the mask never changes the
      activation dtype.
    """

    def __init__(self, cfg: OuroborosConfig) -> None:
        """Build the projections and QK-norms.

        Args:
            cfg: Model configuration. Reads ``dim``, ``n_heads``, ``n_kv_heads``,
                ``norm_eps``, and ``dropout``. Derives ``head_dim = dim // n_heads``
                and ``groups = n_heads // n_kv_heads``.
        """
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        causal: bool = True,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
        use_sdpa: bool = True,
    ) -> torch.Tensor:
        """Compute grouped-query attention over the input sequence.

        Args:
            x: Input activations of shape ``(B, T, dim)``.
            freqs_cis: RoPE phasors of shape ``(T, head_dim // 2)`` (complex),
                already sliced by the caller to positions
                ``start_pos : start_pos + T``.
            causal: Apply a causal mask. ``True`` for training and prefill;
                ``False`` for a single-token decode step. See the class gotchas.
            kv_cache: Optional dict mutated in place. On exit it holds the
                concatenated post-RoPE K and V for ``cache_key``. ``None``
                disables caching (training / full-context forward).
            cache_key: Unique string for this layer (and, inside the loop, this
                loop iteration) so distinct caches never collide.
            use_sdpa: ``True`` uses ``F.scaled_dot_product_attention``;
                ``False`` uses the manual reference path. Both must agree.

        Returns:
            Attention output of shape ``(B, T, dim)``.
        """
        raise NotImplementedError
