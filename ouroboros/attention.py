"""Grouped Query Attention for Ouroboros."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import OuroborosConfig


class GQAttention(nn.Module):
    """Grouped Query Attention with QK-norm and an optional KV cache.

    Each of the ``n_kv_heads`` K/V heads is shared by ``n_heads // n_kv_heads``
    query heads, which shrinks the KV cache. Projections ``wq``, ``wk``, ``wv``,
    ``wo`` are bias-free.

    forward: project q/k/v -> QK-norm (RMSNorm on q and k) -> RoPE -> append to
    the KV cache -> attention (SDPA or manual reference) -> ``wo``.

    Gotchas:
        - Expand K/V heads with ``repeat_interleave`` before attention (or pass
          ``enable_gqa=True`` to SDPA).
        - ``causal=True`` assumes the cache was empty (training or prefill).
          Single-token decode uses ``causal=False``.
        - Manual path: boolean ``triu`` mask applied with ``masked_fill(-inf)``.
    """

    def __init__(self, cfg: OuroborosConfig) -> None:
        """Build the projections and QK-norms.

        Args:
            cfg: Reads ``dim``, ``n_heads``, ``n_kv_heads``, ``norm_eps``, ``dropout``.
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
        """Compute grouped-query attention.

        Args:
            x: ``(B, T, dim)``.
            freqs_cis: RoPE phasors already sliced to the current positions,
                ``(T, head_dim // 2)``.
            causal: Apply a causal mask.
            kv_cache: Optional dict, updated in place as
                ``{cache_key: {"k": ..., "v": ...}}`` with post-RoPE tensors of shape
                ``(B, S, n_kv_heads, head_dim)``.
            cache_key: This layer's (and loop's) cache slot.
            use_sdpa: SDPA path if ``True``, manual reference path if ``False``.

        Returns:
            ``(B, T, dim)``.
        """
        raise NotImplementedError
