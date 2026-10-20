"""Pre-norm transformer block used in the Prelude, Recurrent, and Coda stages."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import OuroborosConfig

__all__ = ["TransformerBlock"]


class TransformerBlock(nn.Module):
    """Pre-norm block: GQA attention then an FFN, each with a residual::

        x = x + dropout(attn(attn_norm(x)))
        x = x + dropout(ffn(ffn_norm(x)))

    The FFN is ``MoEFFN`` if ``use_moe`` (recurrent block only), else a dense
    ``FeedForward(dim, dim * 8 // 3)``. If the loop trains unstably, try Huginn's
    sandwich norm.
    """

    def __init__(self, cfg: OuroborosConfig, use_moe: bool = False) -> None:
        """Build the norms, attention, and FFN.

        Args:
            cfg: Model configuration.
            use_moe: Use the MoE FFN instead of a dense one.
        """
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        rope_angles: torch.Tensor,
        causal: bool = True,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """Apply attention then FFN, each pre-normed with a residual.

        Args:
            x: ``(B, T, dim)``.
            rope_angles, causal, kv_cache, cache_key: Passed through to attention.

        Returns:
            ``(B, T, dim)``.
        """
        raise NotImplementedError
