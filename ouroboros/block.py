"""Pre-norm transformer block used in the Prelude, Recurrent, and Coda stages."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import OuroborosConfig
from .moe import MoEFFN
from .attention import GQAttention
from .norm import RMSNorm
from .feedforward import FeedForward

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
        
        self.attn = GQAttention(cfg=cfg)
        self.attn_norm = RMSNorm(dim=cfg.dim, eps=cfg.norm_eps)
        self.attn_dropout = nn.Dropout(cfg.dropout)

        if use_moe:
            self.ffn = MoEFFN(cfg)
        else:
            self.hidden_dim = (8 * cfg.dim) // 3
            self.ffn = FeedForward(dim=cfg.dim, hidden_dim=self.hidden_dim)
        
        self.ffn_norm = RMSNorm(dim=cfg.dim, eps=cfg.norm_eps)
        self.ffn_dropout = nn.Dropout(cfg.dropout)

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
        
        # attention
        x_ = self.attn_norm(x)
        x_ = self.attn(x_, rope_angles=rope_angles, causal=causal, kv_cache=kv_cache, cache_key=cache_key)
        x = x + self.attn_dropout(x_)

        # ffn
        x_ = self.ffn_norm(x)
        x_ = self.ffn(x_)
        x = x + self.ffn_dropout(x_)
        
        return x
