"""Pre-norm transformer block with a swappable FFN for Ouroboros.

``TransformerBlock`` is the one layer type used everywhere:

- In the **Prelude** and **Coda** (run once each), with a dense SwiGLU FFN.
- Inside the **RecurrentBlock** (looped), with a Mixture-of-Experts FFN.

Attention is always :class:`~ouroboros.attention.GQAttention`. The FFN is chosen
by ``use_moe``: ``True`` gives :class:`~ouroboros.moe.MoEFFN`, ``False`` gives a
dense :class:`~ouroboros.moe.Expert` of width ``dim * 4 // 3``::

    x = x + dropout(attn(rmsnorm(x), ...))
    x = x + dropout(ffn(rmsnorm(x)))

Huginn (Geiping et al., 2025) reports needing a "sandwich" norm (an extra
RMSNorm on each sublayer's output) to train its loop stably at scale.
Plain pre-norm is fine at this size; sandwich norm is the first thing to try if
the loop misbehaves.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import OuroborosConfig

__all__ = ["TransformerBlock"]


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with GQA attention and a swappable FFN.

    Submodules:

    - ``attn``: :class:`~ouroboros.attention.GQAttention`.
    - ``ffn``: :class:`~ouroboros.moe.MoEFFN` if ``use_moe`` else
      ``Expert(cfg.dim, cfg.dim * 4 // 3)``.
    - ``attn_norm``, ``ffn_norm``: two :class:`~ouroboros.norm.RMSNorm`.

    ``use_moe=True`` is used only by the
    :class:`~ouroboros.recurrence.RecurrentBlock`; Prelude and Coda blocks are
    dense.
    """

    def __init__(self, cfg: OuroborosConfig, use_moe: bool = False) -> None:
        """Build the norms, attention sublayer, and FFN sublayer.

        Args:
            cfg: Model configuration (``dim``, ``dropout``, plus the attention
                and MoE fields consumed by the sublayers).
            use_moe: ``True`` for the recurrent block's MoE FFN; ``False``
                (default) for a dense SwiGLU FFN.
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
    ) -> torch.Tensor:
        """Apply pre-norm attention then FFN, each with a residual connection.

        Args:
            x: Input of shape ``(B, T, dim)``.
            freqs_cis: RoPE phasors sliced to the current positions.
            causal: Passed to attention; ``False`` only for single-token decode.
            kv_cache: Optional cache dict, passed through to attention.
            cache_key: This layer's cache slot (e.g. ``"prelude_0"``,
                ``"recurrent_loop_3"``).

        Returns:
            Output tensor of shape ``(B, T, dim)``.
        """
        raise NotImplementedError
