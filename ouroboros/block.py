"""Pre-norm transformer block with a swappable FFN for Ouroboros.

``TransformerBlock`` is the single reusable layer type used everywhere a
standard transformer step is needed:

- In the **Prelude** and **Coda** (run once each), with a dense SwiGLU FFN.
- Inside the **RecurrentBlock** (looped), with a Mixture-of-Experts FFN.

Attention is always :class:`~ouroboros.attention.GQAttention`. The **FFN** is
chosen by the ``use_moe`` constructor flag: ``True`` uses
:class:`~ouroboros.moe.MoEFFN`; ``False`` uses a dense
:class:`~ouroboros.moe.Expert` with hidden width ``dim * 4 // 3``.

The block is pre-norm with a residual connection (and dropout) around each
sublayer::

    x = x + dropout(attn(rmsnorm(x), ...))
    x = x + dropout(ffn(rmsnorm(x)))

NOTE: This is a scaffold. Every body raises ``NotImplementedError``; signatures,
type hints, and docstrings are the contract.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import OuroborosConfig

__all__ = ["TransformerBlock"]


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with GQA attention and a swappable FFN.

    Composition:

    - ``attn``: :class:`~ouroboros.attention.GQAttention`.
    - ``ffn``: :class:`~ouroboros.moe.MoEFFN` if ``use_moe`` else a dense
      :class:`~ouroboros.moe.Expert` of hidden width ``cfg.dim * 4 // 3``.
    - Two :class:`~ouroboros.norm.RMSNorm` instances (one before attention, one
      before the FFN) and dropout on each residual branch.

    The forward signature mirrors the attention modules so the block can simply
    forward ``freqs_cis``, ``mask``, ``kv_cache``, and ``cache_key`` through to
    its attention sublayer.

    ``use_moe`` semantics:
        ``use_moe=True`` is used ONLY inside the
        :class:`~ouroboros.recurrence.RecurrentBlock`, where the looped layer
        carries the model's MoE capacity. ``use_moe=False`` (the default) is used
        for every Prelude and Coda layer, which run once and use a dense SwiGLU
        FFN. The choice is fixed at construction; a block is either dense or MoE
        for its whole lifetime.
    """

    def __init__(self, cfg: OuroborosConfig, use_moe: bool = False) -> None:
        """Build the norms, attention sublayer, and FFN sublayer.

        Args:
            cfg: Model configuration. ``cfg.dim`` sizes the norms and the dense
                FFN width (``cfg.dim * 4 // 3``); ``cfg.dropout`` sets the
                residual-branch dropout. Attention fields (``n_heads``,
                ``n_kv_heads``) are consumed via
                :class:`~ouroboros.attention.GQAttention`; MoE fields are
                consumed via :class:`~ouroboros.moe.MoEFFN` when ``use_moe`` is
                ``True``.
            use_moe: If ``True``, the FFN is a :class:`~ouroboros.moe.MoEFFN`
                (Recurrent block). If ``False`` (default), the FFN is a dense
                :class:`~ouroboros.moe.Expert` of width ``cfg.dim * 4 // 3``
                (Prelude/Coda blocks).
        """
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """Apply pre-norm attention then FFN, each with a residual connection.

        Computes::

            x = x + dropout(attn(rmsnorm(x), freqs_cis, mask, kv_cache, cache_key))
            x = x + dropout(ffn(rmsnorm(x)))

        Args:
            x: Input of shape ``(B, T, dim)``.
            freqs_cis: Precomputed RoPE frequencies sliced to the current
                positions, sized for the GQA head width ``dim // n_heads``.
            mask: Additive causal mask of shape ``(1, 1, T, S)``, or ``None`` for
                single-token decode. Its dtype must match the activation dtype.
            kv_cache: Optional dict mutated in place by the attention sublayer to
                store/retrieve cached keys/values; passed through unchanged.
            cache_key: Unique string identifying this layer's slot in
                ``kv_cache`` (e.g. ``"prelude_0"``, ``"recurrent_loop_3"``).

        Returns:
            Output tensor of shape ``(B, T, dim)``.
        """
        raise NotImplementedError
