"""The full Ouroboros model and its generation loop::

    embed -> Prelude (once) -> Recurrent (n_loops) -> Coda (once) -> norm -> head
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

# These component imports are the build contract for the model and are used once
# the stubbed bodies are implemented; the noqa keeps the scaffold lint-clean.
from .block import TransformerBlock  # noqa: F401
from .config import OuroborosConfig
from .norm import RMSNorm  # noqa: F401
from .recurrence import RecurrentBlock  # noqa: F401
from .rope import precompute_rope_freqs  # noqa: F401


class Ouroboros(nn.Module):
    """Recurrent-depth transformer language model.

    Submodules: ``embed`` (weight-tied to ``head``), ``freqs_cis`` buffer,
    ``prelude`` and ``coda`` (dense blocks), ``recurrent``, ``norm``, ``head``.
    Also stores ``self.cfg``.
    """

    def __init__(self, cfg: OuroborosConfig) -> None:
        """Build the model, tie the head to the embedding, and init weights."""
        super().__init__()
        raise NotImplementedError

    def _init_weights(self) -> None:
        """Init every ``nn.Linear`` and ``nn.Embedding`` weight with ``N(0, init_std)``.

        If early training is unstable, try scaling ``wo`` and each FFN ``down`` by
        ``1 / sqrt(2 * n_eff)`` (GPT-2 style, ``n_eff`` counting loops).
        """
        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Embed, run Prelude, Recurrent, and Coda, then the tied head.

        Slice ``freqs_cis[start_pos : start_pos + T]``, use ``causal = T > 1``, and
        cache keys ``prelude_{i}`` / ``coda_{i}``. The recurrent block starts from
        ``h = e = x``.

        Args:
            input_ids: ``(B, T)``.
            n_loops: Loop count; ``None`` means ``cfg.mean_loops``. Keep it the same
                across steps that share a cache.
            kv_cache: Optional dict, updated in place. Pass ``{}`` at prefill.
            start_pos: Tokens already in the cache. Raise ``ValueError`` if
                ``start_pos + T > cfg.max_seq_len``.

        Returns:
            Logits, ``(B, T, vocab_size)``.
        """
        raise NotImplementedError

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        n_loops: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressively sample ``max_new_tokens`` tokens.

        With the cache: prefill the prompt once, then feed one token per step at the
        right ``start_pos``. Without it: rerun the full sequence every step (the
        baseline for the KV-cache speedup metric). Sampling is temperature, top-k,
        softmax, then ``torch.multinomial``.

        Args:
            input_ids: Prompt, ``(B, T)``.
            max_new_tokens: Tokens to append.
            n_loops: Loop count for every step; ``None`` means ``cfg.mean_loops``.
            temperature: Must be ``> 0``.
            top_k: ``0`` disables; clamp to ``vocab_size``.
            use_cache: Use the KV cache.

        Returns:
            ``(B, T + max_new_tokens)``.
        """
        raise NotImplementedError
