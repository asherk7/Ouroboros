"""The full Ouroboros model and its generation loop::

    embed -> Prelude (once) -> Recurrent (n_loops) -> Coda (once) -> norm -> head
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .block import TransformerBlock
from .config import OuroborosConfig
from .norm import RMSNorm
from .recurrence import RecurrentBlock
from .rope import precompute_rope_angles


class Ouroboros(nn.Module):
    """Recurrent-depth transformer language model.

    Submodules: ``embed`` (weight-tied to ``head``), ``rope_angles`` buffer,
    ``prelude`` and ``coda`` (dense blocks), ``recurrent``, ``norm``, ``head``.
    Also stores ``self.cfg``.
    """

    rope_angles: torch.Tensor  # buffer; annotated so mypy knows it's a tensor

    def __init__(self, cfg: OuroborosConfig) -> None:
        """Build the model, tie the head to the embedding, and init weights."""
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        # angles are per head, so size them to head_dim; non-persistent since they're derived
        angles = precompute_rope_angles(
            cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("rope_angles", angles, persistent=False)

        self.prelude = nn.ModuleList(
            TransformerBlock(cfg) for _ in range(cfg.prelude_layers)
        )
        self.recurrent = RecurrentBlock(cfg)
        self.coda = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.coda_layers))

        self.norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        self._init_weights()
        # tie after init so both names point at one tensor (the embedding's init wins)
        self.head.weight = self.embed.weight

    def _init_weights(self) -> None:
        """Init every ``nn.Linear`` and ``nn.Embedding`` weight with ``N(0, init_std)``.

        If early training is unstable, try scaling ``wo`` and each FFN ``down`` by
        ``1 / sqrt(2 * n_eff)`` (GPT-2 style, ``n_eff`` counting loops).
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Embed, run Prelude, Recurrent, and Coda, then the tied head.

        Slice ``rope_angles[start_pos : start_pos + T]``, use ``causal = T > 1``, and
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
        T = input_ids.shape[1]
        # slicing past the end of rope_angles just returns fewer rows, so check here
        if start_pos + T > self.cfg.max_seq_len:
            raise ValueError(
                f"start_pos + T = {start_pos + T} exceeds max_seq_len "
                f"({self.cfg.max_seq_len})"
            )

        rope_angles = self.rope_angles[start_pos : start_pos + T]  # (T, head_dim // 2)
        # prefill (T > 1) masks the future; a single decode token sees the whole cache
        causal = T > 1

        x = self.embed(input_ids)  # (B, T, dim)
        for i, block in enumerate(self.prelude):
            x = block(x, rope_angles, causal, kv_cache, cache_key=f"prelude_{i}")

        # e = prelude output, re-injected every loop; the loop also starts from it
        x = self.recurrent(x, x, rope_angles, causal, n_loops, kv_cache)

        for i, block in enumerate(self.coda):
            x = block(x, rope_angles, causal, kv_cache, cache_key=f"coda_{i}")

        return self.head(self.norm(x))  # (B, T, vocab_size)

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
        if temperature <= 0:
            raise ValueError(f"temperature ({temperature}) must be > 0")
        top_k = min(top_k, self.cfg.vocab_size)

        # eval mode so sampling doesn't count MoE expert load; restored at the end
        was_training = self.training
        self.eval()

        ids = input_ids
        kv_cache: Optional[dict] = {} if use_cache else None
        for _ in range(max_new_tokens):
            if use_cache and ids.shape[1] > input_ids.shape[1]:
                # decode: only the newest token; the cache holds everything before it
                logits = self(ids[:, -1:], n_loops, kv_cache, start_pos=ids.shape[1] - 1)
            else:
                # prefill (cache) or full rerun (no cache)
                logits = self(ids, n_loops, kv_cache)

            logits = logits[:, -1, :] / temperature  # (B, vocab_size)
            if top_k > 0:
                # mask everything below the k-th largest logit
                kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            ids = torch.cat([ids, next_id], dim=1)

        self.train(was_training)
        return ids
