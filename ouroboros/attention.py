"""Grouped Query Attention for Ouroboros."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import OuroborosConfig
from .norm import RMSNorm
from .rope import apply_rope


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
        # config stuff
        self.dim = cfg.dim
        self.n_heads = cfg.n_heads
        self.kv_heads = cfg.n_kv_heads
        self.q_groupsize = self.n_heads // self.kv_heads

        # get the head dimensions
        self.head_dim = self.dim // self.n_heads
        self.scale = self.head_dim**-0.5

        # initialize norm & rope
        self.eps = cfg.norm_eps
        self.q_norm = RMSNorm(self.head_dim, self.eps)
        self.k_norm = RMSNorm(self.head_dim, self.eps)

        # create the weight matrices
        # we have one matrix for all heads for simple matmuls
        # each one can be split, as the combined matrix is really just each head beside each other
        self.wq = nn.Linear(self.dim, self.head_dim * self.n_heads, bias=False)
        self.wk = nn.Linear(self.dim, self.head_dim * self.kv_heads, bias=False)
        self.wv = nn.Linear(self.dim, self.head_dim * self.kv_heads, bias=False)
        self.wo = nn.Linear(self.head_dim * self.n_heads, self.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_angles: torch.Tensor,
        causal: bool = True,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
        use_sdpa: bool = True,
    ) -> torch.Tensor:
        """Compute grouped-query attention.

        Args:
            x: ``(B, T, dim)``.
            rope_angles: RoPE angles already sliced to the current positions,
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
        B, S, D = x.shape

        q = self.wq(x)  # (batch, seq_len, head_dim * n_heads)
        k = self.wk(x)  # (batch, seq_len, head_dim * kv_heads)
        v = self.wv(x)

        # turn each one into (batch, seq_len, heads, head_dim), split the combined heads
        q = q.view(B, S, self.n_heads, self.head_dim)
        k = k.view(B, S, self.kv_heads, self.head_dim)
        v = v.view(B, S, self.kv_heads, self.head_dim)

        # QK-norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q = apply_rope(q, rope_angles)
        k = apply_rope(k, rope_angles)

        # KV cache
        if kv_cache is not None:
            # skips the cache during training since it's not needed/doesn't exist
            if cache_key in kv_cache:
                # if this layer already has a K/V matrix, store the new values at the end
                # (batch, seq_len_so_far, n_kv_heads, head_dim) + (batch, 1, n_kv_heads, head_dim)
                # = (batch, seq_len + 1, n_kv_heads, head_dim)
                k = torch.cat([kv_cache[cache_key]["k"], k], dim=1)
                v = torch.cat([kv_cache[cache_key]["v"], v], dim=1)
            kv_cache[cache_key] = {
                "k": k,
                "v": v,
            }  # either we store the updated, or the new KV values

        # attention
        # change shape to (batch, heads, seq_len, head_dim) so when we do matmul:
        # (seq_len, head_dim) * (head_dim, seq_len)
        # matmul works on the last 2 dimensions, so it will do this for every batch & head
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # proceed with attention
        if use_sdpa:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=causal, enable_gqa=True, scale=self.scale
            )
            # this gives us the result from probs @ V
            # (batch, num_heads, seq_len, head_dim)
        else:
            # manual GQA implementation
            # give every query head its KV head: if we do matmul with the previous shapes,
            # q has 8 heads, kv has 2, there's a mismatch
            # takes the first dim (heads) and turns it into [h0, h0, h0, h0, h1, h1, h1, h1]
            k = k.repeat_interleave(self.q_groupsize, dim=1)
            v = v.repeat_interleave(self.q_groupsize, dim=1)

            # proceed with attention as normal, qkv are the same shapes now
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # causal mask
            if causal:
                # keep what's above the diagonal, for shape seq_len x seq_len
                mask = torch.triu(
                    torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1
                )
                # fill the mask area with -infinity for softmax to turn to 0
                attn_scores = attn_scores.masked_fill(mask, float("-inf"))

            # softmax
            probs = attn_scores.softmax(
                dim=-1
            )  # last dimension to get probabilities for the keys

            # output
            out = (
                probs @ v
            )  # multiply each query row's probabilities against V, (B, heads, S, head_dim)

        out = out.transpose(1, 2)  # flip heads and seq_len, (B, S, heads, head_dim)
        out = out.reshape(B, S, self.n_heads * self.head_dim)

        output = self.wo(out)
        return output
