"""Attention for Ouroboros.

This module provides :class:`GQAttention` — Grouped Query Attention (Ainslie et
al., 2023) — the single attention mechanism used across the Prelude, Recurrent,
and Coda stages of the model. It exposes a FlashAttention-2 fast path and
degrades gracefully to ``torch.nn.functional.scaled_dot_product_attention``
(flash / mem-efficient backend) and finally to a manual SDPA fallback.

The ``forward`` signature
``(x, freqs_cis, mask=None, kv_cache=None, cache_key="default")`` is the
attention contract consumed by :class:`~ouroboros.block.TransformerBlock`.

FlashAttention-2 / T4 reality
-----------------------------
FlashAttention-2's official prebuilt wheels target Ampere (sm80)+ / Hopper. On the
target hardware for this project — a Google Colab T4 (Turing, sm75) — the ``flash-attn``
package is forward-only and frequently fails to build. The realistic, robust fast path on
T4 is therefore ``F.scaled_dot_product_attention`` under
``torch.backends.cuda.sdp_kernel`` (flash / mem-efficient backends), with a manual matmul
fallback for correctness everywhere. The ``flash_attn_func`` path is kept as an *optional*
accelerator that engages only when the package is importable (e.g. on a rented Ampere GPU);
its import is gated below so the module loads cleanly on T4.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import OuroborosConfig

# FlashAttention-2 is an optional Ampere+ accelerator. On Turing (T4) the package is
# usually unavailable, so its import is gated and the flag steers GQAttention to the
# SDPA-flash / manual fallback instead of a hard import error.
try:  # pragma: no cover - environment-dependent optional dependency
    from flash_attn import flash_attn_func

    _HAS_FLASH_ATTN = True
except ImportError:  # pragma: no cover - expected on T4 / sm75
    flash_attn_func = None  # type: ignore[assignment]
    _HAS_FLASH_ATTN = False


class GQAttention(nn.Module):
    """Grouped Query Attention with a FlashAttention-2 / SDPA-flash fast path.

    Grouped Query Attention (Ainslie et al., 2023, https://arxiv.org/abs/2305.13245)
    uses fewer key/value heads than query heads: ``n_kv_heads < n_heads``. Each KV head
    is shared across ``groups = n_heads // n_kv_heads`` query heads, shrinking the KV
    cache (and KV projection cost) by the ``groups`` factor while preserving full query
    expressiveness. It interpolates between Multi-Head Attention (``n_kv_heads ==
    n_heads``) and Multi-Query Attention (``n_kv_heads == 1``).

    Projections
    -----------
    ``wq, wk, wv, wo`` are all ``bias=False`` ``nn.Linear`` layers with
    ``head_dim = dim // n_heads``:

    - ``wq``: ``dim -> n_heads * head_dim``
    - ``wk``: ``dim -> n_kv_heads * head_dim``
    - ``wv``: ``dim -> n_kv_heads * head_dim``
    - ``wo``: ``n_heads * head_dim -> dim``

    RoPE and KV caching
    -------------------
    Rotary position embeddings (RoPE) are applied to **Q and K** before attention. K and
    V are written to ``kv_cache`` *after* RoPE has been applied, so cached keys are
    already positionally encoded and need no re-rotation when retrieved on a later decode
    step. The cache value layout is ``{cache_key: {"k": ..., "v": ...}}`` where ``k`` and
    ``v`` are ``.detach()``-ed and concatenated along the sequence dimension across decode
    steps.

    Attention backends (in priority order)
    --------------------------------------
    1. **FlashAttention-2 fast path** (``_HAS_FLASH_ATTN``): ``flash_attn_func`` consumes
       ``(B, T, H, head_dim)`` directly and handles GQA natively — no ``repeat_interleave``
       KV-head expansion is needed. The kernel requires fp16/bf16 inputs; activations
       already in fp16/bf16 are passed through **unchanged** (the T4 target trains in
       fp16 — do NOT round-trip fp16 through bfloat16), and only fp32 inputs are cast
       (to fp16) and restored afterward. Causality is signalled with
       ``causal=(mask is not None)``: full-sequence prefill/training (where
       ``q_len == kv_len``) is causal; a single-token T=1 decode passes ``mask=None`` /
       ``causal=False`` (one query attending to the whole cache needs no mask). Note
       flash-attn ≥ 2.1 aligns causal masks to the *bottom-right* when
       ``q_len < kv_len`` — irrelevant here precisely because decode steps are T=1 and
       non-causal, but it is why ``causal=True`` with a partially-filled cache must not
       be combined with multi-token decode without re-checking alignment.
    2. **SDPA-flash path (the realistic T4 fast path):** when ``flash_attn`` is absent but
       a CUDA flash / mem-efficient backend is available,
       ``F.scaled_dot_product_attention`` under ``torch.backends.cuda.sdp_kernel`` provides
       fused attention on Turing without a custom kernel build.
    3. **Manual SDPA fallback (always correct):** expand KV heads with
       ``repeat_interleave(groups, dim=...)``, transpose to ``(B, H, T, head_dim)``,
       compute ``softmax(Q·Kᵀ · head_dim**-0.5 + mask) · V`` with dropout. This is the
       reference path used on CPU and for numerical-equivalence tests.

    Shapes
    ------
    Input ``x``: ``(B, T, dim)`` → Output: ``(B, T, dim)``.

    Gotchas
    -------
    - **Mask dtype must match the activation dtype.** In the manual fallback the additive
      causal mask is added to the attention *logits*; an fp32 mask on fp16/bf16
      activations silently upcasts the logits, which then breaks the matmul against ``V``.
      The mask must be created in the same dtype as the activations.
    - ``n_heads % n_kv_heads == 0`` must hold (enforced by
      ``OuroborosConfig.__post_init__``) so that ``groups`` is an integer and
      ``repeat_interleave`` lines KV heads up with query-head groups.
    - On decode, RoPE is applied only to the *new* tokens; cached K already carries its
      rotation, which is exactly why K/V are cached post-RoPE.
    """

    def __init__(self, cfg: OuroborosConfig) -> None:
        """Initialize a Grouped Query Attention block.

        Args:
            cfg: Model configuration. Reads ``dim``, ``n_heads``, ``n_kv_heads``, and
                ``dropout``. Derives ``head_dim = dim // n_heads`` and
                ``groups = n_heads // n_kv_heads``.

        Note:
            Per the planning-phase contract, this stub declares no layers or parameters;
            the ``wq``/``wk``/``wv``/``wo`` projections and ``dropout_p`` are documented in
            the class docstring and instantiated when the body is implemented.
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
        """Compute grouped-query attention over the input sequence.

        Args:
            x: Input activations of shape ``(B, T, dim)``.
            freqs_cis: Precomputed RoPE phasors sized for ``head_dim``, shape
                ``(T, head_dim // 2)`` (complex), already sliced by the caller to the
                positions ``start_pos : start_pos + T``.
            mask: Optional additive causal mask of shape ``(1, 1, T, S)`` (``S`` is the
                full cached sequence length); ``-inf`` above the diagonal, ``0`` on/below.
                Its dtype must match ``x.dtype``. ``None`` for a single-token (T=1) decode
                step. See the dtype gotcha in the class docstring.
            kv_cache: Optional dict mutated in place. On entry it may hold
                ``{cache_key: {"k": ..., "v": ...}}`` from previous steps; on exit it holds
                the concatenated, ``.detach()``-ed K and V (post-RoPE) for ``cache_key``.
                ``None`` disables caching (training / full-context forward).
            cache_key: Unique string identifying this attention layer (and, inside the
                recurrent loop, this loop depth) so distinct caches never collide.

        Returns:
            Attention output of shape ``(B, T, dim)``.

        Raises:
            NotImplementedError: This is a planning-phase stub.
        """
        raise NotImplementedError
