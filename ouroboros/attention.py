"""Attention mechanisms for Ouroboros.

This module provides the two switchable attention implementations used across the
Prelude, Recurrent, and Coda stages of the model, selected at runtime via
``OuroborosConfig.attn_type`` ("gqa" | "mla"):

- :class:`GQAttention` — Grouped Query Attention (Ainslie et al., 2023). The simpler,
  default mechanism; it exposes a FlashAttention-2 fast path and degrades gracefully to
  ``torch.nn.functional.scaled_dot_product_attention`` (flash / mem-efficient backend)
  and finally to a manual SDPA fallback.
- :class:`MLAttention` — Multi-Latent Attention (DeepSeek-V2, 2024). Compresses the KV
  path into a low-rank latent that is the only thing cached, dramatically shrinking the
  KV-cache footprint during autoregressive decode.

Both classes share the exact same ``forward`` signature
``(x, freqs_cis, mask=None, kv_cache=None, cache_key="default")`` so that
:class:`~ouroboros.block.TransformerBlock` can dispatch to either interchangeably.

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
       KV-head expansion is needed. Activations are cast to ``bfloat16`` for the kernel and
       restored to the original dtype afterward; causality is signalled with
       ``causal=(mask is not None)`` (full-sequence prefill/training is causal; a
       single-token T=1 decode passes ``mask=None`` / ``causal=False``).
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
    - ``n_heads % n_kv_heads == 0`` must hold (validated by the model) so that ``groups``
      is an integer and ``repeat_interleave`` lines KV heads up with query-head groups.
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


class MLAttention(nn.Module):
    """Multi-Latent Attention (DeepSeek-V2) with compressed, cache-efficient KV.

    Multi-Latent Attention (DeepSeek-V2, 2024, https://arxiv.org/abs/2405.04434)
    replaces the standard per-token K/V cache with a single low-rank latent. Instead of
    caching ``n_heads × head_dim`` keys and values per token, MLA caches only the
    compressed latent ``c_kv`` plus the (already-rotated) shared RoPE keys, then
    *reconstructs* the per-head ``k_nope`` and ``v`` from the latent on every step. This
    trades a small amount of recompute for a dramatically smaller KV cache — the main
    decode-time memory and bandwidth win over GQA.

    Head dimensions
    ---------------
    The per-head query/key dimension is split into a non-positional part and a rotary
    part: ``q_head_dim = qk_nope_head_dim + qk_rope_head_dim``. The value head dimension
    ``v_head_dim`` is independent. Note that ``n_kv_heads`` is **irrelevant** to MLA — it
    is a GQA-only field.

    Q path (compress → reconstruct, split nope/rope)
    ------------------------------------------------
    ``x → q_down (dim → q_lora_rank) → q_norm (RMSNorm)`` then two up-projections:

    - ``q_up_nope``: ``q_lora_rank → n_heads * qk_nope_head_dim`` — **no RoPE**.
    - ``q_up_rope``: ``q_lora_rank → n_heads * qk_rope_head_dim`` — **RoPE applied**.

    The two are concatenated per head into ``q`` of width ``q_head_dim``.

    KV path (what is cached vs reconstructed)
    -----------------------------------------
    ``kv_down`` projects ``dim → kv_lora_rank + qk_rope_head_dim`` and the output is split:

    - ``c_kv`` (the first ``kv_lora_rank`` channels): the compressed KV latent —
      **CACHED**.
    - ``k_rope_raw`` (the trailing ``qk_rope_head_dim`` channels): a *single* RoPE key
      shared across all heads. It is broadcast to ``n_heads`` and rotated:
      ``k_rope = RoPE(expand_to_heads(k_rope_raw))`` — and this rotated tensor is
      **CACHED** (so it never needs re-rotation on retrieval).

    Each step, the per-head ``k_nope`` and ``v`` are **RECONSTRUCTED** (never cached) from
    the latent: ``c_kv → kv_norm (RMSNorm) → kv_up (kv_lora_rank → n_heads *
    (qk_nope_head_dim + v_head_dim))`` is split into ``k_nope`` and ``v``. The final key is
    ``k = concat(k_nope, k_rope)`` per head, giving width ``q_head_dim``. Attention is then
    a standard scaled dot-product over ``q``/``k`` with the value width ``v_head_dim``, and
    the output projection ``wo`` maps ``n_heads * v_head_dim → dim``.

    Cache layout
    ------------
    ``{cache_key: {"c_kv": (B, S, kv_lora_rank), "k_rope": (B, S, n_heads,
    qk_rope_head_dim)}}`` — only the compressed latent and the rotary keys are stored,
    which is far smaller than GQA's full ``(B, S, n_kv_heads, head_dim)`` K and V.

    Shapes
    ------
    Input ``x``: ``(B, T, dim)`` → Output: ``(B, T, dim)``.

    Gotchas
    -------
    - **Separate RoPE sizing.** RoPE here acts on ``qk_rope_head_dim`` channels, *not* on
      ``dim // n_heads`` as in GQA. The model therefore precomputes a **separate** RoPE
      buffer (``freqs_cis_mla``) sized to ``qk_rope_head_dim`` and passes it in; mixing it
      up with the GQA buffer rotates the wrong number of channels.
    - ``n_kv_heads`` has no effect here; do not read it.
    - ``c_kv`` and the rotated ``k_rope`` are concatenated along the sequence dim across
      decode steps and ``.detach()``-ed before storage, exactly as in GQA's K/V cache.
    """

    def __init__(self, cfg: OuroborosConfig) -> None:
        """Initialize a Multi-Latent Attention block.

        Args:
            cfg: Model configuration. Reads ``dim``, ``n_heads``, ``kv_lora_rank``,
                ``q_lora_rank``, ``qk_rope_head_dim``, ``qk_nope_head_dim``, ``v_head_dim``,
                and ``dropout``. Derives ``q_head_dim = qk_nope_head_dim +
                qk_rope_head_dim``. ``n_kv_heads`` is intentionally ignored.

        Note:
            Per the planning-phase contract, this stub declares no layers or parameters;
            the ``q_down``/``q_norm``/``q_up_nope``/``q_up_rope``/``kv_down``/``kv_norm``/
            ``kv_up``/``wo`` modules are documented in the class docstring and instantiated
            when the body is implemented.
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
        """Compute multi-latent attention over the input sequence.

        Args:
            x: Input activations of shape ``(B, T, dim)``.
            freqs_cis: Precomputed RoPE phasors sized for ``qk_rope_head_dim``, shape
                ``(T, qk_rope_head_dim // 2)`` (complex) — the dedicated MLA RoPE buffer,
                already sliced to ``start_pos : start_pos + T``. This is *not* the GQA
                ``head_dim``-sized buffer (see the gotcha in the class docstring).
            mask: Optional additive causal mask of shape ``(1, 1, T, S)`` (``S`` is the
                full cached sequence length); ``-inf`` above the diagonal, dtype matching
                ``x.dtype``. ``None`` for a single-token (T=1) decode step.
            kv_cache: Optional dict mutated in place. On entry it may hold
                ``{cache_key: {"c_kv": ..., "k_rope": ...}}`` from previous steps; on exit
                it holds the concatenated, ``.detach()``-ed compressed latent ``c_kv``
                ``(B, S, kv_lora_rank)`` and rotated keys ``k_rope`` ``(B, S, n_heads,
                qk_rope_head_dim)``. ``None`` disables caching. ``k_nope`` and ``v`` are
                reconstructed from ``c_kv`` and are never stored.
            cache_key: Unique string identifying this attention layer (and loop depth
                inside the recurrent loop) so distinct caches never collide.

        Returns:
            Attention output of shape ``(B, T, dim)``.

        Raises:
            NotImplementedError: This is a planning-phase stub.
        """
        raise NotImplementedError
