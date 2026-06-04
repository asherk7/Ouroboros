"""Rotary Positional Embeddings (RoPE) for Ouroboros.

RoPE (Su et al., 2021) encodes absolute token position as a per-position rotation
applied to query and key vectors, so that attention logits depend only on the
*relative* offset between tokens. Ouroboros precomputes the rotation phasors once
(at model build time) and reuses them across every attention call.

Because GQA and MLA rotate different per-head widths, the model precomputes **two**
frequency buffers: one sized ``dim // n_heads`` for :class:`~ouroboros.attention.GQAttention`
and one sized ``qk_rope_head_dim`` for the decoupled RoPE keys of
:class:`~ouroboros.attention.MLAttention`. The model selects the appropriate buffer
per ``attn_type`` and slices it to the current positions before calling
:func:`apply_rope`.

This module exposes two functions:

* :func:`precompute_rope_freqs` — build the complex phasor table once.
* :func:`apply_rope` — rotate a Q/K tensor using a (pre-sliced) phasor table.
"""

import torch


def precompute_rope_freqs(
    dim: int, max_len: int, theta: float = 10000.0
) -> torch.Tensor:
    """Precompute the complex RoPE phasors for all positions ``0 .. max_len - 1``.

    For each frequency index ``k`` (there are ``dim // 2`` of them) the inverse
    frequency is ``theta_k = theta ** (-2k / dim)``. For each position ``m`` the
    phasor is the unit complex number ``e^{i * m * theta_k}`` (computed via
    ``torch.polar`` with magnitude 1). Storing the rotation as a complex phasor
    lets :func:`apply_rope` perform the rotation as a single elementwise complex
    multiply instead of an explicit 2x2 sin/cos matrix.

    Args:
        dim: Per-head feature dimension to rotate over (**must be even**); the
            table holds ``dim // 2`` frequency pairs. For GQA pass
            ``dim // n_heads``; for MLA pass ``qk_rope_head_dim``.
        max_len: Number of positions to precompute (typically
            ``OuroborosConfig.max_seq_len``). Positions are ``0 .. max_len - 1``.
        theta: RoPE base frequency. Larger values decay the frequencies more
            slowly and extend usable context (``10000.0`` small-model default,
            ``500000.0`` for long context). Matches ``OuroborosConfig.rope_theta``.

    Returns:
        A ``complex64`` tensor of shape ``(max_len, dim // 2)``. Row ``m`` holds
        the per-frequency phasors for position ``m``; row 0 is all ``1 + 0j`` (the
        identity rotation).

    Shapes:
        ``-> (max_len, dim // 2)`` (complex64)
    """
    raise NotImplementedError


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Rotate a query or key tensor by precomputed RoPE phasors.

    Each adjacent feature pair ``(x[..., 2k], x[..., 2k + 1])`` is viewed as a 2-D
    complex number and multiplied by the per-position phasor ``freqs_cis[t, k]``,
    which rotates it by the position-dependent angle in the complex plane; the
    result is viewed back as two real features. Rotation is a pure isometry, so it
    **preserves the per-vector L2 norm** — only the phase (relative orientation)
    changes, which is what makes attention logits depend on relative position.

    Args:
        x: Query or key tensor of shape ``(B, T, H, head_dim)`` where ``head_dim``
            **must be even**. Typically fp16/bf16 under mixed precision.
        freqs_cis: Complex phasor table of shape ``(T, head_dim // 2)``, already
            **sliced by the caller** to exactly the positions being processed
            (``start_pos : start_pos + T``). ``apply_rope`` does **not** know
            ``start_pos`` — passing an unsliced or wrongly-offset table silently
            applies the wrong rotation (the classic incremental-decode bug where
            cached tokens get position-0 rotations).

    Returns:
        The rotated tensor with the **same shape and dtype as** ``x``. The rotation
        is computed in float32 internally (view-as-complex requires float) and cast
        back to ``x.dtype`` before returning.

    Shapes:
        ``(B, T, H, head_dim) x (T, head_dim // 2) -> (B, T, H, head_dim)``

    Notes:
        * Position 0 is the identity: ``freqs_cis[0]`` is ``1 + 0j``, so the first
          token passes through unrotated.
        * Norm-preserving: ``||apply_rope(x)|| == ||x||`` per head vector — useful
          as a unit-test invariant (alongside the position-0 identity check).
        * The caller is responsible for slicing ``freqs_cis`` to the current
          positions; this function operates purely on the slice it is given.
    """
    raise NotImplementedError
