"""Rotary Positional Embeddings (RoPE) for Ouroboros."""

import torch


def precompute_rope_freqs(
    dim: int, max_len: int, theta: float = 10000.0
) -> torch.Tensor:
    """Precompute RoPE phasors ``e^{i * m * theta_k}`` for positions ``0 .. max_len - 1``.

    ``theta_k = theta ** (-2k / dim)`` for ``k < dim // 2``.

    Args:
        dim: Per-head width (even).
        max_len: Number of positions.
        theta: Base frequency.

    Returns:
        ``complex64`` tensor, ``(max_len, dim // 2)``.
    """
    raise NotImplementedError


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Rotate each adjacent feature pair of ``x`` by its position's phasor.

    Args:
        x: Query or key, ``(B, T, H, head_dim)``.
        freqs_cis: Phasors already sliced to these positions, ``(T, head_dim // 2)``.

    Returns:
        Same shape and dtype as ``x`` (computed in float32).
    """
    raise NotImplementedError
