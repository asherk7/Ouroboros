"""Rotary Positional Embeddings (RoPE) for Ouroboros."""

import torch


def precompute_rope_angles(
    dim: int, max_len: int, theta: float = 10000.0
) -> torch.Tensor:
    """Precompute RoPE rotation angles ``m * theta_k`` for positions ``0 .. max_len - 1``.

    ``theta_k = theta ** (-2k / dim)`` for ``k < dim // 2``.

    Args:
        dim: Per-head width (even).
        max_len: Number of positions.
        theta: Base frequency.

    Returns:
        ``float32`` tensor, ``(max_len, dim // 2)``: entry ``[m, k]`` is the angle
        that pair ``k`` at position ``m`` rotates by.
    """
    # create the pair speeds, k = 0, 2, 4...; as we go down the pairs, rotations slow
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # create the positions in the sequence, m = 0, 1, 2...; further position = more rotation 
    m = torch.arange(max_len).float()

    # create our main theta, for every m, we have every inv_freq
    # this can be applied to every token in a sequence 
    # for every m, we have every inv_freq to pairs (every index is m * inv_freq)
    angles = torch.outer(m, inv_freq) # (seq_len, num_pairs), num_pairs = dim/2
    return angles

def apply_rope(x: torch.Tensor, rope_angles: torch.Tensor) -> torch.Tensor:
    """Rotate each adjacent feature pair ``(2k, 2k + 1)`` of ``x`` by its angle.

    Applies the 2x2 rotation matrix elementwise:
    ``x0' = x0 cos - x1 sin``, ``x1' = x0 sin + x1 cos``.

    Args:
        x: Query or key, ``(B, seq_len, H, head_dim)``.
        rope_angles: Angles already sliced to these positions, ``(seq_len, head_dim // 2)``.

    Returns:
        Same shape and dtype as ``x`` (computed in float32).

    Gotcha:
        Reshape the ``(seq_len, head_dim // 2)`` cos/sin to ``(1, seq_len, 1, head_dim // 2)``
        before multiplying. Broadcasting aligns from the right, so unreshaped they
        line up with ``(H, head_dim // 2)``: an error if ``seq_len != H``, silently wrong
        if ``seq_len == H``.
    """
    x_ = x.float() # x is (batch, seq_len, heads, head_dim)
    seq_len = x.shape[1]

    # take the angles and apply cos/sin to them to get our cos/sin theta
    cos = rope_angles.cos().view(1, seq_len, 1, -1) # (1, seq_len, 1, num_pairs)
    sin = rope_angles.sin().view(1, seq_len, 1, -1) 

    # get the pairs
    x0 = x_[..., 0::2] # (batch, seq_len, heads, num_pairs); features 0, 2, 4... (first of each pair)
    x1 = x_[..., 1::2] # (batch, seq_len, heads, num_pairs); features 1, 3, 5...  (second of each pair)

    # apply the rotation matrix equation
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos

    # put the pairs back in their original slots
    out = torch.stack([out0, out1], dim=-1) # (batch, seq_len, heads, num_pairs, 2)
    out = out.flatten(-2) # (batch, seq_len, heads, head_dim)
    return out.type_as(x)
