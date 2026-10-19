"""Dense SwiGLU feed-forward network.

Used as the Prelude/Coda FFN and as each MoE expert; only ``hidden_dim``
differs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["FeedForward"]


class FeedForward(nn.Module):
    """SwiGLU FFN: ``down(silu(gate(x)) * up(x))``, all projections bias-free."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        """Build the gate, up, and down projections.

        Args:
            dim: Input and output width.
            hidden_dim: Inner width.
        """
        super().__init__()
        self.activation = nn.SiLU()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., dim) -> (..., dim)``."""

        x_gated = self.activation(self.gate(x))
        x_content = self.up(x)
        # potential optimization: have one linear for "gate" and "up", do one matmul then split in half
        x = x_gated * x_content
        output = self.down(x)
        return output
