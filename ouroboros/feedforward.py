"""The feed-forward layer for Ouroboros: a dense SwiGLU network.

``FeedForward`` is the second sublayer of every transformer block (attention is
the first). It is always a dense SwiGLU network::

    FeedForward(x) = down(silu(gate(x)) * up(x))

It is used in three places, differing only in hidden width:

- the dense FFN in every Prelude and Coda block (``hidden_dim = dim * 8 // 3``);
- each routed expert inside :class:`~ouroboros.moe.MoEFFN`
  (``hidden_dim = cfg.expert_dim``);
- each shared expert inside :class:`~ouroboros.moe.MoEFFN`
  (``hidden_dim = cfg.expert_dim * cfg.n_experts_per_tok``).

An "expert" in the MoE is just one of these networks behind a router.

Reference: GLU Variants Improve Transformer (https://arxiv.org/abs/2002.05202).
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["FeedForward"]


class FeedForward(nn.Module):
    """Dense SwiGLU feed-forward network: ``down(silu(gate(x)) * up(x))``.
        
    Three bias-free projections: 
    - ``gate`` and ``up`` map ``dim -> hidden_dim``
    - ``down`` maps ``hidden_dim -> dim``
    - ``silu(z) = z * sigmoid(z)``
    - the gate multiplies ``up(x)`` elementwise.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        """Initialize the SwiGLU projections.

        Args:
            dim: Input and output width (the residual-stream width).
            hidden_dim: Inner width. ``dim * 8 // 3`` for the dense Prelude/Coda
                FFN; ``cfg.expert_dim`` for a routed expert;
                ``cfg.expert_dim * cfg.n_experts_per_tok`` for a shared expert.
        """
        super().__init__()
        self.activation = nn.SiLU()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SwiGLU transform.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Tensor of shape ``(..., dim)``.
        """

        x_gated = self.activation(self.gate(x))
        x_content = self.up(x)
        # potential optimization: have one linear for "gate" and "up", do one matmul then split in half
        x = x_gated * x_content
        output = self.down(x)
        return output
