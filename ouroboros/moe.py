"""Fine-grained Mixture-of-Experts FFN, used only in the recurrent block.

References: DeepSeekMoE (https://arxiv.org/abs/2401.06066), DeepSeek-V3
(https://arxiv.org/abs/2412.19437).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import OuroborosConfig
from .feedforward import FeedForward  # noqa: F401  (used once implemented)

__all__ = ["MoEFFN"]


class MoEFFN(nn.Module):
    """MoE FFN with routed + shared experts and aux-loss-free load balancing::

        logits   = router(x)
        topk_idx = topk(logits + router_bias, K)     # bias affects selection only
        topk_w   = softmax(logits).gather(topk_idx)  # then renormalize over K
        out      = sum_k topk_w[k] * routed[topk_idx[k]](x) + sum_s shared[s](x)

    Routed experts are ``FeedForward(dim, expert_dim)``. Shared experts are
    ``FeedForward(dim, expert_dim * n_experts_per_tok)`` and see every token.
    ``router_bias`` and ``expert_load`` are buffers (no gradient), changed only by
    :meth:`update_router_bias`.
    """

    def __init__(self, cfg: OuroborosConfig) -> None:
        """Build the router, experts, and buffers.

        Args:
            cfg: Model configuration.
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route tokens to their top-K experts and add the shared experts.

        Count selections into ``expert_load`` only when ``self.training``.
        ``(B, T, dim) -> (B, T, dim)``.
        """
        raise NotImplementedError

    @torch.no_grad()
    def update_router_bias(self) -> None:
        """Nudge ``router_bias`` toward balanced load, then reset ``expert_load``::

            router_bias -= router_bias_update_rate * sign(load - mean_load)

        Call once per optimizer step, not per micro-batch.
        """
        raise NotImplementedError
