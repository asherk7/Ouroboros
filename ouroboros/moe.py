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
        self.n_experts = cfg.n_experts # number of total (routed) experts, not including shared
        self.top_k = cfg.n_experts_per_tok # top-k routed per token, not including shared
        self.update_rate = cfg.router_bias_update_rate 

        self.expert_dim = cfg.expert_dim # hidden dimension of the fine-grained experts
        self.dim = cfg.dim # tokens regular hidden dimension
        self.n_shared_experts = cfg.n_shared_experts # number of shared experts

        self.router = nn.Linear(self.dim, self.n_experts, bias=False)
        self.routed_experts = nn.ModuleList(
            [FeedForward(self.dim, self.expert_dim) for e in range(self.n_experts)]
        )

        self.shared_experts = nn.ModuleList(
            [FeedForward(self.dim, self.expert_dim * self.top_k) for e in range(self.n_shared_experts)]
        )

        self.register_buffer("router_bias", torch.zeros(self.n_experts))
        self.register_buffer("expert_load", torch.zeros(self.n_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route tokens to their top-K experts and add the shared experts.

        Count selections into ``expert_load`` only when ``self.training``.
        ``(B, T, dim) -> (B, T, dim)``.
        """
        # flatten the tokens into one list so the router can route each token 
        B, S, D = x.shape
        x_flat = x.view(-1, D) # (batch * seq_len, dim)

        # route em
        logits = self.router(x_flat)
        scores = logits.softmax(dim=-1) # (batch * seq_len, n_experts), use these as weights for the sum

        # pick the top-k experts using the added bias, (batch * seq_len, topk), topk[batch * seq_len] = [idx1, idx2]
        _, topk_idx = (logits + self.router_bias).topk(self.top_k, dim=-1) 

        # normalize the topk probabilities to get weights that sum to 1
        topk_w = scores.gather(-1, topk_idx) 
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True) 

        if self.training: # set automatically with model.train() or model.eval()
            self.expert_load += torch.bincount(topk_idx.flatten(), minlength=self.n_experts)

        # send tokens to their experts
        out = torch.zeros_like(x_flat) # (seq_len*batch, dim)
        for e, expert in enumerate(self.routed_experts):
            token_idx, slot = torch.where(topk_idx == e) # which tokens picked expert e, and in which of their K slots
            if token_idx.numel() == 0: # if no tokens picked the expert
                continue 
            # n_e is the number of tokens that picked expert e
            weight = topk_w[token_idx, slot].unsqueeze(-1) # (n_e, 1)
            out[token_idx] += weight * expert(x_flat[token_idx]) # (n_e, dim)

        for expert in self.shared_experts:
            out = out + expert(x_flat) # (batch * seq_len, dim)
        
        return out.view(B, S, D)

    @torch.no_grad()
    def update_router_bias(self) -> None:
        """Nudge ``router_bias`` toward balanced load, then reset ``expert_load``::

            router_bias -= router_bias_update_rate * sign(load - mean_load)

        Call once per optimizer step, not per micro-batch.
        """
        self.router_bias -= self.update_rate * torch.sign(self.expert_load - self.expert_load.mean())
        self.expert_load.zero_()

