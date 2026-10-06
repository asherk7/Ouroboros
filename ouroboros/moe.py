"""Mixture-of-Experts feed-forward components for Ouroboros.

- ``Expert``: one SwiGLU feed-forward network. Used as a routed expert, as a
  shared expert, and as the dense FFN in the Prelude/Coda blocks (width
  ``dim * 4 // 3`` there).
- ``MoEFFN``: fine-grained MoE with ``n_experts`` routed experts plus
  ``n_shared_experts`` always-on shared experts. Only the recurrent block uses it.

Design notes:

- **Fine-grained experts (DeepSeekMoE).** Many small experts give a richer set
  of expert combinations than a few large ones at the same active parameter
  count. Rule of thumb: ``expert_dim ~= dim // (n_experts // n_experts_per_tok)``.
- **Shared experts (DeepSeekMoE).** Fire for every token and absorb common
  patterns so routed experts can specialize. Width
  ``expert_dim * n_experts_per_tok``.
- **Aux-loss-free load balancing (DeepSeek-V3).** A non-gradient
  ``router_bias`` buffer is added to the logits for expert SELECTION only. The
  gating WEIGHTS come from the unbiased ``softmax(logits)``, so the bias never
  touches the gradient or the LM loss.
- **The bias update.** Many reference implementations register ``router_bias``
  but never update it, so balancing never happens. Here an ``expert_load``
  buffer counts selections during ``forward`` and
  :meth:`MoEFFN.update_router_bias` (called once per optimizer step) nudges the
  bias toward balance.
- **Dispatch.** A simple masked loop over experts is fine at this size.

References: DeepSeekMoE (https://arxiv.org/abs/2401.06066); DeepSeek-V3
(https://arxiv.org/abs/2412.19437); SwiGLU (https://arxiv.org/abs/2002.05202).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import OuroborosConfig

__all__ = ["Expert", "MoEFFN"]


class Expert(nn.Module):
    """A single SwiGLU feed-forward network: ``down(silu(gate(x)) * up(x))``.

    All three projections are bias-free. Reused as the dense Prelude/Coda FFN
    with ``expert_dim = dim * 4 // 3``.
    """

    def __init__(self, dim: int, expert_dim: int) -> None:
        """Initialize the SwiGLU projections.

        Args:
            dim: Input and output width (the residual-stream width).
            expert_dim: Hidden width. Routed experts use ``cfg.expert_dim``;
                shared experts use ``cfg.expert_dim * cfg.n_experts_per_tok``;
                dense Prelude/Coda FFNs use ``dim * 4 // 3``.
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SwiGLU transform.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Tensor of shape ``(..., dim)``.
        """
        raise NotImplementedError


class MoEFFN(nn.Module):
    """Fine-grained MoE FFN with shared experts and aux-loss-free balancing.

    Routing, for ``N = B * T`` tokens::

        logits   = router(x)                          # (N, n_experts)
        scores   = softmax(logits)                    # weights: UNBIASED
        topk_idx = topk(logits + router_bias, K)      # selection: biased
        topk_w   = scores.gather(topk_idx)
        topk_w   = topk_w / topk_w.sum(-1, keepdim)   # renormalize over K
        out      = sum_k topk_w[k] * routed[topk_idx[k]](x) + sum_s shared[s](x)

    Submodules:
        router: ``Linear(dim, n_experts, bias=False)``.
        routed: ``n_experts`` x ``Expert(dim, expert_dim)``.
        shared: ``n_shared_experts`` x ``Expert(dim, expert_dim * n_experts_per_tok)``.

    Buffers (non-gradient, ``register_buffer``):
        router_bias: ``(n_experts,)``, zeros at init. Updated only by
            :meth:`update_router_bias`, never by autograd.
        expert_load: ``(n_experts,)``. Selection counts since the last bias
            update; consumed and reset by :meth:`update_router_bias`.

    Gotchas:
        - Renormalize the top-K weights, or the routed output's scale depends on
          how confident the router happened to be.
        - ``router_bias`` must be a buffer. If it ever gets a gradient, the
          aux-loss-free property is gone.
    """

    def __init__(self, cfg: OuroborosConfig) -> None:
        """Build the router, routed experts, shared experts, and buffers.

        Args:
            cfg: Uses ``dim``, ``n_experts``, ``n_shared_experts``,
                ``n_experts_per_tok``, ``expert_dim``, and
                ``router_bias_update_rate``.
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route tokens to their top-K experts and add the shared experts.

        Accumulate per-expert selection counts into ``expert_load`` only while
        ``self.training`` is ``True``: eval and generation passes must not
        pollute the balance signal.

        Args:
            x: Input tensor of shape ``(B, T, dim)``.

        Returns:
            Tensor of shape ``(B, T, dim)``.
        """
        raise NotImplementedError

    @torch.no_grad()
    def update_router_bias(self) -> None:
        """Apply one aux-loss-free load-balancing step, then reset the counter.

        Moves the bias down for overloaded experts and up for underloaded ones::

            router_bias -= router_bias_update_rate * sign(load - mean_load)

        Call once per optimizer step, not per micro-batch: under gradient
        accumulation ``expert_load`` should cover the whole effective batch.
        """
        raise NotImplementedError
