"""Recurrence for Ouroboros: the looped core and its helpers.

Loop body::

    h_loop   = loop_index_embedding(h, t, loop_dim)
    combined = RMSNorm(h_loop + e)
    R        = TransformerBlock(combined, ...) - combined   # R(h, e): the block's change only
    h        = LTIInjection(h, e, R)   # A*h + B*e + R(h, e)
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn

from .block import TransformerBlock
from .config import OuroborosConfig
from .norm import RMSNorm

__all__ = [
    "loop_index_embedding",
    "sample_n_loops",
    "LTIInjection",
    "RecurrentBlock",
]


def loop_index_embedding(
    h: torch.Tensor,
    loop_t: int,
    loop_dim: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """Add a sinusoidal loop-index signal to the first ``loop_dim`` channels.

    Like RoPE but over loop index, so the shared block can tell loops apart, even
    at loop counts never seen in training. For pair ``k``,
    ``theta_k = theta ** (-2k / loop_dim)`` and the added signal is
    ``[sin(loop_t * theta_k), cos(loop_t * theta_k)]``.

    Args:
        h: ``(B, T, dim)``.
        loop_t: Current loop index (0-based).
        loop_dim: Even number of leading channels to modify.
        theta: Base frequency.

    Returns:
        ``(B, T, dim)``, same dtype and device as ``h``.
    """
    # pair speeds, same as RoPE: early pairs spin fast, later pairs slow
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, loop_dim, 2, device=h.device).float() / loop_dim)
    )  # (loop_dim // 2,)
    angles = loop_t * inv_freq

    # interleave so pair k gets [sin, cos] in channels (2k, 2k + 1)
    signal = torch.stack([angles.sin(), angles.cos()], dim=-1).flatten()  # (loop_dim,)

    # add onto the first loop_dim channels only; the rest pass through untouched
    # (broadcasts over B and T: every token gets the same loop signal)
    head = h[..., :loop_dim] + signal.to(h.dtype)
    return torch.cat([head, h[..., loop_dim:]], dim=-1)


def sample_n_loops(
    mean_loops: int,
    sigma: float = 0.5,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Sample one training step's loop count (Huginn's log-normal Poisson)::

        tau = Normal(log(mean_loops - 1) - sigma**2 / 2, sigma)
        n   = Poisson(exp(tau)) + 1

    So ``E[n] = mean_loops``, with a heavy right tail. Return 1 directly if
    ``mean_loops == 1``.

    Args:
        mean_loops: ``cfg.mean_loops``.
        sigma: Log-normal spread.
        generator: Optional RNG.

    Returns:
        Loop count ``>= 1``. Sample once per optimizer step.
    """
    if mean_loops == 1:
        return 1

    # log-normal rate: the -sigma**2 / 2 shift makes E[exp(tau)] = mean_loops - 1
    mu = math.log(mean_loops - 1) - sigma**2 / 2
    tau = mu + sigma * torch.randn(1, generator=generator)

    # Poisson on top gives integer counts; + 1 so we always run at least one loop
    n = torch.poisson(tau.exp(), generator=generator) + 1
    return int(n.item())


class LTIInjection(nn.Module):
    """Stable input injection (Parcae): ``h = A * h + B * e + R(h, e)``.

    ``A`` is a learned diagonal forced into ``(0, 1)``, so ``rho(A) < 1`` and the
    state cannot blow up across loops::

        A = exp(-exp((log_dt + log_A).clamp(-12, 4)))

    Parameters: ``log_A (dim,)``, ``log_dt (1,)``, ``B (dim,)`` (init about 0.1).
    """

    def __init__(self, dim: int) -> None:
        """Create the LTI parameters.

        Args:
            dim: Hidden width.
        """
        super().__init__()
        # A = exp(-exp(0 + 0)) = exp(-1) ~ 0.37 at init: the old state decays each loop
        self.log_A = nn.Parameter(torch.zeros(dim))
        self.log_dt = nn.Parameter(torch.zeros(1))
        # small B so e is re-injected gently at first
        self.B = nn.Parameter(torch.full((dim,), 0.1))

    def get_A(self) -> torch.Tensor:
        """Return ``A``, shape ``(dim,)``, all in ``(0, 1)``. ``max(A)`` is ``rho(A)``.

        Gotchas:
            - Keep the ``clamp(-12, 4)`` bounds; looser ones let ``A`` round to 1.0
              in fp32.
            - Add in log space; ``exp(log_dt) * exp(log_A)`` can give
              ``0 * inf = NaN``.
        """
        # add in log space, then clamp so exp(-exp(.)) can't round to exactly 0 or 1
        log_rate = (self.log_dt + self.log_A).clamp(-12, 4)
        # exp(log_rate) > 0, so the exponent is negative and A lands in (0, 1)
        return torch.exp(-torch.exp(log_rate))

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """One step: ``A * h + B * e + R(h, e)``, all ``(B, T, dim)``.

        ``r`` is ``R(h, e)``: the shared block's change to its input, not its full
        output.
        """
        # A and B are diagonal, so elementwise multiplies broadcast over (B, T)
        out = self.get_A() * h + self.B * e + r
        return out.type_as(h)


class RecurrentBlock(nn.Module):
    """The looped core: one shared ``TransformerBlock(use_moe=True)`` run ``n_loops`` times.

    Fields: ``block``, ``injection`` (``LTIInjection`` if ``cfg.use_lti`` else
    ``None``), ``norm``, ``loop_dim``, ``backprop_loops``.

    - ``R(h, e) = block(combined) - combined``: only the block's change, so the
      block's own residual copy of ``combined`` stays out of the LTI update.
    - ``use_lti=False`` swaps in the naive ``h = block(combined) + e`` (the
      comparison arm for the stability metric).
    - Truncated backprop: in training, run the first
      ``n_loops - backprop_loops`` loops under ``torch.no_grad()``.
    - Loop ``t`` uses cache key ``recurrent_loop_{t}``, so the cache grows with
      ``n_loops``.
    """

    def __init__(self, cfg: OuroborosConfig) -> None:
        """Build the shared block, the injection, and the loop norm.

        Args:
            cfg: Model configuration.
        """
        super().__init__()
        self.block = TransformerBlock(cfg, use_moe=True)
        self.injection = LTIInjection(cfg.dim) if cfg.use_lti else None
        self.norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.loop_dim = cfg.loop_dim
        self.backprop_loops = cfg.backprop_loops
        self.mean_loops = cfg.mean_loops

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        rope_angles: torch.Tensor,
        causal: bool = True,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        """Run the loop and return the final hidden state.

        Args:
            h: Initial state, ``(B, T, dim)``. The model passes ``e``.
            e: Prelude output, re-injected every loop.
            rope_angles, causal: Passed to the block.
            n_loops: ``None`` means ``cfg.mean_loops``.
            kv_cache: Optional cache dict.

        Returns:
            ``(B, T, dim)``.
        """
        if n_loops is None:
            n_loops = self.mean_loops

        # truncated backprop: in training, only the last backprop_loops build a graph
        n_frozen = max(0, n_loops - self.backprop_loops) if self.training else 0

        for t in range(n_loops):
            with torch.no_grad() if t < n_frozen else nullcontext():
                h = self._step(h, e, t, rope_angles, causal, kv_cache)
        return h

    def _step(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        t: int,
        rope_angles: torch.Tensor,
        causal: bool,
        kv_cache: Optional[dict],
    ) -> torch.Tensor:
        """One loop iteration ``t``: ``h_t -> h_{t+1}``."""
        # tell the shared block which loop this is, then inject e and normalize
        h_loop = loop_index_embedding(h, t, self.loop_dim)
        combined = self.norm(h_loop + e)

        # every loop gets its own cache entry, since each loop sees different K/V
        out = self.block(
            combined,
            rope_angles=rope_angles,
            causal=causal,
            kv_cache=kv_cache,
            cache_key=f"recurrent_loop_{t}",
        )

        if self.injection is None:
            # naive arm: no A, no B, no stability guarantee
            return out + e

        # R(h, e): keep only the block's change, drop its residual copy of combined
        r = out - combined
        # A * h uses h from the start of this loop (no loop index, no norm)
        return self.injection(h, e, r)
