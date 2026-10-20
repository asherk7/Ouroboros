"""Recurrence for Ouroboros: the looped core and its helpers.

Loop body::

    h_loop   = loop_index_embedding(h, t, loop_dim)
    combined = RMSNorm(h_loop + e)
    trans    = TransformerBlock(combined, ...)
    h        = LTIInjection(h, e, trans)   # A*h + B*e + trans
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import OuroborosConfig

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
    raise NotImplementedError


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
    raise NotImplementedError


class LTIInjection(nn.Module):
    """Stable input injection (Parcae): ``h = A * h + B * e + transformer_out``.

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
        raise NotImplementedError

    def get_A(self) -> torch.Tensor:
        """Return ``A``, shape ``(dim,)``, all in ``(0, 1)``. ``max(A)`` is ``rho(A)``.

        Gotchas:
            - Keep the ``clamp(-12, 4)`` bounds; looser ones let ``A`` round to 1.0
              in fp32.
            - Add in log space; ``exp(log_dt) * exp(log_A)`` can give
              ``0 * inf = NaN``.
        """
        raise NotImplementedError

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        """One step: ``A * h + B * e + transformer_out``, all ``(B, T, dim)``."""
        raise NotImplementedError


class RecurrentBlock(nn.Module):
    """The looped core: one shared ``TransformerBlock(use_moe=True)`` run ``n_loops`` times.

    Fields: ``block``, ``injection`` (``LTIInjection`` if ``cfg.use_lti`` else
    ``None``), ``norm``, ``loop_dim``, ``backprop_loops``.

    - ``use_lti=False`` swaps in the naive ``h = trans + e`` (the comparison arm
      for the stability metric).
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
        raise NotImplementedError

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
        raise NotImplementedError
