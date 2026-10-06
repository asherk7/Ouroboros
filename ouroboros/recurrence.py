"""Recurrence machinery for the Ouroboros recurrent-depth transformer.

These components turn one shared ``TransformerBlock`` into a looped core. The
loop body (see ``ARCHITECTURE.md`` for the full diagram)::

    h_loop   = loop_index_embedding(h, t, loop_dim)      # which loop is this
    combined = RMSNorm(h_loop + e)                        # re-read the input
    trans    = TransformerBlock(combined, ...)            # GQA + MoE
    h        = LTIInjection(h, e, trans)                  # h = A*h + B*e + trans

Training uses Huginn's recipe (Geiping et al., 2025): each step samples a random
loop count from :func:`sample_n_loops`, and gradients flow through only the last
``backprop_loops`` iterations (truncated backprop). This is what lets the loop
count be changed at test time. Every forward pass runs all of its loops for every
position; there is no halting.

Public API:
    - ``loop_index_embedding``: sinusoidal loop-index signal on the first
      ``loop_dim`` channels.
    - ``sample_n_loops``: Huginn's random loop count for one training step.
    - ``LTIInjection``: Parcae's stable input injection, ``rho(A) < 1`` by
      construction.
    - ``RecurrentBlock``: the looped core.
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

    RoPE's idea applied over loop index instead of token position. The block
    reuses the same weights every loop; without a loop signal it cannot tell an
    early loop from a late one. A fixed sinusoid (no parameters) is defined for
    any ``loop_t``, so it keeps working at loop counts never seen in training,
    unlike a learned per-loop table.

    For frequency pair ``k``: ``theta_k = theta ** (-2k / loop_dim)``; the signal
    is ``[sin(loop_t * theta_k), cos(loop_t * theta_k)]``, added to
    ``h[..., :loop_dim]``. Channels ``loop_dim:`` pass through unchanged.

    Args:
        h: Hidden state, shape ``(B, T, dim)``.
        loop_t: Current loop index (0-based); may exceed anything seen in
            training.
        loop_dim: Number of leading channels that receive the signal. Even,
            ``<= dim``.
        theta: Base frequency. Defaults to ``10000.0``.

    Returns:
        Tensor of shape ``(B, T, dim)``, same dtype and device as ``h``.
    """
    raise NotImplementedError


def sample_n_loops(
    mean_loops: int,
    sigma: float = 0.5,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Sample the loop count for one training step (Huginn's distribution).

    A log-normal Poisson, as in Huginn::

        tau = Normal(log(mean_loops - 1) - sigma**2 / 2, sigma)
        n   = Poisson(exp(tau)) + 1

    ``E[exp(tau)] = mean_loops - 1``, so ``E[n] = mean_loops``. If
    ``mean_loops == 1``, return 1 without sampling (``log(0)``). The result is
    usually near ``mean_loops`` but has a heavy right tail, so the model
    regularly sees much deeper loops than average. Training on many loop counts
    is what makes the test-time loop count adjustable.

    One difference from Huginn: Huginn's draw is only the no-grad prefix, and
    its backprop depth is added on top. Here ``n`` is the total loop count and
    the last ``backprop_loops`` of it get gradients.

    Args:
        mean_loops: Mean of the distribution (``cfg.mean_loops``).
        sigma: Log-normal spread. Huginn uses 0.5.
        generator: Optional RNG for reproducibility.

    Returns:
        A loop count ``>= 1``. Call once per optimizer step and use the same
        value for every micro-batch in that step.
    """
    raise NotImplementedError


class LTIInjection(nn.Module):
    """Stable LTI input injection (Parcae, Prairie et al., 2026).

    The recurrent state follows the linear time-invariant update::

        h_{t+1} = A * h_t  +  B * e  +  Transformer(h_t, e)

    with ``A`` a learned diagonal and ``e`` the frozen Prelude output. If
    ``rho(A) >= 1`` the state grows every loop and training diverges. Gradient
    clipping or renormalizing ``h`` each loop only hide that. Here every entry of
    ``A`` is forced into ``(0, 1)`` for any parameter values, so
    ``rho(A) = max(A) < 1`` always.

    Construction (zero-order-hold discretization of a negative continuous
    system)::

        A_continuous = -exp(log_A)        # strictly negative
        dt           = exp(log_dt)        # strictly positive
        A            = exp(dt * A_continuous)  in (0, 1)

    Computed in log space as ``exp(-exp((log_dt + log_A).clamp(-12, 4)))``.

    Parameters (created when implemented): ``log_A (dim,)``, ``log_dt (1,)``,
    ``B (dim,)`` (init about 0.1).
    """

    def __init__(self, dim: int) -> None:
        """Initialize the LTI parameters.

        Args:
            dim: Hidden width. ``A`` and ``B`` are per-channel; ``dt`` is one
                shared scalar.
        """
        super().__init__()
        raise NotImplementedError

    def get_A(self) -> torch.Tensor:
        """Return the discretized diagonal ``A``, shape ``(dim,)``, all in ``(0, 1)``.

        ``max(get_A())`` is ``rho(A)``, the stability number to log every
        training step (also log the mean and min to see the whole spectrum).

        Gotchas:
            - The ``clamp(-12, 4)`` is essential, and so are its bounds. It
              keeps the inner ``exp`` finite under a huge gradient step and
              keeps ``A`` in about ``[1.9e-24, 0.999994]``, strictly inside
              ``(0, 1)`` in fp32. A looser ``clamp(-20, 20)`` fails: in fp32
              ``exp(-exp(-20))`` rounds to exactly 1.0.
            - Never form ``exp(log_dt) * exp(log_A)`` as a product; ``0 * inf``
              gives NaN. Add in log space first.
        """
        raise NotImplementedError

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        """One LTI step: ``A * h + B * e + transformer_out``.

        Args:
            h: Current state, ``(B, T, dim)``.
            e: Frozen Prelude output, ``(B, T, dim)``.
            transformer_out: This loop's block output, ``(B, T, dim)``.

        Returns:
            ``h_{t+1}``, shape ``(B, T, dim)``. ``A`` and ``B`` broadcast over
            batch and sequence.
        """
        raise NotImplementedError


class RecurrentBlock(nn.Module):
    """The looped core: one shared block run ``n_loops`` times.

    Submodules / fields:
        - ``block``: ``TransformerBlock(cfg, use_moe=True)`` (shared weights).
        - ``injection``: ``LTIInjection(dim)`` if ``cfg.use_lti``, else ``None``.
        - ``norm``: ``RMSNorm(dim)`` applied to ``h_loop + e``.
        - ``loop_dim``: ``cfg.loop_dim``.
        - ``backprop_loops``: ``cfg.backprop_loops``.

    Loop body for ``t in range(n_loops)``:
        1. ``h_loop = loop_index_embedding(h, t, loop_dim)``
        2. ``combined = norm(h_loop + e)``
        3. ``trans = block(combined, freqs_cis, causal, kv_cache,
           cache_key=f"recurrent_loop_{t}")``
        4. ``h = injection(h, e, trans)``. With ``cfg.use_lti=False`` this is
           the naive update ``h = trans + e`` instead: no ``A``, no ``B``, no
           stability guarantee. That arm exists only for the stability metric;
           everything else in the loop is identical, so any difference is the
           LTI update's doing.

    **Truncated backprop (training only).** When ``self.training``, run the
    first ``max(0, n_loops - backprop_loops)`` iterations under
    ``torch.no_grad()`` and the rest with gradients. Memory and backward cost
    then stay flat no matter how large the sampled ``n_loops`` is. ``e`` still
    gets gradient, because it is injected in every loop that has a graph.

    **Design choices (vs Huginn):**
        - ``h_0 = e`` (the Prelude output), where Huginn starts from random
          noise. Simpler and deterministic.
        - The input is added (``h_loop + e`` then norm), where Huginn
          concatenates and projects. Huginn reports adding works as well at
          small scale.

    **KV cache.** Each loop iteration writes its own key,
    ``recurrent_loop_{t}``, because the K/V differ per loop even with shared
    weights. So the cache grows linearly with ``n_loops``.
    """

    def __init__(self, cfg: OuroborosConfig) -> None:
        """Build the shared block, the injection, and the loop norm.

        Args:
            cfg: Uses ``dim``, ``mean_loops``, ``backprop_loops``,
                ``loop_index_dim``, and ``use_lti``.
        """
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        causal: bool = True,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        """Run the loop and return the final hidden state.

        Args:
            h: Initial state ``h_0``, ``(B, T, dim)``. The model passes ``e``.
            e: Frozen Prelude output, re-injected every loop, ``(B, T, dim)``.
            freqs_cis: RoPE phasors sliced to the current positions.
            causal: Passed to attention; ``False`` only for single-token decode.
            n_loops: Number of iterations; ``None`` means ``cfg.mean_loops``.
                Training passes a value from :func:`sample_n_loops`; evaluation
                can pass anything, including far more than seen in training.
            kv_cache: Optional cache dict; loop ``t`` uses key
                ``f"recurrent_loop_{t}"``.

        Returns:
            Final hidden state, ``(B, T, dim)``.
        """
        raise NotImplementedError
