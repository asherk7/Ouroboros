"""Recurrence machinery for the Ouroboros recurrent-depth transformer.

This module contains the components that turn a single shared
``TransformerBlock`` into a recurrent-depth reasoning core that is looped a
fixed ``n_loops`` times with stable input injection. Together these implement
the inner ``for t in range(n_loops)`` body of the architecture (see
ARCHITECTURE.md / the canonical data-flow diagram):

    h_loop   = loop_index_embedding(h, t, loop_dim)      # depth signal
    combined = RMSNorm(h_loop + e)                        # re-inject input
    trans    = TransformerBlock(combined, ...)            # GQA + MoE
    h        = LTIInjection(h, e, trans)                  # h = A·h + B·e + trans

The loop runs a fixed number of iterations (``n_loops``) and returns the final
hidden state — there is no adaptive halting; every position is refined for the
same number of loops.

Public API (exact signatures are the contract):
    - ``loop_index_embedding`` — sinusoidal depth signal over the first
      ``loop_dim`` channels (component 8).
    - ``LTIInjection`` — LTI-constrained stable input injection guaranteeing
      ``ρ(A) < 1`` by construction (component 9).
    - ``RecurrentBlock`` — the looped core that owns the above and runs the
      fixed-depth loop body (component 10).

This is an *independent implementation inspired by the recurrent-depth
transformer literature* — Parcae (Prairie et al., 2026) for the LTI injection
and spectral-radius constraint, Universal Transformers (Dehghani et al., 2018)
for the looped-block formulation, and Saunshi et al. (2025) for the
depth-extrapolation property of looped transformers.

NOTE: This is a planning/documentation-phase stub. Bodies raise
``NotImplementedError``; no implementation logic, tensor math, or layer/parameter
declarations live here yet (see the project build spec).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import OuroborosConfig

__all__ = [
    "loop_index_embedding",
    "LTIInjection",
    "RecurrentBlock",
]


# ---------------------------------------------------------------------------
# (8) Loop-index embedding — sinusoidal signal over recurrence DEPTH
# ---------------------------------------------------------------------------


def loop_index_embedding(
    h: torch.Tensor,
    loop_t: int,
    loop_dim: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """Inject a sinusoidal loop-index signal into the first ``loop_dim`` channels.

    This is the depth-domain analogue of RoPE: instead of encoding *token
    position* it encodes the *recurrence iteration index* ``loop_t``. The same
    recurrent-block weights are reused at every depth, so without a per-loop
    signal the block cannot distinguish an early refinement step from a late
    one. Adding a fixed sinusoidal bias keyed on ``loop_t`` lets the shared
    parameters behave functionally differently at each depth.

    Construction (no learned parameters; computed deterministically from
    ``loop_t``): for each frequency pair ``k`` use
    ``θ_k = theta ** (-2k / loop_dim)`` and build the angle ``loop_t · θ_k``,
    then take ``sin`` / ``cos`` to form a ``loop_dim``-length signal. That signal
    is *added as a bias* to the first ``loop_dim`` channels of ``h``; the
    remaining ``dim - loop_dim`` channels pass through unmodified.

    Args:
        h: Hidden state, shape ``(B, T, dim)``.
        loop_t: Current loop iteration index (0-based). ``loop_t = 0`` is the
            first loop; larger values shift the sinusoid. May exceed the trained
            depth at inference (depth extrapolation) — the signal is still
            well-defined for any non-negative integer.
        loop_dim: Number of leading channels that receive the embedding. Must be
            even (it is consumed as adjacent sin/cos pairs) and ``<= dim``.
        theta: Sinusoidal base frequency controlling how fast successive pairs
            decay in frequency. Defaults to ``10000.0``.

    Returns:
        Tensor of shape ``(B, T, dim)``, equal to ``h`` with the sinusoidal bias
        added to its first ``loop_dim`` channels only. Same dtype/device as ``h``.

    Shapes:
        h: ``(B, T, dim)`` -> returns ``(B, T, dim)``.

    Gotcha:
        Only the first ``loop_dim`` channels are altered; the remaining channels
        must be returned untouched so the residual stream is not globally biased.
        ``loop_dim`` must be even.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# (9) LTIInjection — LTI-constrained stable input injection (ρ(A) < 1)
# ---------------------------------------------------------------------------


class LTIInjection(nn.Module):
    """Stable LTI-constrained input injection (Parcae, Prairie et al., 2026).

    The recurrent hidden state evolves under the discrete linear time-invariant
    update::

        h_{t+1} = A · h_t  +  B · e  +  Transformer(h_t, e)

    where ``e`` is the frozen encoded input (Prelude output) re-injected at every
    loop step to prevent the recurrence from drifting away from the prompt.
    ``A`` is a learned *diagonal* state matrix. Left unconstrained, ``A`` can
    acquire spectral radius ``ρ(A) >= 1`` and the hidden state explodes across
    loop iterations, destabilizing training (especially at high learning rates).

    **Guarantee — ``ρ(A) < 1`` by construction (ZOH discretization).** A diagonal
    matrix's spectral radius is the max absolute diagonal entry, so it suffices
    to force every entry into ``(0, 1)``. We parameterize a strictly *negative*
    continuous-time diagonal and discretize it with a positive step ``Δt`` via
    zero-order hold (ZOH):

        A_continuous = -exp(log_A)            # strictly negative, any log_A
        Δt           = exp(log_dt)            # strictly positive
        A_discrete   = exp(Δt · A_continuous) # element-wise, lands in (0, 1)

    Because ``Δt · A_continuous`` is strictly negative, ``exp(·)`` of it is
    strictly in ``(0, 1)``; hence every diagonal entry — and therefore
    ``ρ(A) = max(A_discrete)`` — is ``< 1`` for *any* values of the learned
    parameters ``log_A`` and ``log_dt``. No gradient clipping, hidden-state
    LayerNorm, or other band-aid is needed; stability is structural.

    **Numerical form (essential).** Compute ``A`` entirely in log space::

        A = exp( -exp( (log_dt + log_A).clamp(-20, 20) ) )

    The fused ``-exp(log_dt + log_A)`` avoids forming ``Δt · A_continuous`` as a
    product ``exp(log_dt) * exp(log_A)``, which can be ``0 · inf -> NaN`` when one
    factor under/overflows. The ``clamp(-20, 20)`` keeps the inner exponent
    finite in float32 under aggressive gradient steps: at ``-20`` the inner
    ``exp`` underflows toward 0 so ``A -> 1``; at ``+20`` the inner ``exp`` is
    large so ``A -> 0``. Both extremes stay safely inside ``(0, 1)``.

    Parameters (declared at implementation time, not in this stub):
        ``log_A`` of shape ``(dim,)``, ``log_dt`` of shape ``(1,)``, and the
        injection gain ``B`` of shape ``(dim,)`` (init ~0.1).

    NOTE: Stub — ``__init__`` declares no parameters yet; the body raises
    ``NotImplementedError`` per the planning-phase contract.
    """

    def __init__(self, dim: int) -> None:
        """Initialize the LTI injection parameters.

        Args:
            dim: Hidden-state dimension. ``A`` and ``B`` are per-channel diagonal
                vectors of length ``dim``; ``Δt`` is a single shared scalar.
        """
        super().__init__()
        raise NotImplementedError

    def get_A(self) -> torch.Tensor:
        """Compute the discretized diagonal state matrix ``A_discrete``.

        Implements the ZOH discretization in log space::

            A = exp( -exp( (log_dt + log_A).clamp(-20, 20) ) )

        Returns:
            1-D tensor of shape ``(dim,)`` with every value strictly in ``(0, 1)``,
            guaranteeing ``ρ(A) = max(A) < 1`` regardless of the learned parameter
            values. This same max is the cheap per-step spectral-radius signal
            logged during training (the centerpiece of the stability experiment).

        Shapes:
            -> ``(dim,)``.

        Gotcha:
            The ``clamp(-20, 20)`` is essential — without it the inner ``exp`` can
            overflow/underflow under aggressive gradient steps and the fused
            ``-exp(...)`` can produce ``NaN``. Compute in log space (never form
            ``exp(log_dt) * exp(log_A)`` as a product) to avoid ``0 · inf``.
        """
        raise NotImplementedError

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one LTI recurrence step: ``A · h + B · e + transformer_out``.

        Args:
            h: Current recurrent hidden state, shape ``(B, T, dim)``.
            e: Frozen encoded input from the Prelude, re-injected every loop,
                shape ``(B, T, dim)``.
            transformer_out: Output of the recurrent ``TransformerBlock`` at this
                loop step, shape ``(B, T, dim)``.

        Returns:
            Updated hidden state ``h_{t+1}`` of shape ``(B, T, dim)``. ``A`` and
            ``B`` broadcast over the batch and sequence axes (per-channel
            diagonal scaling).

        Shapes:
            h, e, transformer_out: ``(B, T, dim)`` -> returns ``(B, T, dim)``.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# (10) RecurrentBlock — the looped core (fixed-depth loop body)
# ---------------------------------------------------------------------------


class RecurrentBlock(nn.Module):
    """The recurrent-depth core of Ouroboros — one block looped ``n_loops`` times.

    Owns a single shared ``TransformerBlock`` (with MoE FFN) plus the input
    injection, and runs the loop body from the canonical data-flow diagram. The
    encoded input ``e`` (Prelude output) is frozen and re-injected at every
    iteration so the original prompt signal survives arbitrary loop depth, while
    the LTI injection keeps the recurrence stable (``ρ(A) < 1``).

    Owned submodules / fields:
        - ``block``     = ``TransformerBlock(cfg, use_moe=True)`` (shared weights).
        - ``injection`` = ``LTIInjection(dim)`` if ``cfg.use_lti`` else ``None``.
        - ``norm``      = ``RMSNorm(dim)`` applied to ``h_loop + e``.
        - ``loop_dim``  = ``loop_index_dim`` if set else ``dim // 8`` — the number
          of channels that receive the loop-index embedding.

    Loop body, for ``t in range(n_loops)`` (fixed depth, no early exit):
        1. ``h_loop  = loop_index_embedding(h, t, loop_dim)``.
        2. ``combined = norm(h_loop + e)``  — re-inject the frozen input.
        3. ``trans   = block(combined, freqs_cis, mask, kv_cache,
           cache_key=f"recurrent_loop_{t}")`` — each depth uses a DISTINCT cache
           key so loop caches never collide.
        4. ``h       = injection(h, e, trans)``  — LTI update. When
           ``cfg.use_lti`` is ``False`` (the stability-ablation arm,
           EXPERIMENTS.md exp 1), this step is instead the naive residual
           injection ``h = trans + e`` — no ``A``/``B`` parameters and no
           spectral-radius guarantee, which is exactly what the ablation
           measures.

    After ``n_loops`` iterations the final hidden state ``h`` is returned. Because
    the loop always runs every depth, a KV cache is populated at every
    ``recurrent_loop_{t}`` key on every forward pass, so later autoregressive
    decode steps always find the keys they need — there is no early-exit subtlety
    in this core loop. (Inference-time per-sequence early exit, used by continuous
    depth-wise batching, lives in
    :meth:`~ouroboros.model.Ouroboros.generate_depthwise_batched` as a separate
    convergence-based optimization.)

    NOTE: Stub — ``__init__`` declares no submodules yet; the body raises
    ``NotImplementedError`` per the planning-phase contract.
    """

    def __init__(self, cfg: OuroborosConfig) -> None:
        """Initialize the recurrent block and its injection machinery.

        Args:
            cfg: ``OuroborosConfig``. Uses ``dim``, ``max_loop_iters``,
                ``loop_index_dim`` (falling back to ``dim // 8`` when
                ``loop_index_dim`` is ``None``), and ``use_lti`` (``False``
                builds no ``LTIInjection`` and uses the naive residual update —
                the ablation arm).
        """
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        """Run the fixed-depth recurrent loop with input injection.

        Executes the loop body (see the class docstring) for exactly ``n_loops``
        iterations and returns the final hidden state. Every depth runs on every
        call, so all ``recurrent_loop_{t}`` cache keys stay populated.

        Args:
            h: Initial hidden state ``h_0``, shape ``(B, T, dim)``. The model
                passes the Prelude output here — i.e. ``h_0 = e``; the two
                arguments start as the same tensor and diverge as ``h`` is
                updated across iterations.
            e: Frozen encoded input re-injected at every step, shape
                ``(B, T, dim)``.
            freqs_cis: Precomputed RoPE frequencies for the active attention type,
                already sliced to the processed positions by the caller.
            mask: Additive causal mask of shape ``(1, 1, T, S)`` (dtype matching
                the activations), or ``None`` during single-token decode.
            n_loops: Number of loop iterations; defaults to
                ``cfg.max_loop_iters``. May be raised at inference for deeper
                reasoning (depth extrapolation).
            kv_cache: KV-cache dict mutated in-place by the inner
                ``TransformerBlock``. Each loop depth uses a distinct cache key
                ``f"recurrent_loop_{t}"``.

        Returns:
            Final hidden state after ``n_loops`` iterations, shape ``(B, T, dim)``.

        Shapes:
            h, e: ``(B, T, dim)`` -> returns ``(B, T, dim)``.
        """
        raise NotImplementedError
