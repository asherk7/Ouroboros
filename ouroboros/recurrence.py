"""Recurrence machinery for the Ouroboros recurrent-depth transformer.

This module contains the components that turn a single shared
``TransformerBlock`` into a recurrent-depth reasoning core that is looped ``T``
times with stable input injection and adaptive halting. Together these
implement the inner ``for t in range(n_loops)`` body of the architecture
(see ARCHITECTURE.md §3 / the canonical data-flow diagram):

    h_loop   = loop_index_embedding(h, t, loop_dim)      # depth signal
    combined = RMSNorm(h_loop + e)                        # re-inject input
    trans    = TransformerBlock(combined, ...)            # MLA/GQA + MoE
    trans    = trans + LoRAAdapter(trans, t)              # depth-wise LoRA
    h        = LTIInjection(h, e, trans)                  # h = A·h + B·e + trans
    p        = ACTHalting(h)                              # per-position halt prob
    accumulate ACT-weighted h into h_out; halt converged positions

Public API (exact signatures are the contract — see the canonical spec §4):
    - ``loop_index_embedding`` — sinusoidal depth signal over the first
      ``loop_dim`` channels (component 9).
    - ``LoRAAdapter`` — depth-wise LoRA delta with per-loop scale and a clamp
      for inference-time depth extrapolation (component 10).
    - ``LTIInjection`` — LTI-constrained stable input injection guaranteeing
      ``ρ(A) < 1`` by construction (component 11).
    - ``ACTHalting`` — Adaptive Computation Time per-position halting probability
      (component 12).
    - ``RecurrentBlock`` — the looped core that owns the above and runs the loop
      body with the ACT remainder trick and KV-cache-aware early exit
      (component 13).

This is an *independent implementation inspired by the recurrent-depth
transformer literature* — Parcae (Prairie et al., 2026) for the LTI injection
and spectral-radius constraint, Universal Transformers (Dehghani et al., 2018)
and Adaptive Computation Time (Graves, 2016) for ACT halting, Relaxed Recursive
Transformers (Bae et al., 2024) for depth-wise LoRA, and Saunshi et al. (2025)
for the depth-extrapolation property of looped transformers.

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
    "LoRAAdapter",
    "LTIInjection",
    "ACTHalting",
    "RecurrentBlock",
]


# ---------------------------------------------------------------------------
# (9) Loop-index embedding — sinusoidal signal over recurrence DEPTH
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
# (10) LoRAAdapter — depth-wise LoRA delta with per-loop scale
# ---------------------------------------------------------------------------


class LoRAAdapter(nn.Module):
    """Depth-wise LoRA adaptation for the recurrent block (Bae et al., 2024).

    Pure weight-tying (identical weights at every loop) limits expressiveness,
    while fully distinct weights per loop forfeit the parameter savings that
    make a recurrent-depth model attractive. This adapter is the relaxed middle
    ground from Relaxed Recursive Transformers: a *shared* low-rank
    down-projection and a *shared* up-projection matrix ``B`` are reused across
    all loops, while a tiny *per-loop* scale vector modulates the bottleneck so
    the effective transformation differs at each depth for almost no extra
    parameters.

    Delta computed at loop ``t``::

        delta(x, t) = (down(x) * scale[t]) @ B

    where ``down: Linear(dim, rank)`` and ``B: Parameter(rank, dim)`` are shared,
    and ``scale: Embedding(max_loops, rank)`` provides the per-loop modulation.
    The result is added to the recurrent ``TransformerBlock`` output.

    NOTE: Stub — ``__init__`` declares no layers/parameters yet; the body raises
    ``NotImplementedError`` per the planning-phase contract.
    """

    def __init__(self, dim: int, rank: int, max_loops: int) -> None:
        """Initialize the depth-wise LoRA adapter.

        Args:
            dim: Model hidden dimension (input and output size of the delta).
            rank: Low-rank bottleneck dimension; ``rank << dim`` keeps the
                adapter cheap.
            max_loops: Number of *trained* loop iterations. Sizes the per-loop
                ``scale`` embedding table (rows ``0 .. max_loops - 1``).
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        """Compute the per-loop LoRA delta for the current depth.

        Args:
            x: Input tensor (the recurrent block output) of shape ``(B, T, dim)``.
            loop_t: Current loop iteration index used to look up the per-loop
                scale row.

        Returns:
            Delta tensor of shape ``(B, T, dim)`` to be added to the block output.

        Shapes:
            x: ``(B, T, dim)`` -> returns ``(B, T, dim)``.

        Gotcha (depth extrapolation):
            At inference ``loop_t`` can exceed ``max_loops - 1`` (the model may
            be run for more loops than it was trained on). The lookup index MUST
            be clamped to the last learned row (``min(loop_t, max_loops - 1)``)
            so the embedding is never indexed out of range — extra loops reuse
            the final trained scale rather than erroring or wrapping.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# (11) LTIInjection — LTI-constrained stable input injection (ρ(A) < 1)
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
            transformer_out: Output of the recurrent ``TransformerBlock`` (with
                the LoRA delta already added) at this loop step, shape
                ``(B, T, dim)``.

        Returns:
            Updated hidden state ``h_{t+1}`` of shape ``(B, T, dim)``. ``A`` and
            ``B`` broadcast over the batch and sequence axes (per-channel
            diagonal scaling).

        Shapes:
            h, e, transformer_out: ``(B, T, dim)`` -> returns ``(B, T, dim)``.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# (12) ACTHalting — Adaptive Computation Time per-position halting probability
# ---------------------------------------------------------------------------


class ACTHalting(nn.Module):
    """Adaptive Computation Time halting head (Graves, 2016).

    Predicts a per-position scalar halting probability from the current hidden
    state via ``sigmoid(Linear(dim, 1)(h))``. The ``RecurrentBlock`` accumulates
    these probabilities across loop iterations: a position whose cumulative
    halting probability crosses ``act_threshold`` is considered converged and
    stops receiving further computation, while unconverged positions keep
    looping. This yields variable compute per position within a single batch —
    easy tokens halt early, hard tokens loop deeper — and underpins the
    continuous depth-wise batching inference optimization.

    NOTE: Stub — ``__init__`` declares no layers yet; the body raises
    ``NotImplementedError`` per the planning-phase contract.
    """

    def __init__(self, dim: int) -> None:
        """Initialize the halting head.

        Args:
            dim: Hidden-state dimension; input width of the halting scalar
                predictor (a ``Linear(dim, 1)``).
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Predict per-position halting probability from the hidden state.

        Args:
            h: Hidden state of shape ``(B, T, dim)``.

        Returns:
            Halting-probability tensor of shape ``(B, T)`` with values in
            ``(0, 1)`` (the trailing size-1 channel axis is squeezed away).

        Shapes:
            h: ``(B, T, dim)`` -> returns ``(B, T)``.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# (13) RecurrentBlock — the looped core (loop body + ACT + KV-cache early exit)
# ---------------------------------------------------------------------------


class RecurrentBlock(nn.Module):
    """The recurrent-depth core of Ouroboros — one block looped ``T`` times.

    Owns a single shared ``TransformerBlock`` (with MoE FFN) plus all the
    recurrence machinery, and runs the loop body from the canonical data-flow
    diagram (spec §3). The encoded input ``e`` (Prelude output) is frozen and
    re-injected at every iteration so the original prompt signal survives
    arbitrary loop depth, while the LTI injection keeps the recurrence stable
    (``ρ(A) < 1``) and ACT halting decides per-position when to stop.

    Owned submodules / fields:
        - ``block``     = ``TransformerBlock(cfg, use_moe=True)`` (shared weights).
        - ``injection`` = ``LTIInjection(dim)``.
        - ``act``       = ``ACTHalting(dim)``.
        - ``lora``      = ``LoRAAdapter(dim, lora_rank, max_loop_iters)``.
        - ``norm``      = ``RMSNorm(dim)`` applied to ``h_loop + e``.
        - ``loop_dim``  = ``loop_index_dim`` if set else ``dim // 8`` — the number
          of channels that receive the loop-index embedding.

    Loop body, for ``t in range(n_loops)`` (matches §3 exactly):
        1. ``h_loop  = loop_index_embedding(h, t, loop_dim)``.
        2. ``combined = norm(h_loop + e)``  — re-inject the frozen input.
        3. ``trans   = block(combined, freqs_cis, mask, kv_cache,
           cache_key=f"recurrent_loop_{t}")`` — each depth uses a DISTINCT cache
           key so loop caches never collide.
        4. ``trans  += lora(trans, t)``  — depth-wise LoRA delta.
        5. ``h       = injection(h, e, trans)``  — LTI update.
        6. ``p       = act(h)``  — per-position halting probability ``(B, T)``.
        7. ACT accumulation (remainder trick, see below); update
           ``cumulative_p`` and ``halted``; accumulate ``h_out``.

    ACT remainder trick:
        Maintain ``halted: (B, T) bool``, ``cumulative_p: (B, T)`` and
        ``h_out: (B, T, dim)``. For positions still running, the per-step weight
        is::

            weight = where(cumulative_p + p >= act_threshold,
                           1 - cumulative_p,   # remainder: close out exactly to 1
                           p)                  # otherwise the raw halting prob

        gated by ``still_running = ~halted`` so a halted position contributes
        exactly once (on its halting step) and zero thereafter — otherwise a
        ``threshold < 1`` would leak a non-zero remainder every subsequent step.
        Then ``h_out += weight.unsqueeze(-1) * h``, ``cumulative_p`` advances by
        ``p`` on still-running positions, and positions crossing the threshold
        are marked ``halted``. The returned ``h_out`` is the ACT-weighted sum of
        hidden states across iterations.

    CRITICAL gotcha (the headline subtlety — KV-cache ⊗ ACT short-circuit):
        Early-exit (``if halted.all(): break``) is ONLY valid when
        ``kv_cache is None``. With a KV cache present, EVERY loop depth must run
        on EVERY forward pass: each iteration writes keys/values under its own
        ``recurrent_loop_{t}`` cache key, and a later autoregressive decode step
        that loops to depth ``t`` must find those keys already populated. If the
        loop short-circuits while caching, deeper decode steps read missing keys
        and generation breaks. This same constraint is the crux of continuous
        depth-wise batching (component 17): so the break condition is
        ``if halted.all() and kv_cache is None``.

    NOTE: Stub — ``__init__`` declares no submodules yet; the body raises
    ``NotImplementedError`` per the planning-phase contract.
    """

    def __init__(self, cfg: OuroborosConfig) -> None:
        """Initialize the recurrent block and its recurrence machinery.

        Args:
            cfg: ``OuroborosConfig``. Uses ``dim``, ``lora_rank``,
                ``max_loop_iters``, ``act_threshold``, and ``loop_index_dim``
                (falling back to ``dim // 8`` when ``loop_index_dim`` is ``None``).
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
        """Run the recurrent loop with input injection and ACT halting.

        Executes the loop body (see the class docstring / spec §3) for up to
        ``n_loops`` iterations, accumulating the ACT-weighted sum of hidden
        states. Early exit when all positions have halted is permitted ONLY when
        ``kv_cache is None`` (see the CRITICAL gotcha in the class docstring).

        Args:
            h: Initial hidden state from the Prelude, shape ``(B, T, dim)``.
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
                ``f"recurrent_loop_{t}"``. When non-``None``, early exit is
                disabled so every depth's cache stays populated.

        Returns:
            ACT-weighted sum of hidden states across iterations, shape
            ``(B, T, dim)``.

        Shapes:
            h, e: ``(B, T, dim)`` -> returns ``(B, T, dim)``.
        """
        raise NotImplementedError
