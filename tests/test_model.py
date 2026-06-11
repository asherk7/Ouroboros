"""Integration-test stubs for the full :class:`~ouroboros.model.Ouroboros` model.

These exercise the assembled model end to end — embedding through prelude,
recurrent block, coda, and the tied LM head. Each test documents its assertion
and skips with ``pytest.skip("stub — implement in Phase N")`` so the suite is
collectable and GREEN (all skips) before the model is implemented.

Phase mapping (project roadmap):

* Phase 5 — full model and KV-cached generation: forward shape, forward no-NaN,
  ``generate`` shape, weight tying, end-to-end ``ρ(A) < 1``, the ``use_lti``
  ablation arm, depth extrapolation, cached-decode ≈ full-context, and the
  single-token (``T=1``) forward.
* Phase 7 — inference optimization: continuous depth-wise batched generation
  (:meth:`~ouroboros.model.Ouroboros.generate_depthwise_batched`).
"""

from __future__ import annotations

from typing import Any

import pytest

# These imports ARE part of the contract: collection fails loudly if any public
# name is renamed or removed. ``Ouroboros`` is referenced only inside skipped
# bodies for now, so F401 is suppressed while every test remains a stub.
from ouroboros import Ouroboros, OuroborosConfig  # noqa: F401


def tiny_config(**overrides: Any) -> OuroborosConfig:
    """Build a tiny, CPU-fast :class:`OuroborosConfig` for integration tests.

    Returns an :class:`OuroborosConfig` shrunk so a full forward/generate runs
    quickly on CPU while preserving every architectural invariant:

    * ``dim=64``, ``n_heads=4`` (``head_dim=16``, even), ``n_kv_heads=2``
      (``n_heads % n_kv_heads == 0``).
    * Small MoE: ``n_experts=4``, ``n_shared_experts=1``,
      ``n_experts_per_tok=2``, ``expert_dim=32``.
    * Shallow stacks: ``prelude_layers=1``, ``coda_layers=1``,
      ``max_loop_iters=4``, ``max_seq_len=64``, ``vocab_size=256``.

    Args:
        **overrides: Field values that replace the tiny defaults (e.g.
            ``use_lti=False`` to build the naive-injection ablation arm).

    Returns:
        A tiny :class:`OuroborosConfig` suitable for fast integration tests.
    """
    tiny: dict[str, Any] = dict(
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=32,
        prelude_layers=1,
        coda_layers=1,
        max_loop_iters=4,
        max_seq_len=64,
        vocab_size=256,
    )
    tiny.update(overrides)
    return OuroborosConfig(**tiny)


# ===========================================================================
# Forward pass — Phase 5
# ===========================================================================


def test_forward_output_shape() -> None:
    """``forward`` maps ``input_ids (B, T)`` to logits ``(B, T, vocab_size)``.

    A forward pass on a random ``(B, T)`` batch of token ids must return logits
    of shape ``(B, T, vocab_size)``.
    """
    pytest.skip("stub — implement in Phase 5")


def test_forward_no_nan() -> None:
    """A forward pass produces finite logits with no NaN or Inf.

    Every logit must be finite (``torch.isfinite``) — a basic numerical-health
    check on the assembled prelude/recurrent/coda stack and the LTI recurrence.
    """
    pytest.skip("stub — implement in Phase 5")


def test_forward_single_token_T1() -> None:
    """A single-token forward (``T=1``) runs and returns ``(B, 1, vocab_size)``.

    With ``T=1`` the model builds no causal mask (``mask=None``); the forward
    must still run and return ``(B, 1, vocab_size)`` — the exact path used by
    every incremental decode step.
    """
    pytest.skip("stub — implement in Phase 5")


# ===========================================================================
# Generation — Phase 5
# ===========================================================================


def test_generate_output_shape() -> None:
    """``generate`` appends ``max_new_tokens`` to the prompt.

    ``generate(input_ids, max_new_tokens=k)`` on a prompt of shape ``(B, T)``
    must return token ids of shape ``(B, T + k)``.
    """
    pytest.skip("stub — implement in Phase 5")


def test_cached_decode_matches_full_context() -> None:
    """KV-cached single-token decode logits match a full-context forward.

    The correctness invariant for KV caching: feeding a prompt then decoding one
    token with a KV cache must yield (within tolerance) the same final-position
    logits as a single full-context forward over the concatenated sequence —
    validating ``start_pos`` RoPE slicing and the per-loop
    ``recurrent_loop_{t}`` cache keys.
    """
    pytest.skip("stub — implement in Phase 5")


# ===========================================================================
# Weight tying — Phase 5
# ===========================================================================


def test_weight_tying_head_shares_embedding() -> None:
    """The LM head weight is the SAME tensor as the embedding weight.

    ``model.head.weight`` must be the identical parameter object as
    ``model.embed.weight`` (``is`` identity, not just equal values), so the tied
    ``vocab_size × dim`` table is stored once and trained jointly.
    """
    pytest.skip("stub — implement in Phase 5")


# ===========================================================================
# LTI stability end-to-end — Phase 5
# ===========================================================================


def test_lti_spectral_radius_below_one_end_to_end() -> None:
    """The assembled model's recurrent ``ρ(A) < 1``.

    Reading ``model.recurrent.injection.get_A()`` from a fully-built model must
    give a finite ``(dim,)`` vector strictly in ``(0, 1)``, so the spectral
    radius ``ρ(A) = max(get_A()) < 1`` holds end to end — the stability guarantee
    behind resume bullet 2.
    """
    pytest.skip("stub — implement in Phase 5")


def test_use_lti_false_builds_and_runs() -> None:
    """The ``use_lti=False`` ablation arm assembles and runs end to end.

    A model built from ``tiny_config(use_lti=False)`` must have
    ``model.recurrent.injection is None``, run a forward pass to finite logits
    of shape ``(B, T, vocab_size)`` via the naive residual update
    ``h = transformer_out + e``, and expose no ``get_A()`` to log — the
    comparison configuration for the stability experiment.
    """
    pytest.skip("stub — implement in Phase 5")


# ===========================================================================
# Depth extrapolation — Phase 5
# ===========================================================================


def test_depth_extrapolation_changes_output() -> None:
    """Running more loops than trained changes the forward output.

    Calling ``forward`` with ``n_loops`` greater than ``cfg.max_loop_iters`` must
    run without error (the loop-index embedding is well-defined for any depth)
    and produce logits different from a shallower ``n_loops`` — confirming
    test-time depth extrapolation is wired through.
    """
    pytest.skip("stub — implement in Phase 5")


# ===========================================================================
# Continuous depth-wise batched generation — Phase 7
# ===========================================================================


def test_generate_depthwise_batched_output_shape() -> None:
    """``generate_depthwise_batched`` appends ``max_new_tokens`` to the prompt.

    Generating with continuous depth-wise batching on a ``(B, T)`` prompt must
    return token ids of shape ``(B, T + max_new_tokens)`` — matching the
    standard ``generate`` contract while sequences exit the loop at different
    convergence-driven depths within the batch.
    """
    pytest.skip("stub — implement in Phase 7")


def test_generate_depthwise_batched_matches_generate_at_tight_tol() -> None:
    """With a near-zero ``convergence_tol``, depth-wise batching reduces to naive.

    When the tolerance is tight enough that no sequence exits early, the
    depth-wise batched generator must produce the same tokens as ``generate``
    under identical sampling seeds — the correctness baseline before any
    early-exit speedup is claimed.
    """
    pytest.skip("stub — implement in Phase 7")
