"""Unit-test stubs for the individual Ouroboros components.

Each test names the property it verifies, documents the concrete assertion in
its docstring, and skips with ``pytest.skip("stub — implement in Phase N")`` so
the suite is collectable and GREEN (all skips, no failures or errors) before any
component is implemented. The phase numbers track the project roadmap:

* Phase 1 — :class:`~ouroboros.norm.RMSNorm`, RoPE
  (:func:`~ouroboros.rope.precompute_rope_freqs`,
  :func:`~ouroboros.rope.apply_rope`).
* Phase 2 — :class:`~ouroboros.attention.GQAttention`.
* Phase 3 — :class:`~ouroboros.moe.Expert`, :class:`~ouroboros.moe.MoEFFN`.
* Phase 4 — :func:`~ouroboros.recurrence.loop_index_embedding`,
  :class:`~ouroboros.recurrence.LTIInjection`,
  :class:`~ouroboros.block.TransformerBlock`,
  :class:`~ouroboros.recurrence.RecurrentBlock`.

Real targets are imported from the ``ouroboros`` package so that the imports
themselves are part of the contract: if a public name is renamed or removed,
collection fails loudly even while every body is a skip.
"""

from __future__ import annotations

from typing import Any

import pytest

# These imports ARE part of the contract: collection fails loudly if any public
# name is renamed or removed. They are referenced only inside skipped bodies for
# now, so F401 is suppressed while every test remains a stub.
from ouroboros import (  # noqa: F401
    Expert,
    GQAttention,
    LTIInjection,
    MoEFFN,
    OuroborosConfig,
    RecurrentBlock,
    RMSNorm,
    TransformerBlock,
    apply_rope,
    loop_index_embedding,
    precompute_rope_freqs,
)


def tiny_config(**overrides: Any) -> OuroborosConfig:
    """Build a tiny, CPU-fast :class:`OuroborosConfig` for unit tests.

    Returns an :class:`OuroborosConfig` whose defaults are shrunk to keep every
    test fast on CPU while preserving the architectural invariants the
    components rely on:

    * ``dim=64`` with ``n_heads=4`` (``head_dim=16``, even — RoPE-safe) and
      ``n_kv_heads=2`` (``n_heads % n_kv_heads == 0`` — GQA-safe).
    * A small MoE: ``n_experts=4``, ``n_shared_experts=1``,
      ``n_experts_per_tok=2``, ``expert_dim=32``.
    * Shallow recurrence: ``prelude_layers=1``, ``coda_layers=1``,
      ``max_loop_iters=4``, ``max_seq_len=64``.

    Args:
        **overrides: Field values that replace the tiny defaults (e.g.
            ``use_lti=False`` to exercise the naive-injection ablation arm).

    Returns:
        A tiny :class:`OuroborosConfig` suitable for fast unit tests.
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
    )
    tiny.update(overrides)
    return OuroborosConfig(**tiny)


# ===========================================================================
# (1) OuroborosConfig — validation (implemented; not a stub)
# ===========================================================================


def test_config_defaults_and_tiny_config_construct() -> None:
    """The default and tiny configs both pass ``__post_init__`` validation."""
    OuroborosConfig()
    cfg = tiny_config()
    assert cfg.dim == 64
    assert tiny_config(use_lti=False).use_lti is False


def test_config_rejects_invalid_combinations() -> None:
    """``__post_init__`` fails fast on each cross-field invariant violation."""
    with pytest.raises(ValueError, match="divisible by n_heads"):
        tiny_config(dim=65)
    with pytest.raises(ValueError, match="must be even"):
        tiny_config(dim=68)  # head_dim = 17, odd
    with pytest.raises(ValueError, match="n_kv_heads"):
        tiny_config(n_kv_heads=3)
    with pytest.raises(ValueError, match="n_experts_per_tok"):
        tiny_config(n_experts_per_tok=5)
    with pytest.raises(ValueError, match="loop_index_dim"):
        tiny_config(loop_index_dim=7)


# ===========================================================================
# (2) RMSNorm — Phase 1
# ===========================================================================


def test_rmsnorm_preserves_shape() -> None:
    """RMSNorm returns a tensor with the same shape as its input.

    For an input of shape ``(B, T, dim)`` the output must also be
    ``(B, T, dim)`` (RMSNorm normalizes over the last axis only).
    """
    pytest.skip("stub — implement in Phase 1")


def test_rmsnorm_unit_rms_with_unit_weight() -> None:
    """With ``weight == 1`` the output has unit root-mean-square per position.

    When the learned ``weight`` is left at its init of all ones, each normalized
    row should satisfy ``mean(out**2, dim=-1) ≈ 1`` (up to ``eps``), confirming
    the RMS reduction and rescale are correct.
    """
    pytest.skip("stub — implement in Phase 1")


def test_rmsnorm_weight_is_learnable_parameter() -> None:
    """The per-channel ``weight`` is a learnable parameter of shape ``(dim,)``.

    ``RMSNorm(dim).weight`` must be an ``nn.Parameter`` with ``requires_grad``,
    shape ``(dim,)``, initialized to all ones, and no bias term should exist.
    """
    pytest.skip("stub — implement in Phase 1")


# ===========================================================================
# (3) RoPE — Phase 1
# ===========================================================================


def test_precompute_rope_freqs_shape_and_complex_dtype() -> None:
    """``precompute_rope_freqs`` returns a complex tensor of shape ``(max_len, dim//2)``.

    For ``precompute_rope_freqs(dim, max_len, theta)`` the result must be a
    complex tensor (``torch.complex64``) of shape ``(max_len, dim // 2)`` holding
    unit-modulus phasors ``e^{i·m·θ_k}``.
    """
    pytest.skip("stub — implement in Phase 1")


def test_apply_rope_preserves_shape() -> None:
    """``apply_rope`` returns a tensor of the same shape and dtype as its input.

    Given ``x`` of shape ``(B, T, H, head_dim)`` and ``freqs_cis`` sliced to
    ``(T, head_dim//2)``, the rotated output must keep shape ``(B, T, H,
    head_dim)`` and ``x``'s dtype.
    """
    pytest.skip("stub — implement in Phase 1")


def test_apply_rope_preserves_norm() -> None:
    """RoPE is an isometry: it preserves the per-vector L2 norm.

    The norm of each rotated ``head_dim`` vector must equal the norm of the
    corresponding input vector (rotation in the complex plane changes direction,
    not magnitude), up to floating-point tolerance.
    """
    pytest.skip("stub — implement in Phase 1")


def test_apply_rope_position_zero_is_identity() -> None:
    """Position 0 uses the phasor ``1+0j`` and leaves the input unchanged.

    Applying ``apply_rope`` with ``freqs_cis`` sliced to position 0 only (a
    ``T=1`` sequence at ``start_pos=0``) must return the input unchanged within
    floating-point tolerance.
    """
    pytest.skip("stub — implement in Phase 1")


def test_apply_rope_raises_when_position_exceeds_max_len() -> None:
    """Indexing RoPE frequencies past ``max_len`` raises rather than silently wrapping.

    Slicing the precomputed ``freqs_cis`` (length ``max_len``) at positions
    ``>= max_len`` must raise (e.g. ``IndexError``) so an out-of-range
    ``start_pos`` during decode fails fast instead of corrupting positions.
    """
    pytest.skip("stub — implement in Phase 1")


# ===========================================================================
# (4) GQAttention — Phase 2
# ===========================================================================


def test_gqattention_output_shape() -> None:
    """GQAttention maps ``(B, T, dim)`` to ``(B, T, dim)``.

    A forward pass with a causal mask and no cache must return a tensor of the
    same ``(B, T, dim)`` shape as the input.
    """
    pytest.skip("stub — implement in Phase 2")


def test_gqattention_kv_cache_accumulates() -> None:
    """The KV cache grows along the sequence axis across decode steps.

    After a prefill of length ``T`` followed by a single-token step, the cached
    ``k``/``v`` under the given ``cache_key`` must have sequence length ``T + 1``
    (keys/values are concatenated along the seq dim and detached).
    """
    pytest.skip("stub — implement in Phase 2")


def test_gqattention_causal_mask_blocks_future() -> None:
    """The causal mask prevents a position from attending to later positions.

    With an additive causal mask, the output at position ``i`` must be invariant
    to perturbations of input positions ``j > i`` (future tokens cannot leak
    into earlier outputs).
    """
    pytest.skip("stub — implement in Phase 2")


# ===========================================================================
# (5) Expert — Phase 3
# ===========================================================================


def test_expert_output_shape() -> None:
    """A SwiGLU ``Expert`` maps ``(..., dim)`` to ``(..., dim)``.

    ``Expert(dim, expert_dim)(x)`` for ``x`` of shape ``(B, T, dim)`` must return
    a tensor of shape ``(B, T, dim)`` (the inner ``expert_dim`` is hidden).
    """
    pytest.skip("stub — implement in Phase 3")


# ===========================================================================
# (6) MoEFFN — Phase 3
# ===========================================================================


def test_moeffn_output_shape() -> None:
    """``MoEFFN`` maps ``(B, T, dim)`` to ``(B, T, dim)``.

    The routed-plus-shared expert combination must preserve the residual-stream
    shape ``(B, T, dim)``.
    """
    pytest.skip("stub — implement in Phase 3")


def test_moeffn_router_bias_is_buffer_not_parameter() -> None:
    """``router_bias`` is a non-gradient buffer, never a learnable parameter.

    ``router_bias`` must appear in ``dict(MoEFFN(cfg).named_buffers())`` (with
    shape ``(n_experts,)``) and must NOT appear among ``named_parameters()`` —
    the aux-loss-free bias is updated manually, never by autograd.
    """
    pytest.skip("stub — implement in Phase 3")


def test_moeffn_shared_experts_always_fire() -> None:
    """Shared experts contribute to every token regardless of routing.

    Zeroing the routed-expert contribution (or comparing against a router that
    selects no routed expert) must still leave a non-zero output coming from the
    always-active shared experts, confirming they fire for every token.
    """
    pytest.skip("stub — implement in Phase 3")


def test_moeffn_update_router_bias_moves_load_toward_balance() -> None:
    """``update_router_bias`` nudges the bias to rebalance expert load.

    After deliberately skewing the ``expert_load`` buffer (one expert
    overloaded, another underloaded) and calling ``update_router_bias()``, the
    overloaded expert's ``router_bias`` entry must DECREASE and the underloaded
    expert's must INCREASE (step magnitude ``router_bias_update_rate``), driving
    routing toward uniform. The update runs under ``no_grad`` and does not touch
    the gating gradient.
    """
    pytest.skip("stub — implement in Phase 3")


# ===========================================================================
# (8) loop_index_embedding — Phase 4
# ===========================================================================


def test_loop_index_embedding_preserves_shape() -> None:
    """``loop_index_embedding`` returns a tensor of the same ``(B, T, dim)`` shape.

    Adding the sinusoidal depth signal must not change the shape of ``h``.
    """
    pytest.skip("stub — implement in Phase 4")


def test_loop_index_embedding_differs_per_iteration() -> None:
    """Different ``loop_t`` values inject different signals.

    For the same ``h``, ``loop_index_embedding(h, t=0, ...)`` and
    ``loop_index_embedding(h, t=1, ...)`` must differ — the per-loop sinusoid is
    what lets the shared recurrent weights behave differently at each depth.
    """
    pytest.skip("stub — implement in Phase 4")


def test_loop_index_embedding_only_modifies_first_loop_dim_channels() -> None:
    """Only the first ``loop_dim`` channels change; the rest pass through.

    The trailing ``dim - loop_dim`` channels of the output must equal the
    corresponding input channels exactly, while the first ``loop_dim`` channels
    are shifted by the sinusoidal bias.
    """
    pytest.skip("stub — implement in Phase 4")


# ===========================================================================
# (9) LTIInjection — Phase 4
# ===========================================================================


def test_lti_injection_output_shape() -> None:
    """``LTIInjection.forward`` returns ``h_{t+1}`` of shape ``(B, T, dim)``.

    Given ``h``, ``e``, and ``transformer_out`` each of shape ``(B, T, dim)``,
    the update ``A·h + B·e + transformer_out`` must return ``(B, T, dim)``.
    """
    pytest.skip("stub — implement in Phase 4")


def test_lti_injection_get_A_strictly_between_zero_and_one() -> None:
    """``get_A()`` returns a ``(dim,)`` vector with every value in ``(0, 1)``.

    The ZOH discretization guarantees ``0 < A < 1`` elementwise at init,
    therefore ``ρ(A) = max(get_A()) < 1`` (A is diagonal). Values must be
    strictly positive and strictly below 1.
    """
    pytest.skip("stub — implement in Phase 4")


def test_lti_injection_spectral_radius_stays_below_one_after_huge_step() -> None:
    """``ρ(A) < 1`` holds even after an enormous gradient step on ``log_A``/``log_dt``.

    After applying a large SGD-style update to ``log_A`` and ``log_dt`` (e.g.
    adding ``+1e4``), ``get_A()`` must still be finite, NaN-free, and strictly in
    ``(0, 1)`` — the ``clamp(-20, 20)`` in log space guarantees stability by
    construction, the headline property for resume bullet 2.
    """
    pytest.skip("stub — implement in Phase 4")


# ===========================================================================
# (7) TransformerBlock — Phase 4
# ===========================================================================


def test_transformer_block_dense_shape() -> None:
    """A dense ``TransformerBlock`` preserves the ``(B, T, dim)`` shape.

    Built with ``use_moe=False`` (the Prelude/Coda configuration), the pre-norm
    block must return ``(B, T, dim)``.
    """
    pytest.skip("stub — implement in Phase 4")


def test_transformer_block_moe_shape() -> None:
    """A ``use_moe=True`` ``TransformerBlock`` preserves the ``(B, T, dim)`` shape.

    With the MoE FFN swapped in (the recurrent-block configuration), the block
    must still return ``(B, T, dim)``.
    """
    pytest.skip("stub — implement in Phase 4")


def test_transformer_block_holds_gqa_attention() -> None:
    """Every ``TransformerBlock`` holds a ``GQAttention`` sublayer.

    The block's ``attn`` submodule must be a ``GQAttention`` instance — the
    single attention mechanism in the architecture.
    """
    pytest.skip("stub — implement in Phase 4")


# ===========================================================================
# (10) RecurrentBlock — Phase 4
# ===========================================================================


def test_recurrent_block_output_shape() -> None:
    """``RecurrentBlock`` returns the final hidden state of shape ``(B, T, dim)``.

    Given ``h`` and ``e`` of shape ``(B, T, dim)``, the fixed-depth looped block
    must return ``(B, T, dim)`` (the hidden state after ``n_loops`` iterations).
    """
    pytest.skip("stub — implement in Phase 4")


def test_recurrent_block_more_loops_changes_output() -> None:
    """Running more loop iterations changes the output.

    Calling the block with ``n_loops=2`` versus ``n_loops=4`` on identical
    ``h``/``e`` must produce different outputs — additional recurrence depth
    performs additional computation (the basis of depth extrapolation).
    """
    pytest.skip("stub — implement in Phase 4")


def test_recurrent_block_single_loop_runs() -> None:
    """A single-loop pass (``n_loops=1``) runs and returns ``(B, T, dim)``.

    The loop body must be well-defined for the minimal depth of one iteration
    (no off-by-one in the loop or cache-key construction).
    """
    pytest.skip("stub — implement in Phase 4")


def test_recurrent_block_use_lti_false_builds_naive_injection() -> None:
    """``use_lti=False`` builds no ``LTIInjection`` and still runs the loop.

    A block built from ``tiny_config(use_lti=False)`` must have
    ``injection is None``, run the naive residual update
    ``h = transformer_out + e`` without error, and return ``(B, T, dim)`` —
    the ablation arm of the stability experiment.
    """
    pytest.skip("stub — implement in Phase 4")
