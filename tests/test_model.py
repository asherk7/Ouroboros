"""Integration-test stubs for the full :class:`~ouroboros.model.Ouroboros` model.

These exercise the assembled model end to end — embedding through prelude,
recurrent block, coda, and the tied LM head — under BOTH attention backends
(``attn_type="gqa"`` and ``attn_type="mla"``). Each test documents its assertion
and skips with ``pytest.skip("stub — implement in Phase N")`` so the suite is
collectable and GREEN (all skips) before the model is implemented.

Phase mapping (project roadmap):

* Phase 5 — full model and KV-cached generation: forward shape, forward no-NaN,
  ``generate`` shape, weight tying, end-to-end ``ρ(A) < 1``, depth extrapolation,
  cached-decode ≈ full-context, single-token (``T=1``) forward, and MLA-vs-GQA
  cache size.
* Phase 7 — inference optimization: INT8 quantization
  (:func:`~ouroboros.quantize.quantize_int8` /
  :func:`~ouroboros.quantize.quantization_error`) and continuous depth-wise
  batched generation
  (:meth:`~ouroboros.model.Ouroboros.generate_depthwise_batched`).

The GQA/MLA matrix is expressed with ``@pytest.mark.parametrize`` so each shape
or correctness property is asserted independently for both attention types.
``quantize_int8`` / ``quantization_error`` are imported from ``ouroboros.quantize``
directly because the Phase-7 quantization utilities are not re-exported from the
top-level package namespace.
"""

from __future__ import annotations

import pytest

# These imports ARE part of the contract: collection fails loudly if any public
# name is renamed or removed. ``Ouroboros``, ``quantize_int8``, and
# ``quantization_error`` are referenced only inside skipped bodies for now, so
# F401 is suppressed while every test remains a stub.
from ouroboros import Ouroboros, OuroborosConfig  # noqa: F401
from ouroboros.quantize import (  # noqa: F401
    quantization_error,
    quantize_int8,
)


def tiny_config(attn_type: str = "gqa", **overrides: object) -> OuroborosConfig:
    """Build a tiny, CPU-fast :class:`OuroborosConfig` for integration tests.

    Returns an :class:`OuroborosConfig` shrunk so a full forward/generate runs
    quickly on CPU while preserving every architectural invariant:

    * ``dim=64``, ``n_heads=4`` (``head_dim=16``, even), ``n_kv_heads=2``
      (``n_heads % n_kv_heads == 0``).
    * MLA dims kept self-consistent: ``qk_rope_head_dim=16`` (even),
      ``qk_nope_head_dim=16``, ``v_head_dim=16``, ``kv_lora_rank=32``,
      ``q_lora_rank=32``.
    * Small MoE: ``n_experts=4``, ``n_shared_experts=1``,
      ``n_experts_per_tok=2``, ``expert_dim=32``.
    * Shallow stacks: ``prelude_layers=1``, ``coda_layers=1``,
      ``max_loop_iters=4``, ``max_seq_len=64``, ``vocab_size=256``.

    Args:
        attn_type: Attention backend to configure (``"gqa"`` or ``"mla"``).
        **overrides: Field values that replace the tiny defaults.

    Returns:
        A tiny :class:`OuroborosConfig` with the requested ``attn_type``.
    """
    pytest.skip("stub — implement in Phase 5")


# ===========================================================================
# Forward pass — Phase 5
# ===========================================================================


@pytest.mark.parametrize("attn_type", ["gqa", "mla"])
def test_forward_output_shape(attn_type: str) -> None:
    """``forward`` maps ``input_ids (B, T)`` to logits ``(B, T, vocab_size)``.

    For both GQA and MLA, a forward pass on a random ``(B, T)`` batch of token
    ids must return logits of shape ``(B, T, vocab_size)``.
    """
    pytest.skip("stub — implement in Phase 5")


@pytest.mark.parametrize("attn_type", ["gqa", "mla"])
def test_forward_no_nan(attn_type: str) -> None:
    """A forward pass produces finite logits with no NaN or Inf.

    For both attention backends, every logit must be finite (``torch.isfinite``)
    — a basic numerical-health check on the assembled prelude/recurrent/coda
    stack and the LTI recurrence.
    """
    pytest.skip("stub — implement in Phase 5")


@pytest.mark.parametrize("attn_type", ["gqa", "mla"])
def test_forward_single_token_T1(attn_type: str) -> None:
    """A single-token forward (``T=1``) runs and returns ``(B, 1, vocab_size)``.

    With ``T=1`` the model builds no causal mask (``mask=None``); the forward
    must still run for both backends and return ``(B, 1, vocab_size)`` — the
    exact path used by every incremental decode step.
    """
    pytest.skip("stub — implement in Phase 5")


# ===========================================================================
# Generation — Phase 5
# ===========================================================================


@pytest.mark.parametrize("attn_type", ["gqa", "mla"])
def test_generate_output_shape(attn_type: str) -> None:
    """``generate`` appends ``max_new_tokens`` to the prompt.

    For both backends, ``generate(input_ids, max_new_tokens=k)`` on a prompt of
    shape ``(B, T)`` must return token ids of shape ``(B, T + k)``.
    """
    pytest.skip("stub — implement in Phase 5")


@pytest.mark.parametrize("attn_type", ["gqa", "mla"])
def test_cached_decode_matches_full_context(attn_type: str) -> None:
    """KV-cached single-token decode logits match a full-context forward.

    The correctness invariant for KV caching: feeding a prompt then decoding one
    token with a KV cache must yield (within tolerance) the same final-position
    logits as a single full-context forward over the concatenated sequence. This
    must hold for both GQA and MLA, validating ``start_pos`` RoPE slicing and the
    per-loop ``recurrent_loop_{t}`` cache keys.
    """
    pytest.skip("stub — implement in Phase 5")


# ===========================================================================
# Weight tying — Phase 5
# ===========================================================================


@pytest.mark.parametrize("attn_type", ["gqa", "mla"])
def test_weight_tying_head_shares_embedding(attn_type: str) -> None:
    """The LM head weight is the SAME tensor as the embedding weight.

    ``model.head.weight`` must be the identical parameter object as
    ``model.embed.weight`` (``is`` identity, not just equal values), so the tied
    ``vocab_size × dim`` table is stored once and trained jointly.
    """
    pytest.skip("stub — implement in Phase 5")


# ===========================================================================
# LTI stability end-to-end — Phase 5
# ===========================================================================


@pytest.mark.parametrize("attn_type", ["gqa", "mla"])
def test_lti_spectral_radius_below_one_end_to_end(attn_type: str) -> None:
    """The assembled model's recurrent ``ρ(A) < 1``.

    Reading ``model.recurrent.injection.get_A()`` from a fully-built model must
    give a finite ``(dim,)`` vector strictly in ``(0, 1)``, so the spectral
    radius ``ρ(A) = max(get_A()) < 1`` holds end to end — the stability guarantee
    behind resume bullet 2. Asserted for both backends.
    """
    pytest.skip("stub — implement in Phase 5")


# ===========================================================================
# Depth extrapolation — Phase 5
# ===========================================================================


@pytest.mark.parametrize("attn_type", ["gqa", "mla"])
def test_depth_extrapolation_changes_output(attn_type: str) -> None:
    """Running more loops than trained changes the forward output.

    Calling ``forward`` with ``n_loops`` greater than ``cfg.max_loop_iters`` must
    run without error (the loop-index embedding is well-defined for any depth) and
    produce logits different from a shallower ``n_loops`` — confirming test-time
    depth extrapolation is wired through for both backends.
    """
    pytest.skip("stub — implement in Phase 5")


# ===========================================================================
# MLA vs GQA cache size — Phase 5
# ===========================================================================


def test_mla_cache_smaller_than_gqa_cache() -> None:
    """The MLA KV cache uses fewer bytes than the GQA KV cache.

    Running one forward with a KV cache under each attention type (matched
    config otherwise) and summing the cached tensors' bytes, the MLA cache
    (compressed ``c_kv`` of width ``kv_lora_rank`` plus ``k_rope``) must total
    strictly fewer bytes than the GQA cache (full ``k``/``v`` of width
    ``n_kv_heads × head_dim``).
    """
    pytest.skip("stub — implement in Phase 5")


# ===========================================================================
# INT8 quantization — Phase 7
# ===========================================================================


@pytest.mark.parametrize("attn_type", ["gqa", "mla"])
def test_quantize_int8_returns_model(attn_type: str) -> None:
    """``quantize_int8`` returns a usable :class:`Ouroboros` model.

    The returned object must be an :class:`Ouroboros` instance that still runs a
    forward pass to ``(B, T, vocab_size)`` logits, with the large attention/FFN
    linears replaced by INT8 equivalents (norms, router, and the LM head left in
    higher precision). Asserted for both backends.
    """
    pytest.skip("stub — implement in Phase 7")


def test_quantize_int8_perplexity_delta_is_measurable() -> None:
    """``quantization_error`` reports a finite, measurable perplexity delta.

    Comparing the FP model against its INT8 counterpart via
    ``quantization_error(fp_model, int8_model, eval_loader)`` must return a dict
    whose ``ppl_delta`` (and ``ppl_fp`` / ``ppl_int8``) are finite floats — the
    INT8 accuracy cost is quantified, not assumed (the headline Phase-7 signal).
    """
    pytest.skip("stub — implement in Phase 7")


# ===========================================================================
# Continuous depth-wise batched generation — Phase 7
# ===========================================================================


@pytest.mark.parametrize("attn_type", ["gqa", "mla"])
def test_generate_depthwise_batched_output_shape(attn_type: str) -> None:
    """``generate_depthwise_batched`` appends ``max_new_tokens`` to the prompt.

    For both backends, generating with continuous depth-wise batching on a
    ``(B, T)`` prompt must return token ids of shape ``(B, T + max_new_tokens)``
    — matching the standard ``generate`` contract while sequences exit the loop
    at different convergence-driven depths within the batch.
    """
    pytest.skip("stub — implement in Phase 7")
