"""Unit-test stubs for the individual Ouroboros components.

Each test names the property it checks, states the assertion in its docstring,
and skips with ``pytest.skip("stub: section N")`` so the suite stays green before
the component exists. Section numbers follow ``ROADMAP.md``:

* Section 2: ``RMSNorm``, ``Expert`` (SwiGLU).
* Section 3: RoPE.
* Section 4: ``GQAttention`` (no cache). Section 5: its KV cache.
* Section 6: ``tiled_attention``.
* Section 7: ``MoEFFN``, ``TransformerBlock``.
* Section 8: ``loop_index_embedding``, ``sample_n_loops``, ``LTIInjection``,
  ``RecurrentBlock``.

The imports are part of the contract: renaming or removing a public name fails
collection even while every body is a skip.
"""

from __future__ import annotations

from typing import Any

import pytest

# Referenced only inside skipped bodies for now, so F401 is suppressed.
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
    sample_n_loops,
    tiled_attention,
)


def tiny_config(**overrides: Any) -> OuroborosConfig:
    """Build a tiny, CPU-fast :class:`OuroborosConfig` for unit tests.

    ``dim=64``, ``n_heads=4`` (``head_dim=16``), ``n_kv_heads=2``, a 4-expert
    MoE, one Prelude and one Coda layer, ``mean_loops=4``, ``backprop_loops=2``.

    Args:
        **overrides: Field values that replace the tiny defaults.

    Returns:
        A tiny :class:`OuroborosConfig`.
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
        mean_loops=4,
        backprop_loops=2,
        max_seq_len=64,
    )
    tiny.update(overrides)
    return OuroborosConfig(**tiny)


# ===========================================================================
# OuroborosConfig: validation (implemented)
# ===========================================================================


def test_config_defaults_and_tiny_config_construct() -> None:
    """The default and tiny configs both pass ``__post_init__`` validation."""
    OuroborosConfig()
    cfg = tiny_config()
    assert cfg.dim == 64
    assert cfg.mean_loops == 4
    assert cfg.use_lti is True
    assert tiny_config(use_lti=False).use_lti is False
    assert cfg.loop_dim == 8
    assert tiny_config(dim=72, n_heads=4).loop_dim == 8  # 72 // 8 = 9 -> 8


def test_config_rejects_invalid_combinations() -> None:
    """``__post_init__`` fails fast on each cross-field invariant violation."""
    with pytest.raises(ValueError, match="must be >= 1"):
        tiny_config(n_kv_heads=0)
    with pytest.raises(ValueError, match="divisible by n_heads"):
        tiny_config(dim=65)
    with pytest.raises(ValueError, match="must be even"):
        tiny_config(dim=68)  # head_dim = 17, odd
    with pytest.raises(ValueError, match="n_kv_heads"):
        tiny_config(n_kv_heads=3)
    with pytest.raises(ValueError, match="n_experts_per_tok"):
        tiny_config(n_experts_per_tok=5)
    with pytest.raises(ValueError, match="mean_loops"):
        tiny_config(mean_loops=0)
    with pytest.raises(ValueError, match="mean_loops"):
        tiny_config(mean_loops=4.5)
    with pytest.raises(ValueError, match="backprop_loops"):
        tiny_config(backprop_loops=0)
    with pytest.raises(ValueError, match="loop_index_dim"):
        tiny_config(loop_index_dim=7)


# ===========================================================================
# RMSNorm: section 2
# ===========================================================================


def test_rmsnorm_preserves_shape() -> None:
    """RMSNorm maps ``(B, T, dim)`` to ``(B, T, dim)``."""
    pytest.skip("stub: section 2")


def test_rmsnorm_unit_rms_with_unit_weight() -> None:
    """With ``weight == 1``, ``mean(out**2, dim=-1)`` is about 1 per row."""
    pytest.skip("stub: section 2")


def test_rmsnorm_weight_is_learnable_parameter() -> None:
    """``weight`` is an ``nn.Parameter`` of shape ``(dim,)``, init to ones, no bias."""
    pytest.skip("stub: section 2")


# ===========================================================================
# RoPE: section 3
# ===========================================================================


def test_precompute_rope_freqs_shape_and_complex_dtype() -> None:
    """``precompute_rope_freqs`` returns ``complex64`` of shape ``(max_len, dim // 2)``."""
    pytest.skip("stub: section 3")


def test_apply_rope_preserves_shape() -> None:
    """``apply_rope`` keeps the ``(B, T, H, head_dim)`` shape and dtype."""
    pytest.skip("stub: section 3")


def test_apply_rope_preserves_norm() -> None:
    """RoPE is a rotation: each head vector keeps its L2 norm."""
    pytest.skip("stub: section 3")


def test_apply_rope_position_zero_is_identity() -> None:
    """With ``freqs_cis`` sliced to position 0 only, the input comes back unchanged."""
    pytest.skip("stub: section 3")


# ===========================================================================
# Expert (SwiGLU): section 2
# ===========================================================================


def test_expert_output_shape() -> None:
    """``Expert(dim, expert_dim)`` maps ``(B, T, dim)`` to ``(B, T, dim)``."""
    pytest.skip("stub: section 2")


# ===========================================================================
# GQAttention (4), KV cache (5), tiled_attention (6)
# ===========================================================================


def test_gqattention_output_shape() -> None:
    """GQAttention maps ``(B, T, dim)`` to ``(B, T, dim)``."""
    pytest.skip("stub: section 4")


def test_gqattention_sdpa_matches_manual() -> None:
    """``use_sdpa=True`` and ``use_sdpa=False`` give the same output (atol ~1e-5, fp32)."""
    pytest.skip("stub: section 4")


def test_gqattention_causal_blocks_future() -> None:
    """With ``causal=True``, output at position ``i`` ignores inputs at ``j > i``."""
    pytest.skip("stub: section 4")


def test_gqattention_kv_cache_accumulates() -> None:
    """After a length-``T`` prefill and one decode step, cached k/v have length ``T + 1``."""
    pytest.skip("stub: section 5")


def test_gqattention_cached_decode_matches_one_shot() -> None:
    """Token-by-token decoding through the cache matches one full causal pass.

    Feed tokens one at a time (``causal=False``, correct RoPE slice each step)
    and compare the last position against a single ``causal=True`` pass over
    the whole sequence.
    """
    pytest.skip("stub: section 5")


def test_tiled_attention_matches_sdpa() -> None:
    """``tiled_attention`` matches ``F.scaled_dot_product_attention``.

    Check both ``causal=True`` and ``causal=False``, with a sequence length that
    is not a multiple of ``block_size`` (to catch edge blocks).
    """
    pytest.skip("stub: section 6")


# ===========================================================================
# MoEFFN and TransformerBlock: section 7
# ===========================================================================


def test_moeffn_output_shape() -> None:
    """``MoEFFN`` maps ``(B, T, dim)`` to ``(B, T, dim)``."""
    pytest.skip("stub: section 7")


def test_moeffn_router_bias_is_buffer_not_parameter() -> None:
    """``router_bias`` is in ``named_buffers()`` with shape ``(n_experts,)``, not in ``named_parameters()``."""
    pytest.skip("stub: section 7")


def test_moeffn_shared_experts_always_fire() -> None:
    """With the routed contribution zeroed, the shared experts still give a non-zero output."""
    pytest.skip("stub: section 2")


def test_moeffn_update_router_bias_moves_load_toward_balance() -> None:
    """``update_router_bias`` lowers the bias of an overloaded expert and raises an underloaded one.

    Skew ``expert_load`` by hand, call the update, and check each changed by
    ``router_bias_update_rate`` in the right direction, and that
    ``expert_load`` is reset to zero.
    """
    pytest.skip("stub: section 7")


def test_moeffn_eval_does_not_accumulate_load() -> None:
    """In ``eval()`` mode a forward pass leaves ``expert_load`` unchanged."""
    pytest.skip("stub: section 7")


def test_transformer_block_dense_and_moe_shapes() -> None:
    """Both ``use_moe=False`` and ``use_moe=True`` blocks map ``(B, T, dim)`` to ``(B, T, dim)``."""
    pytest.skip("stub: section 7")


# ===========================================================================
# Recurrence: section 8
# ===========================================================================


def test_loop_index_embedding_differs_per_iteration() -> None:
    """``loop_t=0`` and ``loop_t=1`` give different outputs for the same ``h``."""
    pytest.skip("stub: section 8")


def test_loop_index_embedding_only_modifies_first_loop_dim_channels() -> None:
    """Channels ``loop_dim:`` equal the input exactly; shape is unchanged."""
    pytest.skip("stub: section 8")


def test_sample_n_loops_range_and_mean() -> None:
    """``sample_n_loops`` returns ints ``>= 1`` with an empirical mean near ``mean_loops``.

    Draw a few thousand samples with a seeded generator for ``mean_loops`` of 4
    and 8; each mean should be within about 5% of ``mean_loops`` and the max
    well above it (heavy tail). ``mean_loops=1`` always returns 1.
    """
    pytest.skip("stub: section 8")


def test_lti_injection_get_A_strictly_between_zero_and_one() -> None:
    """``get_A()`` is ``(dim,)`` with every entry in ``(0, 1)``."""
    pytest.skip("stub: section 8")


def test_lti_injection_spectral_radius_stays_below_one_after_huge_step() -> None:
    """``rho(A) < 1`` survives a huge parameter step.

    After adding ``+1e4`` (and separately ``-1e4``) to ``log_A`` and ``log_dt``,
    ``get_A()`` in fp32 is finite, NaN-free, and strictly inside ``(0, 1)``
    (``max < 1`` and ``min > 0``); the ``clamp(-12, 4)`` is what guarantees it.
    """
    pytest.skip("stub: section 8")


def test_recurrent_block_output_shape_and_single_loop() -> None:
    """``RecurrentBlock`` returns ``(B, T, dim)`` for ``n_loops=1`` and ``n_loops=4``."""
    pytest.skip("stub: section 8")


def test_recurrent_block_more_loops_changes_output() -> None:
    """``n_loops=2`` and ``n_loops=4`` give different outputs on the same input."""
    pytest.skip("stub: section 8")


def test_recurrent_block_naive_arm() -> None:
    """``use_lti=False`` builds no ``LTIInjection`` and still runs the loop.

    ``block.injection is None``, the update is ``h = trans + e``, and the output
    is ``(B, T, dim)``.
    """
    pytest.skip("stub: section 8")


def test_recurrent_block_truncated_backprop() -> None:
    """Truncated backprop changes gradients, not values.

    In train mode with ``n_loops=6`` and ``backprop_loops=2``: the output equals
    the output computed under ``torch.no_grad()`` (same values), and
    ``backward()`` still produces gradients for the block's parameters and for
    ``e``.
    """
    pytest.skip("stub: section 8")
