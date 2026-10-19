"""Unit tests for the individual Ouroboros components.

Each test names the property it checks and states the assertion in its
docstring. Sections 2 to 7 have real bodies and fail until the component is
implemented; later sections still skip with ``pytest.skip("stub: section N")``. Section numbers follow ``ROADMAP.md``:

* Section 2: ``RMSNorm``, ``FeedForward`` (SwiGLU).
* Section 3: RoPE.
* Section 4: ``GQAttention`` (no cache). Section 5: its KV cache.
* Section 6: nothing (theory only; the model uses PyTorch SDPA).
* Section 7: ``MoEFFN``, ``TransformerBlock``.
* Section 8: ``loop_index_embedding``, ``sample_n_loops``, ``LTIInjection``,
  ``RecurrentBlock``.

The imports are part of the contract: renaming or removing a public name fails
collection even while every body is a skip.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Section 8 names are referenced only inside skipped bodies, so F401 is suppressed.
from ouroboros import (  # noqa: F401
    FeedForward,
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


def tiny_freqs(cfg: OuroborosConfig) -> torch.Tensor:
    """RoPE phasors for every position of ``cfg``, sized to the per-head width."""
    return precompute_rope_freqs(
        cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta
    )


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
    """RMSNorm maps ``(B, T, dim)`` to ``(B, T, dim)``, keeping the dtype."""
    norm = RMSNorm(64)
    x = torch.randn(2, 8, 64)
    out = norm(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype

    # Low precision in, low precision out (the fp32 reduction is internal only).
    norm_bf16 = RMSNorm(64).to(torch.bfloat16)
    assert norm_bf16(x.to(torch.bfloat16)).dtype == torch.bfloat16


def test_rmsnorm_unit_rms_with_unit_weight() -> None:
    """With ``weight == 1``, ``mean(out**2, dim=-1)`` is about 1 per row."""
    torch.manual_seed(0)
    norm = RMSNorm(64)
    # Scale the input so the test fails if the norm does nothing.
    x = 5.0 * torch.randn(2, 8, 64) + 3.0
    ms = norm(x).pow(2).mean(dim=-1)
    torch.testing.assert_close(ms, torch.ones_like(ms), atol=1e-4, rtol=0)


def test_rmsnorm_matches_formula() -> None:
    """Output equals ``x / sqrt(mean(x**2) + eps) * weight`` for a random ``weight``.

    Catches a LayerNorm (mean subtraction) in disguise, or a weight that is
    never applied, neither of which the unit-RMS test can see.
    """
    torch.manual_seed(0)
    norm = RMSNorm(64, eps=1e-6)
    weight = norm.get_parameter("weight")
    with torch.no_grad():
        weight.copy_(torch.randn(64))
    x = 5.0 * torch.randn(2, 8, 64) + 3.0
    expected = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * weight
    torch.testing.assert_close(norm(x), expected)


def test_rmsnorm_weight_is_learnable_parameter() -> None:
    """``weight`` is an ``nn.Parameter`` of shape ``(dim,)``, init to ones, no bias."""
    norm = RMSNorm(64)
    assert isinstance(norm.weight, nn.Parameter)
    assert norm.weight.shape == (64,)
    assert torch.equal(norm.weight, torch.ones(64))
    assert [name for name, _ in norm.named_parameters()] == ["weight"]


# ===========================================================================
# RoPE: section 3
# ===========================================================================


def test_precompute_rope_freqs_shape_and_complex_dtype() -> None:
    """``precompute_rope_freqs`` returns ``complex64`` of shape ``(max_len, dim // 2)``."""
    freqs = precompute_rope_freqs(16, 64)
    assert freqs.shape == (64, 8)
    assert freqs.dtype == torch.complex64
    # Every phasor is a pure rotation (magnitude 1).
    torch.testing.assert_close(freqs.abs(), torch.ones(64, 8))
    # Entry [m, k] rotates by angle m * theta**(-2k / dim).
    m, k = 5, 3
    angle = m * 10000.0 ** (-2 * k / 16)
    torch.testing.assert_close(
        freqs[m, k], torch.tensor(complex(math.cos(angle), math.sin(angle)))
    )


def test_apply_rope_rotates_adjacent_pairs() -> None:
    """Features ``(2k, 2k + 1)`` form one pair, rotated by angle ``m * theta_k``.

    With ``x = (1, 0, 0, ...)`` at position ``m``, frequency 0 (``theta_0 = 1``)
    turns the first pair into ``(cos m, sin m)`` and leaves the rest at zero.
    A "rotate half" implementation (pairing ``k`` with ``k + head_dim // 2``)
    fails this.
    """
    freqs = precompute_rope_freqs(16, 64)
    m = 3
    x = torch.zeros(1, 1, 1, 16)
    x[..., 0] = 1.0
    expected = torch.zeros(1, 1, 1, 16)
    expected[..., 0], expected[..., 1] = math.cos(m), math.sin(m)
    torch.testing.assert_close(apply_rope(x, freqs[m : m + 1]), expected)


def test_apply_rope_preserves_shape() -> None:
    """``apply_rope`` keeps the ``(B, T, H, head_dim)`` shape and dtype."""
    freqs = precompute_rope_freqs(16, 64)
    x = torch.randn(2, 8, 4, 16)
    out = apply_rope(x, freqs[:8])
    assert out.shape == x.shape
    assert out.dtype == x.dtype

    # Low precision input comes back in low precision (fp32 rotation is internal).
    assert apply_rope(x.to(torch.bfloat16), freqs[:8]).dtype == torch.bfloat16


def test_apply_rope_preserves_norm() -> None:
    """RoPE is a rotation: each head vector keeps its L2 norm."""
    torch.manual_seed(0)
    freqs = precompute_rope_freqs(16, 64)
    x = torch.randn(2, 32, 4, 16)
    out = apply_rope(x, freqs[:32])
    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1))


def test_apply_rope_position_zero_is_identity() -> None:
    """With ``freqs_cis`` sliced to position 0 only, the input comes back unchanged."""
    freqs = precompute_rope_freqs(16, 64)
    x = torch.randn(2, 1, 4, 16)
    torch.testing.assert_close(apply_rope(x, freqs[:1]), x)

    # Guard: position 1 must actually rotate, so a do-nothing apply_rope fails.
    assert not torch.allclose(apply_rope(x, freqs[1:2]), x)


def test_apply_rope_dot_product_depends_only_on_relative_position() -> None:
    """``<rope(q, m), rope(k, n)>`` depends only on ``m - n``, not on ``m`` and ``n``.

    The same q and k placed at positions (5, 2) and at (13, 10) are 3 apart both
    times, so their attention logit must be the same. At (5, 1) it must differ.
    """
    torch.manual_seed(0)
    freqs = precompute_rope_freqs(16, 64)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    def logit(m: int, n: int) -> torch.Tensor:
        return (apply_rope(q, freqs[m : m + 1]) * apply_rope(k, freqs[n : n + 1])).sum()

    torch.testing.assert_close(logit(5, 2), logit(13, 10), atol=1e-4, rtol=1e-4)
    assert not torch.allclose(logit(5, 2), logit(5, 1))


# ===========================================================================
# FeedForward (SwiGLU): section 2
# ===========================================================================


def test_feedforward_output_shape() -> None:
    """``FeedForward(dim, hidden_dim)`` maps ``(B, T, dim)`` to ``(B, T, dim)``."""
    # hidden_dim != dim, so mixing up the two widths raises a shape error.
    ff = FeedForward(64, 64 * 8 // 3)
    x = torch.randn(2, 8, 64)
    assert ff(x).shape == (2, 8, 64)


def test_feedforward_matches_swiglu_formula() -> None:
    """Output equals ``down(silu(gate(x)) * up(x))``, all three projections bias-free."""
    torch.manual_seed(0)
    ff = FeedForward(64, 85)
    gate, up, down = (ff.get_submodule(name) for name in ("gate", "up", "down"))
    for proj in (gate, up, down):
        assert isinstance(proj, nn.Linear) and proj.bias is None
    x = torch.randn(2, 8, 64)
    torch.testing.assert_close(ff(x), down(F.silu(gate(x)) * up(x)))


# ===========================================================================
# GQAttention (4) and its KV cache (5)
# ===========================================================================


def test_gqattention_output_shape() -> None:
    """GQAttention maps ``(B, T, dim)`` to ``(B, T, dim)``."""
    cfg = tiny_config()
    attn = GQAttention(cfg)
    x = torch.randn(2, 8, cfg.dim)
    assert attn(x, tiny_freqs(cfg)[:8]).shape == (2, 8, cfg.dim)
    assert isinstance(attn.q_norm, RMSNorm) and isinstance(attn.k_norm, RMSNorm)


def test_gqattention_uses_rope() -> None:
    """Shifting every position (same input, ``freqs_cis[5:13]`` instead of ``[:8]``)
    leaves the output unchanged, but scrambling the phasors changes it.

    The first half holds because attention only sees relative offsets. The
    second half fails if RoPE is never applied.
    """
    torch.manual_seed(0)
    cfg = tiny_config()
    attn = GQAttention(cfg).eval()
    freqs = tiny_freqs(cfg)
    x = torch.randn(2, 8, cfg.dim)
    out = attn(x, freqs[:8])
    torch.testing.assert_close(attn(x, freqs[5:13]), out, atol=1e-5, rtol=1e-4)
    assert not torch.allclose(attn(x, freqs[:8].flip(0)), out)


def test_gqattention_sdpa_matches_manual() -> None:
    """``use_sdpa=True`` and ``use_sdpa=False`` give the same output (atol ~1e-5, fp32)."""
    torch.manual_seed(0)
    cfg = tiny_config()
    attn = GQAttention(cfg).eval()
    x = torch.randn(2, 8, cfg.dim)
    freqs = tiny_freqs(cfg)[:8]
    for causal in (True, False):
        sdpa = attn(x, freqs, causal=causal, use_sdpa=True)
        manual = attn(x, freqs, causal=causal, use_sdpa=False)
        torch.testing.assert_close(sdpa, manual, atol=1e-5, rtol=1e-4)


def test_gqattention_causal_blocks_future() -> None:
    """With ``causal=True``, output at position ``i`` ignores inputs at ``j > i``."""
    torch.manual_seed(0)
    cfg = tiny_config()
    attn = GQAttention(cfg).eval()
    freqs = tiny_freqs(cfg)[:8]
    x = torch.randn(2, 8, cfg.dim)
    x_changed = x.clone()
    x_changed[:, 5:] = torch.randn(2, 3, cfg.dim)  # change only the future

    for use_sdpa in (True, False):
        out = attn(x, freqs, use_sdpa=use_sdpa)
        out_changed = attn(x_changed, freqs, use_sdpa=use_sdpa)
        torch.testing.assert_close(out[:, :5], out_changed[:, :5])
        # Guard: the changed positions do change, so the comparison means something.
        assert not torch.allclose(out[:, 5:], out_changed[:, 5:])


def test_gqattention_kv_cache_accumulates() -> None:
    """After a length-``T`` prefill and one decode step, cached k/v have length ``T + 1``."""
    cfg = tiny_config()
    attn = GQAttention(cfg).eval()
    freqs = tiny_freqs(cfg)
    head_dim = cfg.dim // cfg.n_heads
    cache: dict = {}

    attn(torch.randn(2, 8, cfg.dim), freqs[:8], causal=True, kv_cache=cache)
    # Stored per KV head (n_kv_heads), not expanded to n_heads.
    assert cache["default"]["k"].shape == (2, 8, cfg.n_kv_heads, head_dim)
    assert cache["default"]["v"].shape == (2, 8, cfg.n_kv_heads, head_dim)

    attn(torch.randn(2, 1, cfg.dim), freqs[8:9], causal=False, kv_cache=cache)
    assert cache["default"]["k"].shape == (2, 9, cfg.n_kv_heads, head_dim)
    assert cache["default"]["v"].shape == (2, 9, cfg.n_kv_heads, head_dim)


def test_gqattention_cached_decode_matches_one_shot() -> None:
    """Token-by-token decoding through the cache matches one full causal pass.

    Feed tokens one at a time (``causal=False``, correct RoPE slice each step)
    and compare every position against a single ``causal=True`` pass over the
    whole sequence.
    """
    torch.manual_seed(0)
    cfg = tiny_config()
    attn = GQAttention(cfg).eval()
    freqs = tiny_freqs(cfg)
    x = torch.randn(2, 8, cfg.dim)

    full = attn(x, freqs[:8], causal=True)

    cache: dict = {}
    steps = [
        attn(x[:, t : t + 1], freqs[t : t + 1], causal=False, kv_cache=cache)
        for t in range(8)
    ]
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-5, rtol=1e-4)


# ===========================================================================
# MoEFFN and TransformerBlock: section 7
# ===========================================================================


def test_moeffn_output_shape() -> None:
    """``MoEFFN`` maps ``(B, T, dim)`` to ``(B, T, dim)``."""
    cfg = tiny_config()
    moe = MoEFFN(cfg)
    assert moe(torch.randn(2, 8, cfg.dim)).shape == (2, 8, cfg.dim)


def test_moeffn_router_bias_is_buffer_not_parameter() -> None:
    """``router_bias`` is in ``named_buffers()`` with shape ``(n_experts,)``, not in ``named_parameters()``."""
    cfg = tiny_config()
    moe = MoEFFN(cfg)
    buffers = dict(moe.named_buffers())
    assert "router_bias" in buffers
    assert buffers["router_bias"].shape == (cfg.n_experts,)
    assert not buffers["router_bias"].requires_grad
    assert "router_bias" not in dict(moe.named_parameters())


def test_moeffn_shared_experts_always_fire() -> None:
    """With the routed contribution zeroed, the shared experts still give a non-zero output."""
    cfg = tiny_config(n_shared_experts=2)  # 2, so a mean would not pass as a sum
    moe = MoEFFN(cfg).eval()
    with torch.no_grad():
        for p in moe.get_submodule("routed").parameters():
            p.zero_()  # every routed expert now outputs exactly 0

    x = torch.randn(2, 8, cfg.dim)
    out = moe(x)
    assert out.abs().sum() > 0
    # With routed experts silenced, the output is exactly the shared experts' sum.
    shared = moe.get_submodule("shared").children()
    torch.testing.assert_close(out, sum(expert(x) for expert in shared))


def test_moeffn_routed_weights_are_unbiased_softmax() -> None:
    """With ``n_experts_per_tok == n_experts`` every expert is picked, so the
    routed output is ``sum_e softmax(router(x))[e] * routed[e](x)``, and a large
    ``router_bias`` must not change it (bias steers selection, never weights).
    """
    torch.manual_seed(0)
    cfg = tiny_config(n_experts_per_tok=4)  # top-K covers all 4 experts
    moe = MoEFFN(cfg).eval()
    with torch.no_grad():
        for p in moe.get_submodule("shared").parameters():
            p.zero_()  # isolate the routed path

    x = torch.randn(2, 8, cfg.dim)
    weights = moe.get_submodule("router")(x).softmax(dim=-1)  # (B, T, n_experts)
    experts = list(moe.get_submodule("routed").children())
    expected = sum(
        weights[..., e : e + 1] * expert(x) for e, expert in enumerate(experts)
    )
    torch.testing.assert_close(moe(x), expected, atol=1e-5, rtol=1e-4)

    moe.get_buffer("router_bias").copy_(torch.tensor([5.0, -5.0, 0.0, 2.0]))
    torch.testing.assert_close(moe(x), expected, atol=1e-5, rtol=1e-4)


def test_moeffn_update_router_bias_moves_load_toward_balance() -> None:
    """``update_router_bias`` lowers the bias of an overloaded expert and raises an underloaded one.

    Skew ``expert_load`` by hand, call the update, and check each changed by
    ``router_bias_update_rate`` in the right direction, and that
    ``expert_load`` is reset to zero.
    """
    cfg = tiny_config()
    moe = MoEFFN(cfg)
    rate = cfg.router_bias_update_rate
    # Mean load is 4: expert 0 is overloaded, 1 is underloaded, 2 and 3 are balanced.
    moe.get_buffer("expert_load").copy_(torch.tensor([6.0, 2.0, 4.0, 4.0]))

    moe.update_router_bias()

    # Fetch after the update, in case it reassigned the buffers rather than editing them.
    bias, load = moe.get_buffer("router_bias"), moe.get_buffer("expert_load")
    expected = torch.tensor([-rate, rate, 0.0, 0.0])
    torch.testing.assert_close(bias, expected.to(bias.dtype))
    assert torch.all(load == 0)


def test_moeffn_eval_does_not_accumulate_load() -> None:
    """In ``eval()`` mode a forward pass leaves ``expert_load`` unchanged."""
    cfg = tiny_config()
    moe = MoEFFN(cfg)
    x = torch.randn(2, 8, cfg.dim)

    moe.eval()
    moe(x)
    assert torch.all(moe.get_buffer("expert_load") == 0)

    # Guard: in train mode it does count, one selection per token per top-K slot.
    moe.train()
    moe(x)
    assert moe.get_buffer("expert_load").sum().item() == 2 * 8 * cfg.n_experts_per_tok


def test_transformer_block_dense_and_moe_shapes() -> None:
    """Both ``use_moe=False`` and ``use_moe=True`` blocks map ``(B, T, dim)`` to ``(B, T, dim)``."""
    cfg = tiny_config()
    x = torch.randn(2, 8, cfg.dim)
    freqs = tiny_freqs(cfg)[:8]
    for use_moe, ffn_type in ((False, FeedForward), (True, MoEFFN)):
        block = TransformerBlock(cfg, use_moe=use_moe)
        assert isinstance(block.ffn, ffn_type)
        assert block(x, freqs).shape == (2, 8, cfg.dim)


def test_transformer_block_residuals() -> None:
    """With both sublayers' output projections zeroed, the block is the identity.

    Each sublayer adds onto ``x`` (``x = x + sublayer(norm(x))``), so silencing
    ``attn.wo`` and ``ffn.down`` must return the input unchanged. Fails if a
    residual is missing or a norm is applied to the residual stream itself.
    """
    cfg = tiny_config()
    block = TransformerBlock(cfg).eval()
    with torch.no_grad():
        block.get_parameter("attn.wo.weight").zero_()
        block.get_parameter("ffn.down.weight").zero_()
    x = torch.randn(2, 8, cfg.dim)
    torch.testing.assert_close(block(x, tiny_freqs(cfg)[:8]), x)


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
