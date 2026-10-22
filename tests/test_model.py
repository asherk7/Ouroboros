"""Integration-test stubs for the full :class:`~ouroboros.model.Ouroboros` model.

These exercise the assembled model end to end: embedding, Prelude, recurrent
block, Coda, and the tied LM head (``ROADMAP.md`` section 9). Each test states its
assertion in its docstring.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from ouroboros import Ouroboros, OuroborosConfig


def tiny_config(**overrides: Any) -> OuroborosConfig:
    """Build a tiny, CPU-fast :class:`OuroborosConfig` for integration tests.

    Same as the unit-test config plus ``vocab_size=256``.

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
        vocab_size=256,
    )
    tiny.update(overrides)
    return OuroborosConfig(**tiny)


def tiny_model(**overrides: Any) -> Ouroboros:
    """A seeded tiny model in eval mode (no truncated backprop, no router-bias updates)."""
    torch.manual_seed(0)
    return Ouroboros(tiny_config(**overrides)).eval()


def random_ids(B: int = 2, T: int = 8) -> torch.Tensor:
    """Random token ids, ``(B, T)``, from the tiny 256-token vocab."""
    return torch.randint(0, 256, (B, T))


def test_forward_output_shape_and_finite() -> None:
    """``forward`` maps ``(B, T)`` ids to finite logits of shape ``(B, T, vocab_size)``."""
    model = tiny_model()
    logits = model(random_ids(2, 8))
    assert logits.shape == (2, 8, 256)
    assert torch.isfinite(logits).all()


def test_forward_single_token_T1() -> None:
    """A ``T=1`` forward (the decode-step path, ``causal=False``) returns ``(B, 1, vocab_size)``."""
    model = tiny_model()
    logits = model(random_ids(2, 1))
    assert logits.shape == (2, 1, 256)


def test_forward_raises_past_max_seq_len() -> None:
    """``forward`` raises ``ValueError`` when ``start_pos + T > max_seq_len``.

    Slicing ``rope_angles`` past its end does not raise by itself, so the model
    must check; ``generate`` with a long prompt plus ``max_new_tokens`` is the
    realistic trigger.
    """
    model = tiny_model()
    with pytest.raises(ValueError):
        model(random_ids(1, 4), start_pos=61)  # 61 + 4 > 64
    with pytest.raises(ValueError):
        model.generate(random_ids(1, 60), max_new_tokens=10)


def test_weight_tying_head_shares_embedding() -> None:
    """``model.head.weight is model.embed.weight`` (same object, not just equal)."""
    model = tiny_model()
    assert model.head.weight is model.embed.weight


def test_lti_spectral_radius_below_one_end_to_end() -> None:
    """``model.recurrent.injection.get_A()`` is finite and strictly inside ``(0, 1)``."""
    injection = tiny_model().recurrent.injection
    assert injection is not None
    A = injection.get_A()
    assert torch.isfinite(A).all()
    assert (A > 0).all() and (A < 1).all()


def test_cached_decode_matches_full_context() -> None:
    """KV-cached decoding gives the same logits as a full forward pass.

    Prefill a prompt into a fresh cache, decode one token with the right
    ``start_pos``, and compare its logits against the last position of one
    full-context forward over the same sequence (same ``n_loops``). This checks
    the ``start_pos`` RoPE slice and the per-loop ``recurrent_loop_{t}`` keys.
    """
    model = tiny_model()
    ids = random_ids(2, 9)

    cache: dict = {}
    model(ids[:, :8], n_loops=3, kv_cache=cache)
    step = model(ids[:, 8:], n_loops=3, kv_cache=cache, start_pos=8)

    full = model(ids, n_loops=3)
    assert "recurrent_loop_2" in cache
    torch.testing.assert_close(step[:, 0], full[:, -1], atol=1e-5, rtol=1e-4)


def test_generate_with_and_without_cache_match() -> None:
    """``generate`` gives identical tokens with ``use_cache=True`` and ``False``.

    Use greedy sampling (``top_k=1``) so the comparison is deterministic. The
    larger ``init_std`` keeps the untrained tied head from just repeating one
    token, which would hide a wrong decode ``start_pos``.
    """
    model = tiny_model(init_std=0.2)
    prompt = random_ids(2, 5)
    cached = model.generate(prompt, max_new_tokens=8, top_k=1, use_cache=True)
    uncached = model.generate(prompt, max_new_tokens=8, top_k=1, use_cache=False)
    assert torch.equal(cached, uncached)


def test_naive_arm_builds_and_runs() -> None:
    """A ``use_lti=False`` model has ``recurrent.injection is None`` and gives finite logits."""
    model = tiny_model(use_lti=False)
    assert model.recurrent.injection is None
    assert torch.isfinite(model(random_ids(2, 8))).all()


def test_generate_output_shape() -> None:
    """``generate(input_ids, max_new_tokens=k)`` on ``(B, T)`` returns ``(B, T + k)``."""
    model = tiny_model()
    out = model.generate(random_ids(2, 5), max_new_tokens=7)
    assert out.shape == (2, 12)


def test_many_loops_stay_finite_and_change_output() -> None:
    """Far more loops than ``mean_loops`` run, stay finite, and change the output.

    ``n_loops=64`` on the tiny model (``mean_loops=4``) gives finite logits that
    differ from ``n_loops=2``. This is the property the loop sweep relies on.
    """
    model = tiny_model()
    ids = random_ids(2, 8)
    many = model(ids, n_loops=64)
    few = model(ids, n_loops=2)
    assert torch.isfinite(many).all()
    assert not torch.allclose(many, few)
