"""Integration-test stubs for the full :class:`~ouroboros.model.Ouroboros` model.

These exercise the assembled model end to end: embedding, Prelude, recurrent
block, Coda, and the tied LM head. Each test states its assertion and skips with
``pytest.skip("stub: section 9")`` (the ``ROADMAP.md`` section where the
full model is wired up) so the suite stays green before the model exists.
"""

from __future__ import annotations

from typing import Any

import pytest

# Referenced only inside skipped bodies for now, so F401 is suppressed.
from ouroboros import Ouroboros, OuroborosConfig  # noqa: F401


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


def test_forward_output_shape_and_finite() -> None:
    """``forward`` maps ``(B, T)`` ids to finite logits of shape ``(B, T, vocab_size)``."""
    pytest.skip("stub: section 9")


def test_forward_single_token_T1() -> None:
    """A ``T=1`` forward (the decode-step path, ``causal=False``) returns ``(B, 1, vocab_size)``."""
    pytest.skip("stub: section 9")


def test_forward_raises_past_max_seq_len() -> None:
    """``forward`` raises ``ValueError`` when ``start_pos + T > max_seq_len``.

    Slicing ``freqs_cis`` past its end does not raise by itself, so the model
    must check; ``generate`` with a long prompt plus ``max_new_tokens`` is the
    realistic trigger.
    """
    pytest.skip("stub: section 9")


def test_weight_tying_head_shares_embedding() -> None:
    """``model.head.weight is model.embed.weight`` (same object, not just equal)."""
    pytest.skip("stub: section 9")


def test_lti_spectral_radius_below_one_end_to_end() -> None:
    """``model.recurrent.injection.get_A()`` is finite and strictly inside ``(0, 1)``."""
    pytest.skip("stub: section 9")


def test_cached_decode_matches_full_context() -> None:
    """KV-cached decoding gives the same logits as a full forward pass.

    Prefill a prompt into a fresh cache, decode one token with the right
    ``start_pos``, and compare its logits against the last position of one
    full-context forward over the same sequence (same ``n_loops``). This checks
    the ``start_pos`` RoPE slice and the per-loop ``recurrent_loop_{t}`` keys.
    """
    pytest.skip("stub: section 9")


def test_generate_with_and_without_cache_match() -> None:
    """``generate`` gives identical tokens with ``use_cache=True`` and ``False``.

    Use greedy sampling (``top_k=1``) so the comparison is deterministic.
    """
    pytest.skip("stub: section 9")


def test_naive_arm_builds_and_runs() -> None:
    """A ``use_lti=False`` model has ``recurrent.injection is None`` and gives finite logits."""
    pytest.skip("stub: section 9")


def test_generate_output_shape() -> None:
    """``generate(input_ids, max_new_tokens=k)`` on ``(B, T)`` returns ``(B, T + k)``."""
    pytest.skip("stub: section 9")


def test_many_loops_stay_finite_and_change_output() -> None:
    """Far more loops than ``mean_loops`` run, stay finite, and change the output.

    ``n_loops=64`` on the tiny model (``mean_loops=4``) gives finite logits that
    differ from ``n_loops=2``. This is the property the loop sweep relies on.
    """
    pytest.skip("stub: section 9")
