"""Canonical hyperparameter configuration for the Ouroboros model.

This module defines :class:`OuroborosConfig`, the single source of truth for
every architectural and training hyperparameter in the project. Defaults target
a small, Google Colab **T4-friendly** research model (16 GB VRAM, Turing sm75,
FP16) — *not* a frontier-scale model. Every component (attention, MoE, recurrence,
RoPE, norm) reads its dimensions from this dataclass, so the field names here are
the project-wide contract: any doc or module that references a config field must
use these exact names.

Ouroboros is an independent, from-scratch implementation inspired by the
recurrent-depth transformer literature (Universal Transformers, Parcae,
DeepSeekMoE / DeepSeek-V3). It is a Prelude / Recurrent / Coda design with
fine-grained MoE (routed + shared experts), GQA attention, and LTI-constrained
stable injection.

Sizing notes (encode these as guidance, not as logic):

* **Fine-grained MoE rule of thumb:** choose the routed expert width so that
  ``expert_dim ~= dim // (n_experts // n_experts_per_tok)``. With the defaults
  (``dim=512``, ``n_experts=8``, ``n_experts_per_tok=2``) this gives
  ``512 // (8 // 2) = 128``; the default ``expert_dim=256`` deliberately runs a
  touch wider for a small-model capacity margin.
* **Shared-expert width:** the always-active shared experts are sized *larger*
  than routed experts — width ``expert_dim * n_experts_per_tok`` — so they can
  absorb common cross-domain structure that would otherwise be redundantly
  re-learned by many routed experts.
* **Prelude / Coda FFN width:** the dense SwiGLU FFN in the (non-MoE) prelude and
  coda blocks uses width ``dim * 4 // 3`` (a deliberate parameter-budget choice,
  smaller than the common ``8/3 * dim`` SwiGLU sizing).
* **Tiny test config:** a "tiny" smoke-test configuration overrides the defaults
  with e.g. ``dim=64, n_heads=4, n_kv_heads=2, n_experts=4, prelude_layers=1,
  coda_layers=1, max_loop_iters=4`` to keep CPU unit tests fast. It is built by
  the ``tiny_config`` helper in the test suite — it is **not** hardcoded here.
* **T4 training config:** the realistic single-GPU training target is a small
  model of roughly **10-30M parameters** trained on WikiText-103 or a FineWeb-Edu
  slice. At small ``dim`` the (tied) embedding table can dominate the parameter
  count, which motivates the modest ``vocab_size=8192`` default. This config is
  likewise assembled at the training-script boundary, not baked into the dataclass.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class OuroborosConfig:
    """Architecture and training hyperparameters for :class:`~ouroboros.model.Ouroboros`.

    The defaults below define a small, T4-friendly research model. Fields are
    grouped by subsystem (core, MoE FFN, recurrence, load balancing,
    RoPE/norm/init/regularization). Cross-field invariants are validated in
    :meth:`__post_init__`, so an invalid combination fails at construction with
    a clear message rather than as a shape error deep in a forward pass.
    """

    # --- Core ---
    # small BPE vocab keeps the embedding table modest at small dim
    vocab_size: int = 8192
    dim: int = 512  # residual-stream width
    n_heads: int = 8  # query heads
    n_kv_heads: int = 2  # GQA key/value heads (n_heads % n_kv_heads == 0)
    max_seq_len: int = 1024  # RoPE precomputation length
    max_loop_iters: int = 8  # default recurrent depth T at inference
    prelude_layers: int = 2  # standard blocks before the loop
    coda_layers: int = 2  # standard blocks after the loop

    # --- MoE FFN (used only inside the Recurrent Block) ---
    n_experts: int = 8  # routed experts
    n_shared_experts: int = 1  # always-active shared experts
    n_experts_per_tok: int = 2  # top-K routed per token
    expert_dim: int = 256  # fine-grained expert hidden width

    # --- Recurrence ---
    # channels receiving loop-index embedding; None -> dim // 8
    loop_index_dim: Optional[int] = None
    # LTI-constrained injection (the stability mechanism). False replaces the
    # LTI update with a naive residual injection ``h = transformer_out + e`` —
    # the ablation arm of the stability experiment (EXPERIMENTS.md, exp 1).
    use_lti: bool = True

    # --- Load balancing (Ouroboros completes what reference impls leave as a stub) ---
    # aux-loss-free LB: per-step bias nudge magnitude
    router_bias_update_rate: float = 1e-3

    # --- RoPE / norm / init / regularization ---
    # RoPE base (small-model default; 500000 for long context)
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6  # RMSNorm epsilon
    init_std: float = 0.02  # N(0, init_std) weight init
    dropout: float = 0.0  # 0.0 disables; 0.1 typical for pretraining
    max_output_tokens: int = 1024  # generation cap

    def __post_init__(self) -> None:
        """Validate cross-field invariants; raise ``ValueError`` on violation."""
        if self.dim % self.n_heads != 0:
            raise ValueError(
                f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
            )
        head_dim = self.dim // self.n_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim = dim // n_heads = {head_dim} must be even "
                "(RoPE rotates adjacent channel pairs)"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads}) for GQA grouping"
            )
        if self.n_experts_per_tok > self.n_experts:
            raise ValueError(
                f"n_experts_per_tok ({self.n_experts_per_tok}) cannot exceed "
                f"n_experts ({self.n_experts})"
            )
        loop_dim = (
            self.loop_index_dim if self.loop_index_dim is not None else self.dim // 8
        )
        if loop_dim % 2 != 0 or not 0 < loop_dim <= self.dim:
            raise ValueError(
                f"loop_index_dim (resolved to {loop_dim}) must be even and in "
                f"(0, dim={self.dim}] — it is consumed as sin/cos pairs"
            )
