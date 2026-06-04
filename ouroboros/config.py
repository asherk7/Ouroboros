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
DeepSeek-V2 / DeepSeekMoE / DeepSeek-V3, Relaxed Recursive Transformers). It is a
Prelude / Recurrent / Coda design with fine-grained MoE (routed + shared experts),
switchable MLA/GQA attention, LTI-constrained stable injection, ACT halting, and
depth-wise LoRA.

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
  with e.g. ``dim=64, n_heads=4, n_kv_heads=1, n_experts=4, prelude_layers=1,
  coda_layers=1, max_loop_iters=2`` to keep CPU unit tests fast. It is described
  in the docs and constructed at call sites — it is **not** hardcoded here.
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
    grouped by subsystem (core, attention, MoE FFN, recurrence/halting, load
    balancing, RoPE/norm/init/regularization). MLA-only fields are ignored when
    ``attn_type == "gqa"`` and vice versa.
    """

    # --- Core ---
    # small BPE vocab keeps the embedding table modest at small dim
    vocab_size: int = 8192
    dim: int = 512  # residual-stream width
    n_heads: int = 8  # query heads
    # GQA key/value heads (n_heads % n_kv_heads == 0); ignored by MLA
    n_kv_heads: int = 2
    max_seq_len: int = 1024  # RoPE precomputation length
    max_loop_iters: int = 8  # default recurrent depth T at inference
    prelude_layers: int = 2  # standard blocks before the loop
    coda_layers: int = 2  # standard blocks after the loop

    # --- Attention ("gqa" | "mla") ---
    attn_type: str = "gqa"  # default GQA: simpler + has the FA2 fast path
    kv_lora_rank: int = 128  # [MLA] compressed KV latent cached
    q_lora_rank: int = 256  # [MLA] compressed Q latent
    qk_rope_head_dim: int = 32  # [MLA] per-head dims that receive RoPE
    qk_nope_head_dim: int = 64  # [MLA] per-head dims without RoPE
    v_head_dim: int = 64  # [MLA] per-head value dim

    # --- MoE FFN (used only inside the Recurrent Block) ---
    n_experts: int = 8  # routed experts
    n_shared_experts: int = 1  # always-active shared experts
    n_experts_per_tok: int = 2  # top-K routed per token
    expert_dim: int = 256  # fine-grained expert hidden width

    # --- Recurrence / stability / halting ---
    act_threshold: float = 0.99  # ACT cumulative-probability halting threshold
    lora_rank: int = 8  # depth-wise LoRA bottleneck rank
    # channels receiving loop-index embedding; None -> dim // 8
    loop_index_dim: Optional[int] = None

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
