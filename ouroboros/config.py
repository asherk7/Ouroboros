"""Hyperparameter configuration for Ouroboros.

Defaults give a ~20M-parameter model sized for TinyStories on a Colab T4.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class OuroborosConfig:
    """Architecture hyperparameters; invalid combinations raise ``ValueError``."""

    # --- Core ---
    # small BPE vocab keeps the embedding table modest at small dim
    vocab_size: int = 8192
    dim: int = 512  # residual-stream width
    n_heads: int = 8  # query heads
    n_kv_heads: int = 2  # GQA key/value heads (n_heads % n_kv_heads == 0)
    max_seq_len: int = 512  # RoPE precomputation length (TinyStories is short)
    prelude_layers: int = 2  # dense blocks before the loop
    coda_layers: int = 2  # dense blocks after the loop

    # --- MoE FFN (used only inside the Recurrent Block) ---
    n_experts: int = 8  # routed experts
    n_shared_experts: int = 1  # always-active shared experts
    n_experts_per_tok: int = 2  # top-K routed per token
    expert_dim: int = 256  # fine-grained expert hidden width
    # aux-loss-free load balancing: per-step router-bias nudge size
    router_bias_update_rate: float = 1e-3

    # --- Recurrence ---
    # mean of the per-step random loop count during training (Huginn), and the
    # default loop count at inference
    mean_loops: int = 8
    # truncated backprop: gradients flow through only the last k loops
    backprop_loops: int = 4
    # channels receiving the loop-index embedding; None -> see loop_dim
    loop_index_dim: Optional[int] = None
    # LTI-stable injection. False swaps in the naive update h = trans + e: the
    # comparison arm for the stability metric (ROADMAP section 11)
    use_lti: bool = True

    # --- RoPE / norm / init / regularization ---
    rope_theta: float = 10000.0  # RoPE base
    norm_eps: float = 1e-6  # RMSNorm epsilon (also used by QK-norm)
    init_std: float = 0.02  # N(0, init_std) weight init
    dropout: float = 0.0  # 0.0 disables

    @property
    def loop_dim(self) -> int:
        """``loop_index_dim`` if set, else ``dim // 8`` rounded down to even (min 2)."""
        if self.loop_index_dim is not None:
            return self.loop_index_dim
        return max(2, (self.dim // 8) // 2 * 2)

    def __post_init__(self) -> None:
        """Validate cross-field invariants."""
        for name in (
            "vocab_size",
            "dim",
            "n_heads",
            "n_kv_heads",
            "n_experts",
            "n_experts_per_tok",
            "expert_dim",
            "mean_loops",
            "backprop_loops",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} ({value!r}) must be >= 1 and an int")
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
        loop_dim = self.loop_dim
        if loop_dim % 2 != 0 or not 0 < loop_dim <= self.dim:
            raise ValueError(
                f"loop_index_dim (resolved to {loop_dim}) must be even and in "
                f"(0, dim={self.dim}]; it is consumed as sin/cos pairs"
            )
