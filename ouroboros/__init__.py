"""Ouroboros — a recurrent-depth (looped) transformer implemented from scratch in PyTorch.

This package exposes the public API of the Ouroboros architecture: a
Prelude / Recurrent / Coda design with fine-grained Mixture-of-Experts
(routed + shared experts), switchable MLA/GQA attention, LTI-constrained
stable input injection (spectral radius < 1 by construction), and an INT8 /
continuous-depth-wise-batching inference path. The model is designed to be
trained on a single Google Colab T4 GPU (16 GB, Turing sm75, FP16).

The components below are re-exported here so callers can do, e.g.::

    from ouroboros import Ouroboros, OuroborosConfig

    model = Ouroboros(OuroborosConfig())

See ``docs/ARCHITECTURE.md`` for the full component contract and the
forward-pass data-flow diagram.
"""

from .attention import GQAttention, MLAttention
from .block import TransformerBlock
from .config import OuroborosConfig
from .model import Ouroboros
from .moe import Expert, MoEFFN
from .norm import RMSNorm
from .recurrence import (
    LTIInjection,
    RecurrentBlock,
    loop_index_embedding,
)
from .rope import apply_rope, precompute_rope_freqs

__version__ = "0.1.0"

__all__ = [
    # Config
    "OuroborosConfig",
    # Primitives
    "RMSNorm",
    "precompute_rope_freqs",
    "apply_rope",
    # Attention
    "GQAttention",
    "MLAttention",
    # MoE
    "Expert",
    "MoEFFN",
    # Block
    "TransformerBlock",
    # Recurrence
    "loop_index_embedding",
    "LTIInjection",
    "RecurrentBlock",
    # Model
    "Ouroboros",
    # Metadata
    "__version__",
]
