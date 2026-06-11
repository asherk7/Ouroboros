"""Ouroboros — a recurrent-depth (looped) transformer implemented from scratch in PyTorch.

This package exposes the public API of the Ouroboros architecture: a
Prelude / Recurrent / Coda design with fine-grained Mixture-of-Experts
(routed + shared experts), GQA attention, LTI-constrained stable input
injection (spectral radius < 1 by construction), and a KV-cached /
continuous-depth-wise-batching inference path. The model is designed to be
trained on a single Google Colab T4 GPU (16 GB, Turing sm75, FP16).

The components below are re-exported here so callers can do, e.g.::

    from ouroboros import Ouroboros, OuroborosConfig

    model = Ouroboros(OuroborosConfig())

See ``docs/ARCHITECTURE.md`` for the full component contract and the
forward-pass data-flow diagram.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .attention import GQAttention
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

# Single-source the version from pyproject.toml package metadata; fall back for
# the from-source (not pip-installed) case.
try:
    __version__ = _version("ouroboros")
except PackageNotFoundError:  # pragma: no cover - running from a raw checkout
    __version__ = "0.0.0.dev0"

__all__ = [
    # Config
    "OuroborosConfig",
    # Primitives
    "RMSNorm",
    "precompute_rope_freqs",
    "apply_rope",
    # Attention
    "GQAttention",
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
