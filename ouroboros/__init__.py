"""Ouroboros: a recurrent-depth (looped) transformer in PyTorch.

Prelude blocks encode the input once, one shared GQA + MoE block loops a
variable number of times, and Coda blocks decode. See ``ARCHITECTURE.md``.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .attention import GQAttention
from .block import TransformerBlock
from .config import OuroborosConfig
from .model import Ouroboros
from .feedforward import FeedForward
from .moe import MoEFFN
from .norm import RMSNorm
from .recurrence import (
    LTIInjection,
    RecurrentBlock,
    loop_index_embedding,
    sample_n_loops,
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
    # Feed-forward and MoE
    "FeedForward",
    "MoEFFN",
    # Block
    "TransformerBlock",
    # Recurrence
    "loop_index_embedding",
    "sample_n_loops",
    "LTIInjection",
    "RecurrentBlock",
    # Model
    "Ouroboros",
    # Metadata
    "__version__",
]
