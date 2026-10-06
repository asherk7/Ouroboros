"""Ouroboros: a recurrent-depth (looped) transformer written from scratch in PyTorch.

A learning project. The model is a Prelude / Recurrent / Coda design: a few
dense blocks encode the input once, one weight-shared block (GQA attention plus
a fine-grained MoE FFN) is looped a variable number of times with LTI-stable
input injection, and a few dense blocks decode the result. Training samples a
random loop count each step, so the loop count can be changed at test time.

The components below are re-exported so callers can do, e.g.::

    from ouroboros import Ouroboros, OuroborosConfig

    model = Ouroboros(OuroborosConfig())

See ``ARCHITECTURE.md`` for the component reference and the forward-pass
diagram.
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
    sample_n_loops,
)
from .rope import apply_rope, precompute_rope_freqs
from .tiled_attention import tiled_attention

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
    "tiled_attention",
    # MoE
    "Expert",
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
