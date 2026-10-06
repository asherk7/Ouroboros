"""Evaluate one trained checkpoint at many test-time loop counts.

Metric 1 (the main one): validation perplexity as a function of the
loop count used at test time, from a single model trained with a random loop
count (mean ``cfg.mean_loops``). Expect perplexity to improve with more loops
and then level off (Parcae reports the same saturating shape), and to stay
finite even at 64 loops. ``rho(A) < 1`` keeps the linear part of the loop
contractive, though it does not by itself bound the whole nonlinear loop.

A stub: bodies raise ``NotImplementedError``.

Usage:
.. code-block:: bash

    python training/loop_sweep.py --ckpt checkpoints/final.pt
"""

from __future__ import annotations

import argparse
from typing import Sequence

DEFAULT_LOOPS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)


def sweep(ckpt_path: str, loops: Sequence[int] = DEFAULT_LOOPS) -> dict[int, float]:
    """Load a checkpoint and return validation perplexity at each loop count.

    The checkpoint stores configs as plain dicts (see ``train`` in
    ``train.py``), so rebuild with ``OuroborosConfig(**ckpt["model_cfg"])``;
    this loads under ``torch.load(weights_only=True)``. Then build the validation
    loader with ``build_dataloader`` from ``train.py`` (importable as ``train``,
    since running this script puts ``training/`` on ``sys.path``), and call
    ``evaluate`` once per entry of ``loops``.

    Args:
        ckpt_path: Path to a checkpoint saved by ``training/train.py``.
        loops: Test-time loop counts to evaluate.

    Returns:
        ``{n_loops: perplexity}``.
    """
    raise NotImplementedError


def plot(results: dict[int, float], out_path: str) -> None:
    """Plot perplexity against loop count (log2 x-axis) and save it.

    Args:
        results: Output of :func:`sweep`.
        out_path: Image path to write.
    """
    raise NotImplementedError


def main() -> None:
    """Parse ``--ckpt`` and ``--out``, run :func:`sweep`, print a table, :func:`plot`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="loop_sweep.png")
    raise NotImplementedError


if __name__ == "__main__":
    main()
