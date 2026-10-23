"""Evaluate one trained checkpoint at many test-time loop counts.

Metric 1 (the main one): validation perplexity as a function of the
loop count used at test time, from a single model trained with a random loop
count (mean ``cfg.mean_loops``). Expect perplexity to improve with more loops
and then level off (Parcae reports the same saturating shape), and to stay
finite even at 64 loops. ``rho(A) < 1`` keeps the linear part of the loop
contractive, though it does not by itself bound the whole nonlinear loop.

Usage:
.. code-block:: bash

    python training/loop_sweep.py --ckpt checkpoints/final.pt
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # no display on Colab / CI; just write the file
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from training.train import TrainConfig, build_dataloader, evaluate, load_model  # noqa: E402

DEFAULT_LOOPS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)


def sweep(
    ckpt_path: str,
    loops: Sequence[int] = DEFAULT_LOOPS,
    max_batches: Optional[int] = 100,
) -> dict[int, float]:
    """Load a checkpoint and return validation perplexity at each loop count.

    The checkpoint stores configs as plain dicts (see ``train`` in
    ``train.py``), so the model is rebuilt with ``OuroborosConfig(**ckpt["model_cfg"])``
    and loads under ``torch.load(weights_only=True)``. The validation loader is
    unshuffled, so every loop count sees the same batches (a paired comparison).

    Args:
        ckpt_path: Path to a checkpoint saved by ``training/train.py``.
        loops: Test-time loop counts to evaluate.
        max_batches: Validation batches per loop count; ``None`` uses all of them.
            The default (~800k tokens) keeps 64 loops to a few minutes on a T4.

    Returns:
        ``{n_loops: perplexity}``.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(ckpt_path, device)
    train_cfg = TrainConfig(**ckpt["train_cfg"])
    val_loader = build_dataloader(train_cfg, model.cfg, "validation")

    # fp16 on the GPU, like training: fp32 is several times slower on a T4
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device == "cuda"
        else nullcontext()
    )

    results = {}
    for n in loops:
        with autocast:
            results[n] = evaluate(model, val_loader, n, max_batches)
        print(f"loops {n:>3} | val_ppl {results[n]:.3f}", flush=True)
    return results


def plot(results: dict[int, float], out_path: str) -> None:
    """Plot perplexity against loop count (log2 x-axis) and save it.

    Args:
        results: Output of :func:`sweep`.
        out_path: Image path to write.
    """
    loops = sorted(results)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loops, [results[n] for n in loops], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xticks(loops, [str(n) for n in loops])
    ax.set_xlabel("test-time loops")
    ax.set_ylabel("validation perplexity")
    ax.set_title("Ouroboros: perplexity vs loop count")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Parse ``--ckpt`` and ``--out``, run :func:`sweep`, print a table, :func:`plot`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="loop_sweep.png")
    parser.add_argument("--max-batches", type=int, default=100)
    args = parser.parse_args()

    results = sweep(args.ckpt, max_batches=args.max_batches)
    best = min(results, key=lambda n: results[n])
    print(
        f"\n1 loop: {results[1]:.3f} | best: {results[best]:.3f} at {best} loops "
        f"| 64 loops: {results[64]:.3f}"
    )
    plot(results, args.out)
    print(f"plot -> {args.out}")


if __name__ == "__main__":
    main()
