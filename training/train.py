"""Single-GPU training script for the Ouroboros recurrent-depth transformer.

Pipeline (single Colab T4, ~15M-parameter model)
-------------------------------------------------
1. **Config / CLI.** Parse flags into an ``OuroborosConfig`` and a
   ``TrainConfig``. ``--tiny`` selects a sub-1M smoke-test model.
2. **Tokenizer.** Train an 8192-token byte-level BPE on the TinyStories train
   split once (:func:`train_tokenizer`) and save it; reuse it afterwards.
3. **Data.** Stream TinyStories (``roneneldan/TinyStories``) via ``datasets``,
   tokenize, append an end-of-text token per story, concatenate, cache the
   token ids to disk once, and pack into ``max_seq_len + 1`` windows giving
   ``(input_ids, targets)`` pairs.
4. **Model.** ``Ouroboros(cfg)`` on the device.
5. **Optimizer / schedule.** AdamW, ``betas=(0.9, 0.95)``, ``weight_decay=0.1``
   on matmul weights only; linear warmup then cosine decay (:func:`get_lr`).
6. **Mixed precision.** fp16 autocast plus ``GradScaler`` on a T4 (no bf16
   tensor cores); bf16 without a scaler on Ampere+; fp32 on CPU. Clip the
   global grad norm to 1.0 after unscaling.
7. **Random loop count (Huginn).** Each optimizer step samples ``n_loops`` with
   :func:`~ouroboros.recurrence.sample_n_loops` and uses it for every
   micro-batch in that step. The recurrent block backprops through only the
   last ``cfg.backprop_loops`` loops.
8. **Load balancing.** After each optimizer step call
   ``model.recurrent.block.ffn.update_router_bias()``.
9. **Checkpoints.** ``ckpt_dir/latest.pt`` every ``ckpt_interval`` steps and
   ``ckpt_dir/final.pt`` at the end. ``--resume`` restores the model,
   optimizer, scaler, and step, so a Colab disconnect costs at most
   ``ckpt_interval`` steps.
10. **Stability runs** (ROADMAP section 11): ``--no-lti`` sets ``use_lti=False``.
    Run it and a default run at the same high learning rate for a short budget.
11. **Logging** (W&B if ``wandb_project`` is set, always stdout): loss, grad
    norm, lr, tokens/s, ``rho(A)`` with the mean and min of ``A``, the sampled
    ``n_loops``, and the variance of expert load.

Usage (from the repo root, after ``pip install -e .``)
-------------------------------------------------------
.. code-block:: bash

    python training/train.py --tiny --max-steps 50     # smoke run first
    python training/train.py --wandb-project ouroboros  # main run
    python training/train.py --resume checkpoints/latest.pt  # after a disconnect
    python training/train.py --lr 3e-3 --max-steps 1000 --run-name lti-high-lr
    python training/train.py --lr 3e-3 --max-steps 1000 --run-name naive-high-lr --no-lti
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ouroboros.config import OuroborosConfig
from ouroboros.model import Ouroboros
from ouroboros.moe import MoEFFN
from ouroboros.recurrence import sample_n_loops

EOT = "<|endoftext|>"  # separates stories, so the model learns where one ends

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Optimization and runtime hyperparameters for one training run.

    Separate from :class:`OuroborosConfig`, which describes the architecture.

    Attributes:
        dataset: Hugging Face dataset id.
        tokenizer_path: Where the trained BPE tokenizer is saved and loaded. The
            tokenized splits are cached next to it as ``{split}.bin``.
        max_stories: Use only the first N stories per split (smoke runs);
            ``None`` uses them all.
        batch_size: Micro-batch size in sequences.
        grad_accum_steps: Micro-batches per optimizer step.
        max_steps: Total optimizer steps.
        lr: Peak learning rate after warmup.
        min_lr: Learning-rate floor at the end of cosine decay.
        warmup_steps: Linear warmup length in optimizer steps.
        weight_decay: AdamW decoupled weight decay (matmul weights only).
        beta1: AdamW first-moment decay.
        beta2: AdamW second-moment decay (0.95 is the usual LLM choice).
        grad_clip: Global gradient-norm clip threshold.
        precision: ``"fp16"`` (T4, uses ``GradScaler``), ``"bf16"`` (Ampere+),
            or ``"fp32"``. Forced to ``"fp32"`` on CPU.
        seed: RNG seed for ``random``, ``numpy``, and ``torch``.
        log_interval: Steps between metric logs.
        eval_interval: Steps between validation-perplexity evaluations.
        eval_batches: Validation batches per evaluation (a quick estimate).
        ckpt_interval: Steps between checkpoint writes.
        ckpt_dir: Checkpoint directory (gitignored). Point it at Google Drive on
            Colab so checkpoints survive a disconnect.
        resume: Checkpoint to resume from; ``None`` starts fresh.
        wandb_project: W&B project name; ``None`` logs to stdout only.
        run_name: W&B run name.
        device: ``"cuda"`` / ``"cpu"``; auto-detected when ``None``.
    """

    dataset: str = "roneneldan/TinyStories"
    tokenizer_path: str = "data/tokenizer.json"
    max_stories: Optional[int] = None
    batch_size: int = 16
    grad_accum_steps: int = 4
    max_steps: int = 5_000
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    precision: str = "fp16"
    seed: int = 1337
    log_interval: int = 10
    eval_interval: int = 500
    eval_batches: int = 50
    ckpt_interval: int = 1000
    ckpt_dir: str = "checkpoints"
    resume: Optional[str] = None
    wandb_project: Optional[str] = None
    run_name: Optional[str] = None
    device: Optional[str] = None


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------


def get_lr(step: int, train_cfg: TrainConfig) -> float:
    """Learning rate for an optimizer step: linear warmup, then cosine decay.

    Warm up linearly to ``lr`` over ``warmup_steps``, then follow a half-period
    cosine down to ``min_lr`` at ``max_steps``; hold ``min_lr`` after that.

    Args:
        step: Current optimizer step (0-indexed).
        train_cfg: Provides ``lr``, ``min_lr``, ``warmup_steps``, ``max_steps``.

    Returns:
        The learning rate for this step.
    """
    lr, min_lr = train_cfg.lr, train_cfg.min_lr
    warmup = train_cfg.warmup_steps

    # warmup: Adam's second-moment estimates are garbage for the first steps, so
    # big updates then can wreck the init. (step + 1) so step 0 isn't a wasted lr=0
    if step < warmup:
        return lr * (step + 1) / warmup

    # cosine: progress goes 0 -> 1 over the decay phase, coeff goes 1 -> 0
    decay_steps = max(1, train_cfg.max_steps - warmup)
    progress = min(1.0, (step - warmup) / decay_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


# ---------------------------------------------------------------------------
# Tokenizer and data
# ---------------------------------------------------------------------------


def iter_stories(train_cfg: TrainConfig, split: str) -> Iterator[str]:
    """Yield story texts from one split, capped at ``max_stories``.

    Streaming avoids holding the whole dataset in memory; we only pass over it
    once per split anyway (results are cached to disk).
    """
    from datasets import load_dataset

    ds = load_dataset(train_cfg.dataset, split=split, streaming=True)
    texts = (row["text"] for row in ds)
    return itertools.islice(texts, train_cfg.max_stories)


def train_tokenizer(train_cfg: TrainConfig, vocab_size: int) -> None:
    """Train a byte-level BPE tokenizer on the TinyStories train split and save it.

    Uses the ``tokenizers`` library with ``vocab_size`` tokens, including one
    end-of-text special token. Runs once; skips if ``tokenizer_path`` exists.

    Args:
        train_cfg: Provides ``dataset`` and ``tokenizer_path``.
        vocab_size: Must equal ``OuroborosConfig.vocab_size``.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    path = Path(train_cfg.tokenizer_path)
    if path.exists():
        return

    # byte-level: the base alphabet is all 256 bytes, so no text is ever <unk>
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print(f"training tokenizer ({vocab_size} tokens) on {train_cfg.dataset}...")
    tokenizer.train_from_iterator(iter_stories(train_cfg, "train"), trainer)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))


def tokenize_split(train_cfg: TrainConfig, split: str) -> np.ndarray:
    """Return every token id of a split as one flat array, cached to disk.

    Each story gets ``EOT`` appended, then they're all concatenated. Stored as
    ``uint16`` (vocab 8192 < 65536) to halve the size; the first call
    tokenizes and writes ``{split}.bin``, later calls memory-map it.
    """
    from tokenizers import Tokenizer

    cache = Path(train_cfg.tokenizer_path).parent / f"{split}.bin"
    if cache.exists():
        # memmap: read windows straight off disk instead of loading it all into RAM
        return np.memmap(cache, dtype=np.uint16, mode="r")

    tokenizer = Tokenizer.from_file(train_cfg.tokenizer_path)
    eot_id = tokenizer.token_to_id(EOT)

    print(f"tokenizing {split} split...")
    chunks = []
    stories = iter_stories(train_cfg, split)
    # encode_batch in chunks: fast (parallel, in Rust) without one giant list
    while batch := list(itertools.islice(stories, 10_000)):
        for enc in tokenizer.encode_batch(batch):
            chunks.append(np.array(enc.ids + [eot_id], dtype=np.uint16))

    tokens = np.concatenate(chunks)
    # write then rename: a disconnect mid-write can't leave a truncated cache behind
    tmp = cache.with_suffix(".tmp")
    tokens.tofile(tmp)
    os.replace(tmp, cache)
    print(f"{split}: {len(tokens):,} tokens -> {cache}")
    return tokens


class PackedDataset(Dataset):
    """Consecutive ``seq_len + 1`` windows over one long token array.

    Window ``i`` starts at ``i * seq_len``, so neighbouring windows share one
    token (the last target of one is the first input of the next) and every
    token is predicted exactly once per epoch.
    """

    def __init__(self, tokens: np.ndarray, seq_len: int) -> None:
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self) -> int:
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = i * self.seq_len
        w = torch.from_numpy(
            self.tokens[start : start + self.seq_len + 1].astype(np.int64)
        )
        # shift by one: position t's target is token t + 1
        return w[:-1], w[1:]


def build_dataloader(
    train_cfg: TrainConfig,
    model_cfg: OuroborosConfig,
    split: str = "train",
) -> DataLoader:
    """Build a packed-sequence ``DataLoader`` over TinyStories.

    Tokenize every story, append the end-of-text token, concatenate, and cut
    into windows of ``max_seq_len + 1`` tokens. Each window gives
    ``input_ids = w[:-1]`` and ``targets = w[1:]``. Shuffle the train split only.

    Args:
        train_cfg: Provides ``dataset``, ``tokenizer_path``, ``batch_size``.
        model_cfg: Provides ``vocab_size`` and ``max_seq_len``.
        split: ``"train"`` or ``"validation"``.

    Returns:
        A ``DataLoader`` yielding ``(input_ids, targets)``, each
        ``(batch_size, max_seq_len)`` ``torch.long``.
    """
    dataset = PackedDataset(tokenize_split(train_cfg, split), model_cfg.max_seq_len)
    is_train = split == "train"
    return DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=is_train,
        drop_last=is_train,  # every train step sees full batches
    )


# ---------------------------------------------------------------------------
# Optimizer, loss, metrics
# ---------------------------------------------------------------------------


def build_optimizer(model: Ouroboros, train_cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with weight decay on matmul weights only.

    Two parameter groups: 2-D weights get ``weight_decay``; norm gains,
    embeddings, and the LTI parameters (``log_A``, ``log_dt``, ``B``) get none.
    ``router_bias`` is a buffer, so it is not in either group.

    Args:
        model: The model to optimize.
        train_cfg: Provides ``weight_decay``, ``lr``, ``beta1``, ``beta2``.

    Returns:
        A ``torch.optim.AdamW`` with two parameter groups.
    """
    # named_parameters() yields the tied head/embedding weight once, as embed.weight.
    # decay pulls weights toward 0: fine for matmuls (keeps them small), wrong for
    # norm gains (should sit near 1), LTI params (would drag A toward a fixed value),
    # and the embedding (rare tokens get few gradients, so decay would erase them)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.dim() >= 2 and name != "embed.weight":
            decay.append(p)
        else:
            no_decay.append(p)

    groups = [
        {"params": decay, "weight_decay": train_cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, lr=train_cfg.lr, betas=(train_cfg.beta1, train_cfg.beta2)
    )


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy.

    Args:
        logits: ``(B, T, vocab_size)``.
        targets: ``(B, T)`` ``torch.long``. Already shifted by the dataset.

    Returns:
        Scalar mean cross-entropy.
    """
    # cross_entropy wants (N, C) and (N,); float() so fp16 logits don't overflow
    return F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())


def lti_stats(model: Ouroboros) -> dict[str, float]:
    """Summarize the LTI state matrix ``A`` for logging.

    ``A`` is diagonal, so ``rho(A) = max(get_A())``. It must stay below 1.

    Args:
        model: The model.

    Returns:
        ``{"rho_A": max, "A_mean": mean, "A_min": min}`` of
        ``model.recurrent.injection.get_A()``, or ``{}`` for a ``use_lti=False``
        model, which has no ``A``.
    """
    injection = model.recurrent.injection
    if injection is None:
        return {}
    with torch.no_grad():
        A = injection.get_A()
    return {"rho_A": A.max().item(), "A_mean": A.mean().item(), "A_min": A.min().item()}


def recurrent_moe(model: Ouroboros) -> MoEFFN:
    """The recurrent block's MoE FFN (the only MoE in the model)."""
    ffn = model.recurrent.block.ffn
    assert isinstance(ffn, MoEFFN)
    return ffn


def expert_load_variance(model: Ouroboros) -> float:
    """Variance of the normalized per-expert load in the recurrent MoE.

    Read ``model.recurrent.block.ffn.expert_load`` before
    ``update_router_bias()`` resets it. Should trend down as balancing works.

    Args:
        model: The model.

    Returns:
        Variance of ``expert_load / expert_load.sum()``.
    """
    load = recurrent_moe(model).expert_load
    total = load.sum()
    if total == 0:
        return 0.0
    # normalize so the number is comparable across batch sizes and loop counts
    return (load / total).var().item()


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: Ouroboros,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    train_cfg: TrainConfig,
) -> None:
    """Save everything needed to resume, writing atomically.

    Configs are stored as dicts, not dataclass objects: ``TrainConfig`` lives in
    ``__main__`` when this file runs as a script, so a pickled instance would not
    load from ``loop_sweep.py``.
    """
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,  # optimizer steps completed
        "model_cfg": asdict(model.cfg),
        "train_cfg": asdict(train_cfg),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # write then rename: a disconnect mid-save can't corrupt the previous checkpoint
    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str,
    model: Ouroboros,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> int:
    """Restore model, optimizer, and scaler in place; return the step to resume at."""
    device = next(model.parameters()).device
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt["step"])


# ---------------------------------------------------------------------------
# Train loop and evaluation
# ---------------------------------------------------------------------------


def infinite(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Cycle through the loader forever (a new shuffle each epoch)."""
    while True:
        yield from loader


def log(metrics: dict[str, Any], step: int, use_wandb: bool) -> None:
    """Print a compact line, and send to W&B if it's on."""
    line = " | ".join(
        f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
        for k, v in metrics.items()
    )
    print(f"step {step} | {line}")
    if use_wandb:
        import wandb

        wandb.log(metrics, step=step)


def train(
    model: Ouroboros,
    train_cfg: TrainConfig,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    use_wandb: bool = False,
) -> None:
    """Run the training loop.

    Per optimizer step:

    1. Sample ``n_loops = sample_n_loops(model.cfg.mean_loops)``.
    2. For each of ``grad_accum_steps`` micro-batches: forward with that
       ``n_loops`` under autocast, scale the loss (fp16), backward.
    3. Unscale, clip to ``grad_clip``, optimizer step.
    4. Log (every ``log_interval``): loss, grad norm, lr, tokens/s,
       :func:`lti_stats`, ``n_loops``, :func:`expert_load_variance`.
    5. ``model.recurrent.block.ffn.update_router_bias()``.
    6. Every ``eval_interval``: validation perplexity at ``model.cfg.mean_loops``.
    7. Every ``ckpt_interval`` as ``ckpt_dir/latest.pt``, and at the end as
       ``ckpt_dir/final.pt``: see :func:`save_checkpoint`.

    If ``train_cfg.resume`` is set, the model, optimizer, scaler, and step are
    restored first, and training continues up to ``max_steps``.

    Args:
        model: The model, already on the device.
        train_cfg: Training configuration (``device`` and ``precision`` resolved).
        train_loader: From :func:`build_dataloader`.
        val_loader: Optional validation loader.
        use_wandb: Also log to the active W&B run.
    """
    device = train_cfg.device or "cpu"
    optimizer = build_optimizer(model, train_cfg)

    # fp16 has a tiny range, so small gradients underflow to 0. the scaler multiplies
    # the loss by a big factor before backward, then divides it back out before the
    # step. bf16 and fp32 have fp32's range, so they don't need it (enabled=False)
    use_fp16 = train_cfg.precision == "fp16"
    device_type = torch.device(device).type  # "cuda:0" -> "cuda" for amp
    scaler = torch.amp.GradScaler(device_type, enabled=use_fp16)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(train_cfg.precision)
    autocast = (
        torch.autocast(device_type=device_type, dtype=dtype) if dtype else nullcontext()
    )

    start_step = 0
    if train_cfg.resume:
        start_step = load_checkpoint(train_cfg.resume, model, optimizer, scaler)
        # reseed so the resumed run doesn't replay the same shuffle and loop counts
        set_seed(train_cfg.seed + start_step)
        print(f"resumed from {train_cfg.resume} at step {start_step}")

    moe = recurrent_moe(model)
    ckpt_dir = Path(train_cfg.ckpt_dir)
    batches = infinite(train_loader)
    tokens_per_step = (
        train_cfg.batch_size * train_cfg.grad_accum_steps * model.cfg.max_seq_len
    )

    model.train()
    t0 = time.time()
    for step in range(start_step, train_cfg.max_steps):
        # the schedule is a pure function of step, so resuming picks up the right lr
        lr = get_lr(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # one loop count for the whole optimizer step: every micro-batch's gradient
        # then comes from the same depth, so the accumulated gradient is one
        # consistent estimate (and the batch runs as a single tensor)
        n_loops = sample_n_loops(model.cfg.mean_loops)

        loss_sum = 0.0
        for _ in range(train_cfg.grad_accum_steps):
            input_ids, targets = (t.to(device) for t in next(batches))
            with autocast:
                logits = model(input_ids, n_loops=n_loops)
                loss = compute_loss(logits, targets)
            # divide so the accumulated gradient is the mean over micro-batches,
            # same as one big batch
            scaler.scale(loss / train_cfg.grad_accum_steps).backward()
            loss_sum += loss.item()

        # unscale first, so clipping compares the true gradient norm to grad_clip
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_cfg.grad_clip
        ).item()
        # skips the step if the fp16 grads overflowed to inf/nan, then adjusts the scale
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        done = step + 1
        if done % train_cfg.log_interval == 0 or done == train_cfg.max_steps:
            elapsed = time.time() - t0
            t0 = time.time()
            steps_logged = done % train_cfg.log_interval or train_cfg.log_interval
            metrics: dict[str, Any] = {
                "loss": loss_sum / train_cfg.grad_accum_steps,
                "grad_norm": grad_norm,
                "lr": lr,
                "n_loops": n_loops,
                "tokens_per_s": tokens_per_step * steps_logged / elapsed,
                # read before update_router_bias() zeroes the counts
                "expert_load_var": expert_load_variance(model),
                **lti_stats(model),
            }
            log(metrics, done, use_wandb)

        # once per optimizer step (not per micro-batch): uses the load summed over
        # every micro-batch and loop, then resets it
        moe.update_router_bias()

        if val_loader is not None and done % train_cfg.eval_interval == 0:
            ppl = evaluate(
                model, val_loader, model.cfg.mean_loops, train_cfg.eval_batches
            )
            log({"val_ppl": ppl}, done, use_wandb)

        if done % train_cfg.ckpt_interval == 0:
            save_checkpoint(
                ckpt_dir / "latest.pt", model, optimizer, scaler, done, train_cfg
            )

    save_checkpoint(
        ckpt_dir / "final.pt", model, optimizer, scaler, train_cfg.max_steps, train_cfg
    )


@torch.no_grad()
def evaluate(
    model: Ouroboros,
    val_loader: DataLoader,
    n_loops: int,
    max_batches: Optional[int] = None,
) -> float:
    """Validation perplexity at a fixed loop count.

    Args:
        model: The model (switched to eval mode inside, previous mode restored).
        val_loader: Validation loader.
        n_loops: Loop count for every forward pass.
        max_batches: Optional cap on batches, for a faster estimate.

    Returns:
        ``exp(mean cross-entropy)``.
    """
    # eval mode: all loops run (no truncated backprop) and MoE stops counting load
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    losses = []
    for input_ids, targets in itertools.islice(val_loader, max_batches):
        logits = model(input_ids.to(device), n_loops=n_loops)
        losses.append(compute_loss(logits, targets.to(device)).item())

    model.train(was_training)
    if not losses:
        raise ValueError("validation loader yielded no batches")
    # perplexity = exp(cross-entropy): the effective number of tokens it's choosing from
    return math.exp(sum(losses) / len(losses))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Architecture knobs: ``--dim``, ``--mean-loops``, ``--backprop-loops``,
    ``--no-lti`` (naive injection, for the stability metric), ``--tiny``.
    Optimization knobs: ``--lr``, ``--max-steps``, ``--batch-size``,
    ``--grad-accum-steps``, ``--warmup-steps``, ``--precision``. Runtime:
    ``--wandb-project``, ``--run-name``, ``--seed``, ``--ckpt-dir``, ``--resume``.

    Flags default to ``None`` so :func:`build_configs` only overrides what was set.

    Returns:
        The parsed namespace.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # architecture
    p.add_argument("--tiny", action="store_true", help="sub-1M smoke-test model")
    p.add_argument("--dim", type=int)
    p.add_argument("--mean-loops", type=int)
    p.add_argument("--backprop-loops", type=int)
    p.add_argument("--no-lti", action="store_true", help="naive injection")
    # optimization
    p.add_argument("--lr", type=float)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--grad-accum-steps", type=int)
    p.add_argument("--warmup-steps", type=int)
    p.add_argument("--precision", choices=["fp16", "bf16", "fp32"])
    # runtime
    p.add_argument("--wandb-project")
    p.add_argument("--run-name")
    p.add_argument("--seed", type=int)
    p.add_argument("--ckpt-dir")
    p.add_argument("--resume", help="checkpoint to resume from")
    return p.parse_args()


def build_configs(args: argparse.Namespace) -> Tuple[OuroborosConfig, TrainConfig]:
    """Turn parsed args into ``(model_cfg, train_cfg)``.

    ``--tiny`` gives a smoke-test model (``dim=64``, ``n_heads=4``,
    ``n_experts=4``, ``mean_loops=4``, ``backprop_loops=2``) trained on 20k
    stories with its own tokenizer and cache under ``data/tiny/``.

    Args:
        args: From :func:`parse_args`.

    Returns:
        ``(model_cfg, train_cfg)``.
    """
    model_kw: dict[str, Any] = {}
    train_kw: dict[str, Any] = {}
    if args.tiny:
        model_kw = dict(
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
            max_seq_len=128,
        )
        train_kw = dict(
            tokenizer_path="data/tiny/tokenizer.json",
            max_stories=20_000,
            batch_size=8,
            grad_accum_steps=1,
            warmup_steps=10,
            log_interval=5,
            eval_interval=25,
            eval_batches=10,
            ckpt_interval=25,
            ckpt_dir="checkpoints/tiny",
        )

    # explicit flags win over the defaults and --tiny
    for field, arg in [
        ("dim", args.dim),
        ("mean_loops", args.mean_loops),
        ("backprop_loops", args.backprop_loops),
    ]:
        if arg is not None:
            model_kw[field] = arg
    if args.no_lti:
        model_kw["use_lti"] = False

    for field, arg in [
        ("lr", args.lr),
        ("max_steps", args.max_steps),
        ("batch_size", args.batch_size),
        ("grad_accum_steps", args.grad_accum_steps),
        ("warmup_steps", args.warmup_steps),
        ("precision", args.precision),
        ("wandb_project", args.wandb_project),
        ("run_name", args.run_name),
        ("seed", args.seed),
        ("ckpt_dir", args.ckpt_dir),
        ("resume", args.resume),
    ]:
        if arg is not None:
            train_kw[field] = arg

    return OuroborosConfig(**model_kw), TrainConfig(**train_kw)


def set_seed(seed: int) -> None:
    """Seed ``random``, ``numpy``, and ``torch``.

    Args:
        seed: The seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA


def main() -> None:
    """Entry point.

    parse args -> build configs -> ``set_seed`` -> train tokenizer if missing ->
    build loaders -> build model -> init W&B if configured -> :func:`train`
    (which builds the optimizer and handles ``--resume``).
    """
    model_cfg, train_cfg = build_configs(parse_args())
    if train_cfg.device is None:
        train_cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if train_cfg.device == "cpu" and train_cfg.precision != "fp32":
        print(f"cpu: using fp32 instead of {train_cfg.precision}")
        train_cfg.precision = "fp32"
    set_seed(train_cfg.seed)

    train_tokenizer(train_cfg, model_cfg.vocab_size)
    train_loader = build_dataloader(train_cfg, model_cfg, "train")
    val_loader = build_dataloader(train_cfg, model_cfg, "validation")

    model = Ouroboros(model_cfg).to(train_cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M params on {train_cfg.device}")

    use_wandb = train_cfg.wandb_project is not None
    if use_wandb:
        import wandb

        wandb.init(
            project=train_cfg.wandb_project,
            name=train_cfg.run_name,
            config={"model": asdict(model_cfg), "train": asdict(train_cfg)},
        )

    train(model, train_cfg, train_loader, val_loader, use_wandb)


if __name__ == "__main__":
    main()
