"""Tests for the parts of ``training/train.py`` that break silently.

The loop itself is checked by watching a smoke run; these cover the pieces
whose bugs don't crash: the lr schedule, the weight-decay groups, the loss, the
packing shift, and checkpoint resume. No dataset download: data is random ids.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# training/ isn't a package; put it on the path so `import train` works
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

import train  # noqa: E402

from ouroboros import Ouroboros  # noqa: E402
from tests.test_model import tiny_config  # noqa: E402


def test_get_lr_warmup_peak_and_floor() -> None:
    """Warmup rises to ``lr`` by its last step, then cosine falls to ``min_lr`` and holds."""
    cfg = train.TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=100)
    lrs = [train.get_lr(s, cfg) for s in range(120)]

    assert 0 < lrs[0] < cfg.lr / 5  # small but nonzero at step 0
    assert math.isclose(lrs[9], cfg.lr)  # peak at the end of warmup
    assert all(a <= b for a, b in zip(lrs[:10], lrs[1:10]))  # warmup rises
    assert all(a >= b for a, b in zip(lrs[10:100], lrs[11:100]))  # decay falls
    assert math.isclose(lrs[100], cfg.min_lr)
    assert all(math.isclose(lr, cfg.min_lr) for lr in lrs[100:])


def test_optimizer_groups_cover_every_param_once() -> None:
    """Every parameter is in exactly one group; only 2-D non-embedding weights decay."""
    model = Ouroboros(tiny_config())
    opt = train.build_optimizer(model, train.TrainConfig())
    decay, no_decay = opt.param_groups

    ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))

    assert decay["weight_decay"] > 0 and no_decay["weight_decay"] == 0
    assert all(p.dim() >= 2 for p in decay["params"])
    no_decay_ids = {id(p) for p in no_decay["params"]}
    assert id(model.embed.weight) in no_decay_ids
    assert id(model.recurrent.injection.log_A) in no_decay_ids  # type: ignore[union-attr]


def test_compute_loss_matches_cross_entropy() -> None:
    """Uniform logits give ``ln(vocab)``; random ones match flattened ``F.cross_entropy``."""
    V = 50
    targets = torch.randint(0, V, (2, 7))
    uniform = torch.zeros(2, 7, V)
    assert math.isclose(
        train.compute_loss(uniform, targets).item(), math.log(V), rel_tol=1e-5
    )

    logits = torch.randn(2, 7, V)
    expected = F.cross_entropy(logits.view(-1, V), targets.view(-1))
    torch.testing.assert_close(train.compute_loss(logits, targets), expected)


def test_packed_dataset_shifts_targets_by_one() -> None:
    """Window ``i`` is ``tokens[i*L : i*L + L]`` with targets one to the right."""
    tokens = np.arange(23, dtype=np.uint16)
    ds = train.PackedDataset(tokens, seq_len=5)
    assert len(ds) == 4  # (23 - 1) // 5

    for i in range(len(ds)):
        x, y = ds[i]
        assert x.dtype == torch.long and x.shape == (5,)
        assert x.tolist() == list(range(5 * i, 5 * i + 5))
        assert y.tolist() == list(range(5 * i + 1, 5 * i + 6))


def test_train_saves_and_resumes(tmp_path: Path) -> None:
    """A run saves ``final.pt``; resuming from a mid-run checkpoint reaches ``max_steps``.

    Resume must restore the step and the optimizer state, not restart them.
    """
    cfg = tiny_config(max_seq_len=16)
    tokens = np.random.randint(0, cfg.vocab_size, 2000).astype(np.uint16)
    loader = DataLoader(train.PackedDataset(tokens, 16), batch_size=4, shuffle=True)
    tcfg = train.TrainConfig(
        max_steps=4,
        warmup_steps=1,
        grad_accum_steps=1,
        precision="fp32",
        device="cpu",
        log_interval=2,
        ckpt_interval=2,
        ckpt_dir=str(tmp_path),
    )

    torch.manual_seed(0)
    train.train(Ouroboros(cfg), tcfg, loader)
    final = torch.load(tmp_path / "final.pt", weights_only=True)
    latest = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert final["step"] == 4 and latest["step"] == 4
    assert final["model_cfg"]["dim"] == cfg.dim  # configs stored as plain dicts

    # a 2-step run, then resume it into a fresh model and train to step 4
    tcfg.max_steps, tcfg.ckpt_dir = 2, str(tmp_path / "half")
    torch.manual_seed(0)
    train.train(Ouroboros(cfg), tcfg, loader)
    half = tmp_path / "half" / "final.pt"

    tcfg.max_steps, tcfg.resume, tcfg.ckpt_dir = 4, str(half), str(tmp_path / "resumed")
    torch.manual_seed(123)
    model = Ouroboros(cfg)
    train.train(model, tcfg, loader)
    resumed = torch.load(tmp_path / "resumed" / "final.pt", weights_only=True)
    assert resumed["step"] == 4
    # Adam's step counter only advances on real steps, so 2 + 2 = 4 means it resumed
    adam_steps = next(iter(resumed["optimizer"]["state"].values()))["step"]
    assert int(adam_steps) == 4
