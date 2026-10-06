# Ouroboros (work in progress)

Started: 2026-06-04

**A recurrent-depth (looped) transformer written from scratch in PyTorch, as a learning project: a Prelude / Recurrent / Coda design with LTI-stable looping, GQA attention with a KV cache, and a fine-grained MoE.**

Instead of stacking more unique layers, Ouroboros loops one weight-shared
transformer block a variable number of times, so the same parameters do more
computation the longer they run. It is trained with a random loop count each
step, so the number of loops can be turned up or down at test time.

The goal is to understand how looped models work and how they are trained, and
to learn the standard parts of a modern LLM along the way. Reading comes first;
the code makes the reading concrete. The papers behind it are in the
[roadmap](ROADMAP.md), section by section.

---

## Highlights

- **Prelude / Recurrent / Coda** (from Huginn): a looped core between small
  dense encode and decode stacks, rather than a fully looped network.
- **LTI-stable looping** (from Parcae): the recurrent update uses a diagonal
  state matrix whose spectral radius is `ρ(A) < 1` by construction, so the
  linear part of the loop cannot blow up whatever the optimizer does.
- **Test-time loop count**: a random loop count per training step plus
  truncated backprop (from Huginn), and a sinusoidal loop-index signal that is
  defined at any depth.
- **GQA with QK-norm and a KV cache**: each loop iteration keeps its own cache
  entries, so grouped KV heads matter more here than in a plain transformer.
- **FlashAttention, understood**: trains on PyTorch SDPA, with FlashAttention's
  tiled online-softmax algorithm reimplemented in plain PyTorch and checked
  against it.
- **Fine-grained Mixture-of-Experts** in the looped block (from DeepSeekMoE and
  DeepSeek-V3): routed plus shared experts, balanced with an aux-loss-free
  router-bias update.
- **Small**: about 15.6M parameters, trained on TinyStories on a single Colab
  T4.

---

## Architecture

```
 input_ids (B, T)
      │
      ▼
 [Embedding]  vocab_size -> dim, weight-tied with LM head
      │  x (B, T, dim)
      ▼
 [Prelude]  prelude_layers x TransformerBlock (GQA + dense SwiGLU), run once
      │  x (B, T, dim)
      ├──────────────► e := x        (encoded input; frozen, re-injected every loop)
      ▼
 ┌─[Recurrent Block]  h_0 = e ────────────────────────────────────┐
 │  for t in range(n_loops):        # training: n_loops is random │
 │    h_loop   = loop_index_embedding(h, t, loop_dim)             │
 │    combined = RMSNorm(h_loop + e)                              │
 │    trans    = TransformerBlock(combined)   # GQA + MoE         │
 │    h        = A*h + B*e + trans            # LTIInjection      │
 └────────────────────────────────────────────────────────────────┘
      │  x := h (B, T, dim)
      ▼
 [Coda]  coda_layers x TransformerBlock (GQA + dense SwiGLU), run once
      │
      ▼
 [RMSNorm] -> [LM head (tied)]
      │
      ▼
 logits (B, T, vocab_size)
```

The stability core is the LTI recurrence

```
h_{t+1} = A · h_t + B · e + Transformer(h_t, e),   with   ρ(A) < 1.
```

`A` is a diagonal state matrix discretized by zero-order hold from
`A_continuous = -exp(log_A)` (always negative), so every diagonal entry lands in
`(0, 1)` for any parameter values. `ρ(A)` doubles as a cheap stability readout
during training.

The component reference (math, shapes, gotchas, and the reasoning behind each
design choice) is in [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Installation

Requires Python 3.10+ and PyTorch 2.1+.

```bash
git clone <repo-url> Ouroboros
cd Ouroboros
pip install -e .            # or: pip install -r requirements.txt
```

---

## Usage

```python
import torch
from ouroboros import Ouroboros, OuroborosConfig

cfg = OuroborosConfig()                           # ~15.6M-parameter default
model = Ouroboros(cfg)
input_ids = torch.randint(0, cfg.vocab_size, (1, 16))

logits = model(input_ids)                         # (1, 16, vocab_size), 8 loops
logits = model(input_ids, n_loops=32)             # same weights, more loops

# Autoregressive generation with a KV cache
tokens = model.generate(input_ids, max_new_tokens=64, n_loops=8)
```

Training and the loop-count sweep:

```bash
python training/train.py --tiny --max-steps 50          # smoke run
python training/train.py --wandb-project ouroboros       # main run
python training/loop_sweep.py --ckpt checkpoints/final.pt
python training/kv_speedup.py --ckpt checkpoints/final.pt
```

---

## Documentation

| Doc | Contents |
| --- | --- |
| [`ROADMAP.md`](ROADMAP.md) | The learning plan: what to read and what to build, section by section, plus the training setup and the three metrics. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | The architecture: layers, mechanisms, and why each was chosen. |

---

## License

MIT. See [`LICENSE`](LICENSE).
