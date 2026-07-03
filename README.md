# Ouroboros (work in progress)

**A recurrent-depth (looped) transformer in PyTorch — a Prelude / Recurrent / Coda design with fine-grained MoE, GQA attention, and LTI-constrained stable looping.**

Ouroboros is a recurrent-depth transformer (RDT): instead of stacking more unique
layers, a single transformer block is *looped* a variable number of times with
stable input injection, so the same parameters perform deeper computation the
longer they run. This buys additional effective depth without additional
parameters, and supports **depth extrapolation** — running more loops at inference
than were used during training.

The architecture is grounded in the recurrent-depth / looped-transformer
literature: Universal Transformers for the looped core, Parcae for the LTI
stability constraint, and DeepSeekMoE / DeepSeek-V3 for the mixture-of-experts.
See [`docs/READING_LIST.md`](docs/READING_LIST.md) for the full bibliography.

---

## Highlights

- **Prelude / Recurrent / Coda** — a parameter-efficient looped core bracketed by
  cheap dense encode/decode stacks, rather than a fully looped network.
- **LTI-constrained stable looping** — the recurrent update is parameterized so the
  diagonal state matrix has spectral radius `ρ(A) < 1` by construction, keeping the
  loop contractive and training stable at high learning rates without gradient
  clipping or hidden-state normalization.
- **Grouped-Query Attention (GQA)** — fewer KV heads than query heads for a
  smaller KV cache, with a FlashAttention-2 / SDPA-flash fast path and a manual
  fallback.
- **Fine-grained Mixture-of-Experts** in the recurrent block — routed plus
  always-on shared experts, with aux-loss-free load balancing via a router-bias
  update (DeepSeek-V3).
- **Depth extrapolation** — a fixed loop count at training time, with a sinusoidal
  loop-index signal that lets the shared weights run deeper at inference than they
  were trained on.
- **Optimized inference** — KV-cached decoding and continuous depth-wise batching
  (sequences exit the loop at different convergence-driven depths within one
  batch), the source of the headline throughput multiplier.
- **Compact and single-GPU friendly** — defaults target a small model trainable on a
  single consumer / Colab-class GPU (e.g. a 16 GB T4).

---

## Architecture

```
 input_ids (B, T)
      │
      ▼
 [Embedding]  vocab_size → dim,  weight-tied with LM head
      │  x (B, T, dim)
      ▼
 [Prelude]  prelude_layers × TransformerBlock (dense SwiGLU FFN), run ONCE
      │  x (B, T, dim)
      ├──────────────► e := x        (encoded input; FROZEN, re-injected every loop)
      ▼
 ┌─[Recurrent Block]──────────────────────────────────────────────┐
 │  for t in range(n_loops):                                        │
 │    h_loop   = loop_index_embedding(h, t, loop_dim)   # sinusoid  │
 │    combined = RMSNorm(h_loop + e)                                │
 │    trans    = TransformerBlock(combined, ...)  # GQA + MoE       │
 │    h        = LTIInjection(h, e, trans)  # h = A·h + B·e + trans │
 └──────────────────────────────────────────────────────────────────┘
      │  x := h (B, T, dim) — final hidden state after n_loops
      ▼
 [Coda]  coda_layers × TransformerBlock (dense SwiGLU FFN), run ONCE
      │  x (B, T, dim)
      ▼
 [RMSNorm] → [LM head (tied)]
      │
      ▼
 logits (B, T, vocab_size)
```

The stability core is the LTI recurrence

```
h_{t+1} = A · h_t + B · e + Transformer(h_t, e),   with   ρ(A) < 1.
```

`A` is a diagonal state matrix discretized via zero-order hold from
`A_continuous = -exp(log_A)` (always negative), so every diagonal entry lands in
`(0, 1)`. The spectral-radius bound therefore holds for any parameter values, and
`ρ(A)` doubles as a cheap, continuous stability signal to monitor during training.

A full component-by-component reference — signatures, math, tensor shapes, and
implementation notes — is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Installation

Requires Python ≥ 3.10 and PyTorch ≥ 2.1.

```bash
git clone <repo-url> Ouroboros
cd Ouroboros
pip install -e .            # or: pip install -r requirements.txt
```

The optional inference fast path (`flash-attn`) can be installed with the
`fast` extra; it is not required (`flash-attn` targets Ampere+ GPUs — on a T4
the SDPA flash backend is the realistic path):

```bash
pip install -e ".[fast]"
```

---

## Usage

```python
import torch
from ouroboros import Ouroboros, OuroborosConfig

cfg = OuroborosConfig()                           # small default config
model = Ouroboros(cfg)
input_ids = torch.randint(0, cfg.vocab_size, (1, 16))

# Forward pass — logits of shape (1, 16, vocab_size)
logits = model(input_ids)

# Autoregressive generation with KV cache
tokens = model.generate(input_ids, max_new_tokens=64, n_loops=8)
```

Model size and loop depth are set on the config:

```python
cfg = OuroborosConfig(
    dim=512,
    max_loop_iters=8,       # default recurrent depth
    use_lti=True,           # False = naive injection (the stability-ablation arm)
)
```

A looped model trained at one depth can be run deeper at inference:

```python
logits = model(input_ids, n_loops=16)             # depth extrapolation
```

Training and inference benchmarking entry points live in
[`training/train.py`](training/train.py) and
[`benchmarks/throughput.py`](benchmarks/throughput.py).

---

## Documentation

| Doc | Contents |
| --- | --- |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Every component in dependency order — signatures, math, I/O shapes, implementation notes. |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | The build phases — goals, components, and acceptance criteria. |
| [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md) | Major design choices: alternatives, rationale, and tradeoffs. |
| [`docs/READING_LIST.md`](docs/READING_LIST.md) | The papers behind each component, organized by what to focus on. |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Experiment plans and result templates (LTI stability, depth extrapolation, inference throughput). |

---

## License

MIT. See [`LICENSE`](LICENSE).
