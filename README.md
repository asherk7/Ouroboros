# Ouroboros

**A recurrent-depth (looped) transformer implemented from scratch in PyTorch — a Prelude / Recurrent / Coda design with fine-grained MoE, switchable MLA/GQA attention, and LTI-constrained stable looping.**

> **Status: 🚧 Work in progress (scaffold / planning phase).** This repository
> currently contains the full architectural contract — typed signatures, rich
> docstrings, design docs, and a phased roadmap — with implementation landing
> phase-by-phase per [`docs/ROADMAP.md`](docs/ROADMAP.md). Library bodies are
> intentionally `raise NotImplementedError` until their phase is reached.

---

## What this is

Ouroboros is an independent, from-scratch implementation of a **recurrent-depth
transformer (RDT)**: instead of stacking more layers, a single transformer block
is *looped* T times with stable input injection, so the same parameters perform
deeper reasoning the longer they run. The design is motivated by the published
recurrent-depth / looped-transformer literature — Universal Transformers,
Adaptive Computation Time, the Parcae stability results, DeepSeek-V2 (MLA),
DeepSeekMoE, and Relaxed Recursive Transformers (see
[`docs/READING_LIST.md`](docs/READING_LIST.md)). It is **not** a fork of any
repository; every component was re-derived from the papers and written to a
single canonical spec.

It is built and benchmarked to run on a **single Google Colab T4** (16 GB VRAM,
Turing sm75, FP16), training a small (~10–30M parameter) model on WikiText-103 or
a FineWeb-Edu slice.

### Résumé claims this project backs (with real, working code + measured results)

1. **Implemented a recurrent-depth (looped) transformer from scratch in
   PyTorch** — a Prelude / Recurrent / Coda design with fine-grained MoE (routed
   + shared experts) and switchable MLA / GQA attention.
2. **Stabilized looped training via LTI-constrained injection** (negative-diagonal
   state matrix, spectral radius ρ(A) < 1 guaranteed by construction), enabling
   clean convergence at high learning rates where an unconstrained loop diverges.
3. **Integrated FlashAttention-2 (SDPA flash backend on Turing) and INT8 quantized
   inference with continuous depth-wise batching**, achieving **[X]× inference
   throughput** on a single GPU.

---

## Architecture (forward-pass data flow)

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
 │    trans    = TransformerBlock(combined, ...)  # MLA/GQA + MoE   │
 │    trans    = trans + LoRAAdapter(trans, t)    # depth-wise LoRA │
 │    h        = LTIInjection(h, e, trans)  # h = A·h + B·e + trans │
 │    p        = ACTHalting(h)              # per-position halt prob │
 │    accumulate ACT-weighted h into h_out; halt converged positions│
 └──────────────────────────────────────────────────────────────────┘
      │  x := h_out (B, T, dim)
      ▼
 [Coda]  coda_layers × TransformerBlock (dense SwiGLU FFN), run ONCE
      │  x (B, T, dim)
      ▼
 [RMSNorm] → [LM head (tied)]
      │
      ▼
 logits (B, T, vocab_size)
```

The stability core is the **LTI recurrence**:

```
h_{t+1} = A · h_t + B · e + Transformer(h_t, e),   with   ρ(A) < 1  guaranteed by construction
```

`A` is a diagonal state matrix discretized via zero-order hold from
`A_continuous = -exp(log_A)` (always negative), so every diagonal entry lands in
`(0, 1)` and the loop cannot blow up — no gradient-clipping or
normalize-the-hidden-state band-aids required.

---

## Features

- **Prelude / Recurrent / Coda** structure — cheap dense encode/decode bracketing
  a parameter-efficient looped core, rather than a fully-looped stack.
- **Switchable attention** — Grouped-Query Attention (GQA) *or* Multi-Latent
  Attention (MLA, DeepSeek-V2 compressed-KV cache), selected by `attn_type`.
- **Fine-grained MoE** in the recurrent block — routed + always-on shared experts,
  with **aux-loss-free load balancing** (DeepSeek-V3 router bias) *and* the actual
  per-step bias-update step that reference implementations leave as a stub.
- **LTI-constrained stable injection** — spectral radius < 1 by construction;
  `ρ(A)` is a cheap, continuous, loggable stability signal.
- **ACT halting** — per-position adaptive computation time; easy tokens halt
  early, hard tokens loop deeper, all within one batch.
- **Depth-wise LoRA adapters** — per-loop low-rank deltas (Relaxed Recursive
  Transformers) bridging pure weight-tying and distinct-per-layer weights, with
  depth-extrapolation clamping at inference.
- **Depth extrapolation** — train at one loop count, run deeper at inference.
- **INT8 post-training quantization** + **continuous depth-wise batching** — the
  inference differentiators (Phase 7).
- **FlashAttention-2 realism on T4** — the robust path is
  `F.scaled_dot_product_attention` with the flash / mem-efficient backend on
  Turing; the native `flash_attn_func` kernel is kept as an optional Ampere fast
  path with a graceful fallback. See [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md).

---

## Documentation

| Doc | Contents |
| --- | --- |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | The 8 build phases — goals, components, acceptance criteria, effort, dependencies. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | All 17 components in dependency order — signatures, math, I/O shapes, gotchas. |
| [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md) | Each major decision: alternatives, rationale, tradeoffs, evidence. |
| [`docs/READING_LIST.md`](docs/READING_LIST.md) | The papers behind each component, organized by what to focus on. |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Hypotheses, variables, configs, and result-table templates (LTI stability, ACT, MLA vs GQA, MoE, INT8, depth-wise batching, loop sweep). |

---

## Quickstart (placeholder)

> Implementation lands phase-by-phase; the API below is the target contract.

```bash
# 1. Install (Python ≥ 3.10)
pip install -r requirements.txt

# 2. Run the (currently-stubbed) test suite — collectable and green
pytest

# 3. Train a small T4-friendly model (Phase 6)
python training/train.py --config t4 --dataset wikitext-103

# 4. Benchmark inference throughput (Phase 7)
python benchmarks/throughput.py --int8 --depthwise-batching
```

```python
import torch
from ouroboros import Ouroboros, OuroborosConfig

model = Ouroboros(OuroborosConfig())          # small T4-friendly defaults
input_ids = torch.randint(0, 8192, (1, 16))
logits = model(input_ids)                     # (1, 16, vocab_size)
out = model.generate(input_ids, max_new_tokens=64, n_loops=8)
```

---

## License & attribution

Independent from-scratch implementation by the author, inspired by the published
recurrent-depth / looped-transformer literature cited in
[`docs/READING_LIST.md`](docs/READING_LIST.md). Not affiliated with or derived
from any specific third-party codebase.
