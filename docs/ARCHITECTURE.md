# Ouroboros — Architecture Reference

> **Ouroboros** is a recurrent-depth (looped) transformer language model built from
> scratch in PyTorch. It follows a **Prelude → Recurrent → Coda** design with
> fine-grained Mixture-of-Experts (routed + shared experts), **GQA** attention,
> **LTI-constrained** stable injection (spectral radius `< 1` by construction),
> and — beyond the reference literature — **continuous depth-wise batching** for
> inference.

This document is the detailed architecture reference. It opens with the canonical
forward-pass data-flow diagram, then walks all **13 components** in dependency
order. For each component you get: *what it is*, *why it exists* (the problem it
solves in a looped transformer), *how it works* (math/equations, enough to
implement without any reference repo), *exact tensor shapes & dtypes*, *key
implementation gotchas*, and *where it sits* in the Prelude → Recurrent → Coda
pipeline. A **key design properties** summary table closes the document.

Ouroboros is an independent implementation inspired by the published
recurrent-depth transformer literature — primarily *Parcae* (Prairie et al.,
2026), *DeepSeekMoE* (Dai et al., 2024), *DeepSeek-V3* (2024), and *Universal
Transformers* (Dehghani et al., 2018).
See [`READING_LIST.md`](./READING_LIST.md) for the full bibliography.

**Target hardware:** a single Google Colab **T4** (16 GB VRAM, Turing `sm75`,
FP16). Defaults target a small research model (~10–30M params) trained on
WikiText-103 or a FineWeb-Edu slice. All shapes and benchmarks below are framed
for this reality.

---

## Table of contents

1. [Forward-pass data flow](#1-forward-pass-data-flow)
2. [The LTI recurrence in one line](#2-the-lti-recurrence-in-one-line)
3. [Notation & shape conventions](#3-notation--shape-conventions)
4. [The 13 components](#4-the-13-components)
   - [(1) OuroborosConfig](#1-ouroborosconfig--configpy)
   - [(2) RMSNorm](#2-rmsnorm--normpy)
   - [(3) RoPE](#3-rope--ropepy)
   - [(4) GQAttention](#4-gqattention--attentionpy)
   - [(5) Expert (SwiGLU FFN)](#5-expert-swiglu-ffn--moepy)
   - [(6) MoEFFN](#6-moeffn--moepy)
   - [(7) TransformerBlock](#7-transformerblock--blockpy)
   - [(8) loop_index_embedding](#8-loop_index_embedding--recurrencepy)
   - [(9) LTIInjection](#9-ltiinjection--recurrencepy)
   - [(10) RecurrentBlock](#10-recurrentblock--recurrencepy)
   - [(11) Ouroboros](#11-ouroboros--modelpy)
   - [(12) KV cache & autoregressive generation](#12-kv-cache--autoregressive-generation--modelpy)
   - [(13) Continuous depth-wise batching](#13-continuous-depth-wise-batching--modelpy)
5. [Key design properties (summary table)](#5-key-design-properties-summary-table)

---

## 1. Forward-pass data flow

The canonical end-to-end data flow (spec §3). This same diagram appears in the
[`README.md`](../README.md).

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

The three stages map to three functional roles:

| Stage         | What it does                                                                 | FFN type | Cache keys        | Run count            |
| ------------- | ---------------------------------------------------------------------------- | -------- | ----------------- | -------------------- |
| **Prelude**   | Lifts tokens into the residual stream and forms the *encoded input* `e`.     | Dense    | `prelude_{i}`     | Once per forward     |
| **Recurrent** | Iteratively refines a latent state `h` by re-reading `e` at every loop step. | MoE      | `recurrent_loop_{t}` | `n_loops` times   |
| **Coda**      | Reads out the converged latent into vocabulary logits.                       | Dense    | `coda_{i}`        | Once per forward     |

The split is deliberate: a *pure* fully-looped transformer would have to use the
same weights for tokenization-level work, mid-stream reasoning, and final
read-out. By bookending the loop with a small dense Prelude and Coda, the looped
weights specialize purely in **iterative refinement**, and the I/O surfaces get
dedicated (cheap) capacity. See
[`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md#1) for the full argument.

---

## 2. The LTI recurrence in one line

The mathematical core — the recurrent hidden state evolves as a discrete
**linear time-invariant (LTI)** system with a learned, structurally stable state
matrix:

```
 h_{t+1} = A · h_t  +  B · e  +  Transformer(h_t, e)        with  ρ(A) < 1  by construction
```

- `A` is a learned **diagonal** matrix with every entry in `(0, 1)` → its
  spectral radius `ρ(A) = max(diag(A))` is *always* `< 1`, regardless of what
  gradient descent does to the underlying parameters.
- `B · e` re-injects the (frozen) encoded input at every loop step so the latent
  never forgets the prompt across arbitrary depth.
- `Transformer(h_t, e)` is the nonlinear refinement (attention + MoE) at this
  depth.

Because `ρ(A) < 1` holds *by construction*, the homogeneous part `A · h_t` is a
contraction: errors decay across loop iterations instead of compounding. This is
what lets Ouroboros train at high learning rates without gradient-clipping or
hidden-state-normalization band-aids. `ρ(A)` is the single cheapest, most
informative stability signal in the whole system — log it every step (see §9).

---

## 3. Notation & shape conventions

| Symbol  | Meaning                                                        |
| ------- | ------------------------------------------------------------- |
| `B`     | Batch size                                                    |
| `T`     | Query sequence length (this forward pass)                     |
| `S`     | Total key/value length incl. cache (`S ≥ T`; `S = T` without cache) |
| `dim`   | Residual-stream width (`cfg.dim`)                             |
| `H`     | Number of query heads (`cfg.n_heads`)                         |
| `t`     | Loop iteration index (`0 ≤ t < n_loops`)                      |
| `ρ(A)`  | Spectral radius of the LTI state matrix `A`                   |

**Dtype convention.** Parameters and activations run in the autocast dtype
(`fp16` on T4 with `GradScaler`, or `bf16` on Ampere). Two reductions are forced
to **fp32** for numerical safety: the RMSNorm mean-of-squares (§2) and the RoPE /
LTI computations (§3, §9). The **additive attention mask must match the
activation dtype** — an `fp32` mask on `fp16`/`bf16` logits silently upcasts the
attention matrix and breaks the downstream matmul against `V` in the fallback
path (§4 gotcha). The `freqs_cis` buffer is `complex64`.

---

## 4. The 13 components

Presented in dependency order: each component only depends on those above it.
Public signatures are quoted **verbatim** from the spec — the `.py` stubs must
match them exactly.

---

### (1) `OuroborosConfig` — `config.py`

**What it is.** The single frozen-by-convention hyperparameter contract. A
`@dataclass` whose field declarations *are* the API (the "header"). Every other
component reads its dimensions from this object — there are no magic numbers
elsewhere.

**Why it exists.** A looped MoE model has a large, interdependent hyperparameter
surface (head dims must divide, expert width is derived from routing, the
loop-index dim is a fraction of `dim`, etc.). Centralizing
these into one typed object makes the invariants checkable in one place and keeps
the tiny-test / T4-training / frontier configs interchangeable.

**How it works.** Plain dataclass with defaults targeting a small T4-friendly
model. Its only logic is `__post_init__`, which validates the cross-field
invariants (head divisibility, even RoPE/loop-index dims, `n_experts_per_tok <=
n_experts`) so an invalid combination fails at construction with a clear message
instead of as a shape error deep in a forward pass. The canonical fields:

```python
@dataclass
class OuroborosConfig:
    # --- Core ---
    vocab_size: int = 8192          # small BPE vocab keeps the embedding table modest at small dim
    dim: int = 512                  # residual-stream width
    n_heads: int = 8                # query heads
    n_kv_heads: int = 2             # GQA key/value heads (n_heads % n_kv_heads == 0)
    max_seq_len: int = 1024         # RoPE precomputation length
    max_loop_iters: int = 8         # default recurrent depth T at inference
    prelude_layers: int = 2         # standard blocks before the loop
    coda_layers: int = 2            # standard blocks after the loop

    # --- MoE FFN (used only inside the Recurrent Block) ---
    n_experts: int = 8              # routed experts
    n_shared_experts: int = 1       # always-active shared experts
    n_experts_per_tok: int = 2      # top-K routed per token
    expert_dim: int = 256           # fine-grained expert hidden width

    # --- Recurrence ---
    loop_index_dim: Optional[int] = None  # channels receiving loop-index embedding; None -> dim // 8
    use_lti: bool = True            # False -> naive residual injection h = transformer_out + e (ablation)

    # --- Load balancing (Ouroboros completes what reference impls leave as a stub) ---
    router_bias_update_rate: float = 1e-3  # aux-loss-free LB: per-step bias nudge magnitude

    # --- RoPE / norm / init / regularization ---
    rope_theta: float = 10000.0     # RoPE base (small-model default; 500000 for long context)
    norm_eps: float = 1e-6          # RMSNorm epsilon
    init_std: float = 0.02          # N(0, init_std) weight init
    dropout: float = 0.0            # 0.0 disables; 0.1 typical for pretraining
    max_output_tokens: int = 1024   # generation cap
```

**Derived-quantity invariants** (enforced implicitly by the dimensions, worth
asserting in tests):

- `n_heads % n_kv_heads == 0` (GQA grouping must be integral).
- `dim // n_heads` is **even** (RoPE pairs adjacent features).
- Fine-grained MoE rule of thumb: `expert_dim ≈ dim // (n_experts // n_experts_per_tok)`.
- Shared experts use a **larger** hidden width — `expert_dim * n_experts_per_tok`.

**Config presets (documented, not hardcoded into the dataclass):**

| Preset             | dim | n_heads | layers (P/R/C) | n_experts | Target            |
| ------------------ | --- | ------- | -------------- | --------- | ----------------- |
| Tiny test          | 64  | 4       | 1 / loop / 1   | 4         | Unit tests, CI    |
| **T4 training**    | 512 | 8       | 2 / loop / 2   | 8         | ~10–30M params    |
| Frontier (illustrative) | 2048+ | 16+ | deeper     | 64+       | Not for T4        |

> **Gotcha — embedding dominates at small scale.** With weight tying and small
> `dim`, the embedding table (`vocab_size × dim`) can dwarf the transformer. This
> is exactly why `vocab_size = 8192` is the default: it keeps the parameter
> budget on the *transformer*, where the architectural story lives, rather than
> on a giant lookup table. (Spec §5.12.)

**Where it fits.** Constructed once and threaded into every module. It is *the*
contract — it precedes Prelude, Recurrent, and Coda alike.

---

### (2) `RMSNorm` — `norm.py`

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...   # (..., dim) -> (..., dim)
```

**What it is.** Root-Mean-Square layer normalization (Zhang & Sennrich, 2019):
LayerNorm without the mean-subtraction and without a bias term.

**Why it exists.** It is the normalization used *everywhere* — pre-norm in every
TransformerBlock, and on `(h_loop + e)` before each recurrent step. In a looped
model the same RMSNorm is applied many times to
the evolving latent, so it must be cheap and numerically robust. Dropping the
mean-subtraction (vs LayerNorm) removes one reduction and one centering op with no
measured quality loss for decoder LMs.

**How it works.**

```
 RMSNorm(x) = x * rsqrt( mean(x², axis=-1) + eps ) * weight
```

- `weight` is a learned per-channel gain of shape `(dim,)`, initialized to `1`.
- No bias, no mean subtraction.
- The `mean(x²)` reduction (and the `rsqrt`) are computed in **fp32**, then cast
  back to `x.dtype`.

**Inputs / outputs.**

| Tensor | Shape       | Dtype                    |
| ------ | ----------- | ------------------------ |
| `x`    | `(..., dim)`| activation dtype         |
| out    | `(..., dim)`| same as `x`              |

**Gotcha.** Compute the RMS reduction in **fp32** then cast back. In pure `fp16`,
`x²` of mid-magnitude activations underflows or saturates the mantissa and the
norm becomes unstable; the fp32 reduction costs almost nothing and removes the
whole failure mode.

**Where it fits.** Used in every stage. Defined first because attention, MoE,
the recurrent norm, and the final pre-head norm all depend on it.

---

### (3) RoPE — `rope.py`

```python
def precompute_rope_freqs(dim: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
    # returns complex64 (max_len, dim//2)

def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x (B, T, H, head_dim) -> rotated, same shape/dtype; head_dim even
```

**What it is.** Rotary Position Embedding (Su et al., 2021 — *RoFormer*). Encodes
absolute token position as a position-dependent **rotation** of each
2-dimensional feature pair, so that the dot product between a query at position
`m` and a key at position `n` depends only on the relative offset `m − n`.

**Why it exists.** A looped transformer reuses the same attention weights at every
recurrence depth, and the same `freqs_cis` buffer is applied at every loop. RoPE
gives relative positional structure with **no learned position table**, no extra
parameters, and clean extrapolation behavior — all of which matter when the same
weights must work across variable loop counts and (during decode) across cached
positions selected by `start_pos`.

**How it works.** For head dimension `d` (even), define per-pair frequencies
`θ_k = theta^(-2k/d)` for `k = 0 … d/2 − 1`. The phasor for position `m` and pair
`k` is the complex number

```
 freqs_cis[m, k] = e^{ i · m · θ_k } = cos(m·θ_k) + i · sin(m·θ_k)
```

`precompute_rope_freqs` builds this `(max_len, d/2)` `complex64` table once.
`apply_rope` then:

1. views adjacent feature pairs of `x` as complex numbers
   (`(B, T, H, d) → (B, T, H, d/2)` complex),
2. multiplies element-wise by the per-position phasor (broadcast over `B` and `H`),
3. views the result back as real and flattens to `(B, T, H, d)`.

Because multiplying a complex number by `e^{iφ}` is a rotation, the operation is
an **isometry** — it preserves the L2 norm of every feature pair exactly. Position
`0` has phasor `1 + 0i`, i.e. the identity (no rotation).

**Inputs / outputs.**

| Tensor      | Shape                | Dtype       | Notes                                     |
| ----------- | -------------------- | ----------- | ----------------------------------------- |
| `freqs_cis` | `(max_len, d/2)`     | `complex64` | precomputed once; `d` = head dim          |
| `x`         | `(B, T, H, d)`       | act dtype   | Q or K, `d` even                          |
| out         | `(B, T, H, d)`       | `x.dtype`   | rotated, identical norm                   |

**Gotchas.**

- **`head_dim` must be even** — features are rotated in pairs.
- **The caller slices `freqs_cis`**, not `apply_rope`. `apply_rope` is positionless;
  the model slices `freqs_cis[start_pos : start_pos + T]` so cached decode tokens
  get the correct absolute positions. Forgetting this gives every decoded token a
  position-0 (identity) rotation and generation degrades (see §11, §12).
- **Compute in fp32, cast back.** The complex multiply is done on `x.float()` then
  cast to `x.dtype`, mirroring RMSNorm's fp32 safety.

**Where it fits.** Inside the attention layer, applied to Q and K. The single
buffer — sized to the head dim `dim // n_heads` — lives on the top-level
`Ouroboros` module and is passed down.

---

### (4) `GQAttention` — `attention.py`

```python
class GQAttention(nn.Module):
    def __init__(self, cfg: OuroborosConfig): ...
    def forward(self, x, freqs_cis, mask=None, kv_cache=None, cache_key="default"): ...
```

**What it is.** Grouped-Query Attention (Ainslie et al., 2023). Multi-head
attention with **fewer KV heads than query heads** — each KV head is shared across
`groups = n_heads // n_kv_heads` query heads.

**Why it exists.** It is the model's one and only attention mechanism, and —
crucially — it has a **FlashAttention-2 / SDPA-flash fast path** that handles GQA
natively. The KV cache shrinks by a factor of `groups` versus full MHA, which is
the dominant memory cost at decode time. For a looped model that re-runs attention
at every depth, a small KV cache and a fast kernel both matter.

**How it works.** With `head_dim = dim // n_heads`:

1. Project `q = wq(x)`, `k = wk(x)`, `v = wv(x)` (all `bias=False`). `wq` produces
   `n_heads · head_dim`; `wk`, `wv` produce `n_kv_heads · head_dim`. `wo` maps
   `n_heads · head_dim → dim`.
2. Reshape to heads, apply RoPE to Q and K.
3. **Cache after RoPE.** If `kv_cache` is given, concatenate the new (already
   rotated) `k, v` onto the cached tensors along the sequence dim, store the
   `.detach()`ed result back, and reuse — so retrieval never needs re-rotation.
4. Attention itself, via one of:
   - **Fast path (Ampere/Hopper):** if `flash_attn_func` is importable, call it on
     Q/K/V **in the input dtype** (it handles GQA natively, no KV-head expansion),
     `causal=(mask is not None)`. No bf16 round-trip — the T4 target is fp16-only,
     and the kernel takes fp16 directly; only fp32 inputs are cast (to fp16) and
     restored.
   - **Fast path (T4, realistic):** `F.scaled_dot_product_attention` under
     `torch.backends.cuda.sdp_kernel(...)` selecting the flash / mem-efficient
     backend. This is the robust Turing path (see the FA2 reality note below).
   - **Fallback (any device):** expand KV heads with `repeat_interleave(groups)`,
     compute scaled dot-product `softmax(QKᵀ / √head_dim + mask) · V`, with dropout.

**Inputs / outputs.**

| Tensor       | Shape                          | Notes                                     |
| ------------ | ------------------------------ | ----------------------------------------- |
| `x`          | `(B, T, dim)`                  | input                                     |
| `freqs_cis`  | `(T, head_dim/2)` `complex64`  | sliced by caller                          |
| `mask`       | `(1, 1, T, S)` or `None`       | additive; dtype == activation dtype       |
| `kv_cache`   | `dict` or `None`               | `{cache_key: {"k":…, "v":…}}`             |
| out          | `(B, T, dim)`                  | —                                         |

KV cache entry shapes: `k, v` are `(B, S, n_kv_heads, head_dim)`, concatenated
along `S` across decode steps and `.detach()`ed.

**Gotchas.**

- **Mask dtype must match activation dtype.** An `fp32` mask added to `fp16`/`bf16`
  logits upcasts the attention matrix to `fp32`; the subsequent matmul against a
  `bf16` `V` in the fallback path then mismatches dtypes (or silently upcasts and
  costs memory). Build the mask in `x.dtype` (§3, §11).
- **FA2-on-T4 reality (surface this honestly).** FlashAttention-2's prebuilt
  wheels target Ampere `sm80`+/Hopper. On Turing (T4, `sm75`) the `flash-attn`
  package is forward-only and frequently fails to build. The realistic, robust
  path is `F.scaled_dot_product_attention` with the flash/mem-efficient backend
  plus a manual fallback. Keep `flash_attn_func` as an *optional* fast path for a
  rented Ampere GPU — do not pretend a custom FA2 kernel runs natively on T4.
- **Cache *after* RoPE**, not before — otherwise every retrieval would have to
  re-rotate the entire cached history.

**Where it fits.** Inside every `TransformerBlock` — Prelude, Recurrent, and
Coda. The `flash_attn` import is guarded: `try: from flash_attn import
flash_attn_func; _HAS_FLASH_ATTN = True / except ImportError: _HAS_FLASH_ATTN =
False`.

---

### (5) Expert (SwiGLU FFN) — `moe.py`

```python
class Expert(nn.Module):
    def __init__(self, dim: int, expert_dim: int): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

**What it is.** A single gated feed-forward unit using the **SwiGLU** variant
(Shazeer, 2020).

**Why it exists.** It is the one FFN primitive reused everywhere: as each *routed
expert* inside `MoEFFN`, as each *shared expert*, **and** as the dense FFN in the
Prelude/Coda `TransformerBlock`s. One class, three roles — only the hidden width
differs. SwiGLU consistently outperforms ReLU/GELU MLPs at equal parameter budget
(see [`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md#7)).

**How it works.**

```
 Expert(x) = down( SiLU(gate(x)) ⊙ up(x) )
```

with three `bias=False` linears: `gate: dim → expert_dim`, `up: dim → expert_dim`,
`down: expert_dim → dim`. `⊙` is element-wise; `SiLU(z) = z · sigmoid(z)`.

**Inputs / outputs.** `x: (..., dim) → (..., dim)`, dtype preserved.

**Sizing gotcha — `dim * 4 // 3` in Prelude/Coda.** When used as the *dense* FFN,
`expert_dim = dim * 4 // 3`. This is deliberately **smaller** than the common
`8/3 · dim` SwiGLU sizing — a parameter-budget choice. SwiGLU has three weight
matrices instead of two, so matching a `4·dim` ReLU MLP's parameter count would
use `8/3 · dim`; Ouroboros trims further to `4/3 · dim` to keep the small model's
budget on the recurrent MoE block rather than the I/O FFNs. (Spec §5.8.)

**Where it fits.** Prelude/Coda (dense, `dim*4//3`) and inside the Recurrent
block's MoE (routed: `expert_dim`; shared: `expert_dim · n_experts_per_tok`).

---

### (6) `MoEFFN` — `moe.py`

```python
class MoEFFN(nn.Module):
    def __init__(self, cfg: OuroborosConfig): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    @torch.no_grad()
    def update_router_bias(self) -> None: ...   # aux-loss-free LB bias step (Ouroboros completion)
```

**What it is.** A fine-grained Mixture-of-Experts feed-forward layer (DeepSeekMoE,
Dai et al., 2024) with two expert classes: **routed** experts (top-K activated per
token) and **shared** experts (always active), plus DeepSeek-V3 **aux-loss-free**
load balancing.

**Why it exists.** It is the FFN inside the **Recurrent** block. The recurrent
weights are reused at every depth, so a single dense FFN would be a bottleneck for
*breadth* (it must cover all domains at every loop). MoE gives the loop a large
parameter pool while activating only top-K experts per token — sparse compute,
broad capacity. **Shared experts** absorb the common cross-domain patterns
(syntax, basic reasoning) so the routed experts don't redundantly relearn them.

**How it works.**

*Routing (aux-loss-free, DeepSeek-V3):*
```
 logits      = router(x)                         # (N, n_experts), unbiased, N = B·T
 scores      = softmax(logits)                   # gating WEIGHTS come from UNBIASED logits
 topk_idx    = topk(logits + router_bias, K)     # SELECTION uses biased logits
 topk_w      = scores.gather(topk_idx)
 topk_w      = topk_w / topk_w.sum(-1, keepdim)  # renormalize over the K chosen
```

The key trick: the **bias steers selection but never the gradient**, because the
gating weights come from `softmax(logits)` *without* the bias. So load can be
balanced without distorting the language-modeling loss (no auxiliary loss term,
no tug-of-war between LM loss and balance loss).

*Combination:*
```
 out = Σ_{i in topk}  topk_w[i] · routed_expert[i](x)       # routed, weighted
     + Σ_{s}          shared_expert[s](x)                    # shared, always on
```

- `router = Linear(dim, n_experts, bias=False)`.
- `router_bias` is a **non-gradient buffer** (`register_buffer`), shape `(n_experts,)`.
- Routed experts: `n_experts × Expert(dim, expert_dim)`.
- Shared experts: `n_shared_experts × Expert(dim, expert_dim · n_experts_per_tok)`
  (wider, as noted in component 5).

*`update_router_bias()` — the Ouroboros completion.* Naive reference
implementations register the bias buffer but **never update it**, leaving load
balancing inert. Ouroboros closes that gap:

1. During `forward`, track per-expert selection counts in an `expert_load` buffer.
2. `update_router_bias()` nudges the bias **down** for overloaded experts and
   **up** for underloaded ones:
   ```
   router_bias -= router_bias_update_rate · sign(load − mean_load)
   ```
   so overloaded experts (`load > mean`) get a negative nudge (picked less),
   underloaded experts get a positive nudge (picked more).
3. The **training loop calls it every step** (`@torch.no_grad()`).

This is a concrete engineering-maturity win — call it out explicitly. (Spec §5.4,
[`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md#8).)

**Inputs / outputs.** `x: (B, T, dim) → (B, T, dim)`.

**Gotchas.**

- **Renormalize the top-K gates** so they sum to 1 — otherwise the routed
  contribution's magnitude depends on how confident the router happened to be.
- **The naive dispatch is an `O(K · n_experts)` masked Python loop** — for each of
  the K slots and each expert, mask the tokens routed there and run that expert.
  Correct but slow. Note the **grouped/batched-gather** dispatch (sort tokens by
  expert, one batched matmul per expert) as a future optimization and a
  benchmarkable item. (Spec §5.5.)
- **`router_bias` must be a buffer, not a `Parameter`** — if it ever receives a
  gradient, the aux-loss-free property is broken.

**Where it fits.** Recurrent block **only** (`use_moe=True`). Prelude/Coda use the
dense `Expert`. `update_router_bias()` is driven by the training loop (Phase 6).

---

### (7) `TransformerBlock` — `block.py`

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg: OuroborosConfig, use_moe: bool = False): ...
    def forward(self, x, freqs_cis, mask=None, kv_cache=None, cache_key="default"): ...
```

**What it is.** A standard **pre-norm** transformer block with GQA attention and
a swappable FFN.

**Why it exists.** It is the single reusable layer primitive. The *same* class
serves the dense Prelude/Coda layers (`use_moe=False`) and the looped Recurrent
core (`use_moe=True`). One block definition, used everywhere, keeps the
architecture honest and the cache plumbing uniform.

**How it works.** Pre-norm residual structure:

```
 x = x + dropout( attn( RMSNorm(x), freqs_cis, mask, kv_cache, cache_key) )
 x = x + dropout( ffn(  RMSNorm(x) ) )
```

- `attn = GQAttention(cfg)`.
- `ffn  = MoEFFN(cfg) if use_moe else Expert(dim, dim * 4 // 3)`.
- Two independent `RMSNorm`s (one before attention, one before the FFN).
- `cache_key` and `kv_cache` are threaded straight to the attention layer.

**Inputs / outputs.** `x: (B, T, dim) → (B, T, dim)`; same `freqs_cis` / `mask` /
`kv_cache` / `cache_key` contract as the attention layer it wraps.

**Gotcha.** `use_moe=True` is used **only** inside `RecurrentBlock`. Prelude and
Coda always pass `use_moe=False`. The MoE's `update_router_bias` therefore only
exists on the recurrent block's FFN.

**Where it fits.** All three stages: `prelude_layers` instances (dense),
one instance inside `RecurrentBlock` (MoE), `coda_layers` instances (dense).

---

### (8) `loop_index_embedding` — `recurrence.py`

```python
def loop_index_embedding(h, loop_t: int, loop_dim: int, theta: float = 10000.0) -> torch.Tensor:
    # (B,T,dim) -> (B,T,dim), sinusoidal signal added to first loop_dim channels
```

**What it is.** A sinusoidal embedding of the **recurrence depth** `t`, added to
the leading `loop_dim` channels of the hidden state — RoPE's idea applied over
loop index instead of token position.

**Why it exists.** The recurrent block reuses the **same weights at every loop**.
Without a depth signal, those weights must perform early-stage pattern-matching
and late-stage refinement *with no way to tell which loop they're on*. Injecting
the loop index lets the shared parameters implement functionally distinct
operations per iteration — the cheapest possible per-depth conditioning (zero
learned parameters).

**How it works.** For the first `loop_dim` channels (`loop_dim` even), build
frequencies `θ_k = theta^(-2k/loop_dim)`, compute `angles = loop_t · θ_k`, and form
`[sin(angles), cos(angles)]`. Add this as a **bias** to the first `loop_dim`
channels of `h`; the remaining `dim − loop_dim` channels pass through unchanged.

```
 h[..., :loop_dim] += [sin(loop_t · θ), cos(loop_t · θ)]
 h[..., loop_dim:]  unchanged
```

**Inputs / outputs.** `h: (B, T, dim) → (B, T, dim)`, dtype preserved; `loop_t`
and `loop_dim` are Python ints.

**Gotchas.**

- **Only the first `loop_dim` channels change** — the rest are untouched, by
  design. This keeps the bulk of the residual stream free for content.
- `loop_dim` must be even (it indexes sin/cos pairs). It defaults to `dim // 8` via
  `cfg.loop_index_dim` (see component 10).
- It is a **sinusoidal** embedding, chosen over a learned per-loop table on purpose
  — sinusoids extrapolate to loop counts beyond training (depth extrapolation; see
  [`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md#6)).

**Where it fits.** First operation inside the recurrent loop body, applied to `h`
before it is combined with `e` and normalized.

---

### (9) `LTIInjection` — `recurrence.py`

```python
class LTIInjection(nn.Module):
    def __init__(self, dim: int): ...
    def get_A(self) -> torch.Tensor: ...                       # (dim,), values in (0,1)
    def forward(self, h, e, transformer_out) -> torch.Tensor:  # h_{t+1}
```

**What it is.** The stable input-injection update for the recurrent loop (Parcae,
Prairie et al., 2026): a learned **diagonal linear time-invariant** state update
whose spectral radius is guaranteed `< 1` *by construction*. This is the stability
core of the whole project (resume bullet 2).

**Why it exists.** A naive recurrent update `h_{t+1} = f(h_t, e)` can let the
hidden state's effective state matrix develop spectral radius `≥ 1`, causing `h`
to **explode across loop iterations** and the training to diverge — especially at
high learning rates. The usual workarounds (gradient clipping, normalizing the
hidden state every loop) are band-aids that mask the instability rather than
removing it. LTI injection removes it structurally: the homogeneous part is a
provable contraction, so there is *no* unstable mode to clip.

**How it works.** Diagonal state matrix `A` derived via a **zero-order-hold (ZOH)**
discretization of a continuous system with a guaranteed-negative continuous
eigenvalue:

```
 A_continuous = − exp(log_A)            # always negative (each channel)
 A_discrete   = exp( Δt · A_continuous )  ∈ (0, 1)     where Δt = exp(log_dt)
```

Computed entirely in **log space** to avoid `0 · ∞ = NaN`:

```
 get_A()  =  exp( − exp( (log_dt + log_A).clamp(-20, 20) ) )
```

The update rule:

```
 forward(h, e, transformer_out)  =  A ⊙ h  +  B ⊙ e  +  transformer_out
```

- Parameters: `log_A (dim,)`, `log_dt (1,)`, `B (dim,)` (init ~`0.1`).
- Because every entry of `A_discrete` lies strictly in `(0, 1)` for **any** values
  of `log_A`/`log_dt`, the spectral radius `ρ(A) = max(get_A()) < 1` *always* — no
  matter how aggressive the gradient step. That guarantee is the whole point.

**Inputs / outputs.**

| Tensor            | Shape          | Notes                                  |
| ----------------- | -------------- | -------------------------------------- |
| `h`               | `(B, T, dim)`  | current hidden state                   |
| `e`               | `(B, T, dim)`  | frozen encoded input (Prelude output)  |
| `transformer_out` | `(B, T, dim)`  | this depth's refinement                |
| `get_A()`         | `(dim,)`       | diagonal of `A`, all in `(0, 1)`       |
| out (`h_{t+1}`)   | `(B, T, dim)`  | next hidden state                      |

**Gotchas.**

- **The `(-20, 20)` clamp is essential.** It keeps the inner `exp` finite in fp32
  under aggressive gradient steps; without it, `log_dt + log_A → +∞` overflows and
  the outer `exp` produces `NaN` via `0 · ∞`. (Spec §5.7.)
- **Compute in log space**, never materialize `A_continuous` or `Δt` separately —
  that is where the `0 · ∞` lurks.
- **`ρ(A) = max(get_A())`** because `A` is diagonal. This is the cheap, continuous
  stability scalar — log it every training step to W&B (spec §5.10). The
  centerpiece stability experiment (LTI vs no-LTI, toggled by `cfg.use_lti`) plots
  `ρ(A)` and loss together
  (see [`EXPERIMENTS.md`](./EXPERIMENTS.md), experiment 1).

**Where it fits.** The recurrence's update step:
`h = LTIInjection(h, e, transformer_out)`, run once per loop iteration after the
transformer output is available.

---

### (10) `RecurrentBlock` — `recurrence.py`

```python
class RecurrentBlock(nn.Module):
    def __init__(self, cfg: OuroborosConfig): ...
    def forward(self, h, e, freqs_cis, mask=None, n_loops=None, kv_cache=None): ...
```

**What it is.** The looped core of Ouroboros — **one** `TransformerBlock` (with
MoE) run for a fixed `n_loops` iterations, wrapped in loop-index conditioning and
LTI injection. This is where the entire architecture's identity lives.

**Why it exists.** It implements "same weights, more loops → deeper reasoning, no
parameter growth." The recurrence machinery (components 8–9) is orchestrated here
into the canonical loop body from §1.

**How it works — owned submodules.**

```python
self.block     = TransformerBlock(cfg, use_moe=True)
self.injection = LTIInjection(cfg.dim)
self.norm      = RMSNorm(cfg.dim)
self.loop_dim  = cfg.loop_index_dim or cfg.dim // 8
self.use_lti   = cfg.use_lti
```

**Loop body** (per iteration `t = 0 … n_loops−1`, fixed depth, no early exit):

```
 h_loop   = loop_index_embedding(h, t, loop_dim)              # depth signal
 combined = norm(h_loop + e)                                   # re-read frozen input
 trans    = block(combined, freqs_cis, mask, kv_cache,
                  cache_key=f"recurrent_loop_{t}")             # GQA + MoE
 h        = injection(h, e, trans)                             # h = A·h + B·e + trans
```

With `use_lti=False` the injection step is replaced by a naive residual injection
`h = trans + e` — the no-LTI arm of the stability experiment (see
[`EXPERIMENTS.md`](./EXPERIMENTS.md), experiment 1). Everything else in the loop
body is identical, so any stability difference is attributable to the LTI update
alone.

After `n_loops` iterations the final hidden state `h` is returned. The loop runs a
**fixed** number of iterations for every position — there is no adaptive halting,
which keeps the core loop simple and the KV cache trivially correct (see gotcha
below). See [`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md#5) for why a fixed loop
count is preferred over adaptive halting at this scope.

**Inputs / outputs.**

| Tensor      | Shape                              | Notes                                  |
| ----------- | ---------------------------------- | -------------------------------------- |
| `h`         | `(B, T, dim)`                      | initial state `h_0 = e` (Prelude output) |
| `e`         | `(B, T, dim)`                      | frozen encoded input                   |
| `freqs_cis` | `(T, head_dim/2)` `complex64`      | sliced upstream by `start_pos`         |
| `mask`      | `(1, 1, T, S)` or `None`           | additive, activation dtype             |
| `n_loops`   | `int` or `None` (→ `max_loop_iters`)| recurrent depth                       |
| `kv_cache`  | `dict` or `None`                   | distinct key per loop depth            |
| out         | `(B, T, dim)`                      | final hidden state after `n_loops`     |

**Gotchas.**

- **Distinct cache key per depth** — `f"recurrent_loop_{t}"` — so caches across
  loop depths never collide.
- **The fixed-depth loop keeps the KV cache trivially correct.** Because every
  forward pass runs all `n_loops` depths, a cache is populated at every
  `recurrent_loop_{t}` key, so later decode steps always find the keys they need.
  Per-sequence early exit — used only by inference-time continuous depth-wise
  batching — is handled separately in component 13, where its cache implications
  are addressed.
- **`e` is re-read every loop** (inside `norm(h_loop + e)` *and* in the LTI `B·e`
  term) — it is the anti-drift anchor that keeps the prompt alive across arbitrary
  depth.

**Where it fits.** *The* Recurrent stage. Constructed once on the top-level model;
receives `h = e = (Prelude output)` and returns the latent fed to the Coda.

---

### (11) `Ouroboros` — `model.py`

```python
class Ouroboros(nn.Module):
    def __init__(self, cfg: OuroborosConfig): ...
    def _init_weights(self) -> None: ...
    @staticmethod
    def _causal_mask(seq_len, device, dtype) -> torch.Tensor: ...   # (1,1,S,S) additive, -inf above diag
    def forward(self, input_ids, n_loops=None, kv_cache=None, start_pos=0) -> torch.Tensor: ...
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, n_loops=8, temperature=1.0, top_k=50): ...
    @torch.no_grad()
    def generate_depthwise_batched(self, input_ids, max_new_tokens=64, max_loops=None,
                                   convergence_tol=1e-3, temperature=1.0, top_k=50): ...  # Phase 7, see (13)
```

**What it is.** The full top-level model: `Embedding → Prelude → RecurrentBlock →
Coda → RMSNorm → tied LM head`, plus generation entry points.

**Why it exists.** It wires the 10 components above into the end-to-end forward
pass of §1, owns the RoPE buffer, the embedding/head weight tying, weight
init, and the causal mask — i.e. everything that is global to the architecture
rather than local to a layer.

**How it works — construction.**

```python
self.embed      = nn.Embedding(vocab_size, dim)
self.freqs_cis  = precompute_rope_freqs(dim // n_heads, max_seq_len, rope_theta)  # head-dim sized
self.prelude    = ModuleList([TransformerBlock(cfg, use_moe=False) for _ in range(prelude_layers)])
self.recurrent  = RecurrentBlock(cfg)
self.coda       = ModuleList([TransformerBlock(cfg, use_moe=False) for _ in range(coda_layers)])
self.norm       = RMSNorm(dim)
self.head       = nn.Linear(dim, vocab_size, bias=False)
self.head.weight = self.embed.weight    # weight tying
self._init_weights()                    # N(0, init_std) on Linear & Embedding
```

**How it works — forward.**

```
 x         = embed(input_ids)                                  # (B, T, dim)
 freqs_cis = freqs_cis[start_pos : start_pos + T]              # absolute positions
 mask      = _causal_mask(T, device, x.dtype) if T > 1 else None   # decode (T=1) needs no mask
 for i, layer in enumerate(prelude):  x = layer(x, freqs_cis, mask, kv_cache, f"prelude_{i}")
 e = x                                                          # freeze encoded input
 x = recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)        # h_0 = e
 for i, layer in enumerate(coda):     x = layer(x, freqs_cis, mask, kv_cache, f"coda_{i}")
 return head(norm(x))                                          # (B, T, vocab_size)
```

`_causal_mask(seq_len, device, dtype)` returns a `(1, 1, S, S)` additive mask: `0`
on/below the diagonal, `-inf` above (`torch.triu(full(-inf), diagonal=1)`),
**built in the activation dtype**.

**Inputs / outputs.**

| Tensor      | Shape                 | Dtype  | Notes                                         |
| ----------- | --------------------- | ------ | --------------------------------------------- |
| `input_ids` | `(B, T)`              | `long` | token indices                                 |
| `n_loops`   | `int` / `None`        | —      | recurrent depth (→ `max_loop_iters`)          |
| `kv_cache`  | `dict` / `None`       | —      | mutated in place                              |
| `start_pos` | `int`                 | —      | first absolute position (RoPE slicing)        |
| logits      | `(B, T, vocab_size)`  | act    | —                                             |

**Gotchas.**

1. **`start_pos` selects the RoPE slice during decode.** Without it, decoded
   tokens get position-0 rotations and generation degrades. Prefill uses
   `start_pos=0`; each subsequent decode step passes only the last token with
   `start_pos = prompt_len + step − 1`. (Spec §5.3.)
2. **Mask only when `T > 1`.** Single-token decode (`T=1`) needs no causal mask;
   building one would be wrong shape-wise against the cached `S`. (See §12.)
3. **Weight init ignores residual-depth scaling (improvement note).** Naive
   `N(0, init_std)` on all weights does not account for the variance a *looped*
   model accumulates across loops. Document GPT-2-style `1/√(2·n_eff)` scaling on
   residual output projections (and optionally QK-norm / logit z-loss) as a
   stability improvement to ablate (spec §5.9, §11 init note).

**Where it fits.** It *is* the model — Prelude, Recurrent, and Coda all hang off
it, and it owns everything global.

---

### (12) KV cache & autoregressive generation — `model.py`

*(Covered by `forward` and `generate` above; this section makes the caching
contract explicit.)*

**What it is.** The plain-`dict` KV cache and the greedy/top-K sampling loop that
drives incremental decoding.

**Why it exists.** Re-encoding the entire prefix at every decode step is `O(S²)`
per token. The cache makes decode `O(S)` per step by storing each layer/loop's
keys and values under a deterministic string key and concatenating the one new
token's contribution each step.

**How it works.**

- The cache is a plain `dict` **mutated in place**; pass `{}` and reuse it across
  decode steps. Deterministic keys (`prelude_{i}`, `recurrent_loop_{t}`,
  `coda_{i}`) guarantee caches never collide across layers or loop depths.
- Each entry holds `{"k", "v"}` — grouped KV heads, stored **already rotated**
  (component 4), so retrieval never re-rotates.
- `generate`:
  - **Step 0** processes the *full prompt* (`start_pos=0`, builds the causal mask).
  - **Each later step** passes only the last token (`input_ids[:, -1:]`, `T=1`)
    with `start_pos = prompt_len + step − 1` and `mask=None`; all prior keys come
    from the cache.
  - Sampling: `logits[:, -1] / temperature`, optional top-K truncation, `softmax`,
    `multinomial`.

**Inputs / outputs.** `generate(input_ids (B, T)) → (B, T + max_new_tokens)`
`long`.

**Gotchas.**

- **`start_pos` is load-bearing** (see §11 gotcha 1).
- **A correctness invariant worth a test:** cached single-token decode logits must
  match a full-context forward pass over the same sequence (within numerical
  tolerance). This is the canonical KV-cache regression test (Phase 5 / Phase 6).
- **The fixed-depth recurrent loop keeps caching simple** (§10) — every
  `recurrent_loop_{t}` key is populated on every forward, so the standard
  `generate` path has no cache/recurrence edge case. Per-sequence early exit is
  introduced only by continuous depth-wise batching, which handles its own cache
  implications (§13).

**Where it fits.** Spans all three stages (every layer/loop writes to the same
cache dict). Lives on the model; exercised by `generate` and
`generate_depthwise_batched`.

---

### (13) Continuous depth-wise batching — `model.py`

```python
@torch.no_grad()
def generate_depthwise_batched(self, input_ids, max_new_tokens=64, max_loops=None,
                               convergence_tol=1e-3, temperature=1.0, top_k=50): ...
```

**What it is.** **THE** inference differentiator (resume bullet 3). Because all
sequences share the same recurrent weights, different sequences in
one batch can **exit the loop at different depths** via a non-learned, *convergence
based* early-exit: a sequence stops looping once its hidden state stops changing
(`‖h_{t+1} − h_t‖_F / (‖h_t‖_F + 1e-8) < convergence_tol`, Frobenius norm over
that sequence's `(T, dim)` hidden block). The **relative** form is scale-free — a
raw L2 norm would grow with `√(T · dim)` — so the `1e-3` default is a meaningful
"0.1% change per iteration" criterion at every model size. Easy sequences converge
fast and exit early, hard ones loop deeper — all in a single batch — instead of
every sequence paying the maximum depth. (No learned halting head; this is the
cut-scope replacement for adaptive halting — see
[`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md#5).)

**Why it exists.** With a uniform fixed loop count, the whole batch runs to
`max_loops` even if most sequences converged early — wasted compute proportional to
the gap between mean and max convergence depth. Continuous depth-wise batching
tightens that to the *active* depth, tying throughput directly to the
convergence-depth distribution. It also carries the inference-optimization claim
on its own: KV-cached decode (component 12) and the SDPA flash/mem-efficient fast
path (component 4) make each decode step cheap; depth-wise batching cuts how many
loop depths each step pays for. (Quantization is deliberately out of scope: at the
10–30M-param scale the fp16 model is ~40 MB, so INT8 has no honest memory/latency
story.)

**How it works — and the central challenge.** The hard part is the
**KV-cache ⊗ early-exit interaction**. A sequence that converges and exits at depth
`d` leaves cache keys `recurrent_loop_{d..n}` **unpopulated**; a later decode step
for that sequence that needs to loop deeper would read missing keys. Three
solutions to weigh (document the chosen one with measured tradeoffs):

1. **Run-to-max-active + mask.** Each step, run all sequences to the **maximum
   active depth currently in the batch**, masking converged ones. Simple, correct,
   keeps the cache dense; gains scale with the spread of convergence depths.
2. **Ragged / compacted per-sequence cache.** Track per-sequence depth and
   maintain a ragged or compacted cache so each sequence stores only the depths it
   actually ran. Most memory-efficient, most bookkeeping.
3. **Bucket by predicted depth.** Sort/bucket sequences so a batch shares a depth,
   removing the ragged-cache problem at the cost of a scheduling pass.

**Chosen approach: (1) run-to-max-active + mask** — the only option whose cache
stays dense and rectangular, so the existing `recurrent_loop_{t}` keying and the
standard KV-cache path are reused verbatim. Options 2–3 are future optimizations.

**Inputs / outputs.** `generate_depthwise_batched(input_ids (B, T)) →
(B, T + max_new_tokens)` `long`. `max_loops` caps the deepest any sequence may run;
`convergence_tol` sets the per-sequence relative-change exit threshold.

**Throughput expectation.** The headline multiplier is **depth-wise batched decode
vs naive fixed-depth decode** (same model, same prompts), with KV-cache on/off as
the supporting ablation. Literature suggests ~**2–3×** depending on the
convergence-depth distribution; the *measured* number on T4 fills in resume bullet
3. Tie the result back to the observed convergence-depth histogram (see
[`EXPERIMENTS.md`](./EXPERIMENTS.md), experiment 3).

**Where it fits.** An inference-time orchestration over the Recurrent stage,
implemented on the model (Phase 7). It does not change the math of any component —
it changes *which depths run for which sequences*, subject to the cache-population
invariant above.

---

## 5. Key design properties (summary table)

| Property                          | Component(s)                  | Mechanism                                                            | Why it matters                                                                  |
| --------------------------------- | ----------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Stable looped training**        | LTIInjection (9)              | Diagonal `A` via ZOH, `ρ(A) < 1` by construction; log-space clamp `(-20,20)` | Converges at high LR with no clipping/normalization band-aids (resume bullet 2) |
| **Depth = compute, not params**   | RecurrentBlock (10)           | One shared block looped a fixed `n_loops` times                     | Deeper reasoning with zero parameter growth                                     |
| **Depth extrapolation**           | RecurrentBlock (10), loop-index (8) | Sinusoidal depth signal, well-defined at any loop count       | Train at `n_loops=8`, run deeper at inference                                   |
| **Breadth in the loop**           | MoEFFN (6), Expert (5)        | Fine-grained routed + always-on shared experts                      | Large capacity, sparse top-K compute, no per-domain bottleneck                  |
| **Real load balancing**           | MoEFFN.update_router_bias (6) | Aux-loss-free bias nudged `-= rate·sign(load−mean)` each step        | Balanced experts without an auxiliary loss; closes the reference-impl stub gap  |
| **Cheap decode memory**           | GQAttention (4), KV cache (12)| Grouped KV heads — cache shrinks `n_heads // n_kv_heads`× vs MHA    | Longer context & larger batch on a 16 GB T4                                      |
| **Anti-drift recurrence**         | LTIInjection (9), RecurrentBlock (10) | `B·e` + `norm(h_loop+e)` re-inject frozen input every loop  | Prompt stays alive across arbitrary depth                                       |
| **No learned position table**     | RoPE (3)                      | Complex-phasor rotation, norm-preserving isometry                   | Relative positions, clean extrapolation, zero position params                   |
| **Inference throughput**          | GQAttention (4), KV cache (12), depth-wise batching (13) | KV-cached decode + SDPA fast path + per-sequence convergence early exit | ~2–3× over naive fixed-depth decode on T4 (resume bullet 3)                     |
| **Correctness under caching**     | RecurrentBlock (10), gen (12/13) | Fixed-depth loop populates every cache key; deterministic keys   | Cached decode matches full-context forward                                      |
| **fp16 numerical safety**         | RMSNorm (2), RoPE (3), LTI (9) | fp32 reductions; dtype-matched additive mask                       | Stable training/inference in fp16 on Turing                                     |
| **Small-scale parameter economy** | OuroborosConfig (1), Expert (5) | `vocab_size=8192`, tied embeddings, `dim*4//3` dense FFN            | Keeps the budget on the transformer, not a giant lookup table (T4-realistic)    |

---

*Cross-references:* roadmap & phase gates → [`ROADMAP.md`](./ROADMAP.md);
rationale & alternatives per decision → [`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md);
experiment designs & result tables → [`EXPERIMENTS.md`](./EXPERIMENTS.md);
papers per component → [`READING_LIST.md`](./READING_LIST.md).
