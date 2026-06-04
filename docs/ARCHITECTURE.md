# Ouroboros — Architecture Reference

> **Ouroboros** is a recurrent-depth (looped) transformer language model built from
> scratch in PyTorch. It follows a **Prelude → Recurrent → Coda** design with
> fine-grained Mixture-of-Experts (routed + shared experts), switchable
> **MLA / GQA** attention, **LTI-constrained** stable injection (spectral radius
> `< 1` by construction), **ACT** adaptive halting, depth-wise **LoRA** adapters,
> and — beyond the reference literature — **INT8** post-training quantization with
> **continuous depth-wise batching** for inference.

This document is the detailed architecture reference. It opens with the canonical
forward-pass data-flow diagram, then walks all **17 components** in dependency
order. For each component you get: *what it is*, *why it exists* (the problem it
solves in a looped transformer), *how it works* (math/equations, enough to
implement without any reference repo), *exact tensor shapes & dtypes*, *key
implementation gotchas*, and *where it sits* in the Prelude → Recurrent → Coda
pipeline. A **key design properties** summary table closes the document.

Ouroboros is an independent implementation inspired by the published
recurrent-depth transformer literature — primarily *Parcae* (Prairie et al.,
2026), *DeepSeek-V2* (2024), *DeepSeekMoE* (Dai et al., 2024), *DeepSeek-V3*
(2024), *Universal Transformers* (Dehghani et al., 2018), *Adaptive Computation
Time* (Graves, 2016), and *Relaxed Recursive Transformers* (Bae et al., 2024).
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
4. [The 17 components](#4-the-17-components)
   - [(1) OuroborosConfig](#1-ouroborosconfig--configpy)
   - [(2) RMSNorm](#2-rmsnorm--normpy)
   - [(3) RoPE](#3-rope--ropepy)
   - [(4) GQAttention](#4-gqattention--attentionpy)
   - [(5) MLAttention](#5-mlattention--attentionpy)
   - [(6) Expert (SwiGLU FFN)](#6-expert-swiglu-ffn--moepy)
   - [(7) MoEFFN](#7-moeffn--moepy)
   - [(8) TransformerBlock](#8-transformerblock--blockpy)
   - [(9) loop_index_embedding](#9-loop_index_embedding--recurrencepy)
   - [(10) LoRAAdapter](#10-loraadapter--recurrencepy)
   - [(11) LTIInjection](#11-ltiinjection--recurrencepy)
   - [(12) ACTHalting](#12-acthalting--recurrencepy)
   - [(13) RecurrentBlock](#13-recurrentblock--recurrencepy)
   - [(14) Ouroboros](#14-ouroboros--modelpy)
   - [(15) KV cache & autoregressive generation](#15-kv-cache--autoregressive-generation--modelpy)
   - [(16) INT8 quantization](#16-int8-quantization--quantizepy)
   - [(17) Continuous depth-wise batching](#17-continuous-depth-wise-batching--modelpy)
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
informative stability signal in the whole system — log it every step (see §11).

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
LTI computations (§3, §11). The **additive attention mask must match the
activation dtype** — an `fp32` mask on `fp16`/`bf16` logits silently upcasts the
attention matrix and breaks the downstream matmul against `V` in the fallback
path (§4 gotcha). `freqs_cis` buffers are `complex64`.

---

## 4. The 17 components

Presented in dependency order: each component only depends on those above it.
Public signatures are quoted **verbatim** from the spec — the `.py` stubs must
match them exactly.

---

### (1) `OuroborosConfig` — `config.py`

**What it is.** The single frozen-by-convention hyperparameter contract. A
`@dataclass` whose field declarations *are* the API (the "header"). Every other
component reads its dimensions from this object — there are no magic numbers
elsewhere.

**Why it exists.** A looped MoE model with two attention back-ends has a large,
interdependent hyperparameter surface (head dims must divide, expert width is
derived from routing, the loop-index dim is a fraction of `dim`, etc.). Centralizing
these into one typed object makes the invariants checkable in one place and keeps
the tiny-test / T4-training / frontier configs interchangeable.

**How it works.** Plain dataclass with defaults targeting a small T4-friendly
model. No methods, no logic. The canonical fields:

```python
@dataclass
class OuroborosConfig:
    # --- Core ---
    vocab_size: int = 8192          # small BPE vocab keeps the embedding table modest at small dim
    dim: int = 512                  # residual-stream width
    n_heads: int = 8                # query heads
    n_kv_heads: int = 2             # GQA key/value heads (n_heads % n_kv_heads == 0); ignored by MLA
    max_seq_len: int = 1024         # RoPE precomputation length
    max_loop_iters: int = 8         # default recurrent depth T at inference
    prelude_layers: int = 2         # standard blocks before the loop
    coda_layers: int = 2            # standard blocks after the loop

    # --- Attention ("gqa" | "mla") ---
    attn_type: str = "gqa"          # default GQA: simpler + has the FA2 fast path
    kv_lora_rank: int = 128         # [MLA] compressed KV latent cached
    q_lora_rank: int = 256          # [MLA] compressed Q latent
    qk_rope_head_dim: int = 32      # [MLA] per-head dims that receive RoPE
    qk_nope_head_dim: int = 64      # [MLA] per-head dims without RoPE
    v_head_dim: int = 64            # [MLA] per-head value dim

    # --- MoE FFN (used only inside the Recurrent Block) ---
    n_experts: int = 8              # routed experts
    n_shared_experts: int = 1       # always-active shared experts
    n_experts_per_tok: int = 2      # top-K routed per token
    expert_dim: int = 256           # fine-grained expert hidden width

    # --- Recurrence / stability / halting ---
    act_threshold: float = 0.99     # ACT cumulative-probability halting threshold
    lora_rank: int = 8              # depth-wise LoRA bottleneck rank
    loop_index_dim: Optional[int] = None  # channels receiving loop-index embedding; None -> dim // 8

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
- `qk_rope_head_dim` is **even** (RoPE again).
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
TransformerBlock, on the Q and KV latents inside MLA, and on `(h_loop + e)` before
each recurrent step. In a looped model the same RMSNorm is applied many times to
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
  position-0 (identity) rotation and generation degrades (see §3, §14, §15).
- **Compute in fp32, cast back.** The complex multiply is done on `x.float()` then
  cast to `x.dtype`, mirroring RMSNorm's fp32 safety.
- **Two differently-sized buffers exist** (one for GQA, one for MLA) — see §14,
  component 14. GQA rotates the full `dim // n_heads`; MLA rotates only
  `qk_rope_head_dim`.

**Where it fits.** Inside both attention back-ends, applied to Q and K. The
buffers live on the top-level `Ouroboros` module and are passed down.

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

**Why it exists.** It is the default attention back-end (`attn_type="gqa"`):
simpler than MLA, and — crucially — it has a **FlashAttention-2 / SDPA-flash fast
path** that handles GQA natively. The KV cache shrinks by a factor of `groups`
versus full MHA, which is the dominant memory cost at decode time. For a looped
model that re-runs attention at every depth, a small KV cache and a fast kernel
both matter.

**How it works.** With `head_dim = dim // n_heads`:

1. Project `q = wq(x)`, `k = wk(x)`, `v = wv(x)` (all `bias=False`). `wq` produces
   `n_heads · head_dim`; `wk`, `wv` produce `n_kv_heads · head_dim`. `wo` maps
   `n_heads · head_dim → dim`.
2. Reshape to heads, apply RoPE to Q and K.
3. **Cache after RoPE.** If `kv_cache` is given, concatenate the new (already
   rotated) `k, v` onto the cached tensors along the sequence dim, store the
   `.detach()`ed result back, and reuse — so retrieval never needs re-rotation.
4. Attention itself, via one of:
   - **Fast path (Ampere/Hopper):** if `flash_attn_func` is importable, cast Q/K/V
     to `bf16`, call it (it handles GQA natively, no KV-head expansion),
     `causal=(mask is not None)`, restore the original dtype.
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
  costs memory). Build the mask in `x.dtype` (§3, §14).
- **FA2-on-T4 reality (surface this honestly).** FlashAttention-2's prebuilt
  wheels target Ampere `sm80`+/Hopper. On Turing (T4, `sm75`) the `flash-attn`
  package is forward-only and frequently fails to build. The realistic, robust
  path is `F.scaled_dot_product_attention` with the flash/mem-efficient backend
  plus a manual fallback. Keep `flash_attn_func` as an *optional* fast path for a
  rented Ampere GPU — do not pretend a custom FA2 kernel runs natively on T4.
- **Cache *after* RoPE**, not before — otherwise every retrieval would have to
  re-rotate the entire cached history.

**Where it fits.** Selected inside every `TransformerBlock` when
`attn_type == "gqa"` — i.e. in Prelude, Recurrent, and Coda. The `flash_attn`
import is guarded: `try: from flash_attn import flash_attn_func; _HAS_FLASH_ATTN
= True / except ImportError: _HAS_FLASH_ATTN = False`.

---

### (5) `MLAttention` — `attention.py`

```python
class MLAttention(nn.Module):
    def __init__(self, cfg: OuroborosConfig): ...
    def forward(self, x, freqs_cis, mask=None, kv_cache=None, cache_key="default"): ...
```

**What it is.** Multi-Latent Attention (DeepSeek-V2, 2024). Instead of caching
full `K` and `V` per token, MLA caches a **low-rank latent** `c_kv` and a small
shared RoPE key, reconstructing `K_nope` and `V` on the fly each step.

**Why it exists.** It is the alternative attention back-end (`attn_type="mla"`).
The KV cache is the dominant decode-time memory cost; MLA compresses it
dramatically (caching `kv_lora_rank + qk_rope_head_dim` per token instead of
`n_kv_heads · head_dim · 2`). On a 16 GB T4 this directly extends the feasible
context length and batch size. Switchability (GQA ↔ MLA via one config flag) is
itself a portfolio talking point — see [`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md#4).

**How it works.** Per-head query dim is `q_head_dim = qk_nope_head_dim +
qk_rope_head_dim`.

*Q path (decoupled RoPE):*
```
 x → q_down (dim → q_lora_rank) → q_norm (RMSNorm)
   → q_up_nope (q_lora_rank → H·qk_nope_head_dim)   # no RoPE
   → q_up_rope (q_lora_rank → H·qk_rope_head_dim)   # RoPE applied
 q = concat(q_nope, q_rope)  per head               # (B, T, H, q_head_dim)
```

*KV path (compressed, decoupled RoPE):*
```
 x → kv_down (dim → kv_lora_rank + qk_rope_head_dim)
   splits into:
     c_kv       (B, T, kv_lora_rank)         ← CACHED (the latent)
     k_rope_raw (B, T, qk_rope_head_dim)     ← shared across heads
 k_rope = RoPE( expand_to_heads(k_rope_raw) )        # (B, T, H, qk_rope_head_dim) ← CACHED, already rotated
 # reconstructed every step (NOT cached):
 c_kv → kv_norm (RMSNorm) → kv_up (kv_lora_rank → H·(qk_nope_head_dim + v_head_dim))
        → split into k_nope (B, S, H, qk_nope_head_dim) and v (B, S, H, v_head_dim)
 k = concat(k_nope, k_rope)  per head                # (B, S, H, q_head_dim)
```

Then scaled dot-product attention `softmax(QKᵀ / √q_head_dim + mask) · V`, and the
output projection `wo: n_heads·v_head_dim → dim`.

**Inputs / outputs.**

| Tensor      | Shape                                | Notes                                      |
| ----------- | ------------------------------------ | ------------------------------------------ |
| `x`         | `(B, T, dim)`                        | input                                      |
| `freqs_cis` | `(T, qk_rope_head_dim/2)` `complex64`| **MLA-sized** RoPE buffer                  |
| `mask`      | `(1, 1, T, S)` or `None`             | additive; matches activation dtype         |
| `kv_cache`  | `dict` or `None`                     | see below                                  |
| out         | `(B, T, dim)`                        | —                                          |

KV cache entry:
`{cache_key: {"c_kv": (B, S, kv_lora_rank), "k_rope": (B, S, H, qk_rope_head_dim)}}`
— far smaller than GQA's full `K`/`V`.

**Gotchas.**

- **`n_kv_heads` is irrelevant to MLA** — it is a GQA-only knob. With
  `attn_type="mla"` the config's `n_kv_heads` is simply ignored.
- **MLA needs its own RoPE buffer** sized to `qk_rope_head_dim`, separate from the
  GQA buffer sized to `dim // n_heads`. The model precomputes both (§14, gotcha 2).
- **`k_rope` is shared across heads** (computed once from `k_rope_raw`, then
  expanded), and it is cached **already rotated** — like GQA, retrieval needs no
  re-rotation. `c_kv` is cached **unrotated** because `K_nope` carries no
  positional signal.
- **MLA has no flash fast path here.** Because `K`/`V` are reconstructed each step
  and the decoupled-RoPE layout differs from the standard one, MLA uses the manual
  SDPA path. (A fused MLA kernel is future work.)

**Where it fits.** Selected inside every `TransformerBlock` when
`attn_type == "mla"` — Prelude, Recurrent, and Coda. Compared head-to-head against
GQA in [`EXPERIMENTS.md`](./EXPERIMENTS.md) (experiment 3).

---

### (6) Expert (SwiGLU FFN) — `moe.py`

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
(see [`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md#8)).

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

### (7) `MoEFFN` — `moe.py`

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
  (wider, as noted in component 6).

*`update_router_bias()` — the Ouroboros completion.* Naive reference
implementations register the bias buffer but **never update it**, leaving load
balancing inert. Ouroboros closes that gap:

1. During `forward`, track per-expert selection counts in an `expert_load` buffer.
2. `update_router_bias()` nudges the bias **down** for overloaded experts and
   **up** for underloaded ones:
   ```
   router_bias += router_bias_update_rate · sign(mean_load − load)
   ```
   so overloaded experts (`load > mean`) get a negative nudge (picked less),
   underloaded experts get a positive nudge (picked more).
3. The **training loop calls it every step** (`@torch.no_grad()`).

This is a concrete engineering-maturity win — call it out explicitly. (Spec §5.4,
[`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md#9).)

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

### (8) `TransformerBlock` — `block.py`

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg: OuroborosConfig, use_moe: bool = False): ...
    def forward(self, x, freqs_cis, mask=None, kv_cache=None, cache_key="default"): ...
```

**What it is.** A standard **pre-norm** transformer block with a swappable
attention back-end and a swappable FFN.

**Why it exists.** It is the single reusable layer primitive. The *same* class
serves the dense Prelude/Coda layers (`use_moe=False`) and the looped Recurrent
core (`use_moe=True`), and it transparently picks GQA or MLA from `cfg.attn_type`.
One block definition, used everywhere, keeps the architecture honest and the cache
plumbing uniform.

**How it works.** Pre-norm residual structure:

```
 x = x + dropout( attn( RMSNorm(x), freqs_cis, mask, kv_cache, cache_key) )
 x = x + dropout( ffn(  RMSNorm(x) ) )
```

- `attn = MLAttention(cfg) if attn_type == "mla" else GQAttention(cfg)`.
- `ffn  = MoEFFN(cfg) if use_moe else Expert(dim, dim * 4 // 3)`.
- Two independent `RMSNorm`s (one before attention, one before the FFN).
- `cache_key` and `kv_cache` are threaded straight to the attention layer.

**Inputs / outputs.** `x: (B, T, dim) → (B, T, dim)`; same `freqs_cis` / `mask` /
`kv_cache` / `cache_key` contract as the attention layers it wraps.

**Gotcha.** `use_moe=True` is used **only** inside `RecurrentBlock`. Prelude and
Coda always pass `use_moe=False`. The MoE's `update_router_bias` therefore only
exists on the recurrent block's FFN.

**Where it fits.** All three stages: `prelude_layers` instances (dense),
one instance inside `RecurrentBlock` (MoE), `coda_layers` instances (dense).

---

### (9) `loop_index_embedding` — `recurrence.py`

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
  `cfg.loop_index_dim` (see component 13).
- It is a **sinusoidal** embedding, chosen over a learned per-loop table on purpose
  — sinusoids extrapolate to loop counts beyond training (depth extrapolation; see
  [`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md#7)).

**Where it fits.** First operation inside the recurrent loop body, applied to `h`
before it is combined with `e` and normalized.

---

### (10) `LoRAAdapter` — `recurrence.py`

```python
class LoRAAdapter(nn.Module):
    def __init__(self, dim: int, rank: int, max_loops: int): ...
    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor: ...
```

**What it is.** A depth-wise **Low-Rank Adaptation** delta (Relaxed Recursive
Transformers, Bae et al., 2024): a small per-loop correction added to the shared
block's output.

**Why it exists.** It bridges two extremes. *Pure weight-tying* (identical weights
every loop) is parameter-efficient but limits expressiveness. *Fully distinct
weights per loop* is expressive but throws away the parameter savings that make a
looped model worthwhile. The LoRA adapter sits in between: a shared low-rank
transform with a tiny **per-loop scale**, adding distinct behavior at each depth
for negligible parameter cost.

**How it works.**

```
 delta(x, t) = ( down(x) ⊙ scale[t] ) @ B
```

- `down: Linear(dim, rank)` — shared across all loops.
- `B: Parameter(rank, dim)` (init ~`0.02`) — shared across all loops.
- `scale: Embedding(max_loops, rank)` — **per-loop** element-wise gain on the
  rank-`r` bottleneck. This is the only depth-specific parameter.

**Inputs / outputs.** `x: (B, T, dim) → (B, T, dim)` (the delta), added to the
transformer-block output. `loop_t` is a Python int.

**Gotcha — depth extrapolation clamp.** At inference, `loop_t` can exceed
`max_loops − 1` (running deeper than training). **Clamp** the lookup index to the
last learned scale (`min(loop_t, max_loops − 1)`) rather than indexing the
embedding out of range. Without this, depth extrapolation crashes instead of
gracefully reusing the deepest learned correction. (Spec §5, component 10 gotcha.)

**Where it fits.** Inside the recurrent loop, applied to the `TransformerBlock`
output before the LTI injection: `trans = trans + LoRAAdapter(trans, t)`.
Constructed with `max_loops = cfg.max_loop_iters`.

---

### (11) `LTIInjection` — `recurrence.py`

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
  centerpiece stability experiment (LTI vs no-LTI) plots `ρ(A)` and loss together
  (see [`EXPERIMENTS.md`](./EXPERIMENTS.md), experiment 1).

**Where it fits.** The recurrence's update step:
`h = LTIInjection(h, e, transformer_out)`, run once per loop iteration after the
LoRA-adjusted transformer output is available.

---

### (12) `ACTHalting` — `recurrence.py`

```python
class ACTHalting(nn.Module):
    def __init__(self, dim: int): ...
    def forward(self, h: torch.Tensor) -> torch.Tensor: ...    # (B,T,dim) -> (B,T) in (0,1)
```

**What it is.** The Adaptive Computation Time halting head (Graves, 2016): a
learned per-position **halting probability** predicted at each loop iteration.

**Why it exists.** Not every token needs the same amount of refinement — easy
tokens converge in a couple of loops, hard ones need more. A fixed loop count
either under-computes hard tokens or wastes compute on easy ones. ACT lets each
position **halt independently**, all within the same batch, and is the basis for
the continuous depth-wise batching differentiator (component 17). It also gives
looped transformers their Turing-completeness story (Universal Transformers,
Dehghani et al., 2018).

**How it works.**

```
 p = sigmoid( Linear(dim, 1)(h) ).squeeze(-1)        # (B, T), per-position halt prob in (0, 1)
```

The *accumulation logic* (the ACT remainder trick, cumulative-probability halting)
lives in `RecurrentBlock` (component 13) — this module only produces the
per-position halting score from the current hidden state.

**Inputs / outputs.** `h: (B, T, dim) → (B, T)`, values in `(0, 1)`.

**Where it fits.** Called once per loop iteration on the updated `h`, feeding the
`RecurrentBlock`'s halting accumulator.

---

### (13) `RecurrentBlock` — `recurrence.py`

```python
class RecurrentBlock(nn.Module):
    def __init__(self, cfg: OuroborosConfig): ...
    def forward(self, h, e, freqs_cis, mask=None, n_loops=None, kv_cache=None): ...
```

**What it is.** The looped core of Ouroboros — **one** `TransformerBlock` (with
MoE) run for `n_loops` iterations, wrapped in loop-index conditioning, depth-wise
LoRA, LTI injection, and ACT halting. This is where the entire architecture's
identity lives.

**Why it exists.** It implements "same weights, more loops → deeper reasoning, no
parameter growth." All the recurrence machinery (components 9–12) is orchestrated
here into the canonical loop body from §1.

**How it works — owned submodules.**

```python
self.block     = TransformerBlock(cfg, use_moe=True)
self.injection = LTIInjection(cfg.dim)
self.act       = ACTHalting(cfg.dim)
self.lora      = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
self.norm      = RMSNorm(cfg.dim)
self.loop_dim  = cfg.loop_index_dim or cfg.dim // 8
```

**Loop body** (per iteration `t = 0 … n_loops−1`):

```
 h_loop   = loop_index_embedding(h, t, loop_dim)              # depth signal
 combined = norm(h_loop + e)                                   # re-read frozen input
 trans    = block(combined, freqs_cis, mask, kv_cache,
                  cache_key=f"recurrent_loop_{t}")             # MLA/GQA + MoE
 trans    = trans + lora(trans, t)                             # per-depth LoRA delta
 h        = injection(h, e, trans)                             # h = A·h + B·e + trans
 p        = act(h)                                             # (B, T) halt prob
 # --- ACT remainder accumulation ---
 still_running = ~halted
 remainder     = (1 − cumulative_p).clamp(min=0)
 weight        = where(cumulative_p + p >= act_threshold, remainder, p)
 weight        = weight * still_running          # halted positions contribute 0
 h_out        += weight.unsqueeze(-1) * h
 cumulative_p += p * still_running
 halted        = halted | (cumulative_p >= act_threshold)
 if halted.all() and kv_cache is None:           # ← see CRITICAL gotcha
     break
```

State maintained across the loop: `halted (B, T) bool`, `cumulative_p (B, T)`,
`h_out (B, T, dim)`. The returned value is the **ACT-weighted sum** of hidden
states — each position's `h` is accumulated with the weight assigned on the step
it converged, so every position contributes a total probability mass of ~1.

**The ACT remainder trick (why `still_running` gating matters).** Once a position's
`cumulative_p + p` crosses `act_threshold`, its weight becomes the *remaining*
mass `1 − cumulative_p` (so its total weight sums to ~1), and it must contribute
**exactly once** more, then nothing. Because `act_threshold < 1`, an un-gated
remainder would leak a non-zero weight every subsequent step. Multiplying by
`still_running` ensures a halted position contributes on its halting step and
zero thereafter.

**Inputs / outputs.**

| Tensor      | Shape                              | Notes                                  |
| ----------- | ---------------------------------- | -------------------------------------- |
| `h`         | `(B, T, dim)`                      | initial state (= Prelude output)       |
| `e`         | `(B, T, dim)`                      | frozen encoded input                   |
| `freqs_cis` | `(T, rope/2)` `complex64`          | GQA- or MLA-sized, selected upstream   |
| `mask`      | `(1, 1, T, S)` or `None`           | additive, activation dtype             |
| `n_loops`   | `int` or `None` (→ `max_loop_iters`)| recurrent depth                       |
| `kv_cache`  | `dict` or `None`                   | distinct key per loop depth            |
| out         | `(B, T, dim)`                      | ACT-weighted sum                       |

**CRITICAL gotcha — the headline subtlety (KV-cache ⊗ ACT conflict).** Early-exit
(`if halted.all(): break`) is valid **only when `kv_cache is None`**. With a KV
cache, **every loop depth must run on every forward pass**, because a later decode
step that loops to depth `d` will read keys from cache key
`f"recurrent_loop_{d}"` — and if an earlier step short-circuited before depth `d`,
that key is unpopulated and decode reads garbage (or crashes). So:

```python
if halted.all() and kv_cache is None:
    break
```

This same constraint is the crux of continuous depth-wise batching (component 17)
and the single most important correctness invariant in the system. (Spec §5.1.)

**Other gotchas.**

- **Distinct cache key per depth** — `f"recurrent_loop_{t}"` — so caches across
  loop depths never collide.
- **`e` is re-read every loop** (inside `norm(h_loop + e)` *and* in the LTI `B·e`
  term) — it is the anti-drift anchor that keeps the prompt alive across arbitrary
  depth.

**Where it fits.** *The* Recurrent stage. Constructed once on the top-level model;
receives `h = e = (Prelude output)` and returns the latent fed to the Coda.

---

### (14) `Ouroboros` — `model.py`

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
                                   temperature=1.0, top_k=50) -> torch.Tensor: ...   # Phase 7, see (17)
```

**What it is.** The full top-level model: `Embedding → Prelude → RecurrentBlock →
Coda → RMSNorm → tied LM head`, plus generation entry points.

**Why it exists.** It wires the 13 components above into the end-to-end forward
pass of §1, owns the two RoPE buffers, the embedding/head weight tying, weight
init, and the causal mask — i.e. everything that is global to the architecture
rather than local to a layer.

**How it works — construction.**

```python
self.embed      = nn.Embedding(vocab_size, dim)
# TWO RoPE buffers (gotcha 2):
self.freqs_cis     = precompute_rope_freqs(dim // n_heads,      max_seq_len, rope_theta)  # GQA-sized
self.freqs_cis_mla = precompute_rope_freqs(qk_rope_head_dim,    max_seq_len, rope_theta)  # MLA-sized
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
 freqs_cis = (freqs_cis_mla if attn_type=="mla" else freqs_cis)[start_pos : start_pos + T]
 mask      = _causal_mask(T, device, x.dtype) if T > 1 else None   # decode (T=1) needs no mask
 for i, layer in enumerate(prelude):  x = layer(x, freqs_cis, mask, kv_cache, f"prelude_{i}")
 e = x                                                          # freeze encoded input
 x = recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)
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
2. **Dual RoPE buffers.** GQA rotates the full `dim // n_heads`; MLA rotates only
   `qk_rope_head_dim`. The model precomputes **both** and selects per `attn_type`
   at forward. Mixing them up silently breaks one back-end. (Spec §5.2.)
3. **Mask only when `T > 1`.** Single-token decode (`T=1`) needs no causal mask;
   building one would be wrong shape-wise against the cached `S`. (See §15.)
4. **Weight init ignores residual-depth scaling (improvement note).** Naive
   `N(0, init_std)` on all weights does not account for the variance a *looped*
   model accumulates across loops. Document GPT-2-style `1/√(2·n_eff)` scaling on
   residual output projections (and optionally QK-norm / logit z-loss) as a
   stability improvement to ablate (spec §5.9, §14 init note).

**Where it fits.** It *is* the model — Prelude, Recurrent, and Coda all hang off
it, and it owns everything global.

---

### (15) KV cache & autoregressive generation — `model.py`

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
- **GQA** caches `{"k", "v"}`; **MLA** caches the much smaller compressed latent
  `{"c_kv", "k_rope"}`.
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

- **`start_pos` is load-bearing** (see §14 gotcha 1).
- **A correctness invariant worth a test:** cached single-token decode logits must
  match a full-context forward pass over the same sequence (within numerical
  tolerance). This is the canonical KV-cache regression test (Phase 5 / Phase 6).
- **The ACT short-circuit must be disabled when caching** (§13 critical gotcha) —
  the cache and the recurrence interact precisely here.

**Where it fits.** Spans all three stages (every layer/loop writes to the same
cache dict). Lives on the model; exercised by `generate` and
`generate_depthwise_batched`.

---

### (16) INT8 quantization — `quantize.py`

```python
def quantize_int8(model: "Ouroboros", method: str = "dynamic") -> "Ouroboros": ...

class INT8Linear(nn.Module):
    def __init__(self, linear: nn.Linear): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

def calibrate(model, calibration_loader) -> None: ...        # for static PTQ

def quantization_error(fp_model, int8_model, eval_loader) -> dict: ...   # ppl delta, etc.
```

**What it is.** Post-training INT8 quantization for **inference only** — one half
of the "beyond the literature" differentiator (resume bullet 3). Replaces the big
`nn.Linear` layers with INT8-weighted equivalents (`INT8Linear`).

**Why it exists.** Memory and throughput. The T4 has **INT8 tensor cores**, so
8-bit weights cut memory roughly in half *and* can run faster than FP16 on
matmul-bound layers — directly feeding the throughput multiplier in resume bullet
3. The architectural cost is measured (perplexity delta), not assumed.

**How it works.**

- **Per-channel weight quantization** of the big linears: attention projections
  (`wq/wk/wv/wo`, MLA's down/up projections) and the expert FFNs. **Keep in higher
  precision:** RMSNorm weights, the router, and the tied LM head — these are small,
  sensitive, and cheap to leave in FP16.
- `INT8Linear` wraps an existing `nn.Linear`, storing INT8 weights + per-channel
  scales and dequantizing/computing in the forward pass.
- `method`: `"dynamic"` (activations quantized per-batch at runtime — simplest,
  no calibration) or `"static"` (activation ranges fixed ahead of time via
  `calibrate`). Backends: `torch.ao.quantization` (dynamic/static) or
  `bitsandbytes` Int8 linear.
- `quantization_error(fp_model, int8_model, eval_loader)` reports the perplexity
  delta (and related metrics) so the accuracy cost is on record.

**Inputs / outputs.** `quantize_int8(model) → model` (an Ouroboros with INT8
linears swapped in). `quantization_error(...) → dict` of metrics.

**Gotchas.**

- **Do not quantize norms / router / LM head** — small layers with outsized
  sensitivity; quantizing them hurts perplexity for negligible savings.
- **Pick the realistic backend.** Document the actual choice (`torch.ao` vs
  `bitsandbytes`) and the T4 INT8-tensor-core caveat — measured, not assumed.
- **It is inference-only** — no gradients flow through INT8 weights here.

**Where it fits.** Applied to a *trained* `Ouroboros` (Phase 7), orthogonal to the
forward-pass structure: it swaps layer internals across all three stages without
changing the data flow.

---

### (17) Continuous depth-wise batching — `model.py`

```python
@torch.no_grad()
def generate_depthwise_batched(self, input_ids, max_new_tokens=64, max_loops=None,
                               temperature=1.0, top_k=50) -> torch.Tensor: ...
```

**What it is.** **THE** inference differentiator (the other half of resume bullet
3). Because all sequences share the same recurrent weights, different sequences in
one batch can **exit the loop at different depths** (ACT-driven): easy sequences
halt early, hard ones loop more — all in a single batch — instead of every
sequence paying the maximum depth.

**Why it exists.** With a fixed loop count, the whole batch runs to max depth even
if most sequences converged early — wasted compute proportional to the gap between
mean and max halting depth. Continuous depth-wise batching tightens that to the
*active* depth, tying throughput directly to the ACT halting distribution.

**How it works — and the central challenge.** The hard part is the
**KV-cache ⊗ ACT interaction** (the same invariant as §13). A sequence that exits
at depth `d` leaves cache keys `recurrent_loop_{d..n}` **unpopulated**; a later
decode step for that sequence that needs to loop deeper would read missing keys.
Three solutions to weigh (document the chosen one with measured tradeoffs):

1. **Run-to-max-active + mask.** Each step, run all sequences to the **maximum
   active depth currently in the batch**, masking finished ones. Simple, correct,
   keeps the cache dense; gains scale with the spread of halting depths.
2. **Ragged / compacted per-sequence cache.** Track per-sequence depth and
   maintain a ragged or compacted cache so each sequence stores only the depths it
   actually ran. Most memory-efficient, most bookkeeping.
3. **Bucket by predicted depth.** Sort/bucket sequences so a batch shares a depth,
   removing the ragged-cache problem at the cost of a scheduling pass.

**Inputs / outputs.** `generate_depthwise_batched(input_ids (B, T)) →
(B, T + max_new_tokens)` `long`. `max_loops` caps the deepest any sequence may run.

**Throughput expectation.** Literature suggests ~**2–3×** depending on the ACT
halting distribution; the *measured* number on T4 fills in resume bullet 3. Tie
the result back to the observed halting-depth histogram (see
[`EXPERIMENTS.md`](./EXPERIMENTS.md), experiment 6).

**Where it fits.** An inference-time orchestration over the Recurrent stage,
implemented on the model (Phase 7). It does not change the math of any component —
it changes *which depths run for which sequences*, subject to the §13 cache
invariant.

---

## 5. Key design properties (summary table)

| Property                          | Component(s)                  | Mechanism                                                            | Why it matters                                                                  |
| --------------------------------- | ----------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Stable looped training**        | LTIInjection (11)             | Diagonal `A` via ZOH, `ρ(A) < 1` by construction; log-space clamp `(-20,20)` | Converges at high LR with no clipping/normalization band-aids (resume bullet 2) |
| **Depth = compute, not params**   | RecurrentBlock (13)           | One shared block looped `n_loops` times                             | Deeper reasoning with zero parameter growth                                     |
| **Depth extrapolation**           | RecurrentBlock (13), LoRA (10), loop-index (9) | Sinusoidal depth signal + clamped per-loop scale  | Train at `n_loops=8`, run deeper at inference                                   |
| **Adaptive per-token compute**    | ACTHalting (12), RecurrentBlock (13) | Cumulative-probability halting + remainder trick             | Easy tokens halt early; hard tokens loop more — in one batch                    |
| **Breadth in the loop**           | MoEFFN (7), Expert (6)        | Fine-grained routed + always-on shared experts                      | Large capacity, sparse top-K compute, no per-domain bottleneck                  |
| **Real load balancing**           | MoEFFN.update_router_bias (7) | Aux-loss-free bias nudged `±rate·sign(mean−load)` each step          | Balanced experts without an auxiliary loss; closes the reference-impl stub gap  |
| **Switchable attention**          | GQAttention (4), MLAttention (5) | `attn_type` flag; dual RoPE buffers                              | GQA (FA2 fast path) ↔ MLA (compressed KV cache) without code changes            |
| **Cheap decode memory**           | MLAttention (5), KV cache (15)| Cache compressed `c_kv` + shared `k_rope` (MLA) / grouped KV (GQA)  | Longer context & larger batch on a 16 GB T4                                      |
| **Anti-drift recurrence**         | LTIInjection (11), RecurrentBlock (13) | `B·e` + `norm(h_loop+e)` re-inject frozen input every loop  | Prompt stays alive across arbitrary depth                                       |
| **No learned position table**     | RoPE (3)                      | Complex-phasor rotation, norm-preserving isometry                   | Relative positions, clean extrapolation, zero position params                   |
| **Inference throughput**          | INT8 (16), depth-wise batching (17) | Per-channel INT8 weights + per-sequence early exit            | Memory ↓ + ~2–3× throughput on T4 (resume bullet 3)                             |
| **Correctness under caching**     | RecurrentBlock (13), gen (15/17) | Early-exit only when `kv_cache is None`; deterministic cache keys | The headline subtlety — cached decode matches full-context forward              |
| **fp16 numerical safety**         | RMSNorm (2), RoPE (3), LTI (11) | fp32 reductions; dtype-matched additive mask                       | Stable training/inference in fp16 on Turing                                     |
| **Small-scale parameter economy** | OuroborosConfig (1), Expert (6) | `vocab_size=8192`, tied embeddings, `dim*4//3` dense FFN            | Keeps the budget on the transformer, not a giant lookup table (T4-realistic)    |

---

*Cross-references:* roadmap & phase gates → [`ROADMAP.md`](./ROADMAP.md);
rationale & alternatives per decision → [`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md);
experiment designs & result tables → [`EXPERIMENTS.md`](./EXPERIMENTS.md);
papers per component → [`READING_LIST.md`](./READING_LIST.md).
