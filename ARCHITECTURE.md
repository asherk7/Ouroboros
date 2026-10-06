# Ouroboros Architecture

Ouroboros is a recurrent-depth (looped) transformer language model. A small
dense **Prelude** encodes the input once, one weight-shared **Recurrent** block
(GQA attention plus a mixture-of-experts feed-forward layer) is looped a
variable number of times with stable input injection, and a small dense
**Coda** decodes the result once. This document describes the architecture: the layers, the mechanisms, and why each was chosen.

**Credits.** Prelude/Recurrent/Coda, the random-loop-count training recipe, and
truncated backprop come from Huginn (Geiping et al., 2025). The stable
injection comes from Parcae (Prairie et al., 2026). The MoE design comes from
DeepSeekMoE and DeepSeek-V3. The looped-block idea goes back to Universal
Transformers (Dehghani et al., 2018).

---

## 1. Forward pass

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
 │  (training: gradients flow through the last backprop_loops)    │
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

| Stage | Role | Feed-forward | Runs |
|---|---|---|---|
| Prelude | lift tokens into a latent space; produce `e` | dense | once |
| Recurrent | refine `h` while re-reading `e` | MoE | `n_loops` times |
| Coda | read the final latent out to logits | dense | once |

**Why Prelude/Recurrent/Coda instead of looping the whole model.** A fully
looped stack makes one set of weights do token encoding, iterative refinement,
and readout. Giving the two ends their own cheap weights lets the loop
specialize in refinement, and gives the injected `e` a stable encoded space.

## 2. The LTI recurrence

```
 h_{t+1} = A * h_t  +  B * e  +  Transformer(h_t, e)        with  ρ(A) < 1
```

- `A` is a learned diagonal with every entry in `(0, 1)`, so
  `ρ(A) = max(diag(A)) < 1` for any parameter values. The linear part is a
  contraction: perturbations shrink across loops instead of compounding.
- `B * e` re-injects the frozen input every loop, so the prompt is never lost
  however deep the loop runs.
- `Transformer(h_t, e)` is the nonlinear refinement at this loop.

---

## 3. Building blocks

Every stage is built from the same transformer block: attention then a
feed-forward layer, each wrapped in a residual connection with RMSNorm in
front (pre-norm).

### Embedding and output head

Tokens come from an 8192-entry byte-level BPE vocabulary. The embedding table
and the output head share one weight matrix (weight tying). At `dim = 512`, a
50k vocabulary would be most of the model; 8192 keeps the shared table at 4.2M
parameters, so the budget goes to the transformer.

### RMSNorm, and where it sits

RMSNorm rescales each vector by its root mean square, with a learned per-channel
gain:

```
 RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
```

Unlike LayerNorm it does not subtract the mean and has no bias. It is cheaper
and works as well. It appears in four places:

- **Pre-norm** in front of every attention and feed-forward sublayer.
- **QK-norm**: on the queries and keys inside attention (see below).
- **The loop input**: `RMSNorm(h_loop + e)` before each recurrent step.
- **The final norm** before the output head.

Huginn needed a "sandwich" norm (an extra RMSNorm on each sublayer's output) to
train its loop stably at 3.5B parameters. Plain pre-norm should be fine at this
size; sandwich norm is the first thing to try if the loop misbehaves.

### Positions: RoPE

Rotary position embedding encodes a token's position by rotating its query and
key vectors. Each head's features are split into pairs, and pair `k` at
position `m` is rotated by the angle `m · θ_k`, with `θ_k = 10000^(-2k/d)`
(fast-spinning pairs for fine position, slow ones for coarse position). Two
rotations compose, so the dot product between a query at `m` and a key at `n`
depends only on the offset `m - n`: attention sees relative position for free,
with no learned position table. Rotation also preserves vector length, and
position 0 is no rotation at all.

### Attention: GQA with QK-norm

Causal multi-head self-attention with **grouped-query attention**: 8 query
heads but only 2 key/value heads, each shared by a group of 4 query heads.
Plain multi-head attention gives every query head its own K/V; multi-query
attention shares one K/V across all heads; GQA sits in between, keeping most of
the quality while cutting the K/V that must be stored during generation by 4.

**Why GQA here.** Every loop iteration stores its own K/V (section 5), so a
looped model's KV cache is far deeper than its parameter count suggests. The
4x saving matters more than in a plain transformer.

**QK-norm** applies RMSNorm to queries and keys (per head, before RoPE). This
bounds the attention logits, which otherwise can grow until softmax saturates
and training spikes. That risk is higher when the same attention weights run
many times per token. QK-norm is standard in Qwen3, OLMo 2, and Gemma 3.

### FlashAttention

Standard attention computes the full `T × T` score matrix, writes it to GPU
main memory (HBM), reads it back for softmax, writes again, and reads again to
multiply by V. For long sequences that memory traffic, not the math, is the
bottleneck. FlashAttention computes the exact same result without ever storing
the score matrix:

- **Tiling.** Split Q, K, V into blocks small enough to fit in the GPU's fast
  on-chip SRAM, and process one block of keys at a time for each block of
  queries.
- **Online softmax.** Softmax needs the max and sum over a whole row, which a
  single block cannot see. So each query row keeps a running max `m`, a running
  denominator `l`, and a running output `o`. When a new block arrives with a
  larger max, the old `l` and `o` are rescaled by `exp(m_old - m_new)`. After
  the last block, `o / l` is exactly softmax attention.
- **Recomputation.** The backward pass recomputes the scores block by block
  instead of storing them, trading cheap FLOPs for expensive memory traffic.

FlashAttention-2 keeps the algorithm and reorganizes the work (fewer non-matmul
operations, better parallelism across the sequence). Ouroboros trains on
PyTorch's fused attention, which uses a FlashAttention-style kernel; the
algorithm itself is reimplemented in plain PyTorch as an exercise and checked
against it.

### Feed-forward: SwiGLU

```
 FFN(x) = W_down( SiLU(W_gate x) ⊙ W_up x )
```

A gated feed-forward layer: one projection, passed through SiLU, gates another
elementwise. It beats ReLU and GELU MLPs per parameter and is standard in
LLaMA- and DeepSeek-class models. The Prelude and Coda use a dense SwiGLU of
hidden width `4/3 · dim`, narrower than the usual `8/3 · dim` to leave parameter
budget for the MoE.

### Mixture of experts (recurrent block only)

The recurrent block's feed-forward layer is a **fine-grained MoE**: 8 small
routed experts (each a SwiGLU of width 256) plus 1 always-on shared expert
(width 512). A router scores every expert for every token; each token uses its
top 2 routed experts plus the shared expert:

```
 scores = softmax(router(x))
 chosen = top2(router(x) + bias)               # selection uses the bias
 out    = Σ_chosen w_i · expert_i(x)  +  shared(x),   w = renormalized scores
```

- **Fine-grained experts** (DeepSeekMoE): many small experts give far more
  combinations than a few big ones at the same active compute.
- **Shared expert** (DeepSeekMoE): absorbs common patterns every token needs,
  so the routed experts can specialize.
- **Aux-loss-free load balancing** (DeepSeek-V3): without balancing, the router
  collapses onto a few experts. The classic fix, an auxiliary balance loss,
  fights the language-modeling loss. Instead, a per-expert bias is added to the
  router scores for **selection only**; the mixing weights come from the
  unbiased scores, so the bias never touches the gradient. After every
  optimizer step the bias moves down for overloaded experts and up for
  underloaded ones.

**Why MoE only in the loop.** The looped block does the most work per
parameter, so cheap extra capacity pays off most there. The Prelude and Coda
run once, and keeping them dense keeps `e` free of routing noise.

---

## 4. The recurrent block

### Loop body

Each iteration: add a loop-index signal to `h`, add the input `e`, normalize,
run the shared transformer block (GQA + MoE), then apply the LTI update. The
loop starts from `h_0 = e`. Every position runs the same number of loops; there
is no early exit (halting).

**Choices vs Huginn.** Huginn starts the loop from random noise and merges `e`
by concatenating and projecting. Ouroboros starts from `e` and simply adds it;
Huginn reports adding works as well at small scale, and both choices are
simpler.

### Loop index

The block reuses the same weights every loop, so without a signal it cannot
tell loop 0 from loop 7. A sinusoidal signal of the loop number, built like
RoPE's frequencies, is added to the first `dim / 8` channels of `h`. Unlike a
learned per-loop table, a sinusoid is defined for any loop number, including
ones never seen in training, and has no parameters. Universal Transformers used
the same kind of timestep signal.

### LTI injection

Parcae's update `h = A·h + B·e + Transformer(h, e)` borrows from state-space
models. `A` is built as a discretized continuous-time system with a strictly
negative rate:

```
 A_continuous = -exp(log_A)              (always negative)
 Δt           = exp(log_dt)              (always positive)
 A            = exp(Δt · A_continuous)   (zero-order hold, always in (0, 1))
```

Because every entry of `A` lies in `(0, 1)` whatever values `log_A` and
`log_dt` take, `ρ(A) < 1` holds by construction. Across loops the state
behaves like `A^t · h_0 + ...`; with `ρ(A) ≥ 1` it would explode, and so would
the gradients through the loop. Gradient clipping or renormalizing `h` every
loop would only hide that; this removes it. `ρ(A)` is also a free stability
readout to log during training.

The comparison arm for the stability metric replaces this update with the
naive `h = Transformer(h, e) + e`: no `A`, no `B`, no guarantee.

### Training: random loop count and truncated backprop

Both from Huginn:

- **Random loop count.** Each training step samples a loop count from a
  heavy-tailed distribution with mean 8 (a Poisson whose rate is itself
  log-normal), so the model sees short loops often and much longer ones now
  and then. A model
  trained at exactly 8 loops has no reason to behave at 4 or 32; this is what
  makes the loop count a real dial at test time.
- **Truncated backprop.** Gradients flow through only the last 4 loops; earlier
  loops run without building a graph. Memory and backward cost stay flat
  however many loops were sampled. The input `e` still receives gradient,
  because it is injected into every loop that has a graph.

---

## 5. Inference: the KV cache in a looped model

During generation, each new token attends to every earlier token's keys and
values. Recomputing those every step makes generation quadratic in length; the
**KV cache** stores them so each step processes only the new token.

In a looped model each loop iteration needs its own cache entry: the weights
are shared, but the block's input differs every loop, so its K/V differ too.
Per token, the cache holds:

```
 2 (K and V) × kv_heads × head_dim × (prelude + n_loops + coda) layers × bytes
 = 2 × 2 × 64 × (2 + 8 + 2) × 2 bytes = 6 KB      (fp16, 8 loops)
```

That is 24 KB with plain multi-head attention, and it grows linearly with the
loop count. A consequence: the loop count must stay fixed for a whole
generation, since a deeper step would find no cached keys for its extra loops.

**Variable depth (not built here).** If tokens exited the loop at different
depths, a token at loop `t` would need keys from earlier tokens at loop `t`
that were never computed. Known answers: run everyone to the deepest active
depth and mask the finished ones; keep a ragged per-token cache; reuse the
most recent loop's K/V for a token when an entry is missing (Huginn); or cache
only active tokens per recursion, or share the first recursion's K/V across all
recursions (Mixture-of-Recursions). Raschka notes that sharing K/V across
passes hurt quality in Nanbeige's experiments, so the shortcut is not free.

---

## 6. Size

| | |
|---|---|
| Width | 512 |
| Attention | 8 query heads, 2 KV heads, head size 64 |
| Layers | 2 Prelude + 1 looped (8 loops on average) + 2 Coda |
| MoE | 8 routed experts (top 2) + 1 shared, recurrent block only |
| Vocabulary / context | 8192 / 512 tokens |
| Parameters | about 15.6M total (4.2M of it the tied embedding), about 13.2M active per token |
