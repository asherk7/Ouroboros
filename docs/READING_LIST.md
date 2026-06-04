# Reading List

This is the curated literature behind Ouroboros, organized by the architectural
component each paper informs. Ouroboros is an **independent from-scratch
implementation inspired by the recurrent-depth transformer literature** — these
papers are the source of the *ideas*; none is a codebase we forked or copied.
The value of "adopting" a component is depth of understanding: for each, be able
to answer *what problem does it solve, what is the alternative, and why is this
the better trade-off here?*

For every entry you get: title, authors, year, link, the Ouroboros component(s)
it maps to, what to focus on while reading (specific sections / figures /
equations), and a priority tag:

- **MUST-READ** — you cannot correctly implement the mapped component without it.
- **CONTEXT** — sharpens understanding, motivates a design decision, or supports
  a resume talking point, but the component can be built without a deep read.

---

## Suggested reading order (implement phase by phase)

Read just-in-time, in lock-step with the [roadmap](./ROADMAP.md), so each paper
lands right before you build the thing it describes:

> **Phase 1 (Skeleton):** RoFormer/RoPE → RMSNorm → SwiGLU.
> **Phase 2 (Attention):** GQA → DeepSeek-V2 (MLA) → FlashAttention-2.
> **Phase 3 (MoE):** DeepSeekMoE → DeepSeek-V3 (aux-loss-free bias).
> **Phase 4 (Recurrence — the heart of the project):** Universal Transformers →
> Adaptive Computation Time → Relaxed Recursive Transformers → **Parcae**
> (read this one closely; it is the stability thesis of the whole project).
> **Phase 5 (Full model / depth extrapolation):** Reasoning with Latent Thoughts
> (looped-transformer theory) → COCONUT (context).
> **Phase 7 (Inference optimization):** LLM.int8().

If you only have time for five, read in this order: **Parcae**, **DeepSeek-V2**,
**DeepSeekMoE**, **Universal Transformers**, **DeepSeek-V3**.

---

## 1. Recurrence & Stability

The defining axis of Ouroboros: a Prelude/Recurrent/Coda model whose middle
block is looped to variable depth, with stability guaranteed by construction
rather than patched with gradient clipping. These papers justify the
[`RecurrentBlock`](./ARCHITECTURE.md), [`LTIInjection`](./ARCHITECTURE.md),
[`ACTHalting`](./ARCHITECTURE.md), and [`LoRAAdapter`](./ARCHITECTURE.md)
components and back resume bullets 1 and 2.

### Parcae: Scaling Laws for Stable Looped Language Models
- **Authors / Year:** Prairie et al., 2026
- **Link:** https://arxiv.org/abs/2604.12946
  (blog: https://sandyresearch.github.io/parcae/)
- **Maps to:** `LTIInjection`, `RecurrentBlock`, and the whole
  Prelude/Recurrent/Coda framing.
- **Focus on:** the **LTI-constrained injection** scheme — the recurrence
  `h_{t+1} = A·h_t + B·e + Transformer(h_t, e)` with diagonal, **negative**
  continuous state matrix and **spectral radius ρ(A) < 1 by construction**.
  Study how the ZOH discretization (`A_discrete = exp(dt · A_continuous)`,
  `A_continuous = -exp(log_A)`) keeps `A ∈ (0, 1)` for free, and read the scaling
  curves showing that constrained loops converge at high LR where unconstrained
  loops diverge. This is the centerpiece of the stability experiment
  (log `ρ(A) = max(get_A())` every step; see [EXPERIMENTS](./EXPERIMENTS.md)).
- **Priority:** **MUST-READ.** This is the single most load-bearing paper in the
  project — it is the source of resume bullet 2.

### Universal Transformers
- **Authors / Year:** Dehghani et al., 2018
- **Link:** https://arxiv.org/abs/1807.03819
- **Maps to:** `RecurrentBlock`, `ACTHalting`, the weight-tied-looped-depth idea.
- **Focus on:** the recurrent (depth-as-time) formulation of a transformer with
  **shared weights applied repeatedly**, and the per-position **ACT halting**
  applied to recurrence depth (their §2.2 on dynamic halting; the halting figure).
  This is the conceptual ancestor of the whole looped design — note how a single
  shared block run T times differs from T distinct stacked layers, which directly
  motivates our per-loop LoRA and loop-index embedding.
- **Priority:** **MUST-READ.**

### Adaptive Computation Time for Recurrent Neural Networks
- **Authors / Year:** Graves, 2016
- **Link:** https://arxiv.org/abs/1603.08983
- **Maps to:** `ACTHalting`, the ACT remainder/accumulation logic in
  `RecurrentBlock`.
- **Focus on:** the **halting-probability mechanism** and the **remainder trick** —
  accumulate per-step halting probabilities until a threshold, then weight the
  final step by `1 - cumulative_p` so contributions sum to 1. This is *exactly*
  the `weight = where(cumulative_p + p >= act_threshold, 1 - cumulative_p, p)`
  accumulation in our loop body. Read the ponder-cost discussion to understand why
  the threshold (`act_threshold = 0.99`) trades compute against accuracy.
- **Priority:** **MUST-READ.**

### Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA
- **Authors / Year:** Bae et al., 2024
- **Link:** https://arxiv.org/abs/2410.20672
- **Maps to:** `LoRAAdapter` (depth-wise LoRA).
- **Focus on:** the argument that **pure weight-tying is too rigid** and
  **fully-distinct layers are too expensive**, and the LoRA-per-depth middle
  ground — a shared down-projection and base matrix with a small per-depth
  adapter. Map this onto our `delta(x, t) = (down(x) * scale[t]) @ B` with a
  per-loop `scale` embedding. Note the depth-extrapolation concern: at inference
  `loop_t` can exceed `max_loops - 1`, so the index must be **clamped** to the
  last learned scale.
- **Priority:** **MUST-READ.**

### Reasoning with Latent Thoughts: On the Power of Looped Transformers
- **Authors / Year:** Saunshi et al., 2025
- **Link:** https://arxiv.org/abs/2502.17416
- **Maps to:** `RecurrentBlock`, depth-extrapolation experiments.
- **Focus on:** the theory of *why* looping adds effective depth/expressivity at
  fixed parameter count, and the evidence that loop count can be **extrapolated at
  test time** beyond the trained depth. This underwrites the loop-count sweep
  (`n_loops ∈ {2,4,8,16}` for a model trained at 8) in
  [EXPERIMENTS](./EXPERIMENTS.md) and the depth-extrapolation talking point.
- **Priority:** **CONTEXT.**

---

## 2. Attention

Ouroboros ships **switchable GQA / MLA** attention (`attn_type`), each with its
own RoPE buffer, and integrates a flash-attention fast path. Maps to
[`GQAttention`](./ARCHITECTURE.md) and [`MLAttention`](./ARCHITECTURE.md), and
backs resume bullets 1 and 3.

### GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
- **Authors / Year:** Ainslie et al., 2023
- **Link:** https://arxiv.org/abs/2305.13245
- **Maps to:** `GQAttention`.
- **Focus on:** how **grouped-query attention** interpolates between multi-head
  and multi-query by sharing each KV head across a group of query heads
  (`groups = n_heads // n_kv_heads`), and the quality/KV-cache-size trade-off
  curve. Confirms why `n_heads % n_kv_heads == 0` must hold and why GQA shrinks
  the KV cache versus full MHA — directly relevant to the MLA-vs-GQA cache-memory
  ablation.
- **Priority:** **MUST-READ.**

### DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
- **Authors / Year:** DeepSeek-AI, 2024
- **Link:** https://arxiv.org/abs/2405.04434
- **Maps to:** `MLAttention` (Multi-head Latent Attention).
- **Focus on:** the **MLA** section — the compressed-KV scheme that caches a small
  latent `c_kv` (size `kv_lora_rank`) plus a shared rotary key, instead of full
  K/V. Trace every shape: the Q down/up path (`q_lora_rank` → split into
  `qk_nope_head_dim` and RoPE'd `qk_rope_head_dim`), the KV down path
  (`kv_lora_rank + qk_rope_head_dim`), and the per-step reconstruction of
  `[k_nope | v]` from the cached latent. Note the **decoupled RoPE** detail (RoPE
  applies only to `qk_rope_head_dim`), which is why Ouroboros precomputes a
  separate `freqs_cis_mla` buffer. `n_kv_heads` is irrelevant under MLA.
- **Priority:** **MUST-READ.**

### FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
- **Authors / Year:** Dao, 2023
- **Link:** https://arxiv.org/abs/2307.08691
- **Maps to:** `GQAttention` fast path; inference optimization (resume bullet 3).
- **Focus on:** the IO-aware tiling and improved work partitioning that make
  attention memory-linear and fast, and the **causal-masking / online-softmax**
  mechanics. Important caveat for this project: official FA2 prebuilt support
  targets **Ampere (sm80)+**, while our target is a **Turing T4 (sm75)**, where the
  `flash-attn` package is forward-only and often won't build. Read with the goal
  of understanding *what the kernel does* so the realistic T4 path —
  `F.scaled_dot_product_attention` with the flash / mem-efficient backend via
  `torch.backends.cuda.sdp_kernel`, plus a manual fallback — is a principled
  substitute, with `flash_attn_func` kept only as an optional Ampere fast path.
- **Priority:** **MUST-READ.**

---

## 3. Mixture-of-Experts (MoE)

The Recurrent Block's FFN is a **fine-grained MoE** with routed + shared experts
and aux-loss-free load balancing. Maps to [`MoEFFN`](./ARCHITECTURE.md) and
[`Expert`](./ARCHITECTURE.md); backs resume bullet 1.

### DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models
- **Authors / Year:** Dai et al., 2024
- **Link:** https://arxiv.org/abs/2401.06066
- **Maps to:** `MoEFFN`, `Expert`.
- **Focus on:** the two core ideas — **fine-grained expert segmentation** (many
  small experts; our `expert_dim ≈ dim // (n_experts // n_experts_per_tok)` rule
  of thumb) and **shared experts** that always fire to capture common knowledge
  (ours use the larger width `expert_dim * n_experts_per_tok`). Read the routing /
  top-K selection and the expert-specialization ablations. This defines the shape
  and structure of our routed-plus-shared design.
- **Priority:** **MUST-READ.**

### DeepSeek-V3 Technical Report
- **Authors / Year:** DeepSeek-AI, 2024
- **Link:** https://arxiv.org/abs/2412.19437
- **Maps to:** `MoEFFN` — specifically the `update_router_bias()` step.
- **Focus on:** the **auxiliary-loss-free load-balancing** strategy. The key
  subtlety to internalize: expert *selection* uses `topk(logits + router_bias)`,
  but the gating *weights* come from the **unbiased** `softmax(logits)`, so the
  bias never enters the gradient; the bias is nudged each step toward balance by
  `router_bias_update_rate * sign(load - mean_load)`. **Ouroboros completion:**
  many reference implementations register the bias buffer but never update it —
  we implement the update step and call it from the training loop. Read this
  carefully; it is the concrete engineering-maturity win in
  [DESIGN_DECISIONS](./DESIGN_DECISIONS.md) #9.
- **Priority:** **MUST-READ** (for the bias-update mechanism).

---

## 4. Positional Encoding, Normalization & FFN

The dense building blocks shared across every block. Maps to
[`RoPE`](./ARCHITECTURE.md), [`RMSNorm`](./ARCHITECTURE.md), and
[`Expert`](./ARCHITECTURE.md) (SwiGLU); these are the Phase-1 primitives.

### RoFormer: Enhanced Transformer with Rotary Position Embedding
- **Authors / Year:** Su et al., 2021
- **Link:** https://arxiv.org/abs/2104.09864
- **Maps to:** `RoPE` (`precompute_rope_freqs`, `apply_rope`); reused for the
  sinusoidal `loop_index_embedding`.
- **Focus on:** the rotary formulation — encode position by **rotating** adjacent
  feature pairs by `m · θ_k` with `θ_k = theta^(-2k/dim)` (our `rope_theta`).
  Note the properties our implementation relies on: rotation is **norm-preserving**
  (an isometry), **position 0 is the identity**, and relative position emerges from
  the dot product. These directly justify the Phase-1 RoPE tests (norm
  preservation, position-0 identity) and the `head_dim`-must-be-even gotcha. The
  same `m · θ_k` construction, applied over **recurrence depth** instead of token
  position, is the loop-index embedding.
- **Priority:** **MUST-READ.**

### GLU Variants Improve Transformer (SwiGLU)
- **Authors / Year:** Shazeer, 2020
- **Link:** https://arxiv.org/abs/2002.05202
- **Maps to:** `Expert` (the SwiGLU FFN used as both routed expert and dense
  prelude/coda FFN).
- **Focus on:** the **SwiGLU** formulation `down(silu(gate(x)) * up(x))` and the
  ablation table showing GLU variants beating plain ReLU/GELU FFNs. Note the
  three-matrix (`gate`, `up`, `down`) structure and the convention that GLU FFNs
  use a narrower hidden width to hold parameters constant — relevant to our
  deliberate `dim * 4 // 3` prelude/coda width (smaller than the common `8/3·dim`),
  a parameter-budget choice discussed in [DESIGN_DECISIONS](./DESIGN_DECISIONS.md).
- **Priority:** **MUST-READ.**

### Root Mean Square Layer Normalization (RMSNorm)
- **Authors / Year:** Zhang & Sennrich, 2019
- **Link:** https://arxiv.org/abs/1910.07467
- **Maps to:** `RMSNorm`.
- **Focus on:** the central claim that the **re-centering** in LayerNorm is
  unnecessary — only **re-scaling** matters — so `x * rsqrt(mean(x²) + eps) * weight`
  (no mean subtraction, no bias) is both cheaper and as effective. Read enough to
  justify the per-channel learned `weight` (init 1) and our gotcha: do the RMS
  reduction in **float32** then cast back to avoid fp16 underflow.
- **Priority:** **CONTEXT** (the module is simple; the paper motivates the design).

---

## 5. Inference Optimization

The "beyond the literature" axis: INT8 quantized inference plus continuous
depth-wise batching. Maps to [`quantize_int8` / `INT8Linear`](./ARCHITECTURE.md)
and `generate_depthwise_batched`; backs resume bullet 3.

### LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
- **Authors / Year:** Dettmers et al., 2022
- **Link:** https://arxiv.org/abs/2208.07339
- **Maps to:** `quantize_int8`, `INT8Linear`, `calibrate`,
  `quantization_error`.
- **Focus on:** **vector-wise / per-channel INT8 quantization** of the large
  linear layers and the **outlier-feature** problem (why a few high-magnitude
  feature dimensions break naive INT8 and must be handled). Map this onto our plan
  to per-channel-quantize the big Linears (attention projections, expert FFNs)
  while keeping norms, the router, and the (tied) LM head in higher precision, and
  to measure the **perplexity delta** (`quantization_error`) and throughput gain.
  T4 note: Turing has INT8 tensor cores, which is exactly why INT8 (not FP16/FP8)
  is the chosen quantization target — see
  [DESIGN_DECISIONS](./DESIGN_DECISIONS.md) #10.
- **Priority:** **MUST-READ** (for resume bullet 3).

> **Note — continuous depth-wise batching has no single source paper.** It is the
> inference *differentiator* of Ouroboros and is developed from first principles
> in [ARCHITECTURE](./ARCHITECTURE.md) (component 17) and
> [DESIGN_DECISIONS](./DESIGN_DECISIONS.md) #11. The relevant background is the
> ACT halting literature above (Graves 2016; Dehghani et al. 2018) — because
> sequences halt at different recurrence depths, easy ones can exit the loop early
> while hard ones loop more *within the same batch*. The central engineering
> problem (a sequence exiting at depth `d` leaves `recurrent_loop_{d..n}` KV-cache
> keys unpopulated) is the cache-population subtlety also flagged in component 13.

---

## 6. Theory & Context

Broader motivation for treating extra recurrence depth as latent "thinking." Not
tied to a single component; informs the framing of the looped design and the
depth-extrapolation story.

### Training Large Language Models to Reason in a Continuous Latent Space (COCONUT)
- **Authors / Year:** Hao et al., 2024
- **Link:** https://arxiv.org/abs/2412.06769
- **Maps to:** project framing — the looped recurrent block as "latent reasoning"
  in continuous space; supports the depth-extrapolation narrative.
- **Focus on:** the idea of reasoning by **feeding hidden states back as latent
  thoughts** rather than decoding intermediate tokens. Useful as conceptual
  framing for *why* iterating the recurrent block (vs. emitting chain-of-thought
  text) can add reasoning capacity, and a good comparison point when discussing
  what Ouroboros's loop is and is not.
- **Priority:** **CONTEXT.**

---

## Component → Paper index

Quick lookup from each Ouroboros component to its primary references
(★ = MUST-READ for that component).

| Component | Primary references |
| --- | --- |
| `RMSNorm` | RMSNorm (Zhang & Sennrich, 2019) |
| `RoPE` / `loop_index_embedding` | ★ RoFormer/RoPE (Su et al., 2021) |
| `GQAttention` | ★ GQA (Ainslie et al., 2023); ★ FlashAttention-2 (Dao, 2023) |
| `MLAttention` | ★ DeepSeek-V2 (2024) |
| `Expert` (SwiGLU) | ★ SwiGLU (Shazeer, 2020) |
| `MoEFFN` | ★ DeepSeekMoE (Dai et al., 2024); ★ DeepSeek-V3 (2024, bias update) |
| `loop_index_embedding` | ★ RoFormer/RoPE (Su et al., 2021); Universal Transformers (2018) |
| `LoRAAdapter` | ★ Relaxed Recursive Transformers (Bae et al., 2024) |
| `LTIInjection` | ★ Parcae (Prairie et al., 2026) |
| `ACTHalting` | ★ Adaptive Computation Time (Graves, 2016); ★ Universal Transformers (2018) |
| `RecurrentBlock` | ★ Parcae (2026); ★ Universal Transformers (2018); Reasoning with Latent Thoughts (Saunshi et al., 2025) |
| `Ouroboros` (depth extrapolation) | Reasoning with Latent Thoughts (2025); COCONUT (2024) |
| `quantize_int8` / `INT8Linear` | ★ LLM.int8() (Dettmers et al., 2022) |
| `generate_depthwise_batched` | (no single paper — see ARCHITECTURE component 17; ACT background above) |
