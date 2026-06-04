# Ouroboros — Build Roadmap

> Phased plan for building **Ouroboros**, a recurrent-depth (looped) transformer
> implemented from scratch in PyTorch. This document is the execution plan that
> turns the architecture (see [`ARCHITECTURE.md`](./ARCHITECTURE.md)) into working,
> measured code on a single Google Colab **T4** (16 GB, Turing sm75, FP16).

## What Ouroboros is

Ouroboros is an honest, from-scratch implementation inspired by the recurrent-depth
transformer literature (Universal Transformers, Parcae, DeepSeek-V2/V3, DeepSeekMoE).
It is a **Prelude / Recurrent / Coda** design: a few
dense blocks encode the input once (Prelude), a single weight-tied block is looped `T`
times with stable input injection (Recurrent Block), and a few dense blocks decode the
result once (Coda). The loop carries fine-grained MoE (routed + shared experts),
switchable MLA/GQA attention, a sinusoidal loop-index embedding, and
LTI-constrained injection for stability (a fixed loop count — no adaptive halting).
Beyond the literature, Ouroboros adds **INT8 quantized inference** and **continuous
depth-wise batching** as the inference differentiators.

This is a planning/documentation phase: the scaffold is API contracts (signatures +
docstrings) only. The roadmap below sequences the *implementation* that will fill those
contracts.

### The three resume claims this build backs

1. **Architecture (Phases 1–5).** "Implemented a recurrent-depth (looped) transformer
   from scratch in PyTorch — a Prelude/Recurrent/Coda design with fine-grained MoE
   (routed + shared experts) and switchable MLA/GQA attention."
2. **Stability (Phases 4 & 6).** "Stabilized looped training via LTI-constrained
   injection (negative-diagonal state matrix, spectral radius < 1), enabling clean
   convergence at high learning rates."
3. **Inference (Phase 7).** "Integrated FlashAttention-2 and INT8 quantized inference
   with continuous depth-wise batching, achieving **[X]×** inference throughput on a
   single GPU." (`[X]` is the measured multiplier produced in Phase 7.)

Bullets 1–2 are *build + train the architecture*. Bullet 3 is *go beyond the
literature*: INT8 PTQ and continuous depth-wise batching are the novel pieces;
FlashAttention-2 integration is table-stakes but must be made to *actually work* on T4
(realistic path = `F.scaled_dot_product_attention` flash/mem-efficient backend with a
manual fallback; the `flash_attn_func` kernel is kept as an optional Ampere fast path).

---

## Phase-dependency overview

```
 Phase 1  Skeleton ─────────────────────────────────────┐
 (config, RMSNorm, RoPE)                                 │
        │                                                │
        ├──────────────┬───────────────┐                 │
        ▼              ▼               ▼                  │
 Phase 2 Attention  Phase 3 MoE   Phase 4 Recurrence ◄────┘  (P4 needs P1–P3)
 (GQA, MLA)         (Expert,      (LTIInjection, loop_index_embedding,
        │            MoEFFN)       TransformerBlock, RecurrentBlock)
        │              │
        └──────┬───────┴───────────────┐
               ▼                        ▼
        Phase 5  Full model + generation (Ouroboros, KV cache, generate)
               │
               ▼
        Phase 6  Training (single-GPU; LTI vs no-LTI; W&B; MoE bias step)
               │
               ▼
        Phase 7  Inference optimization (INT8, FA2/SDPA, depth-wise batching)
               │
               ▼
        Phase 8  Polish (README, diagrams, benchmark table, tests/lint green)
```

Critical path: **1 → 2/3 → 4 → 5 → 6 → 7 → 8**. Phases 2 and 3 are independent of each
other (both depend only on Phase 1) and can be built in parallel or interleaved; Phase 4
needs all of Phases 1–3 because `RecurrentBlock` composes `TransformerBlock`, which wires
together attention (P2) and `MoEFFN` (P3).

### Effort summary

Estimates assume ~4–6 focused hours/day. The build-from-reference timeline can compress
because the math and shapes are well understood up front; the spread reflects
"smooth run" vs "debugging the subtle bits" (KV-cache decode correctness, dual RoPE
buffers, `start_pos` decode slicing, fp16 stability). **Phases 1–5 ≈ ~10 days** is the
realistic architecture-build target.

| Phase | Name | Effort (days) | Depends on |
|------:|------|:-------------:|------------|
| 1 | Skeleton | ~1 | — |
| 2 | Attention | ~2–3 | 1 |
| 3 | MoE FFN | ~2 | 1 |
| 4 | Recurrence machinery | ~2–3 | 1, 2, 3 |
| 5 | Full model + generation | ~2 | 1, 2, 3, 4 |
| **1–5 subtotal** | **Architecture** | **~10** | |
| 6 | Training | ~3–4 | 5 |
| 7 | Inference optimization | ~3–4 | 5 (6 for ppl baselines) |
| 8 | Polish | ~2 | 1–7 |
| **Total** | | **~18–20** | |

A note on calendar vs effort: T4 training and benchmark runs (Phases 6–7) include
wall-clock waiting for runs to finish that does not consume focused engineering hours;
budget for it separately from the day estimates above.

---

## Phase 1 — Skeleton

**Goal.** Stand up the project's numerical foundation: the config contract and the two
primitives every block depends on (normalization + positional encoding). Prove tensors
flow end-to-end through the simplest possible path with correct shapes.

**Components built** (exact names):
- `OuroborosConfig` — `config.py` (the dataclass contract; fields per spec §2).
- `RMSNorm` — `norm.py`.
- `precompute_rope_freqs`, `apply_rope` — `rope.py`.
- Minimal scaffolding to exercise them: `nn.Embedding(vocab_size, dim)` and a tied
  `nn.Linear(dim, vocab_size)` head used only for the smoke test (the real `Ouroboros`
  wiring lands in Phase 5).

**Acceptance criteria** (concrete):
- **Shape smoke test.** `embed → RMSNorm → tied head` on a random `(B, T)` Long tensor
  of token ids returns logits of shape `(B, T, vocab_size)` with no NaN/Inf, for the
  tiny config (`dim=64, n_heads=4`) and the default config (`dim=512`).
- **RMSNorm correctness.** Output matches a reference fp32 formula
  `x * rsqrt(mean(x², -1) + eps) * weight` to within `1e-5`; with `weight=1`, the
  per-row RMS of the output is ≈ 1.0. The reduction is computed in fp32 then cast back
  (verify no fp16 underflow on a deliberately tiny-magnitude fp16 input).
- **RoPE norm preservation (isometry).** For random `x` of shape `(B, T, H, head_dim)`,
  `‖apply_rope(x, freqs)‖ == ‖x‖` per `(B, T, H)` to within `1e-4` — rotation cannot
  change vector norm.
- **RoPE position-0 identity.** `freqs_cis[0]` is the identity phasor (`1 + 0j`), so
  `apply_rope(x, freqs_cis[:1])` returns `x` unchanged (within `1e-5`).
- **RoPE shape/dtype.** `precompute_rope_freqs(dim, max_len, theta)` returns a
  `complex64` tensor of shape `(max_len, dim // 2)`; `apply_rope` preserves input shape
  and dtype; asserts `head_dim` is even.
- **Config contract.** `OuroborosConfig()` instantiates with all spec defaults; a "tiny
  test config" and a "T4 training config" (~10–30M params) are documented (in docstrings
  / docs, not hardcoded as dataclass defaults).

**Estimated effort.** ~1 day. The three components are small and well-specified; the
only real care is the fp32-reduction detail in `RMSNorm` and the complex-view mechanics
in `apply_rope`.

**Dependencies.** None.

---

## Phase 2 — Attention

**Goal.** Implement both attention mechanisms behind a single switch (`cfg.attn_type`),
with a correct KV-cache contract and a realistic FlashAttention-2/SDPA fast path plus
fallback. Build GQA first (simpler, has the FA2 fast path), then MLA.

**Components built** (exact names):
- `GQAttention` — `attention.py` (Grouped Query Attention; `head_dim = dim // n_heads`,
  `groups = n_heads // n_kv_heads`; RoPE on Q and K; K/V cached *after* RoPE).
- `MLAttention` — `attention.py` (DeepSeek-V2 compressed-KV attention; Q low-rank path,
  KV low-rank latent `c_kv` cached, decoupled RoPE on `qk_rope_head_dim`).

**Acceptance criteria** (concrete):
- **GQA smoke + shape.** `GQAttention(cfg)(x, freqs_cis, mask)` maps `(B, T, dim) →
  (B, T, dim)` with no NaN; works for the tiny and default configs; asserts
  `n_heads % n_kv_heads == 0`.
- **MLA smoke + shape.** `MLAttention(cfg)(x, freqs_cis_mla, mask)` maps `(B, T, dim) →
  (B, T, dim)` with no NaN; uses the **separate** MLA RoPE buffer sized to
  `qk_rope_head_dim` (verify it does *not* read the GQA-sized `freqs_cis`).
- **Fast-path / fallback parity.** With and without the flash/mem-efficient SDPA backend
  (toggle `torch.backends.cuda.sdp_kernel`), GQA outputs agree to within `1e-2` (bf16
  tolerance). If `_HAS_FLASH_ATTN` and an Ampere GPU is available, `flash_attn_func`
  path agrees with the SDPA path to the same tolerance. On T4, the test exercises the
  SDPA flash backend (not a native FA2 kernel) and the manual fallback.
- **GQA KV-cache correctness.** Feeding a length-`T` sequence in one shot vs feeding it
  token-by-token through a shared `kv_cache` dict produces the same final-token output
  to within `1e-3`. Cache stores `{cache_key: {"k", "v"}}`, concatenated along seq dim,
  `.detach()`ed.
- **MLA cache is compressed.** `MLAttention` caches `{"c_kv": (B,S,kv_lora_rank),
  "k_rope": (B,S,H,qk_rope_head_dim)}` — assert `c_kv.shape[-1] == kv_lora_rank` and
  that **MLA cache bytes < GQA cache bytes** for the same `(B, S)` (the headline MLA win;
  log the ratio).
- **Mask dtype guard.** An additive causal mask whose dtype matches activation dtype is
  required; a test that passes an fp32 mask against fp16/bf16 activations documents/asserts
  the upcast hazard in the fallback path.

**Estimated effort.** ~2–3 days. GQA is ~half a day; MLA's two-path Q/KV reconstruction,
decoupled RoPE, and compressed cache are the time sink. FA2-on-T4 realism (SDPA backend
selection + fallback) adds debugging time.

**Dependencies.** Phase 1 (`OuroborosConfig`, `RMSNorm` used inside MLA's `q_norm`/
`kv_norm`, `apply_rope`, `precompute_rope_freqs`).

---

## Phase 3 — MoE FFN

**Goal.** Build the fine-grained Mixture-of-Experts FFN with routed + shared experts and
**aux-loss-free** load balancing — including the bias *update step* that reference
implementations leave as a stub (the Ouroboros completion).

**Components built** (exact names):
- `Expert` — `moe.py` (SwiGLU: `down(silu(gate(x)) * up(x))`, all `bias=False`; reused as
  the dense Prelude/Coda FFN with `expert_dim = dim * 4 // 3`).
- `MoEFFN` — `moe.py` (router `Linear(dim, n_experts, bias=False)`; `router_bias` buffer;
  `n_experts` routed `Expert`s + `n_shared_experts` shared `Expert`s of width
  `expert_dim * n_experts_per_tok`; `forward`; `update_router_bias`).

**Acceptance criteria** (concrete):
- **Expert smoke + shape.** `Expert(dim, expert_dim)(x)` maps `(..., dim) → (..., dim)`,
  no NaN; SwiGLU formula verified against a reference.
- **MoEFFN smoke + shape.** `MoEFFN(cfg)(x)` maps `(B, T, dim) → (B, T, dim)`, no NaN.
- **Routing diversity.** For a batch of distinct random tokens, `topk` selections are not
  all identical across tokens — different tokens route to different experts (assert the
  set of selected expert ids has size > 1 over a reasonable batch).
- **Top-K gate renormalization.** The top-K gating weights for each token sum to 1.0
  (within `1e-5`).
- **Shared experts always fire.** Zeroing all routed-expert outputs (e.g. forcing empty
  selection) still yields a nonzero output equal to the summed shared-expert contribution;
  shared experts contribute for every token.
- **`router_bias` is a buffer, not a parameter.** Assert `"router_bias"` appears in
  `module.buffers()` and **not** in `module.parameters()`, and that it never receives a
  gradient (the bias enters only `topk(logits + router_bias)` selection; gating weights
  come from the *unbiased* `softmax(logits)`).
- **`update_router_bias` moves load toward balance.** Construct an adversarial routing
  distribution that overloads expert 0 and starves expert `k`; run forward to populate the
  `expert_load` buffer, call `update_router_bias()`, and assert the bias moved **down** for
  overloaded experts and **up** for underloaded ones by `router_bias_update_rate *
  sign(load - mean_load)`. Over repeated steps, per-expert load variance decreases.

**Estimated effort.** ~2 days. The naive token-scatter dispatch is an
`O(topk · n_experts)` masked loop (correct but slow — note the grouped-gather
optimization as future work). The subtle parts are the unbiased-gate / biased-selection
split and the load-tracking + update step.

**Dependencies.** Phase 1 (`OuroborosConfig`). Independent of Phase 2.

---

## Phase 4 — Recurrence machinery

**Goal.** Build the heart of Ouroboros: stable input injection, the per-depth
loop-index signal, the composed transformer block, and the fixed-depth recurrent loop
that ties them together. This phase delivers the stability claim (resume bullet 2) at
the component level.

**Components built** (exact names, in dependency order):
- `loop_index_embedding` — `recurrence.py` (sinusoidal depth signal added to the first
  `loop_dim` channels).
- `LTIInjection` — `recurrence.py` (`log_A`, `log_dt`, `B`; `get_A()`; `forward` →
  `A·h + B·e + transformer_out`).
- `TransformerBlock` — `block.py` (pre-norm; attention from `attn_type`; FFN = `MoEFFN`
  if `use_moe` else dense `Expert`; two `RMSNorm`s; residual + dropout).
- `RecurrentBlock` — `recurrence.py` (owns `block` (use_moe=True), `injection`, `norm`,
  `loop_dim = loop_index_dim or dim // 8`; runs the fixed-depth loop per spec §3).

**Acceptance criteria** (concrete):
- **Spectral radius < 1 by construction (the headline stability check).**
  `ρ(LTIInjection.get_A()) = max(get_A()) < 1` for random init **and** after an
  adversarial optimizer step — e.g. **`ρ(get_A()) < 1` even after a 1e3-LR SGD step**
  on `log_A`/`log_dt`. Also assert all entries of `get_A()` lie strictly in `(0, 1)` and
  that no NaN appears even when `log_dt → +∞, log_A → +∞` (the `(-20, 20)` log-space
  clamp must hold).
- **More loops change the output.** For the same input, `RecurrentBlock` outputs at
  `n_loops=2` vs `n_loops=8` differ measurably (relative L2 difference above a small
  threshold) — depth actually does work.
- **Loop-index differs per iteration.** `loop_index_embedding(h, t=0, ...)` ≠
  `loop_index_embedding(h, t=3, ...)` on the first `loop_dim` channels, and the remaining
  `dim - loop_dim` channels are **unchanged** (passthrough verified exactly).
- **Depth extrapolation runs.** `RecurrentBlock.forward(..., n_loops=16)` for a block
  whose `max_loop_iters=8` runs without error and stays finite — the sinusoidal
  loop-index is defined for any depth, so there is no learned per-loop table to index out
  of range.
- **TransformerBlock parity with `use_moe`.** `TransformerBlock(cfg, use_moe=False)` uses
  a dense `Expert(dim, dim*4//3)`; `use_moe=True` uses `MoEFFN`; both map
  `(B, T, dim) → (B, T, dim)` with no NaN for GQA and MLA `attn_type`.
- **Fixed-depth loop populates all cache keys.** With a non-None `kv_cache`, a forward
  runs every loop depth and populates cache keys `recurrent_loop_{0..n_loops-1}`, so the
  standard decode path (Phase 5) and depth-wise batching (Phase 7) inherit a fully
  populated cache with no early-exit edge case.

**Estimated effort.** ~2–3 days. `LTIInjection`'s log-space ZOH math and the dual-RoPE /
per-depth cache-key wiring in `RecurrentBlock` are the subtle parts; getting them right
here means Phase 5/7 inherit a correct loop.

**Dependencies.** Phase 1 (`RMSNorm`, `OuroborosConfig`), Phase 2 (attention inside
`TransformerBlock`), Phase 3 (`MoEFFN` and `Expert` inside `TransformerBlock`).

---

## Phase 5 — Full model + generation

**Goal.** Assemble the end-to-end `Ouroboros` model (Embedding → Prelude → Recurrent
Block → Coda → tied head) and implement KV-cached autoregressive generation. This
completes resume bullet 1 — a runnable from-scratch recurrent-depth transformer.

**Components built** (exact names):
- `Ouroboros` — `model.py` (`__init__` builds `embed`, dual RoPE buffers `freqs_cis`
  (size `dim//n_heads`) and `freqs_cis_mla` (size `qk_rope_head_dim`), `prelude`
  `ModuleList`, `recurrent` `RecurrentBlock`, `coda` `ModuleList`, final `RMSNorm`,
  tied `head`).
- `Ouroboros._init_weights` — `N(0, init_std)` on Linear & Embedding.
- `Ouroboros._causal_mask` — `(1, 1, S, S)` additive, `-inf` above diagonal, dtype-matched.
- `Ouroboros.forward(input_ids, n_loops, kv_cache, start_pos)`.
- `Ouroboros.generate(input_ids, max_new_tokens, n_loops, temperature, top_k)`.
- KV cache contract (a plain in-place `dict` with deterministic per-layer/per-loop keys:
  `prelude_{i}`, `recurrent_loop_{t}`, `coda_{i}`).
- `Ouroboros.generate_depthwise_batched` is *declared* here as a signature stub; its
  implementation is Phase 7.

**Acceptance criteria** (concrete):
- **Forward shape + no NaN, both attention types.** `Ouroboros(cfg).forward(input_ids)`
  returns `(B, T, vocab_size)` with no NaN for `attn_type="gqa"` **and** `attn_type="mla"`
  (tiny and default configs).
- **Weight tying.** `model.head.weight is model.embed.weight` (same storage); a gradient
  step updates both views consistently.
- **Dual RoPE selection.** With `attn_type="mla"`, `forward` slices `freqs_cis_mla`;
  with `"gqa"`, it slices `freqs_cis` — assert the right buffer is used and that it is
  sliced `[start_pos : start_pos + T]`.
- **Cached-decode logits ≈ full-context.** For a fixed sequence, the per-step
  KV-cached decode logits match a single full-context forward pass to within `1e-3`
  (the core correctness invariant for generation; depends on the Phase 4 cache-population
  rule and on `start_pos` RoPE slicing).
- **`start_pos` matters.** Running incremental decode *without* the `start_pos` offset
  (forcing position-0 rotations) measurably degrades the match above — a guard test that
  documents why `start_pos` is required.
- **Generate shape.** `generate(input_ids, max_new_tokens=k)` returns
  `(B, T + k)` token ids; `top_k` and `temperature` paths run without error.
- **Single-token (T=1) path.** A forward with `T == 1` uses `mask = None` (no causal
  mask) and runs cleanly — the decode step.
- **Depth extrapolation changes output.** `generate(..., n_loops=16)` differs from
  `n_loops=8` for a model whose `max_loop_iters=8` (runs deeper than training without
  error — the sinusoidal loop-index is defined at any depth — and the output changes).

**Estimated effort.** ~2 days. Most components exist by now; the work is wiring,
weight tying, the dual-RoPE selection, and getting cached decode bit-exact against full
context (the `start_pos` slicing and the Phase-4 cache-population rule are where bugs hide).

**Dependencies.** Phases 1–4 (everything the model composes).

---

## Phase 6 — Training

**Goal.** Train a small (~10–30M param) Ouroboros on T4 and **demonstrate the stability
claim empirically**: an LTI-constrained run converges cleanly at a high learning rate
while an otherwise-identical run *without* LTI injection diverges. Wire the MoE bias
update into the loop.

**Components built** (exact names):
- `training/train.py` — single-GPU training script: `argparse` config, data loading
  (WikiText-103 or a FineWeb-Edu slice with an 8192-vocab BPE tokenizer), AdamW, cosine
  schedule with warmup, fp16 + `GradScaler` (T4) / bf16 fallback, gradient clipping, and
  a per-step call to `MoEFFN.update_router_bias()` across the model.
- W&B logging of: loss, **`ρ(A) = max(model.recurrent.injection.get_A())`**, gradient
  norm, tokens/s, and learning rate.
- An ablation switch to run **LTI vs no-LTI** (the no-LTI variant replaces the stable
  update with a naive `h = transformer_out + e` style injection) for the stability
  experiment.

**Acceptance criteria** (concrete):
- **Loss decreases + stable convergence.** On the tiny dataset, training loss decreases
  monotonically-in-trend over a short run; no NaN/Inf in loss or grads.
- **`ρ(A)` stays < 1 throughout.** The logged spectral radius `ρ(A)` remains strictly
  below 1 at **every** logged step of the LTI run, even at high LR (sweep
  `lr ∈ {3e-4, 1e-3, 3e-3}`).
- **No-LTI run destabilizes at high LR.** The ablation (no LTI) **diverges** — loss
  blows up / NaNs / `ρ(A) → ≥ 1` — at a learning rate where the LTI run stays stable.
  This contrast is the evidence for resume bullet 2 and feeds
  [`EXPERIMENTS.md`](./EXPERIMENTS.md) experiment 1.
- **MoE balance improves over training.** Per-expert load variance trends down as
  `update_router_bias()` runs each step (logged), confirming the Ouroboros completion is
  active.
- **Smoke before scale.** The loop runs end-to-end on a tiny batch / few steps without
  error before any long run is launched; config + hyperparameters are logged to W&B with
  the run, not just in comments.
- **Throughput sanity.** tokens/s logged and plausible for a ~10–30M model on T4 FP16.

**Estimated effort.** ~3–4 days focused engineering (plus separate wall-clock for the
runs). Data/tokenizer plumbing, fp16+GradScaler stability on T4, and producing a clean
LTI-vs-no-LTI divergence plot are the time sinks.

**Dependencies.** Phase 5 (a working, generation-capable model to train).

---

## Phase 7 — Inference optimization

**Goal.** Produce the inference multiplier for resume bullet 3: INT8 post-training
quantization, a cleaned-up FA2/SDPA path, and **continuous depth-wise batching**, each
benchmarked against an un-optimized baseline.

**Components built** (exact names):
- `quantize_int8(model, method)` — `quantize.py` (per-channel weight quantization of the
  large Linears — attention projections, expert FFNs — keeping norms/router/LM head in
  higher precision; `torch.ao.quantization` dynamic/static or `bitsandbytes` Int8).
- `INT8Linear` — `quantize.py`.
- `calibrate(model, calibration_loader)` — `quantize.py` (static PTQ).
- `quantization_error(fp_model, int8_model, eval_loader)` — `quantize.py` (perplexity
  delta + related metrics).
- `Ouroboros.generate_depthwise_batched(input_ids, max_new_tokens, max_loops,
  convergence_tol, temperature, top_k)` — `model.py` (the inference differentiator;
  per-sequence convergence-based early exit, solving the cache-key population problem it
  creates).
- `benchmarks/throughput.py` — prefill/decode latency, depth sweep, INT8 on/off, and
  depth-wise batching on/off; reports the end-to-end multiplier.

**Acceptance criteria** (concrete):
- **INT8 perplexity delta measured & acceptable.** `quantization_error(...)` reports the
  perplexity of fp16 vs INT8 on a held-out slice; the delta is small (target: within a
  few percent of fp16 ppl) and documented. INT8 model memory footprint is measurably
  smaller than fp16.
- **INT8 correctness.** `INT8Linear` output matches its fp `nn.Linear` source within the
  expected quantization tolerance on random inputs; norms/router/LM head remain unquantized
  (asserted).
- **Depth-wise batching gives a measured throughput gain.** With a batch whose sequences
  converge at different depths, `generate_depthwise_batched` produces **higher tokens/s**
  than the naive `generate` that pays max depth for every sequence — and produces
  **identical or equivalent** generations (correctness preserved). The gain is tied to the
  measured convergence-depth distribution (theory predicts ~2–3×).
- **Cache-key population solved.** The chosen solution to the
  "sequence exits at depth `d` leaves `recurrent_loop_{d..n}` unpopulated" problem
  (run-to-max-active-depth-with-masking, ragged/compacted cache, or depth-bucketing) is
  implemented and validated: a later decode step that loops deeper never reads a missing
  cache key.
- **FA2/SDPA path clean on T4.** The SDPA flash/mem-efficient backend is used where
  available with a working manual fallback; `flash_attn_func` remains an optional Ampere
  fast path. Benchmarked honestly (no claim of a native FA2 kernel on Turing).
- **End-to-end multiplier reported.** `benchmarks/throughput.py` emits a single
  **[X]× throughput** number (optimized vs un-optimized baseline on one GPU) that fills
  resume bullet 3 and [`EXPERIMENTS.md`](./EXPERIMENTS.md) experiments 4–5.

**Estimated effort.** ~3–4 days focused engineering (plus wall-clock for benchmark
sweeps). Depth-wise batching's cache-key bookkeeping is the hardest part of the whole
project after the Phase-4 loop; INT8 backend selection and honest benchmarking add time.

**Dependencies.** Phase 5 (model + KV-cache + the `generate_depthwise_batched` stub).
Phase 6 is a soft dependency: trained weights and the fp16 perplexity baseline make the
INT8 and throughput comparisons meaningful (you can prototype on random weights, but the
reported numbers should come from a trained model).

---

## Phase 8 — Polish

**Goal.** Make the repository a resume centerpiece: complete documentation, a clean
benchmark table with the measured multiplier, and a green, lint-clean codebase.

**Components built / finalized:**
- `README.md` — reflects the current README, professional, covers the scope.
  the canonical forward-pass ASCII diagram (shared with `ARCHITECTURE.md`), and links to
  every doc in `docs/`.
- Architecture diagram + W&B curve references (stability plot, loss curves,
  convergence-depth distribution) embedded/linked.
- Benchmark table populated with real Phase 6–7 numbers (perplexity, KV-cache memory
  MLA vs GQA, INT8 ppl delta, depth-wise batching throughput, end-to-end **[X]×**).
- Final pass on docstrings/type hints across all modules; `EXPERIMENTS.md`,
  `DESIGN_DECISIONS.md`, `READING_LIST.md`, `ARCHITECTURE.md` cross-checked for
  consistency with the shipped code.

**Acceptance criteria** (concrete):
- **Tests green.** `pytest` passes with zero failures across `tests/test_components.py`
  and `tests/test_model.py` (all real assertions implemented; no remaining `pytest.skip`
  stubs).
- **Lint/format/type clean.** `ruff check .`, `black --check .` (line length 88), and
  `mypy` all pass with no errors.
- **Docs self-contained.** A reader can understand and re-implement Ouroboros from the
  docs without consulting any external reference; every config field / signature named in
  a doc matches the code identically (and vice-versa).
- **Resume numbers filled in.** The `[X]×` throughput multiplier and the stability
  evidence are present, sourced from real runs, and consistent between README,
  `EXPERIMENTS.md`, and the benchmark table.
- **Reproducibility.** Seeds, configs, and hyperparameters for the headline runs are
  recorded (W&B + config files), and the commands to reproduce them are in the README.

**Estimated effort.** ~2 days.

**Dependencies.** Phases 1–7 (this phase aggregates and presents everything built).
