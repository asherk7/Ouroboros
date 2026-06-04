# Ouroboros — Design Decisions

## Purpose of this document

Ouroboros is an honest, from-scratch PyTorch implementation of a recurrent-depth
(looped) transformer, motivated by published research rather than forked from any
single codebase. This document records *why* each non-obvious architectural choice
was made, so the design can be defended end-to-end.

There are two distinct defensibility standards at play, and each decision below is
written against the one that applies to it:

- **Adopted components** (LTI injection, MLA/GQA, MoE, SwiGLU, aux-loss-free
  balancing) are defended on *depth of understanding*. For each
  I state: what problem it solves, the concrete alternative, why this choice is
  better here, and what I would change with more time or compute. The bar is being
  able to answer "why this and not that?" without hand-waving.
- **Novel / engineering-completion components** (the aux-loss-free bias *update
  step* that reference implementations leave stubbed, INT8 post-training
  quantization, and continuous depth-wise batching) are defended on *documented
  reasoning plus measured tradeoffs*. The bar is a written rationale today and a
  number from an `EXPERIMENTS.md` run later.

Every decision carries an **Evidence** line. These are deliberate placeholders tied
to a specific run in `EXPERIMENTS.md`, to be filled with real measurements during
Phase 6 (training) and Phase 7 (inference optimization). An empty evidence line is a
TODO, not a claim. The configuration field names used throughout (`dim`,
`n_loops`, `attn_type`, `n_experts_per_tok`,
`router_bias_update_rate`, etc.) are the exact fields declared in
`ouroboros/config.py` (`OuroborosConfig`).

Each decision follows the same five-part structure: **The decision**,
**Alternatives considered**, **Why this approach**, **What I'd change with more
time/compute**, and **Evidence**.

---

## 1. Prelude / Recurrent / Coda over a fully-looped architecture

**The decision.** Split the model into three functional stages: a `prelude` of
`prelude_layers` dense `TransformerBlock`s run once, a single shared
`RecurrentBlock` looped up to `max_loop_iters` times, and a `coda` of `coda_layers`
dense blocks run once. Only the recurrent block has tied weights reused across
depth; the prelude and coda have their own distinct weights.

**Alternatives considered.**
- *Fully-looped* — a single block (or the entire model) looped from token
  embeddings straight to the LM head, as in the purest Universal Transformer
  framing.
- *Pure depth-unrolled* — N distinct blocks with no weight sharing, i.e. an ordinary
  deep transformer (the looped structure collapses to a no-op).

**Why this approach.** A fully-looped stack forces one shared parameter set to do
three different jobs: surface-level encoding of raw token embeddings, iterative
refinement of an abstract latent, and final decoding into vocabulary logits. Those
jobs have conflicting requirements, and tying them degrades all three. The
prelude/recurrent/coda split (the Parcae-style recurrent-depth design) lets the
*non-shared* prelude lift tokens into a clean latent once, hands that latent to the
shared recurrent core for the part that actually benefits from arbitrary depth, and
uses a *non-shared* coda to read the converged latent out. Concretely, the
prelude output `e` is frozen and re-injected at every loop step (`h = A·h + B·e +
Transformer(h, e)`), which is only coherent if `e` lives in a stable encoded space
that the loop does not have to also produce — exactly what a dedicated prelude
gives. It also keeps parameter count down (the loop is reused) while preserving the
expressiveness that matters at the boundaries.

**What I'd change with more time/compute.** Ablate the `prelude_layers` /
`coda_layers` split at a fixed total parameter budget — is 2/2 optimal, or does
1-prelude/3-coda (more read-out capacity) win? I'd also test whether the prelude and
coda should share an attention KV space with the recurrent block, and whether a
second, shallower recurrent block (a two-scale loop) helps on reasoning-style data.

**Evidence.** TBD — depth-extrapolation run (`EXPERIMENTS.md` §6, loop-count sweep):
expect a model trained at `n_loops=8` to retain or improve validation perplexity at
`n_loops ∈ {4, 8, 16}`, which is only sensible if the recurrent core is isolated
from encode/decode by the prelude/coda. A fully-looped baseline is expected to
extrapolate worse.

---

## 2. LTI-constrained injection over gradient-clipping / LayerNorm-on-hidden-state band-aids

**The decision.** Stabilize the recurrent update with a Linear Time-Invariant
constraint (`LTIInjection`): the per-channel diagonal state matrix `A` is built so
that its spectral radius is **strictly < 1 by construction**, via a ZOH-style
discretization computed in log space —
`A = exp(-exp((log_dt + log_A).clamp(-20, 20)))`, giving every channel a value in
`(0, 1)`. The update is `h_{t+1} = A·h_t + B·e + Transformer(h_t, e)`.

**Alternatives considered.**
- *Gradient clipping* — clip the global grad norm and hope the recurrence does not
  blow up across loop iterations.
- *LayerNorm / RMSNorm on the hidden state every loop* — renormalize `h` after each
  iteration to bound its magnitude.
- *Unconstrained learned `A`* — let `A` be a free parameter and rely on the
  optimizer to keep it contractive.

**Why this approach.** Across `T` loop iterations the hidden state evolves like a
linear dynamical system `h_t ≈ A^t · h_0 + (injection terms)`. If `ρ(A) ≥ 1`, that
`A^t` term grows without bound and the state explodes — and crucially the *gradient*
through the loop explodes with it, which clipping only masks step-to-step while the
underlying dynamics stay unstable. Renormalizing `h` each loop is a band-aid that
fights the symptom: it distorts the very signal the loop is trying to refine and
masks the instability rather than removing it. The LTI constraint removes the
failure mode at the source — `ρ(A) < 1` is *guaranteed* regardless of what the
optimizer does, because the exponential parameterization cannot produce a
non-contractive `A`. That is what enables clean convergence at high learning rate
with no clipping or normalization crutch (resume claim #2). The log-space clamp at
`(-20, 20)` is essential: it prevents the `0 · inf = NaN` that arises when `log_dt →
-∞` and `log_A → +∞` under an aggressive step, keeping the whole thing fp32-robust.

**What I'd change with more time/compute.** Move beyond a diagonal `A` to a
diagonal-plus-low-rank or block-diagonal state matrix (richer dynamics while keeping
the spectral-radius guarantee cheap to enforce). I'd also log the full eigenvalue
*distribution* of `A`, not just `ρ(A) = max(get_A())`, to see whether channels
collapse to a single timescale, and ablate whether learning `B` per-channel
(current) versus a scalar materially changes injection strength.

**Evidence.** TBD — stability run (`EXPERIMENTS.md` §1, LTI vs no-LTI): expect the
unconstrained variant to diverge (`ρ(A) → ≥ 1`, loss → NaN) at `lr > 3e-4`, while
the LTI variant converges cleanly across the LR sweep `{3e-4, 1e-3, 3e-3}`. `ρ(A)`
is logged every step to W&B as the headline stability signal.

---

## 3. MoE in the recurrent block but dense FFN in prelude/coda

**The decision.** Use a fine-grained `MoEFFN` (routed + shared experts) **only**
inside the looped `RecurrentBlock`; the prelude and coda use a dense SwiGLU `Expert`
with hidden width `dim * 4 // 3`. The `TransformerBlock` toggles this with its
`use_moe` flag (`True` only when constructed by `RecurrentBlock`).

**Alternatives considered.**
- *MoE everywhere* — every block (prelude, recurrent, coda) is sparse.
- *Dense everywhere* — no MoE at all; a single dense FFN throughout.

**Why this approach.** The recurrent block is reused at every loop depth, so its
parameters carry the heaviest representational burden in the model — that is exactly
where added *capacity without added FLOPs* pays off, and MoE buys capacity by
activating only `n_experts_per_tok` of `n_experts` routed experts per token. The
prelude and coda run once each; making them sparse adds router overhead, a
load-balancing concern, and dispatch complexity for a single pass that is already
cheap relative to `T` loop iterations. Keeping the boundaries dense also keeps `e`
(the injected encoding) deterministic per token — no routing noise in the signal
that every loop iteration depends on. The shared expert (width `expert_dim *
n_experts_per_tok`) always fires and absorbs common cross-domain structure, so the
routed experts specialize rather than redundantly relearning syntax.

**What I'd change with more time/compute.** Try MoE in the coda only (read-out may
benefit from specialization) at matched FLOPs, and sweep `n_experts` /
`n_experts_per_tok` against the fine-grained rule of thumb `expert_dim ≈ dim //
(n_experts // n_experts_per_tok)`. With more compute I'd measure expert-utilization
entropy across loop depth — do different experts dominate at different loop
iterations, validating that the loop index genuinely changes routing behavior?

**Evidence.** TBD — MoE-vs-dense run (`EXPERIMENTS.md` §3, matched parameter
budget): expect sparse MoE in the recurrent block to match or beat a dense FFN of
equal *total* parameters at lower activated FLOPs per token. Pair with the
expert-routing diversity check (different tokens reach different experts).

---

## 4. Switchable MLA / GQA rather than committing to one attention mechanism

**The decision.** Support both attention mechanisms behind a single
`attn_type: str = "gqa"` config switch. `GQAttention` (default) shares
`n_kv_heads` key/value heads across `n_heads` query heads; `MLAttention` caches a
compressed KV latent `c_kv` of width `kv_lora_rank` plus a decoupled RoPE key. The
`TransformerBlock` picks the class at construction; the model precomputes **two**
RoPE buffers — `freqs_cis` sized `dim // n_heads` for GQA and `freqs_cis_mla` sized
`qk_rope_head_dim` for MLA — and selects per `attn_type` at forward.

**Alternatives considered.**
- *GQA only* — simplest, has the `flash_attn_func` / SDPA-flash fast path, smaller
  code surface.
- *MLA only* — smallest KV cache (caches `c_kv` instead of full K/V), best decode
  memory, but more moving parts and no clean flash fast path on T4.
- *Full multi-head attention* — no KV sharing or compression; the most memory-hungry
  and the weakest baseline at this scale.

**Why this approach.** These two mechanisms optimize different axes, and a portfolio
project that backs resume claim #1 ("switchable MLA/GQA attention") should be able
to *demonstrate* the tradeoff rather than assert it. GQA is the pragmatic default on
a T4: it is simple, and it has a real fast path (`flash_attn_func` when present,
otherwise `F.scaled_dot_product_attention` with the flash / mem-efficient backend).
MLA shrinks the KV cache dramatically by caching a low-rank latent and reconstructing
`k_nope`/`v` on the fly — the better choice when decode memory dominates. Making
them switchable behind one config flag (and dual RoPE buffers) turns "which
attention?" into a measured ablation instead of a guess, and keeps both code paths
honest because both must pass the same shape/no-NaN/cache-correctness tests.

**What I'd change with more time/compute.** Implement the MLA "absorb" trick
(folding `kv_up` into the query projection) so MLA never materializes full per-head
K, closing most of its throughput gap with GQA. I'd also benchmark MLA's compressed
cache at long context where its memory advantage is largest, and try a per-layer
attention type (GQA in prelude/coda for speed, MLA in the recurrent block for cache
size across loop depth).

**Evidence.** TBD — attention ablation (`EXPERIMENTS.md` §2, MLA vs GQA): same
config; expect MLA cache bytes < GQA cache bytes (a correctness check on the
compressed cache) with comparable perplexity, and GQA ahead on raw decode throughput
on T4 owing to the flash fast path.

---

## 5. Fixed loop count over ACT (adaptive halting)

**The decision.** Run the recurrent block for a **fixed** `n_loops` iterations for
every position and take the final hidden state. There is no learned halting head and
no per-position adaptive computation time — the loop body is simply
`loop_index_embedding → TransformerBlock → LTIInjection`, repeated `n_loops` times.
(`n_loops` defaults to `max_loop_iters` and can be raised at inference for depth
extrapolation.)

**Alternatives considered.**
- *Adaptive Computation Time (ACT)* — a learned `sigmoid(Linear(dim, 1))` halting
  head per loop, accumulating an ACT-weighted sum of hidden states with the
  remainder trick so each position halts independently (Graves 2016; Universal
  Transformers).
- *Global convergence early-stop during training* — halt the whole batch when the
  mean state delta falls below a threshold.

**Why this approach.** ACT is elegant, but at this project's scope its cost
outweighs its benefit. It adds a learned halting head, the cumulative-probability
remainder bookkeeping, a halting-threshold hyperparameter, and — most importantly — a
sharp correctness subtlety: under a KV cache the per-position early-exit must be
disabled so every loop depth still runs (otherwise later decode steps read
unpopulated `recurrent_loop_{t}` keys). A fixed loop count removes all of that: the
core loop is trivial to reason about, the KV cache is always fully populated on every
forward, and training is a clean `n_loops`-step unroll with no halting gradient to
tune. The adaptive-compute *throughput* idea is not lost — it is recovered at
inference, and more cheaply, by continuous depth-wise batching (decision 10), which
exits a sequence once its hidden state stops changing
(`‖h_{t+1} − h_t‖ < convergence_tol`) with no learned head. So Ouroboros keeps the
win (variable depth per sequence at inference) while dropping the training-time
complexity. This is a deliberate scope cut: the resume claims are LTI-stable looped
training and the inference optimizations, not adaptive halting.

**What I'd change with more time/compute.** Revisit ACT once the fixed-depth model
trains cleanly: add a learned halting head *for inference only* (train at fixed depth,
halt adaptively at test time) and compare its quality/throughput against the
convergence-based early-exit. A ponder-cost-regularized ACT during training is the
fuller version if the fixed depth proves wasteful on easy tokens.

**Evidence.** TBD — loop-count sweep (`EXPERIMENTS.md` §6): expect a fixed-depth
model trained at `n_loops=8` to converge stably and retain quality across
`n_loops ∈ {4, 8, 16}` at inference (depth extrapolation), confirming a fixed loop
count is sufficient at this scope.

---

## 6. Sinusoidal loop-index embedding over learned per-loop embeddings

**The decision.** Signal the current loop iteration with a fixed sinusoidal
`loop_index_embedding(h, loop_t, loop_dim)` — RoPE-style sin/cos of `loop_t · θ_k`
added as a bias to only the first `loop_dim` channels (`loop_index_dim or dim // 8`),
leaving the remaining channels untouched.

**Alternatives considered.**
- *Learned per-loop embedding table* — an `Embedding(max_loop_iters, dim)` (or
  `loop_dim`) lookup added each iteration.
- *No loop signal* — shared weights with no indication of which iteration they are
  in.

**Why this approach.** Without any depth signal, the tied recurrent weights cannot
tell loop 0 from loop 7 and must behave identically — defeating the point of looping.
A *learned* per-loop table fixes that but is bounded to `max_loop_iters` entries: at
inference, depth extrapolation past the trained range has no embedding to use. A
deterministic sinusoid is defined for *any* `loop_t`, so the depth-extrapolation
property (loop deeper at test time) keeps working with zero special-casing, exactly
mirroring why sinusoidal/RoPE position encodings extrapolate over sequence length.
Restricting it to `loop_dim` channels keeps the signal a small, non-destructive
nudge rather than overwriting the latent. It is also parameter-free, which matters
at the T4 small-model scale where the embedding table already dominates parameters.

**What I'd change with more time/compute.** Sweep `loop_index_dim` (does `dim // 8`
carry enough depth signal, or does the loop need more channels?) and compare against
a learned table *within* the trained range to quantify what, if anything, learnable
loop embeddings buy before extrapolation breaks them. I'd also try injecting the loop
index multiplicatively (RoPE-style rotation over depth) rather than additively.

**Evidence.** TBD — loop-index ablation, folded into the depth-extrapolation run
(`EXPERIMENTS.md` §6): expect sinusoidal loop-index to extrapolate to `n_loops=16`
where a learned table (capped at the trained `max_loop_iters`) cannot, and a
per-iteration check that the block's output genuinely differs across loops.

---

## 7. SwiGLU over ReLU / GELU FFN

**The decision.** Every FFN — the dense prelude/coda `Expert` and each routed/shared
MoE expert — is a SwiGLU gated unit: `down(silu(gate(x)) * up(x))`, all `bias=False`.
Prelude/coda dense width is `dim * 4 // 3`; routed-expert width is `expert_dim`;
shared-expert width is `expert_dim * n_experts_per_tok`.

**Alternatives considered.**
- *ReLU / GELU MLP* — the classic two-matrix `down(act(up(x)))` feed-forward.
- *Plain GLU (sigmoid gate)* or other GLU variants (GEGLU, ReGLU).

**Why this approach.** SwiGLU's data-dependent multiplicative gate (`silu(gate(x)) *
up(x)`) consistently improves transformer quality per parameter over a plain
ReLU/GELU MLP (Shazeer 2020), and it is the de facto standard in modern LLaMA/DeepSeek
-class models — so it is both the strong default and the honest choice for a
literature-grounded build. The `silu` gate is smooth (no dead-unit issue), and the
three-matrix structure pairs naturally with MoE, where the same `Expert` class serves
as both the dense FFN and a routed expert. The `dim * 4 // 3` width is a deliberate,
documented parameter-budget choice: it is *smaller* than the common `8/3 · dim`
SwiGLU sizing, trading a little FFN capacity to keep the T4 small-model under budget
(where the embedding table already dominates).

**What I'd change with more time/compute.** Ablate the `dim * 4 // 3` width against
the standard `8/3 · dim` to quantify exactly what the budget cut costs, and run a
clean SwiGLU-vs-GELU comparison at matched parameters to put a number on the SwiGLU
gain at *this* scale (most published evidence is at larger scale). I'd also test
GEGLU as a cheap alternative gate.

**Evidence.** TBD — FFN-variant ablation (small add-on to `EXPERIMENTS.md` §3):
expect SwiGLU to edge out GELU at matched parameters, and a documented perplexity
cost (expected small) for the `4/3` vs `8/3` width choice. This is an adopted
component, so the bar is reasoning + a confirming number, not novelty.

---

## 8. Aux-loss-free load balancing over an auxiliary loss — *and* actually implementing the bias update

**The decision.** Balance MoE expert load with the DeepSeek-V3 aux-loss-free bias
trick instead of an auxiliary load-balancing loss: expert *selection* uses
`topk(logits + router_bias)`, but the gating *weights* come from the **unbiased**
`softmax(logits)` (renormalized over the top-K), so `router_bias` never enters the
gradient. `router_bias` is a non-gradient buffer. Crucially, Ouroboros **implements
the update step** that reference implementations leave stubbed: a per-expert
`expert_load` buffer tracks selections during forward, and
`update_router_bias()` nudges the bias toward balance each training step by
`router_bias_update_rate * sign(load - mean_load)` (down for overloaded experts, up
for underloaded ones). The training loop calls it every step.

**Alternatives considered.**
- *Auxiliary load-balancing loss* — add a balance term to the training objective
  (the classic Switch-Transformer / GShard approach).
- *No balancing* — let the router collapse onto a few experts (router collapse).
- *Aux-loss-free selection but never updating the bias* — register `router_bias` and
  leave it at zero (what naive reference implementations actually ship).

**Why this approach.** An auxiliary loss couples balancing to the main objective: its
gradient fights the language-modeling gradient, and its weight is a finicky
hyperparameter that trades perplexity for balance. The bias trick decouples the two —
balancing happens entirely through a non-gradient bias on *selection*, while gradients
flow only through unbiased gating weights, so the loss landscape is never distorted by
a balance penalty. That is cleaner and matches the strongest current practice. The
second half is the engineering-maturity point and the *novel-completion* part of this
decision: the trick only works if the bias is actually moved, and many reference
implementations register the buffer but never update it, so their "balancing" is
inert. Ouroboros closes that gap with a real `expert_load` accumulator and a working
`update_router_bias()` — a small but concrete, demonstrable win, and the kind of
detail an interviewer can probe.

**What I'd change with more time/compute.** Tune `router_bias_update_rate`
(`1e-3` default) and consider an adaptive rate (larger when imbalance is high). I'd
log per-expert utilization over training and, with more compute, compare the bias
trick head-to-head against a tuned auxiliary loss to quantify the perplexity-vs-balance
tradeoff. I'd also replace the O(topk · n_experts) masked-loop dispatch with a
grouped/batched-gather dispatch (see the cross-cutting note below) since dispatch
cost, not balancing, is the real bottleneck.

**Evidence.** TBD — load-balancing run (`EXPERIMENTS.md` §3, MoE diagnostics):
expect `update_router_bias()` to drive per-expert load toward uniform over training
(measure the max/mean load ratio falling), where the never-updated baseline stays
imbalanced. Defended on documented reasoning + this measured balance curve, per the
novel-component bar.

---

## 9. INT8 over FP16 / FP8 for inference quantization

**The decision.** Use **post-training INT8** quantization for inference
(`quantize_int8(model, method="dynamic")`, with a static/calibrated path via
`calibrate`). Per-channel weight quantization is applied to the large Linear layers
(attention projections, expert FFNs) while norms, the router, and the tied LM head
stay in higher precision. Quantization error (perplexity delta) and throughput are
measured with `quantization_error(fp_model, int8_model, eval_loader)`.

**Alternatives considered.**
- *Keep FP16* — the training/serving precision on T4; no quantization, no error, but
  no INT8 throughput/memory win either.
- *FP8* — newer low-precision format, but Turing (T4, sm75) has **no** FP8 tensor-core
  support; FP8 only pays off on Hopper+.
- *INT4* — smaller still, but materially larger accuracy loss and weaker, less
  portable kernel support at this scale.

**Why this approach.** The target hardware decides this. The T4 has **INT8 tensor
cores** (sm75) but no FP8 path, so INT8 is the precision that actually maps to
hardware acceleration on the deployment target — choosing FP8 would be hardware
fiction on a T4. INT8 PTQ on the big Linear layers cuts weight memory and lifts
decode throughput while keeping the numerically sensitive pieces (norms, router,
LM head) in higher precision to bound the perplexity hit. Post-training (not
quantization-aware) keeps the scope realistic for a portfolio project: no retraining,
just calibrate-and-measure. This is a *novel-component* decision (going beyond the
reference literature for resume bullet #3), so it is defended by a written rationale
now and a measured perplexity-vs-throughput tradeoff later.

**What I'd change with more time/compute.** Compare the realistic backends head-to-head
on T4 (`torch.ao.quantization` dynamic vs static, and `bitsandbytes` Int8) on both
perplexity delta and actual kernel throughput, since the win depends entirely on
kernel quality. With more compute I'd add a per-layer sensitivity analysis (which
layers tolerate INT8 and which must stay FP16) and explore INT8 *and* INT4 on the most
tolerant layers (mixed-precision PTQ). Quantization-aware fine-tuning would be the next
step if PTQ error proves too high.

**Evidence.** TBD — quantization run (`EXPERIMENTS.md` §4): report perplexity before
vs after INT8 (expect a small, acceptable delta) and the decode-throughput gain on T4.
This number feeds resume bullet #3 (the inference-throughput multiplier).

---

## 10. Continuous depth-wise batching — the inference differentiator

**The decision.** Implement `generate_depthwise_batched` so that sequences in one
batch may **exit the recurrent loop at different depths** via a non-learned
*convergence* criterion: a sequence stops looping once its hidden state stops
changing (`‖h_{t+1} − h_t‖ < convergence_tol`). Easy sequences converge early, hard
ones loop more, in the *same* batch, instead of every sequence paying max depth. This
is the headline inference optimization and the third resume bullet.

**Alternatives considered (for the cache-population problem this creates).** With a
KV cache, a sequence that converges and exits at depth `d` leaves cache keys
`recurrent_loop_{d..n}` unpopulated, so a later decode step that loops deeper would
read missing keys. Three ways to resolve it:
- *(a) Run to the max active depth, mask converged sequences* — every batch runs to
  the deepest still-active sequence; converged ones are masked out (their outputs
  frozen at their exit depth) but their cache keys still get populated.
- *(b) Per-sequence depth with a ragged / compacted cache* — track each sequence's
  depth and store a ragged cache, compacting as sequences finish.
- *(c) Sort / bucket by predicted depth* — group sequences so a batch shares a target
  depth.

**Why this approach.** This is the inference-time payoff that replaces ACT (decision
5): rather than a learned per-position halting head, a simple per-sequence
convergence test gives variable depth across a batch, so a batch no longer has to pay
worst-case depth for its hardest member — and it needs no training-time machinery.
The one hard constraint is the **KV-cache ⊗ early-exit interaction**: a sequence that
exits at depth `d` must not leave later `recurrent_loop_{t>d}` cache keys unpopulated
when a future step needs them. Solution **(a)** is the chosen default because it
respects that constraint directly and is simple and correct: run to the max active
depth and mask converged sequences, which keeps the cache dense and guarantees all
needed keys exist (the standard `recurrent_loop_{t}` keying and KV-cache path are
reused verbatim). **(b)** is the highest-throughput ceiling but the most complex and
bug-prone (ragged cache bookkeeping); **(c)** depends on predicting depth before
running and only helps when the depth distribution is multi-modal. We start correct
(a), then optimize.

**What I'd change with more time/compute.** Implement (b) (a compacted/ragged cache
that drops finished sequences) to capture the throughput (a) leaves on the table when
depths are very skewed, and add (c) as a scheduling layer (bucket by a cheap depth
predictor) on top. I'd tie the measured speedup explicitly to the convergence-depth
distribution — the win is large only when depths are spread out — and stress-test the
cache invariant with a test that cached depth-wise-batched decode logits match an
un-batched full-depth forward. I'd also sweep `convergence_tol` against the
quality/throughput tradeoff.

**Evidence.** TBD — depth-wise batching run (`EXPERIMENTS.md` §5): report throughput
with vs without convergence-based early-exit batching, tied to the convergence-depth
distribution. Literature suggests ~2–3×; the *measured* number on T4 fills resume
bullet #3. Defended on documented reasoning + this measured throughput gain, per the
novel-component bar.

---

## Cross-cutting notes (referenced by several decisions)

These are not standalone decisions but recurring engineering points surfaced above,
collected for quick reference:

- **MoE dispatch cost.** The current routed-expert dispatch is an O(topk · n_experts)
  masked Python loop — correct but slow. A grouped/batched-gather dispatch is the
  obvious next optimization and a benchmarkable item (referenced in decision 8).
- **FlashAttention-2 on T4 is forward-only and often will not build** (Turing sm75 is
  not an officially supported FA2 target). The realistic, robust fast path on T4 is
  `F.scaled_dot_product_attention` with the flash / mem-efficient backend
  (`torch.backends.cuda.sdp_kernel`) plus a manual fallback; `flash_attn_func` is kept
  as an optional fast path for when the package is present (e.g. a rented Ampere GPU).
  This is stated honestly wherever FA2 appears (decision 4, benchmarks) — no claim of a
  native custom FA2 kernel on T4.
- **Weight init ignores residual-depth scaling.** Naive `N(0, init_std)` on all
  weights does not account for variance accumulating across loop iterations; GPT-2-style
  `1/sqrt(2·n_eff)` scaling on residual output projections (and optionally QK-norm /
  logit z-loss) is flagged as a stability improvement to ablate (relevant to
  decisions 1 and 2).
- **`ρ(A)` logging.** Because `A` is diagonal, `ρ(A) = max(get_A())` is a cheap,
  continuous, per-step stability signal — logged to W&B every step and the centerpiece
  of the stability experiment (decision 2).
