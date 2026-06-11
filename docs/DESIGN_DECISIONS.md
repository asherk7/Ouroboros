# Ouroboros — Design Decisions

## Purpose of this document

Ouroboros is an honest, from-scratch PyTorch implementation of a recurrent-depth
(looped) transformer, motivated by published research rather than forked from any
single codebase. This document records *why* each non-obvious architectural choice
was made, so the design can be defended end-to-end.

There are two distinct defensibility standards at play, and each decision below is
written against the one that applies to it:

- **Adopted components** (LTI injection, GQA, MoE, SwiGLU, aux-loss-free
  balancing) are defended on *depth of understanding*. For each
  I state: what problem it solves, the concrete alternative, why this choice is
  better here, and what I would change with more time or compute. The bar is being
  able to answer "why this and not that?" without hand-waving.
- **Novel / engineering-completion components** (the aux-loss-free bias *update
  step* that reference implementations leave stubbed, and continuous depth-wise
  batching) are defended on *documented reasoning plus measured tradeoffs*. The
  bar is a written rationale today and a number from an `EXPERIMENTS.md` run
  later.

Every decision carries an **Evidence** line. These are deliberate placeholders tied
to a specific run in `EXPERIMENTS.md`, to be filled with real measurements during
Phase 6 (training) and Phase 7 (inference optimization). An empty evidence line is a
TODO, not a claim. The configuration field names used throughout (`dim`,
`n_loops`, `use_lti`, `n_experts_per_tok`,
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
prelude output `e` seeds the loop (`h_0 = e`) and is frozen and re-injected at
every loop step (`h = A·h + B·e + Transformer(h, e)`), which is only coherent if
`e` lives in a stable encoded space
that the loop does not have to also produce — exactly what a dedicated prelude
gives. It also keeps parameter count down (the loop is reused) while preserving the
expressiveness that matters at the boundaries.

**What I'd change with more time/compute.** Ablate the `prelude_layers` /
`coda_layers` split at a fixed total parameter budget — is 2/2 optimal, or does
1-prelude/3-coda (more read-out capacity) win? I'd also test whether the prelude and
coda should share an attention KV space with the recurrent block, and whether a
second, shallower recurrent block (a two-scale loop) helps on reasoning-style data.

**Evidence.** TBD — depth-extrapolation run (`EXPERIMENTS.md` §2, loop-count sweep):
expect a model trained at `n_loops=8` to retain or improve validation perplexity at
`n_loops ∈ {2, 4, 8, 16}`, which is only sensible if the recurrent core is isolated
from encode/decode by the prelude/coda. A fully-looped baseline is expected to
extrapolate worse.

---

## 2. LTI-constrained injection over gradient-clipping / LayerNorm-on-hidden-state band-aids

**The decision.** Stabilize the recurrent update with a Linear Time-Invariant
constraint (`LTIInjection`): the per-channel diagonal state matrix `A` is built so
that its spectral radius is **strictly < 1 by construction**, via a ZOH-style
discretization computed in log space —
`A = exp(-exp((log_dt + log_A).clamp(-20, 20)))`, giving every channel a value in
`(0, 1)`. The update is `h_{t+1} = A·h_t + B·e + Transformer(h_t, e)`. The
constraint is toggled by the `use_lti` config field (default `True`);
`use_lti=False` swaps in the naive residual injection `h = transformer_out + e`
used as the stability baseline.

**Alternatives considered.**
- *Gradient clipping* — clip the global grad norm and hope the recurrence does not
  blow up across loop iterations.
- *LayerNorm / RMSNorm on the hidden state every loop* — renormalize `h` after each
  iteration to bound its magnitude.
- *Unconstrained learned `A`* — let `A` be a free parameter and rely on the
  optimizer to keep it contractive.
- *Naive residual injection* — drop the state path entirely
  (`h = transformer_out + e`): no `A`, no `B`, nothing to constrain. This is the
  `use_lti=False` arm of the stability experiment.

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
with no clipping or normalization crutch (resume claim #2). Dropping the state
path entirely (the naive residual injection) removes the explicit `A^t` term but
leaves the loop with no contractive anchor at all — nothing damps depth-wise
drift — which is why it serves as the honest experimental baseline rather than a
competing stability mechanism. The log-space clamp at
`(-20, 20)` is essential: it prevents the `0 · inf = NaN` that arises when `log_dt →
-∞` and `log_A → +∞` under an aggressive step, keeping the whole thing fp32-robust.

**What I'd change with more time/compute.** Move beyond a diagonal `A` to a
diagonal-plus-low-rank or block-diagonal state matrix (richer dynamics while keeping
the spectral-radius guarantee cheap to enforce). I'd also log the full eigenvalue
*distribution* of `A`, not just `ρ(A) = max(get_A())`, to see whether channels
collapse to a single timescale, and ablate whether learning `B` per-channel
(current) versus a scalar materially changes injection strength.

**Evidence.** TBD — stability run (`EXPERIMENTS.md` §1, LTI vs no-LTI): expect the
`use_lti=False` arm (naive residual injection) to diverge (loss → NaN) at
`lr > 3e-4`, while the LTI variant converges cleanly across the LR sweep
`{3e-4, 1e-3, 3e-3}`. `ρ(A)` is logged every step to W&B as the headline stability
signal for the LTI arm (the no-LTI arm has no state matrix to log).

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

**What I'd change with more time/compute.** Run the matched-total-parameter
MoE-vs-dense ablation that was trimmed from the experiment plan (sparsity should
win at fixed total params; locally that claim currently rests on literature, not a
measured number). Try MoE in the coda only (read-out may
benefit from specialization) at matched FLOPs, and sweep `n_experts` /
`n_experts_per_tok` against the fine-grained rule of thumb `expert_dim ≈ dim //
(n_experts // n_experts_per_tok)`. With more compute I'd measure expert-utilization
entropy across loop depth — do different experts dominate at different loop
iterations, validating that the loop index genuinely changes routing behavior?

**Evidence.** TBD — MoE health is validated via the W&B training logs recorded on
**every** run (`EXPERIMENTS.md` §0.6): the per-expert load variance (the spread of
`expert_load`) staying low, the router bias visibly moving, and stable loss curves
with MoE active in the loop. There is no dedicated MoE-vs-dense experiment in the
trimmed three-experiment scope; the matched-budget ablation lives on the more-time
list above.

---

## 4. GQA only over switchable MLA/GQA attention (scope cut)

**The decision.** `GQAttention` is the **only** attention mechanism: `n_kv_heads`
key/value heads shared across `n_heads` query heads, one RoPE phasor table
(`freqs_cis`, sized `dim // n_heads`), one KV-cache layout. An earlier revision
planned MLA (DeepSeek-style compressed-KV latent attention) as a second mechanism
behind a config switch, with its own config fields and a second RoPE buffer; that
entire switch has been removed — class, config fields, dual buffers, and the
MLA-vs-GQA experiment.

**Alternatives considered.**
- *Switchable MLA/GQA* — the previous plan: both mechanisms behind one config flag,
  ablated head-to-head.
- *MLA only* — smallest KV cache (caches a low-rank latent instead of full per-head
  K/V), best decode memory, but more moving parts and no clean flash fast path on T4.
- *Full multi-head attention* — no KV sharing or compression; the most memory-hungry
  and the weakest baseline at this scale.

**Why this approach.** A second attention mechanism doubles the attention, cache,
RoPE, and test surface — two cache layouts, two phasor-table sizes, two sets of
shape/no-NaN/cache-correctness tests — without adding anything to the core
recurrence story (LTI-stable looped training, depth extrapolation, depth-wise
batched inference). And MLA's compressed-KV payoff is a long-context, large-model
story: at `max_seq_len=1024` and 10–30M parameters, GQA with `n_kv_heads=2` already
makes the KV cache a non-problem, so the ablation would have measured a tradeoff
this project does not need to win. GQA keeps the real fast path on the target
hardware (`F.scaled_dot_product_attention` with the flash / mem-efficient backend,
`flash_attn_func` when present). This is the same kind of deliberate scope cut as
ACT (decision 5): keep the surface area that serves the resume claims, cut the rest.

**What I'd change with more time/compute.** Reintroduce MLA only if the project
moves to long context, where the compressed cache pays — implemented with the
"absorb" trick from the start so it never materializes full per-head K, and
benchmarked against GQA at the context lengths where the memory gap is material.

**Evidence.** None owed — a scope cut, not a measured claim; no resume bullet
references MLA. The GQA path is validated by the shape/no-NaN/cache-correctness
tests and carries the decode numbers in the inference-throughput run
(`EXPERIMENTS.md` §3).

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
exits a sequence once its hidden state stops changing (relative change
`‖h_{t+1} − h_t‖_F / (‖h_t‖_F + 1e-8) < convergence_tol`) with no learned head. So
Ouroboros keeps the
win (variable depth per sequence at inference) while dropping the training-time
complexity. This is a deliberate scope cut: the resume claims are LTI-stable looped
training and the inference optimizations, not adaptive halting.

**What I'd change with more time/compute.** Revisit ACT once the fixed-depth model
trains cleanly: add a learned halting head *for inference only* (train at fixed depth,
halt adaptively at test time) and compare its quality/throughput against the
convergence-based early-exit. A ponder-cost-regularized ACT during training is the
fuller version if the fixed depth proves wasteful on easy tokens.

**Evidence.** TBD — loop-count sweep (`EXPERIMENTS.md` §2): expect a fixed-depth
model trained at `n_loops=8` to converge stably and retain quality across
`n_loops ∈ {2, 4, 8, 16}` at inference (depth extrapolation), confirming a fixed
loop count is sufficient at this scope.

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
(`EXPERIMENTS.md` §2): expect sinusoidal loop-index to extrapolate to `n_loops=16`
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

**Evidence.** Deferred — no FFN-variant ablation survives the trimmed
three-experiment scope. SwiGLU stands on literature strength (Shazeer 2020) and is
exercised by every training run; the matched-parameter SwiGLU-vs-GELU and `4/3` vs
`8/3` width comparisons live on the more-time list above. This is an adopted
component, so the bar is reasoning, not novelty.

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

**Evidence.** TBD — load-balancing diagnostics logged on **every** training run
(`EXPERIMENTS.md` §0.6): expect `update_router_bias()` to drive per-expert load
toward uniform over training (the per-expert load variance falling, the bias
visibly moving), where a never-updated bias stays imbalanced. There is no dedicated
MoE experiment in the trimmed scope; this is validated from the training logs of
the runs that exist. Defended on documented reasoning + this measured balance
curve, per the novel-component bar.

---

## 9. FP16 inference over INT8 quantization (scope cut)

**The decision.** Inference runs in the training precision, FP16. An earlier
revision planned post-training INT8 weight quantization of the large Linear layers
(dynamic and calibrated/static, with a third-party INT8 backend) as a headline
inference optimization; that has been cut entirely — module, dependency, and the
quantization experiment.

**Alternatives considered.**
- *Keep the INT8 plan* — per-channel weight PTQ on the big Linear layers, with
  norms, the router, and the tied LM head left in higher precision (the previous
  version of this decision).
- *FP8 / INT4* — already rejected before the cut: Turing (T4, sm75) has no FP8
  tensor-core path, and INT4 costs too much accuracy at this scale. Cutting INT8
  makes them doubly moot.

**Why this approach.** At 10–30M parameters the FP16 model is roughly **40 MB** —
INT8 has no honest memory or latency story at this scale. Nothing meaningful is
saved on a 16 GB card, and weight-only INT8 in pure PyTorch typically *slows*
decode (per-step dequantization in unfused kernels) unless backed by fused INT8
GEMMs that only pay off at much larger matrices. A quantization number measured
here would be resume theater, not engineering. The inference story is instead
carried entirely by mechanisms that do have an honest story at this scale:
KV-cached decode, the SDPA flash / mem-efficient backend, and continuous
depth-wise batching (decision 10) — with the headline [X]× defined as
batched-convergence decode vs naive fixed-depth decode.

**What I'd change with more time/compute.** Revisit quantization only at a scale
where it stops being fiction — when weight memory or memory bandwidth actually
binds (hundreds of MB of weights, long-context serving) — and then with fused INT8
kernels, measured end-to-end against the FP16 baseline.

**Evidence.** None owed — a scope cut, not a measured claim. The
inference-throughput claim now rests solely on the inference-throughput run
(`EXPERIMENTS.md` §3); no quantization number is claimed anywhere.

---

## 10. Continuous depth-wise batching — the inference differentiator

**The decision.** Implement `generate_depthwise_batched` so that sequences in one
batch may **exit the recurrent loop at different depths** via a non-learned
*convergence* criterion: a sequence stops looping once its hidden state stops
changing (relative change `‖h_{t+1} − h_t‖_F / (‖h_t‖_F + 1e-8) < convergence_tol`).
Easy sequences converge early, hard ones loop more, in the *same* batch, instead of
every sequence paying max depth. With INT8 cut (decision 9), this carries the
inference story alone: it is the headline inference optimization, and the third
resume bullet's [X]× is batched-convergence decode vs naive fixed-depth decode.

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

**Evidence.** TBD — inference-throughput run (`EXPERIMENTS.md` §3): report throughput
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
