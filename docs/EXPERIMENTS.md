# Ouroboros — Experiment Log

This is a **pre-planned experiment log**. It defines, ahead of implementation, the
three experiments that validate the Ouroboros architecture and back the three resume
claims (looped transformer from scratch, LTI-stabilized training, KV-cached +
depth-wise batched inference). Each experiment states a falsifiable hypothesis, the
independent variable, the dependent variables, the exact `OuroborosConfig` differences,
a T4-realistic procedure, and an **empty results table to fill in** once the runs
complete (Phases 6–7 of the roadmap).

Treat the result tables as the contract for "done": an experiment is finished when its
table is fully populated and the hypothesis is explicitly accepted or rejected in the
**Verdict** line.

---

## 0. Shared methodology (read first — makes every experiment reproducible)

Everything below is held fixed across experiments unless an experiment explicitly lists
it as the independent variable. Deviations must be recorded in that experiment's "config
differences" block.

### 0.1 Hardware & precision

- **Target hardware:** a single Google Colab **T4** (16 GB VRAM, Turing sm75).
- **Precision:** FP16 mixed precision with `torch.cuda.amp` + `GradScaler` (T4 has no
  bf16 tensor cores, so bf16 is *not* used for training on this GPU). RMSNorm reductions
  and the LTI log-space math run in FP32 internally regardless.
- **Attention backend:** `F.scaled_dot_product_attention` with the flash / mem-efficient
  backend selected via `torch.backends.cuda.sdp_kernel`, with the manual SDPA fallback.
  The `flash_attn_func` fast path is **only** exercised on a rented Ampere GPU when that
  device is explicitly noted; it is not assumed to build on T4. Every throughput number
  records which backend actually ran.

### 0.2 The small ~10–30M-parameter T4 training config

All experiments use the same small research model unless stated otherwise. It is built
from `OuroborosConfig` with these values (the dataclass defaults already target this
regime; only the deltas from the defaults are listed):

| Field                    | Value  | Note                                                        |
|--------------------------|--------|-------------------------------------------------------------|
| `vocab_size`             | 8192   | small BPE vocab; embedding (tied) dominates params at this dim |
| `dim`                    | 512    | residual-stream width                                       |
| `n_heads`                | 8      | query heads                                                 |
| `n_kv_heads`             | 2      | GQA key/value heads                                         |
| `max_seq_len`            | 1024   | RoPE precompute length; training context = 1024             |
| `max_loop_iters`         | 8      | recurrent depth T (training default)                        |
| `prelude_layers`         | 2      | dense blocks before the loop                                |
| `coda_layers`            | 2      | dense blocks after the loop                                 |
| `use_lti`                | `True` | LTI-constrained injection; the stability experiment flips this |
| `n_experts`              | 8      | routed experts                                              |
| `n_shared_experts`       | 1      | always-active shared expert                                 |
| `n_experts_per_tok`      | 2      | top-K routed per token                                      |
| `expert_dim`             | 256    | fine-grained expert width                                   |
| `loop_index_dim`         | `None` | → `dim // 8 = 64` channels receive the loop-index signal    |
| `router_bias_update_rate`| 1e-3   | aux-loss-free load-balancing bias step                      |
| `rope_theta`             | 10000  | small-model default                                         |
| `norm_eps`               | 1e-6   | RMSNorm epsilon                                             |
| `init_std`               | 0.02   | weight init std                                             |
| `dropout`                | 0.0    | pretraining at this scale uses no dropout                   |

This lands at roughly 10–30M total parameters (the embedding/LM-head table is
`vocab_size × dim ≈ 4.2M` of that, tied once). Each experiment records the **exact
measured parameter count** (`sum(p.numel())`) and the **activated-parameter count per
token** (the MoE recurrent block activates only `n_experts_per_tok` of the `n_experts`
routed experts, plus the shared expert), so any params-vs-quality statement is honest
about sparsity.

A separate **tiny test config** (`dim=64`, `n_heads=4`, `n_kv_heads=2`, `n_experts=4`,
`expert_dim=64`, `prelude_layers=1`, `coda_layers=1`, `max_loop_iters=4`) exists purely
for fast correctness tests and is **not** used to produce any experiment numbers.

### 0.3 Data

- **Primary corpus:** WikiText-103 (raw, `wikitext-103-raw-v1` via `datasets`).
- **Alternate / fallback:** a fixed FineWeb-Edu slice (a deterministic shard, recorded by
  shard id and row range) when a cleaner web-text distribution is wanted. An experiment
  states which corpus it used; cross-experiment perplexity is only compared **within the
  same corpus**.
- **Tokenizer:** a BPE tokenizer trained to `vocab_size = 8192` on the training split,
  saved to disk and reused by every experiment so token ids are identical run to run.
- **Validation set:** a held-out split, fixed once. Perplexity is always reported on this
  exact split with `n_loops = max_loop_iters` (=8) unless the experiment varies `n_loops`.
- **Sequence packing:** documents concatenated and chunked to `max_seq_len = 1024`;
  packing order is seeded.

### 0.4 Optimizer & schedule (held fixed unless varied)

- AdamW, `betas=(0.9, 0.95)`, `weight_decay=0.1`, `eps=1e-8`.
- Cosine learning-rate schedule with linear warmup (first ~3% of steps).
- Gradient clipping at norm 1.0 — **kept on for every run**, including the no-LTI
  baseline, so the stability experiment isolates LTI rather than clipping (see
  Experiment 1; the "no-LTI" arm still gets the same clip, which makes the LTI win
  conservative/honest).
- `MoEFFN.update_router_bias()` is called **once per optimizer step** (the aux-loss-free
  load-balancing update that reference implementations leave as a stub).
- A fixed token budget per run (e.g. a set number of steps × global batch × 1024 tokens),
  recorded per experiment, so cost is comparable across arms.
- Effective batch via gradient accumulation tuned to fit 16 GB at FP16; the per-experiment
  table records micro-batch, accumulation steps, and effective tokens/step.

### 0.5 Seeds & reproducibility

- **Two seeds** per arm: `{0, 1}`. Tables report **mean ± spread** (half the seed-to-seed
  range — with two seeds a standard deviation would overstate precision); any single-seed
  number is flagged as such.
- At startup each run sets Python, NumPy, and Torch seeds and enables deterministic algorithms
  where it does not destroy throughput (cuDNN determinism is noted when toggled, since it
  affects the tokens/s numbers).
- Every run logs: git commit hash, full resolved `OuroborosConfig`, tokenizer hash, dataset
  name + split + slice, and the exact command line. These go into the W&B run config so a
  table row is traceable to one reproducible run.

### 0.6 Weights & Biases logging

- **Project:** `ouroboros`. **Entity:** the author's default W&B entity.
- **Run naming:** `{experiment-id}-{arm}-seed{N}` (e.g. `exp1-lti-seed0`,
  `exp1-nolti-seed0`).
- **Tags:** the experiment id (`exp1`…`exp3`) plus the arm name, so a W&B group/filter
  reconstructs each experiment.
- **Logged every step:** training loss, **ρ(A)** (see 0.7; LTI arm only), gradient norm
  (pre-clip), learning rate, tokens/s, and the MoE load-balance health signals — the
  per-expert load variance (the spread of `expert_load`) and the fraction of router-bias
  mass moved.
- **MoE health is validated here, not by a dedicated experiment:** the trimmed scope has
  no MoE-vs-dense ablation, so the load-variance and router-bias-movement curves above —
  recorded on **every** run — are the evidence that the aux-loss-free balancing works.
- **Logged every eval interval:** validation perplexity (at `n_loops = max_loop_iters`).
- The W&B run URL is pasted into each results table row so a number is one click from its
  curve.

### 0.7 How ρ(A) (spectral radius of the LTI state matrix) is logged

The LTI injection uses a **diagonal** discrete state matrix `A_discrete`, so its spectral
radius is simply the largest-magnitude diagonal entry. `LTIInjection.get_A()` returns the
`(dim,)` vector of diagonal entries, each in `(0, 1)` by construction
(`A = exp(-exp((log_dt + log_A).clamp(-20, 20)))`). The logged stability signal is therefore:

```
rho_A = LTIInjection.get_A().max().item()   # scalar in (0, 1) by construction
```

This is computed under `torch.no_grad()` and logged to W&B **every training step** (it is
cheap — one reduction over `dim`). It is the centerpiece of the stability experiment: for
the LTI arm it must stay strictly `< 1` for the entire run. The no-LTI arm
(`use_lti=False`) replaces the LTI update with the naive residual injection
`h = transformer_out + e` and so has **no state matrix to log** — its instability shows up
directly in the loss and gradient-norm curves instead. We additionally log the **mean**
and **min** of `get_A()` so the table can describe the whole spectrum, not just the worst
eigenvalue.

### 0.8 How the convergence-depth distribution is measured (inference)

The core training loop runs a **fixed** `n_loops` for every position — there is no
adaptive halting during training. The analogous depth signal exists only at *inference*,
inside continuous depth-wise batching (`generate_depthwise_batched`): per sequence, record
the loop iteration `t` at which the **relative** hidden-state change falls below the
threshold (`‖h_{t+1} − h_t‖_F / (‖h_t‖_F + 1e-8) < convergence_tol`), i.e. its
**convergence depth** in `{1, …, max_loops}`. We aggregate this over the eval prompts and
log: the **mean** convergence depth, the per-depth histogram (the **distribution** of
convergence depth), and the fraction of sequences that never converge before `max_loops`.
These feed Experiment 3 (inference throughput) — the depth-wise batching win scales with
the spread of this distribution.

---

## Experiment index

| #  | Name                                                  | Independent variable                | Resume bullet |
|----|-------------------------------------------------------|-------------------------------------|---------------|
| 1  | Stability: LTI vs no-LTI                              | `use_lti` on/off × learning rate    | 2             |
| 2  | Loop-count sweep (depth extrapolation)                | inference `n_loops`                 | 1             |
| 3  | Inference throughput (cache × backend × batching)     | decode optimization layer           | 3             |

---

## Experiment 1 — Stability: LTI-constrained injection vs naive residual injection

**Resume bullet:** 2 ("Stabilized looped training via LTI-constrained injection …
enabling clean convergence at high learning rates").

### Hypothesis
Constraining the recurrent state update so that ρ(A) < 1 by construction (the LTI
injection) yields stable, monotone convergence at high learning rates, whereas the naive
residual injection `h = transformer_out + e` — same loop, no contractive state path —
destabilizes: the training loss diverges (NaN / blow-up) at learning rates where the LTI
arm still converges. The gap widens as the learning rate increases.

### Independent variable
1. **Injection rule** — `use_lti=True` (the `LTIInjection` update
   `h_{t+1} = A·h_t + B·e + transformer_out`, ρ(A) < 1 guaranteed) vs `use_lti=False`
   (the LTI update replaced by the naive residual injection `h = transformer_out + e` —
   no `A`, no `B`, no spectral guarantee).
2. **Learning rate** — swept over `{3e-4, 1e-3, 3e-3}` for each injection rule.

(2 injection rules × 3 learning rates × 2 seeds = 12 runs.)

### Dependent variables
- Training loss curve (and whether it diverges / NaNs).
- **ρ(A)** over steps (`get_A().max()`), plus its mean and min — LTI arm only (the
  no-LTI arm has no state matrix; see §0.7).
- Gradient norm (pre-clip) over steps.
- Final validation perplexity (for runs that survive).
- The step at which divergence occurs, if any.

### Config differences (vs the shared T4 config, §0.2)
- `use_lti=False` is the only change in the no-LTI arm. The loop body and the initial
  state `h_0 = e` (the prelude output seeds both the state and the injection) are
  identical in both arms; only the state update differs.
- All other fields identical. **Gradient clipping stays at 1.0 for both arms** so the
  comparison credits LTI, not clipping — and clipping can only help the no-LTI arm,
  keeping the comparison conservative.
- One short fixed token budget (enough to expose divergence within the budget — divergence,
  when it happens, is early).

### Procedure (T4-realistic)
1. Train all 12 runs with identical data order per seed.
2. Log ρ(A) (LTI arm) and grad norm every step; loss every step; perplexity at intervals.
3. A run is "diverged" if loss becomes NaN/Inf or rises monotonically for a sustained window.
4. Overlay loss and grad-norm curves (LTI vs no-LTI) per learning rate from W&B.

### Results — fill in
**Convergence by arm (report mean ± spread over 2 seeds):**

| LR    | Injection | Converged? | Final val PPL | Final ρ(A) | Max ρ(A) over run | Diverge step | W&B |
|-------|-----------|------------|---------------|------------|-------------------|--------------|-----|
| 3e-4  | lti       |            |               |            |                   |              |     |
| 3e-4  | no-lti    |            |               | —          | —                 |              |     |
| 1e-3  | lti       |            |               |            |                   |              |     |
| 1e-3  | no-lti    |            |               | —          | —                 |              |     |
| 3e-3  | lti       |            |               |            |                   |              |     |
| 3e-3  | no-lti    |            |               | —          | —                 |              |     |

(The ρ(A) columns apply to the `lti` arm only; the `no-lti` arm has no state matrix.)

**Verdict:** _(accept/reject hypothesis; state the highest LR at which the no-LTI
arm diverges and the LTI arm still converges — this is the headline number for resume
bullet 2.)_

---

## Experiment 2 — Loop-count sweep (test-time depth extrapolation)

**Resume bullet:** 1 (looped/recurrent design; depth extrapolation).

### Hypothesis
A model trained at `max_loop_iters = 8` can be run at inference with a **different**
`n_loops` and remain coherent — validation perplexity improves (or at least stays stable)
as `n_loops` increases up to 8, and the model **extrapolates** beyond training depth
(`n_loops = 16`): perplexity improves further or degrades gracefully, without collapse,
thanks to the LTI injection keeping the encoded input alive and the sinusoidal
loop-index embedding being well-defined at any depth.

### Independent variable
Inference recurrent depth `n_loops ∈ {2, 4, 8, 16}` (a forward-pass argument; **no
retraining**). 16 exceeds the trained `max_loop_iters = 8` and exercises depth
extrapolation beyond the trained range.

### Dependent variables
- Validation perplexity at each `n_loops`.
- Decode throughput (tokens/s) at each `n_loops`.
- A qualitative stability check (no NaN / no degeneration) at `n_loops = 16`.

### Config differences (vs §0.2)
- One trained model per seed (`max_loop_iters = 8`). Only the forward/`generate`
  `n_loops` argument changes; the config dataclass is unchanged.

### Procedure (T4-realistic)
1. Train one model per seed under §0.2.
2. Evaluate validation perplexity and decode throughput at each `n_loops` value.
3. At `n_loops = 16`, confirm the model runs deeper than training (the sinusoidal
   loop-index embedding is defined for any depth) and outputs stay finite.

### Results — fill in
(Report mean ± spread over 2 seeds.)

| n_loops | Val PPL | Decode tok/s | Stable (no NaN)? | Notes | W&B |
|---------|---------|--------------|------------------|-------|-----|
| 2       |         |              |                  |       |     |
| 4       |         |              |                  |       |     |
| 8       |         |              |                  |       |     |
| 16      |         |              |                  |       |     |

**Verdict:** _(does perplexity improve with depth up to 8, and does the model extrapolate to
16 without collapse? state the perplexity at 8 vs 16.)_

---

## Experiment 3 — Inference throughput: KV cache, SDPA backend, depth-wise batching

**Resume bullet:** 3 ("KV-cached decode + SDPA flash/mem-efficient attention +
continuous depth-wise batching, achieving [X]× inference throughput").

### Hypothesis
Letting sequences in a batch exit the recurrent loop at **different convergence-driven
depths** (`generate_depthwise_batched`, exiting a sequence once its relative state change
`‖h_{t+1} − h_t‖_F / (‖h_t‖_F + 1e-8)` falls below `convergence_tol`) — instead of every
sequence paying the maximum depth (`generate`) — increases batched decode throughput at
matched output quality. The gain scales with the spread of the convergence-depth
distribution (§0.8): more spread → more saving; if every sequence runs to `max_loops`,
the gain collapses to ~1× (the falsification case). Literature suggests ~2–3×; the
measured number fills resume bullet 3. The KV cache and the flash / mem-efficient SDPA
backend are measured as supporting rungs so the headline multiplier is attributed
honestly rather than silently folded in.

### Independent variables
1. **KV cache** — off (full-prefix `forward` per decode step) vs on (`generate`'s
   cached decode).
2. **Attention backend** — SDPA flash / mem-efficient (via
   `torch.backends.cuda.sdp_kernel`) vs the manual SDPA fallback (§0.1; every number
   records which backend actually ran).
3. **Generation strategy** — naive fixed-depth `generate` (every sequence runs to
   `n_loops`) vs `generate_depthwise_batched` (per-sequence convergence-based early
   exit).

### Dependent variables
- Decode throughput (tokens/s) and median per-token latency (ms) for each rung, at
  several batch sizes.
- The realized **distribution** of per-sequence exit depths (§0.8): mean, per-depth
  histogram, never-converged fraction.
- Output quality at the operating `convergence_tol`: with `convergence_tol → 0` the
  depth-wise path must match `generate` **token-for-token** (the cache-population
  correctness gate — the cache-key problem is solved, not silently producing wrong
  tokens); at the chosen tolerance, held-out perplexity must match the fixed-depth
  baseline within seed spread (matched output quality).
- The end-to-end **[X]×** multiplier: batched-convergence decode
  (`generate_depthwise_batched`) vs naive fixed-depth decode (`generate`) at the best
  batch size, same backend, both KV-cached.

### Config differences (vs §0.2)
- Same trained checkpoint for every arm (the seed-0 Experiment 2 model) — this
  experiment is inference-only; no retraining. Throughput is a systems measurement, not
  a training statistic, but per §0.5 any single-checkpoint quality number is flagged as
  single-seed.
- Both generation strategies use the same KV-cache path. **The cache-population
  constraint to honor:** a sequence that converges and exits at depth `d` must not leave
  later `recurrent_loop_{t>d}` cache keys unpopulated when a future step needs them. The
  implemented solution is **run-to-max-active-depth + masking** — converged rows are
  frozen but their cache keys still get populated.
- Sweep `convergence_tol` over a small grid (e.g. `{1e-2, 1e-3, 1e-4}`) to trace the
  quality/throughput tradeoff; the main tables fix the chosen value.
- Sweep batch size `∈ {1, 4, 16, 32}` (T4-realistic; record the largest that fits in 16 GB).

### Procedure (T4-realistic)
1. Benchmarking discipline for every number: fixed prompt set and `max_new_tokens`,
   warmup iterations before timing, `torch.cuda.synchronize()` around timers, median
   over repeats, and the actually-active attention backend recorded (§0.1).
2. Climb the optimization ladder at a fixed batch size: (a) no KV cache, manual SDPA
   fallback, fixed depth; (b) + KV cache (`generate`); (c) + flash / mem-efficient SDPA
   backend; (d) + depth-wise batching (`generate_depthwise_batched`). Record tokens/s
   and latency per rung.
3. Run the depth-wise vs fixed-depth comparison at each batch size; record tokens/s and
   the exit-depth distribution.
4. Correctness before speed: with `convergence_tol → 0`, assert token-for-token equality
   with `generate`; then at the chosen tolerance, verify held-out perplexity matches the
   baseline within seed spread.
5. Tie the measured throughput ratio to the convergence-depth distribution (§0.8) and
   read off the headline [X]×.

### Results — fill in
**Optimization ladder (decode, fixed batch size = 16):**

| Rung                        | KV cache | Backend         | Depth-wise? | Decode tok/s | Median ms/token | × vs (a) | W&B |
|-----------------------------|----------|-----------------|-------------|--------------|-----------------|----------|-----|
| (a) naive fixed-depth       | off      | manual fallback | no          |              |                 | 1.0× (ref) |   |
| (b) + KV cache              | on       | manual fallback | no          |              |                 |          |     |
| (c) + SDPA flash/mem-eff    | on       | flash/mem-eff   | no          |              |                 |          |     |
| (d) + depth-wise batching   | on       | flash/mem-eff   | yes         |              |                 |          |     |

**Depth-wise batched vs naive fixed-depth `generate` (both KV-cached, best backend):**

| Batch size | `generate` tok/s | Depth-wise tok/s | Throughput × | Mean exit depth | Quality match? | W&B |
|------------|------------------|------------------|--------------|-----------------|----------------|-----|
| 1          |                  |                  |              |                 |                |     |
| 4          |                  |                  |              |                 |                |     |
| 16         |                  |                  |              |                 |                |     |
| 32         |                  |                  |              |                 |                |     |

(`Quality match?` = token-for-token equality at `convergence_tol → 0` **and**
baseline-level held-out perplexity at the operating tolerance.)

**Verdict:** _(state the headline [X]× — batched-convergence decode vs naive fixed-depth
decode at the best batch size — and relate it to the convergence-depth spread; report
the cache and backend contributions from the ladder so the multiplier is honestly
attributed. This is the resume-bullet-3 number.)_

---

## Cross-experiment summary (fill in last)

| Resume bullet | Backed by experiment(s) | Headline number to report                          | Status |
|---------------|-------------------------|----------------------------------------------------|--------|
| 1 (architecture) | 2 (+ MoE load-balance health via training logs, §0.6) | depth extrapolation: val PPL at `n_loops=16` vs 8 |        |
| 2 (LTI stability)| 1                   | highest LR where no-LTI diverges but LTI converges |        |
| 3 (inference)    | 3                   | end-to-end [X]× — depth-wise batched vs naive fixed-depth decode |        |
