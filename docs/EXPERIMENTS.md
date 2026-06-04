# Ouroboros — Experiment Log

This is a **pre-planned experiment log**. It defines, ahead of implementation, the
seven experiments that validate the Ouroboros architecture and back the three resume
claims (looped transformer from scratch, LTI-stabilized training, INT8 + depth-wise
batched inference). Each experiment states a falsifiable hypothesis, the independent
variable, the dependent variables, the exact `OuroborosConfig` differences, a
T4-realistic procedure, and an **empty results table to fill in** once the runs
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
| `attn_type`              | `"gqa"`| default; the MLA-vs-GQA experiment flips this               |
| `n_experts`              | 8      | routed experts                                              |
| `n_shared_experts`       | 1      | always-active shared expert                                 |
| `n_experts_per_tok`      | 2      | top-K routed per token                                      |
| `expert_dim`             | 256    | fine-grained expert width                                   |
| `act_threshold`          | 0.99   | ACT cumulative-probability halting threshold                |
| `lora_rank`              | 8      | depth-wise LoRA bottleneck rank                             |
| `loop_index_dim`         | `None` | → `dim // 8 = 64` channels receive the loop-index signal    |
| `router_bias_update_rate`| 1e-3   | aux-loss-free load-balancing bias step                      |
| `rope_theta`             | 10000  | small-model default                                         |
| `norm_eps`               | 1e-6   | RMSNorm epsilon                                             |
| `init_std`               | 0.02   | weight init std                                             |
| `dropout`                | 0.0    | pretraining at this scale uses no dropout                   |

This lands at roughly 10–30M total parameters depending on `attn_type` and whether MoE
is active (the embedding/LM-head table is `vocab_size × dim ≈ 4.2M` of that, tied once).
Each experiment records the **exact measured parameter count** (`sum(p.numel())`) and the
**activated-parameter count per token** (relevant for the MoE-vs-dense comparison), since
matched-budget claims depend on both.

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
- Gradient clipping at norm 1.0 — **kept on for every run**, including the LTI baseline,
  so the stability experiment isolates LTI rather than clipping (see Experiment 1; the
  "no-LTI" arm still gets the same clip, which makes the LTI win conservative/honest).
- `MoEFFN.update_router_bias()` is called **once per optimizer step** (the aux-loss-free
  load-balancing update that reference implementations leave as a stub).
- A fixed token budget per run (e.g. a set number of steps × global batch × 1024 tokens),
  recorded per experiment, so cost is comparable across arms.
- Effective batch via gradient accumulation tuned to fit 16 GB at FP16; the per-experiment
  table records micro-batch, accumulation steps, and effective tokens/step.

### 0.5 Seeds & reproducibility

- **Three seeds** per arm: `{0, 1, 2}`. Tables report **mean ± standard deviation** across
  seeds; single-seed numbers are flagged as such.
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
- **Tags:** the experiment id (`exp1`…`exp7`) plus the arm name, so a W&B group/filter
  reconstructs each experiment.
- **Logged every step:** training loss, **ρ(A)** (see 0.7), gradient norm (pre-clip),
  learning rate, tokens/s, and the fraction of router-bias mass moved (load-balance health).
- **Logged every eval interval:** validation perplexity, mean ACT halting depth, and the
  full ACT halting-depth **distribution** (see 0.8).
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
the LTI arm it must stay strictly `< 1` for the entire run; for the unconstrained arm it is
expected to drift toward / exceed `1` as training diverges. We additionally log the **mean**
and **min** of `get_A()` so the table can describe the whole spectrum, not just the worst
eigenvalue.

### 0.8 How the ACT halting-depth distribution is logged

For each forward pass the recurrent block records, per position, the loop iteration `t` at
which that position crossed `act_threshold` (i.e. its halting depth in `{1, …, n_loops}`).
At each eval interval we aggregate halting depth over the whole validation set and log:
the **mean** halting depth, the per-depth histogram (the **distribution** of halting depth),
and the fraction of positions that never halt before `n_loops` (forced halts). These feed
Experiments 2 and 6.

---

## Experiment index

| #  | Name                                   | Independent variable                          | Resume bullet |
|----|----------------------------------------|-----------------------------------------------|---------------|
| 1  | Stability: LTI vs no-LTI               | LTI injection on/off × learning rate          | 2             |
| 2  | ACT halting vs fixed loop count        | adaptive ACT vs fixed depth                   | 1             |
| 3  | Attention ablation: MLA vs GQA         | `attn_type`                                   | 1             |
| 4  | MoE vs dense (matched param budget)    | sparse MoE FFN vs dense FFN                    | 1             |
| 5  | INT8 quantization                      | weight precision (FP16 vs INT8)               | 3             |
| 6  | Continuous depth-wise batching         | early-exit batching on/off                    | 3             |
| 7  | Loop-count sweep (depth extrapolation) | inference `n_loops`                            | 1             |

---

## Experiment 1 — Stability: LTI-constrained injection vs unconstrained

**Resume bullet:** 2 ("Stabilized looped training via LTI-constrained injection …
enabling clean convergence at high learning rates").

### Hypothesis
Constraining the recurrent state update so that ρ(A) < 1 by construction (the LTI
injection) yields stable, monotone convergence at high learning rates, whereas an
unconstrained injection of the same shape destabilizes — ρ(A) drifts toward or past 1
and the training loss diverges (NaN / blow-up) — at the same learning rate. The gap
widens as the learning rate increases.

### Independent variable
1. **Injection type** — `lti` (the `LTIInjection` module, ρ(A) < 1 guaranteed) vs
   `unconstrained` (same residual update `A·h + B·e + transformer_out` but with `A` and
   `B` as free, unconstrained learned vectors — no log-space ZOH parameterization, no
   spectral guarantee).
2. **Learning rate** — swept over `{3e-4, 1e-3, 3e-3}` for each injection type.

(2 injection types × 3 learning rates × 3 seeds = 18 runs.)

### Dependent variables
- Training loss curve (and whether it diverges / NaNs).
- **ρ(A)** over steps (`get_A().max()`), plus its mean and min.
- Gradient norm (pre-clip) over steps.
- Final validation perplexity (for runs that survive).
- The step at which divergence occurs, if any.

### Config differences (vs the shared T4 config, §0.2)
- Injection type is the only architectural change; the `unconstrained` arm swaps
  `LTIInjection` for a same-interface module with free `A`, `B` vectors and no clamp/ZOH.
- All other fields identical. **Gradient clipping stays at 1.0 for both arms** so the
  comparison credits LTI, not clipping.
- One short fixed token budget (enough to expose divergence within the budget — divergence,
  when it happens, is early).

### Procedure (T4-realistic)
1. Train all 18 runs with identical data order per seed.
2. Log ρ(A) and grad norm every step; loss every step; perplexity at intervals.
3. A run is "diverged" if loss becomes NaN/Inf or rises monotonically for a sustained window.
4. Overlay loss and ρ(A) curves (LTI vs unconstrained) per learning rate from W&B.

### Results — fill in
**ρ(A) and convergence by arm (report mean ± std over 3 seeds):**

| LR    | Injection      | Converged? | Final val PPL | Final ρ(A) | Max ρ(A) over run | Diverge step | W&B |
|-------|----------------|------------|---------------|------------|-------------------|--------------|-----|
| 3e-4  | lti            |            |               |            |                   |              |     |
| 3e-4  | unconstrained  |            |               |            |                   |              |     |
| 1e-3  | lti            |            |               |            |                   |              |     |
| 1e-3  | unconstrained  |            |               |            |                   |              |     |
| 3e-3  | lti            |            |               |            |                   |              |     |
| 3e-3  | unconstrained  |            |               |            |                   |              |     |

**Verdict:** _(accept/reject hypothesis; state the highest LR at which the unconstrained
arm diverges and the LTI arm still converges — this is the headline number for resume
bullet 2.)_

---

## Experiment 2 — ACT halting vs fixed loop count

**Resume bullet:** 1 (the adaptive-depth recurrent design).

### Hypothesis
Adaptive Computation Time halting reaches validation perplexity comparable to (or better
than) a fixed maximum loop count, while spending **fewer average loop iterations** per
token — yielding higher decode throughput. Easy positions halt early; hard positions use
more depth.

### Independent variable
**Halting policy:** adaptive ACT (`act_threshold = 0.99`) vs **fixed** loop count (ACT
disabled; the model runs exactly `n_loops` iterations and returns the last hidden state).
The fixed arm is evaluated at several fixed depths `n_loops ∈ {2, 4, 8}`.

### Dependent variables
- Validation perplexity.
- **Mean ACT halting depth** and the full halting-depth **distribution** (§0.8) for the
  adaptive arm.
- Decode throughput (tokens/s) and average loop iterations executed per token.

### Config differences (vs §0.2)
- Adaptive arm: shared config, ACT active.
- Fixed arms: same trained weights where possible, or a matched-budget run with ACT
  accumulation replaced by "use final-iteration hidden state"; `act_threshold` is inert.
- For a clean comparison, train one adaptive model and one fixed model with the same token
  budget; report the adaptive model's halting distribution at eval.

### Procedure (T4-realistic)
1. Train the adaptive model under §0.2.
2. Evaluate it; record mean halting depth + distribution + throughput.
3. Train/evaluate the fixed-depth arms at `n_loops ∈ {2, 4, 8}`; record perplexity +
   throughput.
4. Compare perplexity-vs-throughput frontiers.

### Results — fill in
| Arm              | n_loops (eval) | Val PPL | Mean halt depth | Tokens/s (decode) | Avg iters/token | W&B |
|------------------|----------------|---------|-----------------|-------------------|-----------------|-----|
| ACT (adaptive)   | up to 8        |         |                 |                   |                 |     |
| Fixed            | 2              |         | n/a             |                   | 2.0             |     |
| Fixed            | 4              |         | n/a             |                   | 4.0             |     |
| Fixed            | 8              |         | n/a             |                   | 8.0             |     |

**Halting-depth distribution (adaptive arm), fraction of positions halting at each depth:**

| Depth t  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 (forced) |
|----------|---|---|---|---|---|---|---|------------|
| Fraction |   |   |   |   |   |   |   |            |

**Verdict:** _(does adaptive halting match fixed-8 perplexity at lower average depth /
higher throughput? state the perplexity gap and the throughput ratio.)_

---

## Experiment 3 — Attention ablation: MLA vs GQA

**Resume bullet:** 1 ("switchable MLA/GQA attention").

### Hypothesis
MLA (compressed-KV attention) and GQA reach comparable validation perplexity at this
scale, but MLA caches a much smaller KV footprint per token (it stores the compressed
latent `c_kv` plus the shared rotary key, not full per-head K/V), trading some compute
for memory. GQA keeps the `flash_attn_func` / SDPA-flash fast path and is simpler.

### Independent variable
`attn_type` ∈ {`"gqa"`, `"mla"`}.

### Dependent variables
- Validation perplexity.
- **KV-cache memory** at a fixed context length (bytes per token, and total at
  `max_seq_len`).
- Prefill and decode throughput (tokens/s).
- Total and activated parameter counts.

### Config differences (vs §0.2)
- GQA arm: `attn_type="gqa"`, uses `n_heads=8`, `n_kv_heads=2`, RoPE over `dim//n_heads=64`.
- MLA arm: `attn_type="mla"`, uses `kv_lora_rank=128`, `q_lora_rank=256`,
  `qk_rope_head_dim=32`, `qk_nope_head_dim=64`, `v_head_dim=64`; `n_kv_heads` is **inert**
  for MLA. RoPE for MLA is the separate `freqs_cis_mla` buffer sized `qk_rope_head_dim`.
- Same data, token budget, optimizer, seeds.

### Procedure (T4-realistic)
1. Train both arms under §0.2.
2. Measure KV-cache bytes by inspecting the cache dict tensors at a fixed context length
   (GQA stores `k`,`v`; MLA stores `c_kv`,`k_rope`). Assert MLA cache bytes < GQA cache bytes.
3. Benchmark prefill and decode throughput with the actually-selected attention backend
   (record which one ran on T4).

### Results — fill in
| attn_type | Val PPL | Total params | KV bytes/token | KV bytes @1024 | Prefill tok/s | Decode tok/s | Backend | W&B |
|-----------|---------|--------------|----------------|----------------|---------------|--------------|---------|-----|
| gqa       |         |              |                |                |               |              |         |     |
| mla       |         |              |                |                |               |              |         |     |

**Verdict:** _(perplexity gap; MLA KV-memory reduction factor vs GQA; throughput
tradeoff.)_

---

## Experiment 4 — MoE vs dense FFN (matched parameter budget)

**Resume bullet:** 1 (fine-grained MoE with routed + shared experts).

### Hypothesis
At a **matched total-parameter budget**, the fine-grained MoE recurrent FFN (routed top-K
experts + always-active shared expert, with aux-loss-free load balancing) reaches lower
validation perplexity than a dense FFN of equal parameter count, because sparsity buys
capacity at fixed activated FLOPs. The benefit is conditioned on the router actually
balancing load (tracked via `update_router_bias`).

### Independent variable
Recurrent-block FFN type: **MoE** (`MoEFFN`, `use_moe=True`) vs **dense** (`Expert(dim,
expert_dim_dense)`, `use_moe=False`) — matched on **total** parameters.

### Dependent variables
- Validation perplexity.
- Total params (matched) and **activated params per token** (MoE activates far fewer).
- Training throughput (tokens/s) — MoE's masked-loop dispatch is slower per step.
- Per-expert load balance (the spread of `expert_load` after the bias update).

### Config differences (vs §0.2)
- MoE arm: `n_experts=8`, `n_shared_experts=1`, `n_experts_per_tok=2`, `expert_dim=256`;
  `update_router_bias()` called every step.
- Dense arm: recurrent FFN is a single `Expert` whose `expert_dim_dense` is chosen so the
  dense model's **total** parameter count matches the MoE model's total (record the exact
  width used and the resulting param delta — aim for < 1% mismatch).
- Same data, token budget, optimizer, seeds.

### Procedure (T4-realistic)
1. Compute the dense FFN width that matches MoE total params; record both counts.
2. Train both arms under §0.2.
3. Log per-expert selection counts (`expert_load`) and the router-bias movement to confirm
   the MoE arm balances rather than collapsing onto a few experts.

### Results — fill in
| Arm   | Total params | Activated params/token | Val PPL | Train tok/s | Load CV (expert balance) | W&B |
|-------|--------------|------------------------|---------|-------------|--------------------------|-----|
| MoE   |              |                        |         |             |                          |     |
| Dense |              |                        |         |             |                          |     |

(`Load CV` = coefficient of variation of per-expert selection counts; lower = better
balanced.)

**Verdict:** _(does sparsity help at fixed total params? state the perplexity delta and the
activated-FLOP saving, and confirm the router did not collapse.)_

---

## Experiment 5 — INT8 post-training quantization

**Resume bullet:** 3 ("INT8 quantized inference … throughput").

### Hypothesis
Per-channel INT8 weight quantization of the large Linear layers (attention projections and
expert FFNs), keeping norms, the router, and the tied LM head in higher precision, yields a
small, acceptable validation-perplexity increase while improving decode throughput and
reducing model memory on T4 (which has INT8 tensor cores).

### Independent variable
Weight precision of the big Linear layers: **FP16** (baseline) vs **INT8** (via
`quantize_int8(model, method=...)`, comparing `dynamic` and `static`/calibrated PTQ).

### Dependent variables
- Validation perplexity before vs after quantization (the **quantization error**, via
  `quantization_error(fp_model, int8_model, eval_loader)`).
- Decode throughput (tokens/s) before vs after.
- Model memory footprint (bytes) before vs after.

### Config differences (vs §0.2)
- Same trained FP16 checkpoint for every arm — this is **post-training**, no retraining.
- Arms differ only in `quantize_int8` `method`: none (FP16 baseline), `dynamic`, `static`
  (with `calibrate(model, calibration_loader)` on a fixed calibration slice).
- Record the chosen INT8 backend (`torch.ao.quantization` vs `bitsandbytes`) and confirm
  norms / router / LM head stayed in higher precision.

### Procedure (T4-realistic)
1. Take the trained FP16 model; measure baseline val perplexity, decode throughput, memory.
2. Quantize dynamically; re-measure all three; compute perplexity delta.
3. Calibrate on a fixed slice and quantize statically; re-measure.
4. Report the perplexity delta and the throughput/memory gains.

### Results — fill in
| Arm            | Val PPL | ΔPPL vs FP16 | Decode tok/s | Throughput × | Model memory | Memory × | Backend | W&B |
|----------------|---------|--------------|--------------|--------------|--------------|----------|---------|-----|
| FP16 baseline  |         | 0 (ref)      |              | 1.0×         |              | 1.0×     |         |     |
| INT8 dynamic   |         |              |              |              |              |          |         |     |
| INT8 static    |         |              |              |              |              |          |         |     |

**Verdict:** _(is the perplexity delta acceptable? state the throughput multiplier — this
contributes to the resume-bullet-3 number.)_

---

## Experiment 6 — Continuous depth-wise batching

**Resume bullet:** 3 ("continuous depth-wise batching, achieving [X]× inference
throughput").

### Hypothesis
Letting sequences in a batch exit the recurrent loop at **different ACT-driven depths**
(`generate_depthwise_batched`) — instead of every sequence paying the maximum depth
(`generate`) — increases batched decode throughput. The gain scales with the spread of the
ACT halting-depth distribution: more spread → more saving. Literature suggests ~2–3×; the
measured number fills resume bullet 3.

### Independent variable
Batched generation strategy: **baseline** (`generate`, every sequence runs to `n_loops`)
vs **depth-wise batched** (`generate_depthwise_batched`, per-sequence early exit with the
chosen cache-population solution).

### Dependent variables
- Batched decode throughput (tokens/s) at several batch sizes.
- The realized **distribution** of per-sequence exit depths in the batch.
- A correctness check: depth-wise-batched outputs match per-sequence `generate` outputs
  (the cache-key population problem is solved, not silently producing wrong tokens).

### Config differences (vs §0.2)
- Same trained model and weights for both arms.
- Both use a KV cache. **Critical constraint to honor (this is the headline subtlety):**
  with a KV cache the loop must not early-exit globally — every loop depth must run on every
  forward pass so later decode steps find populated keys at every `recurrent_loop_{t}` cache
  key. Record which cache-population solution is used (run-to-max-active-depth + mask /
  ragged per-sequence cache / depth bucketing).
- Sweep batch size `∈ {1, 4, 16, 32}` (T4-realistic; record the largest that fits in 16 GB).

### Procedure (T4-realistic)
1. Generate with `generate` (baseline) at each batch size; record tokens/s.
2. Generate with `generate_depthwise_batched`; record tokens/s and the exit-depth
   distribution.
3. Assert token-for-token equality with the baseline outputs (correctness before speed).
4. Tie the measured throughput ratio to the ACT halting-depth distribution from
   Experiment 2.

### Results — fill in
| Batch size | Baseline tok/s | Depth-wise tok/s | Throughput × | Mean exit depth | Outputs match? | W&B |
|------------|----------------|------------------|--------------|-----------------|----------------|-----|
| 1          |                |                  |              |                 |                |     |
| 4          |                |                  |              |                 |                |     |
| 16         |                |                  |              |                 |                |     |
| 32         |                |                  |              |                 |                |     |

**Verdict:** _(state the throughput multiplier vs baseline at the best batch size and relate
it to the halting-depth spread. Combine with Experiment 5 for the end-to-end resume-bullet-3
multiplier.)_

---

## Experiment 7 — Loop-count sweep (test-time depth extrapolation)

**Resume bullet:** 1 (looped/recurrent design; depth extrapolation).

### Hypothesis
A model trained at `max_loop_iters = 8` can be run at inference with a **different**
`n_loops` and remain coherent — validation perplexity improves (or at least stays stable)
as `n_loops` increases up to 8, and the model **extrapolates** beyond training depth
(`n_loops = 16`) without collapse, thanks to the LTI injection keeping the encoded input
alive and the LoRA scale being clamped past `max_loops - 1`.

### Independent variable
Inference recurrent depth `n_loops ∈ {2, 4, 8, 16}` (a forward-pass argument; **no
retraining**). 16 exceeds the trained `max_loop_iters = 8` and exercises the depth-extrapolation
clamp in `LoRAAdapter`.

### Dependent variables
- Validation perplexity at each `n_loops`.
- Decode throughput (tokens/s) at each `n_loops`.
- A qualitative stability check (no NaN / no degeneration) at `n_loops = 16`.

### Config differences (vs §0.2)
- One trained model (`max_loop_iters = 8`). Only the forward/`generate` `n_loops` argument
  changes; the config dataclass is unchanged.
- ACT halting stays active; report whether positions still halt before the (possibly larger)
  `n_loops` cap.

### Procedure (T4-realistic)
1. Train one model under §0.2.
2. Evaluate validation perplexity and decode throughput at each `n_loops` value.
3. At `n_loops = 16`, confirm the `LoRAAdapter` scale index is clamped to the last learned
   loop (no out-of-range indexing) and outputs stay finite.

### Results — fill in
| n_loops | Val PPL | Decode tok/s | Mean halt depth | Stable (no NaN)? | Notes | W&B |
|---------|---------|--------------|-----------------|------------------|-------|-----|
| 2       |         |              |                 |                  |       |     |
| 4       |         |              |                 |                  |       |     |
| 8       |         |              |                 |                  |       |     |
| 16      |         |              |                 |                  |       |     |

**Verdict:** _(does perplexity improve with depth up to 8, and does the model extrapolate to
16 without collapse? state the perplexity at 8 vs 16.)_

---

## Cross-experiment summary (fill in last)

| Resume bullet | Backed by experiment(s) | Headline number to report                          | Status |
|---------------|-------------------------|----------------------------------------------------|--------|
| 1 (architecture) | 2, 3, 4, 7           | adaptive halting / MLA-GQA parity / MoE win / extrapolation |        |
| 2 (LTI stability)| 1                   | highest LR where no-LTI diverges but LTI converges |        |
| 3 (inference)    | 5, 6                | end-to-end throughput × (INT8 × depth-wise batching) |        |
