# Ouroboros Roadmap

Ouroboros is a learning project: build a small recurrent-depth (looped)
transformer from scratch, train it, and change its loop count at test time, in
order to understand how looped models work and how they are trained. Along the
way it covers the standard parts of a modern LLM. Reading comes first; code
exists to make the reading concrete.

Delete this file when done, refer to notes taken in Obsidian

Loop:
1. Read the section in ROADMAP --> write notes in Obsidian
2. Skim the matching part of ARCHITECTURE
3. Open the code stub; docstring has the specs (inputs, shapes, etc.)
4. Write the tests first (to understand the reading) --> write the code
5. Run pytest -q (only section 10/11 require a GPU)

---

## 1. The big picture: looped transformers

What a looped model is, what looping costs, and how Huginn trains one so the
loop count can change at test time.

**Read**
- [GPT-6 Astra, Looped Transformers, and Hidden Reasoning](https://magazine.sebastianraschka.com/p/gpt-6-astra-looped-transformers-and) (Raschka, 2026). READ the looped transformer sections: reusing blocks, looping costs, flexible loop counts, the research roundup.
- [Huginn: Scaling up Test-Time Compute with Latent Reasoning](https://arxiv.org/abs/2502.05171) (Geiping et al., 2025). READ the architecture and training sections: Prelude/Recurrent/Coda, random loop count, truncated backprop, sandwich norm, the failed runs. Then the test-time section (early exit, KV sharing).
- [Universal Transformers](https://arxiv.org/abs/1807.03819) (Dehghani et al., 2018). SKIM section 2: the original weight-tied loop and its per-step timestep signal, which the loop index copies.
- [Reasoning with Latent Thoughts](https://arxiv.org/abs/2502.17416) (Saunshi et al., 2025). SKIM intro and main results: why looping adds effective depth at a fixed parameter count.

**Build:** nothing yet. Skim the stubs to see where each piece will go.

## 2. Normalization and the feed-forward layer

RMSNorm, where norms go in a modern block (pre-norm), and the SwiGLU
feed-forward layer.

**Read**
- [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) (Raschka, 2025). READ the DeepSeek V3 and OLMo 2 sections now (norm placement, RMSNorm); the rest is useful reference for sections 4 and 7.
- [RMSNorm](https://arxiv.org/abs/1910.07467) (Zhang and Sennrich, 2019). SKIM: why dropping LayerNorm's mean subtraction loses nothing.
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (Shazeer, 2020). READ (two pages): the gated feed-forward family and SwiGLU.

**Build:** `RMSNorm`, `Expert` (SwiGLU).

**Done when:** RMSNorm rows have unit RMS; `Expert` keeps the shape.

## 3. Positions: RoPE

How rotating queries and keys encodes relative position.

**Read**
- [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) (EleutherAI, 2021). READ: the intuition and the complex-number view.
- [RoFormer](https://arxiv.org/abs/2104.09864) (Su et al., 2021). SKIM the derivation: why relative position falls out of the dot product.

**Build:** `precompute_rope_freqs`, `apply_rope`.

**Done when:** RoPE preserves norms and leaves position 0 unchanged.

## 4. Attention: GQA and QK-norm

Grouped-query attention, and normalizing queries and keys for stability.

**Read**
- [GQA](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023). SKIM: sharing KV heads across groups of query heads, and the quality vs memory trade-off.
- The Big LLM Architecture Comparison (from section 2). READ the GQA explanation and the OLMo 2 QK-norm part.

**Build:** `GQAttention` without the cache: projections, QK-norm, RoPE, the manual path and the SDPA path.

**Done when:** SDPA and manual outputs match; causal attention ignores future tokens.

## 5. The KV cache

Why generation recomputes everything without a cache, and how the cache fixes
it.

**Read**
- [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) (Raschka, 2025). READ.

**Build:** the cache inside `GQAttention`.

**Done when:** the cache grows by one per step; token-by-token decoding through the cache matches one full pass.

## 6. FlashAttention

Exact attention without storing the score matrix: tiling, online softmax,
recomputation.

**Read**
- [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022). READ the introduction and the algorithm section (tiling and recomputation).
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (Milakov and Gimelshein, 2018). SKIM: the running-max trick FlashAttention builds on.
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) (Dao, 2023). SKIM: what changes from FA1 (work partitioning, fewer non-matmul operations).

**Build:** `tiled_attention`.

**Done when:** it matches SDPA, causal and not, including a sequence length that is not a multiple of the block size.

## 7. Mixture of experts

Routing, fine-grained and shared experts, and load balancing without an
auxiliary loss.

**Read**
- [Mixture of Experts Explained](https://huggingface.co/blog/moe) (Hugging Face, 2023). READ the sections from "What is a Mixture of Experts" through "Load balancing tokens for MoEs".
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) (Dai et al., 2024). READ fine-grained expert segmentation and shared-expert isolation.
- [Auxiliary-Loss-Free Load Balancing Strategy for MoE](https://arxiv.org/abs/2408.15664) (Wang et al., 2024). READ the method: bias steers selection, unbiased scores give the weights. DeepSeek-V3 adopted this.

**Build:** `MoEFFN` (router, routed and shared experts, bias update), `TransformerBlock`.

**Done when:** `router_bias` is a buffer; the bias update moves load toward balance; eval passes leave the load counter alone.

## 8. The loop: LTI injection, loop index, training recipe

The heart of the project: a stable loop that can be trained at random depths.

**Read**
- [Parcae](https://arxiv.org/abs/2604.12946) (Prairie et al., 2026; [blog](https://sandyresearch.github.io/parcae/)). READ: why looped models blow up, the negative-diagonal parameterization that keeps ρ(A) < 1, and the leveling-off test-time scaling curve.
- Huginn (from section 1). Re-read the training section: the loop-count distribution and truncated backprop.

**Build:** `loop_index_embedding`, `sample_n_loops`, `LTIInjection`, `RecurrentBlock` (including the naive `use_lti=False` arm).

**Done when:** ρ(A) < 1 even after a huge parameter step; the sampler's mean is right; truncated backprop changes gradients but not values; the naive arm runs.

## 9. The full model and generation

Wiring everything together, sampling, and caching across loops.

**Read**
- [How to generate text](https://huggingface.co/blog/how-to-generate) (Hugging Face, 2020). SKIM temperature and top-k sampling.
- [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) (Bae et al., 2025). SKIM the KV caching section: caching per recursion vs sharing KV from the first recursion.

**Build:** `Ouroboros`, `generate` (with and without the cache).

**Done when:** cached decode matches a full forward; cache on and off give the same tokens; 64 loops stay finite; head and embedding share weights.

## 10. Training

Tokenizer, data, and a standard single-GPU training loop.

**Read**
- [Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5) (Hugging Face LLM course). READ.
- [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan and Li, 2023). SKIM the intro: why tiny models can write coherent stories on this data.
- [nanoGPT `train.py`](https://github.com/karpathy/nanoGPT/blob/master/train.py) (Karpathy). READ the file: AdamW with weight-decay groups, warmup plus cosine schedule, gradient accumulation, fp16 with `GradScaler`, gradient clipping.

**Build:** `train_tokenizer`, `build_dataloader`, the training loop.

**Setup:** TinyStories packed into 512-token windows; AdamW (0.9, 0.95), weight decay 0.1 on matrices only, peak lr 1e-3 with warmup and cosine decay, clipping at 1.0; fp16 on a T4; one loop count sampled per optimizer step; router-bias update once per step; one seed.

**Watch while training:** loss, grad norm, the sampled loop count, ρ(A) with the mean and min of `A` (ρ(A) ≥ 1 would mean a bug), and expert load variance (should fall).

**Done when:** loss falls on the `--tiny` smoke run, then the main run is going.

## 11. Metrics and results

Three numbers for a one-paragraph write-up.

**Read**
- [SMELT](https://arxiv.org/abs/2609.01343) (Wang et al., 2026). READ the ablations (which layers to loop, how many times) and the attention-sink analysis; skip the scaling-law fits. It loops an MoE transformer like this one, at scale.
- Parcae (from section 8). Revisit the test-time scaling curve to compare against metric 1.

**Build and run:** `loop_sweep.py`, `kv_speedup.py`, the two stability runs.

1. **Test-time loop count** (the main one). Validation perplexity at 1, 2, 4, 8, 16, 32, and 64 loops from the trained model. Expect it to improve with more loops, level off past the training mean, and stay finite at 64. Result: the plot, and perplexity at 1 loop vs the best count.
2. **Stability.** Two short runs (about 1000 steps) at lr 3e-3, identical except the injection: LTI vs `--no-lti`. Clipping stays on for both, so it cannot explain the difference. Expect the naive run to spike or diverge. Result: the two loss curves and the divergence step, if any. If neither diverges, try 1e-2.
3. **KV-cache speedup.** Decode tokens/s with and without the cache, 8 loops, 256 new tokens. Result: the speedup factor.

**Done when:** a short Results paragraph with the three numbers is in the README.

```bash
python training/train.py --tiny --max-steps 50          # smoke run
python training/train.py --wandb-project ouroboros       # main run
python training/loop_sweep.py --ckpt checkpoints/final.pt
python training/kv_speedup.py --ckpt checkpoints/final.pt
python training/train.py --lr 3e-3 --max-steps 1000 --run-name lti-high-lr
python training/train.py --lr 3e-3 --max-steps 1000 --run-name naive-high-lr --no-lti
```
