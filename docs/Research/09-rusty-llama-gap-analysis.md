# 09. rusty_llama vs llama.cpp: Gap Analysis & ROI Roadmap

## Summary

This is the synthesis doc: what `rusty_llama` has today, where it diverges from
llama.cpp (subsystem by subsystem), and — the point of the whole exercise — a
**ROI-ranked** list of what to build next. The headline: `rusty_llama` already
shares llama.cpp's core architecture (backend abstraction over a Llama forward
pass) and is within ~3–4× on its one model; the gaps are mostly **breadth**
(1 architecture vs 132, 4 samplers vs ~18, no server) plus **one depth lever**
that genuinely closes the decode speed gap (a packed-weight DP4A GEMV). Most
breadth is cheap and high-value; raw-throughput parity is a treadmill.

─────────────────────────────────────────────────────────────────────────────

## Where rusty_llama stands today

Grounded in its own source (not aspiration):

| Subsystem | Current state | Source |
|---|---|---|
| Compute model | **Eager, per-op** via a `Backend` trait (`rmsnorm`, `matmul`, `rope`, `attention`, `swiglu`, `add` + batched variants + `forward_step`/`forward_prefill`); no op graph | `src/backend/mod.rs`, `src/model.rs` |
| Backends | CPU (rayon + AVX2 integer quant kernels), wgpu GPU (f32 + int8 Q8_0 decode), CUDA (cuBLASLt f16 prefill + resident f16 decode) | `src/backend/{cpu,gpu,cuda}.rs` |
| Quant (consume) | F32, F16, Q4_0, Q8_0, Q4_K, Q6_K; CPU has Q8/Q8_K activation int8-dot path | `src/quant.rs`, `src/backend/cpu.rs` |
| Quant (produce) | none (consumes externally-made GGUFs) | — |
| GGUF | single-file mmap (v2/v3, full 13-variant metadata) + legacy llama2.c | `src/gguf.rs`, `src/loader.rs` |
| Architectures | **Llama family only** (GQA, RoPE incl. YaRN/llama3 scaling, SwiGLU, RMSNorm); arch chosen by name-prefix, not a registry | `src/model.rs`, `src/config.rs` |
| Tokenizers | SPM (llama) + GPT-2 byte-level BPE (gpt2) | `src/tokenizer.rs` |
| Sampler | greedy / temperature / multinomial / top-p only, xorshift RNG | `src/sampler.rs` |
| KV-cache | **single sequence**, contiguous per-layer K/V buffer in `RunState`; CUDA keeps it resident with a one-time host upload | `src/model.rs`, `src/backend/cuda.rs` |
| Serving / tools | CLI (`llama-cli`-style) + a benchmark harness; no server, multimodal, LoRA, embeddings | `src/main.rs` |
| Perf (TinyLlama-1.1B Q4_K_M, RTX 5070 Ti) | prefill ~**4.4×** behind, decode ~**3.4×** behind llama.cpp | `PERFORMANCE.md` |

─────────────────────────────────────────────────────────────────────────────

## Capability matrix

✅ have · 🟡 partial · ❌ missing — vs the llama.cpp surface in `00`.

| Capability | rusty_llama | Notes / detail doc |
|---|---|---|
| Backend abstraction | ✅ | trait, not a graph+scheduler; `01`/`02` |
| CPU SIMD quant kernels | 🟡 | AVX2 int8 dot; no AVX-512/AMX/repack; `02` |
| CUDA tensor-core **prefill** | 🟡 | cuBLASLt f16 (= llama.cpp's *avoided* fallback), not MMQ int8; `03` |
| CUDA **decode** GEMV | 🟡 | f16 GEMV (~2 B/wt); not packed-weight DP4A (MMVQ); `03` |
| Flash attention | ❌ | tiled cooperative attn, no online-softmax FA; `03` |
| k-quant consume (Q4_K/Q6_K/Q8_0) | ✅ | `04` |
| i-quants / MXFP4 / ternary | ❌ | `04` |
| imatrix / quantize tooling | ❌ | offline tooling; low ROI; `04` |
| GGUF load | ✅ | `05` |
| Sharded GGUF / async upload / validation | ❌ | `05` |
| Architectures | 🟡 | Llama only (1 vs 132); `05` |
| HF→GGUF conversion | ❌ | correctly out of scope; `05` |
| Paged / multi-sequence KV-cache | ❌ | single contiguous buffer; `06` |
| Continuous batching | ❌ | `06` |
| Context shift / SWA / recurrent KV | ❌ | `06` |
| Speculative / n-gram decoding | ❌ | `06` |
| Tokenizers (SPM + BPE) | ✅ | the two that matter for Llama; `07` |
| Tokenizers (WPM/UGM/RWKV/PLaMo) | ❌ | only needed with those archs; `07` |
| Samplers (top-k, min-p, penalties, mirostat, DRY, XTC…) | ❌ | has 4 of ~18; `07` |
| Sampler-chain abstraction | ❌ | monolithic `Sampler`; `07` |
| GBNF grammar / JSON-schema | ❌ | `07` |
| OpenAI/Anthropic server | ❌ | `08` |
| Multimodal | ❌ | `08` |
| LoRA / control vectors | ❌ | `08` |
| Chat templates / tool calling | ❌ | `08` |
| Embeddings / reranking | ❌ | `08` |
| Distributed (RPC) / multi-GPU | ❌ | `08`/`02` |

─────────────────────────────────────────────────────────────────────────────

## The performance gap, decomposed

The ~3–4× is **two different problems** (see `00` and `03`):

- **Decode (~3.4×) is pure weight bandwidth.** `rusty_llama` streams f16 weights
  (~2 B/weight); llama.cpp's MMVQ streams packed Q4_K (~0.56 B/weight) and
  dequantizes in-register with `__dp4a`. The ratio ≈3.5× ≈ the measured gap.
  This is *not* a tuning problem — it is a format-on-the-wire problem.
- **Prefill (~4.4×) is matmul-kernel quality.** `rusty_llama` dequantizes to f16
  and calls cuBLASLt — which is exactly the path (`to_fp16_cuda` + cuBLAS
  GemmEx) that llama.cpp's MMQ deliberately *avoids*. MMQ keeps weights packed
  and uses int8 tensor cores (`mma.sync m16n8k32 s8`).

A prior naive attempt at a packed dequant GEMV (`rusty_llama` closed PR #23) lost
to f16 by ~1.6× — but `03` shows *why*: it lacked warp-cooperative coalesced
loads + q8_1 activation quantization. llama.cpp's 419 tok/s decode *is* that
kernel, so the ceiling is real; the difficulty is in the kernel craft, not the
existence of the win.

─────────────────────────────────────────────────────────────────────────────

## ROI-ranked roadmap

Ordered by (value ÷ effort). Effort tags are honest engineering estimates.

### Tier 1 — cheap breadth, do first

1. **Sampler chain + the missing samplers** — refactor `Sampler` into a
   composable chain (llama.cpp's `llama_sampler_i` apply/accept), then add
   **min-p, top-k, and repetition/frequency/presence penalties**; mirostat/DRY/
   XTC later. *Effort: low. Value: high* — these are the most-requested knobs and
   the default chain order is documented in `07`.
2. **`llama-bench` protocol parity** — mirror its exact methodology (time only
   `decode`, exclude tokenize+sample, warmup, `-r` averaging, `-d` depth) so the
   "3–4× behind" number is apples-to-apples and future wins show cleanly.
   *Effort: trivial. Value: measurement hygiene.* (`08`)
3. **Embeddings / pooling mode** — same forward pass + a pooling head +
   normalize; a `--embedding`/`--pooling` flag opens RAG use. *Effort: low.* (`08`)
4. **LoRA + control vectors** — LoRA is a `scale·B·(A·x)` add on matmul outputs
   with per-request scaling; control vectors are a per-layer `+= dir`. Both are
   small given GGUF loading already exists. *Effort: low–medium. Value: good.* (`08`)

### Tier 2 — the depth lever that actually moves the speed needle

5. **Packed-weight DP4A decode GEMV (MMVQ-style)** — the *only* change that
   closes the decode bandwidth gap. Port the design from `03`: q8_1 activation
   quantization once per step, `vec_dot_q4_K_q8_1`/`q6_K` unpack-in-register +
   `__dp4a`, **warp-cooperative coalesced** loads, weights stay packed in VRAM.
   Scope tight: Blackwell sm_120, Q4_K/Q6_K, batch-1. *Effort: high (real kernel
   work). Value: high, proven ceiling (~3.5×).* This is the answer to "can't we
   learn from llama.cpp" — yes, and this is the thing to learn.
6. **MMQ-style int8 tensor-core prefill** — beats the cuBLASLt wall but harder
   than MMVQ; do it only after decode lands. *Effort: high. Value: medium* (prefill
   is rarer than decode in interactive use). (`03`)

### Tier 3 — large but high-value (capability jumps)

7. **Paged, multi-sequence KV-cache → continuous batching → server.** These
   stack: a paged KV with `seq_id` (`06`) is the prerequisite for continuous
   batching, which is the prerequisite for a useful `llama-server`. A minimal
   `/v1/chat/completions` + `/completion` + `/health` is a reasonable first
   milestone. *Effort: high. Value: the biggest "product" jump.* (`06`/`08`)
8. **GBNF grammar + JSON-schema structured output** — depends on Tier-1's
   sampler layer; then constrained decoding is a per-step token mask. Frequently
   requested. *Effort: medium–high.* (`07`)
9. **More architectures via a registry pattern** — Qwen2/Gemma/Phi are
   near-Llama (mostly norm/rope/activation variants) and cheap once an arch
   registry (à la `llama-arch.cpp`, `05`) replaces the name-prefix dispatch;
   MoE/Mamba/RWKV are genuinely new graphs. *Effort: low per Llama-like arch,
   high for new families.* (`05`)

### Tier 4 — defer or treadmill

- **Multimodal** — needs a separate clip/ViT encoder + projector +
  embedding-splicing; long-horizon. (`08`)
- **Distributed / RPC / tensor-parallel** — premature for a single box; add
  multi-GPU `layer` split first if ever. (`02`/`08`)
- **i-quants / MXFP4 / quantize tooling** — format breadth with low ROI unless a
  target model requires it; an inference engine only consumes GGUFs. (`04`)
- **Portable wgpu cooperative-matrix** — already spiked and **ruled out** on this
  hardware (silent zeros); CUDA is the tensor-core path. (`PERFORMANCE.md`, `03`)
- **Graph / define-then-run rewrite** — a multi-week re-architecture; the eager
  `Backend` trait is not the current bottleneck. Borrow only the *cheap* ideas
  (strided-view tensor type, quant-trait table, inplace-reuse rule). (`01`)

─────────────────────────────────────────────────────────────────────────────

## Honest framing

Matching llama.cpp's raw throughput is a treadmill: it is a mature,
tensor-core-tuned, perpetually-moving target with a decade of per-architecture
micro-tuning (the per-arch warp tables in `03`, the 132-arch registry in `05`,
the ~30 quant formats in `04`). An ~8k-LOC from-scratch Rust engine will not win
that race, and shouldn't try to.

The worthwhile goals, in order:
1. **Cheap breadth that makes it *useful*** (Tier 1) — samplers, LoRA,
   embeddings, honest benchmarking. Days of work, broad payoff.
2. **The one speed lever with a proven ceiling** (Tier 2, MMVQ decode GEMV) —
   the only place where the packed-weight bandwidth edge actually beats f16.
3. **The product jump** (Tier 3, paged-KV → server) when capability-as-a-service
   matters more than raw single-stream speed.

Everything in Tier 4 is either a research dead-end already proven (wgpu coopmat)
or breadth with poor ROI for a single-box Llama engine.

─────────────────────────────────────────────────────────────────────────────

## Provenance

Synthesized from `00`–`08` (llama.cpp snapshot commit
`fe7c8b2414bb39e3dbc192fb494a1eb054b90890`, 2026-06-18) and from `rusty_llama`'s
own source + `PERFORMANCE.md`/`HANDOFF.md` as cited above.
