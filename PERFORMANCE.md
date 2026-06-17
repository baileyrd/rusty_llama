# Performance: rusty_llama vs llama.cpp, and the improvement roadmap

A head-to-head benchmark against [llama.cpp](https://github.com/ggml-org/llama.cpp)
and the prioritized list of work it points to. Measured 2026-06-16.

## Setup

- **Machine:** Intel Core Ultra 9 285H (AVX2, **no** AVX-512), NVIDIA GeForce
  RTX 5070 Ti Laptop GPU (Blackwell, sm_120, 12 GB), CUDA 13.3 toolkit.
- **Model:** TinyLlama-1.1B-Chat v1.0 **Q4_K_M** (the same GGUF for both).
- **llama.cpp:** prebuilt release `b9672` (`llama-bench`), CUDA 13.3 / Vulkan /
  CPU builds. `pp512` = prefill throughput, `tg128` = single-token decode.
- **rusty_llama:** release build; decode measured as a 128-token greedy run with
  a 1-token prompt (decode-dominated), so it lines up with `tg128`.

## Results

### Decode (tok/s) — the clean apples-to-apples

| Path | Decode | vs rusty_llama |
| --- | ---: | --- |
| rusty_llama — CPU (autovectorized scalar) | 36.7 | — |
| rusty_llama — CPU (**AVX2 integer kernels**, roadmap #1) | **49.8** | — |
| llama.cpp — CPU | 73.6 | **1.48× faster** (was 2.1× before #1) |
| rusty_llama — GPU (wgpu) | 45.9 | — |
| llama.cpp — Vulkan (same GPU, same API class) | 375.9 | **8.2× faster** |
| llama.cpp — CUDA (NVIDIA-native) | 397.5 | **8.7× faster** |

### Prefill (tok/s)

llama.cpp: **CPU 5,890 · Vulkan 16,988 · CUDA 18,457**. rusty_llama's batched
prefill helps but uses no tensor cores, so it is one to two orders of magnitude
behind (synthetic prefill bench ~735 tok/s; the real-model figure isn't
separable from the CLI but is far below llama.cpp regardless).

## Why the gaps

- **CPU (~2×):** our integer fast path is **AVX-512 VNNI only** (`vpdpbusd`),
  which is dormant on this Core Ultra and on most consumer CPUs that lack
  AVX-512. We have **no AVX2 integer path** — just the VNNI kernels plus an
  autovectorized scalar loop. llama.cpp's AVX2 quant kernels (`vpmaddubsw` +
  `vpmaddwd`) and threading are simply better-tuned on this class of hardware.
- **GPU (~8× decode, ~24× prefill):** the decisive factor is **tensor cores**.
  llama.cpp's Vulkan backend reports `matrix cores: NV_coopmat2` and its CUDA
  backend uses native `mma`; both exploit the RTX 5070 Ti's tensor cores (plus
  flash attention and years of kernel tuning). Our wgpu (v29) cooperative GEMV
  is portable **f32 with no tensor-core path**, which dominates the prefill gap.

## Improvement roadmap (ranked by ROI vs the gaps above)

1. **CPU: hand-written AVX2 integer quant kernels — DONE.** `vpmaddubsw`+
   `vpmaddwd` int8 dot products (the AVX2 analog of `vpdpbusd`) for
   Q4_K/Q6_K/Q8_0/Q4_0, dispatched as VNNI → AVX2 → scalar (`x86::vec_dot_*_avx2`).
   4-/6-bit weights (Q4_K/Q6_K) don't saturate the i16 step so they're
   **bit-identical** to scalar; the signed 8-bit formats widen to i16
   (`vpmovsxbw` + `vpmaddwd`) to stay exact. **Run-verified on this machine**
   (`*_avx2_matches_scalar` tests) and measured **+36% CPU decode** on TinyLlama
   Q4_K_M (36.7 → 49.8 tok/s), narrowing the gap to llama.cpp's CPU from ~2.1× to
   ~1.48×. Toggle off with `RUSTY_LLAMA_NO_AVX2=1`.
2. **GPU: int8 (DP4A `dot4I8Packed`) decode — Stage 1 (Q8_0) DONE; Q4_K/Q6_K next.**
   Quantize each activation to int8 on-device (per-32 scale) and run the decode
   matmuls as `dot4I8Packed` cooperative GEMVs against relaid int8 weights (the
   GPU analog of the CPU integer path; the Vulkan device reports `int dot: 1`).
   Wired into the fused decode behind an all-Q8_0 eligibility gate (toggle off
   with `RUSTY_LLAMA_NO_INT8` / `set_int8_decode`), with per-op parity vs the
   exact integer dot and an end-to-end coherence check. Measured on a
   TinyLlama-1.1B-**shaped** synthetic Q8_0 model (dim 2048 × 22L, vocab 32k):
   decode **~59 → ~91 tok/s, ≈1.5–1.6×** over the in-shader dequant GEMV. That's
   below the ~2.5× single-kernel microbench because only the seven matmuls +
   classifier go int8 — rmsnorm/rope/attention/swiglu stay f32 and the path adds
   five quantize passes per step (Amdahl). **Stage 2:** Q4_K/Q6_K (TinyLlama
   Q4_K_M's real formats), kept *packed* in VRAM and unpacked-to-int8 in-shader
   before the DP4A (pre-expanding to int8 inflates weight bandwidth ~1.8× and can
   erase the win); only then does the real-model GPU decode row above move.
   Still well behind the tensor-core backends — that gap is item 3.
3. **GPU: tensor cores — the big swing (most of the prefill/decode gap).**
   Either wgpu's `EXPERIMENTAL_COOPERATIVE_MATRIX` (portable: Vulkan coopmat /
   Metal simdgroup-matrix, but experimental in wgpu 29 with an immature WGSL
   surface) or a NVIDIA-only **CUDA backend** (cuBLASLt / `mma`, most certain to
   deliver). This is where ~80% of the GPU gap lives.
4. **Flash attention** (tiled) for long context; **cache-blocked CPU prefill
   GEMM**.
5. **Breadth (usefulness, not speed):** more architectures (Qwen/Gemma/Phi/MoE),
   IQ-quants, KV-cache quantization, richer samplers (min-p, repetition penalty,
   GBNF grammars), batching / server.

### The honest framing

Matching llama.cpp's raw throughput is a treadmill — it is a mature,
tensor-core-tuned, perpetually-moving target, and an ~8k-LOC from-scratch
reference won't win that race. The worthwhile goal is to **close the most
glaring, bounded-effort gaps while staying legible**. By that measure item 1
(AVX2 integer kernels) is the first move: real payoff, broad hardware impact,
and fully verifiable on this machine.
