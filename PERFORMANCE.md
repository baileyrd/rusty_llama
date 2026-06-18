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
2. **GPU: int8 (DP4A `dot4I8Packed`) decode — Stage 1 (Q8_0) DONE; Stage 2 (Q4_K/Q6_K) measured, not adopted.**
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
   five quantize passes per step (Amdahl).

   **Stage 2 (Q4_K/Q6_K) — built, proven, and measured *not* worth adopting.**
   The three int8 kernels (Q8K-activation quantize + Q4_K/Q6_K unpack-in-shader
   `dot4I8Packed`) are implemented and **bit-exact** against the CPU integer-dot
   oracles (`vec_dot_q4_k`/`vec_dot_q6_k`). But k-quant weights stay *packed*, so
   the int8 GEMV streams the **same bytes** as the dequant GEMV — int8 saves **no
   weight bandwidth** — and decode is bandwidth-bound, so the in-shader unpack +
   DP4A is pure added compute. The gating microbench
   (`bench_kquant_int8_vs_dequant_gemv`) measures int8 at **~0.94–0.98× (Q4_K)**
   and **~0.70× (Q6_K)** of the dequant GEMV on this GPU — break-even to slower.
   Per the pre-agreed kill-criterion the kernels are **kept (behind their tests)
   but not wired**, so TinyLlama Q4_K_M decode stays on the dequant path. The
   lesson: scalar DP4A only helps when ALU-bound; over already-bandwidth-minimal
   4-/6-bit weights the real GPU decode win needs **tensor cores** (item 3), not
   `dot4I8Packed`. (Pre-expanding k-quants to int8 in VRAM would inflate weight
   bandwidth ~1.8× and lose outright — never attempted.)
3. **GPU: tensor cores — the big swing (most of the prefill/decode gap).**
   Either wgpu's `EXPERIMENTAL_COOPERATIVE_MATRIX` (portable: Vulkan coopmat /
   Metal simdgroup-matrix, but experimental in wgpu 29 with an immature WGSL
   surface) or a NVIDIA-only **CUDA backend** (cuBLASLt / `mma`, most certain to
   deliver). This is where ~80% of the GPU gap lives.

   **Portable wgpu coopmat path — spiked and ruled out (on this hardware, wgpu
   29).** A cheap go/no-go probe (`probe_cooperative_matrix`, an ignored test in
   `gpu.rs`) found the portable path silently broken here:
   - The RTX 5070 Ti (Blackwell, Vulkan `VK_KHR_cooperative_matrix`) advertises
     **only f16-input configs** — 16×16×16, 16×8×16, 16×8×8 with f16 A/B and
     f16-or-f32 accumulate. It exposes **no 8×8 f32 config**.
   - wgpu 29's coopmat backend **only wires 8×8 f32** (per its own feature doc).
     So the one shape wgpu supports, this driver doesn't expose; the shapes this
     driver exposes (f16 16×16), wgpu doesn't wire.
   - Worst of all, requesting an 8×8 f32 `coopMultiplyAdd` **compiles and runs
     with no validation error** (after the `ExperimentalFeatures` token) yet
     returns **all-zero garbage** — a silent correctness failure, exactly the
     "experimental, may be UB" warning made real.

   Conclusion: the portable coopmat path is a dead end until wgpu wires f16
   16×16 (and naga's WGSL surface matures). The real tensor-core win on this
   machine needs the **NVIDIA-only CUDA backend** (`mma` / cuBLASLt).
   `dot4I8Packed` (item 2) is also exhausted.

   **CUDA backend (cudarc + cuBLASLt) — M0/M1 done, M2a kernels landed.** A third
   `Backend` impl (`src/backend/cuda.rs`, behind the `cuda` cargo feature; cudarc
   `dynamic-loading` so it needs no CUDA libs/nvcc at build time and fails
   gracefully on a CUDA-less host). The `forward_prefill` matmuls + classifier run
   on **cuBLASLt TF32 tensor cores** (`Matmul<f32>` → `CUBLAS_COMPUTE_32F_FAST_TF32`;
   dequantized f32 weights cached on-device). Pure f32 throughout — no f16, so no
   `cuda_fp16.h`/header dependency, and the per-call host f16 conversion that
   throttled the first cut is gone. Per-op parity vs the CPU oracle (maxdiff
   ~6e-3) + prefill coherence (relative L2 ~3e-4, top token matches) confirm the
   column-major `x·Wᵀ` layout. **Measured prefill (synthetic, 256 tok, dim 1024 ×
   4L, same shape for all three): CPU ~70–280 · wgpu f32 GEMV 533 · CUDA TF32
   ~2800 tok/s** (CPU baseline is single-shot and noisy). Still under llama.cpp's
   CUDA prefill (~18.5k on the real Q4_K_M), because this is synthetic f32 and the
   per-op path still round-trips activations to the host per matmul.

   **M2a** adds the on-device primitives to remove those round-trips: five nvrtc
   kernels (add, swiglu, rmsnorm, rope, attention), each parity-verified against
   the CPU op (maxdiff ≤ 3e-7). **M2b (next)** assembles them + the TF32 GEMMs
   into a resident fused `forward_prefill` (activations + KV cache stay on
   device, only embeddings up / logits down), then re-benches the real TinyLlama
   Q4_K_M vs `llama-bench`. Decode stays on the CPU/existing paths (batch=1 is
   GEMV/bandwidth-bound).
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
