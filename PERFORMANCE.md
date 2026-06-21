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
- **Benchmark protocol (roadmap #0.3):** the in-crate throughput benches follow
  `llama-bench`'s protocol — one untimed warmup, then `-r` timed repetitions
  (default 5, override with `RUSTY_LLAMA_BENCH_R`) reported as **mean ± stddev**
  tok/s; tokenize/sampling are excluded (token ids are fed in directly). Run them
  with `cargo test --release --features cuda --lib -- --ignored --nocapture bench_decode_real_tinyllama bench_prefill_real_tinyllama`
  (`RUSTY_LLAMA_GGUF` overrides the model path), and compare against
  `llama-bench -m <gguf> -ngl 99 -p 512 -n 128 -r 5`.

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
behind. The batched prefill matmul is now register-tiled (each thread computes
TN=4 output rows, every weight read once), lifting the synthetic f32 prefill bench
~728 → ~2,170 tok/s (~3.0×) and real TinyLlama Q4_K_M prefill ~42 → ~207 tok/s
(~4.9×) — still far below llama.cpp's tensor-core Vulkan/CUDA regardless.

## Why the gaps

- **CPU (~1.5×):** we now have a hand-written **AVX2 integer path**
  (`vpmaddubsw` + `vpmaddwd`) alongside the **AVX-512 VNNI** kernels (`vpdpbusd`,
  dormant on this Core Ultra and on consumer CPUs lacking AVX-512), dispatched
  VNNI → AVX2 → scalar (roadmap #1, done). The residual gap is that llama.cpp's
  AVX2 quant kernels and threading are still better-tuned on this class of
  hardware.
- **GPU (~8× decode, ~8× prefill):** the decisive factor is **tensor cores**.
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

   **M2a + M2b done — resident fused prefill.** Five nvrtc kernels (add, swiglu,
   rmsnorm, rope, attention), each parity-verified vs the CPU op (maxdiff ≤ 3e-7),
   are assembled with the TF32 GEMMs into a `Backend::forward_prefill` override
   (`generate` dispatches through it): a prompt runs entirely on-device — `x`/`xb`/
   `q`/`hb` + the per-layer KV stay resident, only the token embeddings go up and
   the final logits come down. The prompt's K/V are copied back to the host KV
   cache so subsequent CPU decode is correct (guarded by a prefill→decode
   coherence test, rel L2 ~3e-4). **Fused prefill: ~3300 tok/s** on the synthetic
   shape (256 tok, dim 1024 × 4L) — up from ~2800 (per-op TF32) and 533 (wgpu).

   **Real-model result (the north star), measured on the actual TinyLlama-1.1B
   Q4_K_M — 22 layers, dim 2048, GQA 32/4 — prefill 512 tokens, same machine:**

   | Path | Prefill (pp512) | vs llama.cpp |
   | --- | ---: | --- |
   | rusty_llama — CUDA, naive attention + f32 weights | 744 tok/s | ~26× behind |
   | rusty_llama — CUDA, tiled attention + f32 weights | ~2,700 tok/s | ~7.3× behind |
   | rusty_llama — CUDA, tiled attention + f16 weights | ~3,560 tok/s | ~5.5× behind |
   | rusty_llama — **CUDA, + batched KV download** | ~4,450 tok/s | ~4.4× behind |
   | rusty_llama — **CUDA, + conversion-kill (shared f16 narrow)** | **~5,230 tok/s** | **~3.8× behind** |
   | llama.cpp — CUDA (`llama-bench`, fresh) | **19,637 tok/s** | — |

   Three changes closed most of the gap from the first cut's ~26×. (1) Replacing
   the naive one-thread-per-(row,head) attention with a **cooperative
   block-per-(row,head) kernel** (threads split the key dot-products and the output
   head-dims; scores + softmax reduction in shared memory, no global scratch):
   **744 → ~2,700 tok/s** (the synthetic 4-layer proxy past 10k) — attention was
   the dominant cost. (2) Caching weights as **f16** and running the GEMM as
   `Matmul<f16>` (activations narrowed f32↔f16 on-device via inline-PTX `cvt`
   kernels — no `cuda_fp16.h`): half the weight bandwidth + VRAM, faster f16 tensor
   cores: **~2,700 → ~3,560**. (3) **Batching the per-layer KV download** — each
   layer keeps its own resident device K/V so the whole prefill runs with no
   in-loop `synchronize`; the KV is copied to host once at the end for the decode
   handoff: **~3,560 → ~4,450**. A further per-op overhead cut (disable cudarc's
   single-stream-unnecessary event tracking; skip the memset on fully-overwritten
   GEMM scratch) left the real model ~unchanged (**~4,500 tok/s**) but lifted the
   small synthetic 4-layer proxy **16.7k → 25k** — i.e. **real-model prefill is now
   compute-bound** (per-op overhead is a small fraction of the big GEMMs), while
   op-heavy small models still gained. A later conversion-kill (sharing the per-norm f16 narrow across q/k/v and w1/w3) lifted real-model pp512 ~4,671 → ~5,230 t/s. Now **~3.8× behind llama.cpp**, correctness
   held throughout (per-op parity + prefill/decode coherence, rel L2 ≤6e-4). The
   CPU prefill baseline here is unreliable (laptop thermal throttling under
   sustained GPU load), so CUDA-vs-llama.cpp is the meaningful comparison.

   The residual ~3.8× on real-model prefill is now the **f16-cuBLAS-vs-int8-MMQ**
   structural gap (llama.cpp doesn't use cuBLAS for quantized prefill), not overhead.
   The L7 gate confirmed a header-free int8 tensor-core `mma.sync` is feasible on
   sm_120 (`probe_int8_mma_ptx`), but the full per-sub-block MMQ kernel is not yet built.

   **Decode (`tg128`) — packed-weight DP4A GEMV (Phase 2, adopted).**
   `CudaBackend::forward_step` runs a resident single-token decode (KV cache +
   activations stay on device across steps). Decode is batch-1 GEMV —
   bandwidth-bound, every weight read once — so the lever is **weight bytes
   streamed**. We replaced the f16 cuBLASLt GEMV (~2 B/weight) with a
   warp-cooperative, **coalesced packed-Q4_K/Q6_K `__dp4a` GEMV** that reads the
   weights packed (~0.56 B/weight for Q4_K) and dot-products them against an
   on-device Q8_K-quantized activation (`quantize_q8k` + `gemv_q4_k`/`gemv_q6_k`,
   mirroring the CPU `vec_dot_q4_k`/`vec_dot_q6_k` oracles at the integer level).
   Measured on the real TinyLlama-1.1B Q4_K_M (same session,
   `bench_decode_real_tinyllama`):

   | Path | Decode (tg128) | |
   | --- | ---: | --- |
   | rusty_llama — CPU | ~50 tok/s | — |
   | rusty_llama — CUDA (resident, **f16** GEMV) | ~86 tok/s | 1.7× CPU |
   | rusty_llama — **CUDA (resident, packed DP4A)** | ~207 tok/s | 2.4× over f16 |
   | rusty_llama — **CUDA (resident, DP4A + captured CUDA graph)** | **~275 tok/s** | — |
   | llama.cpp — CUDA (`llama-bench`) | ~415 tok/s | **~1.5× ahead** |

   The per-GEMV microbench (`bench_decode_gemv_quant_vs_f16`) shows **3.5–4.2×**
   at the decode shapes — the full bandwidth ratio; end-to-end is 2.4× because the
   non-GEMV ops (attention, rmsnorm, rope, embedding/logits transfers) don't speed
   up (Amdahl). This is the **inverse** of the wgpu Stage-2 dead-end (item 2):
   there int8 was compared against an *already-packed* dequant GEMV (same bytes →
   no bandwidth delta); here the baseline is **f16** weights, so packing is a real
   ~3.5× saving. Default-on for Q4_K/Q6_K decode; opt out with
   `RUSTY_LLAMA_CUDA_NO_QGEMV`. Correctness: integer-core parity vs the CPU dot +
   multi-step / prefill→decode coherence (rel L2 ≤6e-4). **Phase 2.2** also made
   the prefill→decode KV handoff device-resident (the one-time host round-trip is
   gone). **Phase 6.2** then captured the per-step kernel sequence into a CUDA graph
   (replayed per token on a capturable non-NULL stream), lifting decode to ~275 t/s.
   The residual ~1.5× to llama.cpp is **GEMV/kernel tuning** — flash
   attention (item 4) was measured decode-neutral, ruling attention out as the
   residual decode cost.
4. **Flash (online-softmax) attention — DONE (Phase 5.2), decode-neutral.** A
   single streaming pass with a running max/sum/accumulator, so no
   `seq_len`-sized score buffer: the CPU oracle drops its per-head `att` scratch,
   and the CUDA resident kernel (cooperative, tiled) no longer caps context by
   shared memory (validated to 16 384 tokens). Decode tok/s unchanged within
   noise (attention is a small slice of decode); adopted for the long-context /
   memory win, not speed. Still open: **cache-blocked CPU prefill GEMM**. Plan:
   `docs/Architecture/plans/phase-5-flash-attention.md`.
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
