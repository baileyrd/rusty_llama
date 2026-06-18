# Handoff: GPU tensor cores (roadmap #3)

A cold-start prompt for the tensor-core GPU work — the ~8× decode / ~24× prefill
gap vs llama.cpp. See [`PERFORMANCE.md`](PERFORMANCE.md) for the benchmark + full
roadmap this serves.

---

## ✅ Status update (2026-06-17): #2 merged; #3 portable path spiked + ruled out

- **Roadmap #2 (int8/DP4A) — DONE and merged to `main` (PR #19).** Stage 1 (Q8_0
  int8 decode, ~1.5–1.6×) is adopted behind an all-Q8_0 gate + `RUSTY_LLAMA_NO_INT8`
  toggle. Stage 2 (Q4_K/Q6_K int8) is built + bit-exact but **not wired** — on a
  bandwidth-bound decode the packed k-quant unpack erases the win (~0.94–0.98×
  Q4_K, ~0.70× Q6_K). **Don't re-litigate scalar DP4A for k-quants — it loses.**

- **Roadmap #3 (tensor cores) — portable wgpu coopmat path SPIKED and RULED OUT
  on this hardware.** Branch `feat/gpu-tensor-cores`. A go/no-go probe
  (`probe_cooperative_matrix`, ignored test in `src/backend/gpu.rs`) found that
  the RTX 5070 Ti / Vulkan advertises **only f16 coopmat configs** (16×16, 16×8),
  while wgpu 29 **only wires 8×8 f32** — and an 8×8 f32 `coopMultiplyAdd`
  compiles + runs with **no error but returns all-zero garbage**. Silent
  correctness failure → dead end until wgpu wires f16 16×16. Full write-up in
  `PERFORMANCE.md` item 3.

- **Roadmap #3 (tensor cores) — CUDA backend M0 DONE.** FFI decision settled:
  **cudarc 0.19** (dynamic-loading — no CUDA libs/nvcc at build time). A third
  `Backend` impl `CudaBackend` lives in `src/backend/cuda.rs` behind a new `cuda`
  cargo feature (off by default, NVIDIA-only). M0 = device + cuBLASLt handle init
  with a graceful no-CUDA fallback (gated on `cudarc::driver::sys::is_culib_present`
  because `CudaContext::new` *panics* on a CUDA-less host), all ops delegated to
  `CpuBackend` for now. `--backend cuda` wired; smoke test passes on the RTX 5070
  Ti (device detected + htod/dtoh round-trip). `cargo test`/`clippy --all-targets`
  clean with and without `--features cuda`.

- **Roadmap #3 — CUDA backend M1 DONE + M2a DONE.**
  - **M1:** `CudaBackend::matmul`/`matmul_batch` run on **cuBLASLt TF32 tensor
    cores** (`Matmul<f32>`; dequantized f32 weights cached on-device by data
    pointer). Pure f32 — switched off f16 to drop the `cuda_fp16.h` dependency and
    the per-call host f16 conversion. Layout `out(n,oc)=x(n,ic)·Wᵀ` =
    `transa=true, m=oc, n=rows, k=ic, lda=ldb=ic, ldc=oc, matmul(cfg, a=W, b=X,
    c=out)` (in [`CudaBackend::gemm_dev`]) — verified by per-op parity + prefill
    coherence (rel L2 ~3e-4). **Per-op prefill ~2800 tok/s** (256 tok, dim 1024
    ×4L) vs wgpu 533, CPU noisy.
  - **M2a:** five nvrtc kernels (add/swiglu/rmsnorm/rope/attention) in `KERNEL_SRC`
    + `dev_*` launchers, each parity-verified vs the CPU op (maxdiff ≤ 3e-7).
  - **M2b:** `Backend::forward_prefill` (defaulted to the free fn; `generate`
    dispatches through it) overridden on `CudaBackend` with a **resident fused
    prefill** — embeddings up once, `x`/`xb`/`q`/`hb` + per-layer KV resident,
    TF32 GEMMs + the `dev_*` kernels, logits down. The prompt's K/V are copied
    back to host `state` (`RunState::store_prefill_kv`) so CPU decode is correct;
    guarded by `prefill_then_decode_coherent` (rel L2 ~3e-4). Handles `pos_base==0`
    (delegates otherwise). **Fused prefill ~3300 tok/s** (256 tok, dim 1024 ×4L)
    vs ~2800 per-op TF32, 533 wgpu.

- **Real-model bench + tiled attention + f16 weights DONE.**
  `bench_prefill_real_tinyllama` loads the actual TinyLlama-1.1B Q4_K_M (22
  layers, dim 2048, GQA 32/4), prefills 512 tokens. Progression on real-model
  prefill vs llama.cpp's **19,637**:
  - naive attention + f32 weights: **744 tok/s** (~26× behind);
  - **cooperative block-per-(row,head)** `attention_kernel` (threads split the key
    dot-products + output head-dims; scores+softmax in shared mem; global `att`
    scratch removed): **~2,700** (~7.3×);
  - **f16 weights + `Matmul<f16>`** (host dequant→f16 cache; activations narrowed
    f32↔f16 on-device via inline-PTX `cvt` kernels — no `cuda_fp16.h`; half the
    weight bytes/VRAM + faster f16 tensor cores): **~3,560 tok/s, ~5.5× behind**.
  - **Batched KV download:** each layer keeps its own resident device K/V (Vec of
    buffers), so the whole prefill runs with no in-loop `synchronize`; the KV is
    copied to host once after the loop for the decode handoff: **~4,450 tok/s,
    ~4.4× behind**.
  - **Per-op overhead cut:** disabled cudarc event tracking (single-stream, so
    it's pure overhead) + non-zeroing alloc for the fully-overwritten GEMM scratch.
    Real model ~unchanged (**~4,500**, it's compute-bound) but the small synthetic
    4-layer proxy went **16.7k → 25k** — so this helps op-heavy small models.
  Correctness held throughout (parity + prefill/decode coherence, rel L2 ≤6e-4).
  CPU baseline is throttle-noisy under sustained load — compare CUDA-vs-llama.cpp.

- **CUDA fused decode DONE.** `CudaBackend::forward_step` now runs a resident
  single-token decode (mirrors the wgpu `DecodeState`): a lazily-built
  `DecodeCuda` holds the per-layer KV cache + activation scratch + RMSNorm
  weights on device; a `kv_filled` counter uploads the prompt's host KV once on
  the first step; each step does embedding→layers (batch-1 GEMVs via `gemm_dev`
  + the M2 kernels; K/V appended to row `pos` via `slice_mut`/`memcpy_dtod`)→
  classifier→logits download. **Real TinyLlama Q4_K_M tg128: CUDA ~123 tok/s vs
  CPU ~52 (2.4×), vs llama.cpp 419 (~3.4× behind).** Coherence: multi-step +
  prefill→decode vs CPU, rel L2 ≤6e-4/step.

- **Dequant-in-kernel decode GEMV (Q4_K/Q6_K) — built, proven, NOT adopted.** To
  cut the decode weight bandwidth, `gemv_q4_k`/`gemv_q6_k` stream the **packed**
  bytes + dequant inline (bit-exact: `gemv_q4_k_parity`/`gemv_q6_k_parity`). But
  the naive one-block-per-row kernel runs **~78 tok/s vs f16's ~121 (~1.6×
  SLOWER)** — poorly-coalesced byte loads + per-element dequant erase the
  bandwidth saving and don't approach cuBLASLt's tuned f16 GEMV. Kept behind the
  `RUSTY_LLAMA_CUDA_DEQUANT` toggle (off by default); decode stays on f16.
  **Don't re-litigate a naive dequant GEMV** — the win needs a warp-cooperative,
  coalesced kernel (cf. llama.cpp `mmvq`).

**KEY FINDING: the ~3.4× decode gap is weight bandwidth, but capturing it needs a
*tuned* dequant GEMV (warp-cooperative + coalesced), not a naive one.**

**Next session → if pursuing decode further: a warp-cooperative dequant GEMV.**
Restructure `gemv_q4_k`/`gemv_q6_k` so a warp processes a row with **coalesced**
loads (consecutive threads read consecutive packed bytes) + vectorized
dequant — the only way the ~3.5× fewer weight bytes beats cuBLASLt f16. This is a
real kernel-tuning effort. Lower-ROI: keep KV resident across prefill→decode
(drop the one-time host upload); warp-level/flash attention. Prefill is near the
cuBLASLt wall (~4.4×). `dot4I8Packed` and the portable wgpu coopmat path are
exhausted. Or pivot to **breadth** (roadmap #5: more architectures, samplers).

---

You're picking up **`rusty_llama`** — a from-scratch Llama inference engine in
Rust ("llama.cpp, the Rust way"). The CPU path is mature; a full **GPU backend**
(wgpu, behind the `gpu` cargo feature) is built + merged, CPU **AVX2/AVX-512
integer kernels** and **GPU int8/DP4A decode** are done. You're starting
**roadmap #3: tensor cores** — the single biggest remaining gap vs llama.cpp.

## This is a local GPU machine — confirm hardware first
- **GPU:** NVIDIA RTX 5070 Ti Laptop (Blackwell, sm_120, 12 GB), CUDA 13.3
  toolkit installed (`nvcc`/cuBLAS available); wgpu runs on Vulkan/DX12.
- **Coopmat caveat (already proven):** on this GPU the Vulkan
  `VK_KHR_cooperative_matrix` advertises **only f16 configs** (16×16/16×8); wgpu
  29 only wires **8×8 f32**, which compiles but **silently returns zeros**. So
  tensor cores here mean **CUDA**, not portable wgpu coopmat. See `PERFORMANCE.md`
  item 3 and the `probe_cooperative_matrix` test in `src/backend/gpu.rs`.
- **CPU:** Intel Core Ultra 9 285H — **AVX2 but NO AVX-512** (VNNI kernels dormant
  here; AVX2 ones run). Confirm with `nvidia-smi` and an
  `is_x86_feature_detected!` probe.

## Read first, in order
- `PERFORMANCE.md` — the llama.cpp head-to-head + ROI-ranked roadmap. **North
  star.** TinyLlama-1.1B Q4_K_M: ~1.48× behind on CPU decode, ~8× on GPU decode,
  **~24× on prefill** — the prefill GEMM gap is the clearest tensor-core win.
- `src/backend/mod.rs` (`Backend` trait — every op a backend must implement) and
  `src/model.rs` (`forward`, `forward_prefill`, `generate`) — how a backend is
  selected and driven. A CUDA backend is a **third `Backend` impl** alongside
  CPU/GPU, behind a new `cuda` cargo feature.
- `src/backend/gpu.rs` — the wgpu `GpuBackend` for reference: resident-weight
  caching keyed on data pointer, batched prefill (`matmul_batch`, the GEMM that
  tensor cores should accelerate), fused decode (`build_decode_state`/`fused_step`).
- `src/quant.rs` — quant types + dequant (CUDA will need weights on-device too).

## Branch state
- `main` (merged): CPU + wgpu GPU backends, batched prefill, fused decode,
  on-device quant, cooperative GEMV, AVX2 integer kernels, int8/DP4A Q8_0 decode
  (PRs #10–#19).
- **`feat/gpu-tensor-cores`** (your branch): the coopmat go/no-go spike only —
  `probe_cooperative_matrix` + the `PERFORMANCE.md`/`HANDOFF.md` write-ups ruling
  out the portable path. No CUDA code yet.

## Task: NVIDIA CUDA backend — prefill GEMM first

**Settle the FFI strategy first** (open question): the [`cudarc`] crate (safe-ish
Rust bindings to the CUDA driver API + cuBLAS, no `nvcc` at build time) vs a
hand-rolled `build.rs` + `cc`/`nvcc` compiling `.cu` kernels. `cudarc` is likely
the faster, more legible path and matches the "Rust way" ethos; hand-rolled gives
full control over custom `mma` kernels. Decide, then put everything behind a new
`cuda` cargo feature (off by default — the portable wgpu/CPU build must stay
dependency-light and keep building/testing without CUDA).

**Then build the backend incrementally:**
1. A `CudaBackend` skeleton implementing `Backend`, device init + graceful
   "no CUDA device" fallback (mirror `GpuBackend::new`'s `Result`).
2. **Prefill GEMM via cuBLASLt (f16 tensor cores)** — the ~24× gap and the
   highest-confidence win. Dequant weights to f16 on-device (cached, pointer-keyed
   like the wgpu path), run the prefill matmuls through cuBLASLt. Per-op parity
   vs the CPU backend + an e2e coherence check.
3. Only after prefill lands, consider decode (batch=1 GEMV is bandwidth-bound —
   tensor cores help less; may stay on a tuned GEMV/int8 path).

## Acceptance / conventions
- Per-op parity (new kernels vs the CPU backend, the oracle) + e2e coherence;
  `cargo test` and `cargo clippy --all-targets` clean **with and without** each
  GPU feature (`gpu`, and the new `cuda`). (`cargo fmt --check` is noisy — match
  style by hand.)
- Report prefill/decode tok/s honestly (CUDA vs wgpu vs CPU, same model).
- Feature branch, small commits ending with
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`; PR per
  milestone; **don't merge without the human's say-so**.
- Re-benchmark vs llama.cpp:
  `llamacpp_bench/lc/llama-bench.exe -m tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -ngl 99`
  (CUDA) / `-ngl 0` (CPU); dir is gitignored. The TinyLlama GGUF +
  `stories15M.bin`/`tokenizer.bin` are already in the repo root.

[`cudarc`]: https://crates.io/crates/cudarc

Start by confirming the GPU, checking out `feat/gpu-int8-probe`, reading the
files above and the current `build_decode_state`/`fused_step`, then propose a
short plan for the int8 fused-decode wiring before coding.
