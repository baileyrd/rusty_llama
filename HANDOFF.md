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

- **Roadmap #3 — CUDA backend M1 DONE: prefill tensor cores live.**
  `CudaBackend::matmul`/`matmul_batch` now run on **cuBLASLt f16** (f16 weights
  cached on-device keyed by data pointer, f32 accumulate); other ops still
  delegate to CPU. The column-major `out(n,oc)=x(n,ic)·Wᵀ` layout is `transa=true,
  m=oc, n=rows, k=ic, lda=ldb=ic, ldc=oc, matmul(cfg, a=W, b=X, c=out)` — verified
  by per-op parity (maxdiff ~5e-3) + e2e prefill coherence (rel L2 ~6e-4, top
  token matches). **Measured prefill (256 tok, dim 1024 × 4L): CPU ~70 · wgpu 533
  · CUDA 1304 tok/s** (~2.45× wgpu, ~18× CPU). Benches/tests are ignored
  (`bench_prefill_cuda_vs_cpu`, `*_parity`, `prefill_coherent_with_cpu`).

**Next session → M2: kill the per-op host↔device copies.** M1 is honest but
leaves throughput on the table — every prefill matmul re-uploads its activation
tile and reads the result back. Move the elementwise/attention ops
(rmsnorm/rope/attention/swiglu/add) onto CUDA (cudarc kernels via nvrtc/PTX, or
hand `.cu`) and keep the activation/KV buffers **resident** across a prefill, so
a layer runs without host round-trips — mirror the wgpu fused-prefill approach.
Then **re-bench the real TinyLlama Q4_K_M vs `llama-bench`** (the north-star
number). Decode stays on CPU/existing paths (batch=1 GEMV is bandwidth-bound).
`dot4I8Packed` (item 2) and the portable wgpu coopmat path are both exhausted.

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
