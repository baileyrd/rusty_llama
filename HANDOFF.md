# Handoff: GPU int8/DP4A decode (roadmap #2)

A cold-start prompt for continuing the in-progress int8 GPU decode work. See
[`PERFORMANCE.md`](PERFORMANCE.md) for the benchmark + full roadmap this serves.

---

## ✅ Status update (2026-06-17): roadmap #2 (int8/DP4A) is DONE — next is #3

This handoff's task is complete; the rest of the doc is kept for context. All on
branch `feat/gpu-int8-probe`, **PR #19** (open, not merged):

- **Stage 1 — Q8_0 int8 decode: ADOPTED.** Wired into the fused decode behind an
  all-Q8_0 eligibility gate + toggle (`RUSTY_LLAMA_NO_INT8` / `set_int8_decode`).
  Measured **~1.5–1.6×** decode over the dequant path (TinyLlama-shaped synthetic
  Q8_0). Per-op parity (vs exact int dot) + acc=1 fold + gate/toggle + e2e
  coherence vs CPU.
- **Stage 2 — Q4_K/Q6_K int8: MEASURED, NOT WIRED.** The three k-quant int8
  kernels are built and **bit-exact** (per-op parity), but the gating microbench
  `bench_kquant_int8_vs_dequant_gemv` shows int8 at **~0.94–0.98× (Q4_K) / ~0.70×
  (Q6_K)** of the dequant GEMV — k-quants stay packed, so int8 saves no weight
  bandwidth and the in-shader unpack is pure added compute on a bandwidth-bound
  decode. Kept behind their tests (kill-criterion); Q4_K_M decode stays on
  dequant. **Don't re-litigate scalar DP4A for k-quants** — it loses here.

**Next session → roadmap #3: tensor cores** (the ~8× GPU decode / ~24× prefill
gap). Either wgpu `EXPERIMENTAL_COOPERATIVE_MATRIX` (portable, but immature WGSL
in wgpu 29) or a NVIDIA-only CUDA backend (`mma`/cuBLASLt, most certain). That's
where ~80% of the GPU gap lives; `dot4I8Packed` has been exhausted.

---

You're picking up **`rusty_llama`** — a from-scratch Llama inference engine in
Rust ("llama.cpp, the Rust way"). The CPU path is mature; over recent sessions a
full **GPU backend** (wgpu, behind the `gpu` cargo feature) was built and merged,
and CPU **AVX2/AVX-512 integer kernels** were added. You're continuing **GPU
decode performance work (roadmap #2: int8/DP4A)** — finishing **Stage 1** (a
partly-built feature branch).

## This is a local GPU machine — confirm hardware first
- **GPU:** NVIDIA RTX 5070 Ti Laptop (Blackwell, sm_120, 12 GB), CUDA 13.3
  toolkit installed; wgpu runs on Vulkan/DX12. `dot4I8Packed` (DP4A) works in
  wgpu 29 / naga with no special feature (device reports `int dot: 1`).
- **CPU:** Intel Core Ultra 9 285H — **AVX2 but NO AVX-512**. So the AVX-512 VNNI
  kernels (`x86::vec_dot_*`) are **dormant here** (run-verifiable only on CI / an
  AVX-512 host); the AVX2 kernels (`x86::vec_dot_*_avx2`) *do* run here. Confirm
  with `nvidia-smi` and an `is_x86_feature_detected!` probe.

## Read first, in order
- `PERFORMANCE.md` — the llama.cpp head-to-head (same machine/model) + the
  ROI-ranked roadmap. **North star.** On TinyLlama-1.1B Q4_K_M we're ~1.48×
  behind on CPU decode (after AVX2), ~8× on GPU decode, ~24× on prefill. The GPU
  gap is **tensor cores** (llama.cpp uses `NV_coopmat2`/CUDA `mma`; we don't).
- `src/backend/gpu.rs` (~2400 LOC, the big file) — the wgpu `GpuBackend`. The
  decode hot path is `build_decode_state` + `fused_step` (one command encoder per
  token, resident KV + activations, cooperative-GEMV matmuls with residual-add
  folded in). Weights are kept quantized on-device, dequantized in-shader.
- `src/quant.rs` — quant types + scalar/VNNI/AVX2 integer dot kernels
  (`vec_dot_*`, the `x86` module).
- `src/backend/mod.rs` (`Backend` trait), `src/model.rs` (`forward`,
  `forward_prefill`, `generate`).

## Branch state
- `main` (merged): GPU backend, batched prefill, fused decode, on-device quant,
  cooperative GEMV (GPU decode beats CPU here: ~46 vs ~35 tok/s), residual-fold
  fusion, Q6_K VNNI, AVX2 integer kernels (PRs #10, #12–#18).
- **`feat/gpu-int8-probe`** (NOT merged — your branch): (1) an int8 DP4A
  prototype microbench showing the int8 GEMV is **~2.5× faster** than the f32
  dequant GEMV for Q8_0; (2) **Stage 1 foundation** — production kernels
  `WGSL_QUANTIZE_Q8` (on-device f32→int8 + per-32 scale) and
  `WGSL_MATMUL_Q8_0_I8` (Q8_0 DP4A cooperative GEMV with `acc` residual-fold),
  each with passing GPU parity tests (`matmul_q8_0_i8_matches_int_dot`,
  `quantize_q8_kernel_is_valid`). The two pipeline fields are
  `#[allow(dead_code)]` until wired in.

## Task: finish Stage 1, then Stage 2

**Stage 1 — wire the int8 path into the fused decode (Q8_0) and measure.**
- In `build_decode_state`/`fused_step`, when the model's matmul weights are all
  Q8_0-eligible (and not disabled by a new `RUSTY_LLAMA_NO_INT8` env toggle for
  A/B): add resident int8 activation+scale buffers (`xb_i8`/`xb_scale`,
  `hb_i8`/`hb_scale`), run a `quantize_q8` pass before each matmul group
  (quantize once per activation, reused — 4 points/layer: after attn-rmsnorm,
  after attention, after ffn-rmsnorm, after swiglu), and use `matmul_q8_0_i8`
  bind groups (`acc=1` for `mo`/`m2`). Add `weight_buffer_i8` (Q8_0 → contiguous
  int8 + f32 scales, cached). Cleanest as a parallel `fused_step_int8` + a
  `DecodeInt8` bundle on `DecodeState`, so the f32 path stays untouched.
- Remove the `#[allow(dead_code)]` once wired. **Measure** real decode tok/s on a
  synthetic Q8_0 model (`dummy::synthetic_gguf_typed(.., GgmlType::Q8_0)`), int8
  vs f32 (toggle), and an e2e greedy coherence check (output won't be
  bit-identical — int8 activation error — but must stay coherent; prototype
  maxdiff ~0.07).
- Known Stage-1 shortcut to undo: it's fine to build both f32 and int8 weight
  buffers (2× weight VRAM) on the measurement model to get the number, then drop
  the f32 build when int8 is active.

**Stage 2 — int8 Q4_K + Q6_K.** Do **not** pre-expand the packed k-quant weights
to int8 — that inflates weight bandwidth ~1.8× for Q4_K and can erase the win
(decode is bandwidth-sensitive). Keep weights **packed** and unpack-to-int8
**in-shader** before `dot4I8Packed`, with the Q8K activation. Measure the
bandwidth/compute trade-off; this is where TinyLlama Q4_K_M's real formats live.

## Acceptance / conventions
- Per-op parity (int8 kernels vs the exact integer dot) + e2e coherence;
  `cargo test` and `cargo clippy --all-targets` clean **with and without**
  `--features gpu`. (`cargo fmt --check` is noisy — match style by hand.)
- Report decode tok/s honestly (int8 vs f32, and vs CPU). Expect end-to-end gain
  **< the 2.5× kernel number** (re-quantize passes + non-matmul ops dilute it).
- Feature branch, small commits ending with
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`; PR per
  stage; **don't merge without the human's say-so**.
- Re-benchmark vs llama.cpp:
  `llamacpp_bench/lc/llama-bench.exe -m tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -ngl 99`
  (CUDA) / `-ngl 0` (CPU); dir is gitignored. The TinyLlama GGUF +
  `stories15M.bin`/`tokenizer.bin` are already in the repo root.

Start by confirming the GPU, checking out `feat/gpu-int8-probe`, reading the
files above and the current `build_decode_state`/`fused_step`, then propose a
short plan for the int8 fused-decode wiring before coding.
