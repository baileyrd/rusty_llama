# Phase 6 — Performance review & the road to llama.cpp parity

An independent, on-machine performance review of `rusty_llama` vs `llama.cpp`
(2026-06-19), the parity verdict it produced, and the ranked, ROI-ordered work
plan that follows from it. This supersedes the "high-ROI levers are spent"
framing in `PERFORMANCE.md`/`HANDOFF.md` for the gaps it re-diagnoses.

Machine: Intel Core Ultra 9 285H (AVX2, **no** AVX-512) + NVIDIA RTX 5070 Ti
Laptop (Blackwell sm_120, 12 GB, ~672–896 GB/s GDDR7), CUDA 13.3. Model:
TinyLlama-1.1B-Chat **Q4_K_M** (22 layers, dim 2048, GQA 32/4, vocab 32000,
ffn 5632). Baselines: `llama-bench` build `b9672`, freshly re-run on this box.

─────────────────────────────────────────────────────────────────────────────

## Verdict

**rusty_llama is closer to llama.cpp than its own docs claim, and the dominant
remaining gaps are recoverable engineering overhead — not fundamental hardware
walls.** The repo docs repeatedly conclude "at the cuBLAS wall / it's the tensor
cores / levers are spent." That attribution is wrong in four of the five biggest
gaps. The *parity-is-a-treadmill* framing is correct **for sustained, general
parity**; it is misapplied to the snapshot decode/CPU gaps, which are tractable.

The decisive fact: **it is the same hardware and the same math.** llama.cpp runs
the same CUDA driver, the same cuBLAS/nvrtc, the same AVX2 instructions. The one
real *wall* (wgpu 29 not wiring f16 coopmat) the CUDA path already sidesteps.

## Method

Every number below was reproduced on this machine: `llama-bench` re-run fresh for
baselines; the in-crate `#[ignore]` benches re-run for rusty_llama; the CPU
decode taken from a fresh cold-CPU CLI process; the hot paths in
`cpu.rs`/`cuda.rs`/`gpu.rs`/`model.rs`/`quant.rs` read directly; and six parallel
code-grounded subsystem audits. The repo's own perf markdown was treated as
claims to verify, not ground truth.

## Ground truth (fresh, same machine, same GGUF)

| Path | rusty_llama | llama.cpp | Real gap | Docs claimed | Diagnosis |
|---|---:|---:|---:|---:|---|
| CPU decode | 47.9 t/s | 64.8 t/s | **1.35×** | 1.48× | recoverable (un-tuned kernels) |
| **CPU prefill** (pp512) | ~44 t/s | 6047 t/s | **~130×** | buried as "❌" | **fully recoverable (zero batching)** |
| CUDA decode (tg128) | 191 t/s | 378 t/s | **1.98×** | 2.0× | recoverable (no CUDA graphs) |
| CUDA prefill (pp512) | 4671 t/s | 15860 t/s | **3.4×** | 4.4× | mostly recoverable (f16 not int8) |
| wgpu decode | 45.9 t/s | 375.9 t/s | ~8× | 8.2× | recoverable (launch overhead, **not** tensor cores) |

Fresh llama.cpp numbers are **lower** than the docs cite (15860 vs 19637; 378 vs
415; 64.8 vs 73.6), so most gaps are already smaller than documented. The one the
docs hide is CPU prefill.

## The decisive evidence: roofline

Decode is batch-1 GEMV → every weight read once → **bandwidth-bound**. Neither
engine is near the ceiling:

- **CUDA decode:** rusty 0.667 GB/tok × 191 = **127 GB/s (~14–19% of peak)**;
  llama.cpp 0.667 × 378 = **252 GB/s (~28–37%)**. Both far below peak ⇒
  latency/launch-bound. rusty loses ~half its achievable bandwidth to
  between-kernel bubbles (~5.4 µs × ~445 launches/step ≈ 2.4 ms ≈ 50% of the
  4.83 ms step). The packed GEMV microbench (`bench_decode_gemv_quant_vs_f16`)
  already hits 3.5–4.2× (the full byte ratio) — so the kernels are fine; the loss
  is **between** kernels. llama.cpp erases this with **CUDA graphs** (one replay /
  token) — never mentioned in the repo docs.
- **CPU decode:** rusty ~33 GB/s, llama.cpp ~49 GB/s, DDR5 ceiling ~70 GB/s.
  Same RAM, same AVX2 — llama.cpp just keeps the bus fuller.

## Findings, ranked by ROI

| # | Finding | Impact | Effort | Evidence |
|---|---|---|---|---|
| 1 | **CPU prefill does zero batching** — runs at decode speed | ~50–130× | Med | `cpu.rs:48-235` (no `matmul_batch` override) → `mod.rs:143-148` default loops single-row → `model.rs:1171-1240` |
| 2 | **CUDA decode: no CUDA graphs** — ~445 launches/step | +100–130 t/s | Med | `cuda.rs:1143-1197` (445 submits, 1 sync, no capture) |
| 3 | **CUDA prefill: f16 cuBLAS, not int8 MMQ** | ~1.3–1.8× | Large | `cuda.rs:641-682`, `weight_f16` `:571-596` |
| 4 | **CUDA prefill: 310 redundant f32↔f16 cvt passes** | ~1.05–1.15× | Sm-Med | `cuda.rs:652-653,681,1266-1268,1281-1282` |
| 5 | **CPU decode: AVX2 dots un-tuned** (1 accumulator, many hsums) | part of 1.35× | Med | `quant.rs:895-928,967-976` |
| 6 | **CPU rope recomputes cos/sin per head** (~32× redundant) | ~1–2% | Small | `cpu.rs:124-143` |
| 7 | **MoE: per-token alloc + no batched expert GEMM** | MoE-only | Med | `model.rs:1524,1220-1232` |
| 8 | **wgpu: ~300 serial compute passes/step** (fresh pass/op) | recoverable | Large | `gpu.rs:1064-1071` |

### #1 detail — the buried headline

`CpuBackend` overrides **none** of the batched ops, so `prefill_residual`'s
`matmul_batch` falls through to the default that loops single-row `matmul`
(`mod.rs:143-148`). Each of N prompt tokens re-streams the **entire** quantized
weight set from RAM: pp512 ≈ 512 × 0.62 GB ≈ **317 GB** of traffic vs ~0.62 GB
with cache-blocking. Measured CPU prefill (44 t/s) ≈ CPU decode (48), while
llama.cpp prefill is **93×** its own decode. **Algorithmic, not thermal** (the
docs blame throttling — but thermal cannot make prefill slower than decode).

## Parity analysis

"Parity" is not one number. Per path:

| Path | Bound by | Parity reachable? | Realistic best | What it takes |
|---|---|---|---|---|
| CUDA decode | launch latency (solvable) | **Yes** | ~1.0–1.2× | CUDA graphs + coalesced GEMV |
| CPU decode | memory BW (shared ceiling) | **Yes** | ~1.0–1.2× | tuned AVX2 dots + P-core threading |
| CPU prefill | FMA compute (shared) | **Near** | ~1.3–1.5× | cache-blocked int8 GEMM |
| CUDA prefill | int8 tensor-core kernel quality | **Hardest** | ~1.5–2× | MMQ-class int8 kernels |

- **Bandwidth-bound paths are the reachable ones** (counter to the docs, which
  call decode "near-walled"): the hardware ceiling is shared and llama.cpp is
  nowhere near saturating it on GPU. The bottleneck is a *solved* problem (CUDA
  graphs) or the same-ISA kernel-quality gap.
- **CUDA prefill is the genuinely hard one:** llama.cpp's edge is hand-tuned
  int8 tensor-core MMQ. Bounded effort (cuBLASLt IMMA int8 + conversion-kill +
  attention coalescing) reaches ~1.5–2×; the last stretch to parity *is* the
  treadmill.
- **Snapshot parity** (this model, this GPU, today) is reachable on 3 of 4 paths;
  **sustained general parity** (every arch/quant/GPU, kept matched) is **not**
  realistic for an ~11k-LOC legible engine, and shouldn't be the goal.
- **Tension:** CUDA-decode parity needs device-resident position + graph capture
  + a tuned GEMV — each erodes the "stay legible" value. Parity and legibility
  partly trade off.

**Recommended goal:** *within ~1.2× on the bandwidth-bound paths, within ~2× on
CUDA prefill, while staying legible* — defensible and finishable.

─────────────────────────────────────────────────────────────────────────────

## Phase 6 work plan (ROI order)

Each item: own branch → parity/coherence test + before/after bench on this
machine → PR → merge to `main` before the next.

### 6.1 — CPU cache-blocked prefill GEMM  *(highest ROI)*
Override `matmul_batch` (and `forward_prefill` if needed) on `CpuBackend` with a
cache-blocked int8 GEMM: quantize all N activation rows once to Q8/Q8_K, then
**outer-loop weight super-blocks (stream each once)**, inner-loop the N quantized
columns doing the existing `vec_dot_*` integer dots, tiled by output-row blocks
(~32–64) to stay in L2. Mirrors llama.cpp's chunked-row GEMM.
- **Measured (DONE, 2026-06-19):** clean A/B on this machine (rigorous harness,
  `RUSTY_LLAMA_NO_BLOCKED_GEMM` toggle) — real-model `pp512` **64 → 113 tok/s
  (1.77×)**, decode unchanged. **Far below the projected ~130×**: the workload is
  compute-bound on the un-tuned `vec_dot` AVX2 kernel (~1.4 MACs/cycle, ~45×
  below peak), so removing the redundant weight RAM traffic only recovers the
  fraction that wasn't already hidden behind slow compute. **Batching is
  necessary but the dominant CPU lever is kernel efficiency (6.4); they compound**
  — once the dot is near-peak the GEMM becomes RAM-bound and this blocking pays
  the larger multiple. Still ~53× behind llama.cpp's 6047 (kernel + tiling).
- **Acceptance:** per-op parity vs current `matmul_batch` (bit-identical for the
  int8 formats, since the dots are unchanged) + e2e greedy coherence; before/after
  `bench_prefill_real_tinyllama` (cpu arm) and `llama-bench -ngl 0 -p 512`.

### 6.2 — CUDA-graph decode capture
Capture the per-step kernel sequence once into a `cudarc` `CudaGraph` and replay
per token. Prereqs: (1) move off the legacy NULL stream (`ctx.new_stream()`);
(2) kill per-step kernel-arg drift (`pos`, `off = pos*kv_dim`) — make position
device-resident so one graph replays unchanged.
- **Target:** 191 → ~300–330 t/s (~1.1× behind).
- **Acceptance:** multi-step + prefill→decode coherence vs CPU (rel L2 ≤ 6e-4);
  before/after `bench_decode_real_tinyllama` and `llama-bench -ngl 99 -n 128`.

### 6.3 — CUDA prefill: conversion-kill (+ int8 attempt)
First the bounded win: share each narrowed activation across sibling GEMMs and
fold f16-narrowing into rmsnorm/swiglu (removes ~310 cvt passes/prefill); cache
the ~7 cuBLASLt plans. Then attempt the int8 path (on-device q8k activation +
cuBLASLt IMMA `CUDA_R_8I`/`COMPUTE_32I`, or the existing packed-weight dp4a
kernels) for the GEMMs.
- **Target:** 3.4× → ~1.5–2×.
- **Acceptance:** per-op parity + prefill coherence; before/after
  `bench_prefill_real_tinyllama` and `llama-bench -ngl 99 -p 512`.

### 6.4 — CPU decode AVX2 kernel tuning
Multiple independent vector accumulators + single horizontal reduction per row in
`vec_dot_q4_k`/`q6_k`/`q8_0` AVX2 paths (hide vpmaddwd latency); precompute the
rope cos/sin per position once (`cpu.rs` rope) instead of per head.
- **Target:** 1.35× → ~1.1×.
- **Acceptance:** **bit-exact** parity vs scalar oracle (`*_avx2_matches_scalar`)
  preserved; before/after CPU decode bench.

## Bench-methodology fixes (fold into the relevant PR)
- Add a real-model **CPU/wgpu** decode bench in the rigorous harness (warmup + r
  reps); today the "clean" CPU 49.8 / wgpu 45.9 come from the CLI (include
  sampling + detok + `stdout.flush()`), which *penalizes* rusty_llama.
- CPU baseline inside the CUDA bench uses `reps=1, warmup=0` after CUDA heats the
  package (`cuda.rs:1626-1628`) → contaminated; give it warmup+reps.
- CUDA prefill bench zeroes a ~92 MB host KV cache (`RunState::new`) *inside* the
  timed region (`cuda.rs:1727`) → understates prefill ~3–8%; hoist/reuse.

## Doc-drift / correctness (low severity, verified; fix opportunistically)
- `PERFORMANCE.md:125`, `HANDOFF.md:37-43` call the prefill GEMM "TF32/`Matmul<f32>`";
  it is f16 inputs / f32-accumulate (`cuda.rs:678`, cudarc `safe.rs:449`).
- `gpu.rs:16-18` module doc says quantized weights are dequantized to f32 on host;
  false — kept **packed**, dequantized in-shader (`gpu.rs:1409-1421`).
- `PERFORMANCE.md` "Why the gaps" still says "no AVX2 integer path" — contradicted
  by its own roadmap item 1.
- No functional bugs found; `unsafe` non-zeroing allocs in `gemm_dev` verified
  safe; prefill→decode f16-KV numeric path is parity-guarded.
