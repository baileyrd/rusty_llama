# Phase 7 — CUDA int8 MMQ prefill (BACKLOG L7)

A cold-start plan for the **one remaining bounded performance lever**: replacing
the dequant→f16-cuBLASLt prefill GEMM with a packed int8 tensor-core MMQ kernel,
the path llama.cpp uses and the entire ~3.6× CUDA prefill gap.

**This is days of the hardest kernel work in the repo, with an Amdahl-limited and
unproven ceiling.** It is staged so the cheap **Stage B measures the real ceiling
on this hardware before** the expensive Q4_K kernel is written. If Stage B doesn't
clear the bar, stop — that's a successful outcome, not a failure.

See [`PERFORMANCE.md`](../../../PERFORMANCE.md) item 3 and
[`BACKLOG.md`](../../../BACKLOG.md) L7 for the measured standing this serves.

---

## ⏹️ Outcome — executed through the Stage-B gate (2026-06-21): **NO-GO for bounded effort**

This plan was run through Stages A and B. **The kernels are proven correct on the
RTX 5070 Ti (sm_120); the Stage-B GO/NO-GO measurement says a hand-written MMQ is
~5× off cuBLASLt f16 and basic tuning does not close it — so the work is banked,
not continued.** Doing the cheap stages first to learn this *is* the plan working.

**What was proven correct** (PR #63, all behind `#[ignore]` tests, run with
`--features cuda -- --ignored`):
- **Stage A** (`mmq_stage_a_tiled_int8_gemm`): the proven `mma.sync.m16n8k32.s8`
  fragment generalized to a tiled, K-accumulating int8 GEMM — bit-exact vs host
  int8 (maxerr 0.0). Weights feed in native `W[n][k]` layout (no transpose).
- **Stage B** (`mmq_stage_b_q8_0` / `_tiled` / `_v2`): the per-block-scaled Q8_0
  MMQ (fresh mma per 32-block, scaled by `d_act·d_weight`, f32-accumulated),
  naive + shared-tiled + register-blocked — all bit-exact vs the host Q8_0 oracle
  (max rel ~1e-5) and within Q8_0 quant error of true f32.

**The GO/NO-GO measurement** (`mmq_stage_b_bench_vs_f16`, 512×2048×2048 — one pp512
matmul — vs `gemm_dev_f16`, the cuBLASLt f16 path MMQ would replace):

| Kernel | Time | vs f16 | Correct? |
|---|---:|---:|---|
| f16 cuBLASLt (baseline) | 0.116 ms | 1.00× | — |
| naive tiled MMQ | 0.525 ms | **0.22×** | rel L2 0.0042 |
| one tuning pass (v2: 64×64 tile, register-blocked, 4× fewer syncs) | 0.634 ms | **0.18×** | rel L2 0.0042 |

**Finding.** A *correct* MMQ is ~5× slower than cuBLASLt f16, and one reasonable
tuning pass made it **slower, not faster** — an occupancy/register-pressure
collapse (16 f32 accumulators + 18 KB shared/block dropped resident warps below
what hides latency), the classic ILP-vs-occupancy trap. The int8 ceiling is real
(the L7 spike measured cuBLASLt-*int8* at 2.49× f16), but a custom per-sub-block
kernel can't borrow cuBLAS's tuning, and reaching it is deep, iterative, expert
GEMM engineering (occupancy balance, `cp.async` double-buffering, vectorized/
swizzled loads) — the open-ended treadmill, with the e2e payoff still Amdahl-capped
at ~2× prefill. **Decision: bank the proven kernels, do not build Stage C.** The
correctness foundation is preserved for any future revisit.

The staged plan below stands as written; the Stage-B gate fired exactly as intended.

---

## The gap, precisely

| | rusty_llama | llama.cpp | Gap |
|---|---:|---:|---|
| CUDA prefill (pp512, TinyLlama Q4_K_M) | ~5,230 t/s | ~19,637 t/s | **~3.6–3.8×** |

Today (`src/backend/cuda.rs`):
- `weight_f16` (cuda.rs:632) **dequantizes each weight to a full f16 VRAM buffer**
  (~2.2 GB for TinyLlama) and caches it by source pointer.
- `gemm_dev_f16` (cuda.rs:717) runs every prefill matmul as cuBLASLt
  `Matmul<f16>`. This is structurally **the `ggml_cuda_op_mul_mat_cublas`
  fallback llama.cpp deliberately routes around.**

llama.cpp's MMQ keeps weights **packed** in global memory, dequantizes a tile into
shared/registers, and multiplies on **int8 tensor cores**
(`mma.sync.aligned.m16n8k32.s8.s8.s32`, ~2× f16 TOPS). That structural difference
is the gap.

---

## Why this is now de-risked (the hard unknowns are already retired)

MMQ here is mostly **assembly of proven parts** + one genuinely new piece
(shared-mem tiling that feeds tensor cores). What already exists and is parity-tested:

| Proven asset | Where | What it gives Phase 7 |
|---|---|---|
| Header-free int8 tensor-core op | `probe_int8_mma_ptx` (cuda.rs:2493) | `mma.sync.m16n8k32.s8.s8.s32` compiles + runs **bit-correct on sm_120**, 32-lane fragment layout hand-verified. The blocking unknown is gone. |
| Packed-weight device cache | `weight_packed` (cuda.rs:662) | weights already stay packed on device for the decode DP4A path — reuse instead of `weight_f16`. Also drops VRAM ~2.2 GB → ~0.6 GB. |
| Q4_K/Q6_K integer unpack + dot | `dot_q4_k_row` / `dot_q6_k_row` / `gemv_q4_k` (decode GEMV) | the exact per-sub-block unpack (`get_scale_min_k4`, min-correction `Σq·x − min·Σx`) is **already written and parity-tested** at the integer level vs the CPU oracle. MMQ feeds the same math to tensor cores instead of `__dp4a`. |
| Activation quant | `quantize_q8k` (Q8_K per-256 + bsums) | the correct, already-validated activation format for k-quant MMQ. |
| CPU oracles | `vec_dot_q8_0` / `vec_dot_q4_k` / `vec_dot_q6_k` (`src/quant.rs`) | bit-level parity references for every kernel. |
| Gating/A-B/bench harness | `RUSTY_LLAMA_CUDA_*` env toggles, `bench_prefill_real_tinyllama` | the measurement discipline to land each stage honestly. |

**The genuinely new work** is the tiled shared-memory staging + fragment loads that
keep the tensor cores fed, and expressing the per-sub-block scale/min in tensor-core
(int32-accumulator) form.

## A negative result to respect (do not re-litigate)

The **per-row int8 cuBLASLt spike** (`RUSTY_LLAMA_CUDA_INT8`, #49) got **+18% e2e
prefill but FAILED quality**: a per-row activation scale is too coarse for Q4_K's
per-32-sub-block scale (prefill coherence rel L2 0.098, real generation degenerates
on some prompts). The **per-sub-block MMQ is the quality-correct version** — its
whole point is matching Q4_K's sub-block scale granularity. Do **not** re-attempt
per-row int8; the +18% is the *floor* a clean MMQ must clear, not a target.

---

## Staging (each stage independently GPU-verifiable)

### Stage A — tiled int8×int8→f32 GEMM scaffold · ~½–1 day · ✅ DONE (bit-exact, maxerr 0.0)
Generalize `probe_int8_mma_ptx` from a single 16×8×32 fragment to a **tiled GEMM**
over real prefill shapes `(M=rows, N=oc, K=ic)`: int8 `A` (activation) and int8 `B`
(weight) in global memory + per-tile `f32` scales. Stage tiles into shared memory,
loop `k` in steps of 32 (one `mma.sync` per K-step), accumulate `s32`, scale to
`f32`. **No quant format yet** — pure plumbing.
- **Gate:** bit-correct vs a host int8-GEMM reference at a real shape (e.g.
  2048×2048×512). Proves the tiling / fragment layout / accumulation at GEMM scale.

### Stage B — Q8_0 MMQ + **THE CEILING MEASUREMENT** · ~1 day · **GO/NO-GO** · ⏹️ DONE → NO-GO (correct, but 0.18–0.22× of cuBLAS f16; see Outcome banner)
Feed Stage A from **packed Q8_0** weights (`weight_packed`; load the 32 `int8`
quants per block to shared, keep the per-32 `f16` scale) and Q8_0-quantized
activation. A Q8_0 block is exactly 32 elements = **one `mma.sync` K-step**; the
`s32` accumulator scales by `d_w[blk]·d_a[blk]`. Replace `gemm_dev_f16` for
Q8_0-weighted prefill matmuls behind `RUSTY_LLAMA_CUDA_MMQ`. Parity vs
`vec_dot_q8_0`.
- **Measure `pp512` on a Q8_0 TinyLlama** (requantize the GGUF to Q8_0, or the
  synthetic Q8_0 builder). This is the **simplest possible MMQ** and reports the
  **real e2e prefill ceiling on sm_120** before any Q4_K complexity.
- **Decision rule:** only the 7 matmuls + classifier go int8 — attention, rmsnorm,
  rope, swiglu stay f32 — so the GEMM speedup is **Amdahl-diluted**. Project the
  Q8_0 e2e gain onto Q4_K_M; if it doesn't clear the +18% per-row floor by a margin
  worth days of Q4_K kernel work, **STOP and document** (a legitimate, valuable
  negative result — you'll have proven the ceiling cheaply).

### Stage C — Q4_K / Q6_K MMQ · ~2–4 days · *only if Stage B clears the bar* · ⛔ NOT BUILT (Stage B was a NO-GO)
The real kernel. Reuse `dot_q4_k_row` / `dot_q6_k_row`'s per-sub-block unpack (8×32,
`get_scale_min_k4`, the min term `Σq·x = Σq_u·x − min·Σx`) but **stage the unpacked
int8 quants into shared and feed `mma.sync` per 32-K sub-block**, applying the
per-sub-block scale/min to the `s32` accumulator. Q8_K activation (`quantize_q8k`,
already written). Parity vs `vec_dot_q4_k` / `vec_dot_q6_k`.
- Where the days go: shared-mem layout + bank conflicts, expressing the min term in
  tensor-core form, occupancy/double-buffering tuning. This is llama.cpp-class
  kernel work.

### Stage D — wire into `forward_prefill` + coherence + real bench · ~½–1 day
Route the prefill matmuls (q/k/v, w1/w3, wo, w2, classifier) through MMQ, gated by
`RUSTY_LLAMA_CUDA_MMQ` with `gemm_dev_f16` as the A/B fallback. Drop the
`weight_f16` cache for MMQ-eligible models (the VRAM win). Full **prefill→decode
coherence** (rel L2 ≤6e-4, argmax matches the CPU oracle) and a real-model `pp512`
bench vs llama.cpp's 19,637.

---

## Acceptance & kill criteria (the repo's standing conventions)

- **Per-kernel:** bit-level integer parity vs the CPU `vec_dot_*` oracle (the
  reduction stays in integer arithmetic, so it can be exact — mirror the decode
  GEMV's parity tests).
- **End-to-end:** prefill→decode coherence rel L2 ≤6e-4, argmax == CPU.
- **Gated:** `RUSTY_LLAMA_CUDA_MMQ` on/off A/B toggle; f16 path stays as fallback.
- **Honest bench:** `pp512` vs `llama-bench` on the real TinyLlama Q4_K_M, same
  machine, warmup + r≥5 (the existing harness; `RunState` already hoisted).
- **Kill criterion (Stage B):** if the Q8_0 ceiling doesn't project to a Q4_K_M gain
  worth the Stage-C effort, stop and write it up. This is the point of staging.

## Realistic outcome — set expectations

Even fully landed, this closes prefill from ~3.6× to perhaps **~1.6–2.4× behind**,
not ahead: llama.cpp stacks MMQ **plus** prefill fusion and years of per-arch tuning
on top. Decode (~1.4×) is unaffected (bandwidth-bound, already on packed DP4A +
graph). The wins are: most of the prefill gap, and ~1.6 GB less VRAM (packed vs f16
weights). The goal remains **close, legibly** — not win the throughput race.

**Effort:** ~4–7 days total; the **Stage B GO/NO-GO gate lands ~1.5 days in**, so
the expensive Q4_K work is never started without a measured ceiling on this exact
hardware.
