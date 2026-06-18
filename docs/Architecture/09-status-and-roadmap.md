# 09. Status & Roadmap

## Summary

Where `rusty_llama` actually stands today, what the roadmap has burned down, the
current head-to-head numbers, and what's worth doing next. The CPU path is mature
(within 1.48× of llama.cpp); the wgpu and CUDA backends are built and merged; the
tensor-core CUDA backend is the speed frontier (prefill ~4.4× behind at the
cuBLASLt wall, decode ~3.4× behind at the f16-bandwidth wall). The high-ROI
bounded-effort speed levers are largely spent; the open question is **breadth vs
the one remaining decode lever** — covered here and, from the llama.cpp side, in
`../Research/09-rusty-llama-gap-analysis.md`.

Sources: `PERFORMANCE.md`, `HANDOFF.md`, `BACKLOG.md`, and this capture's
subsystem docs (`01`–`08`). Snapshot: `main` @ `e2a7d5d`, 2026-06-18.

─────────────────────────────────────────────────────────────────────────────

## What's done (roadmap burndown)

The roadmap is defined in `PERFORMANCE.md` §"Improvement roadmap". Status:

| # | Item | Status |
|---|---|---|
| 1 | **CPU AVX2 integer quant kernels** | ✅ DONE — `vpmaddubsw`+`vpmaddwd` int8 dot for Q4_K/Q6_K/Q8_0/Q4_0, dispatched VNNI→AVX2→scalar (`src/quant.rs`); +36% CPU decode (36.7→49.8 tok/s), bit-identical to scalar. VNNI path present but **dormant** on the dev box (no AVX-512). NEON still TODO. (`BACKLOG.md` #3) |
| 2 | **GPU int8 / DP4A decode** | 🟡 PARTIAL — Stage 1 (Q8_0) adopted behind an all-Q8_0 gate (~1.5–1.6×). Stage 2 (Q4_K/Q6_K) built + **bit-exact** but **intentionally not wired**: on bandwidth-bound decode the packed-kquant unpack saves no bytes (~0.94–0.98× Q4_K, ~0.70× Q6_K). Kept behind tests per a pre-agreed kill-criterion. (`02`/`03`/`05`) |
| 3 | **GPU tensor cores** | ✅ (CUDA) / ❌ (portable) — portable wgpu coopmat **ruled out** on this hardware (advertises only f16 16×16; wgpu 29 wires only 8×8 f32 → silent all-zero garbage). CUDA backend M0→M2b + resident decode **DONE and merged** (PRs #20–#22). Closed PR #23: a naive packed dequant-GEMV was ~1.6× *slower* than f16. (`03`/`04`) |
| 4 | Flash attention; cache-blocked CPU prefill | ❌ not started |
| 5 | **Breadth** (archs, samplers, KV-quant, server) | ❌ not started |

─────────────────────────────────────────────────────────────────────────────

## Performance: current numbers

TinyLlama-1.1B Q4_K_M; Core Ultra 9 285H (AVX2, no AVX-512) + RTX 5070 Ti
(Blackwell sm_120, 12 GB), CUDA 13.3; vs llama.cpp `b9672`. From `PERFORMANCE.md`.

| Path | Metric | rusty_llama | llama.cpp | Gap |
|---|---|---:|---:|---|
| CPU | decode | 49.8 tok/s | 73.6 | **1.48×** behind |
| wgpu | decode | 45.9 tok/s | 375.9 (Vulkan) | 8.2× behind |
| CUDA | prefill (pp512) | ~4,450 tok/s | 19,637 | **~4.4×** behind |
| CUDA | decode (tg128) | ~123 tok/s | 419 | **~3.4×** behind |

Two distinct walls (see `04` and `../Research/03-cuda-kernels.md`):
- **Decode ~3.4× = weight bandwidth.** Batch-1 GEMV reads each weight once; we
  stream f16 (~2 B/weight) vs llama.cpp's packed Q4_K (~0.56 B). ≈3.5× ≈ the gap.
- **Prefill ~4.4× = GEMM kernel quality.** cuBLASLt `Matmul<f16>` + our nvrtc
  kernels vs llama.cpp's fused int8 tensor-core MMQ. Now compute-bound (per-op
  overhead already cut), so further gains require beating cuBLASLt.

─────────────────────────────────────────────────────────────────────────────

## Capability snapshot

| Area | State | Doc |
|---|---|---|
| Architectures | Llama family only (name-prefix dispatch, not a registry) | `01`,`06` |
| Quant (consume) | F32/F16/Q4_0/Q8_0/Q4_K/Q6_K | `05` |
| Quant (produce) | only `quantize_q8_0` (test fixtures); no quantize-to-disk | `05` |
| Tokenizers | SPM + GPT-2 BPE (GPT-2/Llama-3/Qwen2 pretokenizers) | `07` |
| Samplers | greedy / temperature / multinomial / top-p (4 behaviors) | `07` |
| KV-cache | single sequence, contiguous per-layer f32; resident on GPU/CUDA | `01`,`03`,`04` |
| Loading | single-file GGUF (v2/3) + llama2.c; mmap zero-copy; no sharding/validation/async | `06` |
| Backends | CPU (oracle) + wgpu + CUDA | `02`,`03`,`04` |
| Serving / multimodal / LoRA / embeddings / grammar | none | — |

─────────────────────────────────────────────────────────────────────────────

## Doc-drift & cleanup found during this capture

Concrete, low-risk fixes the capture surfaced (code is correct; comments lie):

1. **`src/model.rs:133-135`** — `from_gguf`'s doc-comment claims partial rotary
   and RoPE scaling are "not yet handled," but both are implemented
   (`rope_dim` `model.rs:156`, `read_rope_scaling` `model.rs:245`). Stale comment.
2. **`src/backend/cuda.rs`** — stale "TF32" comments (e.g. near `:8`, `:803`); the
   GEMM is now `Matmul<f16>` (f32 accumulate), not TF32 (`gemm_dev` `cuda.rs:407`).
   The M1 path was TF32; M-later switched to f16. Comments not updated.
3. **`PERFORMANCE.md` §"Why the gaps"** — states "our integer fast path is
   AVX-512 VNNI only … We have no AVX2 integer path," which the roadmap section
   just below it (item 1, DONE) supersedes. The current code has both VNNI and
   AVX2. The diagnosis prose predates the AVX2 work.

(Intentional, *not* drift: the Stage-2 Q4_K/Q6_K int8 GPU kernels are
`#[allow(dead_code)]` on purpose — kept behind tests per the kill-criterion.)

These are documentation-only; none affect behavior. Fixing them is a clean
follow-up, not part of this capture.

─────────────────────────────────────────────────────────────────────────────

## Forward roadmap

Two framings, reconciled. `HANDOFF.md` (written from inside the speed work)
ranks breadth highest now that the bounded speed levers are spent;
`../Research/09-rusty-llama-gap-analysis.md` (written after reading llama.cpp's
kernels) agrees on breadth-first but notes the decode lever is **more bounded
than HANDOFF assumed** — llama.cpp's `mmvq` is a concrete, proven design to port.

**Tier 1 — cheap breadth, do first** (`../Research/09`, `07`):
- Refactor `Sampler` into a chain + add **min-p, top-k, repetition/freq/presence
  penalties** (4 behaviors → the common set). Low effort, high user value.
- **`llama-bench` protocol parity** in the bench harness (already close; `08`).
- **Embeddings/pooling mode**; **LoRA + control vectors** (GGUF loading exists).

**Tier 2 — the one decode speed lever with a proven ceiling** (`04`, `../Research/03`):
- **Warp-cooperative, coalesced dequant GEMV (MMVQ-style)**: q8_1 activation
  quant once/step, `vec_dot_q4_K_q8_1`-style unpack-in-register + DP4A, weights
  stay packed. The only thing that closes the ~3.5× decode bandwidth gap. The
  naive version already lost (closed PR #23) — the win needs the warp-cooperative
  craft. High effort, bounded, proven by llama.cpp's 419 tok/s.

**Tier 3 — large capability jumps** (`06`, `../Research/05`,`06`,`08`):
- Arch registry → Qwen/Gemma/Phi (near-Llama) then MoE/Mamba/RWKV.
- Paged multi-sequence KV → continuous batching → minimal server.
- GBNF grammar / JSON-schema structured output (needs the Tier-1 sampler layer).

**Tier 4 — defer / proven dead-ends:**
- Portable wgpu coopmat (ruled out), scalar DP4A for k-quants (loses),
  multimodal, distributed/RPC, i-quant/MXFP4 formats. Keep KV resident across
  prefill→decode and flash attention are minor decode wins if revisited.

─────────────────────────────────────────────────────────────────────────────

## The honest framing

From `PERFORMANCE.md`, unchanged by this capture: matching llama.cpp's raw
throughput is a treadmill — a mature, tensor-core-tuned, perpetually-moving
target an ~11k-LOC from-scratch engine won't out-run. The worthwhile goal is to
**close the glaring, bounded-effort gaps while staying legible**, and to add
**user-visible breadth** where it's cheap. The CPU AVX2 work (1.48× behind) and
the merged CUDA backend (3.4–4.4× behind, down from 8–26×) are that goal
executed; the remaining moves are breadth (Tier 1/3) and the single bounded
decode lever (Tier 2).

─────────────────────────────────────────────────────────────────────────────

## Provenance

Synthesized from `PERFORMANCE.md`, `HANDOFF.md`, `BACKLOG.md`, and docs `01`–`08`
of this capture (source @ `e2a7d5d`, 2026-06-18). External comparison:
`../Research/00`–`09`.
