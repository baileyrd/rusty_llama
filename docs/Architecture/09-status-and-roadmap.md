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
| 2 | **GPU int8 / DP4A decode** | ✅ (CUDA) / 🟡 (wgpu) — **CUDA packed-weight DP4A GEMV ADOPTED** (Phase 2): warp-cooperative, coalesced Q4_K/Q6_K `__dp4a` decode GEMV (`quantize_q8k` + `gemv_q4_k`/`gemv_q6_k`), default-on, **2.4× decode** (86→207 tok/s on real TinyLlama; plan [`plans/phase-2-decode-gemv.md`](plans/phase-2-decode-gemv.md)). The wgpu Stage-2 stays parked (int8 vs an already-packed dequant GEMV saves no bytes); the CUDA win is real because its baseline was f16. (`02`/`03`/`04`) |
| 3 | **GPU tensor cores** | ✅ (CUDA) / ❌ (portable) — portable wgpu coopmat **ruled out** on this hardware (advertises only f16 16×16; wgpu 29 wires only 8×8 f32 → silent all-zero garbage). CUDA backend M0→M2b + resident decode **DONE and merged** (PRs #20–#22). Closed PR #23: a naive packed dequant-GEMV was ~1.6× *slower* than f16. (`03`/`04`) |
| 4 | Flash attention; cache-blocked CPU prefill | ❌ not started |
| 5 | **Breadth** (archs, samplers, KV-quant, server) | 🟡 PARTIAL — **Phase 3.1 archs** (Qwen2/Phi-3/Gemma 2) **+ 4.1 server + 4.2 continuous batching + 5.1 GBNF grammar SHIPPED**: OpenAI HTTP server (`server` feature) with streaming SSE, a continuous-batching scheduler (`Batch::decode_step`), and grammar-constrained / JSON output ([`plans/phase-4-server.md`](plans/phase-4-server.md), [`plans/phase-5-grammar.md`](plans/phase-5-grammar.md)). KV-quant ❌; 4.3 paged KV, 5.2 flash attention, 3.2 MoE, 3.3 recurrent remain. |

─────────────────────────────────────────────────────────────────────────────

## Performance: current numbers

TinyLlama-1.1B Q4_K_M; Core Ultra 9 285H (AVX2, no AVX-512) + RTX 5070 Ti
(Blackwell sm_120, 12 GB), CUDA 13.3; vs llama.cpp `b9672`. From `PERFORMANCE.md`.

| Path | Metric | rusty_llama | llama.cpp | Gap |
|---|---|---:|---:|---|
| CPU | decode | 49.8 tok/s | 73.6 | **1.48×** behind |
| wgpu | decode | 45.9 tok/s | 375.9 (Vulkan) | 8.2× behind |
| CUDA | prefill (pp512) | ~4,450 tok/s | 19,637 | **~4.4×** behind |
| CUDA | decode (tg128) | **~207 tok/s** | ~415 | **~2.0×** behind |

Two distinct walls (see `04` and `../Research/03-cuda-kernels.md`):
- **Decode ~2× (was ~3.4×) — bandwidth wall closed (Phase 2).** The packed
  Q4_K/Q6_K `__dp4a` GEMV streams ~0.56 B/weight vs f16's ~2 B — CUDA decode
  86 → 207 tok/s same-session (2.4×). The residual ~2× is kernel tuning + flash attn.
- **Prefill ~4.4× = GEMM kernel quality.** cuBLASLt `Matmul<f16>` + our nvrtc
  kernels vs llama.cpp's fused int8 tensor-core MMQ. Now compute-bound (per-op
  overhead already cut), so further gains require beating cuBLASLt.

─────────────────────────────────────────────────────────────────────────────

## Capability snapshot

| Area | State | Doc |
|---|---|---|
| Architectures | Llama, Qwen2, Phi-3, Gemma 2 (`Arch` registry seam, `src/arch.rs`; Phase 3.1) | `01`,`06`,`plans/phase-3-archs.md` |
| Quant (consume) | F32/F16/Q4_0/Q8_0/Q4_K/Q6_K | `05` |
| Quant (produce) | only `quantize_q8_0` (test fixtures); no quantize-to-disk | `05` |
| Tokenizers | SPM + GPT-2 BPE (GPT-2/Llama-3/Qwen2 pretokenizers) | `07` |
| Samplers | greedy / temperature / multinomial / top-p / top-k / min-p / penalties; **GBNF grammar-constrained decoding** (Phase 5.1) | `07`,`plans/phase-5-grammar.md` |
| KV-cache | single-sequence flat per-layer f32 (resident on GPU/CUDA) + multi-sequence batched decode (server scheduler, per-slot KV) | `01`,`03`,`04`,`plans/phase-4-server.md` |
| Loading | single-file GGUF (v2/3) + llama2.c; mmap zero-copy; no sharding/validation/async | `06` |
| Backends | CPU (oracle) + wgpu + CUDA | `02`,`03`,`04` |
| Serving | OpenAI HTTP server + continuous batching (`server` feature; Phase 4.1/4.2), with `grammar` / `response_format` JSON constraints (Phase 5.1); LoRA + embeddings shipped (Phase 1); multimodal: none | `plans/phase-4-server.md`,`plans/phase-5-grammar.md` |

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

**Tier 2 — the one decode speed lever — ✅ DONE (Phase 2)** (`04`, `../Research/03`):
- **Warp-cooperative, coalesced packed-weight DP4A GEMV (MMVQ-style)**:
  `quantize_q8k` activation quant once/step + `gemv_q4_k`/`gemv_q6_k`
  unpack-in-register + `__dp4a`, weights stay packed. **Shipped + adopted** —
  CUDA decode 86 → 207 tok/s (2.4×), closing the bandwidth wall the naive PR #23
  couldn't, plus 2.2 (device-resident prefill→decode KV handoff). Plan
  [`plans/phase-2-decode-gemv.md`](plans/phase-2-decode-gemv.md).

**Tier 3 — large capability jumps** (`06`, `../Research/05`,`06`,`08`):
- Arch registry → Qwen2/Phi-3/Gemma2 (near-Llama) ✅ **done — Phase 3.1** (`plans/phase-3-archs.md`); next MoE/Mamba/RWKV.
- Paged multi-sequence KV → continuous batching → minimal server: **4.1 server + 4.2 continuous batching done** (`plans/phase-4-server.md`); 4.3 paged KV remains (memory efficiency).
- GBNF grammar / JSON-schema structured output ✅ **done — Phase 5.1** (`plans/phase-5-grammar.md`): GBNF parser + byte-NFA matcher + a `GrammarStage` on the sampler chain; CLI `--grammar` and server `grammar` / `response_format` (`json_object`).

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
