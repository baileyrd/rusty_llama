# Phase 5.2 — Flash (online-softmax) attention (detailed plan)

**Status: SHIPPED.** Adopted as the attention algorithm on the CPU oracle and the
CUDA resident decode path.

Replace the classic *materialize-scores → softmax → value-weighted-sum* attention
with **online (running-max) softmax** that streams the keys in a single pass,
keeping a running max `m`, running exp-sum `l`, and a running output accumulator —
so **no `seq_len`-sized score buffer** is needed anywhere.

## Why (honest framing)

Attention is **not** the decode bottleneck — the quantized GEMVs are — so flash is
**decode-throughput-neutral** here (measured below). Its value is **capability +
memory**:

- **CUDA**: the old kernel held the whole score row in shared memory
  (`[ sq:head_size | sc:seq_len | red:block ]`), which **caps context by shared
  memory** (~12 K tokens at the 48 KB default). Flash uses a constant
  `[ sq:head_size | sp:block | acc:head_size | red:block ]` — context is no longer
  shared-memory-bound (validated to 16 384 tokens).
- **CPU**: drops the per-head `n_heads × seq_len` score scratch (`att`) entirely —
  it matters for long context and many concurrent server slots.

This is exactly the kill-criterion call the repo uses: a speed change is adopted
only on a win; a non-speed change (here, a capability + memory win at neutral
throughput, fully parity-checked) is adopted on its own merits.

─────────────────────────────────────────────────────────────────────────────

## CPU (`CpuBackend::attention`, the oracle)

Per head (still rayon-parallel over heads), a single streaming pass over keys
`0..=pos`: for each key, score `q·k·scale` (+ optional Gemma2 tanh softcap),
`m_new = max(m, s)`, rescale the running state by `α = exp(m − m_new)`,
`p = exp(s − m_new)`, `l = l·α + p`, and `acc = acc·α + p·v`; finalize `acc / l`.
The trait's `att` scratch and `seq_len` arg are now unused (`let _ = (att, seq_len)`)
— the signature is unchanged so every caller and backend is untouched. The
duplicate `model::softmax_inplace` was folded into `math::softmax`.

## CUDA (`attention_kernel` / `dev_attention`, resident decode)

One thread **block** per `(row, head)`, cooperative, streaming keys in **tiles of
`blockDim.x`** with an online softmax. Per tile: each thread scores one key,
block-reduce the tile max, online-combine `m`/`l`, write the tile probabilities to
a `block`-sized shared row `sp`, then each thread (owning a stride of head dims)
accumulates `Σ_tile p·v` into the shared `acc`. Global K/V traffic is identical to
the old kernel (each read once); the only adds are per-tile reductions. The
resident path is Llama-only, so no logit softcap in the kernel (Gemma2's softcap
runs the per-op path, which delegates attention to the CPU). Shared memory dropped
from `(head_size + seq_len + block)·4` to `(2·head_size + 2·block)·4`.

The wgpu batched path (`WGSL_ATTENTION_BATCH`) already used an online softmax;
this brings CPU and CUDA in line.

## Validation

- **CPU**: `attention_flash_matches_classic_reference` — flash vs an independent
  materialized-softmax reference across GQA, several positions, and softcap on/off
  (max diff < 1e-6).
- **CUDA** (`--features cuda -- --ignored`): `dev_attention_parity` (vs the classic
  `CpuBackend::attention_batch`) and **`dev_attention_flash_long_context_parity`**
  — `seq_len = 16 384`, `pos_base = 16 000` (~125 tiles), which both exercises the
  multi-tile combine and exceeds the old shared-memory budget. Max diff ≈ 1.4e-8.
- All existing parity holds: `prefill_matches_sequential` (+ MoE / Qwen2-MoE
  variants) and `batched_decode_matches_independent_sequences` stay bit-identical
  (both paths use the same flash `attention`).

## Measurement (TinyLlama-1.1B Q4_K_M; Core Ultra 9 285H + RTX 5070 Ti, CUDA 13)

`bench_decode_real_tinyllama` (tg128, r=5):

| Path | Before (classic) | After (flash) |
|---|---:|---:|
| CUDA decode | 136 ± 7 tok/s | 134–135 ± 1–4 tok/s |
| CPU decode | 32 tok/s | 33 tok/s |

Decode is **neutral within noise** — as expected, since attention is a small slice
of decode time. Adopted for the long-context + memory win, not speed.

## Limitations / follow-ons

- The win is capability/memory, not decode throughput. A genuine decode speedup
  would come from the GEMVs (Phase 2 territory), not attention.
- The CUDA flash kernel is the resident-decode kernel; the per-op CUDA/GPU
  attention still delegates to the CPU oracle (unchanged).
- A prefill-tiled flash (query tiles reusing K/V tiles across rows) could cut
  prefill K/V traffic, but prefill is GEMM-quality-bound, so it is not pursued.

─────────────────────────────────────────────────────────────────────────────
