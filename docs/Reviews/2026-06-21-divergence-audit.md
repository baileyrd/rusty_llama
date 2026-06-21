# Divergence Audit — 2026-06-21

A whole-repo review through one lens the prior reviews don't take head-on:
**where does rusty_llama diverge from llama.cpp?** Divergences are exactly what
threaten the "matches llama.cpp" parity claims and explain the residual
performance gap, so they're worth mapping on their own.

**Method.** 10 subsystem finders compared the rusty source against
[`docs/Research/*`](../Research) (which pin llama.cpp's actual behavior at a
commit), then every finding that mattered was adversarially re-verified against
the real code: **139 raw findings → 52 confirmed divergences, 7 refuted, ~30
confirmed *non*-divergences** (matches-where-you'd-fear-otherwise). The
numeric-parity core (`model.rs` forward pass, `config.rs` RoPE, `arch.rs`,
`cpu.rs`, `quant.rs`) was additionally read by hand to check the machine's work;
the two agree, including on what *isn't* a divergence.

This complements the [2026-06-20 deep review](2026-06-20-deep-review.md)
(correctness + perf + over-engineering) rather than repeating it.

---

## Verdict

The divergences sort into four kinds, and the sort *is* the finding:

| Kind | Count | Meaning |
|---|---:|---|
| **A. Intentional, cost performance** | ~8 | These *are* the gap. One structural choice dominates; the rest are documented and mostly correctly ruled out. |
| **B. Intentional, correct & compensated** | ~30 | Clever unifications that look like divergences but reproduce llama.cpp's numerics exactly. Where the codebase shines. |
| **C. Silent correctness divergences** | ~6 | The dangerous ones — all on **breadth archs / edge configs**, none on the TinyLlama benchmark path. |
| **D. Numeric micro-divergences** | ~8 | f32-vs-f64 reductions, rounding modes, NaN handling. Real but sub-perceptual. |

**The performance gap is essentially one divergence** — `f16 cuBLAS` instead of
`int8 MMQ` for quantized prefill. Everything else in A is tuning residue or
already-ruled-out. The *interesting* divergences are all in C, and they're
cheap to close. **C1 and C2 are fixed in this PR; C3 is documented below.**

---

## A. Divergences that cost performance (the gap) — all already tracked

1. **Quantized prefill: dequant→f16 cuBLASLt, not int8-MMQ tensor cores**
   (`cuda.rs:717-758` `gemm_dev_f16`; weights pre-dequantized to a full f16 VRAM
   buffer in `weight_f16`, `cuda.rs:632-657`). Structurally identical to the
   `ggml_cuda_op_mul_mat_cublas` *fallback* that llama.cpp deliberately routes
   around (it sends quantized prefill to MMQ: packed weights, dequant-in-shared,
   `mma.sync.m16n8k32.s8`). **This one divergence is the ~3.6× prefill gap.**
   Documented: `PERFORMANCE.md:46/:52-54/:192-195`, BACKLOG L7. The header-free
   int8 `mma.sync` primitive is already proven on sm_120 (`probe_int8_mma_ptx`).
2. **Decode GEMV is packed-DP4A but not llama.cpp's per-arch-tuned MMVQ** —
   correct kernel; residual ~1.4× is micro-tuning (near structural limit).
3. **CUDA flash attention is a custom block-per-(row,head) online-softmax
   kernel**, not the `fattn` MMA/tile/vec family — no tensor cores. Decode-neutral
   (verified); matters for prefill / long context. (BACKLOG L8.)
4. **CPU GEMM is cache-blocked + register-tiled (MR=4), not a repacking
   micro-kernel** — no AMX, no weight repack, no NR-tiling. The ~28× CPU prefill
   gap (BACKLOG L9).
5. **Eager per-op `Backend` trait vs ggml's compute graph** (`backend/mod.rs`).
   No graph-level fusion/scheduling; recovered by hand (`matmul_shared`,
   CUDA-graph capture, fused decode).
6. **Server: no prompt-cache / KV reuse across requests** (`server.rs:458`
   re-prefills the entire prompt at `pos=0` every request). Biggest
   server-throughput divergence; out-of-scope per `phase-4-server.md`.
7. **Server: per-slot full-length KV reservation + no continuous batching** —
   caps concurrency; llama.cpp packs a unified batch across slots.
8. **KV cache stored f32 on host** (`model.rs`), llama.cpp defaults to f16 (and
   offers KV-quant). 2× memory; rusty is *more* exact, not less.

---

## B. Divergences that aren't — the clever compensations

These look like bugs and aren't. They're load-bearing, so they're worth stating
explicitly (a future refactor that breaks one would be a silent regression):

- **RoPE: one interleaved (NORM) kernel + load-time NEOX→interleaved permute**
  (`model.rs:419-438`, `neox_src`/`permute_neox`). llama.cpp carries *two* rope
  kernels dispatched per-arch; rusty unifies on one and pre-permutes Q/K (+biases)
  for NEOX archs — the inverse of the permute llama.cpp's converter bakes into
  Llama GGUFs. **Verified bit-equivalent.** (But see C1 — this is what creates the
  footgun.)
- **YaRN / llama3 / linear RoPE scaling** (`config.rs`) — corr-dims, ramp, and
  `mscale = attn_factor·(1+0.1·ln s)` all match `rope_yarn` exactly,
  cross-checked against Python.
- **Q4_K/Q6_K/Q8_0/Q4_0 block layouts + `get_scale_min_k4`** — bit-exact ggml.
  **Q8_K per-256 activation quant matches `vec_dot_q4_K_q8_K`**, so the CPU int8
  path *matches* llama.cpp; the wgpu f32 path is the one that's *more* exact than
  llama.cpp (confirms R5.3).
- **MoE** (`moe_ffn_row`): softmax→top-k→Mixtral-renorm / Qwen2-MoE-raw +
  sigmoid-gated shared expert — matches `build_moe_ffn`.
- **Gemma2 sandwich norm, embd `√d` scale, attn+final softcap order, GQA mapping,
  prefill-last-token-only logits** — all verified faithful.
- **GeGLU GELU** (was suspected divergent, *refuted*): ggml's default `gelu` IS
  the tanh-approx f16-LUT — rusty matches (and even replicates the f16 LUT).

---

## C. The divergences that matter — silent correctness on breadth archs

**None touch the TinyLlama Q4_K_M benchmark path.** All are where a from-scratch
engine quietly disagrees with llama.cpp on advertised-but-unbenchmarked models.

### C1. Unrecognized archs silently fall back to Llama → wrong RoPE — **FIXED**
`arch.rs::from_name` mapped any unknown `general.architecture` to `Arch::Llama`
(NORM rope, **no NEOX permute**). Because correctness now *depends* on
`rope_neox()` classifying every arch, any NEOX-rope model rusty doesn't recognize
(`gemma`/`gemma3`, `starcoder2`, `cohere`, `stablelm`, `phi2`, …) loaded
**silently** and rotated Q/K with the wrong convention — garbage output, no error.
llama.cpp dispatches rope-type per recognized arch and never silently mis-rotates.
The unification in (B) is safe *only* while the arch table is complete.

→ **Fix (this PR):** added `Arch::is_known`; `Model::from_gguf` now warns when the
arch is unrecognized (still loads as Llama — the permissive policy is deliberate —
but no longer *silently*). `llama`/`mistral` are whitelisted as NORM-compatible.
Tested by `is_known_flags_only_implemented_archs`.

### C2. SPM ignored GGUF `add_space_prefix` → broke byte-exact tokenization — **FIXED**
`Spm::from_gguf` never read `tokenizer.ggml.add_space_prefix`; `encode` always
prepended the dummy U+2581 space. Some SPM GGUFs set it **false** → llama.cpp adds
no prefix, rusty did → **different token ids**, silently breaking byte-exact
parity for those vocabs.

→ **Fix (this PR):** added an `add_space_prefix` field (default true), read from
the GGUF key, gating the dummy space in `encode`. Tested by
`spm_add_space_prefix_toggles_dummy_space`.

### C3. Gemma2 interleaved sliding-window attention (iSWA) not implemented — **documented**
`cpu.rs:401` always attends `0..=pos`; there is no per-layer `n_swa` window.
Gemma2's even layers attend only to the last 4096 keys. rusty is **numerically
identical below 4096 tokens** (why 2B greedy validation passed) and **divergent
beyond it**. Independently flagged by two finders.

→ **Status:** documented here as a known long-context divergence (the full fix is
a per-layer window mask + `n_swa` config field — out of scope for this small PR).
Short-context Gemma2 is unaffected.

### C4–C6 (medium/low, breadth)
- **Missing quant types** — `Q5_K, Q3_K, Q2_K, Q5_0, Q5_1, Q4_1`, all IQ/MXFP4/TQ
  → **hard error at load** (clean, not silent — good). Pure breadth gap.
- **GGUF: single-file only**, no sharded `-00001-of-000NN` → large models won't
  load.
- **Chat: ~5 hardcoded templates, no Jinja** → silently wrong formatting for any
  model whose template isn't hardcoded. (The "Gemma drops system role" suspicion
  was *refuted* — rusty matches llama.cpp there.)
- **`generate` stops on token id `1`** (`model.rs:1445`) — llama2.c legacy
  treating BOS as EOS; llama.cpp uses only the model's EOG set. Harmless on normal
  models, latent if id 1 is ever a real emitted token.

---

## D. Numeric micro-divergences (low, sub-perceptual)

- **RMSNorm sum-of-squares accumulates in f32** (`cpu.rs`); ggml uses
  `ggml_float` = **f64** for that reduction. Tiny per-norm drift.
- **f32 matmul/attention accumulate sequentially** (`.sum()`) vs ggml's
  multi-accumulator SIMD order — rounding-level; only the integer-dot paths are
  bit-exact to llama.cpp.
- **No FTZ / max-offset softmax stabilizers** (ggml has `SOFTMAX_FTZ_THRESHOLD` +
  KQ max-offset clamps); safe in f32, would matter in f16.
- Sampler: **argmax tie-break is LAST index** (llama.cpp greedy is FIRST); TopP
  `min_keep`, penalty special-token handling, `temp<=0` guard differ slightly. RNG
  differs entirely (llama2.c xorshift vs mt19937), so draws never matched anyway —
  by design.

---

## What the audit refuted (7 knocked-down)

All turned out to be *matches* on verification: GeGLU (ggml uses tanh-approx),
Gemma `1+w` norm (converters bake +1 in — confirmed), grammar EOG gating (a
`cargo test` confirmed correct), MoE shared-expert detection, Gemma template
system-message fold, sampler seed-from-wall-clock (OpenAI-faithful, not a
llama.cpp divergence). The verifiers did real work to refute these (ran tests,
read raw bytes), which is the reason to trust the 52 that survived.

---

## Prioritized recommendations

The performance divergences are well-understood and mostly exhausted; the
high-ROI work is the silent-correctness bucket, which is nearly free.

1. **C1 — warn on unrecognized arch instead of silent Llama fallback.** ✅ done.
2. **C2 — honor `add_space_prefix`.** ✅ done.
3. **C3 — document Gemma2 iSWA as wrong >4096**, add `n_swa` mask when convenient.
   📝 documented.
4. **A1 — int8 Q4_K MMQ kernel (BACKLOG L7).** The *only* remaining real speed
   lever; primitive already banked. Large effort, known ceiling.
5. Everything in D: leave it. f64 rmsnorm accumulation is the only one worth
   reconsidering, and only if quality drift ever appears.

**Strategic read.** rusty_llama's divergences from llama.cpp are overwhelmingly
*deliberate and documented*, and the few silent ones are all confined to the
breadth archs it doesn't benchmark — exactly the failure profile you'd want. The
honest race on the benchmark path is genuinely honest.
