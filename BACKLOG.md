# Backlog

Tasks deferred from the cloud sessions, with enough context to start cold.
Each item notes *why* it's deferred. Pick one, branch, implement, test, PR.

---

## 1. GPU backend behind the `Backend` trait  — **requires a GPU**

**Why deferred:** the cloud execution environment these milestones were built in
has no GPU, so this can't be implemented or verified there. Pick this up on a
**local Claude Code instance with GPU access** (CUDA, Metal, or Vulkan/wgpu).

### What exists today
All heavy math goes through the object-safe `Backend` trait in
[`src/backend/mod.rs`](src/backend/mod.rs); the only implementation is
[`CpuBackend`](src/backend/cpu.rs) (rayon + autovectorized/integer kernels).
The transformer in [`src/model.rs`](src/model.rs) (`forward`) is written purely
in terms of the trait, so a new backend slots in without touching model code.

Trait surface to implement:

| method      | shape contract |
|-------------|----------------|
| `rmsnorm(out, x, weight, eps)`                                   | all length `dim` |
| `matmul(out, x, w: &QMatrix)`                                    | `out=[w.rows()]`, `x=[w.cols()]` |
| `rope(q, k, pos, head_size, kv_dim, inv_freq, mscale)`           | in place; `inv_freq` is the precomputed per-pair table |
| `attention(out, q, key_cache, value_cache, att, pos, n_heads, n_kv_heads, head_size, seq_len, kv_dim)` | GQA, one position |
| `swiglu(hb, hb2)`                                                | `hb = silu(hb) * hb2` |
| `add(out, x)`                                                    | residual |

`QMatrix` ([`src/tensor.rs`](src/tensor.rs)) is either `F32 { data, rows, cols }`
or `Quant { ty, data: &[u8], rows, cols }`. The CPU backend dequantizes a block
at a time (`src/quant.rs`: `dequant_block`, `vec_dot_*`). Quant types: F32, F16,
Q4_0, Q8_0, Q4_K, Q6_K.

### Suggested approach
- **Library:** `wgpu` is the portable choice (Metal/Vulkan/DX12 + WebGPU) and
  keeps the "from scratch, minimal deps" spirit; CUDA via `cudarc` is an option
  if targeting NVIDIA only. Put it behind a cargo feature (e.g. `gpu`) so the
  default build stays dependency-light.
- **Start simple:** implement f32 `matmul` first (upload `QMatrix::F32` weights
  once, keep them resident), then `rmsnorm`/`rope`/`swiglu`/`add`, then
  `attention`, then dequantize quantized `QMatrix` on the device (or convert to
  f16 on upload).
- **Performance note:** single-token decode issues many *tiny* (dim-sized)
  ops. Naive per-op host↔device dispatch will be latency-bound and likely
  *slower* than `CpuBackend`. To actually win, keep weights + KV cache resident
  on the device and minimize round-trips (ideally fuse a layer, or batch the
  prompt prefill). Measure against `CpuBackend` honestly.

### Acceptance criteria
- New `impl Backend` selectable at runtime (CLI flag, e.g. `--backend gpu`).
- Per-op parity tests vs `CpuBackend` within f32 tolerance (mirror the existing
  backend unit tests; reuse the synthetic model in `src/dummy.rs`).
- An end-to-end run of a real GGUF model produces coherent text matching the
  CPU backend's greedy output.
- A benchmark vs `CpuBackend` (extend the `bench_*` pattern in
  `src/backend/cpu.rs`), reported in the PR.

---

## 2. Finish modern-model tokenization / RoPE  (partially done)

**Done:** byte-level BPE (`gpt2`) tokenizer is implemented in
[`src/tokenizer.rs`](src/tokenizer.rs) (`Bpe`): GPT-2 byte↔unicode mapping,
rank-based merges from `tokenizer.ggml.merges`, and `Tokenizer::from_gguf`
dispatches `llama`→SPM / `gpt2`→BPE. So Llama-3 / Qwen2 vocabularies load.

**Still TODO** (none of these needs a GPU — do it on a machine with network
access to a reference tokenizer / small real model):
- **Byte-exact pre-tokenizer.** `pretokenize_gpt2` is a hand-rolled
  approximation of GPT-2's regex (contractions, letter/digit/punct runs,
  whitespace). It does not byte-match Llama-3's / Qwen2's exact regex (which use
  lookahead). Validate against HF `tokenizers` output and either fix the
  hand-rolled splitter or pull in `fancy-regex` and use the model's
  `tokenizer.ggml.pre` pattern.
- **RoPE long-context scaling.** θ (`rope.freq_base`) is read; the scaling types
  (`linear`/`yarn`/`llama3`) in `rope_scaling` are not. `Model::from_gguf`
  currently rejects partial rotary outright — extend it.
- **Special-token handling.** Added/control tokens (`<|...|>`) are decoded as
  literal text and not recognized during encode. Add a special-token table.

---

## 3. Explicit SIMD intrinsics for the integer dot products  (mostly done)

**Done:** `src/quant.rs` now has a hand-written **AVX-512 VNNI** (`vpdpbusd` /
`_mm256_dpbusd_epi32`) path for `vec_dot_q8_0`, `vec_dot_q4_0` and `vec_dot_q4_k`
(the `x86` submodule). Each public `vec_dot_*` is a dispatcher that selects the
SIMD kernel at run time via `is_x86_feature_detected!` (cached), falling back to
the autovectorized scalar loop — which is also kept as the bit-identical oracle.

The reduction stays entirely in integer arithmetic, so the SIMD result is
**bit-for-bit identical** to the scalar one. `vpdpbusd` needs an unsigned weight
operand: Q4_K nibbles are already unsigned `[0,15]` so they map straight on;
Q8_0/Q4_0 weights are signed, so they're biased `+128` and the bias removed
exactly using the activation's per-block sums (`Q8Activation::block_sums`).

- **Parity:** `q8_0/q4_0/q4_k_simd_matches_scalar` fuzz the kernels and assert
  `f32::to_bits` equality. An end-to-end greedy run is byte-identical with the
  SIMD path on vs off (`RUSTY_LLAMA_NO_VNNI=1`).
- **Benchmark:** `bench_{q8_0,q4_0,q4_k}_simd_vs_scalar` — measured **2.2–2.5×**
  over the autovectorized path (2048×2048), and ~**2×** end to end on a
  TinyLlama-width Q8_0 model. So yes, it's worth it.

**Still TODO:**
- **Q6_K.** `vec_dot_q6_k` is still scalar. Its 6-bit weights are reconstructed
  from split `ql`/`qh` planes (the bulk of its work) and its scales are per-16,
  not per-32, so it doesn't fit the 32-wide `vpdpbusd` helper as cleanly; the
  same bias trick (`a+32`, corrected by the per-16 `bsums`) would apply over
  128-bit lanes.
- **NEON.** No aarch64 path yet; the dispatchers fall through to scalar there.
