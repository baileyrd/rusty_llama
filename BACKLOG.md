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
| `rope(q, k, pos, head_size, kv_dim, theta)`                      | in place |
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

## 2. `gpt2` byte-level BPE tokenizer (+ RoPE long-context scaling)

**Why deferred:** large, and hard to validate without a reference tokenizer /
real model to download (both blocked in the cloud sandbox). *Does not* require a
GPU — could be done anywhere with network access to a reference.

Currently [`Tokenizer::from_gguf`](src/tokenizer.rs) only handles `llama`/SPM
tokenizers (it rejects `tokenizer.ggml.model == "gpt2"`). Full Llama-3 / Qwen2
support needs byte-level BPE:
- GPT-2 byte↔unicode mapping; vocab tokens stored in that encoded form.
- `tokenizer.ggml.merges` (ordered; merge by lowest rank, vs our current
  score-based SPM merges).
- Pre-tokenization regex split (Llama-3 / Qwen2 use specific patterns).
- RoPE base θ is already read from GGUF (done); long-context **scaling**
  (`rope_scaling` type `linear`/`yarn`/`llama3`) is still TODO in
  `Model::from_gguf` (partial rotary is currently rejected outright).

Validate against a known reference (HF `tokenizers` output) on a machine with
network access.

---

## 3. Explicit SIMD intrinsics for the integer dot products

**Why deferred:** the integer kernels (`src/quant.rs`: `vec_dot_q8_0`,
`vec_dot_q4_0`, `vec_dot_q4_k`, `vec_dot_q6_k`) already autovectorize well under
`target-cpu=native` (~3× over per-block f32 dequant), so the incremental win
from hand-written AVX-512 VNNI (`_mm512_dpbusd_epi32`) / NEON is uncertain and
the code is `unsafe` + platform-specific.

If pursued: add a runtime-feature-gated (`is_x86_feature_detected!`) intrinsic
path with the existing scalar loop as fallback, and a test asserting the SIMD
result is **bit-identical** to the scalar one. Benchmark against the current
`bench_q8_0_matmul` / `bench_q4_k_matmul` to confirm it's worth the complexity.
