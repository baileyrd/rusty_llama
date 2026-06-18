# 08. Testing, Benchmarking & Parity

## Summary

rusty_llama's correctness story rests on one idea: **the CPU backend is the
parity oracle.** Every kernel — scalar, AVX2/VNNI, wgpu, CUDA — is validated by
computing the same op and comparing against `CpuBackend` (or its bit-identical
scalar integer-dot loop) within a documented tolerance. Three concentric rings
of tests enforce this: (1) **unit tests** pin individual ops against
hand-computed values and the scalar oracle; (2) **integration / GGUF / prefill
parity** tests run the full forward pass and generation loop on synthetic models;
(3) **device-gated GPU/CUDA parity + in-crate benchmarks** compare backends and
report `tok/s` against `llama-bench`. The honest headline (PERFORMANCE.md,
2026-06-16): real TinyLlama-1.1B Q4_K_M prefill **~4.4× behind** llama.cpp CUDA,
decode **~3.4× behind**.

llama.cpp counterpart: `docs/Research/08-capabilities-and-tooling.md`
(`llama-bench` methodology, `pp512`/`tg128` definitions).

─────────────────────────────────────────────────────────────────────────────

## 1. Philosophy: the CPU backend is the oracle

The architecture funnels all heavy math through the `Backend` trait (see
`docs/Architecture/02-backend-trait-and-cpu.md`). That single seam is what makes
parity testing tractable: a GPU or CUDA op is "correct" iff it reproduces the
`CpuBackend` result for the same inputs. The CPU backend itself is anchored two
ways:

- **Unit tests vs hand-computed values.** `src/backend/cpu.rs` checks each
  primitive against a value worked out by hand — e.g. `matmul_f32_basic`
  (`w·[1,1] = [3,7]`, `src/backend/cpu.rs:231`), `rmsnorm_unit_input`
  (`src/backend/cpu.rs:391`), `softmax_uniform` (`:415`), `swiglu_known_value`
  (`:423`), and a battery of RoPE tests pinning pos-0 identity, angle, θ-base,
  linear scaling, YaRN mscale, and partial-rotary tail (`:439`–`:518`).
- **The scalar integer-dot loop is itself the oracle for SIMD.** In
  `src/quant.rs`, every `vec_dot_*` keeps an autovectorized scalar reduction that
  the AVX2/VNNI kernels must reproduce **bit-for-bit** (`f32::to_bits` equality),
  because the reduction stays entirely in integer arithmetic
  (`src/quant.rs:594`–596). `vec_dot_q8_0_equals_dequantized_dot`
  (`src/quant.rs:1073`) cross-checks the integer dot against the
  fully-dequantized f32 dot — closing the loop back to plain f32.

### Exact parity vs. coherence

Two distinct bars, chosen by how lossy the path is:

| Path | Bar | Why |
| --- | --- | --- |
| CPU SIMD (AVX2 / VNNI) integer dot | **bit-exact** (`to_bits` eq) | reduction is pure integer; no rounding introduced |
| Batched prefill vs sequential (CPU) | **bit-exact** (`assert_eq!`) | batched ops loop the single-token ops in the same order (`tests/prefill.rs:1`) |
| wgpu F32 / Q8_0 dequant op vs CPU | **tight tol** `1e-3 + 1e-3·\|w\|` | reassociated / FMA-contracted f32 (`src/backend/gpu.rs:2220`) |
| CUDA TF32 GEMM vs CPU | **TF32 tol** `0.02 + ic·2e-4 + 0.03·\|w\|` | TF32 keeps ~10 mantissa bits (≈2⁻¹¹ rel), error grows with reduction length `ic` (`src/backend/cuda.rs:945`) |
| CUDA nvrtc elementwise kernels vs CPU | `atol + rtol·\|w\|` (`close_approx`) | per-op f32 kernels, maxdiff ≤ 3e-7 (`src/backend/cuda.rs:1372`) |
| int8 (DP4A) fused decode vs CPU | **loose coherence** `< 0.05` | int8 *activation* quant is lossy by design; per-op int8 kernels are pinned bit-exact separately (`tests/gpu_parity.rs:216`) |

The coherence bound is deliberately loose enough to absorb activation-quant error
(observed ~2e-4 on the tiny test model, ~0.07 on real TinyLlama) yet tight enough
that a wiring regression — bad scale, missing barrier — diverges by O(1) and
fails loudly (`tests/gpu_parity.rs:216`–220).

─────────────────────────────────────────────────────────────────────────────

## 2. Test-file inventory (`tests/`)

| File | Scope | Key assertions |
| --- | --- | --- |
| `tests/gpu_parity.rs` | GPU↔CPU end-to-end on a synthetic GGUF (whole file `#![cfg(feature = "gpu")]`) | 5 tests; logits track CPU within `1e-2` and pick the **same greedy token**, F32 + Q8_0 |
| `tests/prefill.rs` | batched prefill ≡ sequential, on CPU | last-position logits **bit-identical** to one-token-at-a-time `forward`; KV-cache parity via one extra decode step (F32 + Q8_0) |
| `tests/gguf.rs` | GGUF parse → load → forward → generate | 9 tests: metadata/tensor table, tokenizer (SPM + GPT-2 BPE), RoPE θ/eps + scaling, special-token round-trip, weights == dequantized tensors, deterministic finite forward, Q8_0 ≈ F32 |
| `tests/integration.rs` | real loader (mmap) + forward + generation on a llama2.c checkpoint | 4 tests: header round-trip, finite/deterministic forward, reproducible greedy gen, truncated checkpoint errors cleanly |
| `tests/pretokenize.rs` | byte-exact GPT-2-family pre-tokenizer | chunks match golden table (≥90 cases) and **reassemble** to input byte-for-byte |
| `tests/fixtures/pretokenize_golden.rs` | `@generated` golden vectors | `PRETOKENIZE_GOLDEN: &[(&str, &[(&str, &[&str])])]` for `gpt-2`, `llama-bpe`, `qwen2` pre-names |

### gpu_parity.rs detail

A shared `cfg()` (dim 32, hidden 64, 2 layers, GQA 4 q / 2 kv, vocab 64, seq 16 —
all multiples of 32 so the same config serializes as Q8_0,
`tests/gpu_parity.rs:16`) and a `gpu()` helper that returns `None` (printing a
skip note) when no adapter is present (`tests/gpu_parity.rs:31`). The five tests:

- `forward_logits_match_cpu` (`:56`) — single-step forward, `max_diff < 1e-2`,
  same argmax, F32 + Q8_0.
- `prefill_matches_cpu` (`:87`) — whole batched prefill path through all layers.
- `decode_step_matches_cpu` (`:118`) — fused on-device decode after prefill
  (exercises the one-time host→device KV-cache sync); `set_int8_decode(false)`
  forces the exact dequant path.
- `greedy_stream_matches_cpu` (`:156`) — 12-token greedy stream must be
  **byte-identical** (`assert_eq!(run(&CpuBackend::new()), run(&g))`).
- `decode_step_int8_coherent_with_cpu` (`:188`) — `set_int8_decode(true)` on the
  Q8_0-eligible model; loose `< 0.05` coherence, same argmax.

### prefill.rs detail

`prefill_matches_sequential` (`tests/prefill.rs:25`) asserts batched
`forward_prefill` last-position logits **exactly equal** the result of looping
`forward` one token at a time (`assert_eq!`, not approximate — the CPU batched ops
loop the single-token ops in identical order). It then runs **one more decode
step** on both states and asserts equality, which can only hold if both KV caches
hold identical K/V for every prompt position (`tests/prefill.rs:57`–63). Both F32
and Q8_0.

─────────────────────────────────────────────────────────────────────────────

## 3. In-crate unit & parity tests (`src/`)

Backends keep their parity tests next to the kernels they validate. Counts of
`#[test]` functions: `src/backend/gpu.rs` 34, `src/quant.rs` 22,
`src/backend/cpu.rs` 17, `src/backend/cuda.rs` 16, `src/tokenizer.rs` 14,
`src/config.rs` 5, `src/sampler.rs` 3.

### CPU (`src/backend/cpu.rs`)
Hand-value unit tests (above) plus `matmul_quant_matches_f32`
(`src/backend/cpu.rs:242`) and `matmul_routes_kquants_to_int8`
(`:304`, confirms Q4_K/Q6_K matmuls route through the int8 activation path and
equal `vec_dot_q4_k`/`vec_dot_q6_k`). Two ignored micro-benches:
`bench_q8_0_matmul` (`:265`) and `bench_q4_k_matmul` (`:350`) time the per-block
dequant dot vs the int8 fast path.

### Quant (`src/quant.rs`)
The integer-dot parity heart of the crate:

- `vec_dot_{q8_0,q4_0,q4_k,q6_k}_equals_dequantized_dot` (`:1073`–1128) — integer
  dot == dequantized f32 dot.
- `q8_0/q4_0/q4_k/q6_k_avx2_matches_scalar` (`:1170`–1240) — AVX2 kernels
  bit-exact vs scalar; gated on `x86::avx2_supported()` (toggle off with
  `RUSTY_LLAMA_NO_AVX2`, `src/quant.rs:826`). 4-/6-bit weights don't saturate the
  i16 step so they're bit-identical; signed 8-bit formats widen to i16 to stay
  exact.
- `*_simd_matches_scalar` (`:1263`–1340) — AVX-512 VNNI kernels, gated on
  `vnni_supported()` (early-return where absent).
- `q6_k_vnni_algorithm_matches_scalar` (`:1414`) — re-checks the VNNI kernel's
  index/bias **algorithm** in scalar on hardware lacking AVX-512 (this dev
  machine is AVX2-only), so the logic is verified without VNNI silicon.
- Ignored benches `bench_{q8_0,q4_0,q4_k,q6_k}_simd_vs_scalar` (`:1435`+).

### wgpu GPU (`src/backend/gpu.rs`)
`close(got, want)` tolerance `1e-3 + 1e-3·|w|` (`:2220`). `gpu()` helper skips
when no adapter. Per-op parity vs CPU oracle: `matmul_f32_basic/parity`,
`matmul_quant_q8_0_matches_exact`, `matmul_q4_k/q6_k_matches_exact`,
`matmul_q4_k_batch_matches_exact`, `q8_0_relayout_roundtrips`,
`weight_buffer_i8_caches_and_sizes`, `matmul_q8_0_i8_matches_int_dot`,
`matmul_q8_0_i8_acc_folds_residual`, `quantize_q8_kernel_is_valid`,
`quantize_q8k_kernel_is_valid`, `matmul_q4_k_i8_matches_int_dot`,
`matmul_q6_k_i8_matches_int_dot`, `rmsnorm_parity`, `add_parity`,
`swiglu_parity`, `rope_parity`, and `int8_decode_gate_and_toggle` (`:2505`,
verifies the all-Q8_0 eligibility gate + `set_int8_decode`). The int8 kernels are
pinned against the **exact integer dot** here — that's why the e2e int8 decode
test is allowed to be merely coherent.

### CUDA (`src/backend/cuda.rs`)
`close_tc(got, want, ic)` (`:945`, TF32 tolerance) and `close_approx`
(`:1372`). A `noise()` xorshift PRNG seeds reproducible inputs (`:927`). **All 16
tests are `#[ignore]`** (need a CUDA device) and additionally early-return via
`cuda()` when no device is present:

| Test | Checks |
| --- | --- |
| `cuda_smoke` (`:966`) | init + device name + htod/dtoh round-trip (driver + cuBLASLt resolve at run time) |
| `matmul_batch_f32_parity` (`:981`) | non-square batched GEMM vs CPU (catches transpose / m·n·k swap) |
| `matmul_single_f32_parity` (`:995`) | classifier `rows=1` path |
| `matmul_batch_q8_0_parity` (`:1011`) | dequant→f16→GEMM vs exact Q8_0 dot |
| `prefill_coherent_with_cpu` (`:1031`) | fused prefill, top token agrees, rel L2 `< 0.1` |
| `prefill_then_decode_coherent` (`:1089`) | device→host KV handoff correct |
| `decode_multistep_coherent` (`:1144`) | resident `forward_step` KV append across steps |
| `dev_{cvt_roundtrip,add,swiglu,rmsnorm,rope,attention}_parity` (`:1394`–1479) | each nvrtc kernel vs CPU op |

─────────────────────────────────────────────────────────────────────────────

## 4. Device-gated tests & how they run

- **GPU parity** tests are *not* `#[ignore]`; they compile only under
  `--features gpu` and **skip cleanly** when no adapter is found (the `gpu()`
  helper returns `None`). The whole `tests/gpu_parity.rs` file is
  `#![cfg(feature = "gpu")]`.
- **CUDA** tests are *all* `#[ignore = "requires a CUDA device; run with
  --features cuda -- --ignored --nocapture"]` and also guard with `cuda()`. They
  do not run in the default `cargo test`.
- **GPU benchmarks + the coopmat probe** are `#[ignore]` timing tests.

Run matrix:

| Command | Runs |
| --- | --- |
| `cargo test` | CPU unit + integration + GGUF + prefill + pretokenize (GPU file compiled out) |
| `cargo test --features gpu` | + wgpu per-op parity + `gpu_parity.rs` (skip if no adapter) |
| `cargo test --features cuda -- --ignored --nocapture` | CUDA device parity + smoke |
| `cargo test --release -- --ignored --nocapture` | CPU/quant SIMD-vs-scalar benches |
| `cargo test --release --features gpu -- --ignored --nocapture` | wgpu benches + `probe_cooperative_matrix` |
| `RUSTY_LLAMA_GGUF=<path> cargo test --release --features cuda bench_prefill_real_tinyllama -- --ignored --nocapture` | real-model pp512 bench |

**Env overrides:** `RUSTY_LLAMA_GGUF` selects the real GGUF for the real-model
benches (default `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`,
`src/backend/cuda.rs:1202` / `:1321`); `RUSTY_LLAMA_NO_AVX2`, `RUSTY_LLAMA_NO_VNNI`,
`RUSTY_LLAMA_NO_INT8` force scalar/non-fast paths for A/B testing;
`GpuBackend::set_int8_decode(bool)` toggles the DP4A decode path in tests. README
documents the canonical `cargo test` entry point (`README.md:162`).

─────────────────────────────────────────────────────────────────────────────

## 5. In-crate benchmarks & methodology

Benchmarks live as `#[ignore]` tests so they share the build, the synthetic-model
helpers, and the `CpuBackend` baseline. Methodology is consistent: **warm up**
once (dequantize + upload weights so they're resident; build the resident decode
state), **then time**; compare the printed `tok/s` against `llama-bench -ngl 99`.

### Synthetic, comparable shape
`bench_prefill_gpu_vs_cpu` (`src/backend/gpu.rs:2767`) and
`bench_prefill_cuda_vs_cpu` (`src/backend/cuda.rs:1262`) use the **same** shape
(dim 1024, hidden 2816, 4 layers, 16 heads, vocab 4096, 256-token prompt) so the
wgpu-f32 and CUDA-TF32 numbers are directly comparable.
`bench_decode_gpu_vs_cpu` (`gpu.rs:2817`) times 64 decode steps;
`bench_decode_quant_vs_f32` (`gpu.rs:2865`) uses the **TinyLlama-1.1B shape**
(dim 2048, hidden 5632, 22 layers, GQA 32/4, vocab 32000) to compare f32 vs Q8_0
dequant vs Q8_0 int8 GEMV.

### Real-model "north star"
`bench_prefill_real_tinyllama` (`src/backend/cuda.rs:1317`) is the north star: it
loads the **actual** TinyLlama-1.1B Q4_K_M GGUF (22 layers, dim 2048, GQA 32/4),
prefills `512.min(seq_len)` tokens (`:1337`), warms up (weights cached as **f16**
on-device, ~2.2 GB, `:1314`), and prints `pp512` tok/s to compare against
`llama-bench ... -ngl 99` (pp512). `bench_decode_real_tinyllama` (`:1199`) runs
128 resident decode steps (warm-up via `backend_warmup`, `:1251`) for `tg128`.

### Headline numbers (PERFORMANCE.md, 2026-06-16)

Setup: Intel Core Ultra 9 285H (AVX2, **no** AVX-512) + RTX 5070 Ti Laptop GPU
(Blackwell sm_120, 12 GB), CUDA 13.3, llama.cpp release `b9672`.

**Prefill (`pp512`, real TinyLlama-1.1B Q4_K_M)** — `PERFORMANCE.md:141`–147:

| Path | pp512 | vs llama.cpp |
| --- | ---: | --- |
| CUDA, naive attn + f32 weights | 744 | ~26× behind |
| CUDA, tiled attn + f32 | ~2,700 | ~7.3× behind |
| CUDA, tiled attn + f16 | ~3,560 | ~5.5× behind |
| **CUDA, + batched KV download** | **~4,450** | **~4.4× behind** |
| llama.cpp CUDA | **19,637** | — |

**Decode (`tg128`, real TinyLlama-1.1B Q4_K_M)** — `PERFORMANCE.md:180`–184:

| Path | tg128 | |
| --- | ---: | --- |
| rusty_llama CPU | ~52 | — |
| **rusty_llama CUDA (resident, f16)** | **~123** | 2.4× CPU |
| llama.cpp CUDA | **419** | **~3.4× ahead** |

**CPU/GPU decode (separate apples-to-apples)** — `PERFORMANCE.md:20`–27:
CPU autovectorized scalar 36.7 → AVX2 integer kernels **49.8 tok/s** (+36%) vs
llama.cpp CPU 73.6 (1.48×); wgpu 45.9 vs llama.cpp Vulkan 375.9 (8.2×).

**Why the gaps:** prefill is now compute-bound — the residual ~4.4× is
GEMM/kernel efficiency vs llama.cpp's fused tensor-core kernels, not per-op
overhead (`PERFORMANCE.md:170`). Decode's ~3.4× **is weight bandwidth**: we stream
**f16** (~2 B/weight) vs llama.cpp's **Q4_K** (~0.56 B) ≈ 3.5×, which roughly *is*
the gap — making quantized-on-device weights the clear next decode lever
(`PERFORMANCE.md:188`).

### The negative-result benches (kept, not wired)
`bench_kquant_int8_vs_dequant_gemv` (`gpu.rs:3053`) measured int8 DP4A at
~0.94–0.98× (Q4_K) / ~0.70× (Q6_K) of the dequant GEMV — break-even to slower,
because k-quant weights stay packed so int8 saves no bandwidth; kept behind tests
but not adopted (`PERFORMANCE.md:73`–87). `probe_cooperative_matrix`
(`gpu.rs:3255`) is a go/no-go spike that found the portable wgpu-29 coopmat path a
dead end on this driver (advertises only f16 16×16; wgpu wires only 8×8 f32,
which returns all-zero garbage). `bench_dispatch_overhead` (`gpu.rs:3117`)
isolates per-compute-pass cost.

─────────────────────────────────────────────────────────────────────────────

## 6. Golden pre-tokenize fixtures

The byte-level BPE pre-tokenizer must split text byte-identically to the
reference tokenizers. Proof without a network dependency at test time:

1. `scripts/gen_pretokenize_golden.py` runs each model's **documented splitting
   regex** through Python's `regex` module — an engine **independent** of Rust's
   `fancy-regex` (`gen_pretokenize_golden.py:21`; the patterns must stay
   byte-identical to the `PRE_*` constants in `src/tokenizer.rs`, `:25`) — over a
   tricky corpus (contractions incl. uppercase, digit runs, CJK, emoji with
   skin-tone + ZWJ, accents, leading/trailing spaces, tabs, LF/CRLF runs).
2. It emits `tests/fixtures/pretokenize_golden.rs`
   (`@generated`, `pretokenize_golden.rs:1`): a nested table keyed by
   `tokenizer.ggml.pre` name (`gpt-2`, `llama-bpe`, `qwen2`).
   Usage: `python3 scripts/gen_pretokenize_golden.py > tests/fixtures/pretokenize_golden.rs`
   (`gen_pretokenize_golden.py:15`; needs `pip install regex`).
3. `tests/pretokenize.rs` `include!`s the fixture and asserts (a)
   `pretokenize(pre, input) == expected` for every case, ≥90 cases as a
   guard against an empty/garbled fixture (`tests/pretokenize.rs:19`,32), and (b)
   the chunks **reassemble** to the input byte-for-byte — no characters dropped
   or duplicated (`:36`). See `docs/Architecture/07-tokenizer-and-sampler.md`.

─────────────────────────────────────────────────────────────────────────────

## 7. Synthetic models for tests

Tests never need real weights; `src/dummy.rs` fabricates models in both supported
formats (full detail in `docs/Architecture/06-gguf-and-loading.md`):

| Helper | Produces |
| --- | --- |
| `synthetic_checkpoint(&Config)` | a llama2.c v1 checkpoint (noise weights) |
| `synthetic_gguf(&Config)` | a minimal Llama GGUF |
| `synthetic_gguf_typed(&Config, GgmlType)` | GGUF with the matmul weights as F32 or Q8_0 (dim/hidden must be ×32) |
| `synthetic_gguf_gpt2 / _gpt2_special` | GGUF with a GPT-2 BPE vocab / special tokens |
| `dummy_tokenizer()` | an in-memory tokenizer for generation tests |

`examples/make_dummy.rs` is the CLI front-end: `cargo run --release --example
make_dummy -- dummy.gguf [q8_0]` writes a randomly-weighted model (dim 64, hidden
192, 2 layers, 8 heads, vocab 32000, seq 256, `examples/make_dummy.rs:30`); the
`.gguf` extension picks `synthetic_gguf_typed`, anything else picks
`synthetic_checkpoint` (llama2.c, needs a separate `tokenizer.bin`,
`:42`). Weights are noise, so output is gibberish — it only proves the pipeline
runs. For *real* assets, `scripts/download_assets.sh` fetches Karpathy's
`stories15M` (`.bin`) and the llama2.c `tokenizer.bin`.

─────────────────────────────────────────────────────────────────────────────

## Status, gaps & notes

- **No CI coverage for the device paths.** GPU/CUDA parity and all benchmarks are
  `#[ignore]`/feature-gated and only ever run on a machine with the hardware; the
  default `cargo test` (and any cloud CI) exercises **only** CPU + synthetic
  paths. The CUDA suite has never run unattended.
- **VNNI kernels can't be executed on the dev machine** (Core Ultra is AVX2-only,
  no AVX-512), so `*_simd_matches_scalar` early-return here; only
  `q6_k_vnni_algorithm_matches_scalar` (a scalar re-implementation check) and the
  AVX2 tests give real on-machine coverage of the integer fast path. NEON has no
  path or tests at all (`BACKLOG.md:103`).
- **Benchmarks are single-shot, not statistical** — one warm-up + one timed run,
  no repetitions/percentiles. The CPU prefill baseline is explicitly unreliable
  under sustained GPU load (laptop thermal throttling, `PERFORMANCE.md:167`), so
  CUDA-vs-llama.cpp is the only trustworthy prefill comparison; synthetic prefill
  numbers (~735 tok/s) are far below the real-model figure's relevance.
- **Parity is logit/argmax-level, not perplexity-level.** Tests assert finite,
  deterministic, same-greedy-token, and bounded L2 — there is no
  perplexity/quality regression harness, so subtle quantization quality drift
  would not be caught.
- **int8 decode path is correctness-gated but perf-shelved** (k-quant DP4A proven
  not worth wiring); the real GPU decode win awaits tensor cores
  (`PERFORMANCE.md:88`).
- The real-model benches depend on a ~640 MB GGUF that isn't vendored; without it
  (or `RUSTY_LLAMA_GGUF`) they skip with a printed note.

llama.cpp counterpart: `docs/Research/08-capabilities-and-tooling.md` — its
`llama-bench` is the external yardstick (`pp512` / `tg128`, `-ngl 99`) every
real-model number here is measured against.
