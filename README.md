# rusty_llama

A small, from-scratch **Llama inference engine in Rust** — think "llama.cpp,
the Rust way", built up one layer at a time so every piece is understandable.

It runs both [Karpathy llama2.c](https://github.com/karpathy/llama2.c)
checkpoints (`stories15M.bin` and friends) **and GGUF files** (llama.cpp's
format, including quantized weights), on the CPU — with an optional
[`wgpu`](https://github.com/gfx-rs/wgpu) GPU backend (Vulkan/DX12/Metal). All
the transformer math — RMSNorm, RoPE, grouped-query attention, a KV cache,
SwiGLU, the BPE tokenizer, and the quant block dequantizers — is written by
hand. No `candle`, no `ggml` bindings.

## Quick start

```sh
# 1. Get a model + tokenizer (≈60 MB; needs open network access)
./scripts/download_assets.sh

# 2. Generate
cargo run --release -- stories15M.bin -z tokenizer.bin -i "Once upon a time" -n 256
```

### GGUF models

GGUF files carry their own tokenizer (SentencePiece for `llama` models, GPT-2
byte-level BPE for `gpt2` models such as Llama-3 / Qwen2), so no `-z` is needed
— just point at the file (weights may be F32/F16/Q4_0/Q8_0/Q4_K/Q6_K):

```sh
cargo run --release -- TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf -i "The meaning of life is" -n 128
```

No network? Try the pipeline with a randomly-weighted model (gibberish, but it
proves the whole path works). The output format is picked by extension:

```sh
# GGUF (self-contained, embedded tokenizer):
cargo run --release --example make_dummy -- dummy.gguf
cargo run --release -- dummy.gguf -i "hi" -n 16

# llama2.c (needs a real tokenizer.bin for readable sub-words):
cargo run --release --example make_dummy -- dummy.bin
cargo run --release -- dummy.bin -z tokenizer.bin -i "Once upon a time" -n 32
```

### CLI

```
rusty_llama <checkpoint.bin> [options]
  -z <path>     tokenizer file            (default: tokenizer.bin)
  -t <float>    temperature, 0 = greedy   (default: 1.0)
  -p <float>    top-p / nucleus sampling  (default: 0.9)
  -s <int>      RNG seed                  (default: wall-clock time)
  -n <int>      number of steps to run    (default: 256)
  -i <text>     prompt                    (default: "")
  --backend <b> compute backend: cpu|gpu  (default: cpu; gpu needs --features gpu)
```

### GPU backend (optional)

A second `Backend` implementation runs the transformer primitives as WGSL
compute shaders via `wgpu`. It is off by default (keeping the base build at
three small deps) and built in with a cargo feature:

```sh
cargo run --release --features gpu -- stories15M.bin -z tokenizer.bin \
  -i "Once upon a time" -n 256 --backend gpu
```

The matmul weights are uploaded to the device once and kept resident (cached by
data pointer). Q8_0/Q4_K/Q6_K weights stay in their **quantized** GGUF blocks
on the device and are dequantized *in the matmul shader* (like the CPU's
per-block path), so the device streams ~4–7× less weight data per token than
expanding to f32 would; other formats are dequantized to f32 on the host at
upload. Greedy output is **byte-identical** to the CPU backend (verified on
`stories15M` and a quantized TinyLlama-1.1B Q4_K_M; see `tests/gpu_parity.rs`).

**Prompt prefill is batched.** The whole prompt runs in one set of *large*
matmuls (`forward_prefill`) rather than one tiny pass per token, via batched
`Backend` ops (`matmul_batch`/`rmsnorm_batch`/`rope_batch`/`attention_batch`,
with default impls that loop the single-token ops so the CPU is correct for
free). That's where the GPU shines: prefilling a 256-token prompt is **~2×
faster on GPU than CPU** (735 vs 377 tok/s, `bench_prefill_gpu_vs_cpu`).

**Decode is fused on-device with a cooperative GEMV.** A single decode step
(`Backend::forward_step`) keeps the KV cache and activations resident and records
the whole step into one command encoder — one submission, one logits read-back.
The matmuls use a **cooperative GEMV** (one workgroup per output row, threads
reduce the dot product with coalesced weight reads) instead of one thread per
row; profiling showed the old kernel — not dispatch overhead — was the
bottleneck (only ~18 GB/s effective). Together these took TinyLlama Q4_K_M decode
from **6 → ~37 tok/s** and **VRAM from ~4.5 GB → ~1.0 GB**.

**Performance summary.** The GPU now wins across the board on this hardware
(RTX 5070 Ti): a resident 4096² matmul ~1.2× a multi-core CPU; batched prefill
~2× (`bench_prefill_gpu_vs_cpu`); and **decode now beats the CPU** — TinyLlama
Q4_K_M ≈37 vs ≈34 tok/s, and a synthetic dim-1024 model 587 vs 229 tok/s
(`bench_decode_gpu_vs_cpu`). One caveat: the cooperative reduction sums in a
different (parallel) order than the CPU's sequential dot, so over a long greedy
run the GPU's output can eventually diverge from the CPU's *exact* token
sequence by a near-tie tie-break — both remain coherent, and per-op outputs
match within f32 tolerance. See the `bench_*_gpu_vs_cpu` benchmarks in
`src/backend/gpu.rs`.

## How it fits together

```
 Checkpoint (mmap)  ──parse──▶  Model { Config, Weights }
                                     │
 Tokenizer.encode(prompt) ─tokens─▶  forward(token, pos) ─logits─▶ Sampler
                                     │  (uses a Backend)              │
                                     └──────────── KV cache ◀─────────┘
```

| Module          | Responsibility                                              |
| --------------- | ----------------------------------------------------------- |
| `config`        | Parse the 7-`int32` checkpoint header into hyper-parameters |
| `loader`        | `mmap` the checkpoint file (zero-copy weights)              |
| `gguf`          | Parse the GGUF container (metadata + tensor table)          |
| `quant`         | GGML type table, per-block dequantizers, integer dot kernels (scalar + AVX-512 VNNI) |
| `tensor`        | `QMatrix`: an f32-or-quantized weight matrix, dequantized on demand |
| `model`         | Weight layout, KV-cache scratch, `forward`, and `from_gguf` |
| `backend`       | `Backend` trait + `CpuBackend` (rayon) + optional `GpuBackend` (wgpu, `gpu` feature) |
| `tokenizer`     | SentencePiece (SPM) + GPT-2 byte-level BPE encode/decode    |
| `sampler`       | Greedy / temperature / top-p sampling                       |

The forward pass only ever calls `Backend` methods (`matmul`, `rmsnorm`,
`rope`, `attention`, `swiglu`, `add`), so a future GPU backend just implements
the trait — the model code doesn't change.

## Performance notes

`.cargo/config.toml` sets `target-cpu=native` so the f32 inner loops compile to
AVX2/AVX-512/FMA on this machine. The two hot kernels (`matmul`, `attention`)
parallelize across cores with rayon. If you copy the binary to a different CPU,
change that to `x86-64-v3` or remove it.

Quantized GGUF weights are **not** expanded to f32 at load time — they're
memory-mapped and the matmul takes an integer fast path: the activation is
quantized once per matmul (Q8 for Q8_0/Q4_0, Q8_K for Q4_K/Q6_K) and the row
dot products run in integer arithmetic, with only the per-block scales in f32.
That's ≈3× faster than per-block f32 dequant on this machine (see the
`bench_q8_0_matmul` / `bench_q4_k_matmul` tests) and uses a fraction of the RAM
an f32 copy would.

On CPUs with **AVX-512 VNNI**, the Q8_0/Q4_0/Q4_K integer dot products use a
hand-written `vpdpbusd` kernel (`src/quant.rs`, the `x86` module) selected at
run time via `is_x86_feature_detected!`, with the autovectorized scalar loop as
the fallback everywhere else. Because the whole reduction stays in integer
arithmetic, the SIMD result is **bit-for-bit identical** to the scalar one (the
`*_simd_matches_scalar` tests assert this), so it's a pure speed swap with no
effect on output. It runs ≈2.2–2.5× faster than the autovectorized path at the
kernel level (`bench_*_simd_vs_scalar`) and ≈2× end to end on a quantized model.
Set `RUSTY_LLAMA_NO_VNNI=1` to force the scalar path (handy for A/B timing).

## Tests

```sh
cargo test
```

Unit tests cover each kernel against hand-computed values; integration tests
mmap a synthetic checkpoint and check the forward pass is finite, deterministic,
and that greedy generation reproduces.

## Roadmap

- [x] llama2.c fp32 checkpoints, CPU backend, BPE tokenizer, sampling
- [x] GGUF container + quantized weights (F16, Q4_0, Q8_0, Q4_K, Q6_K) + GGUF SPM tokenizer
- [x] On-the-fly quantized matmul — weights stay compressed in RAM (mmap), dequantized per block
- [x] Integer (Q8 / Q8_K) activation fast path for Q8_0/Q4_0/Q4_K/Q6_K matmuls (~3× over per-block dequant)
- [x] Per-model RoPE base (θ) and RMSNorm epsilon read from GGUF (Llama-3 / Qwen2 θ honoured)
- [x] Byte-level BPE (`gpt2`) tokenizer — Llama-3 / Qwen2 vocabularies load
- [x] Byte-exact pretokenizer regex (`gpt-2`/`llama-bpe`/`qwen2` via `fancy-regex`) + special-token handling
- [x] RoPE long-context scaling (`linear`/`yarn`/`llama3`)
- [x] Explicit AVX-512 VNNI (`vpdpbusd`) integer dot products for Q8_0/Q4_0/Q4_K (~2.2–2.5×, bit-identical to scalar)
- [x] GPU backend (`wgpu`) behind the existing `Backend` trait — per-op parity + byte-identical e2e output; decode currently latency-bound (honest verdict above)

See [`BACKLOG.md`](BACKLOG.md) for the history of these items.

## Credits

The model format, tokenizer format, and reference math come from Andrej
Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) (MIT).
