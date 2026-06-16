# rusty_llama

A small, from-scratch **Llama inference engine in Rust** — think "llama.cpp,
the Rust way", built up one layer at a time so every piece is understandable.

It runs both [Karpathy llama2.c](https://github.com/karpathy/llama2.c)
checkpoints (`stories15M.bin` and friends) **and GGUF files** (llama.cpp's
format, including quantized weights), on the CPU. All the transformer math —
RMSNorm, RoPE, grouped-query attention, a KV cache, SwiGLU, the BPE tokenizer,
and the quant block dequantizers — is written by hand. No `candle`, no `ggml`
bindings.

## Quick start

```sh
# 1. Get a model + tokenizer (≈60 MB; needs open network access)
./scripts/download_assets.sh

# 2. Generate
cargo run --release -- stories15M.bin -z tokenizer.bin -i "Once upon a time" -n 256
```

### GGUF models

GGUF files carry their own tokenizer, so no `-z` is needed — just point at the
file (llama-architecture SentencePiece models such as Llama-2 / TinyLlama;
weights may be F32/F16/Q4_0/Q8_0/Q4_K/Q6_K):

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
  -z <path>   tokenizer file              (default: tokenizer.bin)
  -t <float>  temperature, 0 = greedy     (default: 1.0)
  -p <float>  top-p / nucleus sampling    (default: 0.9)
  -s <int>    RNG seed                    (default: wall-clock time)
  -n <int>    number of steps to run      (default: 256)
  -i <text>   prompt                      (default: "")
```

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
| `quant`         | GGML type table + per-block dequantizers (Q4_0/Q8_0/Q4_K/Q6_K/F16) |
| `tensor`        | `QMatrix`: an f32-or-quantized weight matrix, dequantized on demand |
| `model`         | Weight layout, KV-cache scratch, `forward`, and `from_gguf` |
| `backend`       | `Backend` trait + `CpuBackend` (rayon; f32 + on-the-fly quant matmul) |
| `tokenizer`     | SentencePiece-style BPE encode/decode (from `.bin` or GGUF) |
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
- [ ] Byte-level BPE (`gpt2`) tokenizer + RoPE long-context scaling → full Llama-3 / Qwen2 support
- [ ] Explicit SIMD intrinsics (e.g. AVX-512 VNNI) for the integer dot products
- [ ] GPU backend (`wgpu`/CUDA) behind the existing `Backend` trait

See [`BACKLOG.md`](BACKLOG.md) for details on the unchecked items (the GPU
backend is tagged for a local instance with GPU access).

## Credits

The model format, tokenizer format, and reference math come from Andrej
Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) (MIT).
