# rusty_llama

A small, from-scratch **Llama inference engine in Rust** — think "llama.cpp,
the Rust way", built up one layer at a time so every piece is understandable.

The first milestone runs [Karpathy llama2.c](https://github.com/karpathy/llama2.c)
checkpoints (`stories15M.bin` and friends) on the CPU, with all the transformer
math — RMSNorm, RoPE, grouped-query attention, a KV cache, SwiGLU, and the BPE
tokenizer — written by hand. No `candle`, no `ggml` bindings.

## Quick start

```sh
# 1. Get a model + tokenizer (≈60 MB; needs open network access)
./scripts/download_assets.sh

# 2. Generate
cargo run --release -- stories15M.bin -z tokenizer.bin -i "Once upon a time" -n 256
```

No network? Try the pipeline with a randomly-weighted model (gibberish, but it
proves the whole path works):

```sh
cargo run --release --example make_dummy -- dummy.bin
# (you still need a real tokenizer.bin for readable sub-words)
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
| `model`         | Weight layout, KV-cache scratch, and the `forward` pass     |
| `backend`       | `Backend` trait + `CpuBackend` (rayon + autovectorized SIMD)|
| `tokenizer`     | SentencePiece-style BPE encode/decode                       |
| `sampler`       | Greedy / temperature / top-p sampling                       |

The forward pass only ever calls `Backend` methods (`matmul`, `rmsnorm`,
`rope`, `attention`, `swiglu`, `add`), so a future GPU backend just implements
the trait — the model code doesn't change.

## Performance notes

`.cargo/config.toml` sets `target-cpu=native` so the f32 inner loops compile to
AVX2/AVX-512/FMA on this machine. The two hot kernels (`matmul`, `attention`)
parallelize across cores with rayon. If you copy the binary to a different CPU,
change that to `x86-64-v3` or remove it.

## Tests

```sh
cargo test
```

Unit tests cover each kernel against hand-computed values; integration tests
mmap a synthetic checkpoint and check the forward pass is finite, deterministic,
and that greedy generation reproduces.

## Roadmap

- [x] llama2.c fp32 checkpoints, CPU backend, BPE tokenizer, sampling
- [ ] GGUF container + quantized weights (Q8_0, Q4_K, …)
- [ ] Explicit SIMD kernels & quantized dot-products
- [ ] GPU backend (`wgpu`/CUDA) behind the existing `Backend` trait
- [ ] Chat templating / instruction models

## Credits

The model format, tokenizer format, and reference math come from Andrej
Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) (MIT).
