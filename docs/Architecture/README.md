# Architecture

Source-grounded capture of `rusty_llama`'s own design and current state — the
internal companion to the external [`../Research/`](../Research/) capture of
llama.cpp.

- **Snapshot:** branch `main`, commit `e2a7d5d`, captured 2026-06-18 (10,772 LOC
  `src/` + 866 `tests/`).
- **Grounding:** every claim is cited to `path:line` in the source at that commit.

## Read order

| # | Doc | Covers |
|---|---|---|
| 00 | [`00-overview.md`](00-overview.md) | **Start here.** The `Backend`-trait design, crate map, public API, e2e generation lifecycle, the three backends, perf snapshot |
| 01 | [`01-model-and-forward-pass.md`](01-model-and-forward-pass.md) | `Config`/`RopeScaling`/`RopeTable`, `Weights`/`RunState`, the eager op-by-op forward pass, `forward_prefill`, `generate` |
| 02 | [`02-backend-trait-and-cpu.md`](02-backend-trait-and-cpu.md) | The `Backend` trait contract + `CpuBackend` (rayon, two-tier quantized-matmul dispatch, the parity oracle) |
| 03 | [`03-gpu-backend-wgpu.md`](03-gpu-backend-wgpu.md) | wgpu backend: 21 WGSL pipelines, packed-weight cooperative GEMV (f32 + int8 DP4A), resident decode, the coopmat ruling |
| 04 | [`04-cuda-backend.md`](04-cuda-backend.md) | CUDA backend: cudarc dynamic-load, cuBLASLt f16 GEMM + weight cache, nvrtc/PTX kernels, resident prefill + decode |
| 05 | [`05-quantization.md`](05-quantization.md) | `GgmlType`, block layouts + bits/weight, quantizers, int8 dot kernels (VNNI/AVX2/scalar), `QMatrix` |
| 06 | [`06-gguf-and-loading.md`](06-gguf-and-loading.md) | GGUF parser, both load paths (GGUF + llama2.c), zero-copy tensor borrow, rope-scaling keys, synthetic builders |
| 07 | [`07-tokenizer-and-sampler.md`](07-tokenizer-and-sampler.md) | SPM + GPT-2 BPE tokenizers + pretokenizers, the `Sampler` (greedy/temp/top-p), the CLI |
| 08 | [`08-testing-benchmarking-parity.md`](08-testing-benchmarking-parity.md) | The CPU-oracle parity strategy, per-file test coverage, device-gated benches + methodology, golden fixtures |
| 09 | [`09-status-and-roadmap.md`](09-status-and-roadmap.md) | **Where it stands.** Roadmap burndown, current numbers, doc-drift found, ROI-ranked next steps |
| 10 | [`10-evolution-plan.md`](10-evolution-plan.md) | **The plan.** Phased, ROI-ranked execution plan to evolve toward llama.cpp-class architecture + capabilities, with dependencies, acceptance gates, risks, decision points |

### TL;DR

`rusty_llama` runs the Llama-family transformer through a single `Backend` trait
(CPU / wgpu / CUDA), eager and per-op, loading GGUF or llama2.c weights borrowed
from an mmap. CPU is mature (1.48× behind llama.cpp) and serves as the parity
oracle; the CUDA backend is the speed frontier (prefill ~4.4×, decode ~3.4×
behind — bounded by the cuBLASLt GEMM wall and f16 weight bandwidth respectively).
What it lacks is breadth (one arch, four samplers, no server/LoRA/grammar). For
the full status and what to build next, see
[`09`](09-status-and-roadmap.md); for the external comparison, [`../Research/`](../Research/).
