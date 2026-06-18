# Research

In-depth technical captures of external systems studied to inform `rusty_llama`.

─────────────────────────────────────────────────────────────────────────────

## llama.cpp architecture & capabilities

A detailed, source-grounded end-to-end capture of [llama.cpp](https://github.com/ggml-org/llama.cpp),
done to steer `rusty_llama`'s roadmap (feature breadth vs raw speed).

- **Snapshot:** commit `fe7c8b2414bb39e3dbc192fb494a1eb054b90890`, captured 2026-06-18 (MIT).
- **Grounding:** every numeric/structural claim is cited to a source file at that
  commit; unverifiable claims are marked `[unverified]`.

### Read order

| # | Doc | Covers |
|---|---|---|
| 00 | [`00-overview.md`](00-overview.md) | **Start here.** Two-layer design, repo map, e2e request lifecycle, the performance model, capability surface |
| 01 | [`01-ggml-core-and-graph.md`](01-ggml-core-and-graph.md) | ggml tensor struct, the ~90-op graph, define-then-run, `ggml-alloc` memory planner, type traits |
| 02 | [`02-backends-and-dispatch.md`](02-backends-and-dispatch.md) | Backend abstraction, registry + dynamic loading, the `ggml_backend_sched` scheduler (hybrid CPU+GPU), backend matrix, CPU SIMD dispatch |
| 03 | [`03-cuda-kernels.md`](03-cuda-kernels.md) | **The speed story.** Matmul routing (MMVQ ≤8 / MMQ >8), MMQ int8 tensor-core GEMM, MMVQ packed-weight DP4A GEMV, flash attention, `q8_1` activation quant |
| 04 | [`04-quantization.md`](04-quantization.md) | All quant families + block layouts + bits/weight, k-quant super-blocks, i-quants, the `_M` mixed-precision scheme, imatrix |
| 05 | [`05-gguf-and-model-loading.md`](05-gguf-and-model-loading.md) | GGUF binary format, model loader (mmap, sharding, async upload), the 132-architecture registry, HF→GGUF conversion |
| 06 | [`06-inference-pipeline.md`](06-inference-pipeline.md) | The `llama_decode` lifecycle, graph build + reuse, KV-cache variants (paged/iSWA/recurrent/hybrid), continuous batching, speculative decoding |
| 07 | [`07-tokenization-and-sampling.md`](07-tokenization-and-sampling.md) | 6 tokenizer families + pretokenizers, the ~18-sampler catalog + default chain, GBNF grammar + JSON-schema |
| 08 | [`08-capabilities-and-tooling.md`](08-capabilities-and-tooling.md) | llama-server (OpenAI/Anthropic API, slots, continuous batching), multimodal, LoRA, chat/tools, embeddings, llama-bench, distributed |
| 09 | [`09-rusty-llama-gap-analysis.md`](09-rusty-llama-gap-analysis.md) | **The payoff.** rusty_llama vs llama.cpp capability matrix + ROI-ranked roadmap |

### TL;DR

llama.cpp = **ggml** (a define-then-run tensor-op graph executed by pluggable
backends) + **llama** (GGUF loading, per-arch graph build, KV-cache, tokenize,
sample). It is fast because it attacks prefill (compute-bound → int8 tensor-core
MMQ + flash attention) and decode (bandwidth-bound → packed-weight DP4A MMVQ) by
their *separate* bottlenecks, and because weights stay in packed quant format
end-to-end. For where that leaves `rusty_llama` and what to build next, see
[`09`](09-rusty-llama-gap-analysis.md).
