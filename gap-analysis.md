# rusty_llama vs. llama.cpp — parity-loop gap analysis (2026-07-23)

Reference: llama.cpp (as characterized in `docs/Research/00`–`08`, snapshot
`fe7c8b2`, 2026-06-18) plus this repo's own `README.md` roadmap, `BACKLOG.md`
(items `L7`–`L11`, `D4`), and `docs/Research/09-rusty-llama-gap-analysis.md`.
Not a symbol-name diff (`rusty_llama` is an app, not a library) — this is a
manual re-audit of `docs/Research/09` against the *current* source, since a lot
of what that doc calls "missing" (samplers, LoRA, server, MoE, grammar,
embeddings, chat templates) has since shipped.

**Environment constraint (new since the last human session):** this run has
**no GPU** (`nvidia-smi`/`vulkaninfo` absent, no Vulkan/CUDA device) and 4
CPU cores. GPU/CUDA items (`BACKLOG.md` L7–L11, tensor-core MMQ, wgpu
megakernel) cannot be built-and-verified here — they need real hardware and
are **excluded from this run's issue list**, not because they're not gaps, but
because "implement + locally verify before push" (the loop's own gate) isn't
possible without a GPU. Everything below is CPU-only and testable in this
environment.

Confirmed already shipped since `docs/Research/09` (re-verified against
current source, so **not** listed as gaps): sampler chain + top-k/min-p/
penalties (`src/sampler.rs`), LoRA + control-vector-shaped adapters
(`src/adapter.rs`), embeddings/pooling (`src/main.rs --embedding`), GBNF
grammar (`src/grammar.rs`), OpenAI-compatible server + continuous batching
(`src/server.rs`), MoE (Mixtral/Qwen2-MoE), flash attention, 5 architectures
(Llama/Qwen2/Phi-3/Gemma2/MoE variants), chat-template detection for 5 known
families (`src/chat.rs`).

| Gap | Category | Platforms | Reference | Breaking? | Est. size | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `GGML_TYPE_Q4_1` consume | quant type | CPU | `ggml.c` legacy quants | no | S | Same 32-elem block shape as existing `Q4_0`; dequant + `QMatrix`/gguf wiring only, scalar dot path (no SIMD kernel required to close the gap) |
| `GGML_TYPE_Q5_0` consume | quant type | CPU | `ggml.c` legacy quants | no | S | 32-elem block + 1-bit high-bit plane; same shape family as `Q4_0`/`Q8_0` |
| `GGML_TYPE_Q5_1` consume | quant type | CPU | `ggml.c` legacy quants | no | S | `Q5_0` + a min/delta pair (mirrors `Q4_1`'s relation to `Q4_0`) |
| `GGML_TYPE_Q5_K` consume | quant type | CPU | `ggml.c` k-quants | no | M | 256-elem superblock like existing `Q4_K`/`Q6_K`; more scale-packing complexity than the legacy quants above |
| `GGML_TYPE_Q3_K` consume | quant type | CPU | `ggml.c` k-quants | no | M | 3-bit + 6-bit-packed scales in a non-byte-aligned layout — trickier bit-unpacking than `Q5_K` |
| `GGML_TYPE_Q2_K` consume | quant type | CPU | `ggml.c` k-quants | no | M | 2-bit weights + 4-bit scales; smallest/lowest-fidelity k-quant, same superblock family |
| Mirostat (v1/v2) sampler | fn (new sampler stage) | CPU | `llama_sampler_init_mirostat{,_v2}` | no | S | New `SamplerStage` impl, same seam `TopK`/`MinP`/`Penalties` already use (`src/sampler.rs`) |
| DRY sampler | fn (new sampler stage) | CPU | `llama_sampler_init_dry` | no | M | Needs a repeat-sequence (z-algorithm-ish) scan over recent tokens; more involved than mirostat but still additive |
| XTC sampler | fn (new sampler stage) | CPU | `llama_sampler_init_xtc` | no | S | Threshold + probability filter over the already-sorted candidate list |
| Sharded GGUF loading | fn (loader) | CPU | `llama.cpp` `--split` / `split.count` GGUF KV | no | M | Multi-file models (`model-00001-of-00005.gguf`, ...) currently unsupported (`grep shard` in `src/gguf.rs`/`src/loader.rs`: no hits) — needed for most large modern GGUF releases |
| Jinja-templated chat rendering | fn (existing, capability gap) | CPU | `llama.cpp` `minja` template engine | **maybe** | L | `src/chat.rs` pattern-matches `tokenizer.chat_template` to 5 known families + hand-written renderers, not the actual embedded Jinja source — an unrecognized/custom template silently falls through to a guess. Full jinja is out of proportion; closing this for real needs either a new (small) template-engine dependency **or** a hand-rolled minimal-subset interpreter — flagging as a dependency/scope decision, not auto-implementing |
| Paged / multi-sequence KV-cache | fn (existing, breaking) | CPU | `llama.cpp` `llama_kv_cache` (paged) | **yes** | L | README's own roadmap already calls this out as the server's "next step." Touches `RunState`/`Backend` KV layout used by every backend (CPU/GPU/CUDA) — a public-surface-shaping change, not a pure addition. Flagging per the loop's breaking-change rule rather than splitting into unattended-mergeable pieces |

## Explicitly out of scope this run

- **GPU/CUDA depth levers** (`BACKLOG.md` L7 int8 MMQ tensor-core prefill, L8
  tensor-core flash attention, L10 wgpu megakernel, L11 further CUDA decode
  fusion) — no GPU in this environment to build+verify against; re-run on a
  CUDA/Vulkan-capable machine.
- **i-quants / MXFP4 / ternary (TQ) formats** — `docs/Research/09` Tier 4,
  deliberately deprioritized by the repo's own ROI analysis (format breadth
  with low real-model demand); not re-litigating that call here.
- **Multimodal, distributed/RPC, HF→GGUF conversion tooling** — `docs/
  Research/09` Tier 4 / "correctly out of scope," same reasoning.
- **L9 (aggressive CPU GEMM micro-kernel tuning)** — CPU-only and technically
  buildable here, but the repo's own `BACKLOG.md` already scoped it as a
  large, high-risk "treadmill" item (llama.cpp-class hand-tuned GEMM, not a
  bounded win) — leaving it out of an unattended issue-per-gap loop; worth a
  deliberate one-off session instead if wanted.
