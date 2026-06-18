# 08. Capabilities & Tooling

## Summary

This doc inventories the **capability breadth** llama.cpp ships *around* its inference core: the OpenAI/Anthropic-compatible `llama-server` (REST + WebUI, slots, continuous batching, prompt-cache reuse, structured output, tool calling, multimodal, speculative decoding, router/multi-model), the `libmtmd` multimodal stack (vision/audio projectors spliced into the token stream), LoRA + control-vector adapters, the Jinja chat-template + tool/reasoning parsing layer, embeddings/reranking modes, the evaluation/benchmark tooling (`llama-bench`, `perplexity`, `imatrix`), distributed inference (RPC backend, multi-GPU split modes), and a WIP training path (`ggml-opt`). The three things that matter most for rusty_llama's roadmap: (1) **all of this is product surface area rusty_llama does not have** — it is a CLI + bench harness only; (2) `llama-bench` is the *exact* harness rusty_llama compares against, so its methodology (pp/tg, warmup, `-r` repetitions, excludes tokenize+sample) defines the scoreboard; (3) the server's slot/continuous-batching/KV-reuse machinery is what turns a single-stream engine into a multi-user service — a large but well-isolated greenfield.

Sampler internals are covered in `07-tokenization-and-sampling.md`; KV-cache/batching internals in `06-inference-pipeline.md`; quantization + imatrix in `04-quantization.md`; CUDA kernels in `03-cuda-kernels.md`. This doc stays at the capability/API layer.

─────────────────────────────────────────────

## 1. `llama-server` — OpenAI-compatible inference service

`tools/server/server.cpp` is a single-binary HTTP server built on `cpp-httplib` + `nlohmann::json` (`tools/server/README.md`). It exposes llama.cpp inference as REST, supports F16/quantized models on CPU+GPU, OpenAI **and** Anthropic Messages compatibility, reranking, parallel multi-user decoding, continuous batching, multimodal, schema-constrained JSON, assistant-prefill, function calling "for ~any model", speculative decoding, and a bundled web UI.

### 1.1 Endpoint catalog

| Endpoint | Method | Purpose / notes |
|---|---|---|
| `/health`, `/v1/health` | GET | Liveness; 503 while loading, 200 `{"status":"ok"}`. Public (no API key). |
| `/completion` | POST | **Non-OAI** raw completion. Accepts string, token array, mixed, or `{prompt_string, multimodal_data:[b64]}`; multi-prompt → array result. All sampler params + `cache_prompt`, `n_cache_reuse`, `lora`, `id_slot`, `n_probs`, `logit_bias`, `t_max_predict_ms`. |
| `/v1/completions` | POST | OAI Completions. |
| `/v1/chat/completions` | POST | OAI Chat. ChatML default; `response_format` json_object / json_schema; `tools`/`parallel_tool_calls`; `reasoning_format`; multimodal via `image_url`/`input_audio`/`input_video`; `chat_template_kwargs` (e.g. `{"enable_thinking":false}`). |
| `/v1/chat/completions/control` | POST | Real-time control of an in-flight completion by `id`; action `reasoning_end` forces end of thinking block (needs `reasoning_control:true`). |
| `/v1/responses` | POST | OAI Responses API. |
| `/v1/models`, `/models` | GET | Model info; `meta` block (n_vocab, n_ctx_train, n_embd, n_params, size); reports `multimodal` capability. |
| `/tokenize` | POST | Text→tokens; `add_special`, `parse_special`, `with_pieces` (id+piece, bytes for invalid UTF-8). |
| `/detokenize` | POST | Tokens→text. |
| `/apply-template` | POST | Render chat template to a prompt string, no inference (lets clients prefill "Sure!"). |
| `/embedding`, `/embeddings` | POST | **Non-OAI** embeddings; `/embeddings` supports `--pooling none` (per-token unnormalized vectors). |
| `/v1/embeddings` | POST | OAI embeddings; `embd_normalize`. |
| `/rerank`, `/reranking`, `/v1/rerank`, `/v1/reranking` | POST | Rerank `documents` by `query`; needs reranker model + `--embedding --pooling rank`. |
| `/infill` | POST | FIM code infill: `input_prefix`/`input_suffix`/`input_extra`/`prompt`; uses `FIM_*` tokens, repo-level pattern when `FIM_REPO`/`FIM_FILE_SEP` exist. |
| `/props` | GET/POST | Global props: default gen settings, `total_slots`, `model_path`, `chat_template`, `chat_template_caps`, `modalities`, `media_marker`, `build_info`, `is_sleeping`. POST gated by `--props`. |
| `/slots` | GET | Per-slot state (params, tokens, speculative flag, `chat_format`, `reasoning_format`); `?fail_on_no_slot=1`; disable with `--no-slots`. |
| `/slots/{id}?action=save\|restore\|erase` | POST | Persist/restore/erase a slot's prompt-cache to/from `--slot-save-path`. |
| `/metrics` | GET | Prometheus metrics (gated by `--metrics`). |
| `/lora-adapters` | GET/POST | List / set global LoRA scales (per-request `lora` overrides). |
| `/v1/audio/speech` | POST | TTS output, enabled by `--talker-model` (qwen3-omni talker + `--code2wav-model`). |

Prometheus metrics (`/metrics`) include `llamacpp:prompt_tokens_total`, `prompt_tokens_seconds` (gauge t/s), `tokens_predicted_total`, `predicted_tokens_seconds`, `requests_processing`, `requests_deferred`, `n_decode_total`, `n_busy_slots_per_decode` — i.e. server-grade throughput observability.

### 1.2 Slots, continuous batching, KV reuse, parallelism

- **Slots** (`-np/--parallel N`, `-1`=auto): each slot is an independent concurrent sequence with its own KV-cache region; `total_slots` is reported in `/props`. `-kvu/--kv-unified` shares one KV buffer across sequences (default when slots auto).
- **Continuous batching** (`-cb`, default on): tokens from many in-flight requests are merged into shared `llama_decode` ubatches — the multi-user throughput lever. `n_busy_slots_per_decode` measures fill.
- **Prompt cache & reuse** (`--cache-prompt` default on; `--cache-reuse N`): on each request the prompt is diffed against the slot's prior prompt and only the unseen suffix is evaluated; `--cache-reuse` reuses non-prefix chunks via KV **shifting**. `-sps/--slot-prompt-similarity` chooses which slot to reuse. RAM cache pool sized by `-cram/--cache-ram` (default 8192 MiB); context checkpoints (`--ctx-checkpoints`, `--checkpoint-min-step`) snapshot SWA/context state per slot. (KV-shift + checkpoint mechanics: `06-inference-pipeline.md`.)
- **Router / multi-model mode**: `--models-dir`, `--models-preset`, `--models-max` (default 4), `--models-autoload` let one server host and lazily load several models; `/metrics` and `/v1/chat/completions/control` then require `?model=`.
- **Sleeping on idle**: `--sleep-idle-seconds` frees compute when idle; `/props` reports `is_sleeping`.

### 1.3 Structured output, multimodal input, speculative, WebUI

- **Structured output**: `grammar`/`json_schema` on `/completion`; `response_format` `{type:json_object|json_schema, schema:…}` on chat (JSON-Schema→GBNF via `common/json-schema-to-grammar.cpp`). Backed by the constrained-sampling grammar engine — see `07-tokenization-and-sampling.md`.
- **Multimodal** (PR #12898, experimental): available on OAI chat, non-OAI `/completion`, and `/embedding`. Clients must check the `multimodal` capability in `/models` first; base64 media goes in `multimodal_data` with one `media_marker` (`mtmd_default_marker()`, e.g. `<__media__>`) per item. Detail in §2.
- **Speculative decoding in-server**: a rich `--spec-type` family — `draft-simple`, `draft-eagle3`, `draft-mtp`, plus `ngram-simple`/`ngram-map-k`/`ngram-map-k4v`/`ngram-mod`/`ngram-cache` (prompt-lookup / no draft model). Draft model fully parameterized (`-md`, `-ngld`, `--spec-draft-n-max/-n-min/-p-min`, its own cache types, threads, device split). Static/dynamic lookup caches via `-lcs`/`-lcd`.
- **Built-in agent tools** (`--tools`, off by default, unsafe): server-side `read_file`, `file_glob_search`, `grep_search`, `exec_shell_command`, `write_file`, `edit_file`, `apply_diff`, `get_datetime` callable by the model from the WebUI.
- **WebUI** (`--ui`, default on): bundled chat front-end served from the same port; `--ui-config`/`--ui-config-file` set defaults; optional MCP CORS proxy (`--ui-mcp-proxy`). Static files via `--path`, API prefix via `--api-prefix`, auth via `--api-key(-file)`, TLS via `--ssl-*` (OpenSSL build).

─────────────────────────────────────────────

## 2. Multimodal — `libmtmd` / clip projector stack

`tools/mtmd` provides multimodal via `libmtmd` (PR #12849, replaced the old `llava.cpp`; unified CLI `mtmd-cli` in #13012). It is explicitly "under very heavy development". Input modalities: **image, audio, video** (`docs/multimodal.md`).

**How an image becomes tokens** (`tools/mtmd/README.md`): a multimodal model is **two GGUF files** — the standard LM, plus a **multimodal projector (`mmproj`)**. The mmproj (built on `clip.cpp`) encodes the raw image/audio into embedding vectors, projects them into the LM's `n_embd` space, and those embeddings are **spliced directly into the token-embedding sequence** at the positions of the media markers — the LM never sees pixels, only embeddings in its hidden dimension. Keeping clip out of `libllama` lets the diverse ViT pre-processing/projection logic evolve independently. mmproj is offloaded to GPU by default (`--no-mmproj-offload` to disable); dynamic-resolution models honor `--image-min-tokens`/`--image-max-tokens`. `convert_hf_to_gguf.py --mmproj` produces the projector for supported families.

| Modality | Example families (GGUF, mostly Q4_K_M) |
|---|---|
| Vision | Gemma 3 / Gemma 4, SmolVLM / SmolVLM2, Pixtral 12B, Qwen2-VL & Qwen2.5-VL, Mistral Small 3.1 24B, InternVL 2.5/3, Llama 4 Scout, Moondream2, MiniCPM-V; OCR variants (PaddleOCR-VL, GLM-OCR, Deepseek-OCR, Dots.OCR, HunyuanOCR) |
| Audio | Ultravox 0.5, Voxtral Mini, Qwen3-ASR, (Qwen2-Audio) |
| Mixed (vision+audio) | Qwen2.5-Omni, Qwen3-Omni, Gemma 4 |

Server multimodal content parts: `image_url` (base64 or URL), `input_audio`, and the OAI-extension `input_video` (base64 only).

─────────────────────────────────────────────

## 3. Adapters — LoRA & control vectors

`src/llama-adapter.cpp` implements two runtime-applied adapter types.

**LoRA** (`llama_adapter_lora`): loaded from a GGUF whose `general.type == "adapter"`, `adapter.type == "lora"`, with matching `general.architecture` (hard mismatch error) and an `adapter.lora.alpha` scalar. Tensors are paired by suffix `.lora_a`/`.lora_b` into `ab_map[name] = {a,b}`; shapes validated against base tensors (a must be transposed — old "finetune" adapters rejected). Each pair is duplicated onto the base tensor's buffer type, except tensors living on **repack/extra** CPU bufts, which fall back to plain CPU (LoRA can't use repacked weights). `_norm.weight` tensors are skipped (TODO). At decode the low-rank delta `scale·B·A·x` is added into the matching weight's output. Multiple adapters load via comma-separated `--lora`/`--lora-scaled FNAME:SCALE`; **hot-swap & per-request scaling** are first-class: `/lora-adapters` GET/POST sets global scales, the per-request `lora:[{id,scale}]` overrides them, and `--lora-init-without-apply` loads at scale 0 for apply-on-demand. Requests with differing LoRA configs are **not batched together**.

**aLoRA** ("activated LoRA"): the adapter GGUF may carry `adapter.alora.invocation_tokens` (UINT32 array); `llama_adapter_get_alora_invocation_tokens()` exposes the trigger sequence so the adapter activates only after a specific token pattern.

**Control vectors** (`llama_adapter_cvec`): a per-layer F32 direction tensor of size `n_embd`; `apply_to()` does `cur = ggml_add(cur, layer_dir)` at each enabled layer. Applied over a layer range (`--control-vector-layer-range START END`), loaded/scaled via `--control-vector` / `--control-vector-scaled FNAME:SCALE`. Layer 0 never has a vector. Passing `data==nullptr` disables but keeps buffers allocated (cheap toggle).

─────────────────────────────────────────────

## 4. Chat templates, tool calling & reasoning parsing

Two layers turn OAI messages into a prompt and parse the model's raw text back into structured fields.

- **Built-in C++ templates** (`src/llama-chat.cpp`): ~60 hard-coded template ids selectable via `--chat-template` (e.g. `chatml`, `llama2`/`llama3`/`llama4`, `gemma`, `deepseek`/`deepseek2`/`deepseek3`, `mistral-v1/v3/v7(+tekken)`, `phi3`/`phi4`, `command-r`, `gpt-oss`, `granite`, `qwen`-family via chatml, `vicuna`, `zephyr`, `rwkv-world`, …). These are the no-Jinja fallback.
- **Jinja engine** (`common/chat.cpp`, on by default `--jinja`): runs the model's own GGUF Jinja template through an in-tree Jinja runtime (`common/jinja/*`), supporting tools, `add_generation_prompt`, `chat_template_kwargs`, and thinking. `common_chat_templates_apply()` returns a `common_chat_params` with prompt, grammar (+ lazy triggers), preserved tokens, and the parser to use.

**Tool / function calling** (`common/chat.h`, `docs/function-calling.md`): OpenAI-style function calling for ~any model, enabled with `--jinja`. At this snapshot output parsing is **PEG-parser-based**: `common_chat_format` ∈ `{CONTENT_ONLY, PEG_SIMPLE, PEG_NATIVE, PEG_GEMMA4}`, with an `autoparser` detecting model-specific shapes. The native parser recognizes the documented family formats — **Llama 3.x** (incl. builtin `wolfram_alpha`/`brave_search`/`code_interpreter`), **Functionary v3.1/3.2**, **Hermes 2/3 & Qwen 2.5(-Coder)**, **Mistral Nemo**, **FireFunction v2**, **Command R7B**, **DeepSeek R1** — falling back to a **Generic** handler for unrecognized templates (more tokens, less efficient). Parsed result is a `common_chat_msg` with `content`, `tool_calls[]` (name/arguments/id), and `reasoning_content`. `parallel_tool_calls` is opt-in and gated by template capabilities. Tool-call argument grammars are constrained via lazy grammar triggers (ties into `07-tokenization-and-sampling.md`).

**Reasoning / thinking** (`--reasoning-format`, server `reasoning_format`): `none` (leave `<think>` in content), `deepseek` (extract to `message.reasoning_content`), `deepseek-legacy` (both), `auto` (detect from template). `--reasoning-budget` caps thinking tokens (`-1` unrestricted, `0` immediate end, N=budget), with `--reasoning-budget-message` injected at exhaustion; `/v1/chat/completions/control action=reasoning_end` cuts it off live. `--prefill-assistant` continues a trailing assistant message (Claude-style prefill).

─────────────────────────────────────────────

## 5. Embeddings & reranking

`examples/embedding` (`llama-embedding`) emits a vector per input. Pooling (`--pooling {none,mean,cls,last,rank}`) selects how token states collapse to a sentence embedding; `--embedding`/`--embeddings` restricts the model to embedding-only mode. Normalization `--embd-normalize N`: `-1` none, `0` max-abs-int16, `1` taxicab/L1, `2` euclidean/L2 (default), `>2` p-norm (formulas in `examples/embedding/README.md`). Output formats: raw, `array`, `json` (OAI style), `json+` (adds cosine-similarity matrix). With `--pooling none` the server `/embeddings` returns **per-token unnormalized** vectors.

`examples/retrieval` (`llama-retrieval`) is a reference RAG pipeline: chunk files (`--context-file`, `--chunk-size`, `--chunk-separator`), embed each chunk, then for each query return top-k chunks by cosine similarity. **Reranking** (`--reranking` + reranker model + `--pooling rank`, e.g. `bge-reranker-v2-m3`) scores query↔document pairs directly via `/rerank` rather than cosine over independent embeddings.

─────────────────────────────────────────────

## 6. Evaluation & benchmark tooling

### 6.1 `llama-bench` — the harness rusty_llama is measured against

`tools/llama-bench` runs three test kinds (`tools/llama-bench/README.md`):
- **pp (prompt processing)**: process a prompt of `-p N` tokens in batches → prefill throughput.
- **tg (text generation)**: autoregressively generate `-n N` tokens → decode throughput.
- **pg (combined)**: `-pg pp,tg`, prompt then generate.

Methodology, exactly: each test repeats `-r` times (default **5**), reporting **average t/s ± stddev**; a warmup run precedes the timed runs unless `--no-warmup`. **Tokenization and sampling times are explicitly excluded** from the measurement — it times only the `llama_decode` compute. `-d N`/`--n-depth` prefills the KV cache with N tokens first to measure throughput at depth (reported as `pp512 @ d512`). Every test param (`-m`, `-p`, `-n`, `-b`, `-ub`, `-ngl`, `-fa`, `-ctk/-ctv`, `-t`, `-sm`, `-ts`, …) can take comma-separated lists and the tool runs the full cartesian product. Output `-o` ∈ md (default) / csv / json / jsonl / sql; rows carry build_commit, cpu_info, gpu_info, backend, model size/params, and the raw `samples_ns`/`samples_ts` arrays. This is precisely the comparison surface for rusty_llama: matching its `pp512`/`tg128` numbers on TinyLlama-1.1B Q4_K_M means matching the same decode-only, warmup-excluded, sampling-excluded measurement.

### 6.2 `perplexity` — quality / quantization-loss evaluation

`tools/perplexity` computes PPL over a corpus (convention: Wikitext-2 test set) to judge quantization quality loss vs FP16 — lower is better, and values are **not** comparable across models/tokenizers/implementations (`tools/perplexity/README.md`). Richer stats come from recording FP16 logits to a `.kld` file (`--kl-divergence-base`, ~11 GiB L2 / 37 GiB L3) then running the quantized model with `--kl-divergence`: **Kullback-Leibler divergence** (0 = identical distributions), PPL ratio/difference, mean Δp ("correct"-token probability change), Pearson correlation, Δp percentiles, RMS Δp, and "same top-p" agreement. The published L3-8B scoreboard quantifies the bits-quality curve, e.g. **q8_0 KLD 0.00136 / Δ0.0027 PPL**, **q6_K 0.0055**, **q4_K_M 0.031 (0.028 w/ imatrix)**, **q4_0 0.072**, **q2_K 0.445**, **iq1_S 2.21** — i.e. q4_K_M ≈ near-lossless while sub-3-bit degrades sharply. imatrix-guided quants (`WT 10m`) consistently beat their non-imatrix counterparts.

### 6.3 `imatrix`

`tools/imatrix` produces the importance matrix consumed by k-/i-quants to weight per-channel error during quantization. Mechanics and its effect on bits-per-weight quality are detailed in `04-quantization.md`; the perplexity scoreboard above is the empirical payoff.

─────────────────────────────────────────────

## 7. Distributed inference

### 7.1 RPC backend (`ggml-rpc`)

`tools/rpc` + `ggml/include/ggml-rpc.h` expose ggml **devices on remote hosts** over TCP (optionally RDMA/RoCEv2 when `libibverbs` is present). A `rpc-server` advertises its local accelerators (`bin/rpc-server`, all CUDA devices by default, or a single CPU); the main host builds with `-DGGML_RPC=ON` and passes `--rpc host:port,host2:port2` to `llama-cli`/`llama-server`. Weights + KV are split across **local and remote** devices proportional to free memory (override with `--tensor-split`). A local tensor cache (`-c`, `$HOME/.cache/llama.cpp/rpc`) avoids re-transferring large tensors. The header pins `RPC_PROTO_*_VERSION` (4.0.1) and asserts `GGML_OP_COUNT == 97` so protocol and op-set stay in lockstep; `GGML_RPC_MAX_SERVERS = 16`. **Explicitly proof-of-concept, fragile, insecure — never expose on an open network.** It plugs in as a normal ggml backend (`ggml_backend_rpc_reg`, `ggml_backend_rpc_init(endpoint, device)`), see `02-backends-and-dispatch.md`.

### 7.2 Multi-GPU split modes (`docs/multi-gpu.md`)

`--split-mode/-sm`:

| Mode | Mechanism | Use |
|---|---|---|
| `none` | One GPU (`--main-gpu`). | Confine to a single device. |
| `layer` (default) | **Pipeline parallel**: contiguous layer slices per GPU; KV for layer *l* on the GPU owning *l*. | Most compatible; max prefill / batch throughput; tolerant of slow interconnect. |
| `row` | Deprecated row tensor-parallel (dense weights only); poor perf. | Avoid. |
| `tensor` | **EXPERIMENTAL** tensor parallel: splits weights *and* KV via a "meta device"; multiple cross-GPU reductions per layer. | Min latency / fast token-gen; needs fast interconnect (NVLink/P2P); CUDA only in practice. |

`tensor` requires flash-attn on and non-quantized KV (`f16`/`bf16`/`f32`), disables `--fit`, and is unimplemented for many MoE/SSM/RWKV/hybrid archs (hard error). Cross-GPU reductions use **NCCL** (build-time `-DGGML_CUDA_NCCL=ON`, default; RCCL for HIP) and optional CUDA P2P (`GGML_CUDA_P2P=1`, opt-in, can corrupt on some boards). `-ts/--tensor-split` sets per-device proportions; `--fit`/`--fit-target` auto-sizes context to VRAM (not in tensor mode).

─────────────────────────────────────────────

## 8. Training / fine-tuning (WIP)

`ggml/include/ggml-opt.h` is a high-level training layer over ggml: datasets (`ggml_opt_dataset`, shuffled/sharded), loss types (`MEAN`, `SUM`, `CROSS_ENTROPY`, `MEAN_SQUARED_ERROR`), optimizers **AdamW** and **SGD** (`ggml_opt_optimizer_params` with lr/betas/eps/weight-decay), forward/grad/opt graph build types, gradient accumulation (`opt_period`), and convenience drivers `ggml_opt_fit` / `ggml_opt_epoch` with val-split and progress callback. `examples/training` (`llama-finetune`) wires this to full-model finetune: **FP32 models only, limited hardware**, "very much WIP" — Stories-260K and LLaMA-3.2-1B finetune in ~24 GB. CPU training needs a CUDA-free build; CUDA training wants max `-ngl`. Validated by `llama-perplexity` dropping after 2 epochs on the test set (`examples/training/README.md`).

## 9. TTS (brief)

`tools/tts` (`llama-tts`) does text→speech with a two-model pipeline: an LLM that emits audio codes (e.g. OuteTTS-0.2-500M) + a vocoder/voice-decoder (WavTokenizer) run with `-mv`/`--model-vocoder`; `--tts-use-guide-tokens` improves word recall. Can run as two `llama-server` instances (LLM + decoder with `--embeddings --pooling none`) driven by `tts-outetts.py`. The server's qwen3-omni path (`--talker-model` + `--code2wav-model`) backs `/v1/audio/speech`.

─────────────────────────────────────────────

## Relevance to rusty_llama

rusty_llama today is a **CLI inference engine + a bench harness**, Llama-arch only (CPU + wgpu + CUDA/cuBLASLt), with Q4_K/Q6_K/Q8_0 quant and resident-CUDA decode. Everything in §1–§9 is breadth it lacks. Honest gap assessment:

- **`llama-bench` parity is the scoreboard, and it is favorable terrain.** The bench excludes tokenize+sampling and times only decode, with warmup and `-r` averaging. rusty_llama should mirror this exact protocol (pp512 / tg128, warmup, repetitions, depth via prefilled KV) so its "3–4× behind" number is apples-to-apples and the wins from `03-cuda-kernels.md`/`04-quantization.md` show cleanly. **Low effort, high leverage** — pure measurement hygiene.
- **The server is the single largest greenfield, and the highest-value one for a "product".** Slots + continuous batching + prompt-cache reuse (KV-shift) convert a single-stream engine into a multi-user service. This is a large, well-isolated subsystem layered on the batching core (`06-inference-pipeline.md`); a minimal `/v1/chat/completions` + `/v1/completions` + `/health` subset is a reasonable first milestone before slots/parallelism. **High effort.**
- **Structured output / grammar** (`json_schema`→GBNF) is a frequently-requested feature that depends entirely on rusty_llama's sampler/grammar layer (`07-tokenization-and-sampling.md`); cheap to expose once that exists, expensive to build from zero.
- **Chat templates + tool calling**: rusty_llama needs at minimum a chatml/llama3 template path to be usable as a chat backend. Full Jinja + PEG tool/reasoning parsing (§4) is a deep rabbit hole; a handful of built-in C++ templates (like `llama-chat.cpp`) covers the common case at modest effort.
- **LoRA + control vectors** (§3) are *small, self-contained* wins given rusty_llama already loads GGUF and runs Llama matmuls: LoRA is a `scale·B·A·x` add on weight outputs with per-request scaling; control vectors are a per-layer `+= dir`. Both are days, not weeks, and unlock finetune-without-merge workflows. **Low–medium effort, good ROI.**
- **Embeddings / reranking** (§5) reuse the same forward pass with a pooling head + normalization — a cheap mode flag (`--embedding`, `--pooling`) that opens RAG use cases. **Low effort.**
- **Multimodal** (§2) is genuinely expensive: it needs a separate clip/ViT encoder + projector and embedding-splicing into the sequence. Treat as a long-horizon item; the architecture (two GGUFs, embeddings spliced at media markers) is the pattern to copy when the time comes.
- **Distributed** (RPC, tensor/pipeline split) is premature for a single-box engine; rusty_llama's multi-backend design (`02-backends-and-dispatch.md`) is the right place to add multi-GPU `layer` split first (pipeline, KV-follows-layer) before anything tensor-parallel. **Defer.**
- **Speculative decoding** (draft + ngram/prompt-lookup) is a real decode-throughput multiplier that fits rusty_llama's resident-CUDA decode; prompt-lookup ngram variants need no second model and are the cheapest entry point. **Medium effort, real speedup.**
- **Training/TTS**: out of scope for an inference engine; note only that `ggml-opt`'s AdamW/SGD + cross-entropy design is the reference if finetune is ever in scope.
