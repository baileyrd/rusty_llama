# 05. GGUF Format & Model Loading

## Summary

GGUF is llama.cpp's single-file, mmap-friendly container: a small magic+header, a typed key→value metadata section, a tensor-info table (name/shape/`ggml_type`/offset), then one aligned tensor-data blob. The container layer (`ggml/src/gguf.cpp`, `ggml/include/gguf.h`) is *model-agnostic* — it knows nothing about transformers. Meaning is layered on top by metadata **conventions** (`general.*`, `<arch>.*`, `tokenizer.ggml.*`) and by `src/llama-model-loader.cpp`, which mmaps the file(s), reconciles split shards into one weight index, validates tensor shapes/types against an arch-specific expectation, and streams data into backend buffers (with async pinned-memory GPU upload). The architecture string in `general.architecture` is resolved through a registry of **132 model architectures** (`src/llama-arch.cpp`) that selects per-arch metadata keys, tensor-name templates, and ultimately the graph builder (see `06-inference-pipeline.md`). The forward path from a HuggingFace checkpoint is `convert_hf_to_gguf.py` + the `gguf-py` writer.

The three things that matter most: (1) the format is deliberately dumb and forward/backward compatible — alignment-padded, offset-checked, version-gated; (2) the loader is where all the real work lives (sharding, buft selection, validation, async upload); (3) the arch registry is a data-driven name/op/layer table, but the actual tensor *set* and wiring per arch is imperative in `llama-model.cpp`.

─────────────────────────────────────────────────

## 1. GGUF binary format (field-by-field)

Layout, per the header doc in `ggml/include/gguf.h` and the parser in `ggml/src/gguf.cpp::gguf_init_from_reader`:

```
┌─ magic        char[4]   = "GGUF"            (GGUF_MAGIC)
├─ version      uint32_t  = 3                 (GGUF_VERSION; v1 rejected, v2 still read)
├─ n_tensors    int64_t                       (ggml tensor count)
├─ n_kv         int64_t                       (metadata KV count)
├─ KV section   [n_kv]   key/typed-value pairs
├─ tensor info  [n_tensors] name/ndims/dims/type/offset
├─ padding      → aligned up to general.alignment (default 32)
└─ tensor data  one contiguous blob, each tensor aligned
```

**Magic & version.** `gguf.cpp` reads 4 bytes and compares against `"GGUF"`. Version is `uint32_t`; the parser rejects `version == 0`, rejects v1 (`"GGUFv1 is no longer supported"`), rejects `version > GGUF_VERSION (3)`, and detects endianness mismatch via the `(version & 0x0000FFFF) == 0` trick (a byte-swapped v3 reads as `0x03000000`).

**Header counts.** `n_tensors` and `n_kv` are signed `int64_t`, bounds-checked against `SIZE_MAX/sizeof(...)` to prevent allocation overflow.

**KV metadata section.** For each pair: a string key, then a `gguf_type` tag (`int32_t`). If the tag is `GGUF_TYPE_ARRAY (9)`, the element type tag and a `uint64_t` element count follow, then the elements; otherwise the scalar value follows directly. The 13 value types (`enum gguf_type`):

| id | type | id | type |
|----|------|----|------|
| 0 | UINT8 | 7 | BOOL (stored as int8) |
| 1 | INT8 | 8 | STRING (u64 len + bytes, no NUL) |
| 2 | UINT16 | 9 | ARRAY (typed, nestable but not array-of-array) |
| 3 | INT16 | 10 | UINT64 |
| 4 | UINT32 | 11 | INT64 |
| 5 | INT32 | 12 | FLOAT64 |
| 6 | FLOAT32 | — | COUNT = 13 (`static_assert`) |

Strings are length-prefixed (`uint64_t`) UTF-8 without a NUL terminator; bools serialize as `int8_t`; all enums as `int32_t`. Duplicate keys are rejected. Strings are capped at `GGUF_MAX_STRING_LENGTH` (1 GiB) and arrays at `GGUF_MAX_ARRAY_ELEMENTS` (1 G) to bound reads. Arrays of arrays are *not* representable (`GGUF_TYPE_ARRAY` as element type is an error).

**Tensor-info section.** Per tensor (`gguf_tensor_info { ggml_tensor t; uint64_t offset; }`): name (string, `< GGML_MAX_NAME`, must be unique), `n_dims` (`uint32_t`, `≤ GGML_MAX_DIMS = 4`), then `n_dims` dimension sizes (`int64_t` each, the remaining `ne[]` default to 1), the `ggml_type` (`int32_t`, range-checked against `GGML_TYPE_COUNT`), and a `uint64_t` `offset` into the data blob. The parser computes strides `nb[]` from type size and block size, and verifies `ne[0] % blck_size == 0` (row length must be a whole number of quant blocks — quant block layouts are in `04-quantization.md`).

**Alignment & data offset.** If `general.alignment` (uint32, must be a power of two) is present it overrides `GGUF_DEFAULT_ALIGNMENT = 32`. After the tensor-info table the reader seeks to `GGML_PAD(pos, alignment)`; that becomes `ctx->offset` (`gguf_get_data_offset`). The data section size is accumulated as `Σ GGML_PAD(ggml_nbytes(t), alignment)`, and crucially **each tensor's stored `offset` must equal the running cumulative size** — i.e. tensors are tightly packed in declaration order with per-tensor alignment padding. Any mismatch aborts the load. This makes the blob a single contiguous, alignment-respecting region directly mmappable: a tensor's bytes live at `data_offset + info.offset`.

**Single-pass vs no_alloc.** `gguf_init_params { bool no_alloc; ggml_context** ctx; }`. With `no_alloc=true` (what the model loader uses) only the metadata + empty tensor descriptors are built and the blob is *not* read — the loader later points tensors at the mmap. With `no_alloc=false`, `gguf.cpp` allocates one `GGML_TYPE_I8` blob tensor and points every tensor's `data` into it. There is also a streaming `gguf_init_from_callback` (chunked reads with `max_chunk_read`) for non-FILE sources. Writing is symmetric (`gguf_write_to_file`, or meta-then-data two-phase via `gguf_get_meta_size`/`gguf_get_meta_data`).

─────────────────────────────────────────────────

## 2. Metadata conventions

Keys are namespaced dot-strings. `src/llama-arch.cpp` (`LLM_KV_NAMES`) and `gguf-py/gguf/constants.py` (`class Keys`) are the two synchronized sources of truth. `<arch>` is substituted from `general.architecture` (e.g. `llama.block_count`).

**General / authorship:** `general.architecture` (string, selects the registry entry), `general.name`, `general.type`, `general.quantization_version`, `general.file_type` (the `llama_ftype` enum), `general.alignment`, plus authorship/licensing/source keys (`general.author`, `general.license`, `general.source.huggingface.repository`, base-model and dataset arrays, `general.tags`, …). Recommended sampler defaults can be embedded under `general.sampling.*` (top_k/top_p/temp/mirostat/…).

**Per-arch hparams (`<arch>.*`):** `block_count`, `context_length`, `embedding_length`, `feed_forward_length`, `vocab_size`, `attention.head_count`, `attention.head_count_kv` (GQA), `attention.key_length` / `value_length`, `attention.layer_norm_rms_epsilon`, `attention.sliding_window`, `attention.scale`, `rope.dimension_count`, `rope.freq_base`, `rope.scaling.{type,factor,original_context_length,yarn_*}`, MoE keys (`expert_count`, `expert_used_count`, `expert_shared_count`, `expert_weights_scale`, `expert_gating_func`), SSM keys (`ssm.{conv_kernel,inner_size,state_size,time_step_rank,group_count}`), and many arch-specific extras (deepstack, altup, nextn, shortconv, wkv/time-mix). The breadth of `LLM_KV` (hundreds of keys) is the cost of supporting 132 architectures in one loader.

**Tokenizer (`tokenizer.ggml.*`):** `model` (`llama`/`gpt2`/`bert`/`rwkv`/…), `pre` (pre-tokenizer regex identity), `tokens` (string array), `token_type`, `scores`, `merges` (BPE), `bos_token_id`/`eos_token_id`/`eot_token_id`/`unknown_token_id`/`padding_token_id`/`mask_token_id`, `add_bos_token`/`add_eos_token`/`add_space_prefix`, FIM/infill ids, `precompiled_charsmap` (Unigram), and `tokenizer.chat_template`. (Tokenizer internals → `07-tokenization-and-sampling.md`.)

**Split/shard:** `split.no`, `split.count`, `split.tensors.count` (see §3). LoRA/control-vector adapters use `adapter.*` keys in their own GGUFs.

─────────────────────────────────────────────────

## 3. The loader: `llama-model-loader.cpp` + `llama-mmap.cpp`

`llama_model_loader` is constructed from a path (or `FILE*`, or pre-built `gguf_context`). It uses `gguf_init_from_file` with `no_alloc=true`, so only metadata + tensor descriptors are parsed; tensor bytes stay on disk until `load_all_data`.

**Weight index.** For every tensor in every context it builds a `weights_map : name → llama_tensor_weight{ uint16_t idx (file index), size_t offs, ggml_tensor* }`. `offs = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, i)` and is **bounds-checked against the file size** at construction (`"data is not within the file bounds, model is corrupted or incomplete"`). Duplicate tensor names across the whole model are rejected. `n_elements`/`n_bytes` accumulate for the BPW report (`print_info` logs `n_bytes*8/n_elements`).

**Sharded / split files.** Naming scheme is `<prefix>-NNNNN-of-NNNNN.gguf` (the `gguf-py` writer constant `SHARD_NAME_FORMAT = "{:s}-{:05d}-of-{:05d}.gguf"`). The loader reads `split.count` from the main file; if `> 1` it asserts the main file is split `0`, derives the sibling paths via `llama_split_prefix`/`llama_split_path` (or accepts a user-supplied list), opens each, **verifies each shard's `split.no` matches its index**, and merges all tensors into the single `weights_map`. Finally it cross-checks the total against `split.tensors.count`. Each `llama_tensor_weight` remembers its source `idx`, so loads route to the right file/mapping.

**mmap (`llama-mmap.cpp`).** `llama_mmap` wraps POSIX `mmap(PROT_READ, MAP_SHARED)` and Win32 `CreateFileMappingA`/`MapViewOfFile`. On Linux it hints `posix_fadvise(SEQUENTIAL)`, optionally `MAP_POPULATE` + `posix_madvise(WILLNEED)` for prefetch, and `POSIX_MADV_RANDOM` under NUMA. The mapping tracks `mapped_fragments` so the loader can `unmap_fragment` (page-aligned `munmap`) the metadata and any GPU-offloaded ranges after upload, releasing RAM for tensors that no longer need a host copy. `llama_mlock` (POSIX `mlock` / Win32 `VirtualLock`) optionally pins resident pages, growing the lock incrementally as tensors are touched. There is also a Linux `O_DIRECT` unbuffered-read path (`use_direct_io`) that is mutually exclusive with mmap.

**Tensor validation & buft selection.** `create_tensor(...)` resolves the canonical name via `LLM_TN`/`LLM_TN_IMPL::str()` (§4), then `check_tensor_dims` compares the GGUF tensor's `ne[]` against the *expected* shape the model code passes in — wrong shape throws `"tensor '...' has wrong shape; expected ... got ..."`, missing-but-required throws, missing-but-optional returns null. Each tensor's target buffer type is chosen by probing whether the op it participates in (`MUL_MAT`, `MUL_MAT_ID`, `ROPE`, `SSM_SCAN`, `GET_ROWS`, …) is supported on a device buffer type (`weight_buft_supported` builds a dummy op graph and calls `ggml_backend_dev_supports_op`); `--override-tensor` regex patterns can force a tensor onto a specific buft. When mmap is on, host/pinned buffer types are avoided so the tensor can be aliased directly into the mapping.

**Lazy/streamed load.** `init_mappings` creates the per-file mmaps (and computes `size_data` for progress). `load_all_data` walks the destination context's tensors and, per tensor:
- **mmap + CPU buffer:** `ggml_backend_tensor_alloc(buf, cur, mapping->addr()+offs)` — zero-copy alias into the page cache; optionally `check_tensors` validates row data asynchronously (`std::async` + `ggml_validate_row_data`).
- **No mmap, host buffer:** `file->seek(offs); read_raw(...)` straight into the tensor.
- **No mmap, device buffer with async-capable backend:** a 4×(64 MiB) pinned host-buffer ring (`n_buffers`, `buffer_size`) double-buffers `read_raw_unsafe` → `ggml_backend_tensor_set_async` uploads, overlapping disk I/O with PCIe transfer; events synchronize buffer reuse. Falls back to a single staging buffer otherwise.
After the final tensor, GPU-offloaded mmap ranges are `unmap_fragment`-ed and validation futures are joined (any failure → `"found tensors with invalid data"`). `done_getting_tensors` checks the created-vs-expected tensor count (allowing a documented `partial` mode for sibling models sharing one file).

─────────────────────────────────────────────────

## 4. The arch registry: `llama-arch.cpp` / `.h`

Three coupled tables, all keyed by `enum llm_arch`:

- **`LLM_ARCH_NAMES`** — `llm_arch → "string"`. `llm_arch_from_string` does a linear lookup of `general.architecture`; an unknown string yields `LLM_ARCH_UNKNOWN`. The loader runs this in its constructor: `llm_kv = LLM_KV(llm_arch_from_string(arch_name))`.
- **`LLM_KV_NAMES`** — `llm_kv → "format"` with `%s` for the arch (e.g. `"%s.block_count"`). `LLM_KV::operator()` formats the arch name in, so `get_key(LLM_KV_BLOCK_COUNT)` becomes `llama.block_count`.
- **`LLM_TENSOR_NAMES`** — `llm_tensor → "template"` (e.g. `"blk.%d.attn_q"`, `"token_embd"`, `"blk.%d.ffn_gate.%d"`). `LLM_TN_IMPL::str()` formats in the block id (`bid`) and expert id (`xid`) and appends a `.weight`/`.bias` suffix.
- **`LLM_TENSOR_INFOS`** — `llm_tensor → {llm_tensor_layer, ggml_op}` (INPUT/REPEATING/OUTPUT + the op the tensor feeds, e.g. `{LAYER_REPEATING, GGML_OP_MUL_MAT}`). The loader uses `layer` to pick the input/output/per-layer buft list and `op` to probe device support; a `"bias"` suffix flips `MUL_MAT→ADD` / `MUL_MAT_ID→ADD_ID`, and `GGML_OP_NONE` marks unused tensors that are silently skipped.

Note the modern design: `LLM_TENSOR_NAMES` is now a *single flat* `map<llm_tensor, const char*>` (not per-arch). **Which** tensors an arch actually has, and **how** they wire into a graph, is encoded imperatively in `src/llama-model.cpp` (per-arch `create_tensor(...)` calls in `load_tensors`, and a per-arch graph-builder struct selected in the arch→graph mapping) — see `06-inference-pipeline.md`. `llama-arch.cpp` also exposes family predicates used to branch memory/graph behavior: `llm_arch_is_recurrent` (Mamba/Mamba2/RWKV*), `llm_arch_is_hybrid` (Jamba, Falcon-H1, PLaMo2, Granite-hybrid, LFM2, Nemotron-H, Qwen3-Next, Kimi-Linear, Qwen3.5), `llm_arch_is_diffusion` (Dream, LLaDA*, RND1).

**Architecture breadth.** `LLM_ARCH_NAMES` defines **132 distinct model architectures** (verified count; 133 map entries including the `LLM_ARCH_UNKNOWN → "(unknown)"` sentinel, and `LLM_ARCH_CLIP → "clip"` is a quantize-only dummy). Major families:

| Family | Entries (examples) |
|--------|--------------------|
| Llama & kin | `llama`, `llama4`, `deci`, `arcee`, `smollm3`, `mistral3`, `mistral4`, `mellum`, `maincoder`, `eagle3` |
| Qwen | `qwen`, `qwen2`, `qwen2moe`, `qwen2vl`, `qwen3`, `qwen3moe`, `qwen3next`, `qwen3vl(moe)`, `qwen35(moe)` |
| Gemma | `gemma`, `gemma2`, `gemma3`, `gemma3n`, `gemma4(-assistant)`, `gemma-embedding` |
| Phi | `phi2`, `phi3`, `phimoe` |
| DeepSeek | `deepseek`, `deepseek2`, `deepseek2-ocr`, `deepseek32`, `glm-dsa` |
| MoE / Mixtral-style | `mixtral`→`llama`, `dbrx`, `arctic`, `grok`, `olmoe`, `bailingmoe(2)`, `dots1`, `grovemoe`, `gpt-oss`(`OPENAI_MOE`), `minimax-m2` |
| SSM / recurrent | `mamba`, `mamba2`, `rwkv6`, `rwkv6qwen2`, `rwkv7`, `arwkv7` |
| Hybrid attn+SSM | `jamba`, `falcon-h1`, `plamo2`, `granitehybrid`, `lfm2(moe)`, `nemotron_h(_moe)`, `kimi-linear` |
| Granite / Command-R | `granite`, `granitemoe`, `granitehybrid`, `command-r`, `cohere2(moe)` |
| BERT/encoder | `bert`, `modern-bert`, `nomic-bert(-moe)`, `neo-bert`, `jina-bert-v2/v3`, `eurobert`, `t5`, `t5encoder` |
| Other | `falcon`, `mpt`, `gpt2`, `gptj`, `gptneox`, `starcoder(2)`, `bloom`, `stablelm`, `bitnet`, `chatglm`, `glm4(moe)`, `exaone(4/-moe)`, `nemotron`, `hunyuan-*`, `ernie4_5(-moe)`, `chameleon`, `wavtokenizer-dec`, `cogvlm`, `paddleocr`, … |

─────────────────────────────────────────────────

## 5. HF → GGUF conversion

`convert_hf_to_gguf.py` (now a thin CLI) + the `conversion/` package + `gguf-py`:

1. **Dispatch.** `ModelBase.load_hparams` reads `config.json`; `get_model_architecture` takes `config["architectures"][0]` (e.g. `"LlamaForCausalLM"`) and maps it through `TEXT_MODEL_MAP`/`MMPROJ_MODEL_MAP` (in `conversion/__init__.py`) to a conversion module name (e.g. `"llama"`). `get_model_class` then returns the registered subclass. Classes register via `@ModelBase.register("LlamaForCausalLM", ...)` (the HOWTO's step 1) into `ModelBase._model_classes`. Hundreds of HF class names fan into ~dozens of converter modules (`MixtralForCausalLM`, `MistralForCausalLM`, `SmolLM3ForCausalLM`, … all map to `"llama"`).
2. **Indexing.** `index_tensors` reads `model.safetensors(.index.json)` or `pytorch_model.bin`, lazily (`LazyTorchTensor`) so tensors are materialized on demand; `--remote` streams safetensors from HF without full download.
3. **Metadata.** Each subclass overrides `set_gguf_parameters()` to emit `<arch>.*` hparams (via `GGUFWriter.add_block_count`, `add_head_count`, `add_rope_freq_base`, …) and `set_vocab()` to emit `tokenizer.ggml.*` (BPE/SPM/WPM/RWKV paths). `model_arch = gguf.MODEL_ARCH.X` ties the writer to the arch string via `MODEL_ARCH_NAMES`.
4. **Tensor renaming.** `modify_tensors()` + `gguf.TensorNameMap` (`tensor_mapping.py`) translate HF tensor names → canonical GGUF names (e.g. `model.layers.{bid}.self_attn.q_proj.weight` → `blk.{bid}.attn_q.weight`), with `{bid}` block substitution. Names must end in `.weight`/`.bias` (quantize relies on it). Dimensions are emitted in reversed (ggml) order.
5. **Output dtype.** `--outtype {f32,f16,bf16,q8_0,tq1_0,tq2_0,auto}`; `auto` sniffs the first 2-D tensor's dtype. Heavier quantization (Q4_K_M etc.) is a *separate* step done by `llama-quantize` on the produced GGUF (→ `04-quantization.md`).

**`gguf-py` writer (`gguf_writer.py`).** Emits exactly the §1 layout: header (`<I` magic, `I` version, `Q` n_tensors, `Q` n_kv) → KV bytes (`_pack_val`) → tensor-info (name, `I` n_dims, dims in reverse, `I` dtype, `Q` running offset padded by `ggml_pad`) → aligned data. A `WriterState` machine enforces ordering. With `--split-max-tensors`/`--split-max-size` it writes shards (`SHARD_NAME_FORMAT`) and injects `split.no`/`split.count`/`split.tensors.count`. `gguf_reader.py` is the inverse `np.memmap` reader (used by tooling/inspection), supporting versions `[2,3]` and endian detection.

─────────────────────────────────────────────────

## Relevance to rusty_llama

rusty_llama already parses GGUF (`src/gguf.rs`, `src/loader.rs`, `src/model.rs::from_gguf`) and also loads the legacy llama2.c binary checkpoint (`Model::parse` via `Config::read_header`). Honest gap assessment vs the above:

- **Format parsing is solid and close to parity.** `src/gguf.rs` mirrors the spec: magic check, version 2/3 gate, KV count, the full 13-variant `MetaValue` enum, tensor-info (name/dims/`GgmlType`/offset), `general.alignment`-padded `data_offset`, and bounds-checked `tensor_bytes`. It is single-pass and borrows straight from the `memmap2` mapping — already the zero-copy design llama.cpp uses for CPU tensors. Low-effort wins: add the v3 endianness-mismatch detection and the per-tensor `offset == cumulative_size` packing assertion for corruption detection.
- **No sharding.** rusty_llama opens exactly one file. Adding `<name>-NNNNN-of-NNNNN.gguf` support (read `split.count`, derive sibling paths, merge a unified tensor index keyed by source-file index) is a contained, well-specified feature worth porting for large models — moderate effort.
- **No arch registry — Llama-only, name-prefixed.** `from_gguf` reads `general.architecture` purely as a `{arch}.*` key prefix and hardcodes `blk.{i}.*`, `token_embd.weight`, `output.weight`. It works for any Llama-tensor-named model but has no `llm_arch` enum, no per-arch tensor/op/layer table, no graph-builder dispatch. The single most valuable architectural pattern to adopt is llama.cpp's **three-table registry** (`LLM_ARCH_NAMES` / `LLM_TENSOR_NAMES` / `LLM_TENSOR_INFOS`): even a 3–5 arch version (Llama, Qwen2/3, Gemma, Phi) would turn "add a model" from a code edit into a table edit. Note llama.cpp keeps the *graph wiring* imperative per-arch (`llama-model.cpp`), so a registry does not eliminate per-arch graph code (see `06`).
- **It already reads the right hparam/RoPE conventions.** `read_rope_scaling` honors `{arch}.rope.scaling.{type,factor,original_context_length,...}` for linear/llama3/yarn — good alignment with §2. Extending to GQA (`attention.head_count_kv`), sliding-window, and MoE keys is the natural next step as new archs land.
- **No tensor validation / async upload.** rusty_llama doesn't do `check_tensors` row validation or the 4×64 MiB pinned-buffer async GPU upload ring. For its resident-CUDA-decode model, porting the **double-buffered pinned-staging upload** could measurably cut model-load latency to VRAM — medium effort, real payoff. Shape/type validation against expected dims is cheap insurance worth adding.
- **No HF→GGUF conversion (and that's fine).** rusty_llama relies on externally produced GGUFs; there is no `gguf-py` equivalent. This is the correct scope — re-implementing `convert_hf_to_gguf.py`'s 130+ model map is not worth it. The one thing to track: the `tokenizer.ggml.pre` pre-tokenizer identity and `chat_template` keys, since correctness there depends on matching llama.cpp's writer (cross-ref `07-tokenization-and-sampling.md`).
- **mmap maturity.** `Checkpoint` uses plain `Mmap::map`; llama.cpp adds `MADV_SEQUENTIAL/WILLNEED`, NUMA `MADV_RANDOM`, `mlock`, and post-offload `unmap_fragment`. Prefetch hints are a trivial, low-risk latency improvement; the rest matters only at multi-GPU/NUMA scale rusty_llama isn't targeting yet.
