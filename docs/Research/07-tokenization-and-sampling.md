# 07. Tokenization & Sampling

**Summary.** This doc covers the front and back of the text path in llama.cpp: the **vocab/tokenizer** layer (`src/llama-vocab.cpp`, `src/unicode.cpp`), the **sampler** layer (`src/llama-sampler.cpp`, `common/sampling.cpp`), and the **grammar / constrained-decoding** engine (`src/llama-grammar.cpp`, `common/json-schema-to-grammar.cpp`, optional llguidance). Three things matter most: (1) six tokenizer families (SPM/BPE/WPM/UGM/RWKV/PLaMo-2) selected from GGUF `tokenizer.ggml.model` + 56 pretokenizer regex variants keyed by `tokenizer.ggml.pre`; (2) a composable **sampler chain** built on a tiny `llama_sampler_i` vtable that mutates a `llama_token_data_array` in place — ~18 samplers vs rusty_llama's four; (3) a stateful **GBNF pushdown-stack** engine that masks the candidate set every step, fed either by hand-written grammars or by JSON-schema→GBNF translation.

Non-goals (covered elsewhere): chat templates and tool-call parsing → `08-capabilities-and-tooling.md`; the decode loop that consumes these tokens/logits → `06-inference-pipeline.md`; GGUF KV plumbing → `05-gguf-and-model-loading.md`.

─────────────────────────────────────────────

## 1. Tokenizers

All vocab state lives behind a PIMPL `llama_vocab` (`src/llama-vocab.h`). Each family has a stateless `llm_tokenizer_*` (built once at load) and a per-call `llm_tokenizer_*_session` that does the actual `tokenize()`.

### 1.1 The six vocab types

`enum llama_vocab_type` (`include/llama.h`):

| Type | Value | Model | Algorithm | Used by |
|------|-------|-------|-----------|---------|
| `NONE`   | 0 | `no_vocab`/`none` | — (embedding-only models) | — |
| `SPM`    | 1 | `llama` | byte-level BPE, **score-ranked** greedy merges, byte fallback | Llama-1/2, TinyLlama, Mistral |
| `BPE`    | 2 | `gpt2`/`gemma4`/`hybriddna`/`whitespace` | GPT-2 byte-level BPE, **rank-ranked** merges + pretokenizer regex | Llama-3, Qwen2/3, GPT-NeoX, Falcon, … |
| `WPM`    | 3 | `bert` | WordPiece longest-match | BERT / embedding models |
| `UGM`    | 4 | `t5` | Unigram (SentencePiece Viterbi) | T5, Gemma's T5 variants |
| `RWKV`   | 5 | `rwkv` | greedy longest-match trie over raw byte tokens | RWKV |
| `PLAMO2` | 6 | `plamo2` | Aho-Corasick + dynamic programming | PLaMo-2 |

The type is chosen in `llama_vocab::impl::load()` by an `if (tokenizer_model == "...")` chain reading the GGUF string key `tokenizer.ggml.model` (`src/llama-vocab.cpp` ~L1912-2060). Unknown model → `throw "unknown tokenizer"`.

### 1.2 Pretokenizer selection (BPE only)

Only `LLAMA_VOCAB_TYPE_BPE` has a pretokenizer. After the type is fixed, `load()` reads the string key `tokenizer.ggml.pre` and maps it through a second giant if-else to one of **56** `enum llama_vocab_pre_type` values (`DEFAULT=0 … MELLUM2=55`, `src/llama-vocab.h`). The mapping also sets per-model flags: `ignore_merges`, `clean_spaces`, `add_bos`, `add_sep`, `escape_whitespaces`, normalizer lowercase, etc. (`src/llama-vocab.cpp` ~L2098-2350). Examples:

| `tokenizer.ggml.pre` | pre_type | side effects |
|----------------------|----------|--------------|
| `llama3`/`llama-bpe`/`falcon3`/`pixtral`/`lfm2` | `LLAMA3` | `ignore_merges`, `add_bos` |
| `gpt-2`/`phi-2`/`jina-*`/`mellum`/`modern-bert` | `GPT2` | — |
| `qwen2`/`deepseek-r1-qwen`/`megrez` | `QWEN2` | `clean_spaces=false` |
| `deepseek-llm` / `deepseek-coder` / `deepseek-v3` | `DEEPSEEK_LLM`/`_CODER`/`DEEPSEEK3_LLM` | `clean_spaces=false` |
| `gpt-4o`/`llama4`/`kanana2` | `GPT4O` | `clean_spaces=false` |
| `tekken` | `TEKKEN` | `clean_spaces=false`, `ignore_merges` |
| `gemma4` | `GEMMA4` | `escape_whitespaces` (uses `<0xXX>` bytes, not byte-encode) |
| `whitespace` | `WHITESPACE` | lowercase off |

If `tokenizer.ggml.pre` is **missing**, llama.cpp falls back to `DEFAULT` and prints a loud `GENERATION QUALITY WILL BE DEGRADED` warning (`src/llama-vocab.cpp` ~L2098) — older "guess by vocab hash" logic is gone; the pre-type must be baked into the GGUF by the converter (`convert_hf_to_gguf.py`).

Each pre_type maps in turn to a *list of PCRE-style regexes* (`regex_exprs`) set in the `llm_tokenizer_bpe` constructor (`src/llama-vocab.cpp` ~L300-555). E.g. LLAMA3 uses the GPT-4 split regex `(?:'[sS]|'[tT]|…)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`. The BPE session pretokenizes text via `unicode_regex_split(text, regex_exprs, byte_encode)` before any merges.

### 1.3 Per-family algorithm notes

- **SPM** (`llm_tokenizer_spm_session`): splits into UTF-8 chars, seeds a max-heap of adjacent bigrams keyed by **token score** (`tok_data.score`), repeatedly merges the highest-scoring pair, then `resegment()`s. Unmatched symbols fall back **byte-by-byte** via `byte_to_token` (`src/llama-vocab.cpp` ~L120-260). Adds a dummy leading space when `add_space_prefix`.
- **BPE** (`llm_tokenizer_bpe_session`): regex-pretokenize → per-word priority queue of bigrams keyed by **merge rank** (`find_bpe_rank`, lower = merge first). `ignore_merges` short-circuits whole words that are already vocab entries. Unknown leftovers are emitted as byte tokens — `byte_encode` true → GPT-2 byte string, false (gemma-4) → `<0xXX>` (~L605-735).
- **WPM** (`llm_tokenizer_wpm_session`): BERT normalize (NFD strip-accents, lowercase, drop control/0xFFFD, split CJK and punctuation into single-char words), prepend phantom space `U+2581`, then **greedy longest match** up to `max_token_len`; unmatched word → `[UNK]` (~L760-905).
- **UGM** (`llm_tokenizer_ugm_session`): SentencePiece Unigram — a **Viterbi** forward pass over a `naive_trie` of tokens scored by log-prob, backtrack for the best path; unknown codepoints get `min_score - 10.0`. Normalization uses the **precompiled charsmap** (an XOR-compressed compact double-array, XCDA) shipped in GGUF (~L935-1130).
- **RWKV** (`llm_tokenizer_rwkv_session`): builds a trie of *unescaped* byte tokens (`llama_unescape_rwkv_token` handles `\xNN`,`\t`,`\n`,`\r`), then greedy longest-match; no match → `[UNK]` (~L1240-1330).
- **PLaMo-2**: Aho-Corasick automaton + DP (per `include/llama.h`); byte fallback via `<0xXX>`.

### 1.4 Special / added tokens & byte fallback

- `token_data { std::string text; float score; llama_token_attr attr; }`. `attr` is a bitfield (`LLAMA_TOKEN_ATTR_*`: NORMAL/UNKNOWN/CONTROL/USER_DEFINED/BYTE/UNUSED + LSTRIP/RSTRIP/SINGLE_WORD/NORMALIZED), so per-token strip/normalize behavior is data-driven (`include/llama.h`).
- **Special-token splitting**: before the family tokenizer runs, `tokenizer_st_partition()` walks `cache_special_tokens` and slices the raw text into a `forward_list<fragment_buffer_variant>` of *special* vs *raw* fragments, so e.g. `<|im_start|>` is matched verbatim and never merged (`src/llama-vocab.cpp` ~L3121, called at ~L3295). `parse_special` gates this.
- **Designated tokens**: BOS/EOS/EOT/EOM/UNK/SEP/PAD/MASK/NL plus the FIM/infill set (`fim_pre/suf/mid/pad/rep/sep`) are resolved at load; `add_bos`/`add_eos`/`add_sep`/`add_space_prefix` flags come from GGUF (`tokenizer.ggml.add_*`). `check_double_bos_eos()` warns on accidental double-BOS.
- **Byte fallback** (`byte_to_token`, ~L3823): SPM/UGM → look up `<0xXX>` then raw 1-byte string; WPM/BPE → `unicode_byte_to_utf8(ch)` (GPT-2 reversible byte→printable-codepoint map); PLaMo-2 → `<0xXX>`. This guarantees any byte sequence is representable.

### 1.5 Role of `unicode.cpp`

`src/unicode.cpp` is the Unicode substrate shared by all families:

| Function | Purpose |
|----------|---------|
| `unicode_len_utf8` / `unicode_cpt_from_utf8` / `unicode_cpt_to_utf8` / `unicode_cpts_from_utf8` | UTF-8 ↔ codepoint conversion |
| `unicode_cpt_flags_from_cpt` | category flags (letter/number/punct/symbol/whitespace/accent/control) — backs `\p{L}`,`\p{N}` etc. |
| `unicode_byte_to_utf8` / `unicode_utf8_to_byte` | GPT-2 byte-level reversible map (the 256-byte ↔ printable scheme) |
| `unicode_tolower`, `unicode_cpts_normalize_nfd` | casing + NFD normalization (WPM/UGM) |
| `unicode_regex_split` | the BPE pretokenizer splitter |

`unicode_regex_split` is the key piece: `std::regex` cannot evaluate `\p{...}` classes, so for the well-known patterns (gpt2/llama3 families) it uses hand-written `unicode_regex_split_custom_*` fast paths over codepoint-flag streams, with a collapsed-class fallback for the rest.

─────────────────────────────────────────────

## 2. Sampler architecture

### 2.1 Data model (`include/llama.h`)

```c
typedef struct llama_token_data       { llama_token id; float logit; float p; } llama_token_data;
typedef struct llama_token_data_array { llama_token_data * data; size_t size;
                                        int64_t selected; bool sorted; } llama_token_data_array;
```

The array is the unit of work: it starts as **all logits** (`size == n_vocab`, `selected == -1`, `sorted == false`). Each sampler may *filter* (`size` shrinks), *rescale* (`logit`), *fill probabilities* (`p`, via softmax), or *select* (`selected` = index, not token id). `sorted` is a cache flag — never assume sorted, always check.

### 2.2 The `llama_sampler_i` vtable

```c
struct llama_sampler_i {
    const char *           (*name)  (...);                       // optional
    void                   (*accept)(..., llama_token token);    // optional: observe chosen token (stateful samplers)
    void                   (*apply) (..., llama_token_data_array * cur_p); // required: transform candidates
    void                   (*reset) (...);                       // optional
    struct llama_sampler * (*clone) (...);                       // optional
    void                   (*free)  (...);                       // optional
    /* [EXPERIMENTAL] backend_{init,accept,apply,set_input} for on-GPU sampling */
};
struct llama_sampler { struct llama_sampler_i * iface; llama_sampler_context_t ctx; };
```

`apply` is the only required hook; `accept` lets stateful samplers (penalties, DRY, grammar, mirostat, adaptive-p) track history. There is now an **[EXPERIMENTAL] backend sampling** path: a sampler can emit ggml ops (e.g. greedy = `ggml_argmax` on `data->logits`) so token selection runs on-device; incompatible with grammar/reasoning-budget, which force CPU sampling (`src/llama-sampler.cpp`).

### 2.3 The chain

`llama_sampler_chain` is itself a `llama_sampler`. `llama_sampler_chain_add` appends (taking ownership); `chain_apply` runs each member's `apply` in order, then a terminal **selector** (`dist`/`greedy`/`mirostat`/`adaptive-p`) sets `selected`. `llama_sampler_sample(smpl, ctx, idx)` is the convenience wrapper: build `cur_p` from `llama_get_logits_ith` (or backend-sampled probs/ids), `apply`, read `data[selected].id`, then `accept` it (`src/llama-sampler.cpp` ~L345-905). Helpers: `softmax_impl` (partial-sorts then normalizes), `top_k_impl` (partial sort to k), `dist` (`std::discrete_distribution` over `p`), `temp_impl` (divide logits; `t<=0` → keep max, rest −∞).

─────────────────────────────────────────────

## 3. Sampler catalog

All `llama_sampler_init_*` are declared in `include/llama.h` and implemented in `src/llama-sampler.cpp`. "Selector" = ends the chain by setting `selected`.

| Sampler | What it does | Key params (defaults from `common/common.h`) |
|---------|--------------|-----------------------------------------------|
| **greedy** *(selector)* | argmax over logits | — |
| **dist** *(selector)* | multinomial draw from softmax probs (`discrete_distribution`) | `seed` |
| **top_k** | keep the `k` highest-logit tokens (partial sort); `k<=0` = noop | `k=40` |
| **top_p** | nucleus: smallest set whose cumulative prob ≥ `p` | `p=0.95`, `min_keep` |
| **min_p** | keep tokens with `p ≥ p · p_max` (relative to top token) | `p=0.05`, `min_keep` |
| **typical** | locally-typical: keep tokens whose surprisal is closest to the entropy | `p=1.0` (off), `min_keep` |
| **temp** | divide logits by `t`; `t<=0` → greedy (max kept, rest −∞) | `t=0.80` |
| **temp_ext** (dynatemp) | entropy-based dynamic temperature in `[t-Δ, t+Δ]` shaped by `exponent` | `t`, `delta=0`, `exponent=1` |
| **top_n_sigma** | keep tokens whose logit ≥ `max_logit − n·σ(logits)` | `n=-1` (off) |
| **xtc** | with prob `p`, drop all but the least-probable of tokens exceeding threshold `t` (diversity) | `p=0`, `t=0.10`, `min_keep`, `seed` |
| **mirostat (v1)** *(selector)* | feedback control to target cross-entropy `tau`; estimates `k` from top-`m` (Zipf) | `n_vocab`, `seed`, `tau=5`, `eta=0.1`, `m=100` |
| **mirostat_v2** *(selector)* | mirostat without the `m` estimation step | `seed`, `tau=5`, `eta=0.1` |
| **penalties** | repeat/freq/presence penalties over the last `n` accepted tokens (ring buffer) | `penalty_last_n=64`, `penalty_repeat=1.0`(off), `penalty_freq=0`, `penalty_present=0` |
| **dry** | DRY: penalize tokens that would extend a repeated n-gram; penalty = `mult · base^(match_len − allowed)` | `n_ctx_train`, `multiplier=0`(off), `base=1.75`, `allowed_length=2`, `penalty_last_n=-1`, `seq_breakers={\n,:,",*}` |
| **logit_bias** | add fixed bias to specific token ids | `n_vocab`, `[{token,bias}]` |
| **grammar** | GBNF mask: set logits of grammar-illegal tokens to −∞ (see §4) | `vocab`, `grammar_str`, `grammar_root` (+ lazy: trigger patterns/tokens) |
| **infill** | FIM: merge same-prefix probs, prefer EOG/EOT, drop low-prob non-EOG (use after top_k+top_p) | `vocab` |
| **adaptive_p** *(selector)* | select tokens near a target probability via an EMA of chosen-token probs | `target=-1`(off), `decay=0.90`, `seed` |

(`common/sampling.cpp` also wraps a **reasoning-budget** sampler that suppresses/forces thinking-block tokens, but it lives in the `common` layer, not the core sampler set.)

### 3.1 Default chain order

Built in `common_sampler_init` (`common/sampling.cpp` ~L300-400) by iterating `common_params_sampling::samplers`. With `mirostat == 0` (the default):

```
[logit_bias if any]
  → PENALTIES → DRY → TOP_N_SIGMA → TOP_K → TYPICAL_P → TOP_P → MIN_P → XTC → TEMPERATURE
  → DIST            (or ADAPTIVE_P if the user explicitly added it)
```

The default `samplers` vector itself is `{PENALTIES, DRY, TOP_N_SIGMA, TOP_K, TYPICAL_P, TOP_P, MIN_P, XTC, TEMPERATURE}` (`common/common.h`). Mirostat replaces the truncation stack: `mirostat==1` → `temp → mirostat`; `mirostat==2` → `temp → mirostat_v2`. `infill` and `adaptive_p` are opt-in.

### 3.2 Grammar application (separate from the chain)

The grammar sampler `grmr` and `rbudget` are **not** in the chain — `common_sampler_sample` applies them around it (`common/sampling.cpp` ~L564-630):
1. `rbudget.apply` first.
2. If `grammar_first`: apply `grmr` *before* the chain (hard constraint up front).
3. Apply the chain → tentative `id`.
4. Otherwise (default): **rejection-test** the chosen `id` against `grmr` (a 1-element array; valid iff its logit isn't −∞). If valid, done.
5. If rejected: **resample** — rebuild `cur_p`, apply `grmr` then the chain. This "grammar last + resample" path avoids paying the full grammar mask on every token while staying correct.

─────────────────────────────────────────────

## 4. Grammar engine (GBNF)

### 4.1 Rule representation (`src/llama-grammar.h`)

A grammar is `llama_grammar_rules = vector<vector<llama_grammar_element>>`. Each `llama_grammar_element { llama_gretype type; uint32_t value; }` where `value` is a codepoint, rule id, or token id depending on `type`:

| `llama_gretype` | meaning |
|-----------------|---------|
| `END` / `ALT` | end of rule / start of an alternate |
| `RULE_REF` | non-terminal reference (`value` = rule id) |
| `CHAR` / `CHAR_NOT` | literal codepoint / negated set `[^…]` |
| `CHAR_RNG_UPPER` / `CHAR_ALT` | range upper bound `[a-z]` / extra alt char `[ab]` |
| `CHAR_ANY` | `.` |
| `TOKEN` / `TOKEN_NOT` | match a specific vocab **token id** / its negation (`<[id]>`, `!<think>`) |

GBNF source syntax (`grammars/README.md`): `rule ::= seq | seq`, char ranges `[a-z]`/`[^\n]`, escapes `\xXX`/`\uXXXX`/`\UXXXXXXXX`, repetition `* + ? {m} {m,n}`, comments `#`, plus **token literals** `<think>` / `<[1000]>` / negated `!<...>`. `root` is the start symbol.

### 4.2 Parse → stacks → masking

- **Parse** (`llama_grammar_parser::parse`): assigns rule ids in `symbol_ids`, desugars repetitions/groups into helper rules (capped by `MAX_REPETITION_THRESHOLD = 2000`), emits the rule vectors. `llama_grammar_init_impl` then runs **left-recursion detection** (`detect_left_recursion`) — unsupported grammars are rejected with an error and `nullptr`.
- **Stacks**: the engine is a set of pushdown stacks (`llama_grammar_stacks = vector<vector<const element*>>`). Init expands the start rule's alternates into initial stacks via `advance_stack` (resolving leading `RULE_REF`s down to character/token positions).
- **Masking** (`llama_grammar_apply_impl`): for every candidate token, its piece is decoded to codepoints (with carried `partial_utf8` for multi-byte boundaries). `llama_grammar_reject_candidates` walks the stacks: a candidate is rejected unless some stack can `match_char`/`match_token` the next codepoint, recursing through `advance_stack` for the remainder. Rejected candidates get `logit = -INFINITY`. So each step the candidate set is pruned to grammar-legal tokens.
- **Accept** (`llama_grammar_accept` / `_accept_str`): advances all stacks by the accepted token's codepoints, updating `partial_utf8`. State is per-grammar, so `clone`/`reset` snapshot/restore the stacks.

### 4.3 Lazy grammars & triggers

`llama_grammar` carries `lazy`, `awaiting_trigger`, `trigger_buffer`, `trigger_tokens`, and `trigger_patterns` (`std::regex`). A lazy grammar buffers output and stays unconstrained until a trigger word/pattern/token matches, then constrains from the first match group onward — used for tool-call/`tool_choice=required` flows (the trigger plumbing is configured in `common/sampling.cpp`; consumers in `08-capabilities-and-tooling.md`).

### 4.4 JSON-schema → GBNF (`common/json-schema-to-grammar.cpp`)

Translates a subset of JSON-Schema to GBNF using `nlohmann::ordered_json` (order-preserving). Core helpers: `build_repetition` (turns `minItems/maxItems`/separators into `?`,`+`,`*`,`{m,n}`) and `build_min_max_int` (encodes integer `minimum/maximum` as digit-range alternations). A `SchemaConverter` resolves `$defs`/`$ref`, emits rules for `object` (`properties`, required-first), `array` (`items`), `enum`, `const`, `oneOf`/`anyOf`, `string` formats (date/time/uuid…), and anchored `pattern`s. Notable defaults/limits (`grammars/README.md`): `additionalProperties` defaults to **false** (faster, fewer hallucinations); unsupported keywords are silently skipped; `not`, `uniqueItems`, `contains`, conditionals, nested/remote `$ref`, `prefixItems` are unsupported; `minimum/maximum` only for `integer`. Exposed via server `json_schema`/`response_format`, CLI `-j/--json`, or `examples/json_schema_to_grammar.py` ahead of time. The schema is **not** injected into the prompt — it only constrains output.

### 4.5 Optional llguidance backend (`docs/llguidance.md`)

Build-time opt-in (`cmake -DLLAMA_LLGUIDANCE=ON`, needs Rust). A grammar string starting with `%llguidance` (or any JSON-schema request) is routed to the [LLGuidance](https://github.com/guidance-ai/llguidance) library instead of the native engine (`common/sampling.cpp` calls `llama_sampler_init_llg`). It accepts a Lark-like CFG and full JSON-Schema, with a **lexer/parser split** that makes it fast: a 128k-vocab llama-3 token mask averages ~50µs (p99 0.5ms). Differences vs native GBNF: `additionalProperties` defaults to **true**, arbitrary whitespace allowed, property order preserved, and unsupported schemas **error** rather than being silently dropped. Native GBNF has no lexer concept, which is why it can be slower on large grammars.

─────────────────────────────────────────────

## Relevance to rusty_llama

rusty_llama's `Sampler` (`src/sampler.rs`) is **four behaviors in one struct**: greedy (argmax), temperature scaling, plain multinomial, and top-p (nucleus), with an xorshift RNG matching llama2.c. Its tokenizer (`src/tokenizer.rs`) implements **SPM** (greedy score merges + byte fallback) and **GPT-2 byte-level BPE**, with verbatim special-token recognition. That maps cleanly onto llama.cpp's SPM + BPE families (the only two that matter for Llama-arch GGUFs), so the *tokenizer* gap is small; the *sampler/grammar* gap is large and is mostly cheap breadth.

- **Sampler chain abstraction is the highest-leverage port.** Replace the monolithic `Sampler::sample` with llama.cpp's `apply/accept/reset` vtable + `token_data_array` (logits → filtered → probs, `sorted` flag). It makes every sampler below a ~10-line drop-in and composes them in any order — no perf cost on the hot path. rusty_llama currently mutates a flat `&mut [f32]` of logits, which already aligns with the `cur_p` contract.
- **min-p, top-k, penalties are near-free and on the roadmap.** `PERFORMANCE.md`/`HANDOFF.md` explicitly list min-p + repetition penalty as high-ROI breadth. top_k = partial-sort to k; min_p = one pass relative to `p_max`; penalties = ring buffer of last `n` accepted tokens with repeat/freq/presence multipliers. All trivial, all big usability wins, zero impact on the 3–4× throughput gap.
- **Cheap extras worth copying:** dynatemp (`temp_ext`), top-n-sigma, XTC, typical-p, logit-bias — each is a small `apply` over `cur_p`. mirostat v1/v2 and DRY are slightly more (stateful via `accept`) but still small.
- **Default chain order matters for output parity:** PENALTIES → DRY → top-n-σ → top-k → typical → top-p → min-p → XTC → temp → dist. If rusty_llama adds these piecemeal, follow this order so results track llama.cpp.
- **Grammars / GBNF + JSON-schema is the big, expensive win.** A pushdown-stack engine plus per-step candidate masking is real work (parser, left-recursion check, UTF-8 partial handling, `O(candidates × stacks)` reject pass). High value for structured output and tool calling, but budget it as a project, not an afternoon. Note llama.cpp's own escape hatch: the lexer/parser-split **llguidance** path is ~50µs/token for a 128k vocab — if grammar perf becomes the bottleneck, that architecture (not naive GBNF) is the target.
- **Byte-fallback parity is already correct — keep it.** BPE must byte-encode via the GPT-2 reversible map (`unicode_byte_to_utf8`); SPM must fall back to `<0xXX>`. rusty_llama does both; the pretokenizer-golden tests (`tests/fixtures/pretokenize_golden.rs`) should keep asserting the llama3/gpt2 regex split matches the model's `tokenizer.ggml.pre`.
- **Out of scope for a Llama-only engine:** WPM/UGM/RWKV/PLaMo-2 tokenizers and the 56-way pretokenizer registry. Don't build them speculatively — they only matter when you add BERT/T5/RWKV/PLaMo architectures (see `05-gguf-and-model-loading.md` for the arch registry).
