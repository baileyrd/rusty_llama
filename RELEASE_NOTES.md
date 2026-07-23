# Release Notes

`rusty_llama` has no version tags yet (pre-1.0) — this log tracks merged pull
requests against `main` instead, reverse chronological, one entry per PR.

---

## PR #90 — Page KvCache: grow capacity lazily instead of eagerly reserving seq_len
**2026-07-23** · [#90](https://github.com/baileyrd/rusty_llama/pull/90)

- **Changed:** `KvCache` starts at `min(KV_PAGE, max_seq_len)` capacity (256
  positions) and grows in `KV_PAGE` increments — relaying out each layer's
  block on growth — instead of eagerly allocating the full context window at
  construction. `RunState`'s public API is unchanged; this is purely an
  internal `KvCache` representation change.
- **Fixed:** the resident GPU/CUDA decode paths' one-time prefill-KV upload
  (`backend/cuda.rs`, `backend/gpu.rs`) previously computed their per-layer
  host-buffer offset from the model's fixed `seq_len` — an assumption paging
  breaks outright (a correctness bug, not just a missed optimization, if left
  as-is). Both now read the KV cache's actual current capacity via the new
  `RunState::kv_capacity()`.
- No GPU in this environment: the CUDA/wgpu offset-math fix is
  compile-verified (`cargo build/test/clippy --all-features` clean) and
  reasoned through against each function's existing contract, but not run on
  real hardware — stated plainly rather than claimed as a runtime guarantee.
- Split from #75 (originally filed as one issue conflating this with
  cross-slot pool sharing, which needs a breaking `RunState` constructor
  change and stays held as #88).
- 3 new `KvCache` unit tests (including a data-survives-relayout check); full
  suite 170+22+9+3+2 / 202+22+9+3+2+1 (default / `--all-features`) passed, 0
  failed.

## PR #89 — Render chat via the GGUF's embedded Jinja template (minijinja)
**2026-07-23** · [#89](https://github.com/baileyrd/rusty_llama/pull/89)

- **Added:** `minijinja` as a new (non-optional) dependency — the base build
  goes from three small deps to four, a deliberate trade approved before
  implementation. `ChatRenderer::resolve` now prefers rendering a GGUF's
  actual `tokenizer.chat_template` Jinja source via `minijinja` over
  pattern-matching known families, falling back to the 5 existing hardcoded
  `ChatTemplate` renderers only when a GGUF has no template string at all. A
  template that fails to parse/render is a hard error, never a silent
  fallback to a guessed family.
- `bos_token` is always passed as `""` to the Jinja context (whether a given
  template embeds it isn't statically knowable per-template), so BOS is
  always added via the tokenizer instead — a deliberate simplification,
  documented on `render_jinja`.
- `ChatTemplate` itself is untouched; every pre-existing test passes
  unmodified.
- No real GGUF files or network access to fetch one in this environment, so
  the "byte-identical to a live upstream template" criterion was adjusted to
  a hand-traced, independently-verified Jinja template exercising the same
  rendering mechanism (loops/if/interpolation) — stated as a real scope
  limitation, not a corner cut for convenience.
- 6 new `chat.rs` unit tests; full suite 167+22+9+3+2 / 199+22+9+3+2+1
  passed, 0 failed.

## PR #85 — Support sharded (multi-file) GGUF loading
**2026-07-23** · [#85](https://github.com/baileyrd/rusty_llama/pull/85)

- **Added:** `Gguf::parse_sharded(&[&[u8]])` alongside the existing `Gguf::parse`,
  merging several shards' tensor tables into one view — each `TensorInfo` now
  records which shard it lives in, and `tensor_bytes` resolves against that
  shard's own bytes/`data_offset`.
- **Added:** `Checkpoint::open` auto-discovers split-GGUF siblings via the
  `split.count` metadata key and llama.cpp's `<prefix>-NNNNN-of-MMMMM.gguf`
  filename convention — point at any one shard, the rest are found and mapped
  automatically. `Checkpoint::gguf()` is the new entry point that transparently
  merges when needed; `main.rs`/`server.rs`'s model/LoRA/control-vector loading
  now go through it.
- Metadata for a merged `Gguf` is taken from shard 0 only — shared KV keys
  (architecture, hyperparameters, tokenizer, ...) aren't cross-validated across
  shards, a deliberate scope cut.
- 8 new unit tests (in-memory shard merge + on-disk auto-discovery
  integration tests); full suite 160+22+9+3+2 passed, 0 failed.

## PR #84 — Add DRY (Don't Repeat Yourself) sampler
**2026-07-23** · [#84](https://github.com/baileyrd/rusty_llama/pull/84)

- **Added:** `Dry`, a new `SamplerStage` that penalizes tokens which would
  extend the context into a repeat of something earlier in it, using a
  Z-algorithm over the reversed recent-token window (O(n), not the naive
  O(n²) scan) truncated to the current run since the nearest sequence-breaker
  token. Mirrors `llama_sampler_dry`, independently re-derived rather than
  ported from the C++ source's exact bookkeeping.
- **Added:** wired into `SamplerConfig`/`from_config` right after `Penalties`;
  CLI `--dry-multiplier`/`--dry-base`/`--dry-allowed-length`/
  `--dry-penalty-last-n`/`--dry-sequence-breaker`; server `SamplingParams`
  (numeric knobs only).
- Sequence breakers are scoped to single tokens — a breaker string that
  tokenizes to more than one token is skipped with a warning (CLI) or simply
  unavailable (server JSON API, which doesn't expose breakers yet).
- 9 new unit tests, including a hand-worked Z-array trace; full suite
  152+22+9+3+2 passed, 0 failed.

## PR #83 — Add Q2_K GGUF quant type support
**2026-07-23** · [#83](https://github.com/baileyrd/rusty_llama/pull/83)

- **Added:** `GgmlType::Q2_K` — a 256-element k-quant super-block, the
  simplest of the k-quant scale schemes (each sub-block scale byte directly
  holds a 4-bit scale and 4-bit min, no cross-byte bit-shuffling). Dequant
  only, no SIMD dot kernel; the matmul dispatch's generic fallback handles it.
- Last of the six quant-breadth issues from this session's gap analysis.
- 2 new unit tests; full suite 143+22+9+3+2 passed, 0 failed.

## PR #82 — Add Q3_K GGUF quant type support
**2026-07-23** · [#82](https://github.com/baileyrd/rusty_llama/pull/82)

- **Added:** `GgmlType::Q3_K` — a 256-element k-quant super-block of 3-bit
  weights (a 2-bit plane plus a separate inverted 1-bit high plane) with 16
  signed 6-bit sub-block scales unpacked via a new `q3_k_scales` helper — a
  different, trickier bit-shuffle than `Q4_K`/`Q5_K`'s `get_scale_min_k4`.
  The scale-unpacking logic was unit-tested in isolation against hand-derived
  values before trusting it inside the full dequant loop.
- 3 new unit tests; full suite 141+22+9+3+2 passed, 0 failed.

## PR #81 — Add Q5_K GGUF quant type support
**2026-07-23** · [#81](https://github.com/baileyrd/rusty_llama/pull/81)

- **Added:** `GgmlType::Q5_K` — `Q4_K`'s 256-element super-block and 6-bit
  packed scale/min scheme, plus a 5th ("high") bit per weight packed
  separately in a 32-byte `qh` plane (the same high-bit-plane idea as `Q5_0`,
  superblocked).
- 2 new unit tests; full suite 138+22+9+3+2 passed, 0 failed.

## PR #80 — Add XTC (exclude top choices) sampler
**2026-07-23** · [#80](https://github.com/baileyrd/rusty_llama/pull/80)

- **Added:** `Xtc`, a filter `SamplerStage` that, with probability
  `probability` each step, drops all but the least-probable of the leading
  run of candidates above `threshold`, pushing the sampler off its most
  confident continuations. Mirrors `llama_sampler_xtc`. Always keeps at least
  one candidate.
- **Added:** wired into `SamplerConfig`/`from_config` after `min_p`, before
  `temp`/`dist` (llama.cpp's chain order); CLI
  `--xtc-probability`/`--xtc-threshold`; server `SamplingParams`.
- 4 new unit tests; full suite 136+22+9+3+2 passed, 0 failed.

## PR #79 — Add Mirostat (v1/v2) sampler
**2026-07-23** · [#79](https://github.com/baileyrd/rusty_llama/pull/79)

- **Added:** `Mirostat`/`MirostatV2`, new *selector* sampler stages (they
  terminate the chain like `Dist`/`Greedy`). v1 estimates the local Zipf
  exponent from the top-`m` candidates and solves for a top-k cutoff that
  keeps expected surprise near an adaptive ceiling; v2 truncates directly to
  the prefix under that ceiling. Both nudge the ceiling toward observed
  surprise each step. Mirrors `llama_sampler_mirostat`/`_v2`.
- **Added:** wired as a mutually-exclusive mode in `SamplerConfig`/
  `from_config` (bypasses top_k/top_p/min_p when on, matching upstream); CLI
  `--mirostat`/`--mirostat-lr`/`--mirostat-ent`; server `SamplingParams`.
- The RNG is this crate's own xorshift64, not upstream's mt19937 — reproducible
  per-seed, not bit-identical to llama.cpp's draw stream.
- 6 new unit tests, including a hand-computed `mu`-update derivation (via a
  NaN-clamp edge case in v1's k formula) verified by hand before trusting the
  test; full suite 132+22+9+3+2 passed, 0 failed.

## PR #78 — Add Q5_1 GGUF quant type support
**2026-07-23** · [#78](https://github.com/baileyrd/rusty_llama/pull/78)

- **Added:** `GgmlType::Q5_1` — `Q5_0`'s 5-bit nibble + high-bit-plane
  packing plus an explicit `(delta, min)` pair instead of a `-16` zero point,
  the same relationship `Q4_1` has to `Q4_0`.
- 2 new unit tests; full suite 127+22+9+3+2 passed, 0 failed.

## PR #77 — Add Q5_0 GGUF quant type support
**2026-07-23** · [#77](https://github.com/baileyrd/rusty_llama/pull/77)

- **Added:** `GgmlType::Q5_0` — the legacy 32-element block family (alongside
  `Q4_0`/`Q8_0`), a 5-bit weight (4-bit nibble plus a 1-bit high plane packed
  in a separate `qh` word) with a single f16 scale.
- 2 new unit tests; full suite 125+22+9+3+2 passed, 0 failed.

## PR #76 — Add Q4_1 GGUF quant type support
**2026-07-23** · [#76](https://github.com/baileyrd/rusty_llama/pull/76)

- **Added:** `GgmlType::Q4_1` — `Q4_0`'s 32-element block shape with an
  explicit `(delta, min)` pair instead of an implicit `-8` zero point.
- First of a run of parity-loop PRs closing quant-type and sampler gaps
  against llama.cpp (see `gap-analysis.md` on the `claude/parity-loop-6cg5rm`
  branch for the full assessment); this and the next nine PRs (#77–#85) were
  all produced and merged unattended by that loop.
- 2 new unit tests; full suite 123+22+9+3+2 passed, 0 failed.
