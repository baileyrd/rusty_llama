# Phase 1 — Cheap breadth (detailed plan)

**Summary.** Phase 1 turns the engine from "greedy/top-p CLI" into something genuinely
usable: the full near-llama.cpp sampler set, an embeddings mode, LoRA + control-vector
adapters, and built-in chat templates. All four items are user-visible breadth with **no
new perf risk** and no change to the decode hot path. **Dependency:** 1.1 builds directly
on the Phase-0.1 sampler chain (`SamplerStage` / `TokenDataArray`, see
[`phase-0-foundations.md`](phase-0-foundations.md) §0.1); 1.2/1.3 ride the existing
per-op `Backend` ops (`src/backend/mod.rs`); 1.4 is pure host string work. **Definition of
done** (per phase): every sampler has a behavior test (not a default-string test),
embeddings validate against a known pooled vector, LoRA matches a reference merge within
tolerance with `scale=0` a byte-identical no-op, and templates render known message lists
to expected prompts — all clean under `cargo test` + `cargo clippy --all-targets` with and
without `gpu`/`cuda`. Master plan: [`../10-evolution-plan.md`](../10-evolution-plan.md)
§"Phase 1". Source captures: [`07`](../../Research/07-tokenization-and-sampling.md),
[`08`](../../Research/08-capabilities-and-tooling.md).

Cross-phase notes: 1.4 templates + 1.1 samplers are the prerequisites the server
(Phase 4.3) consumes; 1.3 adapters are independent of the Phase-2 decode kernel but must
not break its resident caches (analyzed in §1.3).

─────────────────────────────────────────────────────────────────────────────

## 1.1 Core samplers (on the Phase-0.1 chain)

### Design

Phase 0.1 lands the composable chain; Phase 1.1 fills it with the stages. The contract
(locked with `Phase0Plan`, cite [`phase-0-foundations.md`](phase-0-foundations.md) §0.1,
mirroring llama.cpp's `llama_sampler_i` / `llama_token_data_array`, Research/07 §2.1–2.3):

```rust
pub trait SamplerStage: Send {
    fn apply(&mut self, cur: &mut TokenDataArray);   // required: transform candidates
    fn accept(&mut self, _token: u32) {}             // optional: observe chosen token
    fn reset(&mut self) {}                           // optional
    fn name(&self) -> &'static str;
}
pub struct TokenData      { pub id: u32, pub logit: f32, pub p: f32 }
pub struct TokenDataArray { pub data: Vec<TokenData>, pub size: usize,
                            pub selected: Option<usize>, pub sorted: bool }
```

`size` is the live prefix (in-place truncation; a stage shrinks `size`, never reallocates);
`selected` is an **index into `data`**, not a token id; `sorted` is a descending-by-logit
cache flag — a stage that needs order sorts once and sets it, the next stage reuses it.
Probabilities `p` are only valid after a softmax-filling stage. Each stage below is a tiny
`apply` over `cur`; only penalties carry state and use `accept`.

**The four stages (+2 stretch), with exact semantics (Research/07 §3 catalog):**

| Stage | `apply` semantics | Stateful? | Default (off-value) |
|---|---|---|---|
| `TopK { k }` | partial-sort `data[..size]` desc by logit, truncate `size = min(k, size)`; `k <= 0` ⇒ no-op | no | `k = 40` |
| `MinP { p, min_keep }` | over `data[..size]`: `p_max = max logit`; keep tokens with `prob ≥ p · prob_max` (compare in logit space via `logit ≥ p_max + ln p`), but never below `min_keep` survivors | no | `p = 0.05` (`0` = off) |
| `Penalties { last_n, repeat, freq, present, ring }` | over the `last_n` ring of accepted ids: for each candidate seen `c≥1` times — `logit = logit>0 ? logit/repeat : logit*repeat`, then `logit -= c*freq`, then `logit -= (c>0)?present:0` | **yes** | `repeat = 1.0`(off), `freq = 0`, `present = 0`, `last_n = 64` |
| `TempExt { t, delta, exponent }` *(stretch)* | dynatemp: entropy of `softmax(data)` maps to `t' ∈ [t-δ, t+δ]` shaped by `exponent`, then divide logits by `t'`; `δ = 0` collapses to plain temp | no | `delta = 0`, `exponent = 1` |
| `Typical { p, min_keep }` *(stretch)* | fill probs, compute entropy `H`, sort by `|−ln p_i − H|` asc, keep smallest prefix with cumulative prob ≥ `p` | no | `p = 1.0` (off) |
| `Temp { t }` *(already in 0.1)* | divide logits by `t`; `t ≤ 0` ⇒ keep max, rest `−∞` | no | `t = 0.8` |

**Penalty `accept` hook.** `Penalties` owns a fixed-capacity ring of the last `last_n`
accepted token ids. `SamplerChain::accept(token)` (the chain forwards `accept` to every
stage after the selector sets `selected`, Research/07 §2.3) pushes the chosen id into the
ring; `apply` reads it on the next step. `reset()` clears the ring (new sequence). This is
the only place in 1.1 that touches `accept`; `TopK`/`MinP`/`Temp` leave the default no-op.

**Default chain order (must match llama.cpp for output parity, Research/07 §3.1):**
the canonical order is
`PENALTIES → DRY → TOP_N_SIGMA → TOP_K → TYPICAL_P → TOP_P → MIN_P → XTC → TEMP → DIST`.
Our Phase-1.1 subset preserves the relative order:

```
PENALTIES → TOP_K → [TYPICAL] → TOP_P → MIN_P → [TEMP_EXT|TEMP] → DIST(seed)/GREEDY
```

The Phase-0.1 seed config (`temp → top_p → dist`, or `greedy` when `t==0`) is unchanged
when no new flag is set, so existing CLI invocations stay byte-identical (§Behavior).

### CLI (`src/main.rs`)

`parse_args` (`src/main.rs:153-202`) currently steps `i += 2` per flag (every flag takes a
value), and seeds defaults at `src/main.rs:166-175`; `run_generation`
(`src/main.rs:77-110`) builds the sampler at `src/main.rs:80-85`. Phase 1.1:

- Keep `-t` (temp) and `-p` (top-p) as today; add long flags, each value-taking so the
  `i += 2` loop is untouched: `--top-k`, `--min-p`, `--repeat-penalty`, `--repeat-last-n`,
  `--frequency-penalty`, `--presence-penalty`, and (stretch) `--dynatemp-range`,
  `--dynatemp-exp`, `--typical`.
- Replace the `Sampler::new(...)` wire at `src/main.rs:80-85` with a chain builder
  `SamplerChain::from_cli(&args, vocab_size, seed)` that pushes stages in the canonical
  order, omitting any whose flag is at its off-value. Update `USAGE` (`src/main.rs:22-33`).

### Milestones (PR-sized, each independently testable)

1. **`TopK` + `MinP`** stages + `--top-k`/`--min-p` flags + builder ordering. (smallest,
   stateless, no `accept`.)
2. **`Penalties`** stage with the `last_n` ring + `accept` wiring + three flags.
3. **Chain builder `from_cli`** wired into `run_generation`, `USAGE` updated, default path
   proven byte-identical.
4. *(stretch)* **`TempExt` (dynatemp)** + **`Typical`** + flags.

### Behavior preservation / migration

Clean cutover: the monolithic `Sampler` (`src/sampler.rs:11-17`, `sample` at `:37-51`,
xorshift `random_u32` at `:54-61`, `sample_topp` at `:69-106`, `sample_mult` at `:119-128`)
is **replaced** by the chain in 0.1; 1.1 only adds stages. No new flag ⇒ the seed chain is
exactly `temp → top_p → dist` (or `greedy`), reproducing today's stream for a given seed —
the xorshift RNG and draw order are preserved by 0.1. No compat shim.

### Test plan (assert filtering logic, not defaults)

- `top_k_keeps_k_highest`: feed logits `[3,1,4,1,5]`, `TopK{k:2}.apply` ⇒ `size==2` and the
  surviving ids are `{4,2}` (the `5` and `4` rows). `k=0` ⇒ `size` unchanged.
- `min_p_drops_below_relative_threshold`: a distribution where exactly the top-2 clear
  `p ≥ 0.05·p_max` ⇒ `size==2`, and the dropped ids are the low-prob ones; `min_keep`
  forces at least N survivors even when all fall below.
- `penalty_demotes_repeated_token`: pick logits where token `t` is the argmax; push `t` into
  the ring via `accept`; after `Penalties{repeat:1.3}.apply`, `t`'s logit drops below the
  runner-up (assert the **ordering flip**, not a magnitude). `freq`/`present` similarly:
  assert a twice-seen token is penalized more than a once-seen one.
- `penalty_ring_evicts_past_window`: `last_n=2`, accept three tokens, assert the oldest no
  longer affects `apply`.
- `accept_updates_state_not_others`: assert `TopK`/`MinP` `accept` is a no-op (state-free).
- *(stretch)* `dynatemp_widens_on_high_entropy`: flat distribution ⇒ `t' > t`; peaked ⇒
  `t' < t`.

### Effort: **S–M.** Risks: chain-order drift from llama.cpp (mitigate by pinning the
canonical order in `from_cli` + a comment citing Research/07 §3.1); `min_p` numerical edge
(do the compare in logit space to avoid an extra softmax and `0·∞`); ring off-by-one in the
penalty window.

─────────────────────────────────────────────────────────────────────────────

## 1.2 Embeddings / pooling

### Design

Add an embedding mode that runs the forward stack over the prompt and returns a pooled,
L2-normalized hidden vector **instead of** sampling (Research/08 §5). The per-op prefill
already materializes the full per-position residual: `forward_prefill`
(`src/model.rs:491-588`) keeps a row-major `x` of shape `(n, dim)`
(`src/model.rs:508`, seeded per token at `:516-519`) and only norm+classifies the **last**
row at `:582-587`. Embeddings need the *other* rows, pre-classifier.

New pooling enum + a dedicated forward in `src/model.rs`:

```rust
pub enum Pooling { Mean, Last, Cls }      // --pooling {mean,last,cls}

/// Run the layer stack over `tokens`, final-RMSNorm every row, pool to one
/// `dim`-vector, then L2-normalize. Returns the embedding (len = config.dim).
pub fn forward_embed<B: Backend + ?Sized>(
    model: &Model, state: &mut RunState, backend: &B,
    tokens: &[usize], pooling: Pooling, normalize: bool,
) -> Vec<f32>;
```

`forward_embed` is `forward_prefill`'s body up to the layer loop (so it reuses
`rmsnorm_batch`/`matmul_batch`/`attention_batch`/`rope_batch`/`swiglu`/`add` — every
`Backend` has them, GPU/CUDA override the batched ones), but instead of `:582-587` it does:

1. `rmsnorm_batch(out, x, rms_final_weight, eps, n)` over all `n` rows
   (`Weights::rms_final_weight`, `src/model.rs:31`),
2. pool: `Mean` = column-average over rows; `Last` = row `n-1`; `Cls` = row `0`,
3. L2-normalize (`--embd-normalize` default L2; `−1` = none; matches llama.cpp's
   `embd_normalize` default of 2/euclidean, Research/08 §5).

No classifier matmul runs (`wcls` is skipped) — embeddings live in `n_embd = config.dim`.
The fused resident GPU/CUDA `forward_prefill` overrides (which only retain the last row +
KV) are **not** used here; `forward_embed` calls the per-op batched ops directly, so any
backend works without touching the resident decode path.

### CLI / wiring (`src/main.rs`)

`--embedding` switches `run` (`src/main.rs:56-75`) from `generate` to a new
`run_embedding(model, tokenizer, args)` that encodes the prompt (BOS on, EOS off, like
`generate` at `src/model.rs:606`), calls `forward_embed`, and prints the vector (one float
per line, matching `llama-embedding`'s raw format). `--pooling {mean,last,cls}` (default
`mean`), `--embd-normalize {-1,2}` (default `2`).

### Milestones

1. `Pooling` enum + `forward_embed` (CPU), unit-tested against the per-op `forward` hidden
   states.
2. `--embedding`/`--pooling`/`--embd-normalize` flags + `run_embedding` print path.
3. GPU/CUDA validation (per-op batched ops already parity-tested; just confirm
   `forward_embed` matches CPU under each feature).

### Behavior preservation / migration

Purely additive: without `--embedding` nothing changes. `forward_embed` reuses existing
ops, so no backend trait change. WPM/BERT-specific embedding heads are out of scope
(Llama-arch only) — documented as an open question.

### Test plan

- `embed_matches_manual_pool`: run the per-op `forward` (`src/model.rs:393-477`) for each
  prompt position capturing the post-final-norm hidden state, mean-pool + normalize by hand,
  and assert `forward_embed(Mean)` matches within `1e-4` (oracle = the existing forward).
- `pooling_variants`: on a 3-token prompt, `Last` == row-2 normalized, `Cls` == row-0
  normalized, `Mean` == average; assert they differ (not the same vector).
- `l2_normalized_unit_norm`: `normalize=L2` ⇒ `‖v‖₂ ≈ 1`; `normalize=−1` ⇒ raw (norm ≠ 1).
- `embed_reference_vector`: a tiny synthetic model (via `dummy.rs` / `synthetic_gguf`) with
  fixed weights ⇒ a precomputed reference pooled vector (the "known pooled vector" the
  master-plan acceptance asks for), asserted within tolerance.

### Effort: **S.** Risks: which hidden state to pool (we pool the **post-final-RMSNorm**
state for decoder-LM embedders — note as an open question vs raw last-layer); mean-pool over
BOS/special tokens (llama.cpp includes them; we match — assert in the reference test).

─────────────────────────────────────────────────────────────────────────────

## 1.3 LoRA + control vectors

### Design (new `src/adapter.rs`)

Two runtime-applied adapter types (Research/08 §3):

```rust
/// One LoRA low-rank pair for a target weight: out += scale · B · (A · x).
/// A is (rank, in_features), B is (out_features, rank); both borrowed QMatrix.
pub struct LoraPair<'a> { pub a: QMatrix<'a>, pub b: QMatrix<'a> }

pub struct LoraAdapter<'a> {
    /// keyed by base tensor name, e.g. "blk.7.attn_q.weight"
    pub pairs: HashMap<String, LoraPair<'a>>,
    pub alpha: f32,          // adapter.lora.alpha
    pub user_scale: f32,     // per-request --lora-scaled
}
impl LoraAdapter<'_> { fn scale(&self) -> f32 { self.user_scale * self.alpha / rank } }

/// Per-layer steering direction added to the residual stream: x += scale · dir.
pub struct ControlVector { pub dirs: Vec<Option<Vec<f32>>>, pub scale: f32 } // dirs[layer]
```

**Loading.** A LoRA GGUF has `general.type == "adapter"`, `adapter.type == "lora"`, a
matching `general.architecture` (hard error on mismatch — same discipline as
`Model::from_gguf` reading `general.architecture` at `src/model.rs:137`), and a scalar
`adapter.lora.alpha`. Tensors pair by suffix `.lora_a`/`.lora_b` onto the base name; `A`
must be `(rank, in)` and `B` `(out, rank)`, validated against the base weight's shape
(reject transposed/"finetune" layouts, Research/08 §3). `rank` is `A.rows()`. Control
vectors load a per-layer `n_embd` F32 direction over `--control-vector-layer-range`; layer 0
never carries one (Research/08 §3).

**Where it hooks the forward pass.** LoRA targets the `Backend::matmul` consumers in
`forward` (`src/model.rs:421-471`) and `forward_prefill` (`src/model.rs:529-578`): `wq`,
`wk`, `wv`, `wo`, `w1`, `w2`, `w3`. After each base `backend.matmul(out, x, w)`, if the
adapter has a pair for that tensor:

```text
backend.matmul(scratch_r, x, &pair.a);          // r = A·x         (len = rank)
backend.matmul(scratch_o, scratch_r, &pair.b);  // o = B·(A·x)     (len = out)
// out += scale · o   (scaled residual add)
```

`scratch_r` (rank-sized) + `scratch_o` (out-sized) are new `RunState` buffers
(`src/model.rs:323-334`); rank is tiny (≈8–64) so the two extra GEMVs are cheap. The scaled
add needs either a small `Backend::add_scaled(out, x, s)` (a one-line addition to the trait
next to `add` at `src/backend/mod.rs:99`) or fold `scale` into `B` at load time (simpler,
but then per-request scale forces a re-scale — prefer `add_scaled`).

**Control vector** is even simpler: after a layer's residual updates
(`src/model.rs:458,471`) add `scale · dirs[layer]` to `state.x` via `add_scaled` when
`dirs[layer]` is `Some` and the layer is in range.

### The fold-in decision (runtime add vs pre-merge) — analyzed against the resident caches

This is the crux. The GPU and CUDA backends keep **resident weight caches keyed by the base
weight's source data pointer**:

- CUDA: `weights: Mutex<HashMap<usize, Arc<CudaSlice<f16>>>>` (`src/backend/cuda.rs:301-302`),
  filled by `weight_f16` keyed on `data.as_ptr()` (`src/backend/cuda.rs:358-383`); the
  fused decode `DecodeCuda` (`:248-283`) keeps KV + activations + RMSNorm weights resident.
- wgpu: `weights: Mutex<HashMap<usize, GpuWeight>>` (`src/backend/gpu.rs:1085`), key from
  `data.as_ptr()` in `weight_buffer` (`:1399-1403`); `DecodeState` (`:1020-1051`) +
  per-layer `LayerBinds` (`:963-998`) bake the layer ops into bind groups built **once** in
  `build_decode_state` (`:1550`).

**Option A — pre-merge** `W' = W + scale·B·A` into the weight bytes. Fatal frictions:
1. Base weights are quantized and **borrowed from the mmap** (the "compressed stays
   compressed" invariant, `src/model.rs:16-19`). Merging means dequant → add → (requantize
   *or* keep f32), losing precision or doubling RAM/VRAM per merged tensor.
2. The resident caches are **pointer-keyed**: a merged weight is a new buffer ⇒ a new key ⇒
   the f16/f32 device copies and the baked `LayerBinds`/`DecodeState` bind groups must be
   invalidated and rebuilt.
3. **Per-request scale** (`--lora-scaled FNAME:SCALE`, Research/08 §3) means re-merging
   *all* target weights on every scale change — exactly the expensive full dequant+merge.

**Option B — runtime add** (recommended). Keep base `matmul` untouched (resident caches
stay valid), run the two tiny A/B GEMVs + scaled add. The A/B factors are small extra
weights that flow through the **same** pointer-keyed cache for free; per-request scale is a
scalar in `add_scaled`; precision is exact (no requantize); works uniformly on
CPU/GPU/CUDA via existing ops. Cost: two extra GEMVs per LoRA-targeted weight per token —
negligible at rank ≤ 64.

**Decision: runtime add (Option B).** It preserves the compressed-weight invariant, the
pointer-keyed resident caches, and per-request scaling, at trivial cost.

**One wrinkle — the fused resident decode path.** `CudaBackend::forward_step`
(`src/backend/cuda.rs:713+`) and the wgpu `DecodeState` bake a whole layer; injecting the
LoRA GEMVs there means extending those fused kernels/bind groups. **Clean first cut:** when
an adapter is bound, **disable the fused override and fall back to the per-op `forward`**
(`src/model.rs:393-477`) — which routes through `Backend::matmul` where the LoRA hook lives,
so CPU/GPU/CUDA all get correct LoRA immediately. The model already gates the resident path
behind the trait method, so this is a guard at the `forward_step`/`forward_prefill` call
sites, not a kernel change. Folding LoRA into the fused decode kernels (to recover its
speed under an adapter) is a documented **Phase-2-adjacent follow-up**, not Phase 1.

### Milestones

1. `src/adapter.rs`: LoRA GGUF loader + pair/shape validation + `ControlVector` loader
   (no forward wiring yet) — unit-tested on a synthetic adapter GGUF.
2. `Backend::add_scaled` (one method, default loop; GPU/CUDA cheap override) + `RunState`
   scratch buffers.
3. LoRA hook in `forward`/`forward_prefill` `matmul` consumers; per-op path only; CPU
   reference parity.
4. Control-vector residual add + `--control-vector(-scaled)` + `--control-vector-layer-range`.
5. Fused-path guard: bound adapter ⇒ per-op fallback on GPU/CUDA; parity test under each
   feature. CLI: `--lora`, `--lora-scaled FNAME:SCALE`.

### Behavior preservation / migration

No adapter ⇒ zero overhead and the fused resident path is used exactly as today (the hook
is `if let Some(pair) = adapter`). `scale = 0` must be a true no-op (see test). Additive
trait method (`add_scaled`) with a default impl ⇒ no backend breaks.

### Test plan

- `lora_matches_external_merge`: build a base weight `W` and random `A`,`B`; construct the
  merged dense `W' = W + scale·B·A` as a separate `QMatrix`; assert
  `forward`-with-adapter logits == `forward`-with-`W'` logits within `1e-3` (the externally
  merged reference the acceptance demands).
- `lora_scale_zero_is_noop`: `scale = 0` ⇒ logits **byte-identical** to the no-adapter
  forward (not "within tolerance" — exact).
- `lora_shape_mismatch_rejected`: A/B with wrong `(rank,in)`/`(out,rank)` ⇒ load error.
- `control_vector_shifts_residual`: a known `dir` at layer L ⇒ the residual at L moves by
  `scale·dir`; `scale=0` ⇒ no-op; layer 0 never applied.
- `fused_fallback_parity`: GPU/CUDA with an adapter bound matches the all-CPU adapter path
  (guards the per-op fallback).

### Effort: **M.** Risks: `lora_a`/`lora_b` orientation (transposed adapters — validate and
reject); `_norm.weight` LoRA tensors (llama.cpp skips them, Research/08 §3 — skip + warn);
forgetting the fused-path guard (would silently ignore the adapter on GPU/CUDA — the
`fused_fallback_parity` test catches it).

─────────────────────────────────────────────────────────────────────────────

## 1.4 Chat templates (new `src/chat.rs`)

### Design

A small set of **hardcoded** templates (no Jinja) that render OAI-style messages to a
prompt string, mirroring llama.cpp's built-in C++ templates in `llama-chat.cpp`
(Research/08 §4 — ~60 ids, the no-Jinja fallback). Scope for Phase 1: **chatml, llama-3,
qwen2** (qwen2 reuses chatml, Research/07 §1.2 `QWEN2`/Research/08 §4).

```rust
pub enum Role { System, User, Assistant }
pub struct Message { pub role: Role, pub content: String }

pub enum ChatTemplate { ChatMl, Llama3, Qwen2 }

impl ChatTemplate {
    /// Detect from GGUF `tokenizer.chat_template` markers, else arch fallback.
    pub fn detect(gguf: &Gguf, arch: &str) -> Option<ChatTemplate>;
    /// Render messages → prompt; appends the generation header when `add_gen`.
    pub fn render(&self, msgs: &[Message], add_gen: bool) -> String;
}
```

**Selection.** Read the GGUF string key `tokenizer.chat_template` (via the existing
`Gguf::meta_str`, used at `src/model.rs:137` / `src/tokenizer.rs:57`) and detect by
substring: contains `<|im_start|>` ⇒ ChatMl; `<|start_header_id|>` ⇒ Llama3; arch
`qwen2`/`qwen` ⇒ Qwen2 (chatml body). Fallback to arch when the key is absent. A
`--chat-template {chatml,llama3,qwen2}` flag overrides detection (llama.cpp's
`--chat-template`, Research/08 §4).

**Render formats** (literal, not Jinja):

```text
chatml:  per msg → "<|im_start|>{role}\n{content}<|im_end|>\n"
         add_gen → "<|im_start|>assistant\n"
llama-3: prefix "<|begin_of_text|>"; per msg →
         "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
         add_gen → "<|start_header_id|>assistant<|end_header_id|>\n\n"
qwen2:   identical body to chatml (system/user/assistant via <|im_start|>)
```

The rendered string is fed to `Tokenizer::encode(text, bos, eos)`
(`src/tokenizer.rs:68`); the special tokens (`<|im_start|>`, `<|eot_id|>`, …) are matched
**verbatim** by the existing special-token splitter (`SpecialTokens`,
`src/tokenizer.rs:134-149`, recognized during encode at `:311`/`:555`). For llama-3 the
template emits `<|begin_of_text|>` itself, so `encode` is called with `bos=false` to avoid a
double-BOS; chatml/qwen2 follow the GGUF's `add_bos`. (`check_double_bos` discipline,
Research/07 §1.4.)

### CLI / wiring (`src/main.rs`)

Minimal first cut: a `--chat` flag that treats `-i` as a single user turn (optionally
`--system <text>`), renders it via the detected/overridden template with `add_gen = true`,
and runs `generate` on the rendered prompt. Full multi-turn conversation loop is a Phase-4
(server) concern; Phase 1 only needs the render path to exist and be correct.

### Milestones

1. `src/chat.rs`: `Role`/`Message`/`ChatTemplate` + `render` for the three templates
   (pure, unit-tested in isolation).
2. `detect` from `tokenizer.chat_template` + arch fallback + `--chat-template` override.
3. `--chat`/`--system` wiring through `run_generation`; double-BOS handling for llama-3.

### Behavior preservation / migration

Additive: without `--chat` the plain prompt path is unchanged. No tokenizer change — chat
templates are a host-side string layer above `encode`. Templates are data-driven literals;
do not assert default selection in tests (assert render output).

### Test plan (assert rendered strings)

- `chatml_renders_expected`: `[System("S"), User("U")]`, `add_gen=true` ⇒
  `"<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"`.
- `llama3_renders_expected`: same messages ⇒ the `<|begin_of_text|>` + `<|start_header_id|>`
  / `<|eot_id|>` sequence, with the trailing assistant header when `add_gen`.
- `qwen2_is_chatml_body`: qwen2 render == chatml render for the same messages.
- `detect_from_template_marker`: a GGUF stub whose `tokenizer.chat_template` contains
  `<|start_header_id|>` ⇒ `Llama3`; contains `<|im_start|>` ⇒ `ChatMl`; absent ⇒ arch
  fallback. (Behavior, not a default.)
- `chat_specials_round_trip`: render → `encode` → the `<|im_start|>`/`<|eot_id|>` ids appear
  verbatim (reuses the special-token path, `src/tokenizer.rs:884-925`).

### Effort: **M** (mostly the small detection matrix + double-BOS edge). Risks: per-template
BOS handling (llama-3 self-emits, others rely on `add_bos` — covered by the encode-with
`bos=false` rule); models whose special tokens are spelled differently than the literal
(detect from the GGUF template string, not a hardcoded guess).

─────────────────────────────────────────────────────────────────────────────

## Open questions / decisions

1. **CLI flag style (D-1.1).** Keep rusty_llama's `-t`/`-p` (temp/top-p) and add llama.cpp
   long flags for the rest, or fully adopt llama.cpp's `--temp`/`--top-p`/… and deprecate
   the short flags? *Recommended:* keep short flags, add long flags (no breakage).
2. **Embedding hidden state (D-1.2).** Pool the **post-final-RMSNorm** state (chosen here)
   vs the raw last-layer residual? llama.cpp's choice is pooling-type/model dependent.
   Confirm against a real embedding GGUF if one becomes a target; BERT/WPM embedding heads
   are explicitly out of scope (Llama-arch only).
3. **`add_scaled` vs fold-scale-into-B (D-1.3a).** Add a `Backend::add_scaled` trait method
   (clean per-request scale) vs folding `scale` into `B` at load (no trait change but
   re-scale on every request). *Recommended:* `add_scaled`.
4. **Fused-decode LoRA (D-1.3b).** Phase 1 disables the fused GPU/CUDA path under an adapter
   (per-op fallback). Is recovering fused-decode speed *with* an adapter wanted in Phase 1,
   or deferred to a Phase-2-adjacent follow-up? *Recommended:* defer.
5. **Multiple stacked adapters (D-1.3c).** Support comma-separated `--lora` (llama.cpp
   allows N), or single-adapter for Phase 1? *Recommended:* single adapter first; the
   `pairs` map already generalizes to N.
6. **Chat conversation loop (D-1.4).** Phase 1 ships single-turn `--chat` only; the
   multi-turn loop + assistant prefill lands with the server (Phase 4.3). Confirm that's
   the intended split.

## Definition of done

- **1.1** `TopK`, `MinP`, `Penalties` stages (+ stretch `TempExt`/`Typical`) on the
  Phase-0.1 chain, wired to CLI flags, in the canonical llama.cpp order; penalties hook
  `accept` over an `n_prev` ring; each stage has a behavior test (filtering/ordering, not a
  default); no-flag invocations are byte-identical to the 0.1 seed chain.
- **1.2** `--embedding` + `--pooling {mean,last,cls}` + `--embd-normalize` produce a pooled,
  L2-normalized vector via `forward_embed`; validated against a manual-pool oracle and a
  precomputed reference vector within tolerance.
- **1.3** LoRA + control-vector loading (`src/adapter.rs`), runtime-add fold-in into the
  `matmul` consumers (resident caches preserved), per-request scale; output matches an
  external merge within tolerance and `scale=0` is a byte-identical no-op; fused GPU/CUDA
  path falls back to per-op under an adapter with parity.
- **1.4** `src/chat.rs` renders chatml/llama-3/qwen2 message lists to expected prompt
  strings, selected from `tokenizer.chat_template`/arch with a `--chat-template` override;
  special tokens round-trip through `encode`.
- **All:** `cargo test` + `cargo clippy --all-targets` clean **with and without** `gpu` and
  `cuda`; base build stays dependency-light (no new runtime deps for any 1.x item); each
  item shipped on its own feature-branch + PR (HANDOFF conventions,
  [`../10-evolution-plan.md`](../10-evolution-plan.md) §"Standing invariants").

Sibling plans: [`phase-0-foundations.md`](phase-0-foundations.md) (the sampler chain 1.1
builds on), [`phase-2-decode-gemv.md`](phase-2-decode-gemv.md) (decode kernel — 1.3 must
not break its resident caches).
