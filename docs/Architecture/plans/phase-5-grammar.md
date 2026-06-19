# Phase 5.1 — GBNF grammar-constrained decoding (detailed plan)

**Status: SHIPPED.** (Phase 5.2 flash attention is tracked separately.)

Constrain generation to a [GBNF](https://github.com/ggerganov/llama.cpp/tree/master/grammars)
grammar so the engine emits reliable structured output (JSON, enums, fixed
formats). It hangs entirely off the existing `SamplerStage` chain (Phase 1.1) —
no backend or model changes — and is exposed through the CLI and the server.

─────────────────────────────────────────────────────────────────────────────

## Grammar core (`src/grammar.rs`)

- **Parser** (`Grammar::parse`): recursive-descent GBNF → rules. Each rule is a
  list of alternates; each alternate a sequence of elements (`Char(CharSet)` or
  `Ref(rule)`). The postfix operators `* + ?` and groups `( … )` are **desugared
  into anonymous rules** at parse time, so the matcher only ever sees `Char`/`Ref`.
  Supports string literals, char classes (`[a-z]`, `[^…]`, ranges, `\n \t \xHH`
  escapes), `.` (any byte), rule refs, alternation, `#` comments.
- **Matcher**: a stack-of-cursors NFA (à la llama.cpp's `llama_grammar`). State =
  a set of stacks; each stack's top cursor is the next element. `close` is the
  **ε-closure** — a worklist with a visited-set of whole stacks, so it's a
  terminating fixpoint (a grammar cycle re-derives a seen stack and stops; even
  left-recursion can't loop). `advance(byte)` consumes one byte across all stacks;
  `is_accepting` = some stack is empty (a complete parse).
- Operates on **bytes** (UTF-8 literals → byte sequences; classes → byte ranges)
  — exact for ASCII / JSON grammars. Multi-byte char *ranges* are out of scope.
- `JSON_GRAMMAR`: a bundled GBNF for a JSON object (the `response_format` target).

## Sampler integration

- **`GrammarStage`** (impl `SamplerStage`): `apply` masks (to `-∞`) every
  candidate whose token bytes can't extend the grammar — and the EOG tokens until
  the grammar is in an accepting state, so a constrained run can still stop;
  `accept` advances the parse state on the chosen token; `reset` restarts it.
- Placed at the **front** of the chain (`SamplerChain::prepend`) so the
  truncation/selector stages only ever see grammar-valid candidates.
- `Tokenizer::token_piece(id)` gives a token's context-independent output bytes;
  `Tokenizer::eog_ids()` the EOS/EOT ids. The per-token piece table is vocab-sized
  and shared via `Arc` (a server builds it once and reuses it across requests).

## Surfaces

- **CLI**: `--grammar '<gbnf>'` (inline) or `--grammar-file <path>`.
- **Server**: a `grammar` field (raw GBNF) on `/v1/chat/completions` and
  `/v1/completions`, plus `response_format: {"type":"json_object"}` → `JSON_GRAMMAR`.
  The stage is prepended to each request's sampler in `admit`.

## Validation

- Parser + matcher unit tests: literals/alternation, char classes/ranges, negated
  classes, `* + ?`, rule refs + groups, a nested object grammar, and the bundled
  JSON grammar (accepts `{...}`, rejects malformed/`[1,2]`).
- `GrammarStage` masking test: only valid tokens survive; EOG allowed exactly when
  accepting.
- Real-model (CLI, Qwen2-0.5B): `root ::= "yes" | "no"` → `yes`; `[0-9]+` → digits.
- Server e2e (model-gated): a `grammar` request returns one of its alternatives;
  manual smoke: `response_format: json_object` returns **valid JSON**.

## Limitations / follow-ons

- Byte-level (no multi-byte char ranges). ASCII / JSON grammars are exact.
- The per-step mask checks every candidate against the grammar (`O(vocab ×
  state)`); fine for small grammars, but a token-prefix trie or an allowed-byte
  cache would cut large-vocab cost — a follow-on if it shows up in a profile.
- No grammar repetition bounds (`{m,n}`); only `* + ?`.

─────────────────────────────────────────────────────────────────────────────
