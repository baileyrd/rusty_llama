# Phase plans

Detailed, PR-sized implementation plans for the phases in the master plan
[`../10-evolution-plan.md`](../10-evolution-plan.md). Baseline: `main` @ `e2a7d5d`.

## Detail policy (staged)

Implementation detail is written **just-in-time**: phases whose design is durable
and imminent are detailed now; later phases stay as master-plan sketches and are
deepened once their prerequisites land (detailing them earlier would be
speculative and likely rewritten).

| Phase | Detail level | Doc |
|---|---|---|
| 0 — Foundations & seams | **Detailed** (next up) | [`phase-0-foundations.md`](phase-0-foundations.md) |
| 1 — Cheap breadth | **Detailed** | [`phase-1-breadth.md`](phase-1-breadth.md) |
| 2 — Decode speed lever | **Detailed** | [`phase-2-decode-gemv.md`](phase-2-decode-gemv.md) |
| 3 — Architecture breadth | Sketch | `../10-evolution-plan.md` → deepen when Phase 0.2 lands |
| 4 — Paged KV → server | Sketch | `../10-evolution-plan.md` → deepen when Phase 0.4 + Phase 1 land |
| 5 — Structured output & advanced | Sketch | `../10-evolution-plan.md` → deepen opportunistically |

## Locked strategy (human-approved)

Breadth-first (**D1**) · near-Llama archs first (**D2**) · minimal server first
(**D3**) · defer MMQ-prefill + extra quant formats (**D4**). Non-goal: full
llama.cpp parity (treadmill). See `../10-evolution-plan.md` for the guardrails.

## Shared conventions (every milestone)

- Per-op parity vs `CpuBackend` (the oracle) + e2e coherence; behavior tests, not
  default-string tests.
- `cargo test` + `clippy --all-targets` clean **with and without** `gpu`/`cuda`.
- Feature-branch, PR per milestone; re-bench vs `llama-bench` when perf is touched.
- Optional capability ⇒ cargo feature or runtime flag; base build stays
  dependency-light.
- Speed work carries a kill-criterion: measure on the real model, adopt only on a
  win, park proven-but-unwired code behind tests.

## Cross-cutting contracts these plans share

- **Sampler:** `SamplerStage { apply(&mut TokenDataArray); accept(token); reset }`
  + `SamplerChain` (defined in `phase-0-foundations.md` §0.1, consumed by
  `phase-1-breadth.md` §1.1).
- **Arch seam:** `enum Arch` + `TensorNames` table + `Arch::build_model`
  (`phase-0-foundations.md` §0.2, extended in Phase 3).
- **KV seam:** `KvCache` isolating the flat single-seq layout + a forward-referenced
  flagged `attention_seq` (`phase-0-foundations.md` §0.4, consumed by Phase 2.2
  and Phase 4).

## Open decisions awaiting a human call

Each detailed plan ends with an `## Open questions / decisions` section; the live
ones are summarized in the chat handoff. None block starting Phase 0 — all have a
recommended default.
