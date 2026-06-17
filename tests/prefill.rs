//! Batched prompt prefill must be exactly equivalent to processing the prompt
//! one token at a time. On the CPU backend the batched ops loop the
//! single-token ops in the same order, so the equivalence is bit-for-bit.

use rusty_llama::dummy::synthetic_gguf_typed;
use rusty_llama::{forward, forward_prefill, Config, CpuBackend, GgmlType, Gguf, Model, RunState};

fn cfg() -> Config {
    Config {
        dim: 32,
        hidden_dim: 64,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2, // grouped-query attention
        vocab_size: 64,
        seq_len: 16,
        shared_weights: true,
        ..Default::default()
    }
}

/// Prefill's last-position logits, and the KV cache it leaves behind, must match
/// the sequential path — checked for both F32 and Q8_0 weights.
#[test]
fn prefill_matches_sequential() {
    for ty in [GgmlType::F32, GgmlType::Q8_0] {
        let c = cfg();
        let bytes = synthetic_gguf_typed(&c, ty);
        let gguf = Gguf::parse(&bytes).unwrap();
        let model = Model::from_gguf(&gguf).unwrap();
        let backend = CpuBackend::new();

        let tokens: Vec<usize> = vec![3, 8, 1, 5, 9, 2, 40, 17];

        // Batched prefill.
        let mut sa = RunState::new(&model.config);
        forward_prefill(&model, &mut sa, &backend, &tokens, 0);
        let prefill_last = sa.logits().to_vec();
        assert!(prefill_last.iter().all(|v| v.is_finite()));

        // Sequential, one token at a time.
        let mut sb = RunState::new(&model.config);
        for (pos, &t) in tokens.iter().enumerate() {
            forward(&model, &mut sb, &backend, t, pos);
        }
        assert_eq!(
            prefill_last,
            sb.logits(),
            "{ty:?}: prefill last-position logits must equal the sequential result"
        );

        // KV-cache parity: one more decode step (same token, same position) on
        // each must agree — only possible if both caches hold identical K/V for
        // every prompt position.
        let next = 11usize;
        let pos = tokens.len();
        forward(&model, &mut sa, &backend, next, pos);
        forward(&model, &mut sb, &backend, next, pos);
        assert_eq!(
            sa.logits(),
            sb.logits(),
            "{ty:?}: decode after prefill must match decode after sequential (KV parity)"
        );
    }
}
