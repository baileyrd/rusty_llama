//! End-to-end tests that exercise the real loader (mmap), forward pass, and
//! generation loop using a synthetic model.

use std::path::PathBuf;

use rusty_llama::dummy::{dummy_tokenizer, synthetic_checkpoint};
use rusty_llama::{forward, generate, Checkpoint, Config, CpuBackend, RunState, Sampler};

/// A small but non-trivial config (grouped-query attention: 4 q heads, 2 kv).
fn test_config() -> Config {
    Config {
        dim: 16,
        hidden_dim: 32,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        vocab_size: 32,
        seq_len: 8,
        shared_weights: true,
        ..Default::default()
    }
}

/// Write a synthetic checkpoint to a temp file and memory-map it back.
fn mapped_checkpoint(config: &Config, tag: &str) -> (Checkpoint, PathBuf) {
    let bytes = synthetic_checkpoint(config);
    let mut path = std::env::temp_dir();
    path.push(format!("rusty_llama_{tag}_{}.bin", std::process::id()));
    std::fs::write(&path, &bytes).unwrap();
    let cp = Checkpoint::open(&path).unwrap();
    (cp, path)
}

#[test]
fn parses_header_roundtrip() {
    let config = test_config();
    let (cp, path) = mapped_checkpoint(&config, "header");
    let model = cp.model().unwrap();
    assert_eq!(model.config, config);
    let _ = std::fs::remove_file(path);
}

#[test]
fn forward_is_finite_and_deterministic() {
    let config = test_config();
    let (cp, path) = mapped_checkpoint(&config, "forward");
    let model = cp.model().unwrap();
    let backend = CpuBackend::new();
    let mut state = RunState::new(&config);

    forward(&model, &mut state, &backend, 1, 0);
    let first = state.logits().to_vec();
    assert_eq!(first.len(), config.vocab_size);
    assert!(first.iter().all(|v| v.is_finite()), "logits must be finite");

    // Same input at the same (fresh) position reproduces the logits exactly.
    let mut state2 = RunState::new(&config);
    forward(&model, &mut state2, &backend, 1, 0);
    assert_eq!(first, state2.logits());

    // A different input token produces different logits.
    let mut state3 = RunState::new(&config);
    forward(&model, &mut state3, &backend, 2, 0);
    assert_ne!(first, state3.logits());

    let _ = std::fs::remove_file(path);
}

#[test]
fn greedy_generation_is_reproducible() {
    let config = test_config();
    let (cp, path) = mapped_checkpoint(&config, "generate");
    let model = cp.model().unwrap();
    let backend = CpuBackend::new();
    let tokenizer = dummy_tokenizer(config.vocab_size);

    let run = || {
        let mut state = RunState::new(&config);
        let mut sampler = Sampler::new(config.vocab_size, 0.0, 0.9, 42); // greedy
        let mut out = Vec::new();
        let n = generate(
            &model,
            &mut state,
            &backend,
            &tokenizer,
            &mut sampler,
            "", // empty prompt -> generate purely from BOS
            6,
            |bytes| out.extend_from_slice(bytes),
        );
        (n, out)
    };

    let (n1, out1) = run();
    let (n2, out2) = run();
    assert_eq!(out1, out2, "greedy decoding must be deterministic");
    assert_eq!(n1, n2);
    assert!(n1 <= config.seq_len);

    let _ = std::fs::remove_file(path);
}

#[test]
fn truncated_checkpoint_errors_cleanly() {
    // A header that promises far more weight data than the file contains
    // should produce an error, not a panic.
    let config = test_config();
    let mut bytes = synthetic_checkpoint(&config);
    bytes.truncate(Config::HEADER_BYTES + 16);
    let mut path = std::env::temp_dir();
    path.push(format!("rusty_llama_trunc_{}.bin", std::process::id()));
    std::fs::write(&path, &bytes).unwrap();

    let cp = Checkpoint::open(&path).unwrap();
    assert!(cp.model().is_err());
    let _ = std::fs::remove_file(path);
}
