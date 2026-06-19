//! End-to-end tests that exercise the real loader (mmap), forward pass, and
//! generation loop using a synthetic model.

use std::path::PathBuf;

use rusty_llama::dummy::{dummy_tokenizer, synthetic_checkpoint};
use rusty_llama::{
    forward, forward_embed, generate, AdapterBackend, Backend, Batch, Checkpoint, Config,
    ControlVector, CpuBackend, Pooling, RunState, SamplerChain,
};

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
        let mut sampler = SamplerChain::new(config.vocab_size, 0.0, 0.9, 42); // greedy
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

#[test]
fn embed_mean_equals_mean_of_prefix_last() {
    // Mean-pool over the full sequence == average of the per-prefix Last-pool,
    // because causal attention makes position k's hidden depend only on 0..=k.
    let config = test_config();
    let (cp, path) = mapped_checkpoint(&config, "embed_mean");
    let model = cp.model().unwrap();
    let backend = CpuBackend::new();
    let dim = config.dim;
    let tokens = [1usize, 5, 3, 7];

    let mut s = RunState::new(&config);
    let full = forward_embed(&model, &mut s, &backend, &tokens, Pooling::Mean, false);
    assert_eq!(full.len(), dim);

    let mut manual = vec![0.0f32; dim];
    for k in 1..=tokens.len() {
        let mut sk = RunState::new(&config);
        let last = forward_embed(&model, &mut sk, &backend, &tokens[..k], Pooling::Last, false);
        for (m, &v) in manual.iter_mut().zip(&last) {
            *m += v;
        }
    }
    for m in manual.iter_mut() {
        *m /= tokens.len() as f32;
    }
    for (a, b) in full.iter().zip(&manual) {
        assert!((a - b).abs() < 1e-4, "mean-pool mismatch: {a} vs {b}");
    }
    let _ = std::fs::remove_file(path);
}

#[test]
fn embed_l2_normalizes() {
    let config = test_config();
    let (cp, path) = mapped_checkpoint(&config, "embed_l2");
    let model = cp.model().unwrap();
    let backend = CpuBackend::new();
    let tokens = [1usize, 2, 3];

    let mut s = RunState::new(&config);
    let normed = forward_embed(&model, &mut s, &backend, &tokens, Pooling::Mean, true);
    let norm: f32 = normed.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "L2 embedding must be unit-norm, got {norm}");

    let mut s2 = RunState::new(&config);
    let raw = forward_embed(&model, &mut s2, &backend, &tokens, Pooling::Mean, false);
    assert_ne!(raw, normed, "normalize must change the vector");
    let _ = std::fs::remove_file(path);
}

#[test]
fn embed_pooling_variants_differ() {
    let config = test_config();
    let (cp, path) = mapped_checkpoint(&config, "embed_pool");
    let model = cp.model().unwrap();
    let backend = CpuBackend::new();
    let tokens = [1usize, 4, 2, 6];
    let pool = |p| {
        let mut s = RunState::new(&config);
        forward_embed(&model, &mut s, &backend, &tokens, p, false)
    };
    let mean = pool(Pooling::Mean);
    let last = pool(Pooling::Last);
    let cls = pool(Pooling::Cls);
    assert_ne!(mean, last);
    assert_ne!(mean, cls);
    assert_ne!(last, cls);
    let _ = std::fs::remove_file(path);
}

#[test]
fn control_vector_shifts_logits() {
    let config = test_config();
    let (cp, path) = mapped_checkpoint(&config, "ctrlvec");
    let model = cp.model().unwrap();
    let cpu = CpuBackend::new();
    let dim = config.dim;

    // Baseline: one plain decode step.
    let mut s0 = RunState::new(&config);
    forward(&model, &mut s0, &cpu, 1, 0);
    let plain = s0.logits().to_vec();

    // A control vector steering layer 1 must change the output.
    let mut dirs = vec![None; config.n_layers];
    dirs[1] = Some(vec![0.5f32; dim]);
    let cv = ControlVector::from_dirs(dirs, 1.0);
    let ab = AdapterBackend::new(&cpu, None, Some(&cv));
    let mut s1 = RunState::new(&config);
    ab.forward_step(&model, &mut s1, 1, 0);
    assert_ne!(plain, s1.logits().to_vec(), "control vector must change logits");

    // scale = 0 ⇒ byte-identical to the baseline.
    let mut dirs0 = vec![None; config.n_layers];
    dirs0[1] = Some(vec![0.5f32; dim]);
    let cv0 = ControlVector::from_dirs(dirs0, 0.0);
    let ab0 = AdapterBackend::new(&cpu, None, Some(&cv0));
    let mut s2 = RunState::new(&config);
    ab0.forward_step(&model, &mut s2, 1, 0);
    assert_eq!(plain, s2.logits(), "scale 0 ⇒ no-op");

    let _ = std::fs::remove_file(path);
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

#[test]
fn batched_decode_matches_independent_sequences() {
    let config = test_config();
    let (cp, path) = mapped_checkpoint(&config, "batch");
    let model = cp.model().unwrap();
    let backend = CpuBackend::new();
    let prompts: [&[usize]; 3] = [&[1, 5, 3], &[2, 7], &[4, 6, 2, 1]];
    let gen = 3usize;

    // Reference: each sequence generated independently, greedily.
    let reference: Vec<Vec<usize>> = prompts
        .iter()
        .map(|prompt| {
            let mut state = RunState::new(&config);
            backend.forward_prefill(&model, &mut state, prompt, 0);
            let mut tok = argmax(state.logits());
            let mut toks = Vec::new();
            for step in 0..gen {
                toks.push(tok);
                forward(&model, &mut state, &backend, tok, prompt.len() + step);
                tok = argmax(state.logits());
            }
            toks
        })
        .collect();

    // Batched: prefill each prompt into its slot, then decode all together.
    let mut batch = Batch::new(&config, prompts.len());
    let mut states: Vec<RunState> = (0..prompts.len()).map(|_| RunState::new(&config)).collect();
    let mut cur = Vec::new();
    let mut positions = Vec::new();
    for (s, prompt) in prompts.iter().enumerate() {
        backend.forward_prefill(&model, &mut states[s], prompt, 0);
        cur.push(argmax(states[s].logits()));
        positions.push(prompt.len());
    }
    let slots: Vec<usize> = (0..prompts.len()).collect();
    let mut batched: Vec<Vec<usize>> = vec![Vec::new(); prompts.len()];
    for _ in 0..gen {
        for (s, b) in batched.iter_mut().enumerate() {
            b.push(cur[s]);
        }
        let logits = batch.decode_step(&model, &backend, &mut states, &slots, &cur, &positions);
        for s in 0..prompts.len() {
            cur[s] = argmax(&logits[s * config.vocab_size..(s + 1) * config.vocab_size]);
            positions[s] += 1;
        }
    }

    assert_eq!(batched, reference, "batched decode must match independent runs");
    let _ = std::fs::remove_file(path);
}
