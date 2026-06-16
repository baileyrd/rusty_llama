//! End-to-end tests for the GGUF path: parse a synthetic GGUF file, load the
//! weights/tokenizer, and run the forward pass.

use rusty_llama::dummy::synthetic_gguf;
use rusty_llama::quant::dequantize;
use rusty_llama::{forward, generate, Config, CpuBackend, Gguf, Model, RunState, Sampler, Tokenizer};

fn cfg() -> Config {
    Config {
        dim: 16,
        hidden_dim: 32,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2, // grouped-query attention
        vocab_size: 32,
        seq_len: 8,
        shared_weights: true,
    }
}

#[test]
fn parses_metadata_and_tensor_table() {
    let c = cfg();
    let bytes = synthetic_gguf(&c);
    let gguf = Gguf::parse(&bytes).unwrap();

    assert_eq!(gguf.version, 3);
    assert_eq!(gguf.meta_str("general.architecture").unwrap(), "llama");
    assert_eq!(gguf.meta_u64("llama.embedding_length").unwrap(), c.dim as u64);
    assert_eq!(gguf.meta_u64("llama.block_count").unwrap(), c.n_layers as u64);
    assert!((gguf.meta_f32("llama.attention.layer_norm_rms_epsilon").unwrap() - 1e-5).abs() < 1e-9);

    let t = gguf.tensor("token_embd.weight").expect("embedding tensor");
    assert_eq!(t.dims, vec![c.dim as u64, c.vocab_size as u64]);
    assert!(gguf
        .tensor(&format!("blk.{}.ffn_down.weight", c.n_layers - 1))
        .is_some());
    // Shared weights => no separate classifier tensor.
    assert!(gguf.tensor("output.weight").is_none());
}

#[test]
fn tokenizer_loads_from_gguf() {
    let c = cfg();
    let bytes = synthetic_gguf(&c);
    let gguf = Gguf::parse(&bytes).unwrap();
    let tk = Tokenizer::from_gguf(&gguf).unwrap();
    assert_eq!(tk.vocab_size(), c.vocab_size);
    assert_eq!(tk.encode("", true, false), vec![1]); // just BOS
}

#[test]
fn from_gguf_weights_match_dequantized_tensors() {
    let c = cfg();
    let bytes = synthetic_gguf(&c);
    let gguf = Gguf::parse(&bytes).unwrap();
    let model = Model::from_gguf(&gguf).unwrap();

    assert_eq!(model.config.dim, c.dim);
    assert_eq!(model.config.vocab_size, c.vocab_size);
    assert!(model.config.shared_weights);

    // The embedding buffer must equal the raw dequantized tensor.
    let t = gguf.tensor("token_embd.weight").unwrap();
    let expect = dequantize(t.ggml_type, gguf.tensor_bytes(t).unwrap(), t.n_elements()).unwrap();
    assert_eq!(&model.weights.token_embedding_table[..], &expect[..]);

    // Layer-0 query projection must equal the first slice of the concatenated wq.
    let t = gguf.tensor("blk.0.attn_q.weight").unwrap();
    let expect = dequantize(t.ggml_type, gguf.tensor_bytes(t).unwrap(), t.n_elements()).unwrap();
    assert_eq!(&model.weights.wq[..c.dim * c.dim], &expect[..]);

    // Shared classifier aliases the embedding table.
    assert_eq!(&model.weights.wcls[..], &model.weights.token_embedding_table[..]);
}

#[test]
fn gguf_forward_is_finite_deterministic_and_generates() {
    let c = cfg();
    let bytes = synthetic_gguf(&c);
    let gguf = Gguf::parse(&bytes).unwrap();
    let model = Model::from_gguf(&gguf).unwrap();
    let backend = CpuBackend::new();

    let mut s = RunState::new(&model.config);
    forward(&model, &mut s, &backend, 1, 0);
    let logits = s.logits().to_vec();
    assert_eq!(logits.len(), c.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));

    let mut s2 = RunState::new(&model.config);
    forward(&model, &mut s2, &backend, 1, 0);
    assert_eq!(logits, s2.logits());

    // Greedy generation reproduces.
    let tk = Tokenizer::from_gguf(&gguf).unwrap();
    let run = || {
        let mut st = RunState::new(&model.config);
        let mut sm = Sampler::new(c.vocab_size, 0.0, 0.9, 1);
        let mut out = Vec::new();
        generate(&model, &mut st, &backend, &tk, &mut sm, "", 5, |b| {
            out.extend_from_slice(b)
        });
        out
    };
    assert_eq!(run(), run());
}
