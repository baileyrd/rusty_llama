//! End-to-end tests for the GGUF path: parse a synthetic GGUF file, load the
//! weights/tokenizer, and run the forward pass.

use rusty_llama::dummy::{synthetic_gguf, synthetic_gguf_gpt2, synthetic_gguf_typed};
use rusty_llama::quant::dequantize;
use rusty_llama::{
    forward, generate, Config, CpuBackend, GgmlType, Gguf, Model, RopeScaling, RunState, Sampler,
    Tokenizer,
};

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
        ..Default::default()
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
fn gpt2_tokenizer_loads_from_gguf() {
    // A model whose embedded tokenizer is byte-level BPE (gpt2), vocab 256.
    let c = Config {
        dim: 32,
        hidden_dim: 64,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 4,
        vocab_size: 256,
        seq_len: 8,
        shared_weights: true,
        ..Default::default()
    };
    let bytes = synthetic_gguf_gpt2(&c);
    let gguf = Gguf::parse(&bytes).unwrap();
    assert_eq!(gguf.meta_str("tokenizer.ggml.model").unwrap(), "gpt2");

    let tk = Tokenizer::from_gguf(&gguf).unwrap();
    assert_eq!(tk.vocab_size(), 256);

    // Byte-level encode → decode roundtrips arbitrary text.
    let ids = tk.encode("Hi! 7", false, false);
    let mut out = Vec::new();
    for &id in &ids {
        out.extend_from_slice(&tk.decode(0, id));
    }
    assert_eq!(out, b"Hi! 7");

    // The model loads and runs with a gpt2-tokenizer file.
    let model = Model::from_gguf(&gguf).unwrap();
    let backend = CpuBackend::new();
    let mut s = RunState::new(&model.config);
    forward(&model, &mut s, &backend, 5, 0);
    assert!(s.logits().iter().all(|v| v.is_finite()));

    // The full generation loop works with byte-level BPE decode.
    let mut sampler = Sampler::new(c.vocab_size, 0.0, 0.9, 1);
    let mut st = RunState::new(&model.config);
    let n = generate(&model, &mut st, &backend, &tk, &mut sampler, "hi", 4, |_| {});
    assert!(n <= c.seq_len);
}

#[test]
fn from_gguf_reads_rope_base_and_eps() {
    // Distinctive (non-default) values so we know they were read, not defaulted.
    let c = Config {
        rope_freq_base: 500000.0, // Llama-3 style
        rms_eps: 2e-5,
        ..cfg()
    };
    let bytes = synthetic_gguf(&c);
    let gguf = Gguf::parse(&bytes).unwrap();
    let model = Model::from_gguf(&gguf).unwrap();
    assert_eq!(model.config.rope_freq_base, 500000.0);
    assert_eq!(model.config.rms_eps, 2e-5);
}

#[test]
fn from_gguf_accepts_rope_scaling() {
    // A model that used to be rejected (scaling ignored) now loads, and the
    // scaling is read back into the config and folded into the rope table.
    let c = Config {
        rope_freq_base: 500000.0,
        rope_scaling: RopeScaling::Linear { factor: 4.0 },
        ..cfg()
    };
    let bytes = synthetic_gguf(&c);
    let gguf = Gguf::parse(&bytes).unwrap();
    let model = Model::from_gguf(&gguf).unwrap();
    assert_eq!(
        model.config.rope_scaling,
        RopeScaling::Linear { factor: 4.0 }
    );
    // Linear scaling divides every base frequency by the factor.
    let base = Config {
        rope_scaling: RopeScaling::None,
        ..c
    }
    .rope_table();
    for (scaled, b) in model.rope.inv_freq.iter().zip(&base.inv_freq) {
        assert!((scaled - b / 4.0).abs() < 1e-9);
    }

    // The scaled model still runs a finite forward pass.
    let backend = CpuBackend::new();
    let mut s = RunState::new(&model.config);
    forward(&model, &mut s, &backend, 1, 0);
    assert!(s.logits().iter().all(|v| v.is_finite()));
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

    // Per-layer matmul matrices have the expected shapes.
    assert_eq!(model.weights.wq.len(), c.n_layers);
    assert_eq!(model.weights.wq[0].rows(), c.dim);
    assert_eq!(model.weights.wq[0].cols(), c.dim);
    assert_eq!(model.weights.wk[0].rows(), c.kv_dim());
    assert_eq!(model.weights.w2[0].cols(), c.hidden_dim);

    // A dequantized embedding row must equal that row of the raw tensor.
    let t = gguf.tensor("token_embd.weight").unwrap();
    let full = dequantize(t.ggml_type, gguf.tensor_bytes(t).unwrap(), t.n_elements()).unwrap();
    let mut row = vec![0.0; c.dim];
    model.weights.token_embedding_table.dequant_row(3, &mut row);
    assert_eq!(row, full[3 * c.dim..4 * c.dim]);

    // Layer-0 query row 0 must equal blk.0.attn_q row 0.
    let t = gguf.tensor("blk.0.attn_q.weight").unwrap();
    let full = dequantize(t.ggml_type, gguf.tensor_bytes(t).unwrap(), t.n_elements()).unwrap();
    model.weights.wq[0].dequant_row(0, &mut row);
    assert_eq!(row, full[0..c.dim]);

    // Shared classifier aliases the embedding table.
    let mut a = vec![0.0; c.dim];
    let mut b = vec![0.0; c.dim];
    model.weights.wcls.dequant_row(5, &mut a);
    model.weights.token_embedding_table.dequant_row(5, &mut b);
    assert_eq!(a, b);
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

#[test]
fn q8_0_forward_approximates_f32() {
    // Same underlying weights, stored once as F32 and once as Q8_0. Running the
    // quantized weights on the fly should closely track the full-precision
    // result — this exercises the on-the-fly dequant matmul path with a real
    // quant format end-to-end. (dim/hidden must be multiples of 32 for Q8_0.)
    let c = Config {
        dim: 32,
        hidden_dim: 64,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        vocab_size: 64,
        seq_len: 8,
        shared_weights: true,
        ..Default::default()
    };
    let backend = CpuBackend::new();

    let logits_for = |bytes: &[u8]| -> Vec<f32> {
        let gguf = Gguf::parse(bytes).unwrap();
        let model = Model::from_gguf(&gguf).unwrap();
        let mut s = RunState::new(&model.config);
        forward(&model, &mut s, &backend, 7, 0);
        s.logits().to_vec()
    };

    let exact = logits_for(&synthetic_gguf_typed(&c, GgmlType::F32));
    let quant = logits_for(&synthetic_gguf_typed(&c, GgmlType::Q8_0));

    assert!(quant.iter().all(|v| v.is_finite()));
    let max_diff = exact
        .iter()
        .zip(&quant)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 0.05, "Q8_0 logits diverge from f32 by {max_diff}");
}
