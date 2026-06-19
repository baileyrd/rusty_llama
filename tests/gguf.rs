//! End-to-end tests for the GGUF path: parse a synthetic GGUF file, load the
//! weights/tokenizer, and run the forward pass.

use rusty_llama::dummy::{
    synthetic_control_vector_gguf, synthetic_gguf, synthetic_gguf_arch, synthetic_gguf_gpt2,
    synthetic_gguf_gpt2_special, synthetic_gguf_moe, synthetic_gguf_typed, synthetic_lora_gguf,
};
use rusty_llama::quant::dequantize;
use rusty_llama::{
    forward, forward_prefill, generate, Arch, Batch, Config, ControlVector, CpuBackend, GgmlType,
    Gguf, LoraAdapter, Model, RopeScaling, RunState, SamplerChain, Tokenizer,
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
    let mut sampler = SamplerChain::new(c.vocab_size, 0.0, 0.9, 1);
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
fn gpt2_special_tokens_round_trip() {
    // 256 byte tokens, then four special tokens (CONTROL=3 / USER_DEFINED=4)
    // marked via tokenizer.ggml.token_type.
    let specials = [
        ("<|begin_of_text|>", 3u32),
        ("<|im_start|>", 3),
        ("<|im_end|>", 3),
        ("<|tool|>", 4),
    ];
    let c = Config {
        dim: 32,
        hidden_dim: 64,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: 4,
        vocab_size: 256 + specials.len(),
        seq_len: 8,
        shared_weights: true,
        ..Default::default()
    };
    let bytes = synthetic_gguf_gpt2_special(&c, &specials);
    let gguf = Gguf::parse(&bytes).unwrap();
    let tk = Tokenizer::from_gguf(&gguf).unwrap();

    // Each special is recognized as exactly one id (byte tokens occupy 0..256,
    // specials follow in order).
    assert_eq!(tk.encode("<|begin_of_text|>", false, false), vec![256]);
    assert_eq!(tk.encode("<|im_start|>", false, false), vec![257]);
    assert_eq!(tk.encode("<|tool|>", false, false), vec![259]);

    // Mixed text: specials become single ids, the gaps (incl. multi-byte CJK)
    // go through byte-level BPE.
    let text = "<|im_start|>user\nHello 世界!<|im_end|>";
    let ids = tk.encode(text, false, false);
    assert!(ids.contains(&257) && ids.contains(&258));
    // The special ids are single tokens, not split into their bytes.
    assert!(ids.iter().filter(|&&id| id == 257).count() == 1);

    // Full round-trip: decode reproduces the original text, special tokens
    // included.
    let mut out = Vec::new();
    let mut prev = 0;
    for &id in &ids {
        out.extend_from_slice(&tk.decode(prev, id));
        prev = id;
    }
    assert_eq!(out, text.as_bytes());
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
        let mut sm = SamplerChain::new(c.vocab_size, 0.0, 0.9, 1);
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

#[test]
fn lora_adapter_loads_from_gguf() {
    let c = cfg();
    let mb = synthetic_gguf(&c);
    let mg = Gguf::parse(&mb).unwrap();
    let model = Model::from_gguf(&mg).unwrap();

    let kv_dim = c.n_kv_heads * (c.dim / c.n_heads);
    let targets = [
        ("blk.0.attn_q.weight", c.dim, c.dim),
        ("blk.1.ffn_gate.weight", c.hidden_dim, c.dim),
        ("blk.0.attn_v.weight", kv_dim, c.dim),
    ];
    let ab = synthetic_lora_gguf(&targets, 4, 16.0, false);
    let ag = Gguf::parse(&ab).unwrap();
    LoraAdapter::from_gguf(&ag, &model, 1.0).expect("valid LoRA adapter should load");
}

#[test]
fn lora_adapter_rejects_shape_mismatch() {
    let c = cfg();
    let mb = synthetic_gguf(&c);
    let mg = Gguf::parse(&mb).unwrap();
    let model = Model::from_gguf(&mg).unwrap();
    let targets = [("blk.0.attn_q.weight", c.dim, c.dim)];
    let bad = synthetic_lora_gguf(&targets, 4, 16.0, true); // mismatched lora_b rank
    let bg = Gguf::parse(&bad).unwrap();
    assert!(LoraAdapter::from_gguf(&bg, &model, 1.0).is_err());
}

#[test]
fn lora_from_gguf_rejects_non_adapter() {
    // A plain model GGUF has no `adapter.type` ⇒ rejected.
    let c = cfg();
    let mb = synthetic_gguf(&c);
    let mg = Gguf::parse(&mb).unwrap();
    let model = Model::from_gguf(&mg).unwrap();
    assert!(LoraAdapter::from_gguf(&mg, &model, 1.0).is_err());
}

#[test]
fn control_vector_loads_from_gguf() {
    let c = cfg();
    let mb = synthetic_gguf(&c);
    let mg = Gguf::parse(&mb).unwrap();
    let model = Model::from_gguf(&mg).unwrap();
    let cvb = synthetic_control_vector_gguf(c.dim, &[1], 0.5);
    let cvg = Gguf::parse(&cvb).unwrap();
    ControlVector::from_gguf(&cvg, &model, 1.0).expect("control vector should load");
}

#[test]
fn control_vector_rejects_out_of_range_layer() {
    let c = cfg();
    let mb = synthetic_gguf(&c);
    let mg = Gguf::parse(&mb).unwrap();
    let model = Model::from_gguf(&mg).unwrap();
    let cvb = synthetic_control_vector_gguf(c.dim, &[c.n_layers], 0.5); // layer == n_layers (OOB)
    let cvg = Gguf::parse(&cvb).unwrap();
    assert!(ControlVector::from_gguf(&cvg, &model, 1.0).is_err());
}

// --- Breadth architectures (Phase 3.1): Qwen2 / Phi-3 / Gemma2 ---------------
//
// All three are NeoX-RoPE archs, so a successful load + finite forward also
// exercises the load-time NeoX→interleaved Q/K (and bias) permute that
// `model::neox_src`/`permute_neox` implement — there is no separate unit test
// for that private helper.

#[test]
fn qwen2_loads_with_qkv_bias_and_forward_finite() {
    let c = cfg();
    let bytes = synthetic_gguf_arch(&c, "qwen2");
    let gguf = Gguf::parse(&bytes).unwrap();
    let model = Model::from_gguf(&gguf).unwrap();

    assert_eq!(model.config.arch, Arch::Qwen2);
    // Qwen2 carries additive Q/K/V projection biases.
    let q_dim = model.config.q_dim();
    let kv_dim = model.config.kv_dim();
    assert_eq!(model.weights.bq.len(), c.n_layers * q_dim);
    assert_eq!(model.weights.bk.len(), c.n_layers * kv_dim);
    assert_eq!(model.weights.bv.len(), c.n_layers * kv_dim);
    assert!(!model.weights.bq.is_empty());

    // A short prefill over a couple of tokens runs without panic, finite logits.
    let backend = CpuBackend::new();
    let mut s = RunState::new(&model.config);
    forward_prefill(&model, &mut s, &backend, &[1, 2, 3], 0);
    assert_eq!(s.logits().len(), c.vocab_size);
    assert!(s.logits().iter().all(|v| v.is_finite()));
}

#[test]
fn phi3_fused_qkv_and_gate_up_split_shapes() {
    let c = cfg();
    let bytes = synthetic_gguf_arch(&c, "phi3");
    let gguf = Gguf::parse(&bytes).unwrap();
    let model = Model::from_gguf(&gguf).unwrap();

    assert_eq!(model.config.arch, Arch::Phi3);
    // The fused `attn_qkv` and gate+up `ffn_up` tensors split into the standard
    // projections with the right row counts.
    let q_dim = model.config.q_dim();
    let kv_dim = model.config.kv_dim();
    assert_eq!(model.weights.wq[0].rows(), q_dim);
    assert_eq!(model.weights.wk[0].rows(), kv_dim);
    assert_eq!(model.weights.wv[0].rows(), kv_dim);
    assert_eq!(model.weights.w1[0].rows(), c.hidden_dim);
    assert_eq!(model.weights.w3[0].rows(), c.hidden_dim);
    // No QKV bias / sandwich norms for Phi-3.
    assert!(model.weights.bq.is_empty());
    assert!(model.weights.rms_attn_post.is_empty());

    let backend = CpuBackend::new();
    let mut s = RunState::new(&model.config);
    forward_prefill(&model, &mut s, &backend, &[1, 2, 3], 0);
    assert!(s.logits().iter().all(|v| v.is_finite()));
}

#[test]
fn gemma2_loads_softcaps_and_sandwich_norms() {
    let c = cfg();
    let bytes = synthetic_gguf_arch(&c, "gemma2");
    let gguf = Gguf::parse(&bytes).unwrap();
    let model = Model::from_gguf(&gguf).unwrap();

    assert_eq!(model.config.arch, Arch::Gemma2);
    // Gemma scales the input embeddings by sqrt(dim); Gemma2 softcaps the
    // attention scores (50) and the final logits (30).
    assert_eq!(model.config.embd_scale, (c.dim as f32).sqrt());
    assert_eq!(model.config.attn_logit_softcap, 50.0);
    assert_eq!(model.config.final_logit_softcap, 30.0);
    // "Sandwich" post-attention / post-FFN norms are present.
    assert_eq!(model.weights.rms_attn_post.len(), c.n_layers * c.dim);
    assert_eq!(model.weights.rms_ffn_post.len(), c.n_layers * c.dim);
    assert!(!model.weights.rms_attn_post.is_empty());
    assert!(!model.weights.rms_ffn_post.is_empty());

    let backend = CpuBackend::new();
    let mut s = RunState::new(&model.config);
    forward_prefill(&model, &mut s, &backend, &[1, 2, 3], 0);
    assert!(s.logits().iter().all(|v| v.is_finite()));
}

#[test]
fn gemma2_explicit_head_dim_q_dim_ne_dim_forward_finite() {
    // An explicit head_dim decouples q_dim from dim: dim=32, n_heads=4,
    // head_dim=16 ⇒ q_dim = 64 ≠ 32, exercising the q_dim != dim attention path
    // (and the NeoX permute over q_dim rows).
    let c = Config {
        dim: 32,
        hidden_dim: 32,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        vocab_size: 32,
        seq_len: 8,
        shared_weights: true,
        head_dim: 16,
        ..Default::default()
    };
    let bytes = synthetic_gguf_arch(&c, "gemma2");
    let gguf = Gguf::parse(&bytes).unwrap();
    let model = Model::from_gguf(&gguf).unwrap();

    assert_eq!(model.config.arch, Arch::Gemma2);
    assert_eq!(model.config.head_dim, 16);
    assert_eq!(model.config.q_dim(), 64);
    assert_ne!(model.config.q_dim(), model.config.dim);
    assert_eq!(model.config.embd_scale, (c.dim as f32).sqrt());

    let backend = CpuBackend::new();
    let mut s = RunState::new(&model.config);
    forward_prefill(&model, &mut s, &backend, &[1, 2, 3], 0);
    assert_eq!(s.logits().len(), c.vocab_size);
    assert!(s.logits().iter().all(|v| v.is_finite()));
}

// --- Mixture-of-experts (Phase 3.2): Mixtral-style routed MoE -----------------

fn moe_cfg() -> Config {
    Config {
        dim: 16,
        hidden_dim: 32,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        vocab_size: 32,
        seq_len: 16,
        ..Default::default()
    }
}

#[test]
fn moe_loads_and_generates() {
    let c = moe_cfg();
    let bytes = synthetic_gguf_moe(&c, 8, 2);
    let gguf = Gguf::parse(&bytes).unwrap();
    let model = Model::from_gguf(&gguf).unwrap();

    // Mixtral is the `llama` graph plus expert counts; no dense FFN tensors.
    assert_eq!(model.config.arch, Arch::Llama);
    assert_eq!(model.config.n_expert, 8);
    assert_eq!(model.config.n_expert_used, 2);
    assert!(!model.config.uses_resident_decode());
    assert_eq!(model.weights.ffn_gate_inp.len(), c.n_layers);
    assert_eq!(model.weights.w1_exp[0].len(), 8);
    assert!(model.weights.w1.is_empty());

    // Finite prefill, and greedy generation runs and is reproducible.
    let backend = CpuBackend::new();
    let mut s = RunState::new(&model.config);
    forward_prefill(&model, &mut s, &backend, &[1, 2, 3], 0);
    assert_eq!(s.logits().len(), c.vocab_size);
    assert!(s.logits().iter().all(|v| v.is_finite()));

    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let run = || {
        let mut st = RunState::new(&model.config);
        let mut sampler = SamplerChain::new(c.vocab_size, 0.0, 0.9, 1);
        let mut out = Vec::new();
        generate(&model, &mut st, &backend, &tok, &mut sampler, "hi", 6, |b| {
            out.extend_from_slice(b)
        });
        out
    };
    assert_eq!(run(), run(), "MoE greedy generation must be reproducible");
}

#[test]
fn moe_batched_decode_matches_independent_sequences() {
    let c = moe_cfg();
    let bytes = synthetic_gguf_moe(&c, 8, 2);
    let gguf = Gguf::parse(&bytes).unwrap();
    let model = Model::from_gguf(&gguf).unwrap();
    let backend = CpuBackend::new();
    let argmax = |v: &[f32]| -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    };
    let prompts: [&[usize]; 3] = [&[1, 5, 3], &[2, 7], &[4, 6, 2, 1]];
    let gen = 3usize;

    // Reference: each sequence decoded independently, greedily.
    let reference: Vec<Vec<usize>> = prompts
        .iter()
        .map(|prompt| {
            let mut state = RunState::new(&model.config);
            forward_prefill(&model, &mut state, &backend, prompt, 0);
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
    let mut batch = Batch::new(&model.config, prompts.len());
    let mut states: Vec<RunState> =
        (0..prompts.len()).map(|_| RunState::new(&model.config)).collect();
    let (mut cur, mut positions) = (Vec::new(), Vec::new());
    for (s, prompt) in prompts.iter().enumerate() {
        forward_prefill(&model, &mut states[s], &backend, prompt, 0);
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
            cur[s] = argmax(&logits[s * model.config.vocab_size..(s + 1) * model.config.vocab_size]);
            positions[s] += 1;
        }
    }

    assert_eq!(batched, reference, "MoE batched decode must match independent runs");
}
