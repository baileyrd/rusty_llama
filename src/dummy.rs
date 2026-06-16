//! Synthetic models and tokenizers for tests and quick demos.
//!
//! These let you exercise the full pipeline (load → forward → sample → decode)
//! without downloading multi-megabyte weights. The output is, of course,
//! gibberish — the weights are deterministic noise.

use crate::config::Config;
use crate::quant::GgmlType;
use crate::tokenizer::Tokenizer;

/// Serialize a checkpoint with deterministic pseudo-random weights matching
/// `config`'s layout, in the exact byte format [`crate::Model::parse`] expects.
///
/// `config.shared_weights` is honoured: when set, no separate classifier
/// matrix is emitted.
pub fn synthetic_checkpoint(config: &Config) -> Vec<u8> {
    let p = config;
    let head_size = p.head_size();
    let kv_dim = p.kv_dim();
    let l = p.n_layers;

    // Element counts in the same order `Model::parse` reads them, including the
    // two skipped (legacy RoPE) tables.
    let mut counts = vec![
        p.vocab_size * p.dim,    // token_embedding_table
        l * p.dim,               // rms_att_weight
        l * p.dim * p.dim,       // wq
        l * p.dim * kv_dim,      // wk
        l * p.dim * kv_dim,      // wv
        l * p.dim * p.dim,       // wo
        l * p.dim,               // rms_ffn_weight
        l * p.dim * p.hidden_dim, // w1
        l * p.hidden_dim * p.dim, // w2
        l * p.dim * p.hidden_dim, // w3
        p.dim,                   // rms_final_weight
        p.seq_len * head_size / 2, // freq_cis_real (skipped)
        p.seq_len * head_size / 2, // freq_cis_imag (skipped)
    ];
    if !p.shared_weights {
        counts.push(p.vocab_size * p.dim); // wcls
    }
    let total: usize = counts.iter().sum();

    let mut bytes = Vec::with_capacity(Config::HEADER_BYTES + total * 4);
    let vocab_field = if p.shared_weights {
        p.vocab_size as i32
    } else {
        -(p.vocab_size as i32)
    };
    let header = [
        p.dim as i32,
        p.hidden_dim as i32,
        l as i32,
        p.n_heads as i32,
        p.n_kv_heads as i32,
        vocab_field,
        p.seq_len as i32,
    ];
    for h in header {
        bytes.extend_from_slice(&h.to_le_bytes());
    }

    // Small deterministic values keep the forward pass numerically tame.
    let mut rng = 0x2545F491_4F6CDD1Du64 ^ total as u64;
    for _ in 0..total {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        let u = (rng.wrapping_mul(0x2545F491_4F6CDD1D) >> 40) as u32; // 24 bits
        let v = (u as f32 / (1u32 << 24) as f32 - 0.5) * 0.2; // [-0.1, 0.1)
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// A trivial tokenizer of `vocab_size` entries where token `i` decodes to the
/// literal text `<i>` (with the usual specials at 0/1/2).
pub fn dummy_tokenizer(vocab_size: usize) -> Tokenizer {
    let vocab = (0..vocab_size)
        .map(|i| match i {
            0 => b"<unk>".to_vec(),
            1 => b"\n<s>\n".to_vec(),
            2 => b"\n</s>\n".to_vec(),
            _ => format!("<{i}>").into_bytes(),
        })
        .collect();
    Tokenizer::from_vocab(vocab, vec![0.0; vocab_size])
}

/// Names of the dummy vocabulary entries (specials at 0/1/2, then `<i>`).
fn dummy_tokens(vocab_size: usize) -> Vec<Vec<u8>> {
    (0..vocab_size)
        .map(|i| match i {
            0 => b"<unk>".to_vec(),
            1 => b"\n<s>\n".to_vec(),
            2 => b"\n</s>\n".to_vec(),
            _ => format!("<{i}>").into_bytes(),
        })
        .collect()
}

const GGUF_ALIGNMENT: usize = 32;

/// A tiny little-endian writer for assembling GGUF byte streams.
#[derive(Default)]
struct GgufWriter {
    buf: Vec<u8>,
    kv_count: u64,
}

impl GgufWriter {
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn raw_str(&mut self, b: &[u8]) {
        self.u64(b.len() as u64);
        self.buf.extend_from_slice(b);
    }
    fn str(&mut self, s: &str) {
        self.raw_str(s.as_bytes());
    }
    fn kv_u32(&mut self, key: &str, v: u32) {
        self.str(key);
        self.u32(4);
        self.u32(v);
        self.kv_count += 1;
    }
    fn kv_f32(&mut self, key: &str, v: f32) {
        self.str(key);
        self.u32(6);
        self.f32(v);
        self.kv_count += 1;
    }
    fn kv_str(&mut self, key: &str, v: &str) {
        self.str(key);
        self.u32(8);
        self.str(v);
        self.kv_count += 1;
    }
    fn kv_str_array(&mut self, key: &str, items: &[Vec<u8>]) {
        self.str(key);
        self.u32(9); // array
        self.u32(8); // of string
        self.u64(items.len() as u64);
        for it in items {
            self.raw_str(it);
        }
        self.kv_count += 1;
    }
    fn kv_f32_array(&mut self, key: &str, items: &[f32]) {
        self.str(key);
        self.u32(9); // array
        self.u32(6); // of f32
        self.u64(items.len() as u64);
        for &v in items {
            self.f32(v);
        }
        self.kv_count += 1;
    }
}

/// Serialize a synthetic GGUF `llama` model with F32 weights and a dummy
/// tokenizer, in the layout [`crate::Model::from_gguf`] expects.
pub fn synthetic_gguf(config: &Config) -> Vec<u8> {
    synthetic_gguf_typed(config, GgmlType::F32)
}

/// Like [`synthetic_gguf`] but stores the 2-D matmul weights in `matmul_type`
/// (the 1-D RMSNorm vectors stay F32). For quantized types, `dim` and
/// `hidden_dim` must be multiples of the block size.
///
/// `config.shared_weights` is honoured (no `output.weight` when shared).
pub fn synthetic_gguf_typed(config: &Config, matmul_type: GgmlType) -> Vec<u8> {
    let c = config;
    // 2-D tensors carry the (possibly quantized) matmul weights; 1-D norm
    // vectors stay full precision.
    let tensor_type = |dims: &[u64]| {
        if dims.len() == 2 {
            matmul_type
        } else {
            GgmlType::F32
        }
    };
    let kv_dim = c.kv_dim();

    // (name, ggml dims) for every tensor, in file order. ggml dims put the
    // fastest-varying (input) axis first.
    let mut tensors: Vec<(String, Vec<u64>)> = vec![
        ("token_embd.weight".into(), vec![c.dim as u64, c.vocab_size as u64]),
        ("output_norm.weight".into(), vec![c.dim as u64]),
    ];
    if !c.shared_weights {
        tensors.push((
            "output.weight".into(),
            vec![c.dim as u64, c.vocab_size as u64],
        ));
    }
    for i in 0..c.n_layers {
        let p = format!("blk.{i}.");
        tensors.push((format!("{p}attn_norm.weight"), vec![c.dim as u64]));
        tensors.push((format!("{p}attn_q.weight"), vec![c.dim as u64, c.dim as u64]));
        tensors.push((format!("{p}attn_k.weight"), vec![c.dim as u64, kv_dim as u64]));
        tensors.push((format!("{p}attn_v.weight"), vec![c.dim as u64, kv_dim as u64]));
        tensors.push((format!("{p}attn_output.weight"), vec![c.dim as u64, c.dim as u64]));
        tensors.push((format!("{p}ffn_norm.weight"), vec![c.dim as u64]));
        tensors.push((format!("{p}ffn_gate.weight"), vec![c.dim as u64, c.hidden_dim as u64]));
        tensors.push((format!("{p}ffn_down.weight"), vec![c.hidden_dim as u64, c.dim as u64]));
        tensors.push((format!("{p}ffn_up.weight"), vec![c.dim as u64, c.hidden_dim as u64]));
    }

    // Build the (aligned) tensor-data section, recording each tensor's offset.
    let mut data = Vec::new();
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut rng = 0x9E37_79B9_7F4A_7C15u64 ^ tensors.len() as u64;
    for (_, dims) in &tensors {
        let n: usize = dims.iter().product::<u64>() as usize;
        // Deterministic small f32 values for this tensor.
        let mut vals = Vec::with_capacity(n);
        for _ in 0..n {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            let u = (rng.wrapping_mul(0x2545F491_4F6CDD1D) >> 40) as u32;
            vals.push((u as f32 / (1u32 << 24) as f32 - 0.5) * 0.2);
        }
        let encoded = match tensor_type(dims) {
            GgmlType::F32 => vals.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
            GgmlType::Q8_0 => crate::quant::quantize_q8_0(&vals),
            other => panic!("synthetic_gguf_typed does not support {other:?} yet"),
        };
        let off = data.len().next_multiple_of(GGUF_ALIGNMENT);
        data.resize(off, 0);
        offsets.push(off as u64);
        data.extend_from_slice(&encoded);
    }

    // Metadata.
    let tokens = dummy_tokens(c.vocab_size);
    let mut meta = GgufWriter::default();
    meta.kv_str("general.architecture", "llama");
    meta.kv_u32("general.alignment", GGUF_ALIGNMENT as u32);
    meta.kv_u32("llama.embedding_length", c.dim as u32);
    meta.kv_u32("llama.block_count", c.n_layers as u32);
    meta.kv_u32("llama.attention.head_count", c.n_heads as u32);
    meta.kv_u32("llama.attention.head_count_kv", c.n_kv_heads as u32);
    meta.kv_u32("llama.feed_forward_length", c.hidden_dim as u32);
    meta.kv_u32("llama.context_length", c.seq_len as u32);
    meta.kv_f32("llama.attention.layer_norm_rms_epsilon", c.rms_eps);
    meta.kv_f32("llama.rope.freq_base", c.rope_freq_base);
    meta.kv_u32("llama.rope.dimension_count", (c.dim / c.n_heads) as u32);
    meta.kv_str("tokenizer.ggml.model", "llama");
    meta.kv_str_array("tokenizer.ggml.tokens", &tokens);
    meta.kv_f32_array("tokenizer.ggml.scores", &vec![0.0; c.vocab_size]);
    meta.kv_u32("tokenizer.ggml.bos_token_id", 1);
    meta.kv_u32("tokenizer.ggml.eos_token_id", 2);

    // Tensor info table.
    let mut tinfo = GgufWriter::default();
    for ((name, dims), off) in tensors.iter().zip(&offsets) {
        tinfo.raw_str(name.as_bytes());
        tinfo.u32(dims.len() as u32);
        for &d in dims {
            tinfo.u64(d);
        }
        tinfo.u32(tensor_type(dims).to_u32());
        tinfo.u64(*off);
    }

    // Assemble: header | metadata | tensor infos | pad | data.
    let mut out = GgufWriter::default();
    out.buf.extend_from_slice(b"GGUF");
    out.u32(3); // version
    out.u64(tensors.len() as u64);
    out.u64(meta.kv_count);
    out.buf.extend_from_slice(&meta.buf);
    out.buf.extend_from_slice(&tinfo.buf);
    let padded = out.buf.len().next_multiple_of(GGUF_ALIGNMENT);
    out.buf.resize(padded, 0);
    out.buf.extend_from_slice(&data);
    out.buf
}
