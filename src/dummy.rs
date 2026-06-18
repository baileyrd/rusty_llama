//! Synthetic models and tokenizers for tests and quick demos.
//!
//! These let you exercise the full pipeline (load → forward → sample → decode)
//! without downloading multi-megabyte weights. The output is, of course,
//! gibberish — the weights are deterministic noise.

use crate::arch::Arch;
use crate::config::{Config, RopeScaling};
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
    fn kv_i32_array(&mut self, key: &str, items: &[i32]) {
        self.str(key);
        self.u32(9); // array
        self.u32(5); // of i32
        self.u64(items.len() as u64);
        for &v in items {
            self.u32(v as u32);
        }
        self.kv_count += 1;
    }
}

/// Emit the `llama.rope.scaling.*` metadata keys for a [`RopeScaling`].
fn emit_rope_scaling(meta: &mut GgufWriter, scaling: RopeScaling) {
    let k = "llama.rope.scaling.";
    match scaling {
        RopeScaling::None => {}
        RopeScaling::Linear { factor } => {
            meta.kv_str(&format!("{k}type"), "linear");
            meta.kv_f32(&format!("{k}factor"), factor);
        }
        RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            orig_ctx,
        } => {
            meta.kv_str(&format!("{k}type"), "llama3");
            meta.kv_f32(&format!("{k}factor"), factor);
            meta.kv_f32(&format!("{k}low_freq_factor"), low_freq_factor);
            meta.kv_f32(&format!("{k}high_freq_factor"), high_freq_factor);
            meta.kv_u32(&format!("{k}original_context_length"), orig_ctx as u32);
        }
        RopeScaling::Yarn {
            factor,
            orig_ctx,
            attn_factor,
            ..
        } => {
            meta.kv_str(&format!("{k}type"), "yarn");
            meta.kv_f32(&format!("{k}factor"), factor);
            meta.kv_f32(&format!("{k}attn_factor"), attn_factor);
            meta.kv_u32(&format!("{k}original_context_length"), orig_ctx as u32);
        }
    }
}

/// Which embedded tokenizer the synthetic GGUF carries.
#[derive(Clone, Copy)]
enum TokKind {
    Llama,
    Gpt2,
}

/// Serialize a synthetic GGUF `llama` model with F32 weights and a dummy
/// (SentencePiece) tokenizer, in the layout [`crate::Model::from_gguf`] expects.
pub fn synthetic_gguf(config: &Config) -> Vec<u8> {
    build_gguf(config, GgmlType::F32, TokKind::Llama, &[])
}

/// Like [`synthetic_gguf`] but stores the 2-D matmul weights in `matmul_type`
/// (the 1-D RMSNorm vectors stay F32). For quantized types, `dim` and
/// `hidden_dim` must be multiples of the block size.
///
/// `config.shared_weights` is honoured (no `output.weight` when shared).
pub fn synthetic_gguf_typed(config: &Config, matmul_type: GgmlType) -> Vec<u8> {
    build_gguf(config, matmul_type, TokKind::Llama, &[])
}

/// A synthetic GGUF carrying a `gpt2` byte-level BPE tokenizer (all 256 byte
/// tokens, no merges). Requires `config.vocab_size == 256`.
pub fn synthetic_gguf_gpt2(config: &Config) -> Vec<u8> {
    build_gguf(config, GgmlType::F32, TokKind::Gpt2, &[])
}

/// A `gpt2`-tokenizer GGUF whose vocabulary is the 256 byte tokens followed by
/// `specials` — `(text, token_type)` pairs (use 3 for CONTROL, 4 for
/// USER_DEFINED) — with a matching `tokenizer.ggml.token_type` array.
///
/// `config.vocab_size` must equal `256 + specials.len()`.
pub fn synthetic_gguf_gpt2_special(config: &Config, specials: &[(&str, u32)]) -> Vec<u8> {
    build_gguf(config, GgmlType::F32, TokKind::Gpt2, specials)
}

fn build_gguf(
    config: &Config,
    matmul_type: GgmlType,
    tok: TokKind,
    specials: &[(&str, u32)],
) -> Vec<u8> {
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
    meta.kv_u32("llama.rope.dimension_count", c.rotary_dim() as u32);
    emit_rope_scaling(&mut meta, c.rope_scaling);
    match tok {
        TokKind::Llama => {
            meta.kv_str("tokenizer.ggml.model", "llama");
            meta.kv_str_array("tokenizer.ggml.tokens", &dummy_tokens(c.vocab_size));
            meta.kv_f32_array("tokenizer.ggml.scores", &vec![0.0; c.vocab_size]);
        }
        TokKind::Gpt2 => {
            let mut tokens: Vec<Vec<u8>> = crate::tokenizer::gpt2_byte_tokens()
                .into_iter()
                .map(String::into_bytes)
                .collect();
            // GGML_TOKEN_TYPE_NORMAL = 1 for the byte tokens; the appended
            // special tokens carry their given CONTROL/USER_DEFINED type.
            let mut types: Vec<i32> = vec![1; tokens.len()];
            for (text, ty) in specials {
                tokens.push(text.as_bytes().to_vec());
                types.push(*ty as i32);
            }
            assert_eq!(
                tokens.len(),
                c.vocab_size,
                "gpt2 dummy needs vocab_size == 256 + specials"
            );
            meta.kv_str("tokenizer.ggml.model", "gpt2");
            meta.kv_str_array("tokenizer.ggml.tokens", &tokens);
            meta.kv_str_array("tokenizer.ggml.merges", &[]); // no merges
            if !specials.is_empty() {
                meta.kv_i32_array("tokenizer.ggml.token_type", &types);
            }
        }
    }
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

/// A synthetic GGUF for one of the breadth architectures — `"qwen2"`, `"phi3"`,
/// or `"gemma2"` — emitting that arch's tensor set and `{arch}.*` hparam keys so
/// [`crate::Model::from_gguf`] exercises the non-Llama load paths: Qwen2 QKV
/// bias, Phi-3 fused `attn_qkv` / gate+up splits, Gemma2 sandwich norms +
/// logit softcaps + `sqrt(dim)` embedding scale, and the NeoX→interleaved Q/K
/// permute shared by all three. F32 weights with a dummy SentencePiece
/// tokenizer; the byte layout mirrors [`synthetic_gguf`].
///
/// `config.head_dim` is honoured: leave it `0` for the derived `head_size =
/// dim / n_heads` (so `q_dim == dim`), or set it to emit `{arch}.attention.
/// key_length` and drive the `q_dim != dim` path (Gemma2). An unrecognized
/// `arch` falls back to the plain Llama tensor set.
pub fn synthetic_gguf_arch(config: &Config, arch: &str) -> Vec<u8> {
    let c = config;
    let a = Arch::from_name(arch);
    let q_dim = c.q_dim();
    let kv_dim = c.kv_dim();

    // (name, ggml dims) in file order. ggml dims are [cols, rows] = input axis
    // first, output axis second — exactly as [`build_gguf`] lays them out.
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
        if a.fused_qkv() {
            // Phi-3: one fused Q+K+V projection, split at load.
            tensors.push((
                format!("{p}attn_qkv.weight"),
                vec![c.dim as u64, (q_dim + 2 * kv_dim) as u64],
            ));
        } else {
            tensors.push((format!("{p}attn_q.weight"), vec![c.dim as u64, q_dim as u64]));
            tensors.push((format!("{p}attn_k.weight"), vec![c.dim as u64, kv_dim as u64]));
            tensors.push((format!("{p}attn_v.weight"), vec![c.dim as u64, kv_dim as u64]));
        }
        if a.has_qkv_bias() {
            // Qwen2: additive Q/K/V projection biases (1-D).
            tensors.push((format!("{p}attn_q.bias"), vec![q_dim as u64]));
            tensors.push((format!("{p}attn_k.bias"), vec![kv_dim as u64]));
            tensors.push((format!("{p}attn_v.bias"), vec![kv_dim as u64]));
        }
        tensors.push((format!("{p}attn_output.weight"), vec![q_dim as u64, c.dim as u64]));
        tensors.push((format!("{p}ffn_norm.weight"), vec![c.dim as u64]));
        if a.fused_gate_up() {
            // Phi-3: fused gate+up in `ffn_up` (rows = 2*hidden_dim), split at load.
            tensors.push((
                format!("{p}ffn_up.weight"),
                vec![c.dim as u64, (2 * c.hidden_dim) as u64],
            ));
        } else {
            tensors.push((format!("{p}ffn_gate.weight"), vec![c.dim as u64, c.hidden_dim as u64]));
            tensors.push((format!("{p}ffn_up.weight"), vec![c.dim as u64, c.hidden_dim as u64]));
        }
        tensors.push((format!("{p}ffn_down.weight"), vec![c.hidden_dim as u64, c.dim as u64]));
        if a.sandwich_norm() {
            // Gemma2: post-attention / post-FFN "sandwich" norms (1-D).
            tensors.push((format!("{p}post_attention_norm.weight"), vec![c.dim as u64]));
            tensors.push((format!("{p}post_ffw_norm.weight"), vec![c.dim as u64]));
        }
    }

    // Build the (aligned) tensor-data section, recording each tensor's offset.
    // Deterministic small F32 values, byte-for-byte the [`build_gguf`] scheme.
    let mut data = Vec::new();
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut rng = 0x9E37_79B9_7F4A_7C15u64 ^ tensors.len() as u64;
    for (_, dims) in &tensors {
        let n: usize = dims.iter().product::<u64>() as usize;
        let mut bytes = Vec::with_capacity(n * 4);
        for _ in 0..n {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            let u = (rng.wrapping_mul(0x2545F491_4F6CDD1D) >> 40) as u32;
            let v = (u as f32 / (1u32 << 24) as f32 - 0.5) * 0.2;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let off = data.len().next_multiple_of(GGUF_ALIGNMENT);
        data.resize(off, 0);
        offsets.push(off as u64);
        data.extend_from_slice(&bytes);
    }

    // Metadata, all under the `{arch}.` prefix the loader keys on.
    let mut meta = GgufWriter::default();
    meta.kv_str("general.architecture", arch);
    meta.kv_u32("general.alignment", GGUF_ALIGNMENT as u32);
    meta.kv_u32(&format!("{arch}.embedding_length"), c.dim as u32);
    meta.kv_u32(&format!("{arch}.block_count"), c.n_layers as u32);
    meta.kv_u32(&format!("{arch}.attention.head_count"), c.n_heads as u32);
    meta.kv_u32(&format!("{arch}.attention.head_count_kv"), c.n_kv_heads as u32);
    meta.kv_u32(&format!("{arch}.feed_forward_length"), c.hidden_dim as u32);
    meta.kv_u32(&format!("{arch}.context_length"), c.seq_len as u32);
    meta.kv_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"), c.rms_eps);
    meta.kv_f32(&format!("{arch}.rope.freq_base"), c.rope_freq_base);
    meta.kv_u32(&format!("{arch}.rope.dimension_count"), c.rotary_dim() as u32);
    // Explicit attention head dim (drives q_dim != dim); emitted only when set.
    if c.head_dim != 0 {
        meta.kv_u32(&format!("{arch}.attention.key_length"), c.head_dim as u32);
    }
    // Gemma2 tanh logit softcaps on the attention scores and the final logits.
    if a.sandwich_norm() {
        meta.kv_f32(&format!("{arch}.attn_logit_softcapping"), 50.0);
        meta.kv_f32(&format!("{arch}.final_logit_softcapping"), 30.0);
    }
    // A dummy SentencePiece tokenizer so the file is also tokenizer-loadable.
    meta.kv_str("tokenizer.ggml.model", "llama");
    meta.kv_str_array("tokenizer.ggml.tokens", &dummy_tokens(c.vocab_size));
    meta.kv_f32_array("tokenizer.ggml.scores", &vec![0.0; c.vocab_size]);
    meta.kv_u32("tokenizer.ggml.bos_token_id", 1);
    meta.kv_u32("tokenizer.ggml.eos_token_id", 2);

    // Tensor info table (all F32).
    let mut tinfo = GgufWriter::default();
    for ((name, dims), off) in tensors.iter().zip(&offsets) {
        tinfo.raw_str(name.as_bytes());
        tinfo.u32(dims.len() as u32);
        for &d in dims {
            tinfo.u64(d);
        }
        tinfo.u32(GgmlType::F32.to_u32());
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

/// A synthetic LoRA adapter GGUF targeting `targets` (`(base_name, out, in)`),
/// each as an `(A, B)` pair at `rank` with scaling `alpha`. Tensors are F32.
/// When `corrupt`, the first target's `lora_b` is emitted with a mismatched rank
/// so [`crate::adapter::LoraAdapter::from_gguf`] rejects it.
pub fn synthetic_lora_gguf(
    targets: &[(&str, usize, usize)],
    rank: usize,
    alpha: f32,
    corrupt: bool,
) -> Vec<u8> {
    // (name, ggml dims) in file order. ggml dims are [cols, rows] (input first):
    // A is (rank, in) -> [in, rank]; B is (out, rank) -> [rank, out].
    let mut tensors: Vec<(String, Vec<u64>)> = Vec::new();
    for (i, &(name, out, inf)) in targets.iter().enumerate() {
        tensors.push((format!("{name}.lora_a"), vec![inf as u64, rank as u64]));
        let b_rank = if corrupt && i == 0 { rank + 1 } else { rank };
        tensors.push((format!("{name}.lora_b"), vec![b_rank as u64, out as u64]));
    }

    // Deterministic small F32 data, each tensor aligned.
    let mut data = Vec::new();
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut rng = 0xD1B5_4A32_D192_ED03u64 ^ tensors.len() as u64;
    for (_, dims) in &tensors {
        let n: usize = dims.iter().product::<u64>() as usize;
        let mut bytes = Vec::with_capacity(n * 4);
        for _ in 0..n {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            let u = (rng.wrapping_mul(0x2545F491_4F6CDD1D) >> 40) as u32;
            let v = (u as f32 / (1u32 << 24) as f32 - 0.5) * 0.2;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let off = data.len().next_multiple_of(GGUF_ALIGNMENT);
        data.resize(off, 0);
        offsets.push(off as u64);
        data.extend_from_slice(&bytes);
    }

    let mut meta = GgufWriter::default();
    meta.kv_str("general.architecture", "llama");
    meta.kv_str("general.type", "adapter");
    meta.kv_str("adapter.type", "lora");
    meta.kv_u32("general.alignment", GGUF_ALIGNMENT as u32);
    meta.kv_f32("adapter.lora.alpha", alpha);

    let mut tinfo = GgufWriter::default();
    for ((name, dims), off) in tensors.iter().zip(&offsets) {
        tinfo.raw_str(name.as_bytes());
        tinfo.u32(dims.len() as u32);
        for &d in dims {
            tinfo.u64(d);
        }
        tinfo.u32(GgmlType::F32.to_u32());
        tinfo.u64(*off);
    }

    let mut out = GgufWriter::default();
    out.buf.extend_from_slice(b"GGUF");
    out.u32(3);
    out.u64(tensors.len() as u64);
    out.u64(meta.kv_count);
    out.buf.extend_from_slice(&meta.buf);
    out.buf.extend_from_slice(&tinfo.buf);
    let padded = out.buf.len().next_multiple_of(GGUF_ALIGNMENT);
    out.buf.resize(padded, 0);
    out.buf.extend_from_slice(&data);
    out.buf
}

/// A synthetic control-vector GGUF with a constant F32 `direction.{l}` tensor
/// (length `dim`, all `fill`) for each layer index in `layers`.
pub fn synthetic_control_vector_gguf(dim: usize, layers: &[usize], fill: f32) -> Vec<u8> {
    let tensors: Vec<(String, Vec<u64>)> = layers
        .iter()
        .map(|&l| (format!("direction.{l}"), vec![dim as u64]))
        .collect();

    let mut data = Vec::new();
    let mut offsets = Vec::with_capacity(tensors.len());
    for _ in &tensors {
        let off = data.len().next_multiple_of(GGUF_ALIGNMENT);
        data.resize(off, 0);
        offsets.push(off as u64);
        for _ in 0..dim {
            data.extend_from_slice(&fill.to_le_bytes());
        }
    }

    let mut meta = GgufWriter::default();
    meta.kv_str("general.architecture", "llama");
    meta.kv_str("general.type", "controlvector");
    meta.kv_u32("general.alignment", GGUF_ALIGNMENT as u32);

    let mut tinfo = GgufWriter::default();
    for ((name, dims), off) in tensors.iter().zip(&offsets) {
        tinfo.raw_str(name.as_bytes());
        tinfo.u32(dims.len() as u32);
        for &d in dims {
            tinfo.u64(d);
        }
        tinfo.u32(GgmlType::F32.to_u32());
        tinfo.u64(*off);
    }

    let mut out = GgufWriter::default();
    out.buf.extend_from_slice(b"GGUF");
    out.u32(3);
    out.u64(tensors.len() as u64);
    out.u64(meta.kv_count);
    out.buf.extend_from_slice(&meta.buf);
    out.buf.extend_from_slice(&tinfo.buf);
    let padded = out.buf.len().next_multiple_of(GGUF_ALIGNMENT);
    out.buf.resize(padded, 0);
    out.buf.extend_from_slice(&data);
    out.buf
}
