//! Synthetic models and tokenizers for tests and quick demos.
//!
//! These let you exercise the full pipeline (load → forward → sample → decode)
//! without downloading multi-megabyte weights. The output is, of course,
//! gibberish — the weights are deterministic noise.

use crate::config::Config;
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
