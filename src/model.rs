//! Weight layout, run-time scratch buffers, and the transformer forward pass.

use std::borrow::Cow;

use crate::backend::Backend;
use crate::config::Config;
use crate::error::{Error, Result};
use crate::gguf::Gguf;
use crate::quant::dequantize;
use crate::sampler::Sampler;
use crate::tokenizer::Tokenizer;

/// The transformer's weight tensors.
///
/// Each field is a [`Cow`] so a tensor can either *borrow* directly from a
/// memory-mapped checkpoint (the zero-copy llama2.c path) or *own* a freshly
/// dequantized buffer (the GGUF path). Per-layer weights are stored
/// contiguously and indexed with plain slice arithmetic in [`forward`].
pub struct Weights<'a> {
    /// `(vocab_size, dim)` token embedding table.
    pub token_embedding_table: Cow<'a, [f32]>,
    /// `(n_layers, dim)` attention RMSNorm weights.
    pub rms_att_weight: Cow<'a, [f32]>,
    /// `(n_layers, dim, dim)` query projection.
    pub wq: Cow<'a, [f32]>,
    /// `(n_layers, kv_dim, dim)` key projection.
    pub wk: Cow<'a, [f32]>,
    /// `(n_layers, kv_dim, dim)` value projection.
    pub wv: Cow<'a, [f32]>,
    /// `(n_layers, dim, dim)` attention output projection.
    pub wo: Cow<'a, [f32]>,
    /// `(n_layers, dim)` feed-forward RMSNorm weights.
    pub rms_ffn_weight: Cow<'a, [f32]>,
    /// `(n_layers, hidden_dim, dim)` SwiGLU gate projection.
    pub w1: Cow<'a, [f32]>,
    /// `(n_layers, dim, hidden_dim)` SwiGLU down projection.
    pub w2: Cow<'a, [f32]>,
    /// `(n_layers, hidden_dim, dim)` SwiGLU up projection.
    pub w3: Cow<'a, [f32]>,
    /// `(dim,)` final RMSNorm weights.
    pub rms_final_weight: Cow<'a, [f32]>,
    /// `(vocab_size, dim)` output classifier (may alias `token_embedding_table`).
    pub wcls: Cow<'a, [f32]>,
}

/// A parsed model: hyper-parameters plus borrowed weights.
pub struct Model<'a> {
    /// Hyper-parameters read from the checkpoint header.
    pub config: Config,
    /// Borrowed weight tensors.
    pub weights: Weights<'a>,
}

impl<'a> Model<'a> {
    /// Parse a llama2.c checkpoint from raw bytes.
    ///
    /// `bytes` must be 4-byte aligned (a memory-mapped file always is). The
    /// returned model borrows from `bytes`, so the backing buffer must outlive
    /// it.
    // The final `take!` advances `cur` one last time without reading it back.
    #[allow(unused_assignments)]
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let config = Config::read_header(bytes)?;
        let p = &config;
        let head_size = p.head_size();
        let kv_dim = p.kv_dim();
        let l = p.n_layers;

        // `cur` is a running offset measured in f32 elements, starting right
        // after the 7-int32 header. `take!(len)` slices the next `len` f32s and
        // advances the cursor.
        let mut cur = Config::HEADER_BYTES / 4;
        macro_rules! take {
            ($len:expr) => {{
                let len = $len;
                let s = f32_slice(bytes, cur, len)?;
                cur += len;
                s
            }};
        }

        let token_embedding_table = take!(p.vocab_size * p.dim);
        let rms_att_weight = take!(l * p.dim);
        let wq = take!(l * p.dim * p.dim);
        let wk = take!(l * p.dim * kv_dim);
        let wv = take!(l * p.dim * kv_dim);
        let wo = take!(l * p.dim * p.dim);
        let rms_ffn_weight = take!(l * p.dim);
        let w1 = take!(l * p.dim * p.hidden_dim);
        let w2 = take!(l * p.hidden_dim * p.dim);
        let w3 = take!(l * p.dim * p.hidden_dim);
        let rms_final_weight = take!(p.dim);
        // Skip the two vestigial RoPE frequency tables (`freq_cis_real` and
        // `freq_cis_imag`); RoPE is now computed on the fly.
        cur += p.seq_len * head_size / 2;
        cur += p.seq_len * head_size / 2;
        let wcls = if p.shared_weights {
            token_embedding_table
        } else {
            take!(p.vocab_size * p.dim)
        };

        Ok(Model {
            config,
            weights: Weights {
                token_embedding_table: Cow::Borrowed(token_embedding_table),
                rms_att_weight: Cow::Borrowed(rms_att_weight),
                wq: Cow::Borrowed(wq),
                wk: Cow::Borrowed(wk),
                wv: Cow::Borrowed(wv),
                wo: Cow::Borrowed(wo),
                rms_ffn_weight: Cow::Borrowed(rms_ffn_weight),
                w1: Cow::Borrowed(w1),
                w2: Cow::Borrowed(w2),
                w3: Cow::Borrowed(w3),
                rms_final_weight: Cow::Borrowed(rms_final_weight),
                wcls: Cow::Borrowed(wcls),
            },
        })
    }
}

impl Model<'static> {
    /// Build a model from a parsed GGUF file, dequantizing every weight to f32.
    ///
    /// Supports the `llama` architecture (and anything sharing its tensor
    /// naming). The returned model owns its (dequantized) weights, so it does
    /// not borrow from `gguf`.
    ///
    /// Note: RoPE theta is fixed at 10000 and RMSNorm epsilon at 1e-5, which is
    /// correct for Llama-2 / TinyStories / TinyLlama. Models that override
    /// those (e.g. Llama-3's theta of 500000) are not yet handled.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let arch = gguf.meta_str("general.architecture")?.to_owned();
        let key = |k: &str| format!("{arch}.{k}");

        let dim = gguf.meta_u64(&key("embedding_length"))? as usize;
        let n_layers = gguf.meta_u64(&key("block_count"))? as usize;
        let n_heads = gguf.meta_u64(&key("attention.head_count"))? as usize;
        let n_kv_heads = gguf
            .meta_u64(&key("attention.head_count_kv"))
            .unwrap_or(n_heads as u64) as usize;
        let hidden_dim = gguf.meta_u64(&key("feed_forward_length"))? as usize;
        let seq_len = gguf.meta_u64(&key("context_length"))? as usize;

        // Vocabulary size = second dimension of the embedding tensor.
        let tok_embd = gguf
            .tensor("token_embd.weight")
            .ok_or_else(|| Error::Format("GGUF missing token_embd.weight".into()))?;
        let vocab_size = *tok_embd
            .dims
            .get(1)
            .ok_or_else(|| Error::Format("token_embd.weight is not 2-D".into()))?
            as usize;

        // A separate classifier tensor means weights are not shared.
        let shared_weights = gguf.tensor("output.weight").is_none();

        let config = Config {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
            shared_weights,
        };
        config.validate()?;
        let kv_dim = config.kv_dim();

        // Per-layer tensors are concatenated into the contiguous layout
        // `forward` expects.
        let concat = |suffix: &str| -> Result<Vec<f32>> {
            let mut v = Vec::new();
            for i in 0..n_layers {
                v.extend_from_slice(&deq_tensor(gguf, &format!("blk.{i}.{suffix}"))?);
            }
            Ok(v)
        };

        let token_embedding_table = deq_tensor(gguf, "token_embd.weight")?;
        let rms_final_weight = deq_tensor(gguf, "output_norm.weight")?;
        let wcls = if shared_weights {
            token_embedding_table.clone()
        } else {
            deq_tensor(gguf, "output.weight")?
        };
        let rms_att_weight = concat("attn_norm.weight")?;
        let wq = concat("attn_q.weight")?;
        let wk = concat("attn_k.weight")?;
        let wv = concat("attn_v.weight")?;
        let wo = concat("attn_output.weight")?;
        let rms_ffn_weight = concat("ffn_norm.weight")?;
        let w1 = concat("ffn_gate.weight")?;
        let w2 = concat("ffn_down.weight")?;
        let w3 = concat("ffn_up.weight")?;

        // Make sure every tensor matched the shape the forward pass assumes.
        check_len("token_embd", token_embedding_table.len(), vocab_size * dim)?;
        check_len("output_norm", rms_final_weight.len(), dim)?;
        check_len("output/wcls", wcls.len(), vocab_size * dim)?;
        check_len("attn_norm", rms_att_weight.len(), n_layers * dim)?;
        check_len("attn_q", wq.len(), n_layers * dim * dim)?;
        check_len("attn_k", wk.len(), n_layers * kv_dim * dim)?;
        check_len("attn_v", wv.len(), n_layers * kv_dim * dim)?;
        check_len("attn_output", wo.len(), n_layers * dim * dim)?;
        check_len("ffn_norm", rms_ffn_weight.len(), n_layers * dim)?;
        check_len("ffn_gate", w1.len(), n_layers * hidden_dim * dim)?;
        check_len("ffn_down", w2.len(), n_layers * dim * hidden_dim)?;
        check_len("ffn_up", w3.len(), n_layers * hidden_dim * dim)?;

        Ok(Model {
            config,
            weights: Weights {
                token_embedding_table: Cow::Owned(token_embedding_table),
                rms_att_weight: Cow::Owned(rms_att_weight),
                wq: Cow::Owned(wq),
                wk: Cow::Owned(wk),
                wv: Cow::Owned(wv),
                wo: Cow::Owned(wo),
                rms_ffn_weight: Cow::Owned(rms_ffn_weight),
                w1: Cow::Owned(w1),
                w2: Cow::Owned(w2),
                w3: Cow::Owned(w3),
                rms_final_weight: Cow::Owned(rms_final_weight),
                wcls: Cow::Owned(wcls),
            },
        })
    }
}

/// Dequantize a named GGUF tensor to a fresh `f32` buffer.
fn deq_tensor(gguf: &Gguf, name: &str) -> Result<Vec<f32>> {
    let info = gguf
        .tensor(name)
        .ok_or_else(|| Error::Format(format!("GGUF missing tensor '{name}'")))?;
    dequantize(info.ggml_type, gguf.tensor_bytes(info)?, info.n_elements())
}

/// Error unless `got == want` elements were produced for `name`.
fn check_len(name: &str, got: usize, want: usize) -> Result<()> {
    if got != want {
        return Err(Error::Format(format!(
            "tensor '{name}': expected {want} elements, got {got}"
        )));
    }
    Ok(())
}

/// Mutable scratch space reused across forward passes, including the KV cache.
pub struct RunState {
    x: Vec<f32>,           // residual stream (dim)
    xb: Vec<f32>,          // general scratch (dim)
    xb2: Vec<f32>,         // general scratch (dim)
    hb: Vec<f32>,          // FFN scratch (hidden_dim)
    hb2: Vec<f32>,         // FFN scratch (hidden_dim)
    q: Vec<f32>,           // query (dim)
    att: Vec<f32>,         // attention scores (n_heads * seq_len)
    logits: Vec<f32>,      // output logits (vocab_size)
    key_cache: Vec<f32>,   // (n_layers, seq_len, kv_dim)
    value_cache: Vec<f32>, // (n_layers, seq_len, kv_dim)
}

impl RunState {
    /// Allocate all scratch buffers for a given model configuration.
    pub fn new(c: &Config) -> Self {
        let kv_dim = c.kv_dim();
        let cache = c.n_layers * c.seq_len * kv_dim;
        RunState {
            x: vec![0.0; c.dim],
            xb: vec![0.0; c.dim],
            xb2: vec![0.0; c.dim],
            hb: vec![0.0; c.hidden_dim],
            hb2: vec![0.0; c.hidden_dim],
            q: vec![0.0; c.dim],
            att: vec![0.0; c.n_heads * c.seq_len],
            logits: vec![0.0; c.vocab_size],
            key_cache: vec![0.0; cache],
            value_cache: vec![0.0; cache],
        }
    }

    /// Logits produced by the most recent [`forward`] call.
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }

    /// Mutable access to the logits (the sampler scales them in place).
    pub fn logits_mut(&mut self) -> &mut [f32] {
        &mut self.logits
    }
}

/// Run one transformer step for `token` at position `pos`.
///
/// Reads/writes the KV cache in `state` and leaves the next-token logits in
/// `state.logits()`. Mirrors the reference llama2.c `forward()` op-for-op.
pub fn forward(
    model: &Model,
    state: &mut RunState,
    backend: &dyn Backend,
    token: usize,
    pos: usize,
) {
    let p = &model.config;
    let w = &model.weights;
    let dim = p.dim;
    let kv_dim = p.kv_dim();
    let hidden = p.hidden_dim;
    let head_size = p.head_size();

    // Seed the residual stream with this token's embedding.
    state
        .x
        .copy_from_slice(&w.token_embedding_table[token * dim..token * dim + dim]);

    for layer in 0..p.n_layers {
        // --- Attention ---------------------------------------------------
        backend.rmsnorm(&mut state.xb, &state.x, slice(&w.rms_att_weight, layer, dim));

        let loff = layer * p.seq_len * kv_dim;
        let kv_at = loff + pos * kv_dim;

        backend.matmul(
            &mut state.q,
            &state.xb,
            slice(&w.wq, layer, dim * dim),
            dim,
            dim,
        );
        backend.matmul(
            &mut state.key_cache[kv_at..kv_at + kv_dim],
            &state.xb,
            slice(&w.wk, layer, dim * kv_dim),
            dim,
            kv_dim,
        );
        backend.matmul(
            &mut state.value_cache[kv_at..kv_at + kv_dim],
            &state.xb,
            slice(&w.wv, layer, dim * kv_dim),
            dim,
            kv_dim,
        );

        backend.rope(
            &mut state.q,
            &mut state.key_cache[kv_at..kv_at + kv_dim],
            pos,
            head_size,
            kv_dim,
        );

        backend.attention(
            &mut state.xb,
            &state.q,
            &state.key_cache[loff..loff + p.seq_len * kv_dim],
            &state.value_cache[loff..loff + p.seq_len * kv_dim],
            &mut state.att,
            pos,
            p.n_heads,
            p.n_kv_heads,
            head_size,
            p.seq_len,
            kv_dim,
        );

        backend.matmul(
            &mut state.xb2,
            &state.xb,
            slice(&w.wo, layer, dim * dim),
            dim,
            dim,
        );
        backend.add(&mut state.x, &state.xb2);

        // --- Feed-forward (SwiGLU) --------------------------------------
        backend.rmsnorm(&mut state.xb, &state.x, slice(&w.rms_ffn_weight, layer, dim));
        backend.matmul(
            &mut state.hb,
            &state.xb,
            slice(&w.w1, layer, dim * hidden),
            dim,
            hidden,
        );
        backend.matmul(
            &mut state.hb2,
            &state.xb,
            slice(&w.w3, layer, dim * hidden),
            dim,
            hidden,
        );
        backend.swiglu(&mut state.hb, &state.hb2);
        backend.matmul(
            &mut state.xb,
            &state.hb,
            slice(&w.w2, layer, hidden * dim),
            hidden,
            dim,
        );
        backend.add(&mut state.x, &state.xb);
    }

    // Final norm (written into `xb` to avoid aliasing `x`) then classifier.
    backend.rmsnorm(&mut state.xb, &state.x, &w.rms_final_weight);
    backend.matmul(&mut state.logits, &state.xb, &w.wcls, dim, p.vocab_size);
}

/// Autoregressively generate from `prompt`, streaming decoded bytes to
/// `on_piece`. Returns the number of tokens generated (excluding the prompt).
///
/// `steps` is clamped to the model's `seq_len`. Generation stops early if the
/// BOS token (id 1) is produced, matching llama2.c's behaviour.
#[allow(clippy::too_many_arguments)]
pub fn generate(
    model: &Model,
    state: &mut RunState,
    backend: &dyn Backend,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    prompt: &str,
    steps: usize,
    mut on_piece: impl FnMut(&[u8]),
) -> usize {
    let prompt_tokens = tokenizer.encode(prompt, true, false);
    let steps = steps.min(model.config.seq_len);

    let mut token = prompt_tokens[0];
    let mut pos = 0;
    let mut generated = 0;

    while pos < steps {
        forward(model, state, backend, token, pos);

        // While we still have prompt tokens to ingest, force the next token;
        // otherwise sample it.
        let next = if pos < prompt_tokens.len() - 1 {
            prompt_tokens[pos + 1]
        } else {
            sampler.sample(state.logits_mut())
        };
        pos += 1;

        if next == 1 {
            break; // BOS doubles as an end-of-sequence delimiter here.
        }

        on_piece(&tokenizer.decode(token, next));
        token = next;
        generated += 1;
    }

    generated
}

/// Borrow per-layer slice `layer` of a tensor laid out as `(n_layers, stride)`.
#[inline]
fn slice(tensor: &[f32], layer: usize, stride: usize) -> &[f32] {
    &tensor[layer * stride..layer * stride + stride]
}

/// Reinterpret `len` little-endian f32s starting at element offset `elem_off`.
///
/// Returns an error (rather than panicking) on out-of-bounds or misaligned
/// input. `f32` has no invalid bit patterns, so the cast itself is sound once
/// bounds and alignment are checked.
fn f32_slice(bytes: &[u8], elem_off: usize, len: usize) -> Result<&[f32]> {
    let start = elem_off
        .checked_mul(4)
        .ok_or_else(|| Error::Format("weight offset overflow".into()))?;
    let byte_len = len
        .checked_mul(4)
        .ok_or_else(|| Error::Format("weight length overflow".into()))?;
    let end = start
        .checked_add(byte_len)
        .ok_or_else(|| Error::Format("weight offset overflow".into()))?;
    if end > bytes.len() {
        return Err(Error::Format(format!(
            "checkpoint truncated: need {end} bytes for weights, file has {}",
            bytes.len()
        )));
    }
    let region = &bytes[start..end];
    if !(region.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()) {
        return Err(Error::Format(
            "checkpoint data is not 4-byte aligned (load it via a memory-mapped file)".into(),
        ));
    }
    // SAFETY: bounds and alignment checked above; every bit pattern is a valid
    // f32; the crate is little-endian only (asserted at the crate root).
    Ok(unsafe { std::slice::from_raw_parts(region.as_ptr() as *const f32, len) })
}
