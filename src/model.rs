//! Weight layout, run-time scratch buffers, and the transformer forward pass.

use std::borrow::Cow;

use crate::backend::Backend;
use crate::config::{Config, RopeScaling, RopeTable};
use crate::error::{Error, Result};
use crate::gguf::Gguf;
use crate::quant::{dequantize, GgmlType};
use crate::sampler::Sampler;
use crate::tensor::QMatrix;
use crate::tokenizer::Tokenizer;

/// The transformer's weight tensors.
///
/// Matmul weights are [`QMatrix`]es, so they can be borrowed straight from a
/// memory-mapped file — full precision (llama2.c) or quantized (GGUF) — and
/// dequantized on the fly, keeping compressed weights compressed in RAM. The
/// small RMSNorm vectors stay as plain f32.
pub struct Weights<'a> {
    /// `(vocab_size, dim)` token embedding table; rows double as the input
    /// embeddings looked up per token.
    pub token_embedding_table: QMatrix<'a>,
    /// `(vocab_size, dim)` output classifier (may share `token_embedding_table`).
    pub wcls: QMatrix<'a>,
    /// `(n_layers, dim)` attention RMSNorm weights.
    pub rms_att_weight: Cow<'a, [f32]>,
    /// `(n_layers, dim)` feed-forward RMSNorm weights.
    pub rms_ffn_weight: Cow<'a, [f32]>,
    /// `(dim,)` final RMSNorm weights.
    pub rms_final_weight: Cow<'a, [f32]>,
    /// Per-layer query projection, each `(dim, dim)`.
    pub wq: Vec<QMatrix<'a>>,
    /// Per-layer key projection, each `(kv_dim, dim)`.
    pub wk: Vec<QMatrix<'a>>,
    /// Per-layer value projection, each `(kv_dim, dim)`.
    pub wv: Vec<QMatrix<'a>>,
    /// Per-layer attention output projection, each `(dim, dim)`.
    pub wo: Vec<QMatrix<'a>>,
    /// Per-layer SwiGLU gate projection, each `(hidden_dim, dim)`.
    pub w1: Vec<QMatrix<'a>>,
    /// Per-layer SwiGLU down projection, each `(dim, hidden_dim)`.
    pub w2: Vec<QMatrix<'a>>,
    /// Per-layer SwiGLU up projection, each `(hidden_dim, dim)`.
    pub w3: Vec<QMatrix<'a>>,
}

/// A parsed model: hyper-parameters plus weights.
pub struct Model<'a> {
    /// Hyper-parameters.
    pub config: Config,
    /// Weight tensors.
    pub weights: Weights<'a>,
    /// Precomputed RoPE rotation table (derived from `config` at load time).
    pub rope: RopeTable,
}

impl<'a> Model<'a> {
    /// Parse a llama2.c checkpoint from raw (memory-mapped) bytes.
    ///
    /// `bytes` must be 4-byte aligned. The returned model borrows from `bytes`.
    // The final `take!` advances `cur` one last time without reading it back.
    #[allow(unused_assignments)]
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let config = Config::read_header(bytes)?;
        let p = &config;
        let head_size = p.head_size();
        let kv_dim = p.kv_dim();
        let l = p.n_layers;

        // `cur` is a running offset in f32 elements, after the 7-int32 header.
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
        // Skip the two vestigial RoPE frequency tables.
        cur += p.seq_len * head_size / 2;
        cur += p.seq_len * head_size / 2;
        let wcls_slice = if p.shared_weights {
            token_embedding_table
        } else {
            take!(p.vocab_size * p.dim)
        };

        Ok(Model {
            rope: config.rope_table(),
            config,
            weights: Weights {
                token_embedding_table: QMatrix::f32(
                    Cow::Borrowed(token_embedding_table),
                    p.vocab_size,
                    p.dim,
                )?,
                wcls: QMatrix::f32(Cow::Borrowed(wcls_slice), p.vocab_size, p.dim)?,
                rms_att_weight: Cow::Borrowed(rms_att_weight),
                rms_ffn_weight: Cow::Borrowed(rms_ffn_weight),
                rms_final_weight: Cow::Borrowed(rms_final_weight),
                wq: f32_layers(wq, l, p.dim, p.dim)?,
                wk: f32_layers(wk, l, kv_dim, p.dim)?,
                wv: f32_layers(wv, l, kv_dim, p.dim)?,
                wo: f32_layers(wo, l, p.dim, p.dim)?,
                w1: f32_layers(w1, l, p.hidden_dim, p.dim)?,
                w2: f32_layers(w2, l, p.dim, p.hidden_dim)?,
                w3: f32_layers(w3, l, p.hidden_dim, p.dim)?,
            },
        })
    }

    /// Build a model from a parsed GGUF file.
    ///
    /// Matmul weights are kept in their on-disk (quantized) form and borrowed
    /// from `gguf`; only the small RMSNorm vectors are copied to f32. Supports
    /// the `llama` architecture (and anything sharing its tensor naming).
    ///
    /// RoPE base frequency and RMSNorm epsilon are read from the file (so
    /// Llama-3 / Qwen2-style θ values are honoured). Partial rotary
    /// (`rope.dimension_count != head_size`) and long-context RoPE scaling are
    /// not yet handled.
    pub fn from_gguf(gguf: &Gguf<'a>) -> Result<Self> {
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
        let rope_freq_base = gguf.meta_f32(&key("rope.freq_base")).unwrap_or(10000.0);
        let rms_eps = gguf
            .meta_f32(&key("attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);

        // RoPE rotary dimension. Defaults to the full head size; a smaller value
        // (partial rotary) leaves each head's trailing dims unrotated.
        let head_size = dim.checked_div(n_heads).unwrap_or(0);
        let rope_dim = gguf
            .meta_u64(&key("rope.dimension_count"))
            .unwrap_or(head_size as u64) as usize;
        if head_size != 0 && rope_dim > head_size {
            return Err(Error::Format(format!(
                "rope.dimension_count {rope_dim} exceeds head_size {head_size}"
            )));
        }

        // Long-context RoPE frequency scaling (linear / llama3 / yarn), if any.
        let rope_scaling = read_rope_scaling(gguf, &arch)?;

        // Vocabulary size = second (row) dimension of the embedding tensor.
        let tok_embd = gguf
            .tensor("token_embd.weight")
            .ok_or_else(|| Error::Format("GGUF missing token_embd.weight".into()))?;
        let vocab_size = *tok_embd
            .dims
            .get(1)
            .ok_or_else(|| Error::Format("token_embd.weight is not 2-D".into()))?
            as usize;

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
            rope_freq_base,
            rms_eps,
            rope_dim,
            rope_scaling,
        };
        config.validate()?;

        // RMSNorm weights are tiny; concatenate them to owned f32.
        let concat = |suffix: &str| -> Result<Vec<f32>> {
            let mut v = Vec::with_capacity(n_layers * dim);
            for i in 0..n_layers {
                v.extend_from_slice(&deq_tensor(gguf, &format!("blk.{i}.{suffix}"))?);
            }
            Ok(v)
        };
        // Per-layer matmul matrices (borrowed, possibly quantized).
        let layers = |suffix: &str| -> Result<Vec<QMatrix<'a>>> {
            (0..n_layers)
                .map(|i| qmatrix_from_gguf(gguf, &format!("blk.{i}.{suffix}")))
                .collect()
        };

        let token_embedding_table = qmatrix_from_gguf(gguf, "token_embd.weight")?;
        let wcls = if shared_weights {
            qmatrix_from_gguf(gguf, "token_embd.weight")?
        } else {
            qmatrix_from_gguf(gguf, "output.weight")?
        };

        Ok(Model {
            rope: config.rope_table(),
            config,
            weights: Weights {
                token_embedding_table,
                wcls,
                rms_att_weight: Cow::Owned(concat("attn_norm.weight")?),
                rms_ffn_weight: Cow::Owned(concat("ffn_norm.weight")?),
                rms_final_weight: Cow::Owned(deq_tensor(gguf, "output_norm.weight")?),
                wq: layers("attn_q.weight")?,
                wk: layers("attn_k.weight")?,
                wv: layers("attn_v.weight")?,
                wo: layers("attn_output.weight")?,
                w1: layers("ffn_gate.weight")?,
                w2: layers("ffn_down.weight")?,
                w3: layers("ffn_up.weight")?,
            },
        })
    }
}

/// Read `{arch}.rope.scaling.*` into a [`RopeScaling`] (defaults to `None`).
///
/// Mirrors the keys llama.cpp writes: `type`, `factor`,
/// `original_context_length`, `attn_factor`, and the llama3 band factors. YaRN
/// `beta_fast`/`beta_slow` are runtime knobs without standard GGUF keys, so we
/// use llama.cpp's defaults (32 / 1).
fn read_rope_scaling(gguf: &Gguf, arch: &str) -> Result<RopeScaling> {
    let key = |k: &str| format!("{arch}.rope.scaling.{k}");
    let ty = match gguf.meta_str(&key("type")) {
        Ok(t) => t.to_owned(),
        Err(_) => return Ok(RopeScaling::None),
    };
    let factor = gguf.meta_f32(&key("factor")).unwrap_or(1.0);
    let orig_ctx = gguf
        .meta_u64(&key("original_context_length"))
        .unwrap_or(0) as usize;
    Ok(match ty.as_str() {
        "none" => RopeScaling::None,
        "linear" => RopeScaling::Linear { factor },
        "llama3" => RopeScaling::Llama3 {
            factor,
            low_freq_factor: gguf.meta_f32(&key("low_freq_factor")).unwrap_or(1.0),
            high_freq_factor: gguf.meta_f32(&key("high_freq_factor")).unwrap_or(4.0),
            orig_ctx,
        },
        "yarn" => RopeScaling::Yarn {
            factor,
            orig_ctx,
            attn_factor: gguf.meta_f32(&key("attn_factor")).unwrap_or(1.0),
            beta_fast: 32.0,
            beta_slow: 1.0,
        },
        other => {
            return Err(Error::Format(format!(
                "unsupported rope.scaling.type '{other}' (have none/linear/llama3/yarn)"
            )))
        }
    })
}

/// Slice a contiguous `(n_layers, rows, cols)` f32 buffer into per-layer f32
/// matrices that borrow it.
fn f32_layers(big: &[f32], n_layers: usize, rows: usize, cols: usize) -> Result<Vec<QMatrix<'_>>> {
    let stride = rows * cols;
    (0..n_layers)
        .map(|i| QMatrix::f32(Cow::Borrowed(&big[i * stride..i * stride + stride]), rows, cols))
        .collect()
}

/// Build a [`QMatrix`] from a named 2-D GGUF tensor, borrowing its bytes.
///
/// F32 tensors get a zero-copy f32 view when aligned, otherwise fall back to a
/// block-size-1 quantized view (correct, just slower).
fn qmatrix_from_gguf<'a>(gguf: &Gguf<'a>, name: &str) -> Result<QMatrix<'a>> {
    let info = gguf
        .tensor(name)
        .ok_or_else(|| Error::Format(format!("GGUF missing tensor '{name}'")))?;
    let (cols, rows) = match info.dims.as_slice() {
        [c, r] => (*c as usize, *r as usize),
        other => {
            return Err(Error::Format(format!(
                "tensor '{name}' must be 2-D, has {} dims",
                other.len()
            )))
        }
    };
    let bytes = gguf.tensor_bytes(info)?;
    if info.ggml_type == GgmlType::F32 {
        if let Ok(f) = f32_slice(bytes, 0, rows * cols) {
            return QMatrix::f32(Cow::Borrowed(f), rows, cols);
        }
    }
    QMatrix::quant(info.ggml_type, Cow::Borrowed(bytes), rows, cols)
}

/// Dequantize a named GGUF tensor to a fresh `f32` buffer.
fn deq_tensor(gguf: &Gguf, name: &str) -> Result<Vec<f32>> {
    let info = gguf
        .tensor(name)
        .ok_or_else(|| Error::Format(format!("GGUF missing tensor '{name}'")))?;
    dequantize(info.ggml_type, gguf.tensor_bytes(info)?, info.n_elements())
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
    let head_size = p.head_size();

    // Seed the residual stream with this token's (dequantized) embedding.
    w.token_embedding_table.dequant_row(token, &mut state.x);

    for layer in 0..p.n_layers {
        // --- Attention ---------------------------------------------------
        backend.rmsnorm(
            &mut state.xb,
            &state.x,
            slice(&w.rms_att_weight, layer, dim),
            p.rms_eps,
        );

        let loff = layer * p.seq_len * kv_dim;
        let kv_at = loff + pos * kv_dim;

        backend.matmul(&mut state.q, &state.xb, &w.wq[layer]);
        backend.matmul(
            &mut state.key_cache[kv_at..kv_at + kv_dim],
            &state.xb,
            &w.wk[layer],
        );
        backend.matmul(
            &mut state.value_cache[kv_at..kv_at + kv_dim],
            &state.xb,
            &w.wv[layer],
        );

        backend.rope(
            &mut state.q,
            &mut state.key_cache[kv_at..kv_at + kv_dim],
            pos,
            head_size,
            kv_dim,
            &model.rope.inv_freq,
            model.rope.mscale,
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

        backend.matmul(&mut state.xb2, &state.xb, &w.wo[layer]);
        backend.add(&mut state.x, &state.xb2);

        // --- Feed-forward (SwiGLU) --------------------------------------
        backend.rmsnorm(
            &mut state.xb,
            &state.x,
            slice(&w.rms_ffn_weight, layer, dim),
            p.rms_eps,
        );
        backend.matmul(&mut state.hb, &state.xb, &w.w1[layer]);
        backend.matmul(&mut state.hb2, &state.xb, &w.w3[layer]);
        backend.swiglu(&mut state.hb, &state.hb2);
        backend.matmul(&mut state.xb, &state.hb, &w.w2[layer]);
        backend.add(&mut state.x, &state.xb);
    }

    // Final norm (written into `xb` to avoid aliasing `x`) then classifier.
    backend.rmsnorm(&mut state.xb, &state.x, &w.rms_final_weight, p.rms_eps);
    backend.matmul(&mut state.logits, &state.xb, &w.wcls);
}

/// Run the transformer over a whole batch of `tokens` at once (prompt prefill).
///
/// Positions `pos_base .. pos_base + tokens.len()` are written into the KV
/// cache, and the **last** position's next-token logits are left in
/// `state.logits()` (the only ones a greedy/sampling decoder needs from the
/// prompt). This is exactly equivalent to calling [`forward`] once per token in
/// order — the batched [`Backend`] ops have default impls that loop the
/// single-token ones — but lets a batching backend (the GPU) fuse each op into
/// one large kernel and collapse the host↔device round-trips.
///
/// `pos_base` is general, but the prompt is normally prefilled in one call with
/// `pos_base == 0`; the cache rows for the batch are then contiguous.
pub fn forward_prefill(
    model: &Model,
    state: &mut RunState,
    backend: &dyn Backend,
    tokens: &[usize],
    pos_base: usize,
) {
    let p = &model.config;
    let w = &model.weights;
    let dim = p.dim;
    let kv_dim = p.kv_dim();
    let head_size = p.head_size();
    let hidden = p.hidden_dim;
    let n = tokens.len();

    // Row-major (n, *) scratch for the batch. Allocated here (prefill runs once
    // per prompt) rather than living in the single-token RunState.
    let mut x = vec![0.0f32; n * dim];
    let mut xb = vec![0.0f32; n * dim];
    let mut xb2 = vec![0.0f32; n * dim];
    let mut q = vec![0.0f32; n * dim];
    let mut hb = vec![0.0f32; n * hidden];
    let mut hb2 = vec![0.0f32; n * hidden];

    // Seed each row with its token's (dequantized) embedding.
    for (r, &tok) in tokens.iter().enumerate() {
        w.token_embedding_table
            .dequant_row(tok, &mut x[r * dim..r * dim + dim]);
    }

    for layer in 0..p.n_layers {
        // --- Attention --------------------------------------------------
        backend.rmsnorm_batch(&mut xb, &x, slice(&w.rms_att_weight, layer, dim), p.rms_eps, n);

        let loff = layer * p.seq_len * kv_dim;
        let kv_start = loff + pos_base * kv_dim;
        let kv_span = n * kv_dim;

        backend.matmul_batch(&mut q, &xb, &w.wq[layer], n);
        backend.matmul_batch(
            &mut state.key_cache[kv_start..kv_start + kv_span],
            &xb,
            &w.wk[layer],
            n,
        );
        backend.matmul_batch(
            &mut state.value_cache[kv_start..kv_start + kv_span],
            &xb,
            &w.wv[layer],
            n,
        );

        backend.rope_batch(
            &mut q,
            &mut state.key_cache[kv_start..kv_start + kv_span],
            pos_base,
            n,
            head_size,
            kv_dim,
            dim,
            &model.rope.inv_freq,
            model.rope.mscale,
        );

        backend.attention_batch(
            &mut xb,
            &q,
            &state.key_cache[loff..loff + p.seq_len * kv_dim],
            &state.value_cache[loff..loff + p.seq_len * kv_dim],
            &mut state.att,
            pos_base,
            n,
            p.n_heads,
            p.n_kv_heads,
            head_size,
            p.seq_len,
            kv_dim,
        );

        backend.matmul_batch(&mut xb2, &xb, &w.wo[layer], n);
        backend.add(&mut x, &xb2);

        // --- Feed-forward (SwiGLU) -------------------------------------
        backend.rmsnorm_batch(&mut xb, &x, slice(&w.rms_ffn_weight, layer, dim), p.rms_eps, n);
        backend.matmul_batch(&mut hb, &xb, &w.w1[layer], n);
        backend.matmul_batch(&mut hb2, &xb, &w.w3[layer], n);
        backend.swiglu(&mut hb, &hb2);
        backend.matmul_batch(&mut xb, &hb, &w.w2[layer], n);
        backend.add(&mut x, &xb);
    }

    // Only the final position predicts the next token, so norm + classify just
    // that row (the classifier matmul is the widest one — no need to run it n×).
    let last = (n - 1) * dim;
    let mut xb_last = vec![0.0f32; dim];
    backend.rmsnorm(&mut xb_last, &x[last..last + dim], &w.rms_final_weight, p.rms_eps);
    backend.matmul(&mut state.logits, &xb_last, &w.wcls);
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
    let n = prompt_tokens.len();

    // Fast path: batch-prefill the whole prompt in one pass, then decode. Used
    // only when every prompt token would be processed anyway (`steps >= n`) so
    // the emitted token stream is identical to the sequential loop below.
    if n >= 2 && steps >= n {
        return generate_prefilled(
            model,
            state,
            backend,
            tokenizer,
            sampler,
            &prompt_tokens,
            steps,
            on_piece,
        );
    }

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

/// Prompt-prefilled generation: fill the KV cache for the whole prompt in one
/// batched [`forward_prefill`], echo the prompt tokens (matching the sequential
/// loop's streamed output), then decode one token at a time. Produces the exact
/// same token stream as the sequential path; only the prompt's KV fill is
/// batched. Caller guarantees `prompt_tokens.len() >= 2` and `steps >= len`.
#[allow(clippy::too_many_arguments)]
fn generate_prefilled(
    model: &Model,
    state: &mut RunState,
    backend: &dyn Backend,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    prompt_tokens: &[usize],
    steps: usize,
    mut on_piece: impl FnMut(&[u8]),
) -> usize {
    let n = prompt_tokens.len();
    let mut generated = 0;

    forward_prefill(model, state, backend, prompt_tokens, 0);

    // Echo the prompt's internal transitions, exactly as the sequential loop
    // streams them (decode(prev, cur) renders `cur`'s piece).
    for r in 0..n - 1 {
        let next = prompt_tokens[r + 1];
        if next == 1 {
            return generated; // BOS-as-EOS mid-prompt (matches the loop's break)
        }
        on_piece(&tokenizer.decode(prompt_tokens[r], next));
        generated += 1;
    }

    // Sample the first generated token from the last prompt position's logits,
    // then continue single-token decoding from position `n`.
    let mut token = prompt_tokens[n - 1];
    let mut pos = n - 1;
    let mut next = sampler.sample(state.logits_mut());
    loop {
        pos += 1;
        if next == 1 {
            break; // BOS doubles as an end-of-sequence delimiter here.
        }
        on_piece(&tokenizer.decode(token, next));
        token = next;
        generated += 1;
        if pos >= steps {
            break;
        }
        forward(model, state, backend, token, pos);
        next = sampler.sample(state.logits_mut());
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
