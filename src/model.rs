//! Weight layout, run-time scratch buffers, and the transformer forward pass.

use std::borrow::Cow;

use crate::adapter::ControlVector;
use crate::arch::{Arch, FfnActivation};
use crate::backend::Backend;
use crate::config::{Config, RopeScaling, RopeTable};
use crate::error::{Error, Result};
use crate::gguf::Gguf;
use crate::quant::{dequantize, GgmlType};
use crate::sampler::SamplerChain;
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
    /// Per-layer Q/K/V projection biases (Qwen2), flattened `(n_layers, q_dim)`,
    /// `(n_layers, kv_dim)`, `(n_layers, kv_dim)`. Empty unless the arch has QKV bias.
    pub bq: Vec<f32>,
    pub bk: Vec<f32>,
    pub bv: Vec<f32>,
    /// Per-layer post-attention / post-FFN "sandwich" RMSNorm weights (Gemma2),
    /// flattened `(n_layers, dim)`. Empty unless `arch.sandwich_norm()`.
    pub rms_attn_post: Vec<f32>,
    pub rms_ffn_post: Vec<f32>,
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
                bq: Vec::new(),
                bk: Vec::new(),
                bv: Vec::new(),
                rms_attn_post: Vec::new(),
                rms_ffn_post: Vec::new(),
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
    /// Llama-3 / Qwen2-style θ values are honoured), as are partial rotary
    /// (`rope.dimension_count < head_size`, trailing dims left unrotated) and
    /// long-context RoPE scaling (`linear` / `llama3` / `yarn`).
    pub fn from_gguf(gguf: &Gguf<'a>) -> Result<Self> {
        let arch_str = gguf.meta_str("general.architecture")?.to_owned();
        let arch = Arch::from_name(&arch_str);
        let names = arch.tensor_names();
        let key = |k: &str| format!("{arch_str}.{k}");

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

        // Explicit attention head dim (Gemma); 0 ⇒ derive from dim / n_heads.
        let head_dim = gguf.meta_u64(&key("attention.key_length")).unwrap_or(0) as usize;
        let head_size = if head_dim != 0 {
            head_dim
        } else {
            dim.checked_div(n_heads).unwrap_or(0)
        };

        // RoPE rotary dimension. Defaults to the full head size; a smaller value
        // (partial rotary) leaves each head's trailing dims unrotated.
        let rope_dim = gguf
            .meta_u64(&key("rope.dimension_count"))
            .unwrap_or(head_size as u64) as usize;
        if head_size != 0 && rope_dim > head_size {
            return Err(Error::Format(format!(
                "rope.dimension_count {rope_dim} exceeds head_size {head_size}"
            )));
        }

        // Long-context RoPE frequency scaling (linear / llama3 / yarn), if any.
        let rope_scaling = read_rope_scaling(gguf, &arch_str)?;

        // Gemma scales the input embeddings by sqrt(dim); Gemma2 also softcaps
        // attention scores and final logits (keys absent ⇒ 0.0 ⇒ disabled).
        let embd_scale = if matches!(arch, Arch::Gemma2) {
            (dim as f32).sqrt()
        } else {
            1.0
        };
        let attn_logit_softcap = gguf.meta_f32(&key("attn_logit_softcapping")).unwrap_or(0.0);
        let final_logit_softcap = gguf.meta_f32(&key("final_logit_softcapping")).unwrap_or(0.0);

        // Vocabulary size = second (row) dimension of the embedding tensor.
        let tok_embd = gguf
            .tensor(names.token_embd)
            .ok_or_else(|| Error::Format("GGUF missing token_embd.weight".into()))?;
        let vocab_size = *tok_embd
            .dims
            .get(1)
            .ok_or_else(|| Error::Format("token_embd.weight is not 2-D".into()))?
            as usize;

        let shared_weights = gguf.tensor(names.output).is_none();

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
            arch,
            head_dim,
            embd_scale,
            attn_logit_softcap,
            final_logit_softcap,
        };
        config.validate()?;
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();

        // RMSNorm weights are tiny; concatenate to owned f32. (Gemma stores them
        // as `1 + offset`, but modern GGUF converters bake the `+1` in, so they
        // are used as-is like every other arch — matching current llama.cpp.)
        let concat_norm = |suffix: &str| -> Result<Vec<f32>> {
            let mut v = Vec::with_capacity(n_layers * dim);
            for i in 0..n_layers {
                v.extend_from_slice(&deq_tensor(gguf, &format!("blk.{i}.{suffix}"))?);
            }
            Ok(v)
        };
        let final_norm = deq_tensor(gguf, names.output_norm)?;
        // Per-layer bias vectors (Qwen2), concatenated to owned f32.
        let concat_bias = |suffix: &str| -> Result<Vec<f32>> {
            let mut v = Vec::new();
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

        // Q/K/V: a fused `attn_qkv` split into the three projections (Phi-3), or
        // one tensor each (everything else).
        let (wq, wk, wv) = if arch.fused_qkv() {
            let (mut wq, mut wk, mut wv) = (Vec::new(), Vec::new(), Vec::new());
            for i in 0..n_layers {
                let qkv = qmatrix_from_gguf(gguf, &format!("blk.{i}.{}", names.attn_qkv))?;
                let mut parts = split_rows(qkv, &[q_dim, kv_dim, kv_dim])?.into_iter();
                wq.push(parts.next().unwrap());
                wk.push(parts.next().unwrap());
                wv.push(parts.next().unwrap());
            }
            (wq, wk, wv)
        } else {
            (layers(names.attn_q)?, layers(names.attn_k)?, layers(names.attn_v)?)
        };

        // Gate/up FFN: a fused gate+up tensor in `ffn_up` split in half (Phi-3),
        // or separate `ffn_gate` / `ffn_up` tensors.
        let (w1, w3) = if arch.fused_gate_up() {
            let (mut w1, mut w3) = (Vec::new(), Vec::new());
            for i in 0..n_layers {
                let gate_up = qmatrix_from_gguf(gguf, &format!("blk.{i}.{}", names.ffn_up))?;
                let mut parts = split_rows(gate_up, &[hidden_dim, hidden_dim])?.into_iter();
                w1.push(parts.next().unwrap());
                w3.push(parts.next().unwrap());
            }
            (w1, w3)
        } else {
            (layers(names.ffn_gate)?, layers(names.ffn_up)?)
        };

        let (bq, bk, bv) = if arch.has_qkv_bias() {
            (
                concat_bias(names.attn_q_bias)?,
                concat_bias(names.attn_k_bias)?,
                concat_bias(names.attn_v_bias)?,
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };

        // NeoX-RoPE archs (Qwen2/Phi-3/Gemma): permute Q/K weights + biases into
        // the interleaved layout the NORM rope kernel expects — the inverse of the
        // permutation Llama GGUFs ship baked in. V / output projections untouched.
        let (wq, wk, bq, bk) = if arch.rope_neox() {
            let (hs, rot) = (config.head_size(), config.rotary_dim());
            let wq = wq
                .into_iter()
                .map(|m| permute_neox(m, hs, rot))
                .collect::<Result<Vec<_>>>()?;
            let wk = wk
                .into_iter()
                .map(|m| permute_neox(m, hs, rot))
                .collect::<Result<Vec<_>>>()?;
            let (mut bq, mut bk) = (bq, bk);
            permute_neox_bias(&mut bq, n_layers, q_dim, hs, rot);
            permute_neox_bias(&mut bk, n_layers, kv_dim, hs, rot);
            (wq, wk, bq, bk)
        } else {
            (wq, wk, bq, bk)
        };

        let (rms_attn_post, rms_ffn_post) = if arch.sandwich_norm() {
            (concat_norm(names.attn_post_norm)?, concat_norm(names.ffn_post_norm)?)
        } else {
            (Vec::new(), Vec::new())
        };

        let token_embedding_table = qmatrix_from_gguf(gguf, names.token_embd)?;
        let wcls = if shared_weights {
            qmatrix_from_gguf(gguf, names.token_embd)?
        } else {
            qmatrix_from_gguf(gguf, names.output)?
        };

        Ok(Model {
            rope: config.rope_table(),
            config,
            weights: Weights {
                token_embedding_table,
                wcls,
                rms_att_weight: Cow::Owned(concat_norm(names.attn_norm)?),
                rms_ffn_weight: Cow::Owned(concat_norm(names.ffn_norm)?),
                rms_final_weight: Cow::Owned(final_norm),
                wq,
                wk,
                wv,
                wo: layers(names.attn_output)?,
                w1,
                w2: layers(names.ffn_down)?,
                w3,
                bq,
                bk,
                bv,
                rms_attn_post,
                rms_ffn_post,
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
pub(crate) fn qmatrix_from_gguf<'a>(gguf: &Gguf<'a>, name: &str) -> Result<QMatrix<'a>> {
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

/// The transformer's key/value cache.
///
/// PHASE-0 INVARIANT: holds exactly **one** sequence; the layout is the flat
/// `(n_layers, seq_len, kv_dim)` buffer. All slot/window arithmetic lives here
/// and nowhere else, so Phase 4 (paged, multi-sequence KV — see
/// `docs/Architecture/plans/`) can repage these internals or add a `seq_id`
/// dimension without touching `forward`/`forward_prefill` or the `Backend`
/// trait. Returned slices are the exact ranges the old inline `loff`/`kv_at`
/// arithmetic produced.
pub(crate) struct KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    seq_len: usize,
    kv_dim: usize,
}

impl KvCache {
    /// Allocate a zeroed cache for `n_layers` layers of `(seq_len, kv_dim)`.
    fn new(n_layers: usize, seq_len: usize, kv_dim: usize) -> Self {
        let cells = n_layers * seq_len * kv_dim;
        KvCache {
            k: vec![0.0; cells],
            v: vec![0.0; cells],
            seq_len,
            kv_dim,
        }
    }

    /// Flat base offset of layer `layer`'s `(seq_len, kv_dim)` block.
    #[inline]
    fn layer_base(&self, layer: usize) -> usize {
        layer * self.seq_len * self.kv_dim
    }

    /// Layer `layer`'s full `(seq_len, kv_dim)` key window (what attention reads).
    fn layer_k(&self, layer: usize) -> &[f32] {
        let b = self.layer_base(layer);
        &self.k[b..b + self.seq_len * self.kv_dim]
    }

    /// Layer `layer`'s full `(seq_len, kv_dim)` value window.
    fn layer_v(&self, layer: usize) -> &[f32] {
        let b = self.layer_base(layer);
        &self.v[b..b + self.seq_len * self.kv_dim]
    }

    /// The single `kv_dim` key slot at `pos` (what the per-token K matmul writes).
    fn k_slot_mut(&mut self, layer: usize, pos: usize) -> &mut [f32] {
        let b = self.layer_base(layer) + pos * self.kv_dim;
        &mut self.k[b..b + self.kv_dim]
    }

    /// The single `kv_dim` value slot at `pos`.
    fn v_slot_mut(&mut self, layer: usize, pos: usize) -> &mut [f32] {
        let b = self.layer_base(layer) + pos * self.kv_dim;
        &mut self.v[b..b + self.kv_dim]
    }

    /// The contiguous `n`-row key span from `pos_base` (what prefill writes).
    fn k_span_mut(&mut self, layer: usize, pos_base: usize, n: usize) -> &mut [f32] {
        let b = self.layer_base(layer) + pos_base * self.kv_dim;
        &mut self.k[b..b + n * self.kv_dim]
    }

    /// The contiguous `n`-row value span from `pos_base`.
    fn v_span_mut(&mut self, layer: usize, pos_base: usize, n: usize) -> &mut [f32] {
        let b = self.layer_base(layer) + pos_base * self.kv_dim;
        &mut self.v[b..b + n * self.kv_dim]
    }

    /// The whole flat key buffer (a device-resident backend uploads it).
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn all_k(&self) -> &[f32] {
        &self.k
    }

    /// The whole flat value buffer; see [`KvCache::all_k`].
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn all_v(&self) -> &[f32] {
        &self.v
    }

    /// Write a resident prefill's computed K/V back into the host cache at flat
    /// offset `loff` (the layer base).
    #[cfg(feature = "cuda")]
    fn store_at(&mut self, loff: usize, k: &[f32], v: &[f32]) {
        self.k[loff..loff + k.len()].copy_from_slice(k);
        self.v[loff..loff + v.len()].copy_from_slice(v);
    }
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
    kv: KvCache, // single sequence; flat (n_layers, seq_len, kv_dim)
}

impl RunState {
    /// Allocate all scratch buffers for a given model configuration.
    pub fn new(c: &Config) -> Self {
        let kv_dim = c.kv_dim();
        let q_dim = c.q_dim();
        // Attention buffers follow `q_dim` (= dim for Llama-style, but distinct
        // when an explicit head_dim makes n_heads*head_dim ≠ dim, e.g. Gemma).
        let big = c.dim.max(q_dim);
        RunState {
            x: vec![0.0; c.dim],
            xb: vec![0.0; big],
            xb2: vec![0.0; big],
            hb: vec![0.0; c.hidden_dim],
            hb2: vec![0.0; c.hidden_dim],
            q: vec![0.0; q_dim],
            att: vec![0.0; c.n_heads * c.seq_len],
            logits: vec![0.0; c.vocab_size],
            kv: KvCache::new(c.n_layers, c.seq_len, kv_dim),
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

    /// The key cache `(n_layers, seq_len, kv_dim)`. Used by a backend that
    /// mirrors the KV cache on a device (the GPU/CUDA fused decode path).
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    pub(crate) fn key_cache(&self) -> &[f32] {
        self.kv.all_k()
    }

    /// The value cache `(n_layers, seq_len, kv_dim)`; see [`RunState::key_cache`].
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    pub(crate) fn value_cache(&self) -> &[f32] {
        self.kv.all_v()
    }

    /// Write the K/V a resident fused prefill computed on-device into the host
    /// KV cache, starting at flat offset `loff` (the layer's base). Used by the
    /// CUDA backend so subsequent CPU [`forward_step`] decode reads the prompt's
    /// cached keys/values. `k`/`v` are the layer's contiguous prompt rows.
    #[cfg(feature = "cuda")]
    pub(crate) fn store_prefill_kv(&mut self, loff: usize, k: &[f32], v: &[f32]) {
        self.kv.store_at(loff, k, v);
    }
}

/// Run one transformer step for `token` at position `pos`.
///
/// Reads/writes the KV cache in `state` and leaves the next-token logits in
/// `state.logits()`. Mirrors the reference llama2.c `forward()` op-for-op.
pub fn forward(model: &Model, state: &mut RunState, backend: &dyn Backend, token: usize, pos: usize) {
    forward_with(model, state, backend, token, pos, None);
}

/// [`forward`] with an optional control vector added to the residual after each
/// layer (the steering hook for [`crate::adapter::ControlVector`]).
pub fn forward_with(
    model: &Model,
    state: &mut RunState,
    backend: &dyn Backend,
    token: usize,
    pos: usize,
    control: Option<&ControlVector>,
) {
    let p = &model.config;
    let w = &model.weights;
    let dim = p.dim;
    let kv_dim = p.kv_dim();
    let head_size = p.head_size();
    let q_dim = p.q_dim();
    let bias = p.arch.has_qkv_bias();
    let sandwich = p.arch.sandwich_norm();
    let act = p.arch.ffn_activation();

    // Seed the residual stream with this token's (dequantized) embedding,
    // scaled by `embd_scale` (1.0 except Gemma).
    w.token_embedding_table.dequant_row(token, &mut state.x);
    if p.embd_scale != 1.0 {
        for v in state.x.iter_mut() {
            *v *= p.embd_scale;
        }
    }

    for layer in 0..p.n_layers {
        // --- Attention ---------------------------------------------------
        backend.rmsnorm(
            &mut state.xb[..dim],
            &state.x,
            slice(&w.rms_att_weight, layer, dim),
            p.rms_eps,
        );

        backend.matmul(&mut state.q[..q_dim], &state.xb[..dim], &w.wq[layer]);
        backend.matmul(state.kv.k_slot_mut(layer, pos), &state.xb[..dim], &w.wk[layer]);
        backend.matmul(state.kv.v_slot_mut(layer, pos), &state.xb[..dim], &w.wv[layer]);
        if bias {
            add_bias(&mut state.q[..q_dim], slice(&w.bq, layer, q_dim));
            add_bias(state.kv.k_slot_mut(layer, pos), slice(&w.bk, layer, kv_dim));
            add_bias(state.kv.v_slot_mut(layer, pos), slice(&w.bv, layer, kv_dim));
        }

        backend.rope(
            &mut state.q[..q_dim],
            state.kv.k_slot_mut(layer, pos),
            pos,
            head_size,
            kv_dim,
            &model.rope.inv_freq,
            model.rope.mscale,
        );

        backend.attention(
            &mut state.xb[..q_dim],
            &state.q[..q_dim],
            state.kv.layer_k(layer),
            state.kv.layer_v(layer),
            &mut state.att,
            pos,
            p.n_heads,
            p.n_kv_heads,
            head_size,
            p.seq_len,
            kv_dim,
            p.attn_logit_softcap,
        );

        backend.matmul(&mut state.xb2[..dim], &state.xb[..q_dim], &w.wo[layer]);
        if sandwich {
            // Gemma2 post-attention "sandwich" norm on the block output before
            // the residual add.
            backend.rmsnorm(
                &mut state.xb[..dim],
                &state.xb2[..dim],
                slice(&w.rms_attn_post, layer, dim),
                p.rms_eps,
            );
            backend.add(&mut state.x, &state.xb[..dim]);
        } else {
            backend.add(&mut state.x, &state.xb2[..dim]);
        }

        // --- Feed-forward ------------------------------------------------
        backend.rmsnorm(
            &mut state.xb[..dim],
            &state.x,
            slice(&w.rms_ffn_weight, layer, dim),
            p.rms_eps,
        );
        backend.matmul(&mut state.hb, &state.xb[..dim], &w.w1[layer]);
        backend.matmul(&mut state.hb2, &state.xb[..dim], &w.w3[layer]);
        match act {
            FfnActivation::SwiGlu => backend.swiglu(&mut state.hb, &state.hb2),
            FfnActivation::GeGlu => backend.geglu(&mut state.hb, &state.hb2),
        }
        backend.matmul(&mut state.xb[..dim], &state.hb, &w.w2[layer]);
        if sandwich {
            backend.rmsnorm(
                &mut state.xb2[..dim],
                &state.xb[..dim],
                slice(&w.rms_ffn_post, layer, dim),
                p.rms_eps,
            );
            backend.add(&mut state.x, &state.xb2[..dim]);
        } else {
            backend.add(&mut state.x, &state.xb[..dim]);
        }
        if let Some(cv) = control {
            cv.apply_layer(layer, &mut state.x);
        }
    }

    // Final norm (written into `xb` to avoid aliasing `x`) then classifier.
    backend.rmsnorm(&mut state.xb[..dim], &state.x, &w.rms_final_weight, p.rms_eps);
    backend.matmul(&mut state.logits, &state.xb[..dim], &w.wcls);
    softcap(&mut state.logits, p.final_logit_softcap);
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
pub fn forward_prefill<B: Backend + ?Sized>(
    model: &Model,
    state: &mut RunState,
    backend: &B,
    tokens: &[usize],
    pos_base: usize,
) {
    forward_prefill_with(model, state, backend, tokens, pos_base, None);
}

/// [`forward_prefill`] with an optional control vector applied per position after
/// each layer.
pub fn forward_prefill_with<B: Backend + ?Sized>(
    model: &Model,
    state: &mut RunState,
    backend: &B,
    tokens: &[usize],
    pos_base: usize,
    control: Option<&ControlVector>,
) {
    let dim = model.config.dim;
    let n = tokens.len();
    let x = prefill_residual(model, state, backend, tokens, pos_base, control);

    // Only the final position predicts the next token, so norm + classify just
    // that row (the classifier matmul is the widest — no need to run it n×).
    let w = &model.weights;
    let last = (n - 1) * dim;
    let mut xb_last = vec![0.0f32; dim];
    backend.rmsnorm(
        &mut xb_last,
        &x[last..last + dim],
        &w.rms_final_weight,
        model.config.rms_eps,
    );
    backend.matmul(&mut state.logits, &xb_last, &w.wcls);
    softcap(&mut state.logits, model.config.final_logit_softcap);
}

/// Run the batched layer stack over `tokens` (writing their KV at `pos_base`) and
/// return the row-major `(n, dim)` residual stream — the shared body of
/// [`forward_prefill`] and [`forward_embed`].
fn prefill_residual<B: Backend + ?Sized>(
    model: &Model,
    state: &mut RunState,
    backend: &B,
    tokens: &[usize],
    pos_base: usize,
    control: Option<&ControlVector>,
) -> Vec<f32> {
    let p = &model.config;
    let w = &model.weights;
    let dim = p.dim;
    let kv_dim = p.kv_dim();
    let head_size = p.head_size();
    let q_dim = p.q_dim();
    let hidden = p.hidden_dim;
    let n = tokens.len();
    let bias = p.arch.has_qkv_bias();
    let sandwich = p.arch.sandwich_norm();
    let act = p.arch.ffn_activation();

    // Row-major (n, *) scratch for the batch. Attention output `ao` is `(n,
    // q_dim)` and kept separate from the `(n, dim)` norm scratch `xb`, since
    // q_dim need not equal dim (Gemma).
    let mut x = vec![0.0f32; n * dim];
    let mut xb = vec![0.0f32; n * dim];
    let mut xb2 = vec![0.0f32; n * dim];
    let mut q = vec![0.0f32; n * q_dim];
    let mut ao = vec![0.0f32; n * q_dim];
    let mut hb = vec![0.0f32; n * hidden];
    let mut hb2 = vec![0.0f32; n * hidden];

    // Seed each row with its token's (dequantized) embedding, scaled (Gemma).
    for (r, &tok) in tokens.iter().enumerate() {
        w.token_embedding_table
            .dequant_row(tok, &mut x[r * dim..r * dim + dim]);
    }
    if p.embd_scale != 1.0 {
        for v in x.iter_mut() {
            *v *= p.embd_scale;
        }
    }

    for layer in 0..p.n_layers {
        // --- Attention --------------------------------------------------
        backend.rmsnorm_batch(&mut xb, &x, slice(&w.rms_att_weight, layer, dim), p.rms_eps, n);

        backend.matmul_batch(&mut q, &xb, &w.wq[layer], n);
        backend.matmul_batch(state.kv.k_span_mut(layer, pos_base, n), &xb, &w.wk[layer], n);
        backend.matmul_batch(state.kv.v_span_mut(layer, pos_base, n), &xb, &w.wv[layer], n);
        if bias {
            add_bias_batch(&mut q, slice(&w.bq, layer, q_dim), q_dim, n);
            add_bias_batch(state.kv.k_span_mut(layer, pos_base, n), slice(&w.bk, layer, kv_dim), kv_dim, n);
            add_bias_batch(state.kv.v_span_mut(layer, pos_base, n), slice(&w.bv, layer, kv_dim), kv_dim, n);
        }

        backend.rope_batch(
            &mut q,
            state.kv.k_span_mut(layer, pos_base, n),
            pos_base,
            n,
            head_size,
            kv_dim,
            q_dim,
            &model.rope.inv_freq,
            model.rope.mscale,
        );

        backend.attention_batch(
            &mut ao,
            &q,
            state.kv.layer_k(layer),
            state.kv.layer_v(layer),
            &mut state.att,
            pos_base,
            n,
            p.n_heads,
            p.n_kv_heads,
            head_size,
            p.seq_len,
            kv_dim,
            p.attn_logit_softcap,
        );

        backend.matmul_batch(&mut xb2, &ao, &w.wo[layer], n);
        if sandwich {
            backend.rmsnorm_batch(&mut xb, &xb2, slice(&w.rms_attn_post, layer, dim), p.rms_eps, n);
            backend.add(&mut x, &xb);
        } else {
            backend.add(&mut x, &xb2);
        }

        // --- Feed-forward -----------------------------------------------
        backend.rmsnorm_batch(&mut xb, &x, slice(&w.rms_ffn_weight, layer, dim), p.rms_eps, n);
        backend.matmul_batch(&mut hb, &xb, &w.w1[layer], n);
        backend.matmul_batch(&mut hb2, &xb, &w.w3[layer], n);
        match act {
            FfnActivation::SwiGlu => backend.swiglu(&mut hb, &hb2),
            FfnActivation::GeGlu => backend.geglu(&mut hb, &hb2),
        }
        backend.matmul_batch(&mut xb, &hb, &w.w2[layer], n);
        if sandwich {
            backend.rmsnorm_batch(&mut xb2, &xb, slice(&w.rms_ffn_post, layer, dim), p.rms_eps, n);
            backend.add(&mut x, &xb2);
        } else {
            backend.add(&mut x, &xb);
        }
        if let Some(cv) = control {
            for r in 0..n {
                cv.apply_layer(layer, &mut x[r * dim..(r + 1) * dim]);
            }
        }
    }
    x
}

/// How to pool per-position hidden states into one embedding vector.
#[derive(Clone, Copy, Debug)]
pub enum Pooling {
    /// Average over all positions.
    Mean,
    /// The last position's hidden state.
    Last,
    /// The first position's hidden state (`[CLS]`-style).
    Cls,
}

/// Embedding forward pass: run the layer stack over `tokens`, final-RMSNorm every
/// position, pool to one `dim`-vector, then optionally L2-normalize. Returns a
/// vector of length `config.dim`. The classifier (`wcls`) is skipped — embeddings
/// live in `n_embd = config.dim`. Uses the per-op batched ops, so any backend
/// works without touching the resident decode path.
pub fn forward_embed<B: Backend + ?Sized>(
    model: &Model,
    state: &mut RunState,
    backend: &B,
    tokens: &[usize],
    pooling: Pooling,
    normalize: bool,
) -> Vec<f32> {
    let p = &model.config;
    let w = &model.weights;
    let dim = p.dim;
    let n = tokens.len();

    let x = prefill_residual(model, state, backend, tokens, 0, None);
    // Final RMSNorm every row (the classifier is skipped for embeddings).
    let mut normed = vec![0.0f32; n * dim];
    backend.rmsnorm_batch(&mut normed, &x, &w.rms_final_weight, p.rms_eps, n);

    let mut emb = vec![0.0f32; dim];
    match pooling {
        Pooling::Mean => {
            for r in 0..n {
                for (e, &v) in emb.iter_mut().zip(&normed[r * dim..r * dim + dim]) {
                    *e += v;
                }
            }
            let inv = 1.0 / n as f32;
            for e in emb.iter_mut() {
                *e *= inv;
            }
        }
        Pooling::Last => emb.copy_from_slice(&normed[(n - 1) * dim..n * dim]),
        Pooling::Cls => emb.copy_from_slice(&normed[..dim]),
    }

    if normalize {
        let norm = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            let inv = 1.0 / norm;
            for e in emb.iter_mut() {
                *e *= inv;
            }
        }
    }
    emb
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
    sampler: &mut SamplerChain,
    prompt: &str,
    steps: usize,
    on_piece: impl FnMut(&[u8]),
) -> usize {
    let prompt_tokens = tokenizer.encode(prompt, tokenizer.add_bos(), false);
    generate_tokens(model, state, backend, tokenizer, sampler, &prompt_tokens, steps, on_piece)
}

/// Autoregressively generate from already-encoded `prompt_tokens`, streaming
/// decoded bytes to `on_piece`. The token-based core of [`generate`], for callers
/// that control BOS / special-token encoding themselves (e.g. chat templates).
#[allow(clippy::too_many_arguments)]
pub fn generate_tokens(
    model: &Model,
    state: &mut RunState,
    backend: &dyn Backend,
    tokenizer: &Tokenizer,
    sampler: &mut SamplerChain,
    prompt_tokens: &[usize],
    steps: usize,
    mut on_piece: impl FnMut(&[u8]),
) -> usize {
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
            prompt_tokens,
            steps,
            on_piece,
        );
    }

    let mut token = prompt_tokens[0];
    let mut pos = 0;
    let mut generated = 0;
    while pos < steps {
        backend.forward_step(model, state, token, pos);
        let next = if pos < prompt_tokens.len() - 1 {
            prompt_tokens[pos + 1]
        } else {
            sampler.sample(state.logits())
        };
        pos += 1;
        if next == 1 || tokenizer.is_eog(next) {
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
    sampler: &mut SamplerChain,
    prompt_tokens: &[usize],
    steps: usize,
    mut on_piece: impl FnMut(&[u8]),
) -> usize {
    let n = prompt_tokens.len();
    let mut generated = 0;

    // Dispatch through the trait method so a backend with a resident fused
    // prefill (CUDA) is used; the default delegates to `forward_prefill`.
    backend.forward_prefill(model, state, prompt_tokens, 0);

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
    let mut next = sampler.sample(state.logits());
    loop {
        pos += 1;
        if next == 1 || tokenizer.is_eog(next) {
            break; // BOS doubles as an end-of-sequence delimiter here.
        }
        on_piece(&tokenizer.decode(token, next));
        token = next;
        generated += 1;
        if pos >= steps {
            break;
        }
        backend.forward_step(model, state, token, pos);
        next = sampler.sample(state.logits());
    }

    generated
}

/// Borrow per-layer slice `layer` of a tensor laid out as `(n_layers, stride)`.
#[inline]
fn slice(tensor: &[f32], layer: usize, stride: usize) -> &[f32] {
    &tensor[layer * stride..layer * stride + stride]
}

/// Add a bias vector to a single activation row in place (`out[i] += bias[i]`).
#[inline]
fn add_bias(out: &mut [f32], bias: &[f32]) {
    for (o, &b) in out.iter_mut().zip(bias) {
        *o += b;
    }
}

/// Add a bias vector to every row of a `(rows, width)` batch in place.
#[inline]
fn add_bias_batch(out: &mut [f32], bias: &[f32], width: usize, rows: usize) {
    for r in 0..rows {
        add_bias(&mut out[r * width..r * width + width], bias);
    }
}

/// Gemma2 tanh logit softcapping (`x = cap · tanh(x / cap)`); a no-op when
/// `cap <= 0`, so non-Gemma archs pay nothing.
#[inline]
fn softcap(x: &mut [f32], cap: f32) {
    if cap > 0.0 {
        for v in x.iter_mut() {
            *v = cap * (*v / cap).tanh();
        }
    }
}

/// Split a fused row-major weight (Phi-3's `attn_qkv` / gate+up) into parts with
/// the given row counts, each sharing the source bytes (zero-copy for the usual
/// borrowed-from-mmap case). `row_counts` must sum to the source's row count.
fn split_rows<'a>(m: QMatrix<'a>, row_counts: &[usize]) -> Result<Vec<QMatrix<'a>>> {
    let total: usize = row_counts.iter().sum();
    if total != m.rows() {
        return Err(Error::Format(format!(
            "fused tensor has {} rows, split expects {total}",
            m.rows()
        )));
    }
    let cols = m.cols();
    let mut out = Vec::with_capacity(row_counts.len());
    // Per-part [lo, hi) row ranges from the cumulative counts.
    let ranges = || {
        let mut off = 0usize;
        row_counts.iter().map(move |&rc| {
            let r = (off, off + rc);
            off += rc;
            r
        })
    };
    match m {
        QMatrix::F32 { data, .. } => match data {
            Cow::Borrowed(d) => {
                for (lo, hi) in ranges() {
                    out.push(QMatrix::f32(Cow::Borrowed(&d[lo * cols..hi * cols]), hi - lo, cols)?);
                }
            }
            Cow::Owned(d) => {
                for (lo, hi) in ranges() {
                    out.push(QMatrix::f32(
                        Cow::Owned(d[lo * cols..hi * cols].to_vec()),
                        hi - lo,
                        cols,
                    )?);
                }
            }
        },
        QMatrix::Quant { ty, data, .. } => match data {
            Cow::Borrowed(d) => {
                for (lo, hi) in ranges() {
                    let (b0, b1) = (ty.bytes_for(lo * cols), ty.bytes_for(hi * cols));
                    out.push(QMatrix::quant(ty, Cow::Borrowed(&d[b0..b1]), hi - lo, cols)?);
                }
            }
            Cow::Owned(d) => {
                for (lo, hi) in ranges() {
                    let (b0, b1) = (ty.bytes_for(lo * cols), ty.bytes_for(hi * cols));
                    out.push(QMatrix::quant(ty, Cow::Owned(d[b0..b1].to_vec()), hi - lo, cols)?);
                }
            }
        },
    }
    Ok(out)
}

/// For output dim `i` (within a stack of heads each `head_size` wide, the first
/// `rot` dims rotary), the OLD index that the permuted NeoX→interleaved layout
/// puts at NEW position `i`. After this permutation the NORM rope kernel
/// (rotating pairs `2j,2j+1`) reproduces NeoX rope (rotating `j, j+rot/2`).
#[inline]
fn neox_src(i: usize, head_size: usize, rot: usize) -> usize {
    let base = (i / head_size) * head_size;
    let d = i % head_size;
    if d >= rot {
        base + d // unrotated trailing dims pass through
    } else if d.is_multiple_of(2) {
        base + d / 2
    } else {
        base + rot / 2 + d / 2
    }
}

/// Permute the output rows of a Q/K projection from NeoX (half-split) RoPE layout
/// to the interleaved layout the NORM rope kernel expects (the inverse of the
/// permutation llama.cpp's GGUF converter bakes into Llama weights). Materializes
/// an owned copy — q/k only, a fraction of the weights.
fn permute_neox<'a>(m: QMatrix<'_>, head_size: usize, rot: usize) -> Result<QMatrix<'a>> {
    let (rows, cols) = (m.rows(), m.cols());
    match m {
        QMatrix::F32 { data, .. } => {
            let mut out = vec![0.0f32; rows * cols];
            for new in 0..rows {
                let old = neox_src(new, head_size, rot);
                out[new * cols..new * cols + cols]
                    .copy_from_slice(&data[old * cols..old * cols + cols]);
            }
            QMatrix::f32(Cow::Owned(out), rows, cols)
        }
        QMatrix::Quant { ty, data, .. } => {
            let rb = ty.bytes_for(cols);
            let mut out = vec![0u8; rows * rb];
            for new in 0..rows {
                let old = neox_src(new, head_size, rot);
                out[new * rb..new * rb + rb].copy_from_slice(&data[old * rb..old * rb + rb]);
            }
            QMatrix::quant(ty, Cow::Owned(out), rows, cols)
        }
    }
}

/// Permute the per-layer Q/K bias vectors (flattened `(n_layers, width)`) the same
/// NeoX→interleaved way as [`permute_neox`]. No-op on an empty (bias-less) vec.
fn permute_neox_bias(bias: &mut [f32], n_layers: usize, width: usize, head_size: usize, rot: usize) {
    if bias.is_empty() {
        return;
    }
    for l in 0..n_layers {
        let seg = &mut bias[l * width..(l + 1) * width];
        let orig = seg.to_vec();
        for (i, s) in seg.iter_mut().enumerate() {
            *s = orig[neox_src(i, head_size, rot)];
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_offsets_match_flat_layout() {
        // 2 layers × 3 positions × kv_dim 4. Guards the extracted loff/slot math.
        let mut kv = KvCache::new(2, 3, 4);
        assert_eq!(kv.layer_base(0), 0);
        assert_eq!(kv.layer_base(1), 12);
        assert_eq!(kv.layer_k(1).len(), 12);
        assert_eq!(kv.layer_v(0).len(), 12);
        // pos 2 of layer 1 is rows 8..12 within that layer's window.
        kv.k_slot_mut(1, 2).copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&kv.layer_k(1)[8..12], [1.0, 2.0, 3.0, 4.0]);
        // a 2-row span from pos 0 of layer 0 is the first 8 elements.
        assert_eq!(kv.k_span_mut(0, 0, 2).len(), 8);
        kv.v_slot_mut(0, 1).copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(&kv.layer_v(0)[4..8], [5.0, 6.0, 7.0, 8.0]);
    }
}
