//! Weight layout, run-time scratch buffers, and the transformer forward pass.

use std::borrow::Cow;
use std::cell::RefCell;
use std::rc::Rc;

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
    /// MoE router projection per layer, each `(n_expert, dim)`. Empty for a
    /// dense model; populated when `config.n_expert > 0` (Mixtral).
    pub ffn_gate_inp: Vec<QMatrix<'a>>,
    /// MoE routed-expert projections, `[layer][expert]`: gate `(n_ff_exp, dim)`,
    /// down `(dim, n_ff_exp)`, up `(n_ff_exp, dim)`. Empty for a dense model.
    /// When these are populated `w1`/`w2`/`w3` are empty, and vice-versa.
    pub w1_exp: Vec<Vec<QMatrix<'a>>>,
    pub w2_exp: Vec<Vec<QMatrix<'a>>>,
    pub w3_exp: Vec<Vec<QMatrix<'a>>>,
    /// Qwen2-MoE shared expert (per layer). `ffn_gate_inp_shexp` is a `(1, dim)`
    /// sigmoid gate; `w1/w2/w3_shexp` are the always-on SwiGLU FFN at `n_ff_shexp`.
    /// All empty unless the model has a shared expert (`config.n_ff_shexp > 0`).
    pub ffn_gate_inp_shexp: Vec<QMatrix<'a>>,
    pub w1_shexp: Vec<QMatrix<'a>>,
    pub w2_shexp: Vec<QMatrix<'a>>,
    pub w3_shexp: Vec<QMatrix<'a>>,
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
                ffn_gate_inp: Vec::new(),
                w1_exp: Vec::new(),
                w2_exp: Vec::new(),
                w3_exp: Vec::new(),
                ffn_gate_inp_shexp: Vec::new(),
                w1_shexp: Vec::new(),
                w2_shexp: Vec::new(),
                w3_shexp: Vec::new(),
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
        if !Arch::is_known(&arch_str) {
            // Permissive load (as Llama) is deliberate, but make it loud: an
            // unrecognized NeoX-RoPE arch gets NORM rope with no Q/K permute and
            // silently produces garbage. llama.cpp dispatches rope-type per arch.
            eprintln!(
                "rusty_llama: warning: unrecognized architecture '{arch_str}', loading as \
                 Llama — RoPE/normalization may be wrong unless this model is Llama-compatible"
            );
        }
        let names = arch.tensor_names();
        let key = |k: &str| format!("{arch_str}.{k}");

        let dim = gguf.meta_u64(&key("embedding_length"))? as usize;
        let n_layers = gguf.meta_u64(&key("block_count"))? as usize;
        let n_heads = gguf.meta_u64(&key("attention.head_count"))? as usize;
        let n_kv_heads = gguf
            .meta_u64(&key("attention.head_count_kv"))
            .unwrap_or(n_heads as u64) as usize;
        let hidden_dim = gguf.meta_u64(&key("feed_forward_length"))? as usize;
        // Mixture-of-experts counts (absent ⇒ 0 ⇒ a dense FFN).
        let n_expert = gguf.meta_u64(&key("expert_count")).unwrap_or(0) as usize;
        let n_expert_used = gguf.meta_u64(&key("expert_used_count")).unwrap_or(0) as usize;
        // Routed-expert FFN dim (Mixtral has no key ⇒ defaults to hidden_dim);
        // shared-expert FFN dim + presence (Qwen2-MoE only).
        let n_ff_exp = gguf
            .meta_u64(&key("expert_feed_forward_length"))
            .map(|v| v as usize)
            .unwrap_or(hidden_dim);
        let has_shared = n_expert > 0
            && gguf
                .tensor(&format!("blk.0.{}", names.ffn_gate_inp_shexp))
                .is_some();
        let n_ff_shexp = if has_shared {
            gguf.meta_u64(&key("expert_shared_feed_forward_length"))
                .map(|v| v as usize)
                .unwrap_or(hidden_dim)
        } else {
            0
        };
        // Top-k weight renormalization: Mixtral renormalizes, Qwen2-MoE does not.
        let expert_weights_norm = !matches!(arch, Arch::Qwen2Moe);
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

        // Gemma2 interleaved sliding-window attention: even layers attend only to
        // the last `sliding_window` keys. Read from GGUF `attention.sliding_window`;
        // when the key is absent it defaults to 4096 for Gemma2 (llama.cpp hardcodes
        // `n_swa = 4096` then overrides from the key) and 0 (full causal) otherwise.
        // See `Config::attn_window`.
        let sliding_window = gguf
            .meta_u64(&key("attention.sliding_window"))
            .map(|v| v as usize)
            .unwrap_or(if matches!(arch, Arch::Gemma2) { 4096 } else { 0 });

        // Gemma2 divides the attention scores by sqrt(query_pre_attn_scalar),
        // which differs from sqrt(head_size) on Gemma2-27B (144 vs 128). The
        // backends bake in 1/sqrt(head_size), so we pre-scale q by
        // sqrt(head_size / query_pre_attn_scalar); this collapses to 1.0 whenever
        // the key is absent or equals head_size (i.e. every other model). The key
        // is a float in some GGUFs and a uint in others, so accept either.
        let query_pre_attn_scalar = gguf
            .meta_f32(&key("attention.query_pre_attn_scalar"))
            .ok()
            .or_else(|| {
                gguf.meta_u64(&key("attention.query_pre_attn_scalar"))
                    .ok()
                    .map(|v| v as f32)
            })
            .filter(|&s| s > 0.0)
            .unwrap_or(head_size as f32);
        let attn_scale_correction =
            crate::config::attn_scale_correction(head_size, query_pre_attn_scalar);

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
            attn_scale_correction,
            sliding_window,
            n_expert,
            n_expert_used,
            n_ff_exp,
            n_ff_shexp,
            expert_weights_norm,
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

        // FFN weights. A dense model has per-layer gate/up/down (Phi-3's fused
        // gate+up is split here); an MoE model (`n_expert > 0`, e.g. Mixtral)
        // instead has a per-layer router plus stacked expert tensors — and no
        // dense FFN tensors, so `w1`/`w2`/`w3` stay empty.
        let (w1, w2, w3, ffn_gate_inp, w1_exp, w2_exp, w3_exp) = if n_expert > 0 {
            let router = layers(names.ffn_gate_inp)?;
            // Each `*_exps` tensor is a `(n_expert, rows, cols)` stack; split it
            // into the per-expert `(rows, cols)` matrices. Rows are expert-major
            // and contiguous, so the row-split byte math is exact.
            let split_exps = |suffix: &str, rows: usize| -> Result<Vec<Vec<QMatrix<'a>>>> {
                (0..n_layers)
                    .map(|i| {
                        let stack = qmatrix_from_gguf_3d(gguf, &format!("blk.{i}.{suffix}"))?;
                        split_rows(stack, &vec![rows; n_expert])
                    })
                    .collect()
            };
            let w1e = split_exps(names.ffn_gate_exps, n_ff_exp)?;
            let w3e = split_exps(names.ffn_up_exps, n_ff_exp)?;
            let w2e = split_exps(names.ffn_down_exps, dim)?;
            (Vec::new(), Vec::new(), Vec::new(), router, w1e, w2e, w3e)
        } else if arch.fused_gate_up() {
            // Phi-3: a fused gate+up tensor in `ffn_up` (rows = 2*hidden), split.
            let (mut w1, mut w3) = (Vec::new(), Vec::new());
            for i in 0..n_layers {
                let gate_up = qmatrix_from_gguf(gguf, &format!("blk.{i}.{}", names.ffn_up))?;
                let mut parts = split_rows(gate_up, &[hidden_dim, hidden_dim])?.into_iter();
                w1.push(parts.next().unwrap());
                w3.push(parts.next().unwrap());
            }
            let (e1, e2, e3) = (Vec::new(), Vec::new(), Vec::new());
            (w1, layers(names.ffn_down)?, w3, Vec::new(), e1, e2, e3)
        } else {
            let (e1, e2, e3) = (Vec::new(), Vec::new(), Vec::new());
            (
                layers(names.ffn_gate)?,
                layers(names.ffn_down)?,
                layers(names.ffn_up)?,
                Vec::new(),
                e1,
                e2,
                e3,
            )
        };

        // Qwen2-MoE shared expert: a per-layer sigmoid gate + an always-on SwiGLU
        // FFN, added to the routed output. Absent (empty) for Mixtral.
        let (ffn_gate_inp_shexp, w1_shexp, w2_shexp, w3_shexp) = if n_ff_shexp > 0 {
            (
                layers(names.ffn_gate_inp_shexp)?,
                layers(names.ffn_gate_shexp)?,
                layers(names.ffn_down_shexp)?,
                layers(names.ffn_up_shexp)?,
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
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
                w2,
                w3,
                bq,
                bk,
                bv,
                rms_attn_post,
                rms_ffn_post,
                ffn_gate_inp,
                w1_exp,
                w2_exp,
                w3_exp,
                ffn_gate_inp_shexp,
                w1_shexp,
                w2_shexp,
                w3_shexp,
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
        // A 1-D tensor is a single-row matrix (Qwen2-MoE's `ffn_gate_inp_shexp`,
        // a length-`dim` vector that projects to one sigmoid-gate logit).
        [c] => (*c as usize, 1),
        other => {
            return Err(Error::Format(format!(
                "tensor '{name}' must be 1-D or 2-D, has {} dims",
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

/// Build a [`QMatrix`] from a named 3-D GGUF tensor `(experts, rows, cols)` by
/// flattening the outer (expert) axis into the row axis: the result is an
/// `(experts * rows, cols)` matrix borrowing the same bytes. Expert `e`'s
/// `(rows, cols)` block is then a contiguous row range — exactly what
/// [`split_rows`] carves out. Used for MoE expert stacks (`ffn_*_exps.weight`).
fn qmatrix_from_gguf_3d<'a>(gguf: &Gguf<'a>, name: &str) -> Result<QMatrix<'a>> {
    let info = gguf
        .tensor(name)
        .ok_or_else(|| Error::Format(format!("GGUF missing tensor '{name}'")))?;
    let (cols, rows, experts) = match info.dims.as_slice() {
        [c, r, e] => (*c as usize, *r as usize, *e as usize),
        other => {
            return Err(Error::Format(format!(
                "MoE expert tensor '{name}' must be 3-D, has {} dims",
                other.len()
            )))
        }
    };
    let bytes = gguf.tensor_bytes(info)?;
    if info.ggml_type == GgmlType::F32 {
        if let Ok(f) = f32_slice(bytes, 0, experts * rows * cols) {
            return QMatrix::f32(Cow::Borrowed(f), experts * rows, cols);
        }
    }
    QMatrix::quant(info.ggml_type, Cow::Borrowed(bytes), experts * rows, cols)
}

/// Dequantize a named GGUF tensor to a fresh `f32` buffer.
fn deq_tensor(gguf: &Gguf, name: &str) -> Result<Vec<f32>> {
    let info = gguf
        .tensor(name)
        .ok_or_else(|| Error::Format(format!("GGUF missing tensor '{name}'")))?;
    dequantize(info.ggml_type, gguf.tensor_bytes(info)?, info.n_elements())
}

/// Growth increment for [`KvCache`]'s lazy capacity (in positions). Chosen to
/// keep relayout-copy events infrequent (a handful over a full context) while
/// still bounding the worst case (a short generation on a long-context model)
/// well below eagerly allocating the whole `max_seq_len` up front.
const KV_PAGE: usize = 256;

/// Shared KV-memory accounting across several pool-backed [`RunState`]s (one
/// pool per server process, since every slot serves the same model).
///
/// This tracks a **cell budget** (`n_layers * capacity * kv_dim`, doubled for
/// `k`+`v`, summed across every pool-backed [`KvCache`] currently checked
/// out) rather than handing out a literal block table: each `KvCache` still
/// owns one contiguous buffer, so `layer_k`/`layer_v`/[`RunState::key_cache`]/
/// [`RunState::value_cache`] keep working completely unchanged for every
/// backend — including the resident GPU/CUDA decode path, whose stride math
/// only cares about *this* cache's own current capacity, not the pool.
/// "Sharing the pool" means slots draw *budget* from a common ceiling and
/// give it back when their sequence finishes ([`RunState::release_kv`]), not
/// that they gather-read from another slot's physical pages.
///
/// Single-threaded (matches the server worker's ownership model — see
/// `server.rs`'s module doc: one thread owns every slot's `RunState`), hence
/// a plain `Rc<RefCell<..>>` rather than atomics or a mutex.
#[derive(Clone)]
pub struct KvPagePool {
    inner: Rc<RefCell<PoolInner>>,
}

struct PoolInner {
    budget_cells: usize,
    used_cells: usize,
}

impl KvPagePool {
    /// A pool with room for `budget_cells` total `f32` cells — see the
    /// struct doc for exactly what's counted.
    pub fn new(budget_cells: usize) -> Self {
        KvPagePool {
            inner: Rc::new(RefCell::new(PoolInner {
                budget_cells,
                used_cells: 0,
            })),
        }
    }

    /// Cells currently charged against the budget by every checked-out,
    /// pool-backed `KvCache`.
    pub fn used_cells(&self) -> usize {
        self.inner.borrow().used_cells
    }

    /// The total budget this pool was constructed with.
    pub fn budget_cells(&self) -> usize {
        self.inner.borrow().budget_cells
    }

    /// Whether there's currently headroom to admit a new sequence. Meant to
    /// be checked at admission time (before prefill starts) — once a
    /// sequence is running, refusing to grow its KV would corrupt output, so
    /// this is backpressure on *new* work, not a hard mid-flight cap. A
    /// pool can therefore transiently read `used_cells() > budget_cells()`
    /// while already-admitted sequences keep growing past the ceiling — an
    /// accepted limitation (no preemption), not a bug.
    pub fn has_headroom(&self) -> bool {
        let p = self.inner.borrow();
        p.used_cells < p.budget_cells
    }

    fn charge(&self, delta: usize) {
        self.inner.borrow_mut().used_cells += delta;
    }

    fn release(&self, amount: usize) {
        let mut p = self.inner.borrow_mut();
        p.used_cells = p.used_cells.saturating_sub(amount);
    }
}

/// The transformer's key/value cache.
///
/// PHASE-0 INVARIANT: holds exactly **one** sequence; the layout is the flat
/// `(n_layers, cap, kv_dim)` buffer, `cap` a lazily-grown capacity (<=
/// `max_seq_len`, see [`KvCache::ensure_capacity`]) rather than eagerly
/// `max_seq_len` from construction — the memory-efficiency half of Phase 4
/// (paged, multi-sequence KV — see `docs/Architecture/plans/`). Optionally
/// draws that growth from a shared [`KvPagePool`] budget (the multi-sequence
/// half — see the server's continuous-batching slots) instead of growing
/// unbounded on its own. All slot/window arithmetic lives here and nowhere
/// else, so `forward`/`forward_prefill` and the `Backend` trait don't change.
/// Returned slices are the exact ranges the old inline `loff`/`kv_at`
/// arithmetic produced, just against `cap` instead of a fixed `seq_len` —
/// every per-op `Backend::attention`/`attention_batch` impl already only
/// reads `key_cache[..(pos+1)*kv_dim]`, so a shorter (but `>= pos+1`) window
/// is correct for them unchanged. The two backends that mirror the *whole*
/// flat buffer onto a device for resident decode ([`RunState::key_cache`]/
/// [`RunState::value_cache`], read together with [`RunState::kv_capacity`]
/// for the current stride) are the ones that do need to know `cap`
/// explicitly — see `backend/cuda.rs`/`backend/gpu.rs`'s prefill-KV upload,
/// which computes its per-layer offset from [`KvCache::capacity`] rather
/// than the model's `seq_len`.
pub(crate) struct KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    n_layers: usize,
    /// Current per-layer capacity (positions); grows in [`KV_PAGE`]
    /// increments, capped at `max_seq_len`.
    cap: usize,
    /// The model's context window — `cap` never exceeds this.
    max_seq_len: usize,
    kv_dim: usize,
    /// Shared budget this cache draws from, if any — see [`KvPagePool`].
    /// `None` for a [`RunState::new`] (unpooled) instance: unlimited private
    /// growth, exactly the behavior before pooling existed.
    pool: Option<KvPagePool>,
    /// This cache's current charge against `pool` (cells across *both* `k`
    /// and `v`, all layers) — tracked so [`KvCache::shrink_to_initial`] can
    /// give back exactly what it took, without the pool needing to know
    /// anything about individual caches.
    charged: usize,
}

impl KvCache {
    /// Allocate a zeroed cache for `n_layers` layers, capacity starting at
    /// `min(KV_PAGE, max_seq_len)` and growing on demand up to `max_seq_len`.
    /// Charges its initial allocation against `pool` if given.
    fn new(n_layers: usize, max_seq_len: usize, kv_dim: usize, pool: Option<KvPagePool>) -> Self {
        let cap = KV_PAGE.min(max_seq_len.max(1));
        let cells = n_layers * cap * kv_dim;
        let charged = 2 * cells;
        if let Some(p) = &pool {
            p.charge(charged);
        }
        KvCache {
            k: vec![0.0; cells],
            v: vec![0.0; cells],
            n_layers,
            cap,
            max_seq_len,
            kv_dim,
            pool,
            charged,
        }
    }

    /// Current per-layer capacity (positions) — the stride a resident backend
    /// must use when slicing [`RunState::key_cache`]/[`RunState::value_cache`].
    /// Gated on `test` in addition to `gpu`/`cuda` (unlike `key_cache`/
    /// `value_cache`) so it's checkable from plain unit tests without those
    /// features enabled.
    #[cfg(any(test, feature = "gpu", feature = "cuda"))]
    pub(crate) fn capacity(&self) -> usize {
        self.cap
    }

    /// Ensure the cache has room through position `up_to` (inclusive),
    /// growing (and relaying out every layer's block into the new stride) in
    /// [`KV_PAGE`] increments if not. A no-op once `cap` reaches
    /// `max_seq_len` — the caller (`forward`/`forward_prefill`) is
    /// responsible for not writing past the model's context window. Charges
    /// the growth delta against `pool`, if any — see [`KvPagePool`]'s doc on
    /// why this is backpressure on *new* admissions, not a hard mid-flight
    /// cap: growth here always succeeds (refusing would corrupt an
    /// in-progress sequence), it just keeps the shared accounting honest for
    /// other slots' admission checks.
    fn ensure_capacity(&mut self, up_to: usize) {
        if up_to < self.cap || self.cap >= self.max_seq_len {
            return;
        }
        let new_cap = (up_to + 1).next_multiple_of(KV_PAGE).min(self.max_seq_len);
        if new_cap <= self.cap {
            return;
        }
        let block = self.cap * self.kv_dim;
        let new_block = new_cap * self.kv_dim;
        let mut new_k = vec![0.0; self.n_layers * new_block];
        let mut new_v = vec![0.0; self.n_layers * new_block];
        for layer in 0..self.n_layers {
            let old_off = layer * block;
            let new_off = layer * new_block;
            new_k[new_off..new_off + block].copy_from_slice(&self.k[old_off..old_off + block]);
            new_v[new_off..new_off + block].copy_from_slice(&self.v[old_off..old_off + block]);
        }
        let delta = 2 * self.n_layers * (new_block - block);
        if let Some(p) = &self.pool {
            p.charge(delta);
        }
        self.charged += delta;
        self.k = new_k;
        self.v = new_v;
        self.cap = new_cap;
    }

    /// Release any grown capacity back down to the initial `min(KV_PAGE,
    /// max_seq_len)` size, giving back whatever budget it held to `pool` (if
    /// any). Call when this cache's sequence finishes, so the next one
    /// admitted into the same slot starts small again and — for a
    /// pool-backed cache — other slots see the freed headroom immediately.
    /// A no-op if already at (or under) the initial size.
    fn shrink_to_initial(&mut self) {
        let initial_cap = KV_PAGE.min(self.max_seq_len.max(1));
        if self.cap <= initial_cap {
            return;
        }
        let cells = self.n_layers * initial_cap * self.kv_dim;
        let new_charge = 2 * cells;
        if let Some(p) = &self.pool {
            p.release(self.charged - new_charge);
        }
        self.charged = new_charge;
        self.k = vec![0.0; cells];
        self.v = vec![0.0; cells];
        self.cap = initial_cap;
    }

    /// Flat base offset of layer `layer`'s `(cap, kv_dim)` block.
    #[inline]
    fn layer_base(&self, layer: usize) -> usize {
        layer * self.cap * self.kv_dim
    }

    /// Layer `layer`'s full `(cap, kv_dim)` key window (what attention reads;
    /// `cap >= pos+1` for any `pos` already written, so this is always long
    /// enough — see the struct doc).
    fn layer_k(&self, layer: usize) -> &[f32] {
        let b = self.layer_base(layer);
        &self.k[b..b + self.cap * self.kv_dim]
    }

    /// Layer `layer`'s full `(cap, kv_dim)` value window.
    fn layer_v(&self, layer: usize) -> &[f32] {
        let b = self.layer_base(layer);
        &self.v[b..b + self.cap * self.kv_dim]
    }

    /// The single `kv_dim` key slot at `pos` (what the per-token K matmul writes).
    fn k_slot_mut(&mut self, layer: usize, pos: usize) -> &mut [f32] {
        self.ensure_capacity(pos);
        let b = self.layer_base(layer) + pos * self.kv_dim;
        &mut self.k[b..b + self.kv_dim]
    }

    /// The single `kv_dim` value slot at `pos`.
    fn v_slot_mut(&mut self, layer: usize, pos: usize) -> &mut [f32] {
        self.ensure_capacity(pos);
        let b = self.layer_base(layer) + pos * self.kv_dim;
        &mut self.v[b..b + self.kv_dim]
    }

    /// Both the `kv_dim` key and value slots at `pos` as one disjoint borrow of
    /// the separate `k`/`v` buffers — lets the q/k/v projections share a single
    /// quantized activation through [`crate::backend::Backend::matmul_shared`].
    fn kv_slots_mut(&mut self, layer: usize, pos: usize) -> (&mut [f32], &mut [f32]) {
        self.ensure_capacity(pos);
        let b = self.layer_base(layer) + pos * self.kv_dim;
        let kd = self.kv_dim;
        (&mut self.k[b..b + kd], &mut self.v[b..b + kd])
    }

    /// The contiguous `n`-row key span from `pos_base` (what prefill writes).
    fn k_span_mut(&mut self, layer: usize, pos_base: usize, n: usize) -> &mut [f32] {
        self.ensure_capacity(pos_base + n - 1);
        let b = self.layer_base(layer) + pos_base * self.kv_dim;
        &mut self.k[b..b + n * self.kv_dim]
    }

    /// The contiguous `n`-row value span from `pos_base`.
    fn v_span_mut(&mut self, layer: usize, pos_base: usize, n: usize) -> &mut [f32] {
        self.ensure_capacity(pos_base + n - 1);
        let b = self.layer_base(layer) + pos_base * self.kv_dim;
        &mut self.v[b..b + n * self.kv_dim]
    }

    /// The whole flat key buffer (a device-resident backend uploads it; pair
    /// with [`KvCache::capacity`] for the current per-layer stride).
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn all_k(&self) -> &[f32] {
        &self.k
    }

    /// The whole flat value buffer; see [`KvCache::all_k`].
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn all_v(&self) -> &[f32] {
        &self.v
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
    router: Vec<f32>,      // MoE router logits/probs (n_expert; empty if dense)
    moe_tmp: Vec<f32>,     // MoE per-expert FFN output (dim; empty if dense)
    kv: KvCache, // single sequence; flat (n_layers, seq_len, kv_dim)
}

impl RunState {
    /// Allocate all scratch buffers for a given model configuration. The KV
    /// cache grows privately and unbounded (up to `c.seq_len`) — the CLI's
    /// constructor, unaffected by [`RunState::with_pool`].
    pub fn new(c: &Config) -> Self {
        Self::build(c, KvCache::new(c.n_layers, c.seq_len, c.kv_dim(), None))
    }

    /// Like [`RunState::new`], but draws its KV memory from a shared
    /// [`KvPagePool`] instead of growing unbounded on its own — for the
    /// server's continuous-batching slots, so many slots share one memory
    /// ceiling. Pair with [`RunState::release_kv`] when this instance's
    /// sequence finishes, so its capacity (and the pool budget it holds) is
    /// given back before the slot admits the next one.
    pub fn with_pool(c: &Config, pool: &KvPagePool) -> Self {
        Self::build(
            c,
            KvCache::new(c.n_layers, c.seq_len, c.kv_dim(), Some(pool.clone())),
        )
    }

    fn build(c: &Config, kv: KvCache) -> Self {
        let q_dim = c.q_dim();
        // Attention buffers follow `q_dim` (= dim for Llama-style, but distinct
        // when an explicit head_dim makes n_heads*head_dim ≠ dim, e.g. Gemma).
        let big = c.dim.max(q_dim);
        RunState {
            x: vec![0.0; c.dim],
            xb: vec![0.0; big],
            xb2: vec![0.0; big],
            // FFN scratch must cover the largest FFN width: dense `hidden_dim`,
            // routed expert `n_ff_exp`, or shared expert `n_ff_shexp` (MoE).
            hb: vec![0.0; c.hidden_dim.max(c.n_ff_exp).max(c.n_ff_shexp)],
            hb2: vec![0.0; c.hidden_dim.max(c.n_ff_exp).max(c.n_ff_shexp)],
            q: vec![0.0; q_dim],
            att: vec![0.0; c.n_heads * c.seq_len],
            logits: vec![0.0; c.vocab_size],
            router: vec![0.0; c.n_expert],
            moe_tmp: vec![0.0; if c.n_expert > 0 { c.dim } else { 0 }],
            kv,
        }
    }

    /// Release this instance's KV memory back down to its idle size, giving
    /// back any [`KvPagePool`] budget it held (a no-op beyond the shrink
    /// itself for a [`RunState::new`]/unpooled instance — there's no pool to
    /// notify). Call when its sequence finishes, before the slot admits a
    /// new one.
    pub fn release_kv(&mut self) {
        self.kv.shrink_to_initial();
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

    /// Current per-layer capacity (positions) of [`RunState::key_cache`]/
    /// [`RunState::value_cache`] — the stride a resident backend must use to
    /// compute per-layer offsets into them (may be less than the model's
    /// `seq_len` — see [`KvCache`]'s struct doc).
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    pub(crate) fn kv_capacity(&self) -> usize {
        self.kv.capacity()
    }
}

/// Continuous-batching decode scratch: batched activation buffers reused across
/// steps for up to `cap` concurrent sequences. The per-sequence KV lives in each
/// slot's own [`RunState`] (so admitting a prompt can reuse [`forward_prefill`]);
/// one [`Batch::decode_step`] runs a single batched forward over the active slots
/// — the matmuls batch (N batch-1 GEMVs become one N-row GEMM, the throughput
/// win), while rope, the KV write, and attention loop per slot (each at its own
/// position, against its own cache).
pub struct Batch {
    cap: usize,
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    q: Vec<f32>,
    kbuf: Vec<f32>,
    vbuf: Vec<f32>,
    ao: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    att: Vec<f32>,
    logits: Vec<f32>,
    router: Vec<f32>,  // MoE router logits/probs, one row (n_expert; empty if dense)
    moe_tmp: Vec<f32>, // MoE per-expert FFN output, one row (dim; empty if dense)
}

impl Batch {
    /// Allocate batched scratch for up to `cap` concurrent sequences.
    pub fn new(c: &Config, cap: usize) -> Self {
        let (dim, q_dim, kv_dim, hidden, vocab) =
            (c.dim, c.q_dim(), c.kv_dim(), c.hidden_dim, c.vocab_size);
        // FFN scratch width covers dense `hidden_dim` and the routed/shared MoE
        // expert widths (the MoE FFN reuses one row of `hb`/`hb2`).
        let hb_w = hidden.max(c.n_ff_exp).max(c.n_ff_shexp);
        Batch {
            cap,
            x: vec![0.0; cap * dim],
            xb: vec![0.0; cap * dim],
            xb2: vec![0.0; cap * dim],
            q: vec![0.0; cap * q_dim],
            kbuf: vec![0.0; cap * kv_dim],
            vbuf: vec![0.0; cap * kv_dim],
            ao: vec![0.0; cap * q_dim],
            hb: vec![0.0; cap * hb_w],
            hb2: vec![0.0; cap * hb_w],
            att: vec![0.0; c.n_heads * c.seq_len],
            logits: vec![0.0; cap * vocab],
            router: vec![0.0; c.n_expert],
            moe_tmp: vec![0.0; if c.n_expert > 0 { dim } else { 0 }],
        }
    }

    /// Maximum number of concurrent sequences this batch can decode.
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Run one batched decode step over the active slots. Row `r` decodes
    /// `tokens[r]` at `positions[r]` for `states[slots[r]]`: its K/V is appended
    /// to that slot's own cache and it attends only its own cache. Returns the
    /// row-major `(n, vocab)` logits — row `r` is `slots[r]`'s next-token logits.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_step(
        &mut self,
        model: &Model,
        backend: &dyn Backend,
        states: &mut [RunState],
        slots: &[usize],
        tokens: &[usize],
        positions: &[usize],
    ) -> &[f32] {
        let p = &model.config;
        let w = &model.weights;
        let n = slots.len();
        let dim = p.dim;
        let kv_dim = p.kv_dim();
        let head_size = p.head_size();
        let q_dim = p.q_dim();
        let hidden = p.hidden_dim;
        let vocab = p.vocab_size;
        let eps = p.rms_eps;
        let bias = p.arch.has_qkv_bias();
        let sandwich = p.arch.sandwich_norm();
        let act = p.arch.ffn_activation();
        let is_moe = p.n_expert > 0;
        let ff_w = hidden.max(p.n_ff_exp).max(p.n_ff_shexp);
        debug_assert!(n <= self.cap && tokens.len() == n && positions.len() == n);

        // Seed each row with its token's (dequantized) embedding, scaled (Gemma).
        for (r, &tok) in tokens.iter().enumerate() {
            w.token_embedding_table
                .dequant_row(tok, &mut self.x[r * dim..r * dim + dim]);
        }
        if p.embd_scale != 1.0 {
            for v in self.x[..n * dim].iter_mut() {
                *v *= p.embd_scale;
            }
        }

        for layer in 0..p.n_layers {
            // --- Attention: matmuls batch; rope/KV-write/attention loop per slot.
            backend.rmsnorm_batch(&mut self.xb[..n * dim], &self.x[..n * dim], slice(&w.rms_att_weight, layer, dim), eps, n);
            backend.matmul_batch(&mut self.q[..n * q_dim], &self.xb[..n * dim], &w.wq[layer], n);
            backend.matmul_batch(&mut self.kbuf[..n * kv_dim], &self.xb[..n * dim], &w.wk[layer], n);
            backend.matmul_batch(&mut self.vbuf[..n * kv_dim], &self.xb[..n * dim], &w.wv[layer], n);
            if bias {
                add_bias_batch(&mut self.q[..n * q_dim], slice(&w.bq, layer, q_dim), q_dim, n);
                add_bias_batch(&mut self.kbuf[..n * kv_dim], slice(&w.bk, layer, kv_dim), kv_dim, n);
                add_bias_batch(&mut self.vbuf[..n * kv_dim], slice(&w.bv, layer, kv_dim), kv_dim, n);
            }
            for r in 0..n {
                let pos = positions[r];
                backend.rope(
                    &mut self.q[r * q_dim..r * q_dim + q_dim],
                    &mut self.kbuf[r * kv_dim..r * kv_dim + kv_dim],
                    pos,
                    head_size,
                    kv_dim,
                    &model.rope.inv_freq,
                    model.rope.mscale,
                );
                let kv = &mut states[slots[r]].kv;
                kv.k_slot_mut(layer, pos)
                    .copy_from_slice(&self.kbuf[r * kv_dim..r * kv_dim + kv_dim]);
                kv.v_slot_mut(layer, pos)
                    .copy_from_slice(&self.vbuf[r * kv_dim..r * kv_dim + kv_dim]);
                // Gemma2-27B: bend the score scale from 1/sqrt(head_size) (baked
                // into every backend) to 1/sqrt(query_pre_attn_scalar) by pre-scaling
                // q. No-op for every model where the two coincide (correction == 1.0).
                if p.attn_scale_correction != 1.0 {
                    for v in self.q[r * q_dim..r * q_dim + q_dim].iter_mut() {
                        *v *= p.attn_scale_correction;
                    }
                }
                backend.attention(
                    &mut self.ao[r * q_dim..r * q_dim + q_dim],
                    &self.q[r * q_dim..r * q_dim + q_dim],
                    kv.layer_k(layer),
                    kv.layer_v(layer),
                    &mut self.att,
                    pos,
                    p.n_heads,
                    p.n_kv_heads,
                    head_size,
                    p.seq_len,
                    kv_dim,
                    p.attn_logit_softcap,
                    p.attn_window(layer),
                );
            }
            backend.matmul_batch(&mut self.xb2[..n * dim], &self.ao[..n * q_dim], &w.wo[layer], n);
            if sandwich {
                backend.rmsnorm_batch(&mut self.xb[..n * dim], &self.xb2[..n * dim], slice(&w.rms_attn_post, layer, dim), eps, n);
                backend.add(&mut self.x[..n * dim], &self.xb[..n * dim]);
            } else {
                backend.add(&mut self.x[..n * dim], &self.xb2[..n * dim]);
            }

            // --- Feed-forward ---
            backend.rmsnorm_batch(&mut self.xb[..n * dim], &self.x[..n * dim], slice(&w.rms_ffn_weight, layer, dim), eps, n);
            if is_moe {
                // Per-token routing: the FFN runs row-by-row (no batch GEMM).
                for r in 0..n {
                    moe_ffn_row(
                        model,
                        backend,
                        layer,
                        &self.xb[r * dim..r * dim + dim],
                        &mut self.xb2[r * dim..r * dim + dim],
                        &mut self.hb[..ff_w],
                        &mut self.hb2[..ff_w],
                        &mut self.moe_tmp,
                        &mut self.router,
                    );
                }
            } else {
                backend.matmul_batch(&mut self.hb[..n * hidden], &self.xb[..n * dim], &w.w1[layer], n);
                backend.matmul_batch(&mut self.hb2[..n * hidden], &self.xb[..n * dim], &w.w3[layer], n);
                match act {
                    FfnActivation::SwiGlu => backend.swiglu(&mut self.hb[..n * hidden], &self.hb2[..n * hidden]),
                    FfnActivation::GeGlu => backend.geglu(&mut self.hb[..n * hidden], &self.hb2[..n * hidden]),
                }
                backend.matmul_batch(&mut self.xb2[..n * dim], &self.hb[..n * hidden], &w.w2[layer], n);
            }
            // The FFN output now lives in `xb2` for both the dense and MoE paths.
            if sandwich {
                backend.rmsnorm_batch(&mut self.xb[..n * dim], &self.xb2[..n * dim], slice(&w.rms_ffn_post, layer, dim), eps, n);
                backend.add(&mut self.x[..n * dim], &self.xb[..n * dim]);
            } else {
                backend.add(&mut self.x[..n * dim], &self.xb2[..n * dim]);
            }
        }

        backend.rmsnorm_batch(&mut self.xb[..n * dim], &self.x[..n * dim], &w.rms_final_weight, eps, n);
        backend.matmul_batch(&mut self.logits[..n * vocab], &self.xb[..n * dim], &w.wcls, n);
        softcap(&mut self.logits[..n * vocab], p.final_logit_softcap);
        &self.logits[..n * vocab]
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
    let is_moe = p.n_expert > 0;

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

        {
            // q/k/v all project the attention-norm'd `xb`: quantize it once.
            let (k_slot, v_slot) = state.kv.kv_slots_mut(layer, pos);
            let mut qkv: [&mut [f32]; 3] = [&mut state.q[..q_dim], k_slot, v_slot];
            backend.matmul_shared(
                &mut qkv,
                &state.xb[..dim],
                &[&w.wq[layer], &w.wk[layer], &w.wv[layer]],
            );
        }
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

        // Gemma2-27B attention-scale correction (see decode_step); no-op == 1.0.
        if p.attn_scale_correction != 1.0 {
            for v in state.q[..q_dim].iter_mut() {
                *v *= p.attn_scale_correction;
            }
        }
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
            p.attn_window(layer),
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
        if is_moe {
            moe_ffn_row(
                model,
                backend,
                layer,
                &state.xb[..dim],
                &mut state.xb2[..dim],
                &mut state.hb,
                &mut state.hb2,
                &mut state.moe_tmp,
                &mut state.router,
            );
        } else {
            {
                // gate (w1) and up (w3) both project the ffn-norm'd `xb`.
                let mut gu: [&mut [f32]; 2] = [&mut state.hb, &mut state.hb2];
                backend.matmul_shared(&mut gu, &state.xb[..dim], &[&w.w1[layer], &w.w3[layer]]);
            }
            match act {
                FfnActivation::SwiGlu => backend.swiglu(&mut state.hb, &state.hb2),
                FfnActivation::GeGlu => backend.geglu(&mut state.hb, &state.hb2),
            }
            backend.matmul(&mut state.xb2[..dim], &state.hb, &w.w2[layer]);
        }
        // The FFN output now lives in `xb2` for both the dense and MoE paths.
        if sandwich {
            backend.rmsnorm(
                &mut state.xb[..dim],
                &state.xb2[..dim],
                slice(&w.rms_ffn_post, layer, dim),
                p.rms_eps,
            );
            backend.add(&mut state.x, &state.xb[..dim]);
        } else {
            backend.add(&mut state.x, &state.xb2[..dim]);
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
    let is_moe = p.n_expert > 0;
    // FFN scratch width: dense `hidden`, or the larger routed/shared MoE widths.
    let ff_w = hidden.max(p.n_ff_exp).max(p.n_ff_shexp);

    // Row-major (n, *) scratch for the batch. Attention output `ao` is `(n,
    // q_dim)` and kept separate from the `(n, dim)` norm scratch `xb`, since
    // q_dim need not equal dim (Gemma).
    let mut x = vec![0.0f32; n * dim];
    let mut xb = vec![0.0f32; n * dim];
    let mut xb2 = vec![0.0f32; n * dim];
    let mut q = vec![0.0f32; n * q_dim];
    let mut ao = vec![0.0f32; n * q_dim];
    let mut hb = vec![0.0f32; n * ff_w];
    let mut hb2 = vec![0.0f32; n * ff_w];
    // MoE routing scratch (one row at a time); empty for a dense model.
    let mut router = vec![0.0f32; if is_moe { p.n_expert } else { 0 }];
    let mut moe_tmp = vec![0.0f32; if is_moe { dim } else { 0 }];

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

        // Gemma2-27B attention-scale correction (see decode_step). Scales the whole
        // prefill query block; no-op when correction == 1.0.
        if p.attn_scale_correction != 1.0 {
            for v in q.iter_mut() {
                *v *= p.attn_scale_correction;
            }
        }
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
            p.attn_window(layer),
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
        if is_moe {
            // Routing is per-token, so the FFN runs row-by-row (no batch GEMM).
            for r in 0..n {
                moe_ffn_row(
                    model,
                    backend,
                    layer,
                    &xb[r * dim..r * dim + dim],
                    &mut xb2[r * dim..r * dim + dim],
                    &mut hb[..ff_w],
                    &mut hb2[..ff_w],
                    &mut moe_tmp,
                    &mut router,
                );
            }
        } else {
            backend.matmul_batch(&mut hb, &xb, &w.w1[layer], n);
            backend.matmul_batch(&mut hb2, &xb, &w.w3[layer], n);
            match act {
                FfnActivation::SwiGlu => backend.swiglu(&mut hb, &hb2),
                FfnActivation::GeGlu => backend.geglu(&mut hb, &hb2),
            }
            backend.matmul_batch(&mut xb2, &hb, &w.w2[layer], n);
        }
        // The FFN output now lives in `xb2` for both the dense and MoE paths.
        if sandwich {
            backend.rmsnorm_batch(&mut xb, &xb2, slice(&w.rms_ffn_post, layer, dim), p.rms_eps, n);
            backend.add(&mut x, &xb);
        } else {
            backend.add(&mut x, &xb2);
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
        if next == 1 || tokenizer.is_eog(next) {
            return generated; // BOS/EOG mid-prompt (matches the generation loop's break)
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

/// One token's mixture-of-experts FFN: route `xn` (the FFN-normed input row,
/// length `dim`) through the top-k experts and write the result into `out`
/// (length `dim`).
///
/// Mirrors llama.cpp's `build_moe_ffn`: softmax over **all** experts, pick the
/// top `n_expert_used` (ties → lower index), and accumulate each selected
/// expert's SwiGLU/GeGLU output scaled by its weight. When
/// `config.expert_weights_norm` (Mixtral) the selected weights are renormalized
/// to sum to 1; when not (Qwen2-MoE) the raw softmax probabilities are used.
///
/// When the model has a shared expert (`config.n_ff_shexp > 0`, Qwen2-MoE) its
/// always-on SwiGLU FFN, gated by `sigmoid(ffn_gate_inp_shexp · xn)`, is added to
/// the routed output. The dense FFN path never calls this.
///
/// Scratch (caller-owned, contents clobbered): `g`/`u` length
/// `≥ max(n_ff_exp, n_ff_shexp)` (sliced internally), `tmp` length `dim`,
/// `router` length `n_expert`.
#[allow(clippy::too_many_arguments)]
fn moe_ffn_row<B: Backend + ?Sized>(
    model: &Model,
    backend: &B,
    layer: usize,
    xn: &[f32],
    out: &mut [f32],
    g: &mut [f32],
    u: &mut [f32],
    tmp: &mut [f32],
    router: &mut [f32],
) {
    let p = &model.config;
    let w = &model.weights;
    let act = p.arch.ffn_activation();
    let nf = p.n_ff_exp;

    // Router logits → softmax probabilities over all experts.
    backend.matmul(router, xn, &w.ffn_gate_inp[layer]);
    crate::math::softmax(router);

    // Select the top-k experts by probability (ties → lower index).
    let mut sel: Vec<(usize, f32)> = Vec::with_capacity(p.n_expert_used);
    for _ in 0..p.n_expert_used {
        let mut best = 0usize;
        let mut best_p = f32::NEG_INFINITY;
        for (e, &prob) in router.iter().enumerate() {
            if prob > best_p && !sel.iter().any(|&(s, _)| s == e) {
                best_p = prob;
                best = e;
            }
        }
        sel.push((best, best_p));
    }
    // Mixtral renormalizes the selected weights to sum to 1; Qwen2-MoE does not.
    let wsum: f32 = if p.expert_weights_norm {
        sel.iter().map(|&(_, prob)| prob).sum()
    } else {
        1.0
    };

    // Weighted sum of the selected routed experts' FFN outputs.
    out.fill(0.0);
    for (e, prob) in sel {
        let weight = prob / wsum;
        backend.matmul(&mut g[..nf], xn, &w.w1_exp[layer][e]);
        backend.matmul(&mut u[..nf], xn, &w.w3_exp[layer][e]);
        match act {
            FfnActivation::SwiGlu => backend.swiglu(&mut g[..nf], &u[..nf]),
            FfnActivation::GeGlu => backend.geglu(&mut g[..nf], &u[..nf]),
        }
        backend.matmul(tmp, &g[..nf], &w.w2_exp[layer][e]);
        for (o, &t) in out.iter_mut().zip(tmp.iter()) {
            *o += weight * t;
        }
    }

    // Qwen2-MoE shared expert: out += sigmoid(gate · xn) · SwiGLU_shared(xn).
    if p.n_ff_shexp > 0 {
        let ns = p.n_ff_shexp;
        let mut gate = [0.0f32];
        backend.matmul(&mut gate, xn, &w.ffn_gate_inp_shexp[layer]);
        let sig = 1.0 / (1.0 + (-gate[0]).exp());
        backend.matmul(&mut g[..ns], xn, &w.w1_shexp[layer]);
        backend.matmul(&mut u[..ns], xn, &w.w3_shexp[layer]);
        match act {
            FfnActivation::SwiGlu => backend.swiglu(&mut g[..ns], &u[..ns]),
            FfnActivation::GeGlu => backend.geglu(&mut g[..ns], &u[..ns]),
        }
        backend.matmul(tmp, &g[..ns], &w.w2_shexp[layer]);
        for (o, &t) in out.iter_mut().zip(tmp.iter()) {
            *o += sig * t;
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
        let mut kv = KvCache::new(2, 3, 4, None);
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

    #[test]
    fn kv_cache_short_sequence_uses_less_than_max_seq_len_capacity() {
        // The memory win: a context window far larger than what's actually
        // used shouldn't cost that much memory up front.
        let kv = KvCache::new(4, 8192, 16, None);
        assert_eq!(kv.capacity(), KV_PAGE);
        assert!(kv.capacity() < 8192);
    }

    #[test]
    fn kv_cache_grows_lazily_and_preserves_existing_data() {
        let mut kv = KvCache::new(2, 1000, 4, None);
        assert_eq!(kv.capacity(), KV_PAGE);

        // Write within the initial page — no growth yet.
        kv.k_slot_mut(0, 0).copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        kv.k_slot_mut(1, 0).copy_from_slice(&[9.0, 9.0, 9.0, 9.0]);
        assert_eq!(kv.capacity(), KV_PAGE);

        // Writing past the initial page triggers growth (a relayout copy).
        kv.k_slot_mut(0, 300).copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
        assert!(kv.capacity() > KV_PAGE, "should have grown past {KV_PAGE}");
        assert!(kv.capacity() >= 301);
        assert_eq!(kv.capacity() % KV_PAGE, 0);

        // Data written before the growth event must survive the relayout,
        // in both layers (proves the per-layer copy uses the right strides).
        assert_eq!(&kv.layer_k(0)[0..4], [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&kv.layer_k(1)[0..4], [9.0, 9.0, 9.0, 9.0]);
        assert_eq!(&kv.layer_k(0)[300 * 4..300 * 4 + 4], [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn kv_cache_capacity_never_exceeds_max_seq_len() {
        let mut kv = KvCache::new(1, 10, 4, None);
        assert_eq!(kv.capacity(), 10); // KV_PAGE=256 clamped down to max_seq_len=10
        kv.k_slot_mut(0, 9).copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(kv.capacity(), 10);
        assert_eq!(&kv.layer_k(0)[36..40], [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn kv_pool_shares_budget_across_caches_and_reclaims_on_shrink() {
        let pool = KvPagePool::new(1_000_000); // generous budget
        let mut a = KvCache::new(2, 1000, 4, Some(pool.clone()));
        let b_initial = 2 * 2 * KV_PAGE * 4; // n_layers=2, kv_dim=4, *2 for k+v
        assert_eq!(pool.used_cells(), b_initial); // just `a` so far
        let _b = KvCache::new(2, 1000, 4, Some(pool.clone()));
        assert_eq!(pool.used_cells(), 2 * b_initial); // `a` + `b`, independently charged

        // Growing `a` charges the pool on top of both caches' initial charge.
        a.k_slot_mut(0, 300).copy_from_slice(&[0.0; 4]);
        let after_a_grow = pool.used_cells();
        assert!(after_a_grow > 2 * b_initial);

        // Shrinking `a` back gives its growth back to the shared pool — `b`
        // is untouched, so total usage returns to exactly the two initial
        // charges.
        a.shrink_to_initial();
        assert_eq!(pool.used_cells(), 2 * b_initial);
    }

    #[test]
    fn kv_pool_has_headroom_reflects_budget() {
        // Room for exactly one cache's initial charge (n_layers=2, kv_dim=4).
        let budget = 2 * 2 * KV_PAGE * 4;
        let pool = KvPagePool::new(budget);
        assert!(pool.has_headroom());
        let mut a = KvCache::new(2, 1000, 4, Some(pool.clone()));
        assert!(
            !pool.has_headroom(),
            "budget exactly exhausted by one cache's initial charge"
        );
        // Growth is never blocked mid-flight (see KvPagePool's doc) — it can
        // push used past budget, and has_headroom then correctly stays false
        // rather than somehow reporting "true" once already over.
        a.k_slot_mut(0, 300).copy_from_slice(&[0.0; 4]);
        assert!(pool.used_cells() > budget);
        assert!(!pool.has_headroom());
        // Releasing brings usage back down to exactly the budget (no longer
        // over it), though has_headroom() still reports false — the initial
        // charge alone consumes the whole budget, so there was never true
        // spare room to "return to".
        a.shrink_to_initial();
        assert_eq!(pool.used_cells(), budget);
        assert!(!pool.has_headroom());
    }

    #[test]
    fn run_state_with_pool_charges_and_release_kv_reclaims() {
        let c = Config {
            dim: 8,
            hidden_dim: 16,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            vocab_size: 16,
            seq_len: 1000,
            ..Default::default()
        };
        let pool = KvPagePool::new(1_000_000);
        let mut state = RunState::with_pool(&c, &pool);
        let initial_used = pool.used_cells();
        assert!(initial_used > 0);

        // Drive real growth through the same path `forward`/`forward_prefill`
        // use, not a private KvCache method directly. kv_dim = n_kv_heads *
        // (dim / n_heads) = 2 * 4 = 8.
        state.kv.k_slot_mut(0, 500).copy_from_slice(&[0.0; 8]);
        assert!(pool.used_cells() > initial_used);

        state.release_kv();
        assert_eq!(pool.used_cells(), initial_used);
    }

    fn moe_model_bytes(n_expert: usize, n_expert_used: usize) -> Vec<u8> {
        let c = Config {
            dim: 8,
            hidden_dim: 16,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            vocab_size: 16,
            seq_len: 8,
            ..Default::default()
        };
        crate::dummy::synthetic_gguf_moe(&c, n_expert, n_expert_used)
    }

    #[test]
    fn moe_loads_router_and_experts() {
        let bytes = moe_model_bytes(4, 2);
        let gguf = Gguf::parse(&bytes).unwrap();
        let m = Model::from_gguf(&gguf).unwrap();
        assert_eq!(m.config.n_expert, 4);
        assert_eq!(m.config.n_expert_used, 2);
        // A router and an expert list per layer; experts have the FFN shapes.
        assert_eq!(m.weights.ffn_gate_inp.len(), m.config.n_layers);
        assert_eq!(m.weights.ffn_gate_inp[0].rows(), 4);
        assert_eq!(m.weights.ffn_gate_inp[0].cols(), m.config.dim);
        assert_eq!(m.weights.w1_exp[0].len(), 4);
        assert_eq!(m.weights.w1_exp[0][0].rows(), m.config.hidden_dim);
        assert_eq!(m.weights.w1_exp[0][0].cols(), m.config.dim);
        assert_eq!(m.weights.w2_exp[0][0].rows(), m.config.dim);
        assert_eq!(m.weights.w2_exp[0][0].cols(), m.config.hidden_dim);
        // Dense FFN tensors are absent for an MoE model.
        assert!(m.weights.w1.is_empty() && m.weights.w2.is_empty() && m.weights.w3.is_empty());
    }

    #[test]
    fn moe_ffn_row_matches_weighted_expert_sum() {
        // n_expert_used == n_expert: top-k is every expert, so the renormalized
        // weights are exactly the softmax probabilities — checks routing softmax,
        // expert loading/indexing, and the weighted accumulation.
        let bytes = moe_model_bytes(4, 4);
        let gguf = Gguf::parse(&bytes).unwrap();
        let m = Model::from_gguf(&gguf).unwrap();
        let backend = crate::backend::CpuBackend::new();
        let (dim, hidden, ne) = (m.config.dim, m.config.hidden_dim, m.config.n_expert);
        let xn: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.3 - 1.0).sin()).collect();

        let mut out = vec![0.0; dim];
        let (mut g, mut u, mut tmp, mut router) =
            (vec![0.0; hidden], vec![0.0; hidden], vec![0.0; dim], vec![0.0; ne]);
        moe_ffn_row(&m, &backend, 0, &xn, &mut out, &mut g, &mut u, &mut tmp, &mut router);

        let mut probs = vec![0.0; ne];
        backend.matmul(&mut probs, &xn, &m.weights.ffn_gate_inp[0]);
        crate::math::softmax(&mut probs);
        let mut want = vec![0.0; dim];
        for (e, &prob) in probs.iter().enumerate() {
            let (mut ge, mut ue) = (vec![0.0; hidden], vec![0.0; hidden]);
            backend.matmul(&mut ge, &xn, &m.weights.w1_exp[0][e]);
            backend.matmul(&mut ue, &xn, &m.weights.w3_exp[0][e]);
            backend.swiglu(&mut ge, &ue);
            let mut de = vec![0.0; dim];
            backend.matmul(&mut de, &ge, &m.weights.w2_exp[0][e]);
            for (w, &d) in want.iter_mut().zip(&de) {
                *w += prob * d;
            }
        }
        for (o, w) in out.iter().zip(&want) {
            assert!((o - w).abs() < 1e-5, "moe out {o} vs ref {w}");
        }
    }

    #[test]
    fn moe_top1_selects_argmax_expert() {
        // With n_expert_used == 1 the single weight renormalizes to 1.0, so the
        // FFN output is exactly the argmax expert's output.
        let bytes = moe_model_bytes(4, 1);
        let gguf = Gguf::parse(&bytes).unwrap();
        let m = Model::from_gguf(&gguf).unwrap();
        let backend = crate::backend::CpuBackend::new();
        let (dim, hidden, ne) = (m.config.dim, m.config.hidden_dim, m.config.n_expert);
        let xn: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.7 + 0.2).cos()).collect();

        let mut out = vec![0.0; dim];
        let (mut g, mut u, mut tmp, mut router) =
            (vec![0.0; hidden], vec![0.0; hidden], vec![0.0; dim], vec![0.0; ne]);
        moe_ffn_row(&m, &backend, 0, &xn, &mut out, &mut g, &mut u, &mut tmp, &mut router);

        let mut logits = vec![0.0; ne];
        backend.matmul(&mut logits, &xn, &m.weights.ffn_gate_inp[0]);
        let best = (0..ne)
            .max_by(|&a, &b| logits[a].partial_cmp(&logits[b]).unwrap())
            .unwrap();
        let (mut ge, mut ue) = (vec![0.0; hidden], vec![0.0; hidden]);
        backend.matmul(&mut ge, &xn, &m.weights.w1_exp[0][best]);
        backend.matmul(&mut ue, &xn, &m.weights.w3_exp[0][best]);
        backend.swiglu(&mut ge, &ue);
        let mut want = vec![0.0; dim];
        backend.matmul(&mut want, &ge, &m.weights.w2_exp[0][best]);
        for (o, w) in out.iter().zip(&want) {
            assert!((o - w).abs() < 1e-6, "top1 out {o} vs expert {w}");
        }
    }

    #[test]
    fn qwen2moe_ffn_row_routed_no_renorm_plus_shared() {
        // Qwen2-MoE: routed top-k with RAW softmax probs (no renormalization)
        // plus a sigmoid-gated shared SwiGLU expert at its own `n_ff_shexp`.
        let c = Config {
            dim: 8,
            hidden_dim: 16,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            vocab_size: 16,
            seq_len: 8,
            ..Default::default()
        };
        let bytes = crate::dummy::synthetic_gguf_qwen2moe(&c, 4, 2, 12, 20);
        let gguf = Gguf::parse(&bytes).unwrap();
        let m = Model::from_gguf(&gguf).unwrap();
        assert_eq!(m.config.arch, Arch::Qwen2Moe);
        assert_eq!((m.config.n_ff_exp, m.config.n_ff_shexp), (12, 20));
        assert!(!m.config.expert_weights_norm);
        // Shapes: routed experts at n_ff_exp, shared at n_ff_shexp, gate is (1,dim).
        assert_eq!(m.weights.w1_exp[0][0].rows(), 12);
        assert_eq!(m.weights.w1_shexp[0].rows(), 20);
        assert_eq!(m.weights.w2_shexp[0].cols(), 20);
        assert_eq!(m.weights.ffn_gate_inp_shexp[0].rows(), 1);
        assert_eq!(m.weights.ffn_gate_inp_shexp[0].cols(), 8);

        let backend = crate::backend::CpuBackend::new();
        let (dim, ne) = (m.config.dim, m.config.n_expert);
        let (nfe, nfs) = (m.config.n_ff_exp, m.config.n_ff_shexp);
        let xn: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.4 - 0.7).sin()).collect();

        let mut out = vec![0.0; dim];
        let w = nfe.max(nfs);
        let (mut g, mut u, mut tmp, mut router) =
            (vec![0.0; w], vec![0.0; w], vec![0.0; dim], vec![0.0; ne]);
        moe_ffn_row(&m, &backend, 0, &xn, &mut out, &mut g, &mut u, &mut tmp, &mut router);

        // Reference. Routed: softmax over all experts, top-k by prob (ties → lower
        // index), weights = RAW probs (NOT renormalized — the Qwen2-MoE delta).
        let mut probs = vec![0.0; ne];
        backend.matmul(&mut probs, &xn, &m.weights.ffn_gate_inp[0]);
        crate::math::softmax(&mut probs);
        let mut idx: Vec<usize> = (0..ne).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap().then(a.cmp(&b)));
        let mut want = vec![0.0; dim];
        for &e in idx.iter().take(m.config.n_expert_used) {
            let (mut ge, mut ue) = (vec![0.0; nfe], vec![0.0; nfe]);
            backend.matmul(&mut ge, &xn, &m.weights.w1_exp[0][e]);
            backend.matmul(&mut ue, &xn, &m.weights.w3_exp[0][e]);
            backend.swiglu(&mut ge, &ue);
            let mut de = vec![0.0; dim];
            backend.matmul(&mut de, &ge, &m.weights.w2_exp[0][e]);
            for (wv, &d) in want.iter_mut().zip(&de) {
                *wv += probs[e] * d;
            }
        }
        // Shared: sigmoid(gate · xn) * SwiGLU_shared(xn), added to the routed sum.
        let mut gate = [0.0f32];
        backend.matmul(&mut gate, &xn, &m.weights.ffn_gate_inp_shexp[0]);
        let sig = 1.0 / (1.0 + (-gate[0]).exp());
        let (mut gs, mut us) = (vec![0.0; nfs], vec![0.0; nfs]);
        backend.matmul(&mut gs, &xn, &m.weights.w1_shexp[0]);
        backend.matmul(&mut us, &xn, &m.weights.w3_shexp[0]);
        backend.swiglu(&mut gs, &us);
        let mut ds = vec![0.0; dim];
        backend.matmul(&mut ds, &gs, &m.weights.w2_shexp[0]);
        for (wv, &d) in want.iter_mut().zip(&ds) {
            *wv += sig * d;
        }

        for (o, wv) in out.iter().zip(&want) {
            assert!((o - wv).abs() < 1e-5, "qwen2moe out {o} vs ref {wv}");
        }
    }
}
