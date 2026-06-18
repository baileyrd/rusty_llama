//! Runtime LoRA adapters (Research/08 §3).
//!
//! A LoRA adapter holds low-rank pairs `(A, B)` for selected base weights; each
//! adds `scale · B·(A·x)` to that weight's matmul output, with
//! `scale = user_scale · alpha / rank`. [`LoraBackend`] wraps any [`Backend`] and
//! injects this delta after every base `matmul`, matched by the base weight's
//! [`QMatrix::data_ptr`] — so the resident-decode weight caches are untouched and
//! the per-op decode/prefill path applies LoRA uniformly on CPU, GPU, and CUDA.
//!
//! Because `LoraBackend` overrides only `matmul` (and routes `forward_step`
//! through the per-op [`crate::model::forward`]), binding an adapter transparently
//! falls back from the fused resident decode to the per-op path — the documented
//! Phase-1 behavior. Control vectors (a per-layer residual steer) are a planned
//! follow-up; they need a per-layer hook the per-matmul wrapper can't provide.

use std::collections::HashMap;

use crate::arch::LLAMA_NAMES;
use crate::backend::Backend;
use crate::error::{Error, Result};
use crate::gguf::Gguf;
use crate::model::{qmatrix_from_gguf, Model, RunState};
use crate::tensor::QMatrix;

/// One LoRA low-rank pair: `a` is `(rank, in)`, `b` is `(out, rank)`.
pub struct LoraPair<'a> {
    a: QMatrix<'a>,
    b: QMatrix<'a>,
}

/// A loaded LoRA adapter: pairs keyed by the base weight's [`QMatrix::data_ptr`].
pub struct LoraAdapter<'a> {
    pairs: HashMap<usize, LoraPair<'a>>,
    /// `alpha · user_scale`; the per-pair factor divides this by the pair's rank.
    scale_alpha: f32,
}

impl<'a> LoraAdapter<'a> {
    /// Build directly from `(base_ptr, pair)` entries (test/internal constructor).
    fn from_pairs(pairs: HashMap<usize, LoraPair<'a>>, alpha: f32, user_scale: f32) -> Self {
        LoraAdapter {
            pairs,
            scale_alpha: alpha * user_scale,
        }
    }

    /// Load a LoRA adapter GGUF, matching its `<base>.lora_a` / `<base>.lora_b`
    /// tensors to `model`'s base weights. Rejects non-LoRA adapters, dangling
    /// halves, and shape mismatches; skips pairs that target a weight we don't
    /// LoRA (norms, embeddings).
    pub fn from_gguf(gguf: &Gguf<'a>, model: &Model, user_scale: f32) -> Result<Self> {
        match gguf.meta_str("adapter.type") {
            Ok("lora") => {}
            Ok(other) => {
                return Err(Error::Format(format!(
                    "not a LoRA adapter (adapter.type = '{other}')"
                )))
            }
            Err(_) => return Err(Error::Format("missing adapter.type metadata".into())),
        }
        let alpha = gguf.meta_f32("adapter.lora.alpha").unwrap_or(1.0);
        let base_ptrs = base_weight_ptrs(model);

        let mut pairs: HashMap<usize, LoraPair<'a>> = HashMap::new();
        for info in &gguf.tensors {
            let Some(base) = info.name.strip_suffix(".lora_a") else {
                continue;
            };
            let b_name = format!("{base}.lora_b");
            if gguf.tensor(&b_name).is_none() {
                return Err(Error::Format(format!(
                    "LoRA tensor '{}' has no matching '{b_name}'",
                    info.name
                )));
            }
            let Some(&ptr) = base_ptrs.get(base) else {
                continue; // targets a non-LoRA weight (norm/embedding/unknown)
            };
            let a = qmatrix_from_gguf(gguf, &info.name)?;
            let b = qmatrix_from_gguf(gguf, &b_name)?;
            let base_mat = base_weight(model, base).expect("ptr resolved from this model");
            let (out, inf) = (base_mat.rows(), base_mat.cols());
            // A is (rank, in); B is (out, rank); ranks must agree.
            if a.cols() != inf || b.rows() != out || a.rows() != b.cols() {
                return Err(Error::Format(format!(
                    "LoRA '{base}' shape mismatch: A {}x{}, B {}x{}, base {out}x{inf}",
                    a.rows(),
                    a.cols(),
                    b.rows(),
                    b.cols()
                )));
            }
            pairs.insert(ptr, LoraPair { a, b });
        }
        if pairs.is_empty() {
            return Err(Error::Format(
                "LoRA adapter matched no base weights".into(),
            ));
        }
        Ok(Self::from_pairs(pairs, alpha, user_scale))
    }

    /// If `w` has a LoRA pair, add its delta to `out`: `out += s · B·(A·x)`.
    fn apply(&self, inner: &dyn Backend, out: &mut [f32], x: &[f32], w: &QMatrix) {
        let Some(pair) = self.pairs.get(&w.data_ptr()) else {
            return;
        };
        let rank = pair.a.rows();
        let mut r = vec![0.0f32; rank];
        inner.matmul(&mut r, x, &pair.a); // r = A·x
        let mut o = vec![0.0f32; pair.b.rows()];
        inner.matmul(&mut o, &r, &pair.b); // o = B·r
        let s = self.scale_alpha / rank as f32;
        for (oi, ov) in out.iter_mut().zip(&o) {
            *oi += s * *ov;
        }
    }
}

/// Map each LoRA-targetable base weight's GGUF name to its [`QMatrix::data_ptr`].
fn base_weight_ptrs(model: &Model) -> HashMap<String, usize> {
    let n = &LLAMA_NAMES;
    let w = &model.weights;
    let per_layer: [(&str, &Vec<QMatrix>); 7] = [
        (n.attn_q, &w.wq),
        (n.attn_k, &w.wk),
        (n.attn_v, &w.wv),
        (n.attn_output, &w.wo),
        (n.ffn_gate, &w.w1),
        (n.ffn_down, &w.w2),
        (n.ffn_up, &w.w3),
    ];
    let mut map = HashMap::new();
    for (suffix, mats) in per_layer {
        for (l, m) in mats.iter().enumerate() {
            map.insert(format!("blk.{l}.{suffix}"), m.data_ptr());
        }
    }
    map
}

/// Resolve a base GGUF tensor name to its [`QMatrix`] in `model` (shape check).
fn base_weight<'m>(model: &'m Model, name: &str) -> Option<&'m QMatrix<'m>> {
    let n = &LLAMA_NAMES;
    let w = &model.weights;
    let (l, suffix) = name.strip_prefix("blk.")?.split_once('.')?;
    let l: usize = l.parse().ok()?;
    let mats = match suffix {
        s if s == n.attn_q => &w.wq,
        s if s == n.attn_k => &w.wk,
        s if s == n.attn_v => &w.wv,
        s if s == n.attn_output => &w.wo,
        s if s == n.ffn_gate => &w.w1,
        s if s == n.ffn_down => &w.w2,
        s if s == n.ffn_up => &w.w3,
        _ => return None,
    };
    mats.get(l)
}

/// A [`Backend`] that injects a [`LoraAdapter`] after every base `matmul`. All
/// other ops delegate to `inner`; `forward_step` runs the per-op
/// [`crate::model::forward`] through `self`, so the resident fused decode is
/// bypassed (the per-op fallback under an adapter) while LoRA applies everywhere.
pub struct LoraBackend<'a> {
    inner: &'a dyn Backend,
    adapter: &'a LoraAdapter<'a>,
}

impl<'a> LoraBackend<'a> {
    pub fn new(inner: &'a dyn Backend, adapter: &'a LoraAdapter<'a>) -> Self {
        LoraBackend { inner, adapter }
    }
}

impl Backend for LoraBackend<'_> {
    fn rmsnorm(&self, out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
        self.inner.rmsnorm(out, x, weight, eps);
    }

    fn matmul(&self, out: &mut [f32], x: &[f32], w: &QMatrix) {
        self.inner.matmul(out, x, w);
        self.adapter.apply(self.inner, out, x, w);
    }

    #[allow(clippy::too_many_arguments)]
    fn rope(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        pos: usize,
        head_size: usize,
        kv_dim: usize,
        inv_freq: &[f32],
        mscale: f32,
    ) {
        self.inner
            .rope(q, k, pos, head_size, kv_dim, inv_freq, mscale);
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        out: &mut [f32],
        q: &[f32],
        key_cache: &[f32],
        value_cache: &[f32],
        att: &mut [f32],
        pos: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_size: usize,
        seq_len: usize,
        kv_dim: usize,
    ) {
        self.inner.attention(
            out, q, key_cache, value_cache, att, pos, n_heads, n_kv_heads, head_size, seq_len,
            kv_dim,
        );
    }

    fn swiglu(&self, hb: &mut [f32], hb2: &[f32]) {
        self.inner.swiglu(hb, hb2);
    }

    fn add(&self, out: &mut [f32], x: &[f32]) {
        self.inner.add(out, x);
    }

    fn forward_step(&self, model: &Model, state: &mut RunState, token: usize, pos: usize) {
        // Per-op path through `self` ⇒ LoRA applies and the resident fused decode
        // (an `inner` override) is bypassed.
        crate::model::forward(model, state, self, token, pos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use std::borrow::Cow;

    fn f32m(data: Vec<f32>, rows: usize, cols: usize) -> QMatrix<'static> {
        QMatrix::f32(Cow::Owned(data), rows, cols).unwrap()
    }

    #[test]
    fn lora_matches_external_merge() {
        // out = (W + s·B·A)·x via the wrapper must equal a densely-merged W'·x.
        let (inf, out, rank) = (4usize, 3usize, 2usize);
        let w: Vec<f32> = (0..out * inf).map(|i| i as f32 * 0.1 - 0.5).collect();
        let a: Vec<f32> = (0..rank * inf).map(|i| i as f32 * 0.05).collect();
        let b: Vec<f32> = (0..out * rank).map(|i| i as f32 * 0.07 - 0.1).collect();
        let x = vec![0.3f32, -0.2, 0.5, 0.1];
        let (alpha, scale) = (2.0f32, 1.0f32);
        let s = alpha * scale / rank as f32;

        let mut merged = w.clone();
        for o in 0..out {
            for i in 0..inf {
                let mut ba = 0.0f32;
                for r in 0..rank {
                    ba += b[o * rank + r] * a[r * inf + i];
                }
                merged[o * inf + i] += s * ba;
            }
        }

        let base = f32m(w.clone(), out, inf);
        let merged_m = f32m(merged, out, inf);
        let mut pairs = HashMap::new();
        pairs.insert(
            base.data_ptr(),
            LoraPair {
                a: f32m(a, rank, inf),
                b: f32m(b, out, rank),
            },
        );
        let adapter = LoraAdapter::from_pairs(pairs, alpha, scale);

        let cpu = CpuBackend::new();
        let lora = LoraBackend::new(&cpu, &adapter);
        let mut out_lora = vec![0.0f32; out];
        lora.matmul(&mut out_lora, &x, &base);
        let mut out_ref = vec![0.0f32; out];
        cpu.matmul(&mut out_ref, &x, &merged_m);
        for (l, m) in out_lora.iter().zip(&out_ref) {
            assert!((l - m).abs() < 1e-4, "lora {l} vs merged {m}");
        }
    }

    #[test]
    fn lora_scale_zero_is_byte_identical() {
        let base = f32m(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 2, 3);
        let mut pairs = HashMap::new();
        pairs.insert(
            base.data_ptr(),
            LoraPair {
                a: f32m(vec![1.0, 2.0, 3.0], 1, 3),
                b: f32m(vec![4.0, 5.0], 2, 1),
            },
        );
        let adapter = LoraAdapter::from_pairs(pairs, 8.0, 0.0); // user_scale = 0
        let cpu = CpuBackend::new();
        let lora = LoraBackend::new(&cpu, &adapter);
        let x = vec![1.0f32, -1.0, 0.5];
        let mut with = vec![0.0f32; 2];
        let mut without = vec![0.0f32; 2];
        lora.matmul(&mut with, &x, &base);
        cpu.matmul(&mut without, &x, &base);
        assert_eq!(with, without); // scale 0 ⇒ exact no-op
    }

    #[test]
    fn unmatched_weight_is_plain_matmul() {
        let base = f32m(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let other = f32m(vec![2.0, 0.0, 0.0, 2.0], 2, 2);
        let mut pairs = HashMap::new();
        pairs.insert(
            base.data_ptr(),
            LoraPair {
                a: f32m(vec![1.0, 1.0], 1, 2),
                b: f32m(vec![1.0, 1.0], 2, 1),
            },
        );
        let adapter = LoraAdapter::from_pairs(pairs, 1.0, 1.0);
        let cpu = CpuBackend::new();
        let lora = LoraBackend::new(&cpu, &adapter);
        let x = vec![3.0f32, 4.0];
        let mut lo = vec![0.0f32; 2];
        let mut co = vec![0.0f32; 2];
        lora.matmul(&mut lo, &x, &other); // 'other' has no pair
        cpu.matmul(&mut co, &x, &other);
        assert_eq!(lo, co);
    }
}
