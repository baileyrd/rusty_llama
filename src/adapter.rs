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

use crate::arch::TENSOR_NAMES;
use crate::backend::Backend;
use crate::error::{Error, Result};
use crate::gguf::Gguf;
use crate::model::{qmatrix_from_gguf, Model, RunState};
use crate::quant::GgmlType;
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
    let n = &TENSOR_NAMES;
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
    let n = &TENSOR_NAMES;
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

/// A control vector: a per-layer steering direction added to the residual stream
/// (`x += scale · dir[layer]`). `dirs[layer]` is `None` when a layer has no
/// steering (layer 0 conventionally never does). Applied by the per-op forward
/// via [`AdapterBackend`].
pub struct ControlVector {
    dirs: Vec<Option<Vec<f32>>>,
    scale: f32,
}

impl ControlVector {
    /// Build from per-layer directions (`dirs[layer]`, `None` = no steer).
    pub fn from_dirs(dirs: Vec<Option<Vec<f32>>>, scale: f32) -> Self {
        ControlVector { dirs, scale }
    }

    /// Load from a control-vector GGUF: one F32 `direction.{l}` tensor of length
    /// `dim` per steered (0-based) layer `l`. Validates lengths and layer indices.
    pub fn from_gguf(gguf: &Gguf, model: &Model, scale: f32) -> Result<Self> {
        let n_layers = model.config.n_layers;
        let dim = model.config.dim;
        let mut dirs: Vec<Option<Vec<f32>>> = vec![None; n_layers];
        let mut any = false;
        for info in &gguf.tensors {
            let Some(idx) = info.name.strip_prefix("direction.") else {
                continue;
            };
            let l: usize = idx.parse().map_err(|_| {
                Error::Format(format!("bad control-vector tensor name '{}'", info.name))
            })?;
            if l >= n_layers {
                return Err(Error::Format(format!(
                    "control-vector layer {l} out of range (n_layers = {n_layers})"
                )));
            }
            if info.ggml_type != GgmlType::F32 {
                return Err(Error::Format(format!(
                    "control-vector '{}' must be F32",
                    info.name
                )));
            }
            let bytes = gguf.tensor_bytes(info)?;
            if bytes.len() < dim * 4 {
                return Err(Error::Format(format!(
                    "control-vector '{}' too short: need {} bytes",
                    info.name,
                    dim * 4
                )));
            }
            let v: Vec<f32> = bytes[..dim * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            dirs[l] = Some(v);
            any = true;
        }
        if !any {
            return Err(Error::Format(
                "control vector has no direction.* tensors".into(),
            ));
        }
        Ok(ControlVector { dirs, scale })
    }

    /// Add this layer's scaled direction to one residual vector `x` (length dim).
    pub(crate) fn apply_layer(&self, layer: usize, x: &mut [f32]) {
        if let Some(Some(dir)) = self.dirs.get(layer) {
            for (xi, di) in x.iter_mut().zip(dir) {
                *xi += self.scale * *di;
            }
        }
    }
}

/// A [`Backend`] that injects optional LoRA + control-vector adapters. LoRA is
/// added after every base `matmul` (keyed by weight pointer); the control vector
/// is applied per layer inside the per-op forward. `forward_step`/`forward_prefill`
/// run the per-op path through `self`, so the resident fused decode is bypassed
/// (the per-op fallback under an adapter) and both adapters apply on every backend.
pub struct AdapterBackend<'a> {
    inner: &'a dyn Backend,
    lora: Option<&'a LoraAdapter<'a>>,
    control: Option<&'a ControlVector>,
}

impl<'a> AdapterBackend<'a> {
    pub fn new(
        inner: &'a dyn Backend,
        lora: Option<&'a LoraAdapter<'a>>,
        control: Option<&'a ControlVector>,
    ) -> Self {
        AdapterBackend { inner, lora, control }
    }
}

impl Backend for AdapterBackend<'_> {
    fn rmsnorm(&self, out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
        self.inner.rmsnorm(out, x, weight, eps);
    }

    fn matmul(&self, out: &mut [f32], x: &[f32], w: &QMatrix) {
        self.inner.matmul(out, x, w);
        if let Some(lora) = self.lora {
            lora.apply(self.inner, out, x, w);
        }
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
        logit_softcap: f32,
    ) {
        self.inner.attention(
            out, q, key_cache, value_cache, att, pos, n_heads, n_kv_heads, head_size, seq_len,
            kv_dim, logit_softcap,
        );
    }

    fn swiglu(&self, hb: &mut [f32], hb2: &[f32]) {
        self.inner.swiglu(hb, hb2);
    }

    fn add(&self, out: &mut [f32], x: &[f32]) {
        self.inner.add(out, x);
    }

    fn forward_step(&self, model: &Model, state: &mut RunState, token: usize, pos: usize) {
        // Per-op path through `self` ⇒ adapters apply and the resident fused
        // decode (an `inner` override) is bypassed.
        crate::model::forward_with(model, state, self, token, pos, self.control);
    }

    fn forward_prefill(
        &self,
        model: &Model,
        state: &mut RunState,
        tokens: &[usize],
        pos_base: usize,
    ) {
        crate::model::forward_prefill_with(model, state, self, tokens, pos_base, self.control);
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
        let lora = AdapterBackend::new(&cpu, Some(&adapter), None);
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
        let lora = AdapterBackend::new(&cpu, Some(&adapter), None);
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
        let lora = AdapterBackend::new(&cpu, Some(&adapter), None);
        let x = vec![3.0f32, 4.0];
        let mut lo = vec![0.0f32; 2];
        let mut co = vec![0.0f32; 2];
        lora.matmul(&mut lo, &x, &other); // 'other' has no pair
        cpu.matmul(&mut co, &x, &other);
        assert_eq!(lo, co);
    }

    #[test]
    fn control_vector_adds_scaled_direction() {
        let cv = ControlVector::from_dirs(vec![None, Some(vec![1.0, 2.0, 3.0])], 0.5);
        let mut x = vec![10.0f32, 10.0, 10.0];
        cv.apply_layer(0, &mut x); // layer 0 -> None -> unchanged
        assert_eq!(x, vec![10.0, 10.0, 10.0]);
        cv.apply_layer(1, &mut x); // += 0.5 * [1,2,3]
        assert_eq!(x, vec![10.5, 11.0, 11.5]);
        cv.apply_layer(9, &mut x); // out of range -> unchanged
        assert_eq!(x, vec![10.5, 11.0, 11.5]);
    }

    #[test]
    fn control_vector_scale_zero_is_noop() {
        let cv = ControlVector::from_dirs(vec![None, Some(vec![5.0, -5.0])], 0.0);
        let mut x = vec![1.0f32, 2.0];
        cv.apply_layer(1, &mut x);
        assert_eq!(x, vec![1.0, 2.0]);
    }
}
