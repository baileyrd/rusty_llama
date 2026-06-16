//! Multi-threaded CPU backend.
//!
//! The two parallelized kernels ([`CpuBackend::matmul`] and
//! [`CpuBackend::attention`]) split their outer dimension across the rayon
//! thread pool. The inner dot-product loops are written so the compiler can
//! autovectorize them — build with `-C target-cpu=native` (see
//! `.cargo/config.toml`) to get AVX2 / AVX-512 / FMA.

use rayon::prelude::*;

use crate::backend::Backend;
use crate::math::{silu, softmax};
use crate::quant::{
    dequant_block, has_int8_path, quantize_activation_q8, vec_dot_q4_0, vec_dot_q8_0, GgmlType,
    Q8Activation, MAX_BLOCK,
};
use crate::tensor::QMatrix;

/// Dot product of one quantized weight row (decompressed block by block) with
/// the activation `x`. Avoids materializing the full f32 row.
fn dot_quant_row(ty: GgmlType, row: &[u8], x: &[f32]) -> f32 {
    let bs = ty.block_size();
    let ts = ty.type_size();
    let mut buf = [0.0f32; MAX_BLOCK];
    let mut acc = 0.0f32;
    for (b, chunk) in row.chunks_exact(ts).enumerate() {
        let block = &mut buf[..bs];
        dequant_block(ty, chunk, block);
        let xs = &x[b * bs..b * bs + bs];
        for (&w, &xi) in block.iter().zip(xs) {
            acc += w * xi;
        }
    }
    acc
}

/// A stateless f32 CPU backend.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

impl CpuBackend {
    /// Create a new CPU backend (it carries no state).
    pub fn new() -> Self {
        CpuBackend
    }
}

impl Backend for CpuBackend {
    fn rmsnorm(&self, out: &mut [f32], x: &[f32], weight: &[f32]) {
        debug_assert_eq!(out.len(), x.len());
        debug_assert_eq!(out.len(), weight.len());
        let n = x.len() as f32;
        let ss = x.iter().map(|&v| v * v).sum::<f32>() / n + 1e-5;
        let inv = 1.0 / ss.sqrt();
        for ((o, &xi), &wi) in out.iter_mut().zip(x).zip(weight) {
            *o = xi * inv * wi;
        }
    }

    fn matmul(&self, out: &mut [f32], x: &[f32], w: &QMatrix) {
        debug_assert_eq!(out.len(), w.rows());
        debug_assert_eq!(x.len(), w.cols());
        match w {
            QMatrix::F32 { data, cols, .. } => {
                // Fast path: a contiguous f32 row dot product autovectorizes.
                out.par_iter_mut().enumerate().for_each(|(i, o)| {
                    let row = &data[i * cols..i * cols + cols];
                    *o = row.iter().zip(x).map(|(&wij, &xj)| wij * xj).sum();
                });
            }
            QMatrix::Quant { ty, data, cols, .. } => {
                let rb = ty.bytes_for(*cols);
                if has_int8_path(*ty) {
                    // Fast path: quantize the activation once, then do integer
                    // block dot products per row.
                    let act = quantize_activation_q8(x);
                    let dot: fn(&[u8], &Q8Activation) -> f32 = match ty {
                        GgmlType::Q8_0 => vec_dot_q8_0,
                        GgmlType::Q4_0 => vec_dot_q4_0,
                        _ => unreachable!(),
                    };
                    out.par_iter_mut().enumerate().for_each(|(i, o)| {
                        *o = dot(&data[i * rb..i * rb + rb], &act);
                    });
                } else {
                    // k-quants (Q4_K/Q6_K) and F16: dequantize each block to f32.
                    out.par_iter_mut().enumerate().for_each(|(i, o)| {
                        *o = dot_quant_row(*ty, &data[i * rb..i * rb + rb], x);
                    });
                }
            }
        }
    }

    fn rope(&self, q: &mut [f32], k: &mut [f32], pos: usize, head_size: usize, kv_dim: usize) {
        // Rotate consecutive (even, odd) pairs within each head by an angle
        // that grows with position and shrinks with the pair's index in the
        // head. Pairs below `kv_dim` rotate both q and k; the rest (extra query
        // heads under grouped-query attention) rotate q only.
        let dim = q.len();
        let mut i = 0;
        while i < dim {
            let head_dim = i % head_size;
            let freq = 1.0 / 10000f32.powf(head_dim as f32 / head_size as f32);
            let val = pos as f32 * freq;
            let (fcr, fci) = (val.cos(), val.sin());

            let (q0, q1) = (q[i], q[i + 1]);
            q[i] = q0 * fcr - q1 * fci;
            q[i + 1] = q0 * fci + q1 * fcr;

            if i < kv_dim {
                let (k0, k1) = (k[i], k[i + 1]);
                k[i] = k0 * fcr - k1 * fci;
                k[i + 1] = k0 * fci + k1 * fcr;
            }
            i += 2;
        }
    }

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
        let kv_mul = n_heads / n_kv_heads;
        let scale = 1.0 / (head_size as f32).sqrt();

        // One independent task per head. `out` and `att` are sliced into
        // disjoint per-head chunks so the closures never alias.
        out.par_chunks_mut(head_size)
            .zip(att.par_chunks_mut(seq_len))
            .enumerate()
            .for_each(|(h, (out_h, att_h))| {
                let q_h = &q[h * head_size..h * head_size + head_size];
                // Which key/value head this query head attends to.
                let kv_off = (h / kv_mul) * head_size;

                // Scaled dot-product scores against every key so far.
                for (t, score) in att_h[..=pos].iter_mut().enumerate() {
                    let base = t * kv_dim + kv_off;
                    let k = &key_cache[base..base + head_size];
                    *score = q_h.iter().zip(k).map(|(&a, &b)| a * b).sum::<f32>() * scale;
                }

                softmax(&mut att_h[..=pos]);

                // Value-weighted sum.
                for o in out_h.iter_mut() {
                    *o = 0.0;
                }
                for (t, &a) in att_h[..=pos].iter().enumerate() {
                    let base = t * kv_dim + kv_off;
                    let v = &value_cache[base..base + head_size];
                    for (o, &vi) in out_h.iter_mut().zip(v) {
                        *o += a * vi;
                    }
                }
            });
    }

    fn swiglu(&self, hb: &mut [f32], hb2: &[f32]) {
        debug_assert_eq!(hb.len(), hb2.len());
        for (h, &g) in hb.iter_mut().zip(hb2) {
            *h = silu(*h) * g;
        }
    }

    fn add(&self, out: &mut [f32], x: &[f32]) {
        debug_assert_eq!(out.len(), x.len());
        for (o, &xi) in out.iter_mut().zip(x) {
            *o += xi;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }

    #[test]
    fn matmul_f32_basic() {
        // w = [[1,2],[3,4]], x = [1,1]  ->  [3, 7]
        let w = QMatrix::f32(vec![1.0, 2.0, 3.0, 4.0].into(), 2, 2).unwrap();
        let x = [1.0, 1.0];
        let mut out = [0.0; 2];
        CpuBackend.matmul(&mut out, &x, &w);
        approx(out[0], 3.0);
        approx(out[1], 7.0);
    }

    #[test]
    fn matmul_quant_matches_f32() {
        use crate::quant::quantize_q8_0;
        // Two rows of 32 cols. Quantize to Q8_0, matmul, and compare to the
        // exact f32 result within Q8_0 tolerance.
        let row0: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let row1: Vec<f32> = (0..32).map(|i| (i as f32 * 0.05) - 0.7).collect();
        let x: Vec<f32> = (0..32).map(|i| ((i * 7) % 5) as f32 - 2.0).collect();

        let mut all = row0.clone();
        all.extend_from_slice(&row1);
        let w = QMatrix::quant(GgmlType::Q8_0, quantize_q8_0(&all).into(), 2, 32).unwrap();

        let mut out = [0.0; 2];
        CpuBackend.matmul(&mut out, &x, &w);

        let exact0: f32 = row0.iter().zip(&x).map(|(a, b)| a * b).sum();
        let exact1: f32 = row1.iter().zip(&x).map(|(a, b)| a * b).sum();
        assert!((out[0] - exact0).abs() < 0.1, "{} vs {}", out[0], exact0);
        assert!((out[1] - exact1).abs() < 0.1, "{} vs {}", out[1], exact1);
    }

    #[test]
    #[ignore = "timing benchmark; run with --release -- --ignored --nocapture"]
    fn bench_q8_0_matmul() {
        use crate::quant::{quantize_activation_q8, quantize_q8_0, vec_dot_q8_0};
        use std::time::Instant;

        let (rows, cols, iters) = (2048usize, 2048usize, 10);
        let wf: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 257) as f32 - 128.0) * 0.01)
            .collect();
        let wbytes = quantize_q8_0(&wf);
        let rb = GgmlType::Q8_0.bytes_for(cols);
        let x: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let mut sink = 0.0f32;

        let t0 = Instant::now();
        for _ in 0..iters {
            for i in 0..rows {
                sink += dot_quant_row(GgmlType::Q8_0, &wbytes[i * rb..i * rb + rb], &x);
            }
        }
        let dequant = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..iters {
            let act = quantize_activation_q8(&x);
            for i in 0..rows {
                sink += vec_dot_q8_0(&wbytes[i * rb..i * rb + rb], &act);
            }
        }
        let int8 = t1.elapsed();

        eprintln!(
            "Q8_0 {rows}x{cols} x{iters}: dequant={dequant:?} int8={int8:?} \
             speedup={:.2}x (sink={sink})",
            dequant.as_secs_f64() / int8.as_secs_f64()
        );
    }

    #[test]
    fn rmsnorm_unit_input() {
        let x = [1.0, 1.0, 1.0, 1.0];
        let w = [1.0; 4];
        let mut out = [0.0; 4];
        CpuBackend.rmsnorm(&mut out, &x, &w);
        let expected = 1.0 / (1.0 + 1e-5f32).sqrt();
        for &o in &out {
            approx(o, expected);
        }
    }

    #[test]
    fn softmax_uniform() {
        let mut v = [1.0, 1.0];
        softmax(&mut v);
        approx(v[0], 0.5);
        approx(v[1], 0.5);
    }

    #[test]
    fn swiglu_known_value() {
        let mut hb = [1.0];
        let hb2 = [2.0];
        CpuBackend.swiglu(&mut hb, &hb2);
        // silu(1) = 1 * sigmoid(1) = 0.7310586, times 2.
        approx(hb[0], 0.7310586 * 2.0);
    }

    #[test]
    fn rope_pos_zero_is_identity() {
        let mut q = [1.0, 2.0, 3.0, 4.0];
        let mut k = [5.0, 6.0, 7.0, 8.0];
        let before = (q, k);
        CpuBackend.rope(&mut q, &mut k, 0, 4, 4);
        assert_eq!(q, before.0);
        assert_eq!(k, before.1);
    }

    #[test]
    fn rope_rotates_by_angle() {
        // head_size 2, single pair, pos 1, head_dim 0 -> freq 1, angle = 1 rad.
        let mut q = [1.0, 0.0];
        let mut k = [1.0, 0.0];
        CpuBackend.rope(&mut q, &mut k, 1, 2, 2);
        approx(q[0], 1.0f32.cos());
        approx(q[1], 1.0f32.sin());
    }

    #[test]
    fn attention_single_step_returns_value() {
        // pos 0, one head: softmax over a single score is 1.0, so the output
        // is exactly the only value vector.
        let head_size = 2;
        let q = [0.5, 0.5];
        let key_cache = [9.0, 9.0]; // one timestep, kv_dim = 2
        let value_cache = [3.0, -4.0];
        let mut att = [0.0; 1];
        let mut out = [0.0; 2];
        CpuBackend.attention(
            &mut out,
            &q,
            &key_cache,
            &value_cache,
            &mut att,
            0,
            1,
            1,
            head_size,
            1,
            2,
        );
        approx(out[0], 3.0);
        approx(out[1], -4.0);
    }

    #[test]
    fn attention_uniform_scores_average_values() {
        // Two timesteps with identical keys -> equal scores -> output is the
        // mean of the two value vectors.
        let head_size = 2;
        let q = [1.0, 1.0];
        let key_cache = [1.0, 1.0, 1.0, 1.0]; // two identical keys
        let value_cache = [2.0, 4.0, 6.0, 8.0];
        let mut att = [0.0; 2];
        let mut out = [0.0; 2];
        CpuBackend.attention(
            &mut out,
            &q,
            &key_cache,
            &value_cache,
            &mut att,
            1,
            1,
            1,
            head_size,
            2,
            2,
        );
        approx(out[0], 4.0); // mean(2, 6)
        approx(out[1], 6.0); // mean(4, 8)
    }
}
