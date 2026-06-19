//! Multi-threaded CPU backend.
//!
//! The two parallelized kernels ([`CpuBackend::matmul`] and
//! [`CpuBackend::attention`]) split their outer dimension across the rayon
//! thread pool. The inner dot-product loops are written so the compiler can
//! autovectorize them — build with `-C target-cpu=native` (see
//! `.cargo/config.toml`) to get AVX2 / AVX-512 / FMA.

use rayon::prelude::*;

use crate::backend::Backend;
use crate::math::silu;
use crate::quant::{
    dequant_block, dequantize_into, quantize_activation_q8, quantize_activation_q8k, vec_dot_q4_0,
    vec_dot_q4_k, vec_dot_q4_k_batch, vec_dot_q6_k, vec_dot_q6_k_batch, vec_dot_q8_0, GgmlType,
    Q8Activation, Q8KActivation, MAX_BLOCK,
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
    fn rmsnorm(&self, out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
        debug_assert_eq!(out.len(), x.len());
        debug_assert_eq!(out.len(), weight.len());
        let n = x.len() as f32;
        let ss = x.iter().map(|&v| v * v).sum::<f32>() / n + eps;
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
                let row = |i: usize| &data[i * rb..i * rb + rb];
                match ty {
                    // Q8 activation (32-block) for the simple formats.
                    GgmlType::Q8_0 | GgmlType::Q4_0 => {
                        let act = quantize_activation_q8(x);
                        let dot: fn(&[u8], &Q8Activation) -> f32 = match ty {
                            GgmlType::Q8_0 => vec_dot_q8_0,
                            _ => vec_dot_q4_0,
                        };
                        out.par_iter_mut()
                            .enumerate()
                            .for_each(|(i, o)| *o = dot(row(i), &act));
                    }
                    // Q8_K activation (256-block) for the k-quants.
                    GgmlType::Q4_K | GgmlType::Q6_K => {
                        let act = quantize_activation_q8k(x);
                        let dot: fn(&[u8], &Q8KActivation) -> f32 = match ty {
                            GgmlType::Q4_K => vec_dot_q4_k,
                            _ => vec_dot_q6_k,
                        };
                        out.par_iter_mut()
                            .enumerate()
                            .for_each(|(i, o)| *o = dot(row(i), &act));
                    }
                    // F16: no integer path; dequantize each block to f32.
                    _ => {
                        out.par_iter_mut()
                            .enumerate()
                            .for_each(|(i, o)| *o = dot_quant_row(*ty, row(i), x));
                    }
                }
            }
        }
    }

    /// Cache-blocked batched matmul — *the prefill GEMM*.
    ///
    /// The default trait impl loops the single-row [`Self::matmul`] per token,
    /// which re-streams the **entire** weight matrix from RAM once per token: an
    /// N-token prefill then pays N passes over the weights and runs at decode
    /// bandwidth (zero reuse). Here every weight row is streamed **once** and
    /// dot-producted against all `rows` activation columns (quantized up front),
    /// so prefill becomes compute-bound like llama.cpp's chunked-row GEMM.
    ///
    /// Bit-identical to the default: it calls the same `vec_dot_*` kernels on the
    /// same per-row quantized activations, only reordered for weight reuse.
    fn matmul_batch(&self, out: &mut [f32], x: &[f32], w: &QMatrix, rows: usize) {
        let (oc, ic) = (w.rows(), w.cols());
        debug_assert_eq!(out.len(), rows * oc);
        debug_assert_eq!(x.len(), rows * ic);
        // A/B toggle: the pre-blocking per-row GEMV (re-streams the whole weight
        // matrix once per token) for honest before/after benchmarking. Off by
        // default; mirrors the `RUSTY_LLAMA_NO_AVX2`/`NO_INT8` perf toggles.
        if std::env::var_os("RUSTY_LLAMA_NO_BLOCKED_GEMM").is_some() {
            for r in 0..rows {
                self.matmul(&mut out[r * oc..r * oc + oc], &x[r * ic..r * ic + ic], w);
            }
            return;
        }
        // No reuse to exploit for 0/1 rows — the plain GEMV is already optimal.
        if rows <= 1 {
            if rows == 1 {
                self.matmul(&mut out[..oc], &x[..ic], w);
            }
            return;
        }
        let xr = |r: usize| &x[r * ic..r * ic + ic];
        // Transposed scratch `(oc, rows)`: each weight row maps to one
        // contiguous column, so it is read from RAM once and reused across every
        // token instead of re-streamed per token.
        let mut tmp = vec![0.0f32; oc * rows];
        match w {
            QMatrix::F32 { data, cols, .. } => {
                let cols = *cols;
                tmp.par_chunks_mut(rows).enumerate().for_each(|(i, col)| {
                    let wrow = &data[i * cols..i * cols + cols];
                    for (r, o) in col.iter_mut().enumerate() {
                        *o = wrow.iter().zip(xr(r)).map(|(&a, &b)| a * b).sum();
                    }
                });
            }
            QMatrix::Quant { ty, data, cols, .. } => {
                let rb = ty.bytes_for(*cols);
                let row = |i: usize| &data[i * rb..i * rb + rb];
                match ty {
                    GgmlType::Q8_0 | GgmlType::Q4_0 => {
                        let acts: Vec<Q8Activation> = (0..rows)
                            .into_par_iter()
                            .map(|r| quantize_activation_q8(xr(r)))
                            .collect();
                        let dot: fn(&[u8], &Q8Activation) -> f32 = match ty {
                            GgmlType::Q8_0 => vec_dot_q8_0,
                            _ => vec_dot_q4_0,
                        };
                        tmp.par_chunks_mut(rows).enumerate().for_each(|(i, col)| {
                            let wr = row(i);
                            for (r, o) in col.iter_mut().enumerate() {
                                *o = dot(wr, &acts[r]);
                            }
                        });
                    }
                    GgmlType::Q4_K | GgmlType::Q6_K => {
                        let acts: Vec<Q8KActivation> = (0..rows)
                            .into_par_iter()
                            .map(|r| quantize_activation_q8k(xr(r)))
                            .collect();
                        // Unpack-amortized batched dot: each weight super-block is
                        // unpacked ONCE per token-block and reused across columns,
                        // vs the per-column path re-unpacking it per token.
                        let batch: fn(&[u8], &[Q8KActivation], &mut [f32]) = match ty {
                            GgmlType::Q4_K => vec_dot_q4_k_batch,
                            _ => vec_dot_q6_k_batch,
                        };
                        tmp.par_chunks_mut(rows)
                            .enumerate()
                            .for_each(|(i, col)| batch(row(i), &acts, col));
                    }
                    // F16 (block size 1): dequantize each weight row once and
                    // reuse it. A sequential dot over the dequantized row is
                    // bit-identical to `dot_quant_row`, which at block size 1 also
                    // accumulates element by element.
                    _ => {
                        let ty = *ty;
                        tmp.par_chunks_mut(rows).enumerate().for_each(|(i, col)| {
                            let mut wbuf = vec![0.0f32; ic];
                            dequantize_into(ty, row(i), &mut wbuf).expect("dequant weight row");
                            for (r, o) in col.iter_mut().enumerate() {
                                *o = wbuf.iter().zip(xr(r)).map(|(&a, &b)| a * b).sum();
                            }
                        });
                    }
                }
            }
        }
        // Transpose the `(oc, rows)` scratch into the row-major `(rows, oc)`
        // output. Cache-blocked T×T tiles so the strided side reuses each loaded
        // cache line across the tile instead of thrashing the 11 MB scratch;
        // parallel over output-row tiles (each owns a disjoint contiguous block).
        const T: usize = 32;
        out.par_chunks_mut(T * oc).enumerate().for_each(|(blk, oblk)| {
            let r0 = blk * T;
            let rb = oblk.len() / oc; // T, or the final remainder
            for i0 in (0..oc).step_by(T) {
                let iend = (i0 + T).min(oc);
                for rl in 0..rb {
                    let dst = &mut oblk[rl * oc + i0..rl * oc + iend];
                    for (k, o) in dst.iter_mut().enumerate() {
                        *o = tmp[(i0 + k) * rows + r0 + rl];
                    }
                }
            }
        });
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
        // Rotate consecutive (even, odd) pairs within each head, looking up the
        // pair's precomputed inverse frequency. Pairs below `kv_dim` rotate both
        // q and k; the rest (extra query heads under grouped-query attention)
        // rotate q only. Pairs whose index is past `inv_freq.len()` (partial
        // rotary) pass through unchanged.
        let dim = q.len();
        let mut i = 0;
        while i < dim {
            let pair = (i % head_size) / 2;
            if pair < inv_freq.len() {
                let val = pos as f32 * inv_freq[pair];
                let (fcr, fci) = (val.cos() * mscale, val.sin() * mscale);

                let (q0, q1) = (q[i], q[i + 1]);
                q[i] = q0 * fcr - q1 * fci;
                q[i + 1] = q0 * fci + q1 * fcr;

                if i < kv_dim {
                    let (k0, k1) = (k[i], k[i + 1]);
                    k[i] = k0 * fcr - k1 * fci;
                    k[i + 1] = k0 * fci + k1 * fcr;
                }
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
        logit_softcap: f32,
    ) {
        // Flash attention: a single streaming pass over the keys per head with an
        // online (running-max) softmax, so no `seq_len`-sized score buffer is
        // needed — `out_h` doubles as the running value accumulator. Algebraically
        // identical to the materialized-softmax form (to fp rounding), and the
        // memory it saves matters for long context / many concurrent slots.
        let _ = (att, seq_len); // online softmax keeps its state in registers.
        let kv_mul = n_heads / n_kv_heads;
        let scale = 1.0 / (head_size as f32).sqrt();

        // One independent task per head; `out` is sliced into disjoint per-head
        // chunks so the closures never alias.
        out.par_chunks_mut(head_size)
            .enumerate()
            .for_each(|(h, out_h)| {
                let q_h = &q[h * head_size..h * head_size + head_size];
                // Which key/value head this query head attends to.
                let kv_off = (h / kv_mul) * head_size;

                let mut m = f32::NEG_INFINITY; // running max score
                let mut l = 0.0f32; // running exp-sum (softmax denominator)
                for o in out_h.iter_mut() {
                    *o = 0.0;
                }
                for t in 0..=pos {
                    let base = t * kv_dim + kv_off;
                    let k = &key_cache[base..base + head_size];
                    let mut s = q_h.iter().zip(k).map(|(&a, &b)| a * b).sum::<f32>() * scale;
                    // Gemma2 tanh logit softcapping on the raw score.
                    if logit_softcap > 0.0 {
                        s = logit_softcap * (s / logit_softcap).tanh();
                    }
                    let m_new = m.max(s);
                    // exp(m - m_new): rescales the running state when the max grows
                    // (== 0 on the first key, where m == -inf, zeroing the seed).
                    let alpha = (m - m_new).exp();
                    let p = (s - m_new).exp();
                    l = l * alpha + p;
                    let v = &value_cache[base..base + head_size];
                    for (o, &vi) in out_h.iter_mut().zip(v) {
                        *o = *o * alpha + p * vi;
                    }
                    m = m_new;
                }
                let inv = 1.0 / l;
                for o in out_h.iter_mut() {
                    *o *= inv;
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

    fn forward_step(
        &self,
        model: &crate::model::Model,
        state: &mut crate::model::RunState,
        token: usize,
        pos: usize,
    ) {
        // The CPU per-op path is already efficient; just run the generic forward.
        crate::model::forward(model, state, self, token, pos);
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
    fn matmul_routes_kquants_to_int8() {
        use crate::quant::{
            f32_to_f16, pack_scales_q4_k, quantize_activation_q8k, vec_dot_q4_k, vec_dot_q6_k,
        };

        let x: Vec<f32> = (0..256).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
        let act = quantize_activation_q8k(&x);

        // Valid Q4_K block.
        let mut q4k = Vec::new();
        q4k.extend_from_slice(&f32_to_f16(0.05).to_le_bytes());
        q4k.extend_from_slice(&f32_to_f16(0.02).to_le_bytes());
        q4k.extend_from_slice(&pack_scales_q4_k(
            [10, 20, 5, 33, 41, 7, 18, 25],
            [3, 9, 14, 1, 22, 6, 30, 11],
        ));
        (0..128u32).for_each(|i| q4k.push(((i * 7 + 3) % 256) as u8));

        let mut bytes = q4k.clone();
        bytes.extend_from_slice(&q4k); // two identical rows
        let w = QMatrix::quant(GgmlType::Q4_K, bytes.into(), 2, 256).unwrap();
        let mut out = [0.0; 2];
        CpuBackend.matmul(&mut out, &x, &w);
        let expect = vec_dot_q4_k(&q4k, &act);
        approx(out[0], expect);
        approx(out[1], expect);

        // Valid Q6_K block.
        let mut q6k = Vec::new();
        (0..128u32).for_each(|i| q6k.push(((i * 5 + 1) % 256) as u8));
        (0..64u32).for_each(|i| q6k.push(((i * 9 + 2) % 256) as u8));
        (0..16i32).for_each(|i| q6k.push((i - 8) as i8 as u8));
        q6k.extend_from_slice(&f32_to_f16(0.03).to_le_bytes());

        let mut bytes = q6k.clone();
        bytes.extend_from_slice(&q6k);
        let w = QMatrix::quant(GgmlType::Q6_K, bytes.into(), 2, 256).unwrap();
        let mut out = [0.0; 2];
        CpuBackend.matmul(&mut out, &x, &w);
        let expect = vec_dot_q6_k(&q6k, &act);
        approx(out[0], expect);
        approx(out[1], expect);
    }

    #[test]
    #[ignore = "timing benchmark; run with --release -- --ignored --nocapture"]
    fn bench_q4_k_matmul() {
        use crate::quant::{quantize_activation_q8k, vec_dot_q4_k};
        use std::time::Instant;

        let (rows, cols, iters) = (2048usize, 2048usize, 10);
        let rb = GgmlType::Q4_K.bytes_for(cols);
        // Pseudo-random weight bytes (validity doesn't matter for timing).
        let mut wbytes = vec![0u8; rows * rb];
        let mut s: u32 = 0x1234_5678;
        for b in wbytes.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (s >> 16) as u8;
        }
        let x: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let mut sink = 0.0f32;

        let t0 = Instant::now();
        for _ in 0..iters {
            for i in 0..rows {
                sink += dot_quant_row(GgmlType::Q4_K, &wbytes[i * rb..i * rb + rb], &x);
            }
        }
        let dequant = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..iters {
            let act = quantize_activation_q8k(&x);
            for i in 0..rows {
                sink += vec_dot_q4_k(&wbytes[i * rb..i * rb + rb], &act);
            }
        }
        let int8 = t1.elapsed();

        eprintln!(
            "Q4_K {rows}x{cols} x{iters}: dequant={dequant:?} int8={int8:?} \
             speedup={:.2}x (sink={sink})",
            dequant.as_secs_f64() / int8.as_secs_f64()
        );
    }

    /// Clean CPU-only real-model throughput (prefill `pp512` + decode `tg128`) on
    /// the actual GGUF — isolated from any GPU work (no thermal contamination)
    /// and on the rigorous `llama-bench` protocol (1 untimed warmup + r timed
    /// reps, `RunState` reused so the KV cache is not re-zeroed in the timed
    /// region). Set `RUSTY_LLAMA_GGUF` to override the path. Compare against
    /// `llamacpp_bench/lc/llama-bench.exe -m <gguf> -ngl 0`.
    #[test]
    #[ignore = "needs the real GGUF; run with --release -- --ignored --nocapture"]
    fn bench_prefill_decode_real_cpu() {
        use crate::bench_util::{bench_reps, bench_stat};
        use std::time::Duration;
        let path = std::env::var("RUSTY_LLAMA_GGUF")
            .unwrap_or_else(|_| "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".to_string());
        let cp = match crate::Checkpoint::open(&path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping CPU real-model bench: can't open {path}: {e}");
                return;
            }
        };
        if !crate::Gguf::is_gguf(cp.bytes()) {
            eprintln!("skipping CPU real-model bench: {path} is not a GGUF");
            return;
        }
        let gguf = crate::Gguf::parse(cp.bytes()).expect("parse gguf");
        let model = crate::Model::from_gguf(&gguf).expect("build model");
        let c = &model.config;
        let b = CpuBackend::new();
        let reps = bench_reps();

        let n = 512.min(c.seq_len);
        let tokens: Vec<usize> = (0..n).map(|i| (i * 7 + 1) % c.vocab_size).collect();
        let mut ps = crate::RunState::new(&model.config);
        let (pp, pp_sd) = bench_stat(n, reps, Duration::from_secs(3), || {
            b.forward_prefill(&model, &mut ps, &tokens, 0);
        });

        let steps = 128usize;
        let mut ds = crate::RunState::new(&model.config);
        let (tg, tg_sd) = bench_stat(steps, reps, Duration::from_secs(3), || {
            for pos in 0..steps {
                b.forward_step(&model, &mut ds, (pos * 13 + 1) % c.vocab_size, pos);
            }
        });

        eprintln!(
            "CPU real-model {path}: pp{n} {pp:.0} ± {pp_sd:.0} tok/s  \
             tg{steps} {tg:.0} ± {tg_sd:.0} tok/s  \
             (compare: llama-bench -m <gguf> -ngl 0 -p {n} -n {steps} -r {reps})"
        );
    }

    #[test]
    fn rmsnorm_unit_input() {
        let x = [1.0, 1.0, 1.0, 1.0];
        let w = [1.0; 4];
        let mut out = [0.0; 4];
        CpuBackend.rmsnorm(&mut out, &x, &w, 1e-5);
        let expected = 1.0 / (1.0 + 1e-5f32).sqrt();
        for &o in &out {
            approx(o, expected);
        }
    }

    #[test]
    fn rmsnorm_uses_eps() {
        // A large eps must visibly shrink the output.
        let x = [1.0, 1.0, 1.0, 1.0];
        let w = [1.0; 4];
        let (mut a, mut b) = ([0.0; 4], [0.0; 4]);
        CpuBackend.rmsnorm(&mut a, &x, &w, 1e-5);
        CpuBackend.rmsnorm(&mut b, &x, &w, 9.0);
        assert!(b[0] < a[0]);
    }

    #[test]
    fn softmax_uniform() {
        let mut v = [1.0, 1.0];
        crate::math::softmax(&mut v);
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

    /// Base RoPE inverse frequencies for a full-rotary head: `θ^(-2j/head_size)`.
    fn base_inv_freq(head_size: usize, theta: f32) -> Vec<f32> {
        (0..head_size / 2)
            .map(|j| (theta as f64).powf(-2.0 * j as f64 / head_size as f64) as f32)
            .collect()
    }

    #[test]
    fn rope_pos_zero_is_identity() {
        let mut q = [1.0, 2.0, 3.0, 4.0];
        let mut k = [5.0, 6.0, 7.0, 8.0];
        let before = (q, k);
        let inv = base_inv_freq(4, 10000.0);
        CpuBackend.rope(&mut q, &mut k, 0, 4, 4, &inv, 1.0);
        assert_eq!(q, before.0);
        assert_eq!(k, before.1);
    }

    #[test]
    fn rope_rotates_by_angle() {
        // head_size 2, single pair, pos 1, head_dim 0 -> freq 1, angle = 1 rad
        // (independent of theta since the exponent is 0).
        let mut q = [1.0, 0.0];
        let mut k = [1.0, 0.0];
        let inv = base_inv_freq(2, 10000.0);
        CpuBackend.rope(&mut q, &mut k, 1, 2, 2, &inv, 1.0);
        approx(q[0], 1.0f32.cos());
        approx(q[1], 1.0f32.sin());
    }

    #[test]
    fn rope_uses_theta() {
        // head_dim 2 of a size-4 head: angle = pos / theta^(2/4) = 1/sqrt(theta),
        // so different bases give different rotations.
        let rotate = |theta: f32| {
            let mut q = [0.0, 0.0, 1.0, 0.0];
            let mut k = q;
            let inv = base_inv_freq(4, theta);
            CpuBackend.rope(&mut q, &mut k, 1, 4, 4, &inv, 1.0);
            (q[2], q[3])
        };
        let (a_cos, _) = rotate(10000.0);
        let (b_cos, _) = rotate(100.0);
        approx(a_cos, (1.0f32 / 10000.0f32.sqrt()).cos());
        approx(b_cos, (1.0f32 / 100.0f32.sqrt()).cos());
        assert!((a_cos - b_cos).abs() > 1e-4);
    }

    #[test]
    fn rope_linear_scaling_halves_angle() {
        // A linearly-scaled table divides every frequency (hence every angle)
        // by `factor`. With factor 2 at pos 1, the head_dim-0 pair rotates by
        // 0.5 rad instead of 1.0.
        let mut q = [1.0, 0.0];
        let mut k = [1.0, 0.0];
        let inv: Vec<f32> = base_inv_freq(2, 10000.0).iter().map(|f| f / 2.0).collect();
        CpuBackend.rope(&mut q, &mut k, 1, 2, 2, &inv, 1.0);
        approx(q[0], 0.5f32.cos());
        approx(q[1], 0.5f32.sin());
    }

    #[test]
    fn rope_mscale_scales_magnitude() {
        // YaRN's mscale multiplies cos/sin, so the rotated vector's magnitude
        // grows by `mscale`.
        let mut q = [1.0, 0.0];
        let mut k = [1.0, 0.0];
        let inv = base_inv_freq(2, 10000.0);
        CpuBackend.rope(&mut q, &mut k, 1, 2, 2, &inv, 2.0);
        approx(q[0], 2.0 * 1.0f32.cos());
        approx(q[1], 2.0 * 1.0f32.sin());
    }

    #[test]
    fn rope_partial_rotary_leaves_tail_untouched() {
        // Only the first pair of each size-4 head is rotated (inv_freq has one
        // entry); the second pair passes through unchanged.
        let mut q = [1.0, 0.0, 7.0, 9.0];
        let mut k = [1.0, 0.0, 7.0, 9.0];
        let inv = vec![1.0f32]; // rotary_dim = 2 of head_size 4
        CpuBackend.rope(&mut q, &mut k, 1, 4, 4, &inv, 1.0);
        approx(q[0], 1.0f32.cos());
        approx(q[1], 1.0f32.sin());
        // Untouched tail.
        approx(q[2], 7.0);
        approx(q[3], 9.0);
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
            0.0,
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
            0.0,
        );
        approx(out[0], 4.0); // mean(2, 6)
        approx(out[1], 6.0); // mean(4, 8)
    }

    /// Flash (online-softmax) attention must equal the classic
    /// scores→softmax→value-weighted-sum form, across GQA, several positions, and
    /// Gemma2 logit softcapping.
    #[test]
    fn attention_flash_matches_classic_reference() {
        let (n_heads, n_kv_heads, head_size, seq_len) = (4usize, 2usize, 8usize, 12usize);
        let kv_dim = n_kv_heads * head_size;
        let kv_mul = n_heads / n_kv_heads;
        let dim = n_heads * head_size;
        // Deterministic pseudo-random fills in [-0.5, 0.5).
        let r = |i: usize| ((i.wrapping_mul(2654435761) >> 8) as f32 / (1u32 << 24) as f32) % 1.0 - 0.5;
        let q: Vec<f32> = (0..dim).map(r).collect();
        let key_cache: Vec<f32> = (0..seq_len * kv_dim).map(|i| r(i + 7)).collect();
        let value_cache: Vec<f32> = (0..seq_len * kv_dim).map(|i| r(i + 99)).collect();

        // Classic reference for one (pos, softcap).
        let reference = |pos: usize, softcap: f32| -> Vec<f32> {
            let scale = 1.0 / (head_size as f32).sqrt();
            let mut out = vec![0.0f32; dim];
            for h in 0..n_heads {
                let q_h = &q[h * head_size..h * head_size + head_size];
                let kv_off = (h / kv_mul) * head_size;
                let mut scores = vec![0.0f32; pos + 1];
                for (t, sc) in scores.iter_mut().enumerate() {
                    let b = t * kv_dim + kv_off;
                    let mut s = q_h
                        .iter()
                        .zip(&key_cache[b..b + head_size])
                        .map(|(&a, &c)| a * c)
                        .sum::<f32>()
                        * scale;
                    if softcap > 0.0 {
                        s = softcap * (s / softcap).tanh();
                    }
                    *sc = s;
                }
                crate::math::softmax(&mut scores);
                let out_h = &mut out[h * head_size..h * head_size + head_size];
                for (t, &a) in scores.iter().enumerate() {
                    let b = t * kv_dim + kv_off;
                    for (o, &vi) in out_h.iter_mut().zip(&value_cache[b..b + head_size]) {
                        *o += a * vi;
                    }
                }
            }
            out
        };

        for &softcap in &[0.0f32, 30.0] {
            for pos in [0usize, 1, 5, seq_len - 1] {
                let mut att = vec![0.0f32; n_heads * seq_len];
                let mut out = vec![0.0f32; dim];
                CpuBackend.attention(
                    &mut out,
                    &q,
                    &key_cache,
                    &value_cache,
                    &mut att,
                    pos,
                    n_heads,
                    n_kv_heads,
                    head_size,
                    seq_len,
                    kv_dim,
                    softcap,
                );
                for (o, w) in out.iter().zip(&reference(pos, softcap)) {
                    assert!((o - w).abs() < 1e-6, "flash {o} vs classic {w} (pos {pos} cap {softcap})");
                }
            }
        }
    }
}
