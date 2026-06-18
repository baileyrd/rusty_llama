//! NVIDIA CUDA backend (cuBLASLt tensor cores).
//!
//! A third [`Backend`] implementation — alongside the [`CpuBackend`] and the
//! portable wgpu [`GpuBackend`](crate::backend::GpuBackend) — targeting NVIDIA
//! tensor cores via cuBLASLt, gated behind the `cuda` cargo feature and built on
//! the [`cudarc`] crate. The portable wgpu cooperative-matrix path was spiked
//! and ruled out on this hardware (wgpu 29 wires only 8×8 f32, which the Vulkan
//! driver doesn't expose; see `PERFORMANCE.md` item 3), so an f16 cuBLASLt GEMM
//! is the route to the prefill tensor-core win.
//!
//! ## Milestones
//!
//! - **M0** (done): device + cuBLASLt handle init with a graceful no-CUDA
//!   fallback, ops delegated to the [`CpuBackend`].
//! - **M1** (this file): the `forward_prefill` matmuls + classifier run on
//!   cuBLASLt f16 tensor cores ([`CudaBackend::gemm_f16`]); the rest still
//!   delegates to the CPU. Decode (`forward_step`) stays on the CPU — at
//!   batch=1 it is GEMV/bandwidth-bound, where tensor cores help little.
//! - **M2+** (later): move the elementwise/attention ops onto CUDA kernels and
//!   keep activations resident to cut the per-op host↔device copies.
//!
//! ## Why dynamic loading can't just `?`-propagate a missing driver
//!
//! cudarc's `dynamic-loading` backend resolves the CUDA driver library lazily on
//! first use and **panics** (via `panic_no_lib_found`) when it is absent — so on
//! a CUDA-less host (headless CI, no NVIDIA GPU) `CudaContext::new` aborts rather
//! than returning `Err`. [`cudarc::driver::sys::is_culib_present`] is the
//! non-panicking gatekeeper; [`CudaBackend::new`] checks it first.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::f16;

use crate::backend::{Backend, CpuBackend};
use crate::model::{Model, RunState};
use crate::quant::dequantize;
use crate::tensor::QMatrix;

/// CUDA backend: a resident device context, stream, and cuBLASLt handle.
///
/// The matmuls (the `forward_prefill` GEMMs and the classifier) run on cuBLASLt
/// f16 tensor cores; the remaining elementwise/attention ops delegate to an
/// internal [`CpuBackend`]. f16-converted weights are uploaded once and cached,
/// keyed on the source weight's data pointer (stable across the run, since
/// weights are borrowed from the model) — mirroring the wgpu backend's strategy.
pub struct CudaBackend {
    // `ctx` is held only to keep the CUDA context alive for the lifetime of the
    // backend (the stream / handle / slices reference it); it is never read
    // directly, hence the allow.
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlasLT,
    /// f16 device weights, keyed by source data pointer (see [`Self::weight_f16`]).
    weights: Mutex<HashMap<usize, Arc<CudaSlice<f16>>>>,
    cpu: CpuBackend,
    device_name: String,
}

impl CudaBackend {
    /// Initialize CUDA on device 0 and a cuBLASLt handle, or return an error
    /// string when no usable CUDA device is present — so callers can fall back to
    /// the CPU exactly like [`GpuBackend::new`](crate::backend::GpuBackend::new).
    pub fn new() -> Result<Self, String> {
        // `CudaContext::new` calls `cuInit` internally, which — under cudarc's
        // `dynamic-loading` — PANICS (not `Err`) when the CUDA driver library is
        // missing. `is_culib_present` only attempts to `dlopen`/`LoadLibrary` the
        // driver and returns a bool, so it is the safe pre-check.
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err("no CUDA driver library found (need an NVIDIA GPU + driver)".into());
        }

        let ctx = CudaContext::new(0).map_err(|e| format!("CUDA init failed: {e:?}"))?;
        let stream = ctx.default_stream();
        let blas =
            CudaBlasLT::new(stream.clone()).map_err(|e| format!("cuBLASLt init failed: {e:?}"))?;
        let device_name = ctx
            .name()
            .map_err(|e| format!("CUDA device-name query failed: {e:?}"))?;

        Ok(CudaBackend {
            ctx,
            stream,
            blas,
            weights: Mutex::new(HashMap::new()),
            cpu: CpuBackend::new(),
            device_name,
        })
    }

    /// The selected CUDA device's name (for logging / benchmarks).
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// The f16 device copy of weight matrix `w` (row-major `rows`×`cols`),
    /// uploaded once and cached by source data pointer. Quantized weights are
    /// dequantized to f32 on the host, then narrowed to f16.
    fn weight_f16(&self, w: &QMatrix) -> Arc<CudaSlice<f16>> {
        let key = match w {
            QMatrix::F32 { data, .. } => data.as_ptr() as usize,
            QMatrix::Quant { data, .. } => data.as_ptr() as usize,
        };
        if let Some(g) = self.weights.lock().unwrap().get(&key) {
            return g.clone();
        }

        let (rows, cols) = (w.rows(), w.cols());
        let f32_data: Cow<[f32]> = match w {
            QMatrix::F32 { data, .. } => Cow::Borrowed(data),
            QMatrix::Quant { ty, data, .. } => {
                Cow::Owned(dequantize(*ty, data, rows * cols).expect("weight dequantization"))
            }
        };
        let f16_data: Vec<f16> = f32_data.iter().map(|&v| f16::from_f32(v)).collect();
        let dev = self
            .stream
            .clone_htod(&f16_data)
            .expect("upload f16 weight to device");

        let arc = Arc::new(dev);
        self.weights.lock().unwrap().insert(key, arc.clone());
        arc
    }

    /// `out(rows × oc) = x(rows × ic) · Wᵀ` on cuBLASLt f16 tensor cores
    /// (f16 inputs, f32 accumulate). `x`/`out` are row-major host slices.
    ///
    /// cuBLASLt is column-major, so we compute the transpose `outᵀ = W · xᵀ` by
    /// feeding our row-major `W` and `x` buffers directly with `transa = true`
    /// (and the dimensions swapped: `m = oc`, `n = rows`, `k = ic`). Derivation
    /// cross-checked against cudarc's `test_matmul_half`, and guarded by the
    /// per-op parity test against the CPU backend.
    fn gemm_f16(&self, out: &mut [f32], x: &[f32], w: &QMatrix, rows: usize) {
        let (oc, ic) = (w.rows(), w.cols());
        debug_assert_eq!(x.len(), rows * ic);
        debug_assert_eq!(out.len(), rows * oc);

        let w_dev = self.weight_f16(w);
        let x_f16: Vec<f16> = x.iter().map(|&v| f16::from_f32(v)).collect();
        let x_dev = self.stream.clone_htod(&x_f16).expect("upload activation");
        let mut c_dev = self
            .stream
            .alloc_zeros::<f16>(rows * oc)
            .expect("alloc gemm output");

        let cfg = MatmulConfig {
            transa: true,
            transb: false,
            transc: false,
            m: oc as u64,
            n: rows as u64,
            k: ic as u64,
            alpha: 1.0,
            lda: ic as i64,
            ldb: ic as i64,
            beta: 0.0,
            ldc: oc as i64,
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };
        // SAFETY: dimensions/leading-dims match the buffer shapes above; bad
        // values would be the only way to trip the documented memory hazard.
        unsafe {
            self.blas
                .matmul(cfg, w_dev.as_ref(), &x_dev, &mut c_dev, None, None)
                .expect("cuBLASLt f16 matmul");
        }
        // The dtoh copy of a pageable Vec is already synchronous, but make the
        // GEMM→readback ordering explicit.
        self.stream.synchronize().expect("stream synchronize");
        let c_host: Vec<f16> = self.stream.clone_dtoh(&c_dev).expect("download gemm output");
        for (o, &h) in out.iter_mut().zip(&c_host) {
            *o = h.to_f32();
        }
    }
}

// The matmuls run on cuBLASLt (f16 tensor cores); every other op delegates to
// the CPU backend. `forward_prefill` drives all its GEMMs through
// `matmul`/`matmul_batch`, so this captures the prefill tensor-core win. (Decode
// via `forward_step` still runs entirely on the CPU — see that method.)
impl Backend for CudaBackend {
    fn rmsnorm(&self, out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
        self.cpu.rmsnorm(out, x, weight, eps);
    }

    fn matmul(&self, out: &mut [f32], x: &[f32], w: &QMatrix) {
        self.gemm_f16(out, x, w, 1);
    }

    fn matmul_batch(&self, out: &mut [f32], x: &[f32], w: &QMatrix, rows: usize) {
        self.gemm_f16(out, x, w, rows);
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
        self.cpu.rope(q, k, pos, head_size, kv_dim, inv_freq, mscale);
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
        self.cpu.attention(
            out, q, key_cache, value_cache, att, pos, n_heads, n_kv_heads, head_size, seq_len,
            kv_dim,
        );
    }

    fn swiglu(&self, hb: &mut [f32], hb2: &[f32]) {
        self.cpu.swiglu(hb, hb2);
    }

    fn add(&self, out: &mut [f32], x: &[f32]) {
        self.cpu.add(out, x);
    }

    fn forward_step(&self, model: &Model, state: &mut RunState, token: usize, pos: usize) {
        self.cpu.forward_step(model, state, token, pos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{quantize_q8_0, GgmlType};

    /// A CUDA backend for tests, or `None` (with a skip note) when no CUDA device
    /// is available — so the suite still passes on a GPU-less host.
    fn cuda() -> Option<CudaBackend> {
        match CudaBackend::new() {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!("skipping CUDA test: {e}");
                None
            }
        }
    }

    /// Cheap deterministic pseudo-random f32s in roughly `[-0.5, 0.5)`.
    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed ^ 0x2545_F491_4F6C_DD1D;
        (0..n)
            .map(|_| {
                s ^= s >> 12;
                s ^= s << 25;
                s ^= s >> 27;
                let u = (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as u32; // 24 bits
                u as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    /// Assert two slices match within an f16-GEMM tolerance: inputs are rounded
    /// to f16 (≈2⁻¹¹ relative) but accumulated in f32, so error grows with the
    /// reduction length `ic`. Generous enough for that, tight enough that a wrong
    /// layout (transpose / m·n·k swap) — which yields wholly different values —
    /// fails loudly.
    fn close_f16(got: &[f32], want: &[f32], ic: usize) {
        assert_eq!(got.len(), want.len());
        let tol = 0.02 + ic as f32 * 2e-4;
        let mut maxdiff = 0.0f32;
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            let d = (g - w).abs();
            maxdiff = maxdiff.max(d);
            assert!(
                d <= tol + 0.03 * w.abs(),
                "idx {i}: cuda {g} vs cpu {w} (diff {d} > tol, ic={ic})"
            );
        }
        eprintln!("  close_f16 ok: maxdiff {maxdiff:.4} (tol {tol:.4} + 3%·|w|, ic={ic})");
    }

    /// M0 smoke test: prove the toolchain end-to-end — init CUDA, read the device
    /// name, and round-trip a buffer host→device→host through the stream (which
    /// also confirms the dynamically-loaded driver + cuBLASLt resolve at run
    /// time). Skips cleanly (no failure) when no CUDA device is present.
    #[test]
    #[ignore = "requires a CUDA device; run with --features cuda -- --ignored --nocapture"]
    fn cuda_smoke() {
        let Some(b) = cuda() else { return };
        eprintln!("CUDA device: {}", b.device_name());

        let host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let dev = b.stream.clone_htod(&host).expect("host->device copy");
        let back: Vec<f32> = b.stream.clone_dtoh(&dev).expect("device->host copy");
        assert_eq!(host, back);
        eprintln!("htod/dtoh round-trip OK: {back:?}");
    }

    /// Batched f32-weight GEMM parity vs the CPU oracle. Non-square (oc ≠ ic) and
    /// several rows, so a transposed layout or an m/n/k swap is caught.
    #[test]
    #[ignore = "requires a CUDA device; run with --features cuda -- --ignored --nocapture"]
    fn matmul_batch_f32_parity() {
        let Some(b) = cuda() else { return };
        let (oc, ic, rows) = (130usize, 96usize, 5usize);
        let w = QMatrix::f32(noise(oc * ic, 1).into(), oc, ic).unwrap();
        let x = noise(rows * ic, 2);
        let (mut got, mut want) = (vec![0.0; rows * oc], vec![0.0; rows * oc]);
        b.matmul_batch(&mut got, &x, &w, rows);
        CpuBackend.matmul_batch(&mut want, &x, &w, rows);
        close_f16(&got, &want, ic);
    }

    /// Single-row (classifier) GEMM parity — the `rows = 1` path.
    #[test]
    #[ignore = "requires a CUDA device; run with --features cuda -- --ignored --nocapture"]
    fn matmul_single_f32_parity() {
        let Some(b) = cuda() else { return };
        let (oc, ic) = (64usize, 128usize);
        let w = QMatrix::f32(noise(oc * ic, 3).into(), oc, ic).unwrap();
        let x = noise(ic, 4);
        let (mut got, mut want) = (vec![0.0; oc], vec![0.0; oc]);
        b.matmul(&mut got, &x, &w);
        CpuBackend.matmul(&mut want, &x, &w);
        close_f16(&got, &want, ic);
    }

    /// Quantized-weight (Q8_0) GEMM parity: the weight is dequantized to f16 on
    /// upload, so this checks the dequant→f16→GEMM path against the CPU's exact
    /// Q8_0 dot. Two rows of 64 cols (Q8_0 block size 32).
    #[test]
    #[ignore = "requires a CUDA device; run with --features cuda -- --ignored --nocapture"]
    fn matmul_batch_q8_0_parity() {
        let Some(b) = cuda() else { return };
        let (oc, ic, rows) = (8usize, 64usize, 3usize);
        let wf = noise(oc * ic, 5);
        let w = QMatrix::quant(GgmlType::Q8_0, quantize_q8_0(&wf).into(), oc, ic).unwrap();
        let x = noise(rows * ic, 6);
        let (mut got, mut want) = (vec![0.0; rows * oc], vec![0.0; rows * oc]);
        b.matmul_batch(&mut got, &x, &w, rows);
        CpuBackend.matmul_batch(&mut want, &x, &w, rows);
        // Looser: Q8_0 weight quant error on top of the f16 GEMM rounding.
        close_f16(&got, &want, ic);
    }

    /// End-to-end coherence: a full `forward_prefill` over a small synthetic f32
    /// model on CUDA must track the CPU. The non-GEMM ops are identical (both
    /// delegate to the CPU), so any divergence is the f16 GEMMs — a layout bug in
    /// any of the seven matmul shapes would wreck the logits. Asserts the top
    /// token agrees and the logit vectors stay tightly correlated.
    #[test]
    #[ignore = "requires a CUDA device; run with --features cuda -- --ignored --nocapture"]
    fn prefill_coherent_with_cpu() {
        let Some(b) = cuda() else { return };
        let c = crate::Config {
            dim: 256,
            hidden_dim: 512,
            n_layers: 4,
            n_heads: 8,
            n_kv_heads: 8,
            vocab_size: 512,
            seq_len: 64,
            shared_weights: true,
            ..Default::default()
        };
        let bytes = crate::dummy::synthetic_gguf_typed(&c, GgmlType::F32);
        let gguf = crate::Gguf::parse(&bytes).unwrap();
        let model = crate::Model::from_gguf(&gguf).unwrap();
        let tokens: Vec<usize> = (0..16).map(|i| (i * 7) % c.vocab_size).collect();

        let logits = |backend: &dyn Backend| {
            let mut s = crate::RunState::new(&model.config);
            crate::forward_prefill(&model, &mut s, backend, &tokens, 0);
            s.logits().to_vec()
        };
        let cuda_logits = logits(&b);
        let cpu_logits = logits(&CpuBackend::new());

        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, x| a.1.partial_cmp(x.1).unwrap())
                .unwrap()
                .0
        };
        let (ca, pa) = (argmax(&cuda_logits), argmax(&cpu_logits));

        // Relative L2 error between the two logit vectors.
        let num: f32 = cuda_logits
            .iter()
            .zip(&cpu_logits)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let den: f32 = cpu_logits.iter().map(|v| v * v).sum();
        let rel = (num / den).sqrt();
        eprintln!("prefill coherence: argmax cuda={ca} cpu={pa}, relative L2 error {rel:.4}");

        assert_eq!(ca, pa, "top token diverged (cuda {ca} vs cpu {pa})");
        assert!(rel < 0.1, "logit vectors diverged: relative L2 error {rel}");
    }

    /// Prefill throughput: CUDA (cuBLASLt f16) vs CPU on the same shape the wgpu
    /// `bench_prefill_gpu_vs_cpu` uses, so the number is comparable to the wgpu
    /// f32 prefill figure in PERFORMANCE.md. Honest M1 cost: weights are resident
    /// (warm-up call) but activations still round-trip the host per matmul (M2
    /// removes that).
    #[test]
    #[ignore = "timing benchmark; run with --release --features cuda -- --ignored --nocapture"]
    fn bench_prefill_cuda_vs_cpu() {
        use std::time::Instant;
        let Some(b) = cuda() else { return };

        let c = crate::Config {
            dim: 1024,
            hidden_dim: 2816,
            n_layers: 4,
            n_heads: 16,
            n_kv_heads: 16,
            vocab_size: 4096,
            seq_len: 512,
            shared_weights: true,
            ..Default::default()
        };
        let bytes = crate::dummy::synthetic_gguf_typed(&c, GgmlType::F32);
        let gguf = crate::Gguf::parse(&bytes).unwrap();
        let model = crate::Model::from_gguf(&gguf).unwrap();
        let tokens: Vec<usize> = (0..256).map(|i| (i * 7) % c.vocab_size).collect();

        let prefill = |backend: &dyn Backend| {
            let mut s = crate::RunState::new(&model.config);
            crate::forward_prefill(&model, &mut s, backend, &tokens, 0);
        };

        prefill(&b); // warm up: dequant + upload weights, make them resident
        let t = Instant::now();
        prefill(&b);
        let cuda = t.elapsed();

        let cpu_backend = CpuBackend::new();
        let t = Instant::now();
        prefill(&cpu_backend);
        let cpu = t.elapsed();

        eprintln!(
            "prefill {} tokens, dim {} x{} layers: cuda={cuda:?} cpu={cpu:?} \
             -> cuda is {:.2}x the cpu time ({:.0} vs {:.0} tok/s)",
            tokens.len(),
            c.dim,
            c.n_layers,
            cuda.as_secs_f64() / cpu.as_secs_f64(),
            tokens.len() as f64 / cuda.as_secs_f64(),
            tokens.len() as f64 / cpu.as_secs_f64(),
        );
    }
}
