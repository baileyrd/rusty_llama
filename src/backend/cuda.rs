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
//! This file currently implements **M0**: device + cuBLASLt handle
//! initialization with a graceful no-CUDA fallback, plus op delegation to the
//! [`CpuBackend`] so the backend is already correct end-to-end (if not yet
//! accelerated). **M1** overrides the matmuls with cuBLASLt f16 GEMMs — the
//! resident device context, stream, and handle held here are what that path
//! will use.
//!
//! ## Why dynamic loading can't just `?`-propagate a missing driver
//!
//! cudarc's `dynamic-loading` backend resolves the CUDA driver library lazily on
//! first use and **panics** (via `panic_no_lib_found`) when it is absent — so on
//! a CUDA-less host (headless CI, no NVIDIA GPU) `CudaContext::new` aborts rather
//! than returning `Err`. [`cudarc::driver::sys::is_culib_present`] is the
//! non-panicking gatekeeper; [`CudaBackend::new`] checks it first.

use std::sync::Arc;

use cudarc::cublaslt::CudaBlasLT;
use cudarc::driver::{CudaContext, CudaStream};

use crate::backend::{Backend, CpuBackend};
use crate::model::{Model, RunState};
use crate::tensor::QMatrix;

/// CUDA backend: a resident device context, stream, and cuBLASLt handle.
///
/// Until the GEMM kernels land (M1+), every [`Backend`] op delegates to an
/// internal [`CpuBackend`], so results are already correct — just CPU-speed. The
/// `ctx` / `stream` / `blas` fields are what the cuBLASLt path will use; they are
/// `#[allow(dead_code)]` only until then.
pub struct CudaBackend {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    #[allow(dead_code)]
    stream: Arc<CudaStream>,
    #[allow(dead_code)]
    blas: CudaBlasLT,
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
            cpu: CpuBackend::new(),
            device_name,
        })
    }

    /// The selected CUDA device's name (for logging / benchmarks).
    pub fn device_name(&self) -> &str {
        &self.device_name
    }
}

// M0: delegate every primitive to the CPU backend. The backend is correct (and
// `forward_prefill`/`generate` work over it) — M1 replaces the matmuls with
// cuBLASLt GEMMs and overrides the batched variants.
impl Backend for CudaBackend {
    fn rmsnorm(&self, out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
        self.cpu.rmsnorm(out, x, weight, eps);
    }

    fn matmul(&self, out: &mut [f32], x: &[f32], w: &QMatrix) {
        self.cpu.matmul(out, x, w);
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

    /// M0 smoke test: prove the toolchain end-to-end — init CUDA, read the device
    /// name, and round-trip a buffer host→device→host through the stream (which
    /// also confirms the dynamically-loaded driver + cuBLASLt resolve at run
    /// time). Skips cleanly (no failure) when no CUDA device is present, so CI
    /// without a GPU stays green.
    #[test]
    #[ignore = "requires a CUDA device; run with --features cuda -- --ignored --nocapture"]
    fn cuda_smoke() {
        let b = match CudaBackend::new() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping CUDA smoke test: {e}");
                return;
            }
        };
        eprintln!("CUDA device: {}", b.device_name());

        let host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let dev = b.stream.clone_htod(&host).expect("host->device copy");
        let back: Vec<f32> = b.stream.clone_dtoh(&dev).expect("device->host copy");
        assert_eq!(host, back);
        eprintln!("htod/dtoh round-trip OK: {back:?}");
    }
}
