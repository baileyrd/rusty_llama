//! NVIDIA CUDA backend (cuBLASLt tensor cores).
//!
//! A third [`Backend`] implementation — alongside the [`CpuBackend`] and the
//! portable wgpu [`GpuBackend`](crate::backend::GpuBackend) — targeting NVIDIA
//! tensor cores via cuBLASLt, gated behind the `cuda` cargo feature and built on
//! the [`cudarc`] crate. The portable wgpu cooperative-matrix path was spiked
//! and ruled out on this hardware (wgpu 29 wires only 8×8 f32, which the Vulkan
//! driver doesn't expose; see `PERFORMANCE.md` item 3), so a cuBLASLt TF32 GEMM
//! is the route to the prefill tensor-core win.
//!
//! ## Milestones
//!
//! - **M0** (done): device + cuBLASLt handle init with a graceful no-CUDA
//!   fallback, ops delegated to the [`CpuBackend`].
//! - **M1** (done): the `forward_prefill` matmuls + classifier run on cuBLASLt
//!   **TF32 tensor cores** ([`CudaBackend::gemm`]); the rest delegates to the CPU.
//! - **M2** (this file): on-device kernels for the non-GEMM ops (nvrtc, pure f32)
//!   so a fused, resident prefill runs a whole layer without host round-trips.
//!   Decode (`forward_step`) stays on the CPU — at batch=1 it is
//!   GEMV/bandwidth-bound, where tensor cores help little.
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
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use crate::backend::{Backend, CpuBackend};
use crate::model::{Model, RunState};
use crate::quant::dequantize;
use crate::tensor::QMatrix;

/// On-device kernels for the non-GEMM primitives, compiled once at startup via
/// nvrtc (no build-time `nvcc`, and — being pure f32 — no `cuda_fp16.h`, so
/// nvrtc needs no toolkit include path). Everything is f32: the GEMMs run on
/// **TF32 tensor cores** (cuBLASLt `Matmul<f32>` → `CUBLAS_COMPUTE_32F_FAST_TF32`),
/// which take f32 in/out, so no conversion kernels are needed. Each kernel
/// mirrors the corresponding [`CpuBackend`] op exactly (parity-tested). These run
/// resident inside the fused prefill (M2b) so a whole layer executes without host
/// round-trips.
const KERNEL_SRC: &str = r#"
extern "C" __global__ void add_kernel(float* out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += x[i];
}
extern "C" __global__ void swiglu_kernel(float* hb, const float* hb2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = hb[i]; hb[i] = (v / (1.0f + expf(-v))) * hb2[i]; }
}
// One block per row; blockDim.x (a power of two) threads reduce the sum of
// squares in shared memory. Matches CpuBackend::rmsnorm.
extern "C" __global__ void rmsnorm_kernel(float* out, const float* x, const float* w,
                                          int dim, float eps) {
    int row = blockIdx.x;
    const float* xr = x + (long long)row * dim;
    float* outr = out + (long long)row * dim;
    extern __shared__ float sh[];
    float local = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) { float v = xr[i]; local += v * v; }
    sh[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float inv = 1.0f / sqrtf(sh[0] / (float)dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) outr[i] = xr[i] * inv * w[i];
}
// grid (rows, ceil(n_pairs/block)); one thread per (row, even/odd pair).
// Matches CpuBackend::rope (partial-rotary tail left untouched).
extern "C" __global__ void rope_kernel(float* q, float* k, const float* inv_freq,
                                       int pos_base, int head_size, int kv_dim,
                                       int q_dim, int n_freqs, float mscale) {
    int row = blockIdx.x;
    int pair = blockIdx.y * blockDim.x + threadIdx.x;
    int n_pairs = q_dim / 2;
    if (pair >= n_pairs) return;
    int i = pair * 2;
    int local_pair = (i % head_size) / 2;
    if (local_pair >= n_freqs) return;
    float val = (float)(pos_base + row) * inv_freq[local_pair];
    float fcr = cosf(val) * mscale, fci = sinf(val) * mscale;
    float* qr = q + (long long)row * q_dim;
    float q0 = qr[i], q1 = qr[i + 1];
    qr[i] = q0 * fcr - q1 * fci;
    qr[i + 1] = q0 * fci + q1 * fcr;
    if (i < kv_dim) {
        float* kr = k + (long long)row * kv_dim;
        float k0 = kr[i], k1 = kr[i + 1];
        kr[i] = k0 * fcr - k1 * fci;
        kr[i + 1] = k0 * fci + k1 * fcr;
    }
}
// One thread per (row, head); causal GQA over keys 0..=pos_base+row. `att` is
// scratch of (rows*n_heads, seq_len). Matches CpuBackend::attention_batch.
extern "C" __global__ void attention_kernel(float* out, const float* q,
        const float* kcache, const float* vcache, float* att,
        int rows, int pos_base, int n_heads, int kv_mul, int head_size,
        int seq_len, int kv_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * n_heads) return;
    int r = idx / n_heads;
    int h = idx % n_heads;
    int pos = pos_base + r;
    int dim = n_heads * head_size;
    int kv_off = (h / kv_mul) * head_size;
    const float* qh = q + (long long)r * dim + (long long)h * head_size;
    float* sc = att + (long long)idx * seq_len;
    float scale = 1.0f / sqrtf((float)head_size);
    float maxv = -1e30f;
    for (int t = 0; t <= pos; t++) {
        const float* kt = kcache + (long long)t * kv_dim + kv_off;
        float s = 0.0f;
        for (int i = 0; i < head_size; i++) s += qh[i] * kt[i];
        s *= scale;
        sc[t] = s;
        if (s > maxv) maxv = s;
    }
    float sum = 0.0f;
    for (int t = 0; t <= pos; t++) { float e = expf(sc[t] - maxv); sc[t] = e; sum += e; }
    float invsum = 1.0f / sum;
    float* outh = out + (long long)r * dim + (long long)h * head_size;
    for (int i = 0; i < head_size; i++) {
        float o = 0.0f;
        for (int t = 0; t <= pos; t++) {
            const float* vt = vcache + (long long)t * kv_dim + kv_off;
            o += sc[t] * vt[i];
        }
        outh[i] = o * invsum;
    }
}
"#;

/// Compiled handles for the [`KERNEL_SRC`] kernels.
struct Kernels {
    add: CudaFunction,
    swiglu: CudaFunction,
    rmsnorm: CudaFunction,
    rope: CudaFunction,
    attention: CudaFunction,
}

impl Kernels {
    fn compile(ctx: &Arc<CudaContext>) -> Result<Self, String> {
        let ptx = compile_ptx(KERNEL_SRC).map_err(|e| format!("nvrtc compile failed: {e:?}"))?;
        let m = ctx
            .load_module(ptx)
            .map_err(|e| format!("module load failed: {e:?}"))?;
        let f = |name: &str| {
            m.load_function(name)
                .map_err(|e| format!("missing kernel '{name}': {e:?}"))
        };
        Ok(Kernels {
            add: f("add_kernel")?,
            swiglu: f("swiglu_kernel")?,
            rmsnorm: f("rmsnorm_kernel")?,
            rope: f("rope_kernel")?,
            attention: f("attention_kernel")?,
        })
    }
}

/// CUDA backend: a resident device context, stream, and cuBLASLt handle.
///
/// The matmuls (the `forward_prefill` GEMMs and the classifier) run on cuBLASLt
/// **TF32 tensor cores** (`Matmul<f32>`); the remaining elementwise/attention
/// ops delegate to an internal [`CpuBackend`] (M1) or run as on-device kernels
/// inside the fused prefill (M2). Dequantized f32 weights are uploaded once and
/// cached, keyed on the source weight's data pointer (stable across the run,
/// since weights are borrowed from the model) — mirroring the wgpu backend.
pub struct CudaBackend {
    // `ctx` is held only to keep the CUDA context alive for the lifetime of the
    // backend (the stream / handle / slices reference it); it is never read
    // directly, hence the allow.
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlasLT,
    kernels: Kernels,
    /// f32 device weights, keyed by source data pointer (see [`Self::weight_f32`]).
    weights: Mutex<HashMap<usize, Arc<CudaSlice<f32>>>>,
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
        let kernels = Kernels::compile(&ctx)?;
        let device_name = ctx
            .name()
            .map_err(|e| format!("CUDA device-name query failed: {e:?}"))?;

        Ok(CudaBackend {
            ctx,
            stream,
            blas,
            kernels,
            weights: Mutex::new(HashMap::new()),
            cpu: CpuBackend::new(),
            device_name,
        })
    }

    /// The selected CUDA device's name (for logging / benchmarks).
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// The f32 device copy of weight matrix `w` (row-major `rows`×`cols`),
    /// uploaded once and cached by source data pointer. Quantized weights are
    /// dequantized to f32 on the host first.
    fn weight_f32(&self, w: &QMatrix) -> Arc<CudaSlice<f32>> {
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
        let dev = self
            .stream
            .clone_htod(f32_data.as_ref())
            .expect("upload f32 weight to device");

        let arc = Arc::new(dev);
        self.weights.lock().unwrap().insert(key, arc.clone());
        arc
    }

    /// `out(rows × oc) = x(rows × ic) · Wᵀ` on cuBLASLt TF32 tensor cores
    /// (`Matmul<f32>` → f32 in/out, TF32 compute). `x`/`out` are row-major host
    /// slices; the resident-device variant is [`Self::gemm_dev`].
    fn gemm(&self, out: &mut [f32], x: &[f32], w: &QMatrix, rows: usize) {
        let oc = w.rows();
        let x_dev = self.stream.clone_htod(x).expect("upload activation");
        let mut c_dev = self
            .stream
            .alloc_zeros::<f32>(rows * oc)
            .expect("alloc gemm output");
        self.gemm_dev(&mut c_dev, &x_dev, w, rows);
        self.stream.synchronize().expect("stream synchronize");
        let c_host: Vec<f32> = self.stream.clone_dtoh(&c_dev).expect("download gemm output");
        out.copy_from_slice(&c_host);
    }

    /// Resident-device GEMM: `c(rows × oc) = x(rows × ic) · Wᵀ`, all device
    /// buffers, no host round-trip (the building block for the fused prefill).
    ///
    /// cuBLASLt is column-major, so we compute the transpose `cᵀ = W · xᵀ` by
    /// feeding our row-major `W` and `x` buffers directly with `transa = true`
    /// (dimensions swapped: `m = oc`, `n = rows`, `k = ic`). Derivation
    /// cross-checked against cudarc's `test_matmul_*`, guarded by the per-op
    /// parity test against the CPU backend.
    fn gemm_dev(&self, c: &mut CudaSlice<f32>, x: &CudaSlice<f32>, w: &QMatrix, rows: usize) {
        let (oc, ic) = (w.rows(), w.cols());
        debug_assert_eq!(x.len(), rows * ic);
        debug_assert_eq!(c.len(), rows * oc);
        let w_dev = self.weight_f32(w);
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
                .matmul(cfg, w_dev.as_ref(), x, c, None, None)
                .expect("cuBLASLt TF32 matmul");
        }
    }

    // --- On-device primitives -------------------------------------------
    //
    // These run [`KERNEL_SRC`] kernels on resident device buffers (no host
    // round-trips). They are the building blocks of the fused prefill (M2b) and
    // are each parity-tested against the matching `CpuBackend` op.

    /// `out[i] += x[i]` over the whole buffer.
    #[allow(dead_code)] // wired into the fused prefill in M2b
    fn dev_add(&self, out: &mut CudaSlice<f32>, x: &CudaSlice<f32>, n: usize) {
        let n_i = n as i32;
        let mut b = self.stream.launch_builder(&self.kernels.add);
        b.arg(out).arg(x).arg(&n_i);
        unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("launch add");
    }

    /// `hb[i] = silu(hb[i]) * hb2[i]` over the whole buffer.
    #[allow(dead_code)] // wired into the fused prefill in M2b
    fn dev_swiglu(&self, hb: &mut CudaSlice<f32>, hb2: &CudaSlice<f32>, n: usize) {
        let n_i = n as i32;
        let mut b = self.stream.launch_builder(&self.kernels.swiglu);
        b.arg(hb).arg(hb2).arg(&n_i);
        unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("launch swiglu");
    }

    /// Per-row RMSNorm of `x` (`rows` × `dim`) into `out` against shared `w`.
    #[allow(dead_code)] // wired into the fused prefill in M2b
    fn dev_rmsnorm(
        &self,
        out: &mut CudaSlice<f32>,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        rows: usize,
        dim: usize,
        eps: f32,
    ) {
        let dim_i = dim as i32;
        let block = 256u32; // power of two for the shared-memory reduction
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: block * 4,
        };
        let mut b = self.stream.launch_builder(&self.kernels.rmsnorm);
        b.arg(out).arg(x).arg(w).arg(&dim_i).arg(&eps);
        unsafe { b.launch(cfg) }.expect("launch rmsnorm");
    }

    /// Batched RoPE: `q` (`rows`×`q_dim`) and `k` (`rows`×`kv_dim`) rotated in
    /// place; row `r` at absolute position `pos_base + r`.
    #[allow(clippy::too_many_arguments, dead_code)] // wired into the fused prefill in M2b
    fn dev_rope(
        &self,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
        inv: &CudaSlice<f32>,
        rows: usize,
        pos_base: usize,
        head_size: usize,
        kv_dim: usize,
        q_dim: usize,
        n_freqs: usize,
        mscale: f32,
    ) {
        let n_pairs = (q_dim / 2) as u32;
        let block = 64u32;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, n_pairs.div_ceil(block), 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (pos_i, hs_i, kv_i, qd_i, nf_i) = (
            pos_base as i32,
            head_size as i32,
            kv_dim as i32,
            q_dim as i32,
            n_freqs as i32,
        );
        let mut b = self.stream.launch_builder(&self.kernels.rope);
        b.arg(q)
            .arg(k)
            .arg(inv)
            .arg(&pos_i)
            .arg(&hs_i)
            .arg(&kv_i)
            .arg(&qd_i)
            .arg(&nf_i)
            .arg(&mscale);
        unsafe { b.launch(cfg) }.expect("launch rope");
    }

    /// Batched causal GQA: query row `r` (at `pos_base + r`) over cached keys
    /// `0..=pos_base+r`. `att` is scratch of `rows*n_heads*seq_len`.
    #[allow(clippy::too_many_arguments, dead_code)] // wired into the fused prefill in M2b
    fn dev_attention(
        &self,
        out: &mut CudaSlice<f32>,
        q: &CudaSlice<f32>,
        kcache: &CudaSlice<f32>,
        vcache: &CudaSlice<f32>,
        att: &mut CudaSlice<f32>,
        rows: usize,
        pos_base: usize,
        n_heads: usize,
        kv_mul: usize,
        head_size: usize,
        seq_len: usize,
        kv_dim: usize,
    ) {
        let (rows_i, pos_i, nh_i, kvm_i, hs_i, sl_i, kv_i) = (
            rows as i32,
            pos_base as i32,
            n_heads as i32,
            kv_mul as i32,
            head_size as i32,
            seq_len as i32,
            kv_dim as i32,
        );
        let mut b = self.stream.launch_builder(&self.kernels.attention);
        b.arg(out)
            .arg(q)
            .arg(kcache)
            .arg(vcache)
            .arg(att)
            .arg(&rows_i)
            .arg(&pos_i)
            .arg(&nh_i)
            .arg(&kvm_i)
            .arg(&hs_i)
            .arg(&sl_i)
            .arg(&kv_i);
        unsafe { b.launch(LaunchConfig::for_num_elems((rows * n_heads) as u32)) }
            .expect("launch attention");
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
        self.gemm(out, x, w, 1);
    }

    fn matmul_batch(&self, out: &mut [f32], x: &[f32], w: &QMatrix, rows: usize) {
        self.gemm(out, x, w, rows);
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

    /// Assert two slices match within a TF32-GEMM tolerance: tensor-core TF32
    /// keeps ~10 mantissa bits (≈2⁻¹¹ relative) and accumulates in f32, so error
    /// grows with the reduction length `ic`. Generous enough for that, tight
    /// enough that a wrong layout (transpose / m·n·k swap) — which yields wholly
    /// different values — fails loudly.
    fn close_tc(got: &[f32], want: &[f32], ic: usize) {
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
        eprintln!("  close_tc ok: maxdiff {maxdiff:.4} (tol {tol:.4} + 3%·|w|, ic={ic})");
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
        close_tc(&got, &want, ic);
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
        close_tc(&got, &want, ic);
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
        close_tc(&got, &want, ic);
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

    // --- M2a: on-device kernel parity (vs the CpuBackend oracle) ---------

    /// Assert two slices match within `atol + rtol·|want|`.
    fn close_approx(got: &[f32], want: &[f32], atol: f32, rtol: f32) {
        assert_eq!(got.len(), want.len());
        let mut maxdiff = 0.0f32;
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            let d = (g - w).abs();
            maxdiff = maxdiff.max(d);
            assert!(d <= atol + rtol * w.abs(), "idx {i}: cuda {g} vs cpu {w} (diff {d})");
        }
        eprintln!("  ok: maxdiff {maxdiff:.2e}");
    }

    /// Base RoPE inverse frequencies for a full-rotary head: `θ^(-2j/head_size)`.
    fn base_inv_freq(head_size: usize, theta: f32) -> Vec<f32> {
        (0..head_size / 2)
            .map(|j| (theta as f64).powf(-2.0 * j as f64 / head_size as f64) as f32)
            .collect()
    }

    #[test]
    #[ignore = "requires a CUDA device; run with --features cuda -- --ignored --nocapture"]
    fn dev_add_parity() {
        let Some(b) = cuda() else { return };
        let n = 257usize;
        let (mut x, y) = (noise(n, 10), noise(n, 11));
        let mut xd = b.stream.clone_htod(&x).unwrap();
        let yd = b.stream.clone_htod(&y).unwrap();
        b.dev_add(&mut xd, &yd, n);
        b.stream.synchronize().unwrap();
        let got = b.stream.clone_dtoh(&xd).unwrap();
        CpuBackend.add(&mut x, &y);
        close_approx(&got, &x, 1e-6, 0.0);
    }

    #[test]
    #[ignore = "requires a CUDA device; run with --features cuda -- --ignored --nocapture"]
    fn dev_swiglu_parity() {
        let Some(b) = cuda() else { return };
        let n = 300usize;
        let (mut hb, hb2) = (noise(n, 12), noise(n, 13));
        let mut hd = b.stream.clone_htod(&hb).unwrap();
        let h2d = b.stream.clone_htod(&hb2).unwrap();
        b.dev_swiglu(&mut hd, &h2d, n);
        b.stream.synchronize().unwrap();
        let got = b.stream.clone_dtoh(&hd).unwrap();
        CpuBackend.swiglu(&mut hb, &hb2);
        close_approx(&got, &hb, 1e-5, 1e-4);
    }

    #[test]
    #[ignore = "requires a CUDA device; run with --features cuda -- --ignored --nocapture"]
    fn dev_rmsnorm_parity() {
        let Some(b) = cuda() else { return };
        let (dim, rows) = (288usize, 7usize);
        let x = noise(rows * dim, 14);
        let w = noise(dim, 15);
        let xd = b.stream.clone_htod(&x).unwrap();
        let wd = b.stream.clone_htod(&w).unwrap();
        let mut outd = b.stream.alloc_zeros::<f32>(rows * dim).unwrap();
        b.dev_rmsnorm(&mut outd, &xd, &wd, rows, dim, 1e-5);
        b.stream.synchronize().unwrap();
        let got = b.stream.clone_dtoh(&outd).unwrap();
        let mut want = vec![0.0; rows * dim];
        CpuBackend.rmsnorm_batch(&mut want, &x, &w, 1e-5, rows);
        close_approx(&got, &want, 1e-4, 1e-3);
    }

    #[test]
    #[ignore = "requires a CUDA device; run with --features cuda -- --ignored --nocapture"]
    fn dev_rope_parity() {
        let Some(b) = cuda() else { return };
        let (head_size, n) = (8usize, 6usize);
        let (q_dim, kv_dim) = (2 * head_size, head_size);
        let inv = base_inv_freq(head_size, 10000.0);
        let (mut q, mut k) = (noise(n * q_dim, 16), noise(n * kv_dim, 17));
        let mut qd = b.stream.clone_htod(&q).unwrap();
        let mut kd = b.stream.clone_htod(&k).unwrap();
        let invd = b.stream.clone_htod(&inv).unwrap();
        b.dev_rope(&mut qd, &mut kd, &invd, n, 3, head_size, kv_dim, q_dim, inv.len(), 1.0);
        b.stream.synchronize().unwrap();
        let gq = b.stream.clone_dtoh(&qd).unwrap();
        let gk = b.stream.clone_dtoh(&kd).unwrap();
        CpuBackend.rope_batch(&mut q, &mut k, 3, n, head_size, kv_dim, q_dim, &inv, 1.0);
        close_approx(&gq, &q, 1e-4, 1e-4);
        close_approx(&gk, &k, 1e-4, 1e-4);
    }

    #[test]
    #[ignore = "requires a CUDA device; run with --features cuda -- --ignored --nocapture"]
    fn dev_attention_parity() {
        let Some(b) = cuda() else { return };
        let (n_heads, n_kv_heads, head_size, seq_len) = (4usize, 2usize, 8usize, 16usize);
        let kv_dim = n_kv_heads * head_size;
        let dim = n_heads * head_size;
        let n = 10usize; // rows / positions 0..10
        let q = noise(n * dim, 1);
        let kc = noise(seq_len * kv_dim, 2);
        let vc = noise(seq_len * kv_dim, 3);

        let qd = b.stream.clone_htod(&q).unwrap();
        let kcd = b.stream.clone_htod(&kc).unwrap();
        let vcd = b.stream.clone_htod(&vc).unwrap();
        let mut outd = b.stream.alloc_zeros::<f32>(n * dim).unwrap();
        let mut attd = b.stream.alloc_zeros::<f32>(n * n_heads * seq_len).unwrap();
        b.dev_attention(
            &mut outd, &qd, &kcd, &vcd, &mut attd, n, 0, n_heads, n_heads / n_kv_heads, head_size,
            seq_len, kv_dim,
        );
        b.stream.synchronize().unwrap();
        let got = b.stream.clone_dtoh(&outd).unwrap();

        let mut want = vec![0.0; n * dim];
        let mut att = vec![0.0; n_heads * seq_len];
        CpuBackend.attention_batch(
            &mut want, &q, &kc, &vc, &mut att, 0, n, n_heads, n_kv_heads, head_size, seq_len, kv_dim,
        );
        close_approx(&got, &want, 1e-4, 1e-4);
    }

}
