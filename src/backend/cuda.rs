//! NVIDIA CUDA backend (cuBLASLt tensor cores).
//!
//! A third [`Backend`] implementation — alongside the [`CpuBackend`] and the
//! portable wgpu [`GpuBackend`](crate::backend::GpuBackend) — targeting NVIDIA
//! tensor cores via cuBLASLt, gated behind the `cuda` cargo feature and built on
//! the [`cudarc`] crate. The portable wgpu cooperative-matrix path was spiked
//! and ruled out on this hardware (wgpu 29 wires only 8×8 f32, which the Vulkan
//! driver doesn't expose; see `PERFORMANCE.md` item 3), so a cuBLASLt f16 GEMM
//! is the route to the prefill tensor-core win.
//!
//! ## Milestones
//!
//! - **M0** (done): device + cuBLASLt handle init with a graceful no-CUDA
//!   fallback, ops delegated to the [`CpuBackend`].
//! - **M1** (done): the `forward_prefill` matmuls + classifier run on cuBLASLt
//!   tensor cores ([`CudaBackend::gemm`]); the rest delegates to the CPU.
//! - **M2** (done): on-device nvrtc kernels for the non-GEMM ops, so a fused,
//!   resident prefill runs a whole layer without host round-trips. Cooperative
//!   tiled attention.
//! - **f16 weights** (done): weights cached as f16 and the GEMM runs as
//!   `Matmul<f16>` — half the weight bandwidth/VRAM of f32, and f16 tensor cores
//!   beat TF32. Activations convert f32↔f16 on-device around each GEMM.
//! - **Resident decode** (done): `forward_step` runs a resident single-token
//!   decode — KV cache + activations stay on device across steps, batch-1 GEMVs
//!   via the f16 GEMM + the nvrtc kernels, the prompt's host KV uploaded once.
//!   At batch=1 decode is GEMV/bandwidth-bound, so f16 weight bandwidth (not
//!   compute) is the limiter (~3.4× behind llama.cpp; see `PERFORMANCE.md`).
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
use half::f16;

use crate::backend::{Backend, CpuBackend};
use crate::model::{Model, RunState};
use crate::quant::dequantize;
use crate::tensor::QMatrix;

/// On-device kernels for the non-GEMM primitives + f32↔f16 conversion, compiled
/// once at startup via nvrtc (no build-time `nvcc`). Activations are kept in f32;
/// the GEMMs run on cuBLASLt **f16 tensor cores** (`Matmul<f16>`), so the
/// `cvt_*` kernels narrow activations to f16 before each GEMM and widen the
/// result back — bridged with **inline PTX** (`cvt.rn.f16.f32` / `cvt.f32.f16`)
/// over `unsigned short` buffers, so nvrtc needs neither `cuda_fp16.h` nor a
/// toolkit include path. Each compute kernel mirrors the corresponding
/// [`CpuBackend`] op exactly (parity-tested); they run resident inside the fused
/// prefill so a whole layer executes without host round-trips.
const KERNEL_SRC: &str = r#"
// f32 -> f16 (stored as u16), round-to-nearest, via inline PTX (no cuda_fp16.h).
extern "C" __global__ void cvt_f32_f16(unsigned short* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned short h;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(in[i]));
        out[i] = h;
    }
}
// f16 (u16) -> f32 via inline PTX.
extern "C" __global__ void cvt_f16_f32(float* out, const unsigned short* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v;
        asm("cvt.f32.f16 %0, %1;" : "=f"(v) : "h"(in[i]));
        out[i] = v;
    }
}
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
// Causal GQA, one thread BLOCK per (row, head) — cooperative, not one thread
// per (row, head). Threads split the keys for the scores and the head dims for
// the output; the score row + softmax reduction live in shared memory (no
// global scratch). `blockDim.x` must be a power of two. Dynamic shared layout:
// [ sq: head_size | sc: seq_len | red: blockDim.x ] floats. Matches
// CpuBackend::attention_batch.
extern "C" __global__ void attention_kernel(float* out, const float* q,
        const float* kcache, const float* vcache,
        int rows, int pos_base, int n_heads, int kv_mul, int head_size,
        int seq_len, int kv_dim) {
    int idx = blockIdx.x;
    if (idx >= rows * n_heads) return;
    int r = idx / n_heads, h = idx % n_heads;
    int pos = pos_base + r;
    int dim = n_heads * head_size;
    int kv_off = (h / kv_mul) * head_size;
    const float* qh = q + (long long)r * dim + (long long)h * head_size;
    int tid = threadIdx.x, nt = blockDim.x;

    extern __shared__ float smem[];
    float* sq = smem;                 // head_size: this (row,head)'s query
    float* sc = sq + head_size;       // seq_len: scores, used 0..=pos
    float* red = sc + seq_len;        // nt: reduction scratch

    for (int i = tid; i < head_size; i += nt) sq[i] = qh[i];
    __syncthreads();

    float scale = 1.0f / sqrtf((float)head_size);
    // Scores q·k_t (threads split the keys), tracking a per-thread max.
    float lmax = -1e30f;
    for (int t = tid; t <= pos; t += nt) {
        const float* kt = kcache + (long long)t * kv_dim + kv_off;
        float s = 0.0f;
        for (int i = 0; i < head_size; i++) s += sq[i] * kt[i];
        s *= scale;
        sc[t] = s;
        if (s > lmax) lmax = s;
    }
    red[tid] = lmax;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    float maxv = red[0];
    __syncthreads();

    // Exponentiate (threads split the keys), reduce the sum.
    float lsum = 0.0f;
    for (int t = tid; t <= pos; t += nt) {
        float e = expf(sc[t] - maxv);
        sc[t] = e;
        lsum += e;
    }
    red[tid] = lsum;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float invsum = 1.0f / red[0];

    // Output: each thread owns a stride of head dims, summed over all keys.
    float* outh = out + (long long)r * dim + (long long)h * head_size;
    for (int i = tid; i < head_size; i += nt) {
        float acc = 0.0f;
        for (int t = 0; t <= pos; t++) {
            acc += sc[t] * vcache[(long long)t * kv_dim + kv_off + i];
        }
        outh[i] = acc * invsum;
    }
}
"#;

/// Compiled handles for the [`KERNEL_SRC`] kernels.
struct Kernels {
    cvt_f32_f16: CudaFunction,
    cvt_f16_f32: CudaFunction,
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
            cvt_f32_f16: f("cvt_f32_f16")?,
            cvt_f16_f32: f("cvt_f16_f32")?,
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
/// Device-resident state for single-token decode, kept across steps (the CUDA
/// analog of the wgpu `DecodeState`): the per-layer KV cache plus the activation
/// scratch all live on device, so a decode step runs without per-op host
/// round-trips. Built lazily on the first [`CudaBackend::forward_step`] and
/// reused while the model shape is unchanged. `kv_filled` tracks how many cache
/// rows are already on device, so a prior prefill's host KV is uploaded once.
struct DecodeCuda {
    // shape (for staleness checks)
    dim: usize,
    n_layers: usize,
    vocab: usize,
    seq_len: usize,
    kv_dim: usize,
    head_size: usize,
    hidden: usize,
    n_heads: usize,
    kv_mul: usize,
    n_freqs: usize,
    mscale: f32,
    // resident activations
    x: CudaSlice<f32>,
    xb: CudaSlice<f32>,
    xb2: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k_row: CudaSlice<f32>,
    v_row: CudaSlice<f32>,
    hb: CudaSlice<f32>,
    hb2: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    inv: CudaSlice<f32>,
    // resident RMSNorm weights (uploaded once, not per step)
    rms_att: Vec<CudaSlice<f32>>,
    rms_ffn: Vec<CudaSlice<f32>>,
    rms_final: CudaSlice<f32>,
    // resident KV cache, one buffer per layer (seq_len * kv_dim each)
    key: Vec<CudaSlice<f32>>,
    value: Vec<CudaSlice<f32>>,
    // host scratch for the per-step embedding lookup
    embed: Vec<f32>,
    // how many KV rows are already resident on device (uploaded or computed)
    kv_filled: usize,
}

/// The matmuls (the `forward_prefill` GEMMs and the classifier) run on cuBLASLt
/// **f16 tensor cores** (`Matmul<f16>`, f32 accumulate); the remaining
/// elementwise/attention ops delegate to an internal [`CpuBackend`] (M1) or run
/// as on-device kernels inside the fused prefill (M2). Weights are dequantized to
/// **f16** on the host, uploaded once, and cached, keyed on the source weight's
/// data pointer (stable across the run, since weights are borrowed from the
/// model) — half the bytes/VRAM of f32, mirroring the wgpu backend's caching.
pub struct CudaBackend {
    // `ctx` is held only to keep the CUDA context alive for the lifetime of the
    // backend (the stream / handle / slices reference it); it is never read
    // directly, hence the allow.
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlasLT,
    kernels: Kernels,
    /// f16 device weights, keyed by source data pointer (see [`Self::weight_f16`]).
    weights: Mutex<HashMap<usize, Arc<CudaSlice<f16>>>>,
    /// Resident single-token decode state (KV cache + activations), built lazily.
    decode: Mutex<Option<DecodeCuda>>,
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
        // SAFETY: this backend runs everything on a single stream
        // (`default_stream`), so device slices never cross streams — the
        // multi-stream synchronization that event tracking provides is pure
        // overhead here (2 events per alloc + per-launch bookkeeping). Disable it
        // before any slice is allocated (incl. the cuBLASLt workspace below).
        unsafe { ctx.disable_event_tracking() };
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
            decode: Mutex::new(None),
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
    /// dequantized to f32 on the host, then narrowed to f16 — half the
    /// bytes/VRAM of f32 and what the `Matmul<f16>` GEMM consumes.
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

    /// `out(rows × oc) = x(rows × ic) · Wᵀ` on cuBLASLt f16 tensor cores.
    /// `x`/`out` are row-major host slices; the resident-device variant is
    /// [`Self::gemm_dev`].
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
    /// Weights are f16; the f32 activation is narrowed to f16, run through
    /// `Matmul<f16>` (f32 accumulate), and the f16 result widened back to `c`.
    ///
    /// cuBLASLt is column-major, so we compute the transpose `cᵀ = W · xᵀ` by
    /// feeding our row-major `W` and `x` buffers directly with `transa = true`
    /// (dimensions swapped: `m = oc`, `n = rows`, `k = ic`). Derivation
    /// cross-checked against cudarc's `test_matmul_half`, guarded by the per-op
    /// parity test against the CPU backend.
    fn gemm_dev(&self, c: &mut CudaSlice<f32>, x: &CudaSlice<f32>, w: &QMatrix, rows: usize) {
        let (oc, ic) = (w.rows(), w.cols());
        debug_assert_eq!(x.len(), rows * ic);
        debug_assert_eq!(c.len(), rows * oc);
        let w_dev = self.weight_f16(w);

        // Narrow the activation to f16, run the f16 GEMM, widen back into `c`.
        // `x16` is fully written by the cvt kernel and `c16` by the GEMM
        // (beta = 0, so C is not read), so neither needs zeroing — skip the
        // memset with an uninitialized alloc.
        // SAFETY: both buffers are written in full before they are read.
        let mut x16 = unsafe { self.stream.alloc::<f16>(rows * ic) }.expect("alloc x16");
        self.dev_cvt_to_f16(&mut x16, x, rows * ic);
        let mut c16 = unsafe { self.stream.alloc::<f16>(rows * oc) }.expect("alloc c16");

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
                .matmul(cfg, w_dev.as_ref(), &x16, &mut c16, None, None)
                .expect("cuBLASLt f16 matmul");
        }
        self.dev_cvt_to_f32(c, &c16, rows * oc);
    }

    // --- On-device primitives -------------------------------------------
    //
    // These run [`KERNEL_SRC`] kernels on resident device buffers (no host
    // round-trips). They are the building blocks of the fused prefill (M2b) and
    // are each parity-tested against the matching `CpuBackend` op.

    /// Narrow `src` (f32) into `dst` (f16) elementwise.
    fn dev_cvt_to_f16(&self, dst: &mut CudaSlice<f16>, src: &CudaSlice<f32>, n: usize) {
        let n_i = n as i32;
        let mut b = self.stream.launch_builder(&self.kernels.cvt_f32_f16);
        b.arg(dst).arg(src).arg(&n_i);
        unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("launch cvt_f32_f16");
    }

    /// Widen `src` (f16) into `dst` (f32) elementwise.
    fn dev_cvt_to_f32(&self, dst: &mut CudaSlice<f32>, src: &CudaSlice<f16>, n: usize) {
        let n_i = n as i32;
        let mut b = self.stream.launch_builder(&self.kernels.cvt_f16_f32);
        b.arg(dst).arg(src).arg(&n_i);
        unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("launch cvt_f16_f32");
    }

    /// `out[i] += x[i]` over the whole buffer.
    fn dev_add(&self, out: &mut CudaSlice<f32>, x: &CudaSlice<f32>, n: usize) {
        let n_i = n as i32;
        let mut b = self.stream.launch_builder(&self.kernels.add);
        b.arg(out).arg(x).arg(&n_i);
        unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("launch add");
    }

    /// `hb[i] = silu(hb[i]) * hb2[i]` over the whole buffer.
    fn dev_swiglu(&self, hb: &mut CudaSlice<f32>, hb2: &CudaSlice<f32>, n: usize) {
        let n_i = n as i32;
        let mut b = self.stream.launch_builder(&self.kernels.swiglu);
        b.arg(hb).arg(hb2).arg(&n_i);
        unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("launch swiglu");
    }

    /// Per-row RMSNorm of `x` (`rows` × `dim`) into `out` against shared `w`.
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
    #[allow(clippy::too_many_arguments)]
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
    /// `0..=pos_base+r`. One thread block per (row, head); scores + softmax live
    /// in shared memory, so no global scratch is needed.
    #[allow(clippy::too_many_arguments)]
    fn dev_attention(
        &self,
        out: &mut CudaSlice<f32>,
        q: &CudaSlice<f32>,
        kcache: &CudaSlice<f32>,
        vcache: &CudaSlice<f32>,
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
        // One block per (row, head); 128 (power-of-two) threads cooperate.
        // Shared: [ query head_size | scores seq_len | reduction 128 ] floats.
        const BLOCK: u32 = 128;
        let cfg = LaunchConfig {
            grid_dim: ((rows * n_heads) as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: (head_size as u32 + seq_len as u32 + BLOCK) * 4,
        };
        let mut b = self.stream.launch_builder(&self.kernels.attention);
        b.arg(out)
            .arg(q)
            .arg(kcache)
            .arg(vcache)
            .arg(&rows_i)
            .arg(&pos_i)
            .arg(&nh_i)
            .arg(&kvm_i)
            .arg(&hs_i)
            .arg(&sl_i)
            .arg(&kv_i);
        unsafe { b.launch(cfg) }.expect("launch attention");
    }

    /// Allocate the resident decode state for `model` (KV cache + activation
    /// scratch + RMSNorm weights, all on device). `kv_filled = 0`, so the first
    /// step uploads any host KV a prior prefill produced.
    fn build_decode(&self, model: &Model) -> DecodeCuda {
        let p = &model.config;
        let (dim, kv_dim, hidden, seq_len) = (p.dim, p.kv_dim(), p.hidden_dim, p.seq_len);
        let st = &self.stream;
        let alloc = |n: usize| st.alloc_zeros::<f32>(n).expect("alloc decode buffer");
        let up = |s: &[f32]| st.clone_htod(s).expect("upload decode weight");
        DecodeCuda {
            dim,
            n_layers: p.n_layers,
            vocab: p.vocab_size,
            seq_len,
            kv_dim,
            head_size: p.head_size(),
            hidden,
            n_heads: p.n_heads,
            kv_mul: p.n_heads / p.n_kv_heads,
            n_freqs: model.rope.inv_freq.len(),
            mscale: model.rope.mscale,
            x: alloc(dim),
            xb: alloc(dim),
            xb2: alloc(dim),
            q: alloc(dim),
            k_row: alloc(kv_dim),
            v_row: alloc(kv_dim),
            hb: alloc(hidden),
            hb2: alloc(hidden),
            logits: alloc(p.vocab_size),
            inv: up(&model.rope.inv_freq),
            rms_att: (0..p.n_layers)
                .map(|l| up(&model.weights.rms_att_weight[l * dim..l * dim + dim]))
                .collect(),
            rms_ffn: (0..p.n_layers)
                .map(|l| up(&model.weights.rms_ffn_weight[l * dim..l * dim + dim]))
                .collect(),
            rms_final: up(&model.weights.rms_final_weight[0..dim]),
            key: (0..p.n_layers).map(|_| alloc(seq_len * kv_dim)).collect(),
            value: (0..p.n_layers).map(|_| alloc(seq_len * kv_dim)).collect(),
            embed: vec![0.0; dim],
            kv_filled: 0,
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
        logit_softcap: f32,
    ) {
        self.cpu.attention(
            out, q, key_cache, value_cache, att, pos, n_heads, n_kv_heads, head_size, seq_len,
            kv_dim, logit_softcap,
        );
    }

    fn swiglu(&self, hb: &mut [f32], hb2: &[f32]) {
        self.cpu.swiglu(hb, hb2);
    }

    fn add(&self, out: &mut [f32], x: &[f32]) {
        self.cpu.add(out, x);
    }

    /// Resident fused single-token decode: the KV cache + activations stay on
    /// device across steps, so a step runs without per-op host round-trips (only
    /// the embedding goes up and the logits come down). The matmuls are batch-1
    /// GEMVs (bandwidth-bound — where f16/quantized weights pay off). Mirrors
    /// [`crate::model::forward`] op-for-op; result must match the CPU path.
    fn forward_step(&self, model: &Model, state: &mut RunState, token: usize, pos: usize) {
        // Non-Llama archs (Qwen2/Phi-3/Gemma2) run the generic per-op path; the
        // resident fused decode below is the Llama-only fast path.
        if !model.config.arch.uses_resident_decode() {
            crate::model::forward(model, state, self, token, pos);
            return;
        }
        let p = &model.config;
        let w = &model.weights;

        let mut guard = self.decode.lock().unwrap();
        let stale = match &*guard {
            Some(d) => {
                d.dim != p.dim
                    || d.n_layers != p.n_layers
                    || d.vocab != p.vocab_size
                    || d.seq_len != p.seq_len
            }
            None => true,
        };
        if stale {
            *guard = Some(self.build_decode(model));
        }
        let d = guard.as_mut().unwrap();
        let st = &self.stream;
        let (dim, kv_dim, head_size, hidden) = (d.dim, d.kv_dim, d.head_size, d.hidden);
        let (n_heads, kv_mul, n_freqs, mscale) = (d.n_heads, d.kv_mul, d.n_freqs, d.mscale);
        let eps = p.rms_eps;

        // One-time host→device sync of any KV rows a prior prefill filled.
        if d.kv_filled < pos {
            for l in 0..d.n_layers {
                let loff = l * d.seq_len * kv_dim;
                let (lo, hi) = (loff + d.kv_filled * kv_dim, loff + pos * kv_dim);
                let off = d.kv_filled * kv_dim;
                let span = hi - lo;
                st.memcpy_htod(
                    &state.key_cache()[lo..hi],
                    &mut d.key[l].slice_mut(off..off + span),
                )
                .expect("upload prefill K");
                st.memcpy_htod(
                    &state.value_cache()[lo..hi],
                    &mut d.value[l].slice_mut(off..off + span),
                )
                .expect("upload prefill V");
            }
            d.kv_filled = pos;
        }

        // This token's embedding → resident x.
        w.token_embedding_table.dequant_row(token, &mut d.embed);
        st.memcpy_htod(&d.embed, &mut d.x).expect("upload embedding");

        for layer in 0..d.n_layers {
            // --- Attention (batch-1 GEMVs) ---
            self.dev_rmsnorm(&mut d.xb, &d.x, &d.rms_att[layer], 1, dim, eps);
            self.gemm_dev(&mut d.q, &d.xb, &w.wq[layer], 1);
            self.gemm_dev(&mut d.k_row, &d.xb, &w.wk[layer], 1);
            self.gemm_dev(&mut d.v_row, &d.xb, &w.wv[layer], 1);
            self.dev_rope(
                &mut d.q, &mut d.k_row, &d.inv, 1, pos, head_size, kv_dim, dim, n_freqs, mscale,
            );
            // Append this step's K/V to the resident cache at row `pos`.
            let off = pos * kv_dim;
            st.memcpy_dtod(&d.k_row, &mut d.key[layer].slice_mut(off..off + kv_dim))
                .expect("write K");
            st.memcpy_dtod(&d.v_row, &mut d.value[layer].slice_mut(off..off + kv_dim))
                .expect("write V");
            // Attention over cached rows 0..=pos.
            self.dev_attention(
                &mut d.xb, &d.q, &d.key[layer], &d.value[layer], 1, pos, n_heads, kv_mul,
                head_size, pos + 1, kv_dim,
            );
            self.gemm_dev(&mut d.xb2, &d.xb, &w.wo[layer], 1);
            self.dev_add(&mut d.x, &d.xb2, dim);

            // --- Feed-forward (SwiGLU) ---
            self.dev_rmsnorm(&mut d.xb, &d.x, &d.rms_ffn[layer], 1, dim, eps);
            self.gemm_dev(&mut d.hb, &d.xb, &w.w1[layer], 1);
            self.gemm_dev(&mut d.hb2, &d.xb, &w.w3[layer], 1);
            self.dev_swiglu(&mut d.hb, &d.hb2, hidden);
            self.gemm_dev(&mut d.xb, &d.hb, &w.w2[layer], 1);
            self.dev_add(&mut d.x, &d.xb, dim);
        }
        d.kv_filled = pos + 1;

        // Final RMSNorm + classifier → logits, downloaded to host `state`.
        self.dev_rmsnorm(&mut d.xb, &d.x, &d.rms_final, 1, dim, eps);
        self.gemm_dev(&mut d.logits, &d.xb, &w.wcls, 1);
        st.synchronize().expect("stream synchronize");
        let logits_host: Vec<f32> = st.clone_dtoh(&d.logits).expect("download logits");
        state.logits_mut().copy_from_slice(&logits_host);
    }

    /// Resident fused prefill: the whole prompt runs on-device — f16 GEMMs plus
    /// the [`KERNEL_SRC`] kernels — with `x`/`xb`/`q`/`hb` and the per-layer KV
    /// kept resident, so a layer executes with no host round-trips. Only the
    /// token embeddings go up and the final logits come down (plus the prompt's
    /// K/V, copied back to host `state` for subsequent CPU decode).
    ///
    /// Handles the from-scratch case (`pos_base == 0`, how `generate` prefills a
    /// prompt); for a non-zero base the prior KV lives on the host, so it
    /// delegates to the per-op default.
    fn forward_prefill(
        &self,
        model: &Model,
        state: &mut RunState,
        tokens: &[usize],
        pos_base: usize,
    ) {
        if pos_base != 0 || !model.config.arch.uses_resident_decode() {
            crate::model::forward_prefill(model, state, self, tokens, pos_base);
            return;
        }

        let p = &model.config;
        let w = &model.weights;
        let (dim, kv_dim, head_size, hidden) = (p.dim, p.kv_dim(), p.head_size(), p.hidden_dim);
        let (n_heads, kv_mul, eps) = (p.n_heads, p.n_heads / p.n_kv_heads, p.rms_eps);
        let n = tokens.len();
        let st = &self.stream;

        // Seed x with the (dequantized) token embeddings; one upload.
        let mut x_host = vec![0.0f32; n * dim];
        for (r, &tok) in tokens.iter().enumerate() {
            w.token_embedding_table
                .dequant_row(tok, &mut x_host[r * dim..r * dim + dim]);
        }
        let mut x = st.clone_htod(&x_host).expect("upload embeddings");

        // Resident scratch. Each layer keeps its own K/V buffer (rows 0..n, since
        // pos_base == 0) so the whole prefill runs with NO in-loop synchronize —
        // they are downloaded to the host once after the loop for the decode
        // handoff. `att`'s row stride is therefore n, not seq_len.
        let mut xb = st.alloc_zeros::<f32>(n * dim).expect("alloc xb");
        let mut xb2 = st.alloc_zeros::<f32>(n * dim).expect("alloc xb2");
        let mut q = st.alloc_zeros::<f32>(n * dim).expect("alloc q");
        let mut hb = st.alloc_zeros::<f32>(n * hidden).expect("alloc hb");
        let mut hb2 = st.alloc_zeros::<f32>(n * hidden).expect("alloc hb2");
        let mut key_dev: Vec<CudaSlice<f32>> = (0..p.n_layers)
            .map(|_| st.alloc_zeros::<f32>(n * kv_dim).expect("alloc key"))
            .collect();
        let mut value_dev: Vec<CudaSlice<f32>> = (0..p.n_layers)
            .map(|_| st.alloc_zeros::<f32>(n * kv_dim).expect("alloc value"))
            .collect();
        let inv = st
            .clone_htod(&model.rope.inv_freq)
            .expect("upload inv_freq");
        let (n_freqs, mscale) = (model.rope.inv_freq.len(), model.rope.mscale);

        for layer in 0..p.n_layers {
            // --- Attention ---
            let w_att = st
                .clone_htod(&w.rms_att_weight[layer * dim..layer * dim + dim])
                .expect("upload rms_att");
            self.dev_rmsnorm(&mut xb, &x, &w_att, n, dim, eps);
            self.gemm_dev(&mut q, &xb, &w.wq[layer], n);
            self.gemm_dev(&mut key_dev[layer], &xb, &w.wk[layer], n);
            self.gemm_dev(&mut value_dev[layer], &xb, &w.wv[layer], n);
            self.dev_rope(
                &mut q, &mut key_dev[layer], &inv, n, 0, head_size, kv_dim, dim, n_freqs, mscale,
            );
            self.dev_attention(
                &mut xb, &q, &key_dev[layer], &value_dev[layer], n, 0, n_heads, kv_mul, head_size,
                n, kv_dim,
            );
            self.gemm_dev(&mut xb2, &xb, &w.wo[layer], n);
            self.dev_add(&mut x, &xb2, n * dim);

            // --- Feed-forward (SwiGLU) ---
            let w_ffn = st
                .clone_htod(&w.rms_ffn_weight[layer * dim..layer * dim + dim])
                .expect("upload rms_ffn");
            self.dev_rmsnorm(&mut xb, &x, &w_ffn, n, dim, eps);
            self.gemm_dev(&mut hb, &xb, &w.w1[layer], n);
            self.gemm_dev(&mut hb2, &xb, &w.w3[layer], n);
            self.dev_swiglu(&mut hb, &hb2, n * hidden);
            self.gemm_dev(&mut xb, &hb, &w.w2[layer], n);
            self.dev_add(&mut x, &xb, n * dim);
        }

        // Hand the prompt's K/V back to the host cache for decode — one batch of
        // downloads after the whole prefill (no per-layer sync). `clone_dtoh` of
        // pageable host memory completes synchronously.
        for layer in 0..p.n_layers {
            let k_host: Vec<f32> = st.clone_dtoh(&key_dev[layer]).expect("download K");
            let v_host: Vec<f32> = st.clone_dtoh(&value_dev[layer]).expect("download V");
            state.store_prefill_kv(layer * p.seq_len * kv_dim, &k_host, &v_host);
        }

        // Final RMSNorm of the last position + classifier, via the host path.
        let x_host: Vec<f32> = st.clone_dtoh(&x).expect("download residual");
        let last = (n - 1) * dim;
        let mut xb_last = vec![0.0f32; dim];
        self.cpu
            .rmsnorm(&mut xb_last, &x_host[last..last + dim], &w.rms_final_weight, eps);
        self.gemm(state.logits_mut(), &xb_last, &w.wcls, 1);
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

    /// Assert two slices match within an f16-GEMM tolerance: f16 tensor cores
    /// keep ~10 mantissa bits (≈2⁻¹¹ relative) and accumulate in f32, so error
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

        // Dispatch through the trait method: CUDA runs its resident fused
        // prefill, CPU uses the default (per-op free fn).
        let logits = |backend: &dyn Backend| {
            let mut s = crate::RunState::new(&model.config);
            backend.forward_prefill(&model, &mut s, &tokens, 0);
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

    /// Guards the device→host KV handoff: after the fused prefill, a decode step
    /// (which runs on the CPU and reads the *host* KV cache) must match the
    /// all-CPU path. If the fused prefill failed to write `state`'s KV correctly,
    /// the post-prompt decode logits diverge here even though prefill logits
    /// (above) wouldn't.
    #[test]
    #[ignore = "requires a CUDA device; run with --features cuda -- --ignored --nocapture"]
    fn prefill_then_decode_coherent() {
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
        let tokens: Vec<usize> = (0..12).map(|i| (i * 7) % c.vocab_size).collect();
        let n = tokens.len();

        // Prefill the prompt, then decode one token at position n. For CUDA the
        // prefill is fused (writes host KV via the handoff); decode is CPU.
        let run = |backend: &dyn Backend| {
            let mut s = crate::RunState::new(&model.config);
            backend.forward_prefill(&model, &mut s, &tokens, 0);
            backend.forward_step(&model, &mut s, tokens[0], n);
            s.logits().to_vec()
        };
        let cuda = run(&b);
        let cpu = run(&CpuBackend::new());

        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, x| a.1.partial_cmp(x.1).unwrap())
                .unwrap()
                .0
        };
        let num: f32 = cuda.iter().zip(&cpu).map(|(a, b)| (a - b) * (a - b)).sum();
        let den: f32 = cpu.iter().map(|v| v * v).sum();
        let rel = (num / den).sqrt();
        eprintln!(
            "prefill+decode coherence: argmax cuda={} cpu={}, relative L2 error {rel:.4}",
            argmax(&cuda),
            argmax(&cpu)
        );
        assert_eq!(argmax(&cuda), argmax(&cpu), "decode top token diverged");
        assert!(rel < 0.1, "decode logits diverged: relative L2 error {rel}");
    }

    /// Multi-step decode coherence: decode a short sequence one token at a time
    /// from position 0 (no prefill) on CUDA vs CPU, comparing the logits at every
    /// step. Exercises the resident `forward_step` KV append across steps (the
    /// device cache grows by one row per step).
    #[test]
    #[ignore = "requires a CUDA device; run with --features cuda -- --ignored --nocapture"]
    fn decode_multistep_coherent() {
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
        let tokens: Vec<usize> = (0..8).map(|i| (i * 37 + 5) % c.vocab_size).collect();

        let run = |backend: &dyn Backend| {
            let mut s = crate::RunState::new(&model.config);
            tokens
                .iter()
                .enumerate()
                .map(|(pos, &tok)| {
                    backend.forward_step(&model, &mut s, tok, pos);
                    s.logits().to_vec()
                })
                .collect::<Vec<_>>()
        };
        let cuda = run(&b);
        let cpu = run(&CpuBackend::new());

        for (pos, (cu, cp)) in cuda.iter().zip(&cpu).enumerate() {
            let argmax = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .max_by(|a, x| a.1.partial_cmp(x.1).unwrap())
                    .unwrap()
                    .0
            };
            let num: f32 = cu.iter().zip(cp).map(|(a, b)| (a - b) * (a - b)).sum();
            let den: f32 = cp.iter().map(|v| v * v).sum();
            let rel = (num / den).sqrt();
            eprintln!("decode step {pos}: argmax cuda={} cpu={}, rel L2 {rel:.4}", argmax(cu), argmax(cp));
            assert_eq!(argmax(cu), argmax(cp), "step {pos}: top token diverged");
            assert!(rel < 0.1, "step {pos}: logits diverged: rel L2 {rel}");
        }
    }

    /// Decode throughput (`tg128`): CUDA resident `forward_step` vs CPU on the real
    /// TinyLlama-1.1B Q4_K_M, 128 steps. Compare to
    /// `llama-bench -m <gguf> -ngl 99` (tg128). Decode is batch-1 GEMV
    /// (bandwidth-bound). Set `RUSTY_LLAMA_GGUF` to override the path.
    #[test]
    #[ignore = "needs the real GGUF + a CUDA device; run with --release --features cuda -- --ignored --nocapture"]
    fn bench_decode_real_tinyllama() {
        let Some(b) = cuda() else { return };
        let path = std::env::var("RUSTY_LLAMA_GGUF")
            .unwrap_or_else(|_| "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".to_string());
        let cp = match crate::Checkpoint::open(&path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping real-model decode bench: can't open {path}: {e}");
                return;
            }
        };
        if !crate::Gguf::is_gguf(cp.bytes()) {
            eprintln!("skipping real-model decode bench: {path} is not a GGUF");
            return;
        }
        let gguf = crate::Gguf::parse(cp.bytes()).expect("parse gguf");
        let model = crate::Model::from_gguf(&gguf).expect("build model");
        let c = &model.config;
        let steps = 128usize;

        let decode = |backend: &dyn Backend| {
            let mut s = crate::RunState::new(&model.config);
            for pos in 0..steps {
                backend.forward_step(&model, &mut s, (pos * 13 + 1) % c.vocab_size, pos);
            }
        };

        // llama-bench protocol: one untimed warmup (builds the resident decode
        // state + weight cache on device), then `-r` timed reps as mean ± stddev.
        let reps = crate::bench_util::bench_reps();
        let (cuda_tps, cuda_sd) =
            crate::bench_util::bench_stat(steps, reps, std::time::Duration::from_secs(6), || decode(&b));

        let cpu_backend = CpuBackend::new();
        let (cpu_tps, _) =
            crate::bench_util::bench_stat(steps, 1, std::time::Duration::ZERO, || decode(&cpu_backend));

        eprintln!(
            "real decode tg{steps} (r={reps}): cuda {cuda_tps:.0} ± {cuda_sd:.0} tok/s  \
             cpu {cpu_tps:.0} tok/s  -> cuda {:.1}x cpu  \
             (compare: llama-bench -m <gguf> -ngl 99 -n {steps} -r {reps})",
            cuda_tps / cpu_tps,
        );
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
            backend.forward_prefill(&model, &mut s, &tokens, 0);
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

    /// Real-model prefill: the fused CUDA prefill vs CPU on the actual
    /// TinyLlama-1.1B Q4_K_M GGUF (22 layers, dim 2048) — the north-star shape,
    /// not the 4-layer synthetic proxy. Set `RUSTY_LLAMA_GGUF` to override the
    /// path (default: the repo-root TinyLlama). Compare the CUDA tok/s against
    /// `llamacpp_bench/lc/llama-bench.exe -m <gguf> -ngl 99` (pp512). Weights are
    /// cached as **f16** on-device (~2.2 GB for this model).
    #[test]
    #[ignore = "needs the real GGUF + a CUDA device; run with --release --features cuda -- --ignored --nocapture"]
    fn bench_prefill_real_tinyllama() {
        let Some(b) = cuda() else { return };

        let path = std::env::var("RUSTY_LLAMA_GGUF")
            .unwrap_or_else(|_| "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".to_string());
        let cp = match crate::Checkpoint::open(&path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping real-model bench: can't open {path}: {e}");
                return;
            }
        };
        if !crate::Gguf::is_gguf(cp.bytes()) {
            eprintln!("skipping real-model bench: {path} is not a GGUF");
            return;
        }
        let gguf = crate::Gguf::parse(cp.bytes()).expect("parse gguf");
        let model = crate::Model::from_gguf(&gguf).expect("build model");
        let c = &model.config;
        let n = 512.min(c.seq_len);
        let tokens: Vec<usize> = (0..n).map(|i| (i * 7 + 1) % c.vocab_size).collect();
        eprintln!(
            "model {path}: dim {} hidden {} layers {} heads {}/{} vocab {} seq {} | prefill {n} tok",
            c.dim, c.hidden_dim, c.n_layers, c.n_heads, c.n_kv_heads, c.vocab_size, c.seq_len,
        );

        let prefill = |backend: &dyn Backend| {
            let mut s = crate::RunState::new(&model.config);
            backend.forward_prefill(&model, &mut s, &tokens, 0);
        };

        // llama-bench protocol: one untimed warmup (dequant + upload weights,
        // make them resident), then `-r` timed reps as mean ± stddev.
        let reps = crate::bench_util::bench_reps();
        let (cuda_tps, cuda_sd) =
            crate::bench_util::bench_stat(n, reps, std::time::Duration::from_secs(6), || prefill(&b));

        let cpu_backend = CpuBackend::new();
        let (cpu_tps, _) =
            crate::bench_util::bench_stat(n, 1, std::time::Duration::ZERO, || prefill(&cpu_backend));

        eprintln!(
            "real prefill pp{n} (r={reps}): cuda {cuda_tps:.0} ± {cuda_sd:.0} tok/s  \
             cpu {cpu_tps:.0} tok/s  -> cuda {:.1}x cpu  \
             (compare: llama-bench -m <gguf> -ngl 99 -p {n} -r {reps})",
            cuda_tps / cpu_tps,
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

    /// Round-trip f32 → f16 → f32 through the inline-PTX conversion kernels.
    /// Confirms the `asm("cvt...")` actually compiles + runs (not skipped).
    #[test]
    #[ignore = "requires a CUDA device; run with --features cuda -- --ignored --nocapture"]
    fn dev_cvt_roundtrip() {
        let Some(b) = cuda() else { return };
        let n = 100usize;
        let x = noise(n, 18);
        let xd = b.stream.clone_htod(&x).unwrap();
        let mut x16 = b.stream.alloc_zeros::<f16>(n).unwrap();
        b.dev_cvt_to_f16(&mut x16, &xd, n);
        let mut backd = b.stream.alloc_zeros::<f32>(n).unwrap();
        b.dev_cvt_to_f32(&mut backd, &x16, n);
        b.stream.synchronize().unwrap();
        let got = b.stream.clone_dtoh(&backd).unwrap();
        // f16 has ~3 decimal digits; values are in [-0.5, 0.5).
        close_approx(&got, &x, 1e-3, 1e-2);
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
        b.dev_attention(
            &mut outd, &qd, &kcd, &vcd, n, 0, n_heads, n_heads / n_kv_heads, head_size, seq_len,
            kv_dim,
        );
        b.stream.synchronize().unwrap();
        let got = b.stream.clone_dtoh(&outd).unwrap();

        let mut want = vec![0.0; n * dim];
        let mut att = vec![0.0; n_heads * seq_len];
        CpuBackend.attention_batch(
            &mut want, &q, &kc, &vc, &mut att, 0, n, n_heads, n_kv_heads, head_size, seq_len, kv_dim, 0.0,
        );
        close_approx(&got, &want, 1e-4, 1e-4);
    }

}
