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
    CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use half::f16;

use crate::backend::{Backend, CpuBackend};
use crate::model::{Model, RunState};
use crate::quant::{dequantize, GgmlType};
use crate::tensor::QMatrix;

/// `CudaGraph` is not `Send` (it holds raw CUDA graph handles), but the resident
/// decode graph is only ever touched while holding `CudaBackend::decode`'s mutex
/// (single-threaded), so wrapping it keeps `DecodeCuda` `Send` for that
/// `Mutex<Option<…>>` in the `Send + Sync` backend.
struct SendGraph(CudaGraph);
// SAFETY: every access is serialized by the `decode` mutex — the graph is never
// used from two threads at once (cudarc marks its other handles `Send` likewise).
unsafe impl Send for SendGraph {}

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
                                       const int* pos_base, int head_size, int kv_dim,
                                       int q_dim, int n_freqs, float mscale) {
    int row = blockIdx.x;
    int pair = blockIdx.y * blockDim.x + threadIdx.x;
    int n_pairs = q_dim / 2;
    if (pair >= n_pairs) return;
    int i = pair * 2;
    int local_pair = (i % head_size) / 2;
    if (local_pair >= n_freqs) return;
    float val = (float)(pos_base[0] + row) * inv_freq[local_pair];
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
// Append one K/V row into the resident cache at the device-resident position:
// cache[pos[0]*kv_dim + i] = row[i]. The captured decode graph uses this instead
// of a dtod memcpy, whose destination offset would be baked at capture time.
extern "C" __global__ void kv_append(float* cache, const float* row,
                                     const int* pos, int kv_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_dim) cache[(long long)pos[0] * kv_dim + i] = row[i];
}
// Causal GQA flash attention: one thread BLOCK per (row, head), streaming the
// keys in tiles of `blockDim.x` with an online (running-max) softmax. No
// `seq_len`-sized score buffer — only a per-tile prob row — so context is not
// bounded by shared memory. `blockDim.x` must be a power of two. Dynamic shared
// layout: [ sq: head_size | sp: blockDim.x | acc: head_size | red: blockDim.x ]
// floats. The resident decode path is Llama-only, so no logit softcap here
// (Gemma2's softcap runs the per-op path, which delegates attention to the CPU).
extern "C" __global__ void attention_kernel(float* out, const float* q,
        const float* kcache, const float* vcache,
        int rows, const int* pos_base, int n_heads, int kv_mul, int head_size,
        int kv_dim) {
    int idx = blockIdx.x;
    if (idx >= rows * n_heads) return;
    int r = idx / n_heads, h = idx % n_heads;
    int pos = pos_base[0] + r;
    int dim = n_heads * head_size;
    int kv_off = (h / kv_mul) * head_size;
    const float* qh = q + (long long)r * dim + (long long)h * head_size;
    int tid = threadIdx.x, nt = blockDim.x;

    extern __shared__ float smem[];
    float* sq  = smem;                // head_size: this (row,head)'s query
    float* sp  = sq + head_size;      // nt: current tile's softmax probabilities
    float* acc = sp + nt;             // head_size: running value accumulator
    float* red = acc + head_size;     // nt: reduction scratch

    for (int i = tid; i < head_size; i += nt) { sq[i] = qh[i]; acc[i] = 0.0f; }
    __syncthreads();

    float scale = 1.0f / sqrtf((float)head_size);
    float m = -1e30f; // running max score (all threads track it identically)
    float l = 0.0f;   // running exp-sum

    for (int tile = 0; tile <= pos; tile += nt) {
        int t = tile + tid;
        // 1. score for this thread's key (-inf past `pos`).
        float s = -1e30f;
        if (t <= pos) {
            const float* kt = kcache + (long long)t * kv_dim + kv_off;
            float dot = 0.0f;
            for (int i = 0; i < head_size; i++) dot += sq[i] * kt[i];
            s = dot * scale;
        }
        // 2. tile max.
        red[tid] = s;
        __syncthreads();
        for (int o = nt / 2; o > 0; o >>= 1) {
            if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]);
            __syncthreads();
        }
        float m_new = fmaxf(m, red[0]);
        __syncthreads();
        float alpha = expf(m - m_new); // rescales running state when the max grows
        // 3. probability for this thread's key.
        float p = (t <= pos) ? expf(s - m_new) : 0.0f;
        sp[tid] = p;
        // 4. tile sum.
        red[tid] = p;
        __syncthreads();
        for (int o = nt / 2; o > 0; o >>= 1) {
            if (tid < o) red[tid] += red[tid + o];
            __syncthreads();
        }
        l = l * alpha + red[0];
        __syncthreads();
        // 5. accumulate this tile's value-weighted contribution; each thread owns
        //    a stride of head dims and loops the tile's keys.
        int tile_count = (pos + 1 - tile < nt) ? (pos + 1 - tile) : nt;
        for (int i = tid; i < head_size; i += nt) {
            float a = acc[i] * alpha;
            for (int j = 0; j < tile_count; j++) {
                a += sp[j] * vcache[(long long)(tile + j) * kv_dim + kv_off + i];
            }
            acc[i] = a;
        }
        m = m_new;
        __syncthreads(); // sp/acc consumed before the next tile overwrites sp
    }

    float invl = 1.0f / l;
    float* outh = out + (long long)r * dim + (long long)h * head_size;
    for (int i = tid; i < head_size; i += nt) outh[i] = acc[i] * invl;
}
// Activation -> Q8_K (block-256). Mirrors quantize_activation_q8k (src/quant.rs:436)
// EXACTLY (not llama.cpp's block-32 q8_1): per 256-elem super-block, signed-max
// iscale = -128/max where `max` is the signed value at the FIRST position of
// maximal |v| (first-wins tie-break), d = 1/iscale, q = round(iscale*v) clamped to
// [-128,127], and absums = sum of quants per group of 16. All-zero block -> zeros.
// One block per super-block, 256 threads (one per element).
extern "C" __global__ void quantize_q8k(signed char* aq, float* ad, int* absums,
                                        const float* x, int n) {
    int sb = blockIdx.x;
    int base = sb * 256;
    int tid = threadIdx.x;              // 0..255, one element each
    const float* xb = x + base;
    __shared__ float amaxr[256];
    __shared__ int   idxr[256];
    __shared__ int   qsh[256];

    float v = xb[tid];
    float av = fabsf(v);
    amaxr[tid] = av;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) amaxr[tid] = fmaxf(amaxr[tid], amaxr[tid + s]);
        __syncthreads();
    }
    float amax = amaxr[0];
    __syncthreads();

    if (amax == 0.0f) {
        aq[base + tid] = 0;
        if (tid < 16) absums[sb * 16 + tid] = 0;
        if (tid == 0) ad[sb] = 0.0f;
        return;
    }
    // First-max-abs-wins: smallest index whose |v| equals amax.
    idxr[tid] = (av == amax) ? tid : 0x7fffffff;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) idxr[tid] = min(idxr[tid], idxr[tid + s]);
        __syncthreads();
    }
    float maxv = xb[idxr[0]];
    float iscale = -128.0f / maxv;
    if (tid == 0) ad[sb] = 1.0f / iscale;
    int q = (int)roundf(iscale * v);
    q = max(-128, min(127, q));
    qsh[tid] = q;
    aq[base + tid] = (signed char)q;
    __syncthreads();
    if (tid < 16) {
        int s = 0;
        for (int k = 0; k < 16; k++) s += qsh[tid * 16 + k];
        absums[sb * 16 + tid] = s;
    }
}
// --- packed-weight DP4A decode GEMV (Phase 2) -------------------------------
// __dp4a via inline PTX (sm_61+): d = dot4(a.s8, b.s8) + c. Matches the file's
// header-free inline-PTX convention so nvrtc needs no sm_61_intrinsics.h.
__device__ __forceinline__ int dp4a_(int a, int b, int c) {
    int r;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
// f16 (u16 bits) -> f32, inline PTX (device-side counterpart of cvt_f16_f32).
__device__ __forceinline__ float h2f(unsigned short h) {
    float v;
    asm("cvt.f32.f16 %0, %1;" : "=f"(v) : "h"(h));
    return v;
}
// 6-bit scale/min for sub-block j from the 12 packed bytes. Mirrors
// get_scale_min_k4 (src/quant.rs:223).
__device__ __forceinline__ void gsmk4(int j, const unsigned char* q, int* sc, int* m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}
// Per-row Q4_K dot: dequant(w_row) . act at the dp4a level, warp-cooperative; the
// shared core of gemv_q4_k and the fused gemv_qkv. `wr` = the row's nsb 144-byte
// super-blocks. Lane l reads word l (qs[l*4..]) so consecutive lanes touch
// consecutive 4-byte words = one coalesced 128-B transaction/warp step. Result is
// warp-reduced (valid on lane 0). Mirrors vec_dot_q4_k_scalar (src/quant.rs:490).
__device__ __forceinline__ float dot_q4_k_row(const unsigned char* wr,
        const signed char* aq, const float* ad, const int* absums, int nsb, int lane) {
    int chunk = lane >> 3;          // 0..3: which 32-byte qs chunk
    int rem = (lane & 7) * 4;       // this lane's word offset within the chunk
    int jlo = 2 * chunk, jhi = 2 * chunk + 1;
    float racc = 0.0f;
    for (int sb = 0; sb < nsb; sb++) {
        const unsigned char* blk = wr + (long long)sb * 144;
        float d = h2f(*(const unsigned short*)blk);
        float dmin = h2f(*(const unsigned short*)(blk + 2));
        const unsigned char* scales = blk + 4;
        const unsigned char* qs = blk + 16;
        float bigd = ad[sb];
        int word = *(const int*)(qs + lane * 4);
        int wlo = word & 0x0F0F0F0F;
        int whi = (word >> 4) & 0x0F0F0F0F;
        const signed char* aqb = aq + (long long)sb * 256;
        int axlo = *(const int*)(aqb + jlo * 32 + rem);
        int axhi = *(const int*)(aqb + jhi * 32 + rem);
        int plow = dp4a_(wlo, axlo, 0);
        int phigh = dp4a_(whi, axhi, 0);
        int sc_lo, m_lo, sc_hi, m_hi;
        gsmk4(jlo, scales, &sc_lo, &m_lo);
        gsmk4(jhi, scales, &sc_hi, &m_hi);
        racc += d * bigd * (float)(sc_lo * plow + sc_hi * phigh);
        if (lane < 16) {
            int s_unused, mj;
            gsmk4(lane >> 1, scales, &s_unused, &mj);
            racc -= dmin * bigd * (float)(absums[sb * 16 + lane] * mj);
        }
    }
    for (int off = 16; off > 0; off >>= 1)
        racc += __shfl_down_sync(0xffffffff, racc, off);
    return racc;  // valid on lane 0
}
// Warp-cooperative packed Q4_K GEMV: out[row] = dequant(w[row]) . act. One warp
// per row; NWARPS = blockDim.x/32 rows per block (launch-set).
extern "C" __global__ void gemv_q4_k(float* out, const unsigned char* w,
        const signed char* aq, const float* ad, const int* absums, int oc, int nsb) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= oc) return;
    float r = dot_q4_k_row(w + (long long)row * nsb * 144, aq, ad, absums, nsb, lane);
    if (lane == 0) out[row] = r;
}
// Assemble a little-endian u32 from 4 unaligned bytes (Q6_K's 210-byte blocks are
// only 2-aligned, so `*(int*)` would fault).
__device__ __forceinline__ int ld4(const unsigned char* p) {
    return (int)p[0] | ((int)p[1] << 8) | ((int)p[2] << 16) | ((int)p[3] << 24);
}
// Per-row Q6_K dot: the shared core of gemv_q6_k and gemv_qkv. Q6_K is symmetric:
// Σ(q-32)·x = Σq·x − 32·Σx, with Σx = absums[sub] (exact). `wr` = the row's nsb
// 210-byte super-blocks; signed i8 scales, no min term. Mirrors
// vec_dot_q6_k_scalar (src/quant.rs:544).
__device__ __forceinline__ float dot_q6_k_row(const unsigned char* wr,
        const signed char* aq, const float* ad, const int* absums, int nsb, int lane) {
    float racc = 0.0f;
    for (int sb = 0; sb < nsb; sb++) {
        const unsigned char* blk = wr + (long long)sb * 210;
        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;
        const signed char* scales = (const signed char*)(blk + 192);
        float d = h2f(*(const unsigned short*)(blk + 208));
        float bigd = ad[sb];
        const signed char* aqb = aq + (long long)sb * 256;
        for (int pass = 0; pass < 2; pass++) {
            int ri = lane + pass * 32;     // run of 4 elements
            int i = ri * 4;
            int half = i >> 7;             // 0/1 (which 128 block)
            int local = i & 127;
            int group = local >> 5;        // 0..3 reconstruction group
            int lb = local & 31;
            int ql_off = half * 64 + ((group == 1 || group == 3) ? lb + 32 : lb);
            int qh_off = half * 32 + lb;
            int qlw = ld4(ql + ql_off);
            int qhw = ld4(qh + qh_off);
            int nib = (group < 2) ? (qlw & 0x0F0F0F0F) : ((qlw >> 4) & 0x0F0F0F0F);
            int hib = ((qhw >> (group * 2)) & 0x03030303) << 4;
            int qpacked = nib | hib;       // 4 weights, each 0..63
            int qxw = *(const int*)(aqb + i);
            int qdot = dp4a_(qpacked, qxw, 0);
            int sub = i >> 4;              // 16-elem sub-block
            racc += d * bigd * (float)((int)scales[sub] * qdot);
        }
        if (lane < 16) {
            racc -= d * bigd * (float)((int)scales[lane] * 32 * absums[sb * 16 + lane]);
        }
    }
    for (int off = 16; off > 0; off >>= 1)
        racc += __shfl_down_sync(0xffffffff, racc, off);
    return racc;  // valid on lane 0
}
// Warp-cooperative packed Q6_K GEMV: out[row] = dequant(w[row]) . act.
extern "C" __global__ void gemv_q6_k(float* out, const unsigned char* w,
        const signed char* aq, const float* ad, const int* absums, int oc, int nsb) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= oc) return;
    float r = dot_q6_k_row(w + (long long)row * nsb * 210, aq, ad, absums, nsb, lane);
    if (lane == 0) out[row] = r;
}
// Fused q/k/v decode GEMV: one launch computes all three projections (which share
// the Q8_K activation) instead of three, absorbing the small under-saturating k/v
// kernels (kv_oc rows each) into one saturating grid. Q4_K_M layout: q,k are Q4_K,
// v is Q6_K. Rows [0,q_oc) -> q; [q_oc,q_oc+kv_oc) -> k; rest -> v. Same nsb (all
// project the dim-length activation).
extern "C" __global__ void gemv_qkv(float* out_q, float* out_k, float* out_v,
        const unsigned char* wq, const unsigned char* wk, const unsigned char* wv,
        const signed char* aq, const float* ad, const int* absums,
        int q_oc, int kv_oc, int nsb) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= q_oc + 2 * kv_oc) return;
    if (row < q_oc) {
        float r = dot_q4_k_row(wq + (long long)row * nsb * 144, aq, ad, absums, nsb, lane);
        if (lane == 0) out_q[row] = r;
    } else if (row < q_oc + kv_oc) {
        int lr = row - q_oc;
        float r = dot_q4_k_row(wk + (long long)lr * nsb * 144, aq, ad, absums, nsb, lane);
        if (lane == 0) out_k[lr] = r;
    } else {
        int lr = row - q_oc - kv_oc;
        float r = dot_q6_k_row(wv + (long long)lr * nsb * 210, aq, ad, absums, nsb, lane);
        if (lane == 0) out_v[lr] = r;
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
    quantize_q8k: CudaFunction,
    gemv_q4_k: CudaFunction,
    gemv_q6_k: CudaFunction,
    gemv_qkv: CudaFunction,
    kv_append: CudaFunction,
}

impl Kernels {
    fn compile(ctx: &Arc<CudaContext>) -> Result<Self, String> {
        // The packed-GEMV kernels' inline-PTX `dp4a` (sm_61+) assembles under
        // nvrtc's default virtual arch (CUDA 13 targets >= sm_75) and JITs to the
        // device (sm_120); no explicit --gpu-architecture needed.
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
            quantize_q8k: f("quantize_q8k")?,
            gemv_q4_k: f("gemv_q4_k")?,
            gemv_q6_k: f("gemv_q6_k")?,
            gemv_qkv: f("gemv_qkv")?,
            kv_append: f("kv_append")?,
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
    // Device-resident absolute position: rope/attention read `pos` from here and
    // the KV-append kernel writes row `pos`, so the captured decode graph replays
    // unchanged each step (only this one int is updated before each launch).
    pos_dev: CudaSlice<i32>,
    // resident Q8_K activation scratch for the packed decode GEMV (Phase 2),
    // sized max(dim, hidden); reused across the GEMVs that share an activation.
    aq: CudaSlice<i8>,
    ad: CudaSlice<f32>,
    absums: CudaSlice<i32>,
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
    // whether the per-step decode is all custom capturable kernels (qgemv on +
    // all matmul weights k-quant), so the step can be CUDA-graph captured.
    graphable: bool,
    // the captured per-step decode graph, built lazily on the first eligible step.
    graph: Option<SendGraph>,
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
    /// Raw packed quant bytes (Q4_K/Q6_K) for the decode GEMV, keyed by source
    /// data pointer — the bandwidth win vs the f16 cache (~0.56 vs 2 B/weight).
    packed: Mutex<HashMap<usize, Arc<CudaSlice<u8>>>>,
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
        // SAFETY: this backend runs everything on a single stream, so device
        // slices never cross streams — the multi-stream synchronization that event
        // tracking provides is pure overhead here (2 events per alloc + per-launch
        // bookkeeping). Disable it before any slice is allocated (incl. the
        // cuBLASLt workspace below).
        unsafe { ctx.disable_event_tracking() };
        // A NON-default stream: the legacy NULL stream cannot be captured into a
        // CUDA graph (`cuStreamBeginCapture` rejects it), and the resident decode
        // captures its per-step kernel sequence into a graph (see `forward_step`).
        let stream = ctx
            .new_stream()
            .map_err(|e| format!("CUDA stream create failed: {e:?}"))?;
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
            packed: Mutex::new(HashMap::new()),
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

    /// The raw packed quant bytes of `w` uploaded once and cached by source data
    /// pointer — the decode GEMV reads these directly (no f16 expansion), so
    /// `weight_packed` streams ~0.56 B/weight (Q4_K) vs the f16 cache's 2 B.
    fn weight_packed(&self, w: &QMatrix) -> Arc<CudaSlice<u8>> {
        let (key, bytes): (usize, &[u8]) = match w {
            QMatrix::Quant { data, .. } => (data.as_ptr() as usize, data),
            QMatrix::F32 { .. } => panic!("weight_packed requires a quantized matrix"),
        };
        if let Some(g) = self.packed.lock().unwrap().get(&key) {
            return g.clone();
        }
        let dev = self.stream.clone_htod(bytes).expect("upload packed weight");
        let arc = Arc::new(dev);
        self.packed.lock().unwrap().insert(key, arc.clone());
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
        let ic = w.cols();
        debug_assert_eq!(x.len(), rows * ic);
        // Narrow the activation to f16 once, then run the f16 GEMM. `x16` is
        // fully written by the cvt kernel before the GEMM reads it.
        // SAFETY: written in full before read.
        let mut x16 = unsafe { self.stream.alloc::<f16>(rows * ic) }.expect("alloc x16");
        self.dev_cvt_to_f16(&mut x16, x, rows * ic);
        self.gemm_dev_f16(c, &x16, w, rows);
    }

    /// [`Self::gemm_dev`] with the activation ALREADY narrowed to f16. Lets the
    /// fused prefill narrow a *shared* activation once and feed it to the sibling
    /// GEMMs (q/k/v from one norm; w1/w3 from one norm) instead of re-narrowing
    /// the same activation per GEMM — removing ~3 cvt passes + allocs per layer.
    fn gemm_dev_f16(&self, c: &mut CudaSlice<f32>, x16: &CudaSlice<f16>, w: &QMatrix, rows: usize) {
        let (oc, ic) = (w.rows(), w.cols());
        debug_assert_eq!(x16.len(), rows * ic);
        // `c` may be LONGER than rows*oc: the K/V projections write their n rows
        // into the seq_len-sized resident KV-cache buffers. `dev_cvt_to_f32` below
        // only writes the first rows*oc, so a larger destination is fine.
        debug_assert!(
            c.len() >= rows * oc,
            "gemm output {} < rows*oc {}",
            c.len(),
            rows * oc
        );
        let w_dev = self.weight_f16(w);
        // `c16` is fully written by the GEMM (beta = 0, C not read), so skip the
        // memset. SAFETY: written in full before read.
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
        // SAFETY: dimensions/leading-dims match the buffer shapes above.
        unsafe {
            self.blas
                .matmul(cfg, w_dev.as_ref(), x16, &mut c16, None, None)
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

    /// Activation -> Q8_K (block-256): `aq` (i8), `ad` (per-256 scale), `absums`
    /// (per-16 sums) mirror [`crate::quant::quantize_activation_q8k`]. One block
    /// per super-block, 256 threads; `n` must be a multiple of 256.
    fn dev_quantize_q8k(
        &self,
        aq: &mut CudaSlice<i8>,
        ad: &mut CudaSlice<f32>,
        absums: &mut CudaSlice<i32>,
        x: &CudaSlice<f32>,
        n: usize,
    ) {
        debug_assert!(n.is_multiple_of(256));
        let n_i = n as i32;
        let cfg = LaunchConfig {
            grid_dim: ((n / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.kernels.quantize_q8k);
        b.arg(aq).arg(ad).arg(absums).arg(x).arg(&n_i);
        unsafe { b.launch(cfg) }.expect("launch quantize_q8k");
    }

    /// Packed Q4_K GEMV: `out[oc] = dequant(w) · act`, weights read packed and
    /// dot-producted with `__dp4a`. `w` is `oc` rows of `nsb` 144-byte super-blocks;
    /// `aq`/`ad`/`absums` are a [`dev_quantize_q8k`](Self::dev_quantize_q8k) result.
    /// One warp per row, 2 warps per block.
    #[allow(clippy::too_many_arguments)]
    fn dev_gemv_q4_k(
        &self,
        out: &mut CudaSlice<f32>,
        w: &CudaSlice<u8>,
        aq: &CudaSlice<i8>,
        ad: &CudaSlice<f32>,
        absums: &CudaSlice<i32>,
        oc: usize,
        nsb: usize,
    ) {
        const NWARPS: usize = 2;
        let (oc_i, nsb_i) = (oc as i32, nsb as i32);
        let cfg = LaunchConfig {
            grid_dim: (oc.div_ceil(NWARPS) as u32, 1, 1),
            block_dim: ((NWARPS * 32) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.kernels.gemv_q4_k);
        b.arg(out).arg(w).arg(aq).arg(ad).arg(absums).arg(&oc_i).arg(&nsb_i);
        unsafe { b.launch(cfg) }.expect("launch gemv_q4_k");
    }

    /// Packed Q6_K GEMV: `out[oc] = dequant(w) · act`. Like
    /// [`dev_gemv_q4_k`](Self::dev_gemv_q4_k) but 210-byte super-blocks (signed
    /// scales, no min term). One warp per row, 2 warps per block.
    #[allow(clippy::too_many_arguments)]
    fn dev_gemv_q6_k(
        &self,
        out: &mut CudaSlice<f32>,
        w: &CudaSlice<u8>,
        aq: &CudaSlice<i8>,
        ad: &CudaSlice<f32>,
        absums: &CudaSlice<i32>,
        oc: usize,
        nsb: usize,
    ) {
        const NWARPS: usize = 2;
        let (oc_i, nsb_i) = (oc as i32, nsb as i32);
        let cfg = LaunchConfig {
            grid_dim: (oc.div_ceil(NWARPS) as u32, 1, 1),
            block_dim: ((NWARPS * 32) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.kernels.gemv_q6_k);
        b.arg(out).arg(w).arg(aq).arg(ad).arg(absums).arg(&oc_i).arg(&nsb_i);
        unsafe { b.launch(cfg) }.expect("launch gemv_q6_k");
    }

    /// Fused q/k/v decode GEMV — one launch for all three projections (Q4_K_M
    /// layout: q,k are Q4_K with 144-byte blocks, v is Q6_K with 210-byte blocks),
    /// absorbing the small under-saturating k/v grids into one. See [`gemv_qkv`].
    #[allow(clippy::too_many_arguments)]
    fn dev_gemv_qkv(
        &self,
        out_q: &mut CudaSlice<f32>,
        out_k: &mut CudaSlice<f32>,
        out_v: &mut CudaSlice<f32>,
        wq: &CudaSlice<u8>,
        wk: &CudaSlice<u8>,
        wv: &CudaSlice<u8>,
        aq: &CudaSlice<i8>,
        ad: &CudaSlice<f32>,
        absums: &CudaSlice<i32>,
        q_oc: usize,
        kv_oc: usize,
        nsb: usize,
    ) {
        const NWARPS: usize = 4;
        let total = q_oc + 2 * kv_oc;
        let (q_i, kv_i, nsb_i) = (q_oc as i32, kv_oc as i32, nsb as i32);
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(NWARPS) as u32, 1, 1),
            block_dim: ((NWARPS * 32) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.kernels.gemv_qkv);
        b.arg(out_q).arg(out_k).arg(out_v).arg(wq).arg(wk).arg(wv).arg(aq).arg(ad).arg(absums)
            .arg(&q_i).arg(&kv_i).arg(&nsb_i);
        unsafe { b.launch(cfg) }.expect("launch gemv_qkv");
    }

    /// Decode GEMV `out(oc) = w · x` (batch-1): the packed DP4A path for Q4_K/Q6_K
    /// when `quantized` (the activation scratch `aq/ad/absums` is current), else the
    /// f16 cuBLASLt GEMM. The dispatch that makes Phase 2 a clean, gated cutover.
    #[allow(clippy::too_many_arguments)]
    fn gemv_decode(
        &self,
        out: &mut CudaSlice<f32>,
        x: &CudaSlice<f32>,
        aq: &CudaSlice<i8>,
        ad: &CudaSlice<f32>,
        absums: &CudaSlice<i32>,
        w: &QMatrix,
        quantized: bool,
    ) {
        if quantized {
            if let QMatrix::Quant { ty, cols, rows, .. } = w {
                let nsb = cols / 256;
                match ty {
                    GgmlType::Q4_K => {
                        let wp = self.weight_packed(w);
                        self.dev_gemv_q4_k(out, &wp, aq, ad, absums, *rows, nsb);
                        return;
                    }
                    GgmlType::Q6_K => {
                        let wp = self.weight_packed(w);
                        self.dev_gemv_q6_k(out, &wp, aq, ad, absums, *rows, nsb);
                        return;
                    }
                    _ => {}
                }
            }
        }
        self.gemm_dev(out, x, w, 1);
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
        pos: &CudaSlice<i32>,
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
        let (hs_i, kv_i, qd_i, nf_i) = (
            head_size as i32,
            kv_dim as i32,
            q_dim as i32,
            n_freqs as i32,
        );
        let mut b = self.stream.launch_builder(&self.kernels.rope);
        b.arg(q)
            .arg(k)
            .arg(inv)
            .arg(pos)
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
        pos: &CudaSlice<i32>,
        n_heads: usize,
        kv_mul: usize,
        head_size: usize,
        kv_dim: usize,
    ) {
        let (rows_i, nh_i, kvm_i, hs_i, kv_i) = (
            rows as i32,
            n_heads as i32,
            kv_mul as i32,
            head_size as i32,
            kv_dim as i32,
        );
        // One block per (row, head); 128 (power-of-two) threads cooperate, the
        // keys streamed in tiles. Shared: [ query head_size | tile probs BLOCK |
        // accumulator head_size | reduction BLOCK ] floats — no seq_len term, so
        // context isn't bounded by shared memory.
        const BLOCK: u32 = 128;
        let cfg = LaunchConfig {
            grid_dim: ((rows * n_heads) as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: (2 * head_size as u32 + 2 * BLOCK) * 4,
        };
        let mut b = self.stream.launch_builder(&self.kernels.attention);
        b.arg(out)
            .arg(q)
            .arg(kcache)
            .arg(vcache)
            .arg(&rows_i)
            .arg(pos)
            .arg(&nh_i)
            .arg(&kvm_i)
            .arg(&hs_i)
            .arg(&kv_i);
        unsafe { b.launch(cfg) }.expect("launch attention");
    }

    /// Append one K/V row into the resident cache at the device-resident position
    /// `pos` — replaces a fixed-offset dtod memcpy so the captured decode graph
    /// replays correctly as `pos` advances.
    fn dev_kv_append(
        &self,
        cache: &mut CudaSlice<f32>,
        row: &CudaSlice<f32>,
        pos: &CudaSlice<i32>,
        kv_dim: usize,
    ) {
        let kv_i = kv_dim as i32;
        let mut b = self.stream.launch_builder(&self.kernels.kv_append);
        b.arg(cache).arg(row).arg(pos).arg(&kv_i);
        unsafe { b.launch(LaunchConfig::for_num_elems(kv_dim as u32)) }.expect("launch kv_append");
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
        // The captured decode graph is valid only when every per-step op is a
        // custom capturable kernel: every matmul weight must be k-quant (the only
        // types the packed dp4a GEMV handles; anything else falls to cuBLASLt,
        // which we don't capture). The qgemv/dim conditions are re-checked per step.
        let kq = |w: &QMatrix| matches!(w, QMatrix::Quant { ty: GgmlType::Q4_K | GgmlType::Q6_K, .. });
        let wts = &model.weights;
        let graphable = kq(&wts.wcls)
            && (0..p.n_layers).all(|l| {
                kq(&wts.wq[l]) && kq(&wts.wk[l]) && kq(&wts.wv[l]) && kq(&wts.wo[l])
                    && kq(&wts.w1[l]) && kq(&wts.w2[l]) && kq(&wts.w3[l])
            });
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
            // Q8_K activation scratch, sized for the larger of the dim/hidden
            // activations (q8k blocks are 256, groups are 16).
            aq: unsafe { st.alloc::<i8>(dim.max(hidden)) }.expect("alloc aq"),
            ad: st
                .alloc_zeros::<f32>(dim.max(hidden).div_ceil(256))
                .expect("alloc ad"),
            absums: st
                .alloc_zeros::<i32>(dim.max(hidden).div_ceil(16))
                .expect("alloc absums"),
            inv: up(&model.rope.inv_freq),
            pos_dev: st.alloc_zeros::<i32>(1).expect("alloc pos_dev"),
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
            graphable,
            graph: None,
        }
    }

    /// The per-step decode compute — the layer loop, final RMSNorm, and
    /// classifier — reading the absolute position from `d.pos_dev`. Factored out
    /// of [`Self::forward_step`] so the identical sequence can run directly OR be
    /// recorded once into a CUDA graph and replayed: it touches only resident
    /// buffers, so the graph replays unchanged as `pos` advances.
    #[allow(clippy::too_many_arguments)]
    fn decode_step_kernels(
        &self,
        d: &mut DecodeCuda,
        w: &crate::model::Weights,
        eps: f32,
        q_dim_ok: bool,
        q_hid_ok: bool,
    ) {
        let (dim, kv_dim, head_size, hidden) = (d.dim, d.kv_dim, d.head_size, d.hidden);
        let (n_heads, kv_mul, n_freqs, mscale) = (d.n_heads, d.kv_mul, d.n_freqs, d.mscale);
        for layer in 0..d.n_layers {
            // --- Attention (batch-1 GEMVs) ---
            self.dev_rmsnorm(&mut d.xb, &d.x, &d.rms_att[layer], 1, dim, eps);
            if q_dim_ok {
                self.dev_quantize_q8k(&mut d.aq, &mut d.ad, &mut d.absums, &d.xb, dim);
            }
            // q/k/v share the activation; fuse them into one launch when the layout
            // is the gemv_qkv-supported Q4_K/Q4_K/Q6_K (the Q4_K_M shape), else run
            // the three separate GEMVs. Absorbs the small k/v grids into q's.
            if q_dim_ok
                && qkv_fuse_enabled()
                && matches!(&w.wq[layer], QMatrix::Quant { ty: GgmlType::Q4_K, .. })
                && matches!(&w.wk[layer], QMatrix::Quant { ty: GgmlType::Q4_K, .. })
                && matches!(&w.wv[layer], QMatrix::Quant { ty: GgmlType::Q6_K, .. })
            {
                let (wqp, wkp, wvp) = (
                    self.weight_packed(&w.wq[layer]),
                    self.weight_packed(&w.wk[layer]),
                    self.weight_packed(&w.wv[layer]),
                );
                self.dev_gemv_qkv(
                    &mut d.q, &mut d.k_row, &mut d.v_row, &wqp, &wkp, &wvp, &d.aq, &d.ad,
                    &d.absums, n_heads * head_size, kv_dim, dim / 256,
                );
            } else {
                self.gemv_decode(&mut d.q, &d.xb, &d.aq, &d.ad, &d.absums, &w.wq[layer], q_dim_ok);
                self.gemv_decode(&mut d.k_row, &d.xb, &d.aq, &d.ad, &d.absums, &w.wk[layer], q_dim_ok);
                self.gemv_decode(&mut d.v_row, &d.xb, &d.aq, &d.ad, &d.absums, &w.wv[layer], q_dim_ok);
            }
            self.dev_rope(
                &mut d.q, &mut d.k_row, &d.inv, 1, &d.pos_dev, head_size, kv_dim, dim, n_freqs,
                mscale,
            );
            self.dev_kv_append(&mut d.key[layer], &d.k_row, &d.pos_dev, kv_dim);
            self.dev_kv_append(&mut d.value[layer], &d.v_row, &d.pos_dev, kv_dim);
            self.dev_attention(
                &mut d.xb, &d.q, &d.key[layer], &d.value[layer], 1, &d.pos_dev, n_heads, kv_mul,
                head_size, kv_dim,
            );
            if q_dim_ok {
                self.dev_quantize_q8k(&mut d.aq, &mut d.ad, &mut d.absums, &d.xb, dim);
            }
            self.gemv_decode(&mut d.xb2, &d.xb, &d.aq, &d.ad, &d.absums, &w.wo[layer], q_dim_ok);
            self.dev_add(&mut d.x, &d.xb2, dim);

            // --- Feed-forward (SwiGLU) ---
            self.dev_rmsnorm(&mut d.xb, &d.x, &d.rms_ffn[layer], 1, dim, eps);
            if q_dim_ok {
                self.dev_quantize_q8k(&mut d.aq, &mut d.ad, &mut d.absums, &d.xb, dim);
            }
            self.gemv_decode(&mut d.hb, &d.xb, &d.aq, &d.ad, &d.absums, &w.w1[layer], q_dim_ok);
            self.gemv_decode(&mut d.hb2, &d.xb, &d.aq, &d.ad, &d.absums, &w.w3[layer], q_dim_ok);
            self.dev_swiglu(&mut d.hb, &d.hb2, hidden);
            if q_hid_ok {
                self.dev_quantize_q8k(&mut d.aq, &mut d.ad, &mut d.absums, &d.hb, hidden);
            }
            self.gemv_decode(&mut d.xb, &d.hb, &d.aq, &d.ad, &d.absums, &w.w2[layer], q_hid_ok);
            self.dev_add(&mut d.x, &d.xb, dim);
        }
        // Final RMSNorm + classifier → logits.
        self.dev_rmsnorm(&mut d.xb, &d.x, &d.rms_final, 1, dim, eps);
        if q_dim_ok {
            self.dev_quantize_q8k(&mut d.aq, &mut d.ad, &mut d.absums, &d.xb, dim);
        }
        self.gemv_decode(&mut d.logits, &d.xb, &d.aq, &d.ad, &d.absums, &w.wcls, q_dim_ok);
    }
}

/// Whether the packed-weight DP4A decode GEMV (Phase 2) is enabled. Process-once
/// env read, mirroring `RUSTY_LLAMA_NO_INT8` / `RUSTY_LLAMA_NO_AVX2`. **Adopted**
/// on a measured 2.4× decode win on the real TinyLlama Q4_K_M (86 → 207 tok/s,
/// `PERFORMANCE.md`), so it is **default-on** for Q4_K/Q6_K with an opt-out:
/// set `RUSTY_LLAMA_CUDA_NO_QGEMV` to fall back to the f16 cuBLASLt GEMV.
fn qgemv_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("RUSTY_LLAMA_CUDA_NO_QGEMV").is_err())
}

/// Whether the fused q/k/v decode GEMV is used (default on, when the layout is
/// Q4_K/Q4_K/Q6_K). Opt out with `RUSTY_LLAMA_CUDA_NO_QKV_FUSE` to A/B against the
/// three separate GEMVs.
fn qkv_fuse_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("RUSTY_LLAMA_CUDA_NO_QKV_FUSE").is_err())
}

/// Whether the captured-CUDA-graph decode replay is enabled (default on). Opt out
/// with `RUSTY_LLAMA_CUDA_NO_GRAPH` to run the per-step kernels directly — for A/B
/// benchmarking the graph and as an escape hatch.
fn graph_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("RUSTY_LLAMA_CUDA_NO_GRAPH").is_err())
}

// Per-op trait surface: the matmuls run on cuBLASLt (f16 tensor cores) and the
// elementwise/norm ops delegate to the CPU backend. `forward_prefill` drives all
// its GEMMs through `matmul`/`matmul_batch`, so it captures the prefill tensor-core
// win. Decode is NOT this path: `forward_step` (below) is a fully on-device resident
// loop — KV cache + activations stay on the GPU across steps, weights are dot-producted
// with packed `__dp4a` GEMVs, and the per-step kernel sequence is captured into a CUDA
// graph and replayed. (Only non-dense/MoE archs fall back to the generic per-op path.)
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
        // Non-Llama archs (Qwen2/Phi-3/Gemma2) and MoE models run the generic
        // per-op path; the resident fused decode below is the dense-Llama-only
        // fast path.
        if !model.config.uses_resident_decode() {
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
        let (dim, kv_dim, hidden) = (d.dim, d.kv_dim, d.hidden);
        let eps = p.rms_eps;
        // Packed DP4A decode GEMV (Phase 2), gated. The activation is quantized
        // ONCE per shared input (q8k needs len % 256 == 0) and reused across the
        // sibling GEMVs; gemv_decode routes Q4_K/Q6_K to the packed kernels and
        // anything else (or when off) to the f16 GEMM.
        let q_dim_ok = qgemv_enabled() && dim % 256 == 0;
        let q_hid_ok = qgemv_enabled() && hidden % 256 == 0;

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
        // The absolute position, device-resident so rope/attention/kv-append read
        // it at run time — lets the per-step kernel sequence be captured once.
        st.memcpy_htod(&[pos as i32], &mut d.pos_dev).expect("upload pos");

        d.kv_filled = pos + 1;
        // Collapse the ~445 per-step kernel launches into one CUDA-graph replay —
        // decode is launch/enqueue-bound, so this is the lever. Eligible only when
        // every op is a custom capturable kernel (q-gemv on + all weights
        // k-quant); otherwise run the kernels directly each step.
        let use_graph = d.graphable && q_dim_ok && q_hid_ok && graph_enabled();
        let replay = use_graph && d.graph.is_some();
        if replay {
            d.graph.as_ref().unwrap().0.launch().expect("decode graph replay");
        } else {
            if use_graph {
                // Warm the packed-weight cache so its one-time host->device uploads
                // are NOT recorded into the graph; flush them before capture.
                for layer in 0..d.n_layers {
                    for wt in [
                        &w.wq[layer], &w.wk[layer], &w.wv[layer], &w.wo[layer], &w.w1[layer],
                        &w.w2[layer], &w.w3[layer],
                    ] {
                        self.weight_packed(wt);
                    }
                }
                self.weight_packed(&w.wcls);
                st.synchronize().expect("warm sync");
                st.begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
                .expect("begin capture");
            }
            self.decode_step_kernels(d, w, eps, q_dim_ok, q_hid_ok);
            if use_graph {
                let g = self
                    .stream
                    .end_capture(
                        cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY,
                    )
                    .expect("end capture")
                    .expect("captured graph is non-null");
                d.graph = Some(SendGraph(g));
                d.graph.as_ref().unwrap().0.launch().expect("decode graph first launch");
            }
        }
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
        if pos_base != 0 || !model.config.uses_resident_decode() {
            crate::model::forward_prefill(model, state, self, tokens, pos_base);
            return;
        }

        let p = &model.config;
        let w = &model.weights;
        let (dim, kv_dim, head_size, hidden) = (p.dim, p.kv_dim(), p.head_size(), p.hidden_dim);
        let (n_heads, kv_mul, eps) = (p.n_heads, p.n_heads / p.n_kv_heads, p.rms_eps);
        let n = tokens.len();
        let st = &self.stream;

        // Resident decode state: prefill writes K/V straight into it (2.2), so the
        // prefill→decode handoff needs no host round-trip. Build/reuse exactly like
        // forward_step; the resident RMSNorm weights + inv_freq are reused too.
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
        let (n_freqs, mscale) = (d.n_freqs, d.mscale);

        // Seed x with the (dequantized) token embeddings; one upload.
        let mut x_host = vec![0.0f32; n * dim];
        for (r, &tok) in tokens.iter().enumerate() {
            w.token_embedding_table
                .dequant_row(tok, &mut x_host[r * dim..r * dim + dim]);
        }
        let mut x = st.clone_htod(&x_host).expect("upload embeddings");

        // Batch activation scratch (sized n; the resident decode scratch is
        // single-token). K/V are written into the resident per-layer cache at rows
        // 0..n, so `att`'s row stride is n (pos_base == 0).
        let mut xb = st.alloc_zeros::<f32>(n * dim).expect("alloc xb");
        let mut xb2 = st.alloc_zeros::<f32>(n * dim).expect("alloc xb2");
        let mut q = st.alloc_zeros::<f32>(n * dim).expect("alloc q");
        let mut hb = st.alloc_zeros::<f32>(n * hidden).expect("alloc hb");
        let mut hb2 = st.alloc_zeros::<f32>(n * hidden).expect("alloc hb2");
        // Shared f16 narrowing of each norm's activation: q/k/v reuse one narrow,
        // w1/w3 another, instead of `gemm_dev` re-narrowing the same activation
        // per GEMM. SAFETY: fully written by `dev_cvt_to_f16` before each read.
        let mut xb16 = unsafe { st.alloc::<f16>(n * dim) }.expect("alloc xb16");
        // Prefill positions are `0 + row` (rows 0..n at base 0); rope/attention now
        // read the base from device memory, so set it once.
        st.memcpy_htod(&[0i32], &mut d.pos_dev).expect("upload pos_base=0");

        for layer in 0..p.n_layers {
            // --- Attention ---
            self.dev_rmsnorm(&mut xb, &x, &d.rms_att[layer], n, dim, eps);
            self.dev_cvt_to_f16(&mut xb16, &xb, n * dim);
            self.gemm_dev_f16(&mut q, &xb16, &w.wq[layer], n);
            self.gemm_dev_f16(&mut d.key[layer], &xb16, &w.wk[layer], n);
            self.gemm_dev_f16(&mut d.value[layer], &xb16, &w.wv[layer], n);
            self.dev_rope(
                &mut q, &mut d.key[layer], &d.inv, n, &d.pos_dev, head_size, kv_dim, dim, n_freqs,
                mscale,
            );
            self.dev_attention(
                &mut xb, &q, &d.key[layer], &d.value[layer], n, &d.pos_dev, n_heads, kv_mul,
                head_size, kv_dim,
            );
            self.gemm_dev(&mut xb2, &xb, &w.wo[layer], n);
            self.dev_add(&mut x, &xb2, n * dim);

            // --- Feed-forward (SwiGLU) ---
            self.dev_rmsnorm(&mut xb, &x, &d.rms_ffn[layer], n, dim, eps);
            self.dev_cvt_to_f16(&mut xb16, &xb, n * dim);
            self.gemm_dev_f16(&mut hb, &xb16, &w.w1[layer], n);
            self.gemm_dev_f16(&mut hb2, &xb16, &w.w3[layer], n);
            self.dev_swiglu(&mut hb, &hb2, n * hidden);
            self.gemm_dev(&mut xb, &hb, &w.w2[layer], n);
            self.dev_add(&mut x, &xb, n * dim);
        }
        // The prompt's K/V is now resident in the decode cache — decode reads it
        // with no upload (kv_filled = n).
        d.kv_filled = n;

        // Final RMSNorm of the last position + classifier, on-device: copy the last
        // residual row into the resident decode scratch (d.x) and reuse the exact
        // decode final-norm + packed-DP4A classifier (d.rms_final / d.xb / d.aq.. /
        // gemv_decode against w.wcls). Only the vocab-sized logits come back, versus
        // pulling the whole n*dim residual to the host and running the rmsnorm +
        // classifier GEMV there. Same kernels the decode path uses, so prefill->decode
        // logit coherence holds (and the last-token logits now match decode's
        // classifier exactly).
        let last = (n - 1) * dim;
        st.memcpy_dtod(&x.slice(last..last + dim), &mut d.x)
            .expect("copy last residual row to decode scratch");
        let q_dim_ok = qgemv_enabled() && dim % 256 == 0;
        // Normalize into a dim-length scratch (NOT the shared d.xb, which is sized to
        // hidden_dim): gemv_decode's gemm_dev fallback for a non-packable classifier
        // asserts the input is exactly `cols` long.
        let mut xbn = st.alloc_zeros::<f32>(dim).expect("alloc normed last row");
        self.dev_rmsnorm(&mut xbn, &d.x, &d.rms_final, 1, dim, eps);
        if q_dim_ok {
            self.dev_quantize_q8k(&mut d.aq, &mut d.ad, &mut d.absums, &xbn, dim);
        }
        self.gemv_decode(&mut d.logits, &xbn, &d.aq, &d.ad, &d.absums, &w.wcls, q_dim_ok);
        let logits_host: Vec<f32> = st.clone_dtoh(&d.logits).expect("download logits");
        state.logits_mut().copy_from_slice(&logits_host);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{
        quantize_activation_q8k, quantize_q8_0, vec_dot_q4_k, vec_dot_q6_k, GgmlType,
    };

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

    /// Coherence guard for the CUDA-**graph** decode path on a real k-quant model
    /// (graphable=true, so the captured graph is exercised — the synthetic
    /// coherence tests use F32 weights and take the direct device-`pos` path).
    /// Decodes several steps on CUDA (graph capture + replay, `pos` advancing) and
    /// on the CPU oracle, asserting the greedy argmax matches each step. Needs the
    /// real GGUF + a CUDA device.
    #[test]
    #[ignore = "needs the real GGUF + a CUDA device; run with --features cuda -- --ignored --nocapture"]
    fn decode_graph_coherent_real() {
        let Some(b) = cuda() else { return };
        let path = std::env::var("RUSTY_LLAMA_GGUF")
            .unwrap_or_else(|_| "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".to_string());
        let cp = match crate::Checkpoint::open(&path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping graph coherence: can't open {path}: {e}");
                return;
            }
        };
        if !crate::Gguf::is_gguf(cp.bytes()) {
            eprintln!("skipping graph coherence: {path} is not a GGUF");
            return;
        }
        let gguf = crate::Gguf::parse(cp.bytes()).expect("parse gguf");
        let model = crate::Model::from_gguf(&gguf).expect("build model");
        let cpu = CpuBackend::new();
        let argmax = |v: &[f32]| {
            v.iter().enumerate().max_by(|a, x| a.1.partial_cmp(x.1).unwrap()).unwrap().0
        };
        let mut sc = crate::RunState::new(&model.config);
        let mut sg = crate::RunState::new(&model.config);
        let steps = 16usize;
        // Greedy self-feed: both backends consume the SAME (CPU-chosen) tokens, so
        // the activations stay in-distribution — the realistic decode path, where
        // the int8 GEMV's quant error is small (arbitrary tokens amplify it).
        let mut tok = 1usize;
        let mut maxrel = 0.0f32;
        for pos in 0..steps {
            cpu.forward_step(&model, &mut sc, tok, pos);
            b.forward_step(&model, &mut sg, tok, pos);
            let (lc, lg) = (sc.logits(), sg.logits());
            let num: f32 = lc.iter().zip(lg).map(|(a, x)| (a - x) * (a - x)).sum();
            let den: f32 = lc.iter().map(|v| v * v).sum();
            let rel = (num / den).sqrt();
            maxrel = maxrel.max(rel);
            // int8 decode differs from the CPU oracle by small amounts that can
            // flip a near-tie argmax (cf. the coherent CLI run); rel L2 is the
            // meaningful coherence metric, matching the other coherence tests.
            assert!(rel < 0.1, "step {pos}: cuda-graph decode diverged from CPU, rel L2 {rel}");
            tok = argmax(lc);
        }
        eprintln!("graph decode coherent over {steps} greedy steps (max rel L2 {maxrel:.4})");
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

        // llama-bench parity: the RunState (KV cache + scratch) is allocated ONCE,
        // outside the timed region, and reused across reps — exactly as llama.cpp
        // keeps its KV cache resident. Each rep rewrites KV slots 0..steps in order
        // before reading them, so no per-rep reset is needed. (Previously
        // RunState::new lived inside the closure, so its ~tens-of-MB alloc+zero was
        // timed on every rep and understated tok/s; the CPU benches never did this.)
        let decode = |backend: &dyn Backend, s: &mut crate::RunState| {
            for pos in 0..steps {
                backend.forward_step(&model, s, (pos * 13 + 1) % c.vocab_size, pos);
            }
        };

        // bench_stat's first call is the untimed warmup (builds the resident decode
        // state + weight cache on device), then `-r` timed reps as mean ± stddev.
        let reps = crate::bench_util::bench_reps();
        let mut state = crate::RunState::new(&model.config);
        let (cuda_tps, cuda_sd) = crate::bench_util::bench_stat(
            steps, reps, std::time::Duration::from_secs(6), || decode(&b, &mut state));

        let cpu_backend = CpuBackend::new();
        let mut cpu_state = crate::RunState::new(&model.config);
        let (cpu_tps, _) = crate::bench_util::bench_stat(
            steps, 1, std::time::Duration::ZERO, || decode(&cpu_backend, &mut cpu_state));

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

        // RunState allocated once, outside timing, and reused (each call rewrites
        // KV slots 0..n before reading them) — so the timed region is pure compute.
        let prefill = |backend: &dyn Backend, s: &mut crate::RunState| {
            backend.forward_prefill(&model, s, &tokens, 0);
        };
        let mut state = crate::RunState::new(&model.config);

        prefill(&b, &mut state); // warm up: dequant + upload weights, make them resident
        let t = Instant::now();
        prefill(&b, &mut state);
        let cuda = t.elapsed();

        let cpu_backend = CpuBackend::new();
        let mut cpu_state = crate::RunState::new(&model.config);
        let t = Instant::now();
        prefill(&cpu_backend, &mut cpu_state);
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

        // RunState (~92 MB for this model) allocated ONCE, outside the timed
        // region, and reused across reps — like llama.cpp's resident KV cache.
        // Each rep rewrites KV positions 0..n before reading them, so no reset is
        // needed. (Previously RunState::new was inside the closure, so the
        // alloc+zero was timed every rep and understated pp512 by ~4-8%.)
        let prefill = |backend: &dyn Backend, s: &mut crate::RunState| {
            backend.forward_prefill(&model, s, &tokens, 0);
        };

        // bench_stat's first call is the untimed warmup (dequant + upload weights,
        // make them resident), then `-r` timed reps as mean ± stddev.
        let reps = crate::bench_util::bench_reps();
        let mut state = crate::RunState::new(&model.config);
        let (cuda_tps, cuda_sd) = crate::bench_util::bench_stat(
            n, reps, std::time::Duration::from_secs(6), || prefill(&b, &mut state));

        let cpu_backend = CpuBackend::new();
        let mut cpu_state = crate::RunState::new(&model.config);
        let (cpu_tps, _) = crate::bench_util::bench_stat(
            n, 1, std::time::Duration::ZERO, || prefill(&cpu_backend, &mut cpu_state));

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
        let posd = b.stream.clone_htod(&[3i32]).unwrap();
        b.dev_rope(&mut qd, &mut kd, &invd, n, &posd, head_size, kv_dim, q_dim, inv.len(), 1.0);
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
        let posd = b.stream.clone_htod(&[0i32]).unwrap();
        b.dev_attention(
            &mut outd, &qd, &kcd, &vcd, n, &posd, n_heads, n_heads / n_kv_heads, head_size,
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

    #[test]
    #[ignore = "requires a CUDA device; run with --features cuda -- --ignored --nocapture"]
    fn dev_attention_flash_long_context_parity() {
        // seq_len far beyond one tile (BLOCK=128) AND beyond what the old
        // shared-memory score buffer could hold ((head_size + seq_len + 128) * 4
        // > 48 KB): proves the tiled online-softmax combine is correct and the
        // context is no longer bounded by shared memory.
        let Some(b) = cuda() else { return };
        let (n_heads, n_kv_heads, head_size, seq_len) = (2usize, 1usize, 8usize, 16384usize);
        let kv_dim = n_kv_heads * head_size;
        let dim = n_heads * head_size;
        let (rows, pos_base) = (2usize, 16000usize); // rows attend ~125 tiles
        let q = noise(rows * dim, 11);
        let kc = noise(seq_len * kv_dim, 12);
        let vc = noise(seq_len * kv_dim, 13);

        let qd = b.stream.clone_htod(&q).unwrap();
        let kcd = b.stream.clone_htod(&kc).unwrap();
        let vcd = b.stream.clone_htod(&vc).unwrap();
        let mut outd = b.stream.alloc_zeros::<f32>(rows * dim).unwrap();
        let posd = b.stream.clone_htod(&[pos_base as i32]).unwrap();
        b.dev_attention(
            &mut outd, &qd, &kcd, &vcd, rows, &posd, n_heads, n_heads / n_kv_heads, head_size,
            kv_dim,
        );
        b.stream.synchronize().unwrap();
        let got = b.stream.clone_dtoh(&outd).unwrap();

        let mut want = vec![0.0; rows * dim];
        let mut att = vec![0.0; n_heads * seq_len];
        CpuBackend.attention_batch(
            &mut want, &q, &kc, &vc, &mut att, pos_base, rows, n_heads, n_kv_heads, head_size,
            seq_len, kv_dim, 0.0,
        );
        close_approx(&got, &want, 1e-4, 1e-4);
    }

    #[test]
    #[ignore = "requires a CUDA device; run with --features cuda -- --ignored --nocapture"]
    fn dev_quantize_q8k_parity() {
        let Some(b) = cuda() else { return };
        let st = &b.stream;
        let n = 256 * 3;
        let mut x = noise(n, 21);
        // Block 1: all zero -> the zero path (d=0, qs=0, bsums=0).
        for v in x[256..512].iter_mut() {
            *v = 0.0;
        }
        // Block 2: a |v| tie at the max — index 512 (+0.9) must win over 600 (-0.9),
        // else iscale flips sign and every quant negates. noise elsewhere is < 0.5.
        x[512] = 0.9;
        x[600] = -0.9;

        let oracle = crate::quant::quantize_activation_q8k(&x);

        let x_dev = st.clone_htod(&x).unwrap();
        let mut aq = unsafe { st.alloc::<i8>(n) }.unwrap();
        let mut ad = unsafe { st.alloc::<f32>(n / 256) }.unwrap();
        let mut absums = unsafe { st.alloc::<i32>(n / 16) }.unwrap();
        b.dev_quantize_q8k(&mut aq, &mut ad, &mut absums, &x_dev, n);
        st.synchronize().unwrap();
        let aq_h: Vec<i8> = st.clone_dtoh(&aq).unwrap();
        let ad_h: Vec<f32> = st.clone_dtoh(&ad).unwrap();
        let absums_h: Vec<i32> = st.clone_dtoh(&absums).unwrap();

        assert_eq!(aq_h, oracle.qs, "device aq must match the oracle exactly");
        for (g, (&dev, &orc)) in absums_h.iter().zip(&oracle.bsums).enumerate() {
            assert_eq!(dev, orc as i32, "absums[{g}]");
        }
        for (sb, (&dev, &orc)) in ad_h.iter().zip(&oracle.d).enumerate() {
            assert!(
                (dev - orc).abs() <= orc.abs() * 1e-6 + 1e-9,
                "ad[{sb}] {dev} vs {orc}"
            );
        }
    }

    /// Random-but-valid Q4_K bytes: `nblocks` 144-byte super-blocks with non-NaN
    /// f16 d/dmin and random scales/qs. Parity only needs both the kernel and the
    /// oracle to read the same bytes, so the values need not be a real quantization.
    fn rand_q4_k(nblocks: usize, seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        let mut v: Vec<u8> = (0..nblocks * 144)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s & 0xff) as u8
            })
            .collect();
        for blk in 0..nblocks {
            let base = blk * 144;
            let d = crate::quant::f32_to_f16(0.03 + (blk % 7) as f32 * 0.002);
            let dm = crate::quant::f32_to_f16(0.015 + (blk % 5) as f32 * 0.001);
            v[base..base + 2].copy_from_slice(&d.to_le_bytes());
            v[base + 2..base + 4].copy_from_slice(&dm.to_le_bytes());
        }
        v
    }

    #[test]
    #[ignore = "requires a CUDA device; run with --features cuda -- --ignored --nocapture"]
    fn dev_gemv_q4_k_parity() {
        let Some(b) = cuda() else { return };
        let st = &b.stream;
        let (oc, ic) = (20usize, 512usize); // non-square row count; 2 super-blocks/row
        let nsb = ic / 256;
        let wbytes = rand_q4_k(oc * nsb, 7);
        let x = noise(ic, 9);
        let act = quantize_activation_q8k(&x);

        let w_dev = st.clone_htod(&wbytes).unwrap();
        let aq_dev = st.clone_htod(&act.qs).unwrap();
        let ad_dev = st.clone_htod(&act.d).unwrap();
        let absums: Vec<i32> = act.bsums.iter().map(|&v| v as i32).collect();
        let absums_dev = st.clone_htod(&absums).unwrap();
        let mut out_dev = st.alloc_zeros::<f32>(oc).unwrap();
        b.dev_gemv_q4_k(&mut out_dev, &w_dev, &aq_dev, &ad_dev, &absums_dev, oc, nsb);
        st.synchronize().unwrap();
        let got: Vec<f32> = st.clone_dtoh(&out_dev).unwrap();

        let rb = nsb * 144;
        let want: Vec<f32> = (0..oc)
            .map(|r| vec_dot_q4_k(&wbytes[r * rb..r * rb + rb], &act))
            .collect();
        close_approx(&got, &want, 1e-2, 2e-3);
    }

    /// Random-but-valid Q6_K bytes: `nblocks` 210-byte super-blocks with a non-NaN
    /// f16 `d` and random ql/qh/scales (signed). For parity only (see `rand_q4_k`).
    fn rand_q6_k(nblocks: usize, seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        let mut v: Vec<u8> = (0..nblocks * 210)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s & 0xff) as u8
            })
            .collect();
        for blk in 0..nblocks {
            let base = blk * 210;
            let d = crate::quant::f32_to_f16(0.02 + (blk % 6) as f32 * 0.001);
            v[base + 208..base + 210].copy_from_slice(&d.to_le_bytes());
        }
        v
    }

    #[test]
    #[ignore = "requires a CUDA device; run with --features cuda -- --ignored --nocapture"]
    fn dev_gemv_q6_k_parity() {
        let Some(b) = cuda() else { return };
        let st = &b.stream;
        let (oc, ic) = (18usize, 512usize); // non-square; 2 super-blocks/row
        let nsb = ic / 256;
        let wbytes = rand_q6_k(oc * nsb, 11);
        let x = noise(ic, 13);
        let act = quantize_activation_q8k(&x);

        let w_dev = st.clone_htod(&wbytes).unwrap();
        let aq_dev = st.clone_htod(&act.qs).unwrap();
        let ad_dev = st.clone_htod(&act.d).unwrap();
        let absums: Vec<i32> = act.bsums.iter().map(|&v| v as i32).collect();
        let absums_dev = st.clone_htod(&absums).unwrap();
        let mut out_dev = st.alloc_zeros::<f32>(oc).unwrap();
        b.dev_gemv_q6_k(&mut out_dev, &w_dev, &aq_dev, &ad_dev, &absums_dev, oc, nsb);
        st.synchronize().unwrap();
        let got: Vec<f32> = st.clone_dtoh(&out_dev).unwrap();

        let rb = nsb * 210;
        let want: Vec<f32> = (0..oc)
            .map(|r| vec_dot_q6_k(&wbytes[r * rb..r * rb + rb], &act))
            .collect();
        close_approx(&got, &want, 1e-2, 2e-3);
    }

    #[test]
    #[ignore = "requires a CUDA device; run with --features cuda -- --ignored --nocapture"]
    fn dev_gemv_qkv_parity() {
        let Some(b) = cuda() else { return };
        let st = &b.stream;
        // Q4_K_M GQA shape: q large, k/v small; shared `ic` (one activation). q/k
        // are Q4_K, v is Q6_K. Row counts cross warp/range boundaries.
        let (q_oc, kv_oc, ic) = (20usize, 6usize, 512usize);
        let nsb = ic / 256;
        let wq = rand_q4_k(q_oc * nsb, 21);
        let wk = rand_q4_k(kv_oc * nsb, 22);
        let wv = rand_q6_k(kv_oc * nsb, 23);
        let x = noise(ic, 24);
        let act = quantize_activation_q8k(&x);

        let wq_d = st.clone_htod(&wq).unwrap();
        let wk_d = st.clone_htod(&wk).unwrap();
        let wv_d = st.clone_htod(&wv).unwrap();
        let aq_d = st.clone_htod(&act.qs).unwrap();
        let ad_d = st.clone_htod(&act.d).unwrap();
        let absums: Vec<i32> = act.bsums.iter().map(|&v| v as i32).collect();
        let absums_d = st.clone_htod(&absums).unwrap();
        let mut q_d = st.alloc_zeros::<f32>(q_oc).unwrap();
        let mut k_d = st.alloc_zeros::<f32>(kv_oc).unwrap();
        let mut v_d = st.alloc_zeros::<f32>(kv_oc).unwrap();
        b.dev_gemv_qkv(
            &mut q_d, &mut k_d, &mut v_d, &wq_d, &wk_d, &wv_d, &aq_d, &ad_d, &absums_d, q_oc,
            kv_oc, nsb,
        );
        st.synchronize().unwrap();
        let gq: Vec<f32> = st.clone_dtoh(&q_d).unwrap();
        let gk: Vec<f32> = st.clone_dtoh(&k_d).unwrap();
        let gv: Vec<f32> = st.clone_dtoh(&v_d).unwrap();

        let (rb4, rb6) = (nsb * 144, nsb * 210);
        let wantq: Vec<f32> = (0..q_oc).map(|r| vec_dot_q4_k(&wq[r * rb4..r * rb4 + rb4], &act)).collect();
        let wantk: Vec<f32> = (0..kv_oc).map(|r| vec_dot_q4_k(&wk[r * rb4..r * rb4 + rb4], &act)).collect();
        let wantv: Vec<f32> = (0..kv_oc).map(|r| vec_dot_q6_k(&wv[r * rb6..r * rb6 + rb6], &act)).collect();
        close_approx(&gq, &wantq, 1e-2, 2e-3);
        close_approx(&gk, &wantk, 1e-2, 2e-3);
        close_approx(&gv, &wantv, 1e-2, 2e-3);
    }

    #[test]
    #[ignore = "timing benchmark; run with --features cuda -- --ignored --nocapture"]
    fn bench_decode_gemv_quant_vs_f16() {
        let Some(b) = cuda() else { return };
        let st = &b.stream;
        use std::time::Instant;
        let iters = 100;
        // Real TinyLlama decode GEMV shapes (oc × ic): attn q/o, FFN up/gate, down.
        for (oc, ic) in [(2048usize, 2048usize), (5632, 2048), (2048, 5632)] {
            let nsb = ic / 256;
            let wbytes = rand_q4_k(oc * nsb, 3);
            let w = QMatrix::quant(GgmlType::Q4_K, wbytes.clone().into(), oc, ic).unwrap();
            let x = noise(ic, 5);
            let act = quantize_activation_q8k(&x);
            let wp = st.clone_htod(&wbytes).unwrap();
            let aq = st.clone_htod(&act.qs).unwrap();
            let ad = st.clone_htod(&act.d).unwrap();
            let absums: Vec<i32> = act.bsums.iter().map(|&v| v as i32).collect();
            let absums_d = st.clone_htod(&absums).unwrap();
            let mut out = st.alloc_zeros::<f32>(oc).unwrap();
            let x_dev = st.clone_htod(&x).unwrap();
            let mut c = st.alloc_zeros::<f32>(oc).unwrap();
            // Warm up both paths.
            b.gemm_dev(&mut c, &x_dev, &w, 1);
            b.dev_gemv_q4_k(&mut out, &wp, &aq, &ad, &absums_d, oc, nsb);
            st.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..iters {
                b.gemm_dev(&mut c, &x_dev, &w, 1);
            }
            st.synchronize().unwrap();
            let f16_t = t0.elapsed();
            let t1 = Instant::now();
            for _ in 0..iters {
                b.dev_gemv_q4_k(&mut out, &wp, &aq, &ad, &absums_d, oc, nsb);
            }
            st.synchronize().unwrap();
            let pk_t = t1.elapsed();
            eprintln!(
                "GEMV {oc}x{ic} x{iters}: f16 {f16_t:?}  packed-Q4_K {pk_t:?}  speedup {:.2}x",
                f16_t.as_secs_f64() / pk_t.as_secs_f64()
            );
        }
    }

    /// L7 feasibility spike: can an int8 cuBLASLt GEMM (IMMA) beat the f16
    /// tensor-core GEMM at a prefill shape with **ordinary col-major layout** (no
    /// COL32 transform)? Random int8 data — timing only. This is the kill gate for
    /// the int8 MMQ treadmill: if ordinary layout is unsupported (needs COL32
    /// transforms) or int8 isn't meaningfully faster, the integration isn't worth it.
    /// L7 gate: can nvrtc compile + run an int8 TENSOR-CORE op for sm_120, HEADER-FREE
    /// (no <mma.h>, which nvrtc can't find here)? cuBLASLt int8 is proven fast but can't
    /// do the per-sub-block scaling MMQ needs, so MMQ requires a hand-written tensor-core
    /// kernel. This probes raw inline-PTX `mma.sync.m16n8k32.s8` with a hand-laid-out
    /// 16x8x32 fragment, the same header-free convention used for cvt/dp4a.
    #[test]
    #[ignore = "L7 feasibility probe; run with --features cuda -- --ignored --nocapture"]
    fn probe_int8_mma_ptx() {
        let Some(b) = cuda() else {
            eprintln!("no CUDA device");
            return;
        };
        // D[16x8] = A[16x32] * B[32x8], int8 inputs, int32 accumulate. One warp.
        // Fragment layout for m16n8k32 (PTX ISA): gid=lane/4, tig=lane%4.
        const SRC: &str = r#"
extern "C" __global__ void mma_i8(const signed char* A, const signed char* B, int* D) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;      // 0..7
    int tig = lane & 3;       // 0..3
    unsigned a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
    for (int t=0; t<4; t++) {
        a0 |= ((unsigned)(unsigned char)A[gid*32      + tig*4 + t])      << (8*t);
        a1 |= ((unsigned)(unsigned char)A[(gid+8)*32  + tig*4 + t])      << (8*t);
        a2 |= ((unsigned)(unsigned char)A[gid*32      + tig*4 + 16 + t]) << (8*t);
        a3 |= ((unsigned)(unsigned char)A[(gid+8)*32  + tig*4 + 16 + t]) << (8*t);
        b0 |= ((unsigned)(unsigned char)B[(tig*4 + t)*8      + gid])     << (8*t);
        b1 |= ((unsigned)(unsigned char)B[(tig*4 + 16 + t)*8 + gid])     << (8*t);
    }
    int c0=0,c1=0,c2=0,c3=0, d0,d1,d2,d3;
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(c0),"r"(c1),"r"(c2),"r"(c3));
    D[gid*8 + tig*2]         = d0;
    D[gid*8 + tig*2 + 1]     = d1;
    D[(gid+8)*8 + tig*2]     = d2;
    D[(gid+8)*8 + tig*2 + 1] = d3;
}
"#;
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some("compute_120"),
            ..Default::default()
        };
        let ptx = match cudarc::nvrtc::compile_ptx_with_opts(SRC, opts) {
            Ok(p) => p,
            Err(e) => panic!("int8 mma.sync nvrtc compile FAILED (gate): {e:?}"),
        };
        let st = &b.stream;
        let m = b.ctx.load_module(ptx).expect("load module");
        let f = m.load_function("mma_i8").expect("load fn");
        let ah: Vec<i8> = (0..512).map(|i| ((i % 7) as i32 - 3) as i8).collect(); // 16x32
        let bh: Vec<i8> = (0..256).map(|i| ((i % 5) as i32 - 2) as i8).collect(); // 32x8
        let ad = st.clone_htod(&ah).unwrap();
        let bd = st.clone_htod(&bh).unwrap();
        let mut dd = st.alloc_zeros::<i32>(128).unwrap(); // 16x8
        {
            let mut lb = st.launch_builder(&f);
            lb.arg(&ad).arg(&bd).arg(&mut dd);
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg) }.expect("launch");
        }
        let d: Vec<i32> = st.clone_dtoh(&dd).unwrap();
        // D[i][j] = sum_k A[i*32+k] * B[k*8+j]
        let mut bad = 0;
        for i in 0..16 {
            for j in 0..8 {
                let mut acc = 0i32;
                for k in 0..32 {
                    acc += ah[i * 32 + k] as i32 * bh[k * 8 + j] as i32;
                }
                if d[i * 8 + j] != acc {
                    if bad < 8 {
                        eprintln!("[{i}][{j}] gpu={} ref={}", d[i * 8 + j], acc);
                    }
                    bad += 1;
                }
            }
        }
        assert_eq!(bad, 0, "int8 mma.sync mismatch in {bad}/128 cells");
        eprintln!("GATE PASSED: inline-PTX int8 mma.sync.m16n8k32 runs correctly on sm_120");
    }

    /// Phase 7 / Stage A: generalize the proven m16n8k32 fragment (probe above)
    /// into a TILED, K-accumulating int8 GEMM `C[M,N] = scale * (A[M,K] · Bᵀ)`,
    /// where `A` is the activation (M×K row-major) and `B` is the weight (N×K
    /// row-major, the `W[oc][ic]` prefill layout — i.e. K×N column-major, what
    /// `mma.row.col` wants). One warp owns one 16×8 output tile and loops K in
    /// steps of 32, chaining the s32 accumulator across mma.sync calls; the final
    /// accumulator is scaled to f32. No shared-memory reuse yet (Stage B adds it
    /// with the Q8_0 quant) — this only proves the tiling/fragment/accumulation.
    #[test]
    #[ignore = "L7 Stage A; run with --features cuda -- --ignored --nocapture"]
    fn mmq_stage_a_tiled_int8_gemm() {
        let Some(b) = cuda() else {
            eprintln!("no CUDA device");
            return;
        };
        // C[m][n] = scale * sum_k A[m*K+k] * B[n*K+k]. One warp per 16x8 tile;
        // grid.x over N/8 col-tiles, grid.y over M/16 row-tiles. Fragment layout
        // identical to probe_int8_mma_ptx, generalized to (row_base, col_base, kk).
        const SRC: &str = r#"
extern "C" __global__ void mmq_a(
    const signed char* A, const signed char* B, float* C,
    int M, int N, int K, float scale)
{
    int row_base = blockIdx.y * 16;
    int col_base = blockIdx.x * 8;
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;   // 0..7
    int tig = lane & 3;    // 0..3
    int c0=0, c1=0, c2=0, c3=0;
    for (int kk = 0; kk < K; kk += 32) {
        unsigned a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
        for (int t = 0; t < 4; t++) {
            a0 |= ((unsigned)(unsigned char)A[(row_base+gid)*K   + kk + tig*4 + t])      << (8*t);
            a1 |= ((unsigned)(unsigned char)A[(row_base+gid+8)*K + kk + tig*4 + t])      << (8*t);
            a2 |= ((unsigned)(unsigned char)A[(row_base+gid)*K   + kk + 16 + tig*4 + t]) << (8*t);
            a3 |= ((unsigned)(unsigned char)A[(row_base+gid+8)*K + kk + 16 + tig*4 + t]) << (8*t);
            b0 |= ((unsigned)(unsigned char)B[(col_base+gid)*K   + kk + tig*4 + t])      << (8*t);
            b1 |= ((unsigned)(unsigned char)B[(col_base+gid)*K   + kk + 16 + tig*4 + t]) << (8*t);
        }
        int d0,d1,d2,d3;
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(c0),"r"(c1),"r"(c2),"r"(c3));
        c0=d0; c1=d1; c2=d2; c3=d3;
    }
    C[(row_base+gid)*N   + col_base + tig*2]     = scale * (float)c0;
    C[(row_base+gid)*N   + col_base + tig*2 + 1] = scale * (float)c1;
    C[(row_base+gid+8)*N + col_base + tig*2]     = scale * (float)c2;
    C[(row_base+gid+8)*N + col_base + tig*2 + 1] = scale * (float)c3;
}
"#;
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some("compute_120"),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(SRC, opts).expect("nvrtc compile");
        let st = &b.stream;
        let m = b.ctx.load_module(ptx).expect("load module");
        let f = m.load_function("mmq_a").expect("load fn");

        // 2x2 output tiles, 2 K-steps: exercises tiling in all three dims.
        let (mm, nn, kk) = (32usize, 16usize, 64usize);
        let scale = 0.0125f32;
        let ah: Vec<i8> = (0..mm * kk).map(|i| ((i % 11) as i32 - 5) as i8).collect();
        let bh: Vec<i8> = (0..nn * kk).map(|i| ((i * 3 % 13) as i32 - 6) as i8).collect();
        let ad = st.clone_htod(&ah).unwrap();
        let bd = st.clone_htod(&bh).unwrap();
        let mut cd = st.alloc_zeros::<f32>(mm * nn).unwrap();
        {
            let mut lb = st.launch_builder(&f);
            let (mi, ni, ki) = (mm as i32, nn as i32, kk as i32);
            lb.arg(&ad).arg(&bd).arg(&mut cd).arg(&mi).arg(&ni).arg(&ki).arg(&scale);
            let cfg = LaunchConfig {
                grid_dim: ((nn / 8) as u32, (mm / 16) as u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg) }.expect("launch");
        }
        let c: Vec<f32> = st.clone_dtoh(&cd).unwrap();
        let mut bad = 0;
        let mut maxerr = 0.0f32;
        for i in 0..mm {
            for j in 0..nn {
                let mut acc = 0i32;
                for k in 0..kk {
                    acc += ah[i * kk + k] as i32 * bh[j * kk + k] as i32;
                }
                let want = scale * acc as f32;
                let err = (c[i * nn + j] - want).abs();
                maxerr = maxerr.max(err);
                if err > 1e-3 {
                    if bad < 8 {
                        eprintln!("[{i}][{j}] gpu={} ref={want}", c[i * nn + j]);
                    }
                    bad += 1;
                }
            }
        }
        assert_eq!(bad, 0, "tiled int8 GEMM mismatch in {bad}/{} cells", mm * nn);
        eprintln!("STAGE A PASSED: tiled int8 mma GEMM {mm}x{nn}x{kk} exact (maxerr {maxerr:.2e})");
    }

    /// Phase 7 / Stage B (correctness): Q8_0 MMQ — the per-block-scaled tiled mma.
    /// Q8_0 carries one f32 scale per 32 elements, so unlike Stage A the s32
    /// accumulator can't run across blocks: each 32-wide block is a fresh mma
    /// whose int32 dot is scaled by `d_act · d_weight` and accumulated in f32.
    /// This is the simplest MMQ and the Stage-B GO/NO-GO kernel. Verified here vs
    /// a host Q8_0×Q8_0 reference (same math) and sanity-checked vs the true f32
    /// dot (within Q8_0 quant error). Activation/weight are passed pre-separated
    /// (int8 quants + f32 per-block scales); reading the packed ggml block layout
    /// in-kernel is the next refinement.
    #[test]
    #[ignore = "L7 Stage B; run with --features cuda -- --ignored --nocapture"]
    fn mmq_stage_b_q8_0() {
        let Some(b) = cuda() else {
            eprintln!("no CUDA device");
            return;
        };
        const SRC: &str = r#"
extern "C" __global__ void mmq_q8_0(
    const signed char* qa, const float* da,   // activation: qa[M*K], da[M*(K/32)]
    const signed char* qw, const float* dw,   // weight:     qw[N*K], dw[N*(K/32)]
    float* C, int M, int N, int K)
{
    int row_base = blockIdx.y * 16;
    int col_base = blockIdx.x * 8;
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tig = lane & 3;
    int nblk = K / 32;
    float f0=0.f, f1=0.f, f2=0.f, f3=0.f;
    int z = 0;
    for (int blk = 0; blk < nblk; blk++) {
        int kk = blk * 32;
        unsigned a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
        for (int t = 0; t < 4; t++) {
            a0 |= ((unsigned)(unsigned char)qa[(row_base+gid)*K   + kk + tig*4 + t])      << (8*t);
            a1 |= ((unsigned)(unsigned char)qa[(row_base+gid+8)*K + kk + tig*4 + t])      << (8*t);
            a2 |= ((unsigned)(unsigned char)qa[(row_base+gid)*K   + kk + 16 + tig*4 + t]) << (8*t);
            a3 |= ((unsigned)(unsigned char)qa[(row_base+gid+8)*K + kk + 16 + tig*4 + t]) << (8*t);
            b0 |= ((unsigned)(unsigned char)qw[(col_base+gid)*K   + kk + tig*4 + t])      << (8*t);
            b1 |= ((unsigned)(unsigned char)qw[(col_base+gid)*K   + kk + 16 + tig*4 + t]) << (8*t);
        }
        int d0,d1,d2,d3;
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(z),"r"(z),"r"(z),"r"(z));
        float da0 = da[(row_base+gid)*nblk   + blk];
        float da8 = da[(row_base+gid+8)*nblk + blk];
        float dw0 = dw[(col_base+tig*2)*nblk     + blk];
        float dw1 = dw[(col_base+tig*2+1)*nblk   + blk];
        f0 += (float)d0 * da0 * dw0;
        f1 += (float)d1 * da0 * dw1;
        f2 += (float)d2 * da8 * dw0;
        f3 += (float)d3 * da8 * dw1;
    }
    C[(row_base+gid)*N   + col_base + tig*2]     = f0;
    C[(row_base+gid)*N   + col_base + tig*2 + 1] = f1;
    C[(row_base+gid+8)*N + col_base + tig*2]     = f2;
    C[(row_base+gid+8)*N + col_base + tig*2 + 1] = f3;
}
"#;
        // Host Q8_0 quant of one length-K row into per-32 (int8 q, f32 scale),
        // mirroring ggml quantize_row_q8_0 (d = amax/127, q = round(v/d)).
        fn quant_q8_0(v: &[f32], k: usize) -> (Vec<i8>, Vec<f32>) {
            let nblk = k / 32;
            let mut q = vec![0i8; k];
            let mut d = vec![0f32; nblk];
            for blk in 0..nblk {
                let s = &v[blk * 32..blk * 32 + 32];
                let amax = s.iter().fold(0f32, |m, &x| m.max(x.abs()));
                let scale = amax / 127.0;
                d[blk] = scale;
                let id = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                for (i, &x) in s.iter().enumerate() {
                    q[blk * 32 + i] = (x * id).round().clamp(-127.0, 127.0) as i8;
                }
            }
            (q, d)
        }

        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some("compute_120"),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(SRC, opts).expect("nvrtc compile");
        let st = &b.stream;
        let m = b.ctx.load_module(ptx).expect("load module");
        let f = m.load_function("mmq_q8_0").expect("load fn");

        let (mm, nn, kk) = (32usize, 16usize, 128usize); // 4 Q8_0 blocks per row
        let nblk = kk / 32;
        // Pseudo-random f32 activation and weight (deterministic, no rng dep).
        let af: Vec<f32> = (0..mm * kk).map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.21).collect();
        let wf: Vec<f32> = (0..nn * kk).map(|i| ((i * 5 % 19) as f32 - 9.0) * 0.17).collect();
        let (mut qa, mut da) = (vec![0i8; mm * kk], vec![0f32; mm * nblk]);
        for r in 0..mm {
            let (q, d) = quant_q8_0(&af[r * kk..r * kk + kk], kk);
            qa[r * kk..r * kk + kk].copy_from_slice(&q);
            da[r * nblk..r * nblk + nblk].copy_from_slice(&d);
        }
        let (mut qw, mut dw) = (vec![0i8; nn * kk], vec![0f32; nn * nblk]);
        for r in 0..nn {
            let (q, d) = quant_q8_0(&wf[r * kk..r * kk + kk], kk);
            qw[r * kk..r * kk + kk].copy_from_slice(&q);
            dw[r * nblk..r * nblk + nblk].copy_from_slice(&d);
        }

        let qad = st.clone_htod(&qa).unwrap();
        let dad = st.clone_htod(&da).unwrap();
        let qwd = st.clone_htod(&qw).unwrap();
        let dwd = st.clone_htod(&dw).unwrap();
        let mut cd = st.alloc_zeros::<f32>(mm * nn).unwrap();
        {
            let mut lb = st.launch_builder(&f);
            let (mi, ni, ki) = (mm as i32, nn as i32, kk as i32);
            lb.arg(&qad).arg(&dad).arg(&qwd).arg(&dwd).arg(&mut cd).arg(&mi).arg(&ni).arg(&ki);
            let cfg = LaunchConfig {
                grid_dim: ((nn / 8) as u32, (mm / 16) as u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg) }.expect("launch");
        }
        let c: Vec<f32> = st.clone_dtoh(&cd).unwrap();

        // Oracle 1 (hard gate): host Q8_0xQ8_0 — the exact same math the kernel
        //   does, so per-cell rel error must be ~0 (only f32 add-order differs).
        // Oracle 2 (quality): true f32 dot — measured as aggregate relative L2 over
        //   the whole matrix (per-cell rel error is meaningless under the heavy
        //   cancellation of random ± dot products; ‖C_q8 − C_f32‖ / ‖C_f32‖ is the
        //   standard quant-quality metric and stays small).
        let mut bad = 0;
        let mut maxrel_q = 0f32;
        let (mut err2, mut ref2) = (0f64, 0f64);
        for i in 0..mm {
            for j in 0..nn {
                let mut q8 = 0f32;
                let mut f32dot = 0f32;
                for blk in 0..nblk {
                    let mut idot = 0i32;
                    for t in 0..32 {
                        let k = blk * 32 + t;
                        idot += qa[i * kk + k] as i32 * qw[j * kk + k] as i32;
                        f32dot += af[i * kk + k] * wf[j * kk + k];
                    }
                    q8 += idot as f32 * da[i * nblk + blk] * dw[j * nblk + blk];
                }
                let got = c[i * nn + j];
                let rel_q = (got - q8).abs() / q8.abs().max(1e-3);
                maxrel_q = maxrel_q.max(rel_q);
                if rel_q > 1e-4 {
                    if bad < 8 {
                        eprintln!("[{i}][{j}] gpu={got} q8ref={q8} f32={f32dot}");
                    }
                    bad += 1;
                }
                err2 += (got - f32dot) as f64 * (got - f32dot) as f64;
                ref2 += f32dot as f64 * f32dot as f64;
            }
        }
        let rel_l2 = (err2 / ref2).sqrt();
        assert_eq!(bad, 0, "Q8_0 MMQ disagrees with host Q8_0 oracle in {bad} cells");
        assert!(rel_l2 < 0.05, "Q8_0 MMQ vs true f32 aggregate rel L2 too high: {rel_l2}");
        eprintln!(
            "STAGE B PASSED: Q8_0 MMQ {mm}x{nn}x{kk} matches host Q8_0 (max rel {maxrel_q:.2e}), \
             aggregate rel L2 vs true f32 {rel_l2:.4}"
        );
    }

    /// Shared-memory tiled Q8_0 MMQ: 32x32 output tile/block (8 warps, each owning
    /// one 16x8 mma sub-tile), staging A/B int8 sub-tiles + per-block scales into
    /// shared once per K-block so every global byte is read once per block-tile —
    /// the reuse that makes int8 competitive with cuBLASLt f16. Used by the tiled
    /// correctness test and the microbench. Requires M%32==N%32==K%32==0.
    const MMQ_Q8_0_TILED_SRC: &str = r#"
#define BM 32
#define BN 32
#define BK 32
extern "C" __global__ void mmq_q8_0_tiled(
    const signed char* qa, const float* da,   // qa[M*K], da[M*(K/32)]
    const signed char* qw, const float* dw,   // qw[N*K], dw[N*(K/32)]
    float* C, int M, int N, int K)
{
    __shared__ signed char sA[BM*BK];
    __shared__ signed char sB[BN*BK];
    __shared__ float sDa[BM];
    __shared__ float sDw[BN];
    int brow = blockIdx.y * BM;
    int bcol = blockIdx.x * BN;
    int tid = threadIdx.x;       // 0..255 = 8 warps
    int warp = tid >> 5, lane = tid & 31;
    int wr = warp >> 2, wc = warp & 3;   // warp owns rows [wr*16,+16), cols [wc*8,+8)
    int gid = lane >> 2, tig = lane & 3;
    int nblk = K / 32;
    float f0=0.f, f1=0.f, f2=0.f, f3=0.f;
    int z = 0;
    for (int blk = 0; blk < nblk; blk++) {
        int kk = blk * 32;
        for (int idx = tid; idx < BM*BK; idx += blockDim.x)
            sA[idx] = qa[(brow + idx/BK)*K + kk + idx%BK];
        for (int idx = tid; idx < BN*BK; idx += blockDim.x)
            sB[idx] = qw[(bcol + idx/BK)*K + kk + idx%BK];
        if (tid < BM) sDa[tid] = da[(brow + tid)*nblk + blk];
        if (tid < BN) sDw[tid] = dw[(bcol + tid)*nblk + blk];
        __syncthreads();
        unsigned a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
        for (int t = 0; t < 4; t++) {
            a0 |= ((unsigned)(unsigned char)sA[(wr*16+gid)*BK   + tig*4 + t])      << (8*t);
            a1 |= ((unsigned)(unsigned char)sA[(wr*16+gid+8)*BK + tig*4 + t])      << (8*t);
            a2 |= ((unsigned)(unsigned char)sA[(wr*16+gid)*BK   + 16 + tig*4 + t]) << (8*t);
            a3 |= ((unsigned)(unsigned char)sA[(wr*16+gid+8)*BK + 16 + tig*4 + t]) << (8*t);
            b0 |= ((unsigned)(unsigned char)sB[(wc*8+gid)*BK    + tig*4 + t])      << (8*t);
            b1 |= ((unsigned)(unsigned char)sB[(wc*8+gid)*BK    + 16 + tig*4 + t]) << (8*t);
        }
        int d0,d1,d2,d3;
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(z),"r"(z),"r"(z),"r"(z));
        float da0 = sDa[wr*16+gid], da8 = sDa[wr*16+gid+8];
        float dw0 = sDw[wc*8+tig*2], dw1 = sDw[wc*8+tig*2+1];
        f0 += (float)d0*da0*dw0;
        f1 += (float)d1*da0*dw1;
        f2 += (float)d2*da8*dw0;
        f3 += (float)d3*da8*dw1;
        __syncthreads();
    }
    int orow = brow + wr*16 + gid;
    int ocol = bcol + wc*8 + tig*2;
    C[orow*N + ocol]         = f0;
    C[orow*N + ocol + 1]     = f1;
    C[(orow+8)*N + ocol]     = f2;
    C[(orow+8)*N + ocol + 1] = f3;
}
"#;

    /// Quantize each of `rows` length-`k` rows to Q8_0 per-32 (int8 q + f32 scale),
    /// mirroring ggml `quantize_row_q8_0` (`d = amax/127`, `q = round(v/d)`).
    fn quant_rows_q8_0(v: &[f32], rows: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
        let nblk = k / 32;
        let mut q = vec![0i8; rows * k];
        let mut d = vec![0f32; rows * nblk];
        for r in 0..rows {
            for blk in 0..nblk {
                let s = &v[r * k + blk * 32..r * k + blk * 32 + 32];
                let amax = s.iter().fold(0f32, |m, &x| m.max(x.abs()));
                let scale = amax / 127.0;
                d[r * nblk + blk] = scale;
                let id = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                for (i, &x) in s.iter().enumerate() {
                    q[r * k + blk * 32 + i] = (x * id).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }
        (q, d)
    }

    /// Phase 7 / Stage B (shared-mem tiled) correctness: verify the tiled kernel
    /// (`MMQ_Q8_0_TILED_SRC`) vs the host Q8_0 oracle at a multi-block-tile shape.
    #[test]
    #[ignore = "L7 Stage B; run with --features cuda -- --ignored --nocapture"]
    fn mmq_stage_b_q8_0_tiled() {
        let Some(b) = cuda() else {
            eprintln!("no CUDA device");
            return;
        };
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some("compute_120"),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(MMQ_Q8_0_TILED_SRC, opts).expect("compile");
        let st = &b.stream;
        let m = b.ctx.load_module(ptx).expect("load module");
        let f = m.load_function("mmq_q8_0_tiled").expect("load fn");

        let (mm, nn, kk) = (64usize, 64usize, 128usize); // 2x2 block-tiles, 4 K-blocks
        let nblk = kk / 32;
        let af: Vec<f32> = (0..mm * kk).map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.21).collect();
        let wf: Vec<f32> = (0..nn * kk).map(|i| ((i * 5 % 19) as f32 - 9.0) * 0.17).collect();
        let (qa, da) = quant_rows_q8_0(&af, mm, kk);
        let (qw, dw) = quant_rows_q8_0(&wf, nn, kk);

        let qad = st.clone_htod(&qa).unwrap();
        let dad = st.clone_htod(&da).unwrap();
        let qwd = st.clone_htod(&qw).unwrap();
        let dwd = st.clone_htod(&dw).unwrap();
        let mut cd = st.alloc_zeros::<f32>(mm * nn).unwrap();
        {
            let mut lb = st.launch_builder(&f);
            let (mi, ni, ki) = (mm as i32, nn as i32, kk as i32);
            lb.arg(&qad).arg(&dad).arg(&qwd).arg(&dwd).arg(&mut cd).arg(&mi).arg(&ni).arg(&ki);
            let cfg = LaunchConfig {
                grid_dim: ((nn / 32) as u32, (mm / 32) as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg) }.expect("launch");
        }
        let c: Vec<f32> = st.clone_dtoh(&cd).unwrap();

        let mut bad = 0;
        let mut maxrel = 0f32;
        for i in 0..mm {
            for j in 0..nn {
                let mut q8 = 0f32;
                for blk in 0..nblk {
                    let mut idot = 0i32;
                    for t in 0..32 {
                        let k = blk * 32 + t;
                        idot += qa[i * kk + k] as i32 * qw[j * kk + k] as i32;
                    }
                    q8 += idot as f32 * da[i * nblk + blk] * dw[j * nblk + blk];
                }
                let rel = (c[i * nn + j] - q8).abs() / q8.abs().max(1e-3);
                maxrel = maxrel.max(rel);
                if rel > 1e-4 {
                    bad += 1;
                }
            }
        }
        assert_eq!(bad, 0, "tiled Q8_0 MMQ disagrees with host oracle in {bad} cells");
        eprintln!("STAGE B (tiled) PASSED: {mm}x{nn}x{kk} matches host Q8_0 (max rel {maxrel:.2e})");
    }

    /// Phase 7 / Stage B (tuning pass v2): the same Q8_0 MMQ with the three big
    /// algorithmic wins over the naive tiled kernel — a 64x64 block tile,
    /// REGISTER BLOCKING (each warp computes a 2x2 grid of 16x8 mma tiles, reusing
    /// 2 A-fragments x 2 B-fragments -> 4 mmas), and staging 128 K per sync so 4
    /// sub-blocks run between syncs (K/128 syncs instead of K/32). Requires
    /// M%64==0, N%64==0, K%128==0. Scalar shared staging (vectorized int4 loads +
    /// double-buffering are the next pass if the trajectory justifies it).
    const MMQ_Q8_0_V2_SRC: &str = r#"
#define BM 64
#define BN 64
#define BK_TILE 128
#define KSUB 4
#define WARPS_N 4
extern "C" __global__ void mmq_q8_0_v2(
    const signed char* qa, const float* da,
    const signed char* qw, const float* dw,
    float* C, int M, int N, int K)
{
    __shared__ signed char sA[BM*BK_TILE];
    __shared__ signed char sB[BN*BK_TILE];
    __shared__ float sDa[BM*KSUB];
    __shared__ float sDw[BN*KSUB];
    int brow = blockIdx.y * BM;
    int bcol = blockIdx.x * BN;
    int tid = threadIdx.x;
    int warp = tid >> 5, lane = tid & 31;
    int wr = warp / WARPS_N;        // 0..1 -> warp rows [wr*32, +32)
    int wc = warp % WARPS_N;        // 0..3 -> warp cols [wc*16, +16)
    int gid = lane >> 2, tig = lane & 3;
    int nblk_k = K / 32;

    float acc[2][2][4];
    for (int mi=0; mi<2; mi++) for (int ni=0; ni<2; ni++) for (int t=0; t<4; t++) acc[mi][ni][t]=0.f;
    int z = 0;

    for (int k0 = 0; k0 < K; k0 += BK_TILE) {
        for (int idx = tid; idx < BM*BK_TILE; idx += blockDim.x)
            sA[idx] = qa[(brow + idx/BK_TILE)*K + k0 + idx%BK_TILE];
        for (int idx = tid; idx < BN*BK_TILE; idx += blockDim.x)
            sB[idx] = qw[(bcol + idx/BK_TILE)*K + k0 + idx%BK_TILE];
        int sub0 = k0 / 32;
        for (int idx = tid; idx < BM*KSUB; idx += blockDim.x)
            sDa[idx] = da[(brow + idx/KSUB)*nblk_k + sub0 + idx%KSUB];
        for (int idx = tid; idx < BN*KSUB; idx += blockDim.x)
            sDw[idx] = dw[(bcol + idx/KSUB)*nblk_k + sub0 + idx%KSUB];
        __syncthreads();

        for (int sub = 0; sub < KSUB; sub++) {
            int ks = sub*32;
            unsigned af[2][4], bf[2][2];
            for (int mi=0; mi<2; mi++) {
                int rg = wr*32 + mi*16 + gid;
                unsigned a0=0,a1=0,a2=0,a3=0;
                for (int t=0;t<4;t++){
                    a0 |= ((unsigned)(unsigned char)sA[rg*BK_TILE     + ks + tig*4 + t])      << (8*t);
                    a1 |= ((unsigned)(unsigned char)sA[(rg+8)*BK_TILE + ks + tig*4 + t])      << (8*t);
                    a2 |= ((unsigned)(unsigned char)sA[rg*BK_TILE     + ks + 16 + tig*4 + t]) << (8*t);
                    a3 |= ((unsigned)(unsigned char)sA[(rg+8)*BK_TILE + ks + 16 + tig*4 + t]) << (8*t);
                }
                af[mi][0]=a0; af[mi][1]=a1; af[mi][2]=a2; af[mi][3]=a3;
            }
            for (int ni=0; ni<2; ni++) {
                int cg = wc*16 + ni*8 + gid;
                unsigned b0=0,b1=0;
                for (int t=0;t<4;t++){
                    b0 |= ((unsigned)(unsigned char)sB[cg*BK_TILE + ks + tig*4 + t])      << (8*t);
                    b1 |= ((unsigned)(unsigned char)sB[cg*BK_TILE + ks + 16 + tig*4 + t]) << (8*t);
                }
                bf[ni][0]=b0; bf[ni][1]=b1;
            }
            for (int mi=0; mi<2; mi++) {
                float da0 = sDa[(wr*32+mi*16+gid)*KSUB + sub];
                float da8 = sDa[(wr*32+mi*16+gid+8)*KSUB + sub];
                for (int ni=0; ni<2; ni++) {
                    int d0,d1,d2,d3;
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                        : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3)
                        : "r"(af[mi][0]),"r"(af[mi][1]),"r"(af[mi][2]),"r"(af[mi][3]),
                          "r"(bf[ni][0]),"r"(bf[ni][1]),"r"(z),"r"(z),"r"(z),"r"(z));
                    float dw0 = sDw[(wc*16+ni*8+tig*2)*KSUB + sub];
                    float dw1 = sDw[(wc*16+ni*8+tig*2+1)*KSUB + sub];
                    acc[mi][ni][0] += (float)d0*da0*dw0;
                    acc[mi][ni][1] += (float)d1*da0*dw1;
                    acc[mi][ni][2] += (float)d2*da8*dw0;
                    acc[mi][ni][3] += (float)d3*da8*dw1;
                }
            }
        }
        __syncthreads();
    }
    for (int mi=0; mi<2; mi++) for (int ni=0; ni<2; ni++) {
        int orow = brow + wr*32 + mi*16 + gid;
        int ocol = bcol + wc*16 + ni*8 + tig*2;
        C[orow*N + ocol]         = acc[mi][ni][0];
        C[orow*N + ocol + 1]     = acc[mi][ni][1];
        C[(orow+8)*N + ocol]     = acc[mi][ni][2];
        C[(orow+8)*N + ocol + 1] = acc[mi][ni][3];
    }
}
"#;

    /// Correctness of the v2 (register-blocked) Q8_0 MMQ vs the host Q8_0 oracle.
    #[test]
    #[ignore = "L7 Stage B; run with --features cuda -- --ignored --nocapture"]
    fn mmq_stage_b_q8_0_v2() {
        let Some(b) = cuda() else {
            eprintln!("no CUDA device");
            return;
        };
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some("compute_120"),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(MMQ_Q8_0_V2_SRC, opts).expect("compile");
        let st = &b.stream;
        let module = b.ctx.load_module(ptx).expect("module");
        let f = module.load_function("mmq_q8_0_v2").expect("fn");

        let (mm, nn, kk) = (128usize, 128usize, 256usize); // 2x2 blocks, 2 K-tiles
        let nblk = kk / 32;
        let af: Vec<f32> = (0..mm * kk).map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.21).collect();
        let wf: Vec<f32> = (0..nn * kk).map(|i| ((i * 5 % 19) as f32 - 9.0) * 0.17).collect();
        let (qa, da) = quant_rows_q8_0(&af, mm, kk);
        let (qw, dw) = quant_rows_q8_0(&wf, nn, kk);
        let qad = st.clone_htod(&qa).unwrap();
        let dad = st.clone_htod(&da).unwrap();
        let qwd = st.clone_htod(&qw).unwrap();
        let dwd = st.clone_htod(&dw).unwrap();
        let mut cd = st.alloc_zeros::<f32>(mm * nn).unwrap();
        {
            let mut lb = st.launch_builder(&f);
            let (mi, ni, ki) = (mm as i32, nn as i32, kk as i32);
            lb.arg(&qad).arg(&dad).arg(&qwd).arg(&dwd).arg(&mut cd).arg(&mi).arg(&ni).arg(&ki);
            let cfg = LaunchConfig {
                grid_dim: ((nn / 64) as u32, (mm / 64) as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg) }.expect("launch");
        }
        let c: Vec<f32> = st.clone_dtoh(&cd).unwrap();
        let mut bad = 0;
        let mut maxrel = 0f32;
        for i in 0..mm {
            for j in 0..nn {
                let mut q8 = 0f32;
                for blk in 0..nblk {
                    let mut idot = 0i32;
                    for t in 0..32 {
                        let k = blk * 32 + t;
                        idot += qa[i * kk + k] as i32 * qw[j * kk + k] as i32;
                    }
                    q8 += idot as f32 * da[i * nblk + blk] * dw[j * nblk + blk];
                }
                let rel = (c[i * nn + j] - q8).abs() / q8.abs().max(1e-3);
                maxrel = maxrel.max(rel);
                if rel > 1e-4 {
                    bad += 1;
                }
            }
        }
        assert_eq!(bad, 0, "v2 Q8_0 MMQ disagrees with host oracle in {bad} cells");
        eprintln!("STAGE B (v2) PASSED: {mm}x{nn}x{kk} matches host Q8_0 (max rel {maxrel:.2e})");
    }

    /// Phase 7 / Stage B GO/NO-GO microbench: time the tiled Q8_0 MMQ against the
    /// current `gemm_dev_f16` (cuBLASLt f16, the path it would replace) at a pp512
    /// prefill matmul shape. The decision number for whether int8 MMQ justifies
    /// the Q4_K kernel (Stage C). Also cross-checks MMQ vs the f16 GEMM output.
    #[test]
    #[ignore = "L7 Stage B bench; run with --features cuda -- --ignored --nocapture"]
    fn mmq_stage_b_bench_vs_f16() {
        use std::time::Instant;
        let Some(b) = cuda() else {
            eprintln!("no CUDA device");
            return;
        };
        let (rows, oc, ic) = (512usize, 2048usize, 2048usize); // one pp512 matmul
        let nblk = ic / 32;
        let st = &b.stream;

        // Deterministic f32 weight (oc×ic) and activation (rows×ic).
        let wf: Vec<f32> =
            (0..oc * ic).map(|i| ((i.wrapping_mul(2654435761) >> 8) % 255) as f32 * 0.01 - 1.27).collect();
        let af: Vec<f32> =
            (0..rows * ic).map(|i| ((i.wrapping_mul(40503) >> 7) % 255) as f32 * 0.01 - 1.27).collect();

        // --- MMQ inputs (separated int8 + per-block scales) ---
        let (qa, da) = quant_rows_q8_0(&af, rows, ic);
        let (qw, dw) = quant_rows_q8_0(&wf, oc, ic);
        let qad = st.clone_htod(&qa).unwrap();
        let dad = st.clone_htod(&da).unwrap();
        let qwd = st.clone_htod(&qw).unwrap();
        let dwd = st.clone_htod(&dw).unwrap();
        let mut cd = st.alloc_zeros::<f32>(rows * oc).unwrap();
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some("compute_120"),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(MMQ_Q8_0_TILED_SRC, opts).expect("compile");
        let module = b.ctx.load_module(ptx).expect("module");
        let f = module.load_function("mmq_q8_0_tiled").expect("fn");
        let cfg = LaunchConfig {
            grid_dim: ((oc / 32) as u32, (rows / 32) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (mi, ni, ki) = (rows as i32, oc as i32, ic as i32);

        // v2 (register-blocked, 64x64 tile, 128-K stage) — the tuning pass.
        let opts2 = cudarc::nvrtc::CompileOptions {
            arch: Some("compute_120"),
            ..Default::default()
        };
        let ptx2 = cudarc::nvrtc::compile_ptx_with_opts(MMQ_Q8_0_V2_SRC, opts2).expect("compile v2");
        let module2 = b.ctx.load_module(ptx2).expect("module2");
        let f2 = module2.load_function("mmq_q8_0_v2").expect("fn2");
        let cfg2 = LaunchConfig {
            grid_dim: ((oc / 64) as u32, (rows / 64) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut cd2 = st.alloc_zeros::<f32>(rows * oc).unwrap();

        // --- f16 baseline: pack a Q8_0 QMatrix + an f16 activation for gemm_dev_f16 ---
        let mut packed = vec![0u8; oc * nblk * 34];
        for r in 0..oc {
            for blk in 0..nblk {
                let off = (r * nblk + blk) * 34;
                let d = half::f16::from_f32(dw[r * nblk + blk]);
                packed[off..off + 2].copy_from_slice(&d.to_le_bytes());
                for t in 0..32 {
                    packed[off + 2 + t] = qw[r * ic + blk * 32 + t] as u8;
                }
            }
        }
        let qmat =
            QMatrix::quant(GgmlType::Q8_0, std::borrow::Cow::Owned(packed), oc, ic).expect("qmat");
        let x16h: Vec<half::f16> = af.iter().map(|&x| half::f16::from_f32(x)).collect();
        let x16 = st.clone_htod(&x16h).unwrap();
        let mut cf = st.alloc_zeros::<f32>(rows * oc).unwrap();

        // Correctness cross-check (one launch each) before timing.
        {
            let mut lb = st.launch_builder(&f);
            lb.arg(&qad).arg(&dad).arg(&qwd).arg(&dwd).arg(&mut cd).arg(&mi).arg(&ni).arg(&ki);
            unsafe { lb.launch(cfg) }.expect("launch");
        }
        b.gemm_dev_f16(&mut cf, &x16, &qmat, rows);
        st.synchronize().unwrap();
        let (cm, cb): (Vec<f32>, Vec<f32>) =
            (st.clone_dtoh(&cd).unwrap(), st.clone_dtoh(&cf).unwrap());
        let (mut e2, mut r2) = (0f64, 0f64);
        for (m, f) in cm.iter().zip(&cb) {
            e2 += (*m - *f) as f64 * (*m - *f) as f64;
            r2 += *f as f64 * *f as f64;
        }
        let rel = (e2 / r2).sqrt();
        assert!(rel < 0.05, "MMQ vs f16 GEMM diverge: rel L2 {rel}");

        // Timing: warmup then `reps` launches, one synchronize at the end.
        let reps = 50;
        for _ in 0..5 {
            let mut lb = st.launch_builder(&f);
            lb.arg(&qad).arg(&dad).arg(&qwd).arg(&dwd).arg(&mut cd).arg(&mi).arg(&ni).arg(&ki);
            unsafe { lb.launch(cfg) }.expect("launch");
        }
        st.synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..reps {
            let mut lb = st.launch_builder(&f);
            lb.arg(&qad).arg(&dad).arg(&qwd).arg(&dwd).arg(&mut cd).arg(&mi).arg(&ni).arg(&ki);
            unsafe { lb.launch(cfg) }.expect("launch");
        }
        st.synchronize().unwrap();
        let mmq_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;

        for _ in 0..5 {
            b.gemm_dev_f16(&mut cf, &x16, &qmat, rows);
        }
        st.synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..reps {
            b.gemm_dev_f16(&mut cf, &x16, &qmat, rows);
        }
        st.synchronize().unwrap();
        let f16_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;

        // v2: correctness cross-check vs f16, then time.
        {
            let mut lb = st.launch_builder(&f2);
            lb.arg(&qad).arg(&dad).arg(&qwd).arg(&dwd).arg(&mut cd2).arg(&mi).arg(&ni).arg(&ki);
            unsafe { lb.launch(cfg2) }.expect("launch");
        }
        st.synchronize().unwrap();
        let cm2: Vec<f32> = st.clone_dtoh(&cd2).unwrap();
        let (mut e2b, mut r2b) = (0f64, 0f64);
        for (m, f) in cm2.iter().zip(&cb) {
            e2b += (*m - *f) as f64 * (*m - *f) as f64;
            r2b += *f as f64 * *f as f64;
        }
        let rel2 = (e2b / r2b).sqrt();
        assert!(rel2 < 0.05, "v2 MMQ vs f16 diverge: rel L2 {rel2}");
        for _ in 0..5 {
            let mut lb = st.launch_builder(&f2);
            lb.arg(&qad).arg(&dad).arg(&qwd).arg(&dwd).arg(&mut cd2).arg(&mi).arg(&ni).arg(&ki);
            unsafe { lb.launch(cfg2) }.expect("launch");
        }
        st.synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..reps {
            let mut lb = st.launch_builder(&f2);
            lb.arg(&qad).arg(&dad).arg(&qwd).arg(&dwd).arg(&mut cd2).arg(&mi).arg(&ni).arg(&ki);
            unsafe { lb.launch(cfg2) }.expect("launch");
        }
        st.synchronize().unwrap();
        let v2_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;

        eprintln!(
            "STAGE B BENCH {rows}x{oc}x{ic}: f16 cuBLASLt {f16_ms:.3} ms (1.00x baseline)\n  \
             naive-MMQ {mmq_ms:.3} ms => {:.2}x vs f16 (rel L2 {rel:.4})\n  \
             tuned-MMQ {v2_ms:.3} ms => {:.2}x vs f16 (rel L2 {rel2:.4}); \
             tuning lifted MMQ {:.2}x ({mmq_ms:.3} -> {v2_ms:.3} ms)",
            f16_ms / mmq_ms,
            f16_ms / v2_ms,
            mmq_ms / v2_ms,
        );
    }

    #[test]
    #[ignore = "L7 spike; run with --features cuda -- --ignored --nocapture"]
    fn bench_int8_gemm_feasibility() {
        use cudarc::cublaslt::{result, sys, MatmulShared};
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use std::ffi::c_void;
        use std::time::Instant;
        let Some(b) = cuda() else { return };
        let st = &b.stream;
        let iters = 100;
        // Prefill GEMM shape: D[oc x rows] = W[oc x ic] · xᵀ (mirrors gemm_dev_f16).
        let (oc, ic, rows) = (2048usize, 2048usize, 512usize);
        let (m, n, k) = (oc as u64, rows as u64, ic as u64);

        let a8 = st.alloc_zeros::<i8>(oc * ic).unwrap();
        let b8 = st.alloc_zeros::<i8>(ic * rows).unwrap();
        let mut c32 = st.alloc_zeros::<i32>(oc * rows).unwrap();
        let ws_size = 33_554_432usize;
        let ws = unsafe { st.alloc::<u8>(ws_size) }.unwrap();
        let handle = *MatmulShared::handle(&b.blas);

        let desc = match result::create_matmul_desc(
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32I,
            sys::cudaDataType_t::CUDA_R_32I,
        ) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("INT8 SPIKE: create_matmul_desc(32I) failed: {e:?} -> KILL");
                return;
            }
        };
        let (op_t, op_n): (i32, i32) = (1, 0); // CUBLAS_OP_T, CUBLAS_OP_N
        unsafe {
            result::set_matmul_desc_attribute(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                (&op_t) as *const _ as *const c_void,
                4,
            )
            .unwrap();
            result::set_matmul_desc_attribute(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                (&op_n) as *const _ as *const c_void,
                4,
            )
            .unwrap();
        }
        let a_l = result::create_matrix_layout(sys::cudaDataType_t::CUDA_R_8I, k, m, ic as i64).unwrap();
        let b_l = result::create_matrix_layout(sys::cudaDataType_t::CUDA_R_8I, k, n, ic as i64).unwrap();
        let c_l = result::create_matrix_layout(sys::cudaDataType_t::CUDA_R_32I, m, n, oc as i64).unwrap();
        let pref = result::create_matmul_pref().unwrap();
        unsafe {
            result::set_matmul_pref_attribute(
                pref,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&ws_size) as *const _ as *const c_void,
                std::mem::size_of::<usize>(),
            )
            .unwrap();
        }
        let heur = match unsafe {
            result::get_matmul_algo_heuristic(handle, desc, a_l, b_l, c_l, c_l, pref)
        } {
            Ok(h) => h,
            Err(e) => {
                eprintln!("INT8 SPIKE: no int8 algo for ordinary col-major layout: {e:?} -> needs COL32 transforms -> KILL");
                return;
            }
        };
        eprintln!("INT8 SPIKE: ordinary-layout int8 algo found (heuristic ok)");

        let (alpha, beta): (i32, i32) = (1, 0);
        macro_rules! i8gemm {
            () => {unsafe {
                let (pa, _a) = a8.device_ptr(st);
                let (pb, _b) = b8.device_ptr(st);
                let (pc, _c) = c32.device_ptr_mut(st);
                let (pw, _w) = ws.device_ptr(st);
                result::matmul(handle, desc,
                    (&alpha) as *const _ as *const c_void, (&beta) as *const _ as *const c_void,
                    pa as *const c_void, a_l, pb as *const c_void, b_l,
                    pc as *const c_void, c_l, pc as *mut c_void, c_l,
                    (&heur.algo) as *const _, pw as *mut c_void, ws_size, st.cu_stream() as *mut _)
            }};
        }
        if let Err(e) = i8gemm!() {
            eprintln!("INT8 SPIKE: matmul failed (scale/layout): {e:?} -> KILL");
            return;
        }
        st.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            i8gemm!().unwrap();
        }
        st.synchronize().unwrap();
        let i8t = t0.elapsed();

        // f16 reference at the same shape via the safe API.
        let a16 = st.alloc_zeros::<f16>(oc * ic).unwrap();
        let b16 = st.alloc_zeros::<f16>(ic * rows).unwrap();
        let mut c16 = st.alloc_zeros::<f16>(oc * rows).unwrap();
        let cfg = MatmulConfig {
            transa: true, transb: false, transc: false, m, n, k, alpha: 1.0,
            lda: ic as i64, ldb: ic as i64, beta: 0.0, ldc: oc as i64,
            stride_a: None, stride_b: None, stride_c: None, stride_bias: None, batch_size: None,
        };
        unsafe { b.blas.matmul(cfg, &a16, &b16, &mut c16, None, None).unwrap() };
        st.synchronize().unwrap();
        let t1 = Instant::now();
        for _ in 0..iters {
            unsafe { b.blas.matmul(cfg, &a16, &b16, &mut c16, None, None).unwrap() };
        }
        st.synchronize().unwrap();
        let f16t = t1.elapsed();

        eprintln!(
            "INT8 SPIKE {oc}x{ic}x{rows} x{iters}: int8 {i8t:?}  f16 {f16t:?}  int8 {:.2}x f16",
            f16t.as_secs_f64() / i8t.as_secs_f64()
        );
    }

}
