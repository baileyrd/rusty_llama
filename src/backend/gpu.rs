//! GPU backend (wgpu).
//!
//! A second [`Backend`] implementation that runs the transformer primitives as
//! compute shaders via [`wgpu`] (Vulkan / DX12 / Metal). It is selected at run
//! time (`--backend gpu`) and gated behind the `gpu` cargo feature so the
//! default build keeps its tiny dependency set.
//!
//! ## What stays resident
//!
//! The trait hands each op plain host slices plus a borrowed [`QMatrix`], so a
//! naive backend would re-upload every weight on every token — the matmul
//! weights dwarf everything else, so that alone would make the GPU path
//! pointlessly slow. Instead, matmul weights are uploaded **once** and cached,
//! keyed on the weight's data pointer (stable across tokens, since the weights
//! are borrowed from the mmap'd file / owned by the [`Model`](crate::model)).
//! Quantized weights are dequantized to f32 on the host at upload time, so a
//! single f32 matmul shader covers every `GgmlType`. The small RoPE inverse-
//! frequency table is cached the same way.
//!
//! Activations (the residual stream, the KV-cache slices, attention scratch)
//! are small and re-passed by the trait every call, so they are uploaded per
//! call and the result read straight back. That host↔device round-trip per op
//! is the honest cost of fitting a per-op GPU backend behind a trait designed
//! for the CPU; see the `bench_*` notes in `cpu.rs` and the PR.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use wgpu::util::DeviceExt;

use crate::backend::Backend;
use crate::quant::{dequantize, GgmlType};
use crate::tensor::QMatrix;

// --- Shaders ----------------------------------------------------------------
//
// One module per op (each declares its own `@group(0)` bindings, so they can't
// be merged into a single module without binding-number clashes). Parameters
// ride in a small read-only storage buffer to dodge uniform-buffer's 16-byte
// size rules; f32 scalars are passed through `f32::to_bits`.

const WGSL_MATMUL: &str = r#"
struct P { rows: u32, cols: u32 };
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<storage, read> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.rows) { return; }
    var acc = 0.0;
    let base = row * p.cols;
    for (var j = 0u; j < p.cols; j = j + 1u) {
        acc = acc + w[base + j] * x[j];
    }
    outp[row] = acc;
}
"#;

const WGSL_RMSNORM: &str = r#"
struct P { n: u32, eps: f32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<storage, read> p: P;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var ss = 0.0;
    for (var i = tid; i < p.n; i = i + 256u) { let v = x[i]; ss = ss + v * v; }
    red[tid] = ss;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(red[0] / f32(p.n) + p.eps);
    for (var i = tid; i < p.n; i = i + 256u) { outp[i] = x[i] * inv * weight[i]; }
}
"#;

const WGSL_ADD: &str = r#"
struct P { n: u32 };
@group(0) @binding(0) var<storage, read_write> outp: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    outp[i] = outp[i] + x[i];
}
"#;

const WGSL_SWIGLU: &str = r#"
struct P { n: u32 };
@group(0) @binding(0) var<storage, read_write> hb: array<f32>;
@group(0) @binding(1) var<storage, read> hb2: array<f32>;
@group(0) @binding(2) var<storage, read> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let v = hb[i];
    let silu = v * (1.0 / (1.0 + exp(-v)));
    hb[i] = silu * hb2[i];
}
"#;

const WGSL_ROPE: &str = r#"
struct P { head_size: u32, kv_dim: u32, pos: u32, n_freqs: u32, mscale: f32, dim: u32 };
@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read> inv_freq: array<f32>;
@group(0) @binding(3) var<storage, read> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x * 2u;
    if (i >= p.dim) { return; }
    let ph = (i % p.head_size) / 2u;   // pair index within the head
    if (ph >= p.n_freqs) { return; }   // partial rotary: leave the tail alone
    let angle = f32(p.pos) * inv_freq[ph];
    let fcr = cos(angle) * p.mscale;
    let fci = sin(angle) * p.mscale;
    let q0 = q[i]; let q1 = q[i + 1u];
    q[i] = q0 * fcr - q1 * fci;
    q[i + 1u] = q0 * fci + q1 * fcr;
    if (i < p.kv_dim) {
        let k0 = k[i]; let k1 = k[i + 1u];
        k[i] = k0 * fcr - k1 * fci;
        k[i + 1u] = k0 * fci + k1 * fcr;
    }
}
"#;

// One workgroup per head; the workgroup cooperatively scores against every
// cached key, softmaxes (max-subtracted, matching `math::softmax`), then writes
// the value-weighted sum. `att` (the trait's scratch) holds the per-head scores.
const WGSL_ATTENTION: &str = r#"
struct P { head_size: u32, kv_dim: u32, seq_len: u32, kv_mul: u32, np: u32, scale: f32 };
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> keyc: array<f32>;
@group(0) @binding(2) var<storage, read> valc: array<f32>;
@group(0) @binding(3) var<storage, read_write> att: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
@group(0) @binding(5) var<storage, read> p: P;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let tid = lid.x;
    let hs = p.head_size;
    let kv_off = (h / p.kv_mul) * hs;
    let q_base = h * hs;
    let att_base = h * p.seq_len;

    // 1. scaled dot-product scores against every key so far.
    for (var t = tid; t < p.np; t = t + 64u) {
        var s = 0.0;
        let kb = t * p.kv_dim + kv_off;
        for (var i = 0u; i < hs; i = i + 1u) { s = s + q[q_base + i] * keyc[kb + i]; }
        att[att_base + t] = s * p.scale;
    }
    workgroupBarrier();

    // 2. max-reduce for numerical stability.
    var m = -3.4028235e38;
    for (var t = tid; t < p.np; t = t + 64u) { m = max(m, att[att_base + t]); }
    red[tid] = m;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) { red[tid] = max(red[tid], red[tid + stride]); }
        workgroupBarrier();
    }
    let maxv = red[0];
    workgroupBarrier();

    // 3. exp(score - max), summed.
    var ls = 0.0;
    for (var t = tid; t < p.np; t = t + 64u) {
        let e = exp(att[att_base + t] - maxv);
        att[att_base + t] = e;
        ls = ls + e;
    }
    red[tid] = ls;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
        workgroupBarrier();
    }
    let sum = red[0];
    // `att` is in the storage address space; workgroupBarrier() orders only
    // workgroup memory, so a storageBarrier() is required to publish step 3's
    // per-lane writes to the other lanes that read the whole row in step 4
    // (per the WGSL memory model — unlike CUDA's __syncthreads()).
    storageBarrier();

    // 4. value-weighted sum (normalized by the softmax denominator).
    for (var d = tid; d < hs; d = d + 64u) {
        var acc = 0.0;
        for (var t = 0u; t < p.np; t = t + 1u) {
            acc = acc + att[att_base + t] * valc[t * p.kv_dim + kv_off + d];
        }
        outp[q_base + d] = acc / sum;
    }
}
"#;

// --- Batched (prefill) shaders ----------------------------------------------
//
// These process `n` token positions at once so the whole prompt runs as a few
// large dispatches instead of n× the single-token ops. add/swiglu are
// elementwise, so the single-token kernels above already handle a flattened
// batch; only matmul/rmsnorm/rope/attention need batched variants.

// 2-D dispatch: x = output feature, y = batch row. out is (n, rows) row-major.
const WGSL_MATMUL_BATCH: &str = r#"
struct P { rows: u32, cols: u32, n: u32 };
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<storage, read> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let r = gid.y;
    if (row >= p.rows || r >= p.n) { return; }
    var acc = 0.0;
    let wbase = row * p.cols;
    let xbase = r * p.cols;
    for (var j = 0u; j < p.cols; j = j + 1u) {
        acc = acc + w[wbase + j] * x[xbase + j];
    }
    outp[r * p.rows + row] = acc;
}
"#;

// One workgroup per batch row; reduce the row's sum-of-squares over 256 lanes.
const WGSL_RMSNORM_BATCH: &str = r#"
struct P { dim: u32, eps: f32, n: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<storage, read> p: P;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let r = wid.x;
    let tid = lid.x;
    let base = r * p.dim;
    var ss = 0.0;
    for (var i = tid; i < p.dim; i = i + 256u) { let v = x[base + i]; ss = ss + v * v; }
    red[tid] = ss;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(red[0] / f32(p.dim) + p.eps);
    for (var i = tid; i < p.dim; i = i + 256u) { outp[base + i] = x[base + i] * inv * weight[i]; }
}
"#;

// 2-D dispatch: x = pair within the head, y = batch row. Row r is at absolute
// position pos_base + r.
const WGSL_ROPE_BATCH: &str = r#"
struct P { head_size: u32, kv_dim: u32, pos_base: u32, n_freqs: u32, mscale: f32, q_dim: u32, n: u32 };
@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read> inv_freq: array<f32>;
@group(0) @binding(3) var<storage, read> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.y;
    if (r >= p.n) { return; }
    let i = gid.x * 2u;
    if (i >= p.q_dim) { return; }
    let ph = (i % p.head_size) / 2u;
    if (ph >= p.n_freqs) { return; }
    let angle = f32(p.pos_base + r) * inv_freq[ph];
    let fcr = cos(angle) * p.mscale;
    let fci = sin(angle) * p.mscale;
    let qb = r * p.q_dim;
    let q0 = q[qb + i]; let q1 = q[qb + i + 1u];
    q[qb + i] = q0 * fcr - q1 * fci;
    q[qb + i + 1u] = q0 * fci + q1 * fcr;
    if (i < p.kv_dim) {
        let kb = r * p.kv_dim;
        let k0 = k[kb + i]; let k1 = k[kb + i + 1u];
        k[kb + i] = k0 * fcr - k1 * fci;
        k[kb + i + 1u] = k0 * fci + k1 * fcr;
    }
}
"#;

// One thread per (batch row, head). Causal: row r attends to keys 0..=pos_base+r.
// Uses an online (running-max) softmax so no score buffer is needed — the only
// limit is head_size <= 128 (the per-thread accumulator). Larger heads fall back
// to the looped default on the host side.
const WGSL_ATTENTION_BATCH: &str = r#"
struct P { head_size: u32, kv_dim: u32, kv_mul: u32, pos_base: u32, n: u32, n_heads: u32, scale: f32 };
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> keyc: array<f32>;
@group(0) @binding(2) var<storage, read> valc: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@group(0) @binding(4) var<storage, read> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.n * p.n_heads) { return; }
    let r = idx / p.n_heads;
    let h = idx % p.n_heads;
    let hs = p.head_size;
    let dim = p.n_heads * hs;
    let pos = p.pos_base + r;
    let kv_off = (h / p.kv_mul) * hs;
    let qh = r * dim + h * hs;
    var m = -3.4028235e38;
    var sumexp = 0.0;
    var acc: array<f32, 128>;
    for (var d = 0u; d < hs; d = d + 1u) { acc[d] = 0.0; }
    for (var t = 0u; t <= pos; t = t + 1u) {
        let kb = t * p.kv_dim + kv_off;
        var s = 0.0;
        for (var i = 0u; i < hs; i = i + 1u) { s = s + q[qh + i] * keyc[kb + i]; }
        s = s * p.scale;
        let new_m = max(m, s);
        let corr = exp(m - new_m);
        let f = exp(s - new_m);
        sumexp = sumexp * corr + f;
        for (var d = 0u; d < hs; d = d + 1u) { acc[d] = acc[d] * corr + f * valc[kb + d]; }
        m = new_m;
    }
    let inv = 1.0 / sumexp;
    for (var d = 0u; d < hs; d = d + 1u) { outp[qh + d] = acc[d] * inv; }
}
"#;

/// Largest head_size the batched attention kernel's per-thread accumulator
/// (`array<f32, 128>`) supports; larger heads use the looped fallback.
const MAX_BATCH_HEAD: usize = 128;

// --- In-shader dequantizing matmul ------------------------------------------
//
// Quantized weights are uploaded as their raw GGUF block bytes (kept resident)
// and dequantized inside the matmul, exactly like the CPU's per-block path.
// This streams ~4x less weight data per token than expanding to f32 on upload.
// Common WGSL prelude: read a byte out of an `array<u32>` and an f16 scale.

const WGSL_QUANT_PRELUDE: &str = r#"
fn gb(i: u32) -> u32 { return (wb[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu; }
fn f16f32(h: u32) -> f32 {
    let s = (h >> 15u) & 1u;
    let e = (h >> 10u) & 0x1fu;
    let m = h & 0x3ffu;
    var v: f32;
    if (e == 0u) { v = f32(m) * exp2(-24.0); }
    else if (e == 0x1fu) { v = 0.0; }   // weight scales are never inf/nan
    else { v = (1.0 + f32(m) / 1024.0) * exp2(f32(i32(e) - 15)); }
    return select(v, -v, s == 1u);
}
"#;

// Q8_0: 34-byte blocks of (f16 scale, 32 x i8). out[row] = sum_b sum_i d*q*x.
const WGSL_MATMUL_Q8_0: &str = r#"
struct P { rows: u32, cols: u32 };
@group(0) @binding(0) var<storage, read> wb: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<storage, read> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.rows) { return; }
    let nb = p.cols / 32u;
    let row_base = row * nb * 34u;
    var acc = 0.0;
    for (var b = 0u; b < nb; b = b + 1u) {
        let bb = row_base + b * 34u;
        let d = f16f32(gb(bb) | (gb(bb + 1u) << 8u));
        let xb = b * 32u;
        for (var i = 0u; i < 32u; i = i + 1u) {
            let q = (i32(gb(bb + 2u + i)) << 24u) >> 24u;
            acc = acc + d * f32(q) * x[xb + i];
        }
    }
    outp[row] = acc;
}
"#;

// Batched Q8_0: 2-D grid, x = output feature, y = batch row. out is (n, rows).
const WGSL_MATMUL_Q8_0_BATCH: &str = r#"
struct P { rows: u32, cols: u32, n: u32 };
@group(0) @binding(0) var<storage, read> wb: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<storage, read> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let r = gid.y;
    if (row >= p.rows || r >= p.n) { return; }
    let nb = p.cols / 32u;
    let row_base = row * nb * 34u;
    let xrow = r * p.cols;
    var acc = 0.0;
    for (var b = 0u; b < nb; b = b + 1u) {
        let bb = row_base + b * 34u;
        let d = f16f32(gb(bb) | (gb(bb + 1u) << 8u));
        let xb = xrow + b * 32u;
        for (var i = 0u; i < 32u; i = i + 1u) {
            let q = (i32(gb(bb + 2u + i)) << 24u) >> 24u;
            acc = acc + d * f32(q) * x[xb + i];
        }
    }
    outp[r * p.rows + row] = acc;
}
"#;

/// The compiled compute pipelines, one per primitive.
struct Pipelines {
    matmul: wgpu::ComputePipeline,
    rmsnorm: wgpu::ComputePipeline,
    add: wgpu::ComputePipeline,
    swiglu: wgpu::ComputePipeline,
    rope: wgpu::ComputePipeline,
    attention: wgpu::ComputePipeline,
    matmul_batch: wgpu::ComputePipeline,
    rmsnorm_batch: wgpu::ComputePipeline,
    rope_batch: wgpu::ComputePipeline,
    attention_batch: wgpu::ComputePipeline,
    matmul_q8_0: wgpu::ComputePipeline,
    matmul_q8_0_batch: wgpu::ComputePipeline,
}

/// A resident weight: its device buffer and the format it's stored in (`F32`
/// for full-precision or host-dequantized weights, or a quant type that the
/// matmul kernel dequantizes in-shader).
#[derive(Clone)]
struct GpuWeight {
    buf: Arc<wgpu::Buffer>,
    ty: GgmlType,
}

/// Pre-built bind groups for one transformer layer's fused decode step. The
/// bind groups reference fixed resident buffers, so they're built once and
/// reused every step; only the *contents* of the per-step params buffers (pos)
/// and the embedding change between steps.
struct LayerBinds {
    rms_att: wgpu::BindGroup,
    mq: wgpu::BindGroup,
    mk: wgpu::BindGroup,
    mv: wgpu::BindGroup,
    rope: wgpu::BindGroup,
    attn: wgpu::BindGroup,
    mo: wgpu::BindGroup,
    add_attn: wgpu::BindGroup,
    rms_ffn: wgpu::BindGroup,
    m1: wgpu::BindGroup,
    m3: wgpu::BindGroup,
    swiglu: wgpu::BindGroup,
    m2: wgpu::BindGroup,
    add_ffn: wgpu::BindGroup,
    /// On-device format of each matmul weight, to pick the matmul pipeline
    /// (order: wq, wk, wv, wo, w1, w3, w2).
    mm_ty: [GgmlType; 7],
}

/// Device-resident state for fused single-token decode: the KV cache, the
/// activation scratch, and every bind group, all kept on the GPU across steps
/// so a step is one submission with a single logits read-back.
struct DecodeState {
    dim: usize,
    kv_dim: usize,
    hidden: usize,
    vocab: usize,
    n_heads: usize,
    n_layers: usize,
    seq_len: usize,
    // Resident buffers the step touches directly. The purely-intermediate
    // activations (xb, q, hb, …) and the constant params buffers are held alive
    // by the bind groups that reference them, so they aren't stored here.
    x: wgpu::Buffer,                 // embedding written here each step
    k_tmp: wgpu::Buffer,             // fresh K, copied into the resident cache
    v_tmp: wgpu::Buffer,             // fresh V, copied into the resident cache
    logits: wgpu::Buffer,            // classifier output, copied to staging
    logits_staging: wgpu::Buffer,    // mapped to read logits back
    // Resident KV cache, one buffer per layer (seq_len * kv_dim each).
    key: Vec<wgpu::Buffer>,
    value: Vec<wgpu::Buffer>,
    // Per-step params, rewritten each step via write_buffer.
    rope_params: wgpu::Buffer,
    attn_params: wgpu::Buffer,
    layers: Vec<LayerBinds>,
    final_rms: wgpu::BindGroup,
    final_cls: wgpu::BindGroup,
    final_cls_ty: GgmlType,
    /// Number of KV positions already resident on the device.
    kv_filled: usize,
    /// Scratch for the per-step token embedding upload.
    embed: Vec<f32>,
}

/// Record one compute dispatch as its own pass (so wgpu inserts the
/// read-after-write barrier against the previous pass on shared buffers).
fn pass(
    enc: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind: &wgpu::BindGroup,
    workgroups: u32,
) {
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    cp.set_pipeline(pipeline);
    cp.set_bind_group(0, bind, &[]);
    cp.dispatch_workgroups(workgroups, 1, 1);
}

/// A wgpu-backed [`Backend`].
pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Pipelines,
    /// Resident matmul weights (f32), keyed by source data pointer.
    ///
    /// The pointer is a stable, unique key only while the source weights stay
    /// alive — which holds for a backend used with a single [`Model`] (the
    /// CLI's pattern). Reusing one backend across *different* models could in
    /// principle alias a freed-then-recycled address, so don't: make one
    /// backend per model.
    weights: Mutex<HashMap<usize, GpuWeight>>,
    /// Resident RoPE inverse-frequency tables, keyed by source data pointer.
    tables: Mutex<HashMap<usize, Arc<wgpu::Buffer>>>,
    /// The granted device limits (used to pre-check oversized weight buffers).
    limits: wgpu::Limits,
    /// Resident decode state (KV cache + activations + bind groups), built lazily
    /// on the first [`Backend::forward_step`] and reused across decode steps.
    decode: Mutex<Option<DecodeState>>,
    /// Human-readable adapter name, for logging.
    adapter_name: String,
}

/// Storage buffers the `attention` shader binds in one group (q, key, value,
/// att, out, params) — the most of any kernel here.
const ATTENTION_STORAGE_BUFFERS: u32 = 6;

impl GpuBackend {
    /// Initialize wgpu and compile the shaders. Prefers a high-performance
    /// (discrete) adapter. Returns an error string if no adapter is available
    /// (e.g. headless CI with no GPU) so callers can fall back gracefully.
    pub fn new() -> Result<Self, String> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|e| format!("no suitable GPU adapter: {e}"))?;

        let adapter_name = adapter.get_info().name;

        // Ask for the adapter's full limits: the largest weight matrices (e.g.
        // a 32000×2048 embedding/classifier dequantized to f32 ≈ 262 MB) exceed
        // the conservative defaults for `max_storage_buffer_binding_size` /
        // `max_buffer_size`.
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("rusty_llama"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|e| format!("failed to create GPU device: {e}"))?;

        // The attention kernel binds 6 storage buffers in one stage; a downlevel
        // adapter capped at 4 would otherwise pass `new()` and then panic on the
        // first token. Reject up front so callers can fall back to the CPU.
        let limits = device.limits();
        if limits.max_storage_buffers_per_shader_stage < ATTENTION_STORAGE_BUFFERS {
            return Err(format!(
                "adapter '{adapter_name}' allows only {} storage buffers per shader stage; \
                 the attention kernel needs {ATTENTION_STORAGE_BUFFERS}",
                limits.max_storage_buffers_per_shader_stage
            ));
        }

        let pipelines = Pipelines {
            matmul: make_pipeline(&device, WGSL_MATMUL, "matmul"),
            rmsnorm: make_pipeline(&device, WGSL_RMSNORM, "rmsnorm"),
            add: make_pipeline(&device, WGSL_ADD, "add"),
            swiglu: make_pipeline(&device, WGSL_SWIGLU, "swiglu"),
            rope: make_pipeline(&device, WGSL_ROPE, "rope"),
            attention: make_pipeline(&device, WGSL_ATTENTION, "attention"),
            matmul_batch: make_pipeline(&device, WGSL_MATMUL_BATCH, "matmul_batch"),
            rmsnorm_batch: make_pipeline(&device, WGSL_RMSNORM_BATCH, "rmsnorm_batch"),
            rope_batch: make_pipeline(&device, WGSL_ROPE_BATCH, "rope_batch"),
            attention_batch: make_pipeline(&device, WGSL_ATTENTION_BATCH, "attention_batch"),
            matmul_q8_0: make_pipeline(
                &device,
                &format!("{WGSL_QUANT_PRELUDE}{WGSL_MATMUL_Q8_0}"),
                "matmul_q8_0",
            ),
            matmul_q8_0_batch: make_pipeline(
                &device,
                &format!("{WGSL_QUANT_PRELUDE}{WGSL_MATMUL_Q8_0_BATCH}"),
                "matmul_q8_0_batch",
            ),
        };

        Ok(GpuBackend {
            device,
            queue,
            pipelines,
            weights: Mutex::new(HashMap::new()),
            tables: Mutex::new(HashMap::new()),
            limits,
            decode: Mutex::new(None),
            adapter_name,
        })
    }

    /// The selected GPU adapter's name (for logging / benchmarks).
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    // --- buffer helpers -----------------------------------------------------

    /// A read-only storage buffer initialized with `data`.
    fn storage_ro(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    /// A read/write storage buffer initialized with `data` and readable back.
    fn storage_rw(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
    }

    /// An uninitialized output buffer of `len` f32s, readable back.
    fn alloc_out(&self, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// A small read-only parameter buffer from a list of 32-bit words.
    fn params(&self, words: &[u32]) -> wgpu::Buffer {
        self.storage_ro("params", u32_bytes(words))
    }

    /// Build a bind group binding `buffers` to slots `0..n` in declaration
    /// order, against the pipeline's auto-derived layout.
    fn bind(&self, pipeline: &wgpu::ComputePipeline, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        let layout = pipeline.get_bind_group_layout(0);
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        })
    }

    /// Dispatch `pipeline` over a 1-D grid, then copy `out_buf` back into `out`.
    fn run(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind: &wgpu::BindGroup,
        workgroups: u32,
        out_buf: &wgpu::Buffer,
        out: &mut [f32],
    ) {
        self.run_grid(pipeline, bind, [workgroups, 1, 1], out_buf, out);
    }

    /// Dispatch `pipeline` over `grid` workgroups, then copy `out_buf` back into
    /// `out`. Compute + copy share a single submission.
    fn run_grid(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind: &wgpu::BindGroup,
        grid: [u32; 3],
        out_buf: &wgpu::Buffer,
        out: &mut [f32],
    ) {
        let byte_len = (out.len() * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, bind, &[]);
            cpass.dispatch_workgroups(grid[0], grid[1], grid[2]);
        }
        enc.copy_buffer_to_buffer(out_buf, 0, &staging, 0, byte_len);
        self.queue.submit(Some(enc.finish()));
        self.map_read(&staging, out);
    }

    /// Dispatch `pipeline` over a 1-D grid without reading anything back (the op
    /// wrote into resident buffers that are read back separately).
    fn dispatch(&self, pipeline: &wgpu::ComputePipeline, bind: &wgpu::BindGroup, workgroups: u32) {
        self.dispatch_grid(pipeline, bind, [workgroups, 1, 1]);
    }

    /// Dispatch `pipeline` over `grid` workgroups without reading anything back.
    fn dispatch_grid(&self, pipeline: &wgpu::ComputePipeline, bind: &wgpu::BindGroup, grid: [u32; 3]) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, bind, &[]);
            cpass.dispatch_workgroups(grid[0], grid[1], grid[2]);
        }
        self.queue.submit(Some(enc.finish()));
    }

    /// Copy `src` into a fresh staging buffer and read it back into `out`.
    fn read_back(&self, src: &wgpu::Buffer, out: &mut [f32]) {
        let byte_len = (out.len() * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, byte_len);
        self.queue.submit(Some(enc.finish()));
        self.map_read(&staging, out);
    }

    /// Map a `MAP_READ` staging buffer, copy its f32s into `out`, unmap.
    /// Blocks on the device until the GPU work and mapping complete.
    fn map_read(&self, staging: &wgpu::Buffer, out: &mut [f32]) {
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll failed");
        {
            let view = slice.get_mapped_range();
            for (o, c) in out.iter_mut().zip(view.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        staging.unmap();
    }

    /// Fetch (uploading on first use) the resident buffer for weight `w`.
    ///
    /// Formats with an in-shader dequant kernel (currently Q8_0) are uploaded as
    /// their raw quantized block bytes and kept that way — the matmul
    /// dequantizes them on the fly, streaming ~4× less data per token. Every
    /// other format is dequantized to f32 on the host once, here. The returned
    /// [`GpuWeight`] carries the on-device format so the caller picks the
    /// matching matmul kernel.
    fn weight_buffer(&self, w: &QMatrix) -> GpuWeight {
        let key = match w {
            QMatrix::F32 { data, .. } => data.as_ptr() as usize,
            QMatrix::Quant { data, .. } => data.as_ptr() as usize,
        };
        if let Some(g) = self.weights.lock().unwrap().get(&key) {
            return g.clone();
        }

        // (bytes-to-upload, on-device format).
        let (bytes, ty): (Cow<[u8]>, GgmlType) = match w {
            QMatrix::F32 { data, .. } => (Cow::Borrowed(f32_bytes(data)), GgmlType::F32),
            // In-shader formats: keep the raw blocks (padded to a u32 boundary
            // so the shader can read them as array<u32>).
            QMatrix::Quant {
                ty: ty @ GgmlType::Q8_0,
                data,
                ..
            } => {
                let mut b = data.to_vec();
                b.resize(b.len().next_multiple_of(4), 0);
                (Cow::Owned(b), *ty)
            }
            // Everything else: host dequant to f32 (the original behaviour).
            QMatrix::Quant {
                ty, data, rows, cols, ..
            } => {
                let f = dequantize(*ty, data, rows * cols).expect("weight dequantization");
                (Cow::Owned(f32_bytes(&f).to_vec()), GgmlType::F32)
            }
        };

        let nbytes = bytes.len() as u64;
        let binding_cap = self.limits.max_storage_buffer_binding_size;
        if nbytes > binding_cap || nbytes > self.limits.max_buffer_size {
            panic!(
                "weight matrix ({} rows × {} cols → {nbytes} bytes on device) exceeds this GPU's \
                 limits (max_storage_buffer_binding_size {binding_cap}, max_buffer_size {}); \
                 this model is too large for the GPU backend — use --backend cpu",
                w.rows(),
                w.cols(),
                self.limits.max_buffer_size,
            );
        }
        let gw = GpuWeight {
            buf: Arc::new(self.storage_ro("weight", &bytes)),
            ty,
        };
        self.weights.lock().unwrap().insert(key, gw.clone());
        gw
    }

    /// The matmul pipeline for an on-device weight format.
    fn matmul_pipeline(&self, ty: GgmlType) -> &wgpu::ComputePipeline {
        match ty {
            GgmlType::Q8_0 => &self.pipelines.matmul_q8_0,
            _ => &self.pipelines.matmul, // F32 (incl. host-dequantized weights)
        }
    }

    /// The batched matmul pipeline for an on-device weight format.
    fn matmul_batch_pipeline(&self, ty: GgmlType) -> &wgpu::ComputePipeline {
        match ty {
            GgmlType::Q8_0 => &self.pipelines.matmul_q8_0_batch,
            _ => &self.pipelines.matmul_batch,
        }
    }

    /// Fetch (uploading on first use) the resident RoPE inverse-frequency table.
    fn table_buffer(&self, inv_freq: &[f32]) -> Arc<wgpu::Buffer> {
        let key = inv_freq.as_ptr() as usize;
        let mut cache = self.tables.lock().unwrap();
        if let Some(b) = cache.get(&key) {
            return b.clone();
        }
        let buf = Arc::new(self.storage_ro("inv_freq", f32_bytes(inv_freq)));
        cache.insert(key, buf.clone());
        buf
    }

    // --- Fused on-device decode -----------------------------------------

    /// An uninitialized device buffer of `len` f32s with the given usage.
    fn buffer(&self, label: &str, len: usize, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len * 4) as u64,
            usage,
            mapped_at_creation: false,
        })
    }

    /// A storage param buffer initialized from `words`, also writable later.
    fn params_dyn(&self, words: &[u32]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: u32_bytes(words),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Build the resident decode state for `model` (weights/KV/activations +
    /// every bind group). Called once, lazily, on the first decode step.
    fn build_decode_state(&self, model: &crate::model::Model) -> DecodeState {
        use wgpu::BufferUsages as U;
        let p = &model.config;
        let w = &model.weights;
        let (dim, kv_dim, hs, hidden, vocab, nl, seq) = (
            p.dim,
            p.kv_dim(),
            p.head_size(),
            p.hidden_dim,
            p.vocab_size,
            p.n_layers,
            p.seq_len,
        );

        // Activation + KV buffers.
        let x = self.buffer("x", dim, U::STORAGE | U::COPY_DST);
        let xb = self.buffer("xb", dim, U::STORAGE);
        let xb2 = self.buffer("xb2", dim, U::STORAGE);
        let q = self.buffer("q", dim, U::STORAGE);
        let k_tmp = self.buffer("k_tmp", kv_dim, U::STORAGE | U::COPY_SRC);
        let v_tmp = self.buffer("v_tmp", kv_dim, U::STORAGE | U::COPY_SRC);
        let att = self.buffer("att", p.n_heads * seq, U::STORAGE);
        let hb = self.buffer("hb", hidden, U::STORAGE);
        let hb2 = self.buffer("hb2", hidden, U::STORAGE);
        let logits = self.buffer("logits", vocab, U::STORAGE | U::COPY_SRC);
        let logits_staging = self.buffer("logits_staging", vocab, U::MAP_READ | U::COPY_DST);
        let key: Vec<wgpu::Buffer> = (0..nl)
            .map(|_| self.buffer("key", seq * kv_dim, U::STORAGE | U::COPY_DST))
            .collect();
        let value: Vec<wgpu::Buffer> = (0..nl)
            .map(|_| self.buffer("value", seq * kv_dim, U::STORAGE | U::COPY_DST))
            .collect();

        // Constant params (shape/eps), kept alive by the bind groups.
        let p_dimdim = self.storage_ro("p", u32_bytes(&[dim as u32, dim as u32]));
        let p_kvdim = self.storage_ro("p", u32_bytes(&[kv_dim as u32, dim as u32]));
        let p_hidden = self.storage_ro("p", u32_bytes(&[hidden as u32, dim as u32]));
        let p_dimhidden = self.storage_ro("p", u32_bytes(&[dim as u32, hidden as u32]));
        let p_vocab = self.storage_ro("p", u32_bytes(&[vocab as u32, dim as u32]));
        let p_rms = self.storage_ro("p", u32_bytes(&[dim as u32, p.rms_eps.to_bits()]));
        let p_add = self.storage_ro("p", u32_bytes(&[dim as u32]));
        let p_swiglu = self.storage_ro("p", u32_bytes(&[hidden as u32]));
        let kv_mul = (p.n_heads / p.n_kv_heads) as u32;
        let scale = 1.0f32 / (hs as f32).sqrt();
        // pos / np filled in per step.
        let rope_params = self.params_dyn(&[
            hs as u32,
            kv_dim as u32,
            0,
            model.rope.inv_freq.len() as u32,
            model.rope.mscale.to_bits(),
            dim as u32,
        ]);
        let attn_params =
            self.params_dyn(&[hs as u32, kv_dim as u32, seq as u32, kv_mul, 0, scale.to_bits()]);

        // Resident weights (cached by pointer) and per-layer norm weights. Each
        // matmul binds the weight against the pipeline matching its on-device
        // format (f32 or in-shader-dequant), recorded in `mm_ty` for dispatch.
        let inv_freq = self.table_buffer(&model.rope.inv_freq);
        let layers: Vec<LayerBinds> = (0..nl)
            .map(|l| {
                let wq = self.weight_buffer(&w.wq[l]);
                let wk = self.weight_buffer(&w.wk[l]);
                let wv = self.weight_buffer(&w.wv[l]);
                let wo = self.weight_buffer(&w.wo[l]);
                let w1 = self.weight_buffer(&w.w1[l]);
                let w2 = self.weight_buffer(&w.w2[l]);
                let w3 = self.weight_buffer(&w.w3[l]);
                let rms_a = self.table_buffer(&w.rms_att_weight[l * dim..l * dim + dim]);
                let rms_f = self.table_buffer(&w.rms_ffn_weight[l * dim..l * dim + dim]);
                let mm = |gw: &GpuWeight, out: &wgpu::Buffer, params: &wgpu::Buffer| {
                    self.bind(self.matmul_pipeline(gw.ty), &[&gw.buf, &xb, out, params])
                };
                LayerBinds {
                    rms_att: self.bind(&self.pipelines.rmsnorm, &[&x, &rms_a, &xb, &p_rms]),
                    mq: mm(&wq, &q, &p_dimdim),
                    mk: mm(&wk, &k_tmp, &p_kvdim),
                    mv: mm(&wv, &v_tmp, &p_kvdim),
                    rope: self.bind(&self.pipelines.rope, &[&q, &k_tmp, &inv_freq, &rope_params]),
                    attn: self.bind(
                        &self.pipelines.attention,
                        &[&q, &key[l], &value[l], &att, &xb, &attn_params],
                    ),
                    mo: mm(&wo, &xb2, &p_dimdim),
                    add_attn: self.bind(&self.pipelines.add, &[&x, &xb2, &p_add]),
                    rms_ffn: self.bind(&self.pipelines.rmsnorm, &[&x, &rms_f, &xb, &p_rms]),
                    m1: mm(&w1, &hb, &p_hidden),
                    m3: mm(&w3, &hb2, &p_hidden),
                    swiglu: self.bind(&self.pipelines.swiglu, &[&hb, &hb2, &p_swiglu]),
                    // w2 reads hb (not xb); bind it explicitly.
                    m2: self.bind(self.matmul_pipeline(w2.ty), &[&w2.buf, &hb, &xb, &p_dimhidden]),
                    add_ffn: self.bind(&self.pipelines.add, &[&x, &xb, &p_add]),
                    mm_ty: [wq.ty, wk.ty, wv.ty, wo.ty, w1.ty, w3.ty, w2.ty],
                }
            })
            .collect();

        let rms_final = self.table_buffer(&w.rms_final_weight);
        let wcls = self.weight_buffer(&w.wcls);
        let final_rms = self.bind(&self.pipelines.rmsnorm, &[&x, &rms_final, &xb, &p_rms]);
        let final_cls = self.bind(self.matmul_pipeline(wcls.ty), &[&wcls.buf, &xb, &logits, &p_vocab]);
        let final_cls_ty = wcls.ty;

        DecodeState {
            dim,
            kv_dim,
            hidden,
            vocab,
            n_heads: p.n_heads,
            n_layers: nl,
            seq_len: seq,
            x,
            k_tmp,
            v_tmp,
            logits,
            logits_staging,
            key,
            value,
            rope_params,
            attn_params,
            layers,
            final_rms,
            final_cls,
            final_cls_ty,
            kv_filled: 0,
            embed: vec![0.0; dim],
        }
    }

    /// Run one decode step entirely on the device: one command encoder, one
    /// submission, one logits read-back. See [`Backend::forward_step`].
    fn fused_step(
        &self,
        model: &crate::model::Model,
        state: &mut crate::model::RunState,
        token: usize,
        pos: usize,
    ) {
        let mut guard = self.decode.lock().unwrap();
        let stale = match &*guard {
            Some(d) => {
                d.dim != model.config.dim
                    || d.n_layers != model.config.n_layers
                    || d.vocab != model.config.vocab_size
                    || d.seq_len != model.config.seq_len
            }
            None => true,
        };
        if stale {
            *guard = Some(self.build_decode_state(model));
        }
        let d = guard.as_mut().unwrap();

        // One-time host->device sync of any KV positions a prior prefill filled.
        if d.kv_filled < pos {
            let (kv_dim, seq) = (d.kv_dim, d.seq_len);
            for l in 0..d.n_layers {
                let loff = l * seq * kv_dim;
                let lo = loff + d.kv_filled * kv_dim;
                let hi = loff + pos * kv_dim;
                let off = (d.kv_filled * kv_dim * 4) as u64;
                self.queue
                    .write_buffer(&d.key[l], off, f32_bytes(&state.key_cache()[lo..hi]));
                self.queue
                    .write_buffer(&d.value[l], off, f32_bytes(&state.value_cache()[lo..hi]));
            }
            d.kv_filled = pos;
        }

        // Per-step inputs: this token's embedding, and the pos-dependent params.
        model
            .weights
            .token_embedding_table
            .dequant_row(token, &mut d.embed);
        self.queue.write_buffer(&d.x, 0, f32_bytes(&d.embed));
        // Patch just the pos-dependent words: rope's `pos` (3rd u32, byte 8) and
        // attention's `np = pos+1` (5th u32, byte 16).
        self.queue
            .write_buffer(&d.rope_params, 8, u32_bytes(&[pos as u32]));
        self.queue
            .write_buffer(&d.attn_params, 16, u32_bytes(&[(pos + 1) as u32]));

        // Record the whole step into one encoder (one pass per op => wgpu
        // inserts the read-after-write barriers between dependent dispatches).
        let wg_dim = ceil_div(d.dim, 64);
        let wg_kv = ceil_div(d.kv_dim, 64);
        let wg_hidden = ceil_div(d.hidden, 64);
        let wg_vocab = ceil_div(d.vocab, 64);
        let wg_rope = ceil_div(d.dim / 2, 64);
        let wg_heads = d.n_heads as u32;
        let pos_off = (pos * d.kv_dim * 4) as u64;
        let kv_bytes = (d.kv_dim * 4) as u64;

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for l in 0..d.n_layers {
            let lb = &d.layers[l];
            let mm = lb.mm_ty.map(|t| self.matmul_pipeline(t)); // [wq,wk,wv,wo,w1,w3,w2]
            pass(&mut enc, &self.pipelines.rmsnorm, &lb.rms_att, 1);
            pass(&mut enc, mm[0], &lb.mq, wg_dim);
            pass(&mut enc, mm[1], &lb.mk, wg_kv);
            pass(&mut enc, mm[2], &lb.mv, wg_kv);
            pass(&mut enc, &self.pipelines.rope, &lb.rope, wg_rope);
            // Stash the rotated K/V for this position into the resident cache.
            enc.copy_buffer_to_buffer(&d.k_tmp, 0, &d.key[l], pos_off, kv_bytes);
            enc.copy_buffer_to_buffer(&d.v_tmp, 0, &d.value[l], pos_off, kv_bytes);
            pass(&mut enc, &self.pipelines.attention, &lb.attn, wg_heads);
            pass(&mut enc, mm[3], &lb.mo, wg_dim);
            pass(&mut enc, &self.pipelines.add, &lb.add_attn, wg_dim);
            pass(&mut enc, &self.pipelines.rmsnorm, &lb.rms_ffn, 1);
            pass(&mut enc, mm[4], &lb.m1, wg_hidden);
            pass(&mut enc, mm[5], &lb.m3, wg_hidden);
            pass(&mut enc, &self.pipelines.swiglu, &lb.swiglu, wg_hidden);
            pass(&mut enc, mm[6], &lb.m2, wg_dim);
            pass(&mut enc, &self.pipelines.add, &lb.add_ffn, wg_dim);
        }
        pass(&mut enc, &self.pipelines.rmsnorm, &d.final_rms, 1);
        pass(&mut enc, self.matmul_pipeline(d.final_cls_ty), &d.final_cls, wg_vocab);
        enc.copy_buffer_to_buffer(&d.logits, 0, &d.logits_staging, 0, (d.vocab * 4) as u64);
        self.queue.submit(Some(enc.finish()));
        d.kv_filled = pos + 1;

        self.map_read(&d.logits_staging, state.logits_mut());
    }
}

impl Backend for GpuBackend {
    fn rmsnorm(&self, out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
        let xb = self.storage_ro("x", f32_bytes(x));
        let wb = self.storage_ro("weight", f32_bytes(weight));
        let ob = self.alloc_out(out.len());
        let pb = self.params(&[x.len() as u32, eps.to_bits()]);
        let bind = self.bind(&self.pipelines.rmsnorm, &[&xb, &wb, &ob, &pb]);
        self.run(&self.pipelines.rmsnorm, &bind, 1, &ob, out); // single-workgroup reduction
    }

    fn matmul(&self, out: &mut [f32], x: &[f32], w: &QMatrix) {
        let gw = self.weight_buffer(w);
        let pipe = self.matmul_pipeline(gw.ty);
        let xb = self.storage_ro("x", f32_bytes(x));
        let ob = self.alloc_out(out.len());
        let pb = self.params(&[w.rows() as u32, w.cols() as u32]);
        let bind = self.bind(pipe, &[&gw.buf, &xb, &ob, &pb]);
        self.run(pipe, &bind, ceil_div(w.rows(), 64), &ob, out);
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
        let qb = self.storage_rw("q", f32_bytes(q));
        let kb = self.storage_rw("k", f32_bytes(k));
        let fb = self.table_buffer(inv_freq);
        let pb = self.params(&[
            head_size as u32,
            kv_dim as u32,
            pos as u32,
            inv_freq.len() as u32,
            mscale.to_bits(),
            q.len() as u32,
        ]);
        let bind = self.bind(&self.pipelines.rope, &[&qb, &kb, &fb, &pb]);
        self.dispatch(&self.pipelines.rope, &bind, ceil_div(q.len() / 2, 64));
        self.read_back(&qb, q);
        self.read_back(&kb, k);
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
        let np = pos + 1;
        let scale = 1.0 / (head_size as f32).sqrt();
        // Only the first `np` timesteps of the cache are populated/needed.
        let qb = self.storage_ro("q", f32_bytes(q));
        let kb = self.storage_ro("keyc", f32_bytes(&key_cache[..np * kv_dim]));
        let vb = self.storage_ro("valc", f32_bytes(&value_cache[..np * kv_dim]));
        let ab = self.alloc_out(att.len()); // scratch scores (not read back)
        let ob = self.alloc_out(out.len());
        let pb = self.params(&[
            head_size as u32,
            kv_dim as u32,
            seq_len as u32,
            (n_heads / n_kv_heads) as u32,
            np as u32,
            scale.to_bits(),
        ]);
        let bind = self.bind(&self.pipelines.attention, &[&qb, &kb, &vb, &ab, &ob, &pb]);
        // One workgroup per head.
        self.run(&self.pipelines.attention, &bind, n_heads as u32, &ob, out);
    }

    fn swiglu(&self, hb: &mut [f32], hb2: &[f32]) {
        let h = self.storage_rw("hb", f32_bytes(hb));
        let h2 = self.storage_ro("hb2", f32_bytes(hb2));
        let pb = self.params(&[hb.len() as u32]);
        let bind = self.bind(&self.pipelines.swiglu, &[&h, &h2, &pb]);
        self.run(&self.pipelines.swiglu, &bind, ceil_div(hb.len(), 64), &h, hb);
    }

    fn add(&self, out: &mut [f32], x: &[f32]) {
        let ob = self.storage_rw("out", f32_bytes(out));
        let xb = self.storage_ro("x", f32_bytes(x));
        let pb = self.params(&[out.len() as u32]);
        let bind = self.bind(&self.pipelines.add, &[&ob, &xb, &pb]);
        self.run(&self.pipelines.add, &bind, ceil_div(out.len(), 64), &ob, out);
    }

    // --- Batched (prefill) overrides ------------------------------------

    fn matmul_batch(&self, out: &mut [f32], x: &[f32], w: &QMatrix, rows: usize) {
        let (oc, ic) = (w.rows(), w.cols());
        let gw = self.weight_buffer(w);
        let pipe = self.matmul_batch_pipeline(gw.ty);
        let xb = self.storage_ro("x", f32_bytes(x));
        let ob = self.alloc_out(out.len());
        let pb = self.params(&[oc as u32, ic as u32, rows as u32]);
        let bind = self.bind(pipe, &[&gw.buf, &xb, &ob, &pb]);
        // 2-D grid: x over output features, y over batch rows.
        let grid = [ceil_div(oc, 64), rows as u32, 1];
        self.run_grid(pipe, &bind, grid, &ob, out);
    }

    fn rmsnorm_batch(&self, out: &mut [f32], x: &[f32], weight: &[f32], eps: f32, rows: usize) {
        let dim = weight.len();
        let xb = self.storage_ro("x", f32_bytes(x));
        let wb = self.storage_ro("weight", f32_bytes(weight));
        let ob = self.alloc_out(out.len());
        let pb = self.params(&[dim as u32, eps.to_bits(), rows as u32]);
        let bind = self.bind(&self.pipelines.rmsnorm_batch, &[&xb, &wb, &ob, &pb]);
        // One workgroup per row (each reduces its own row over 256 lanes).
        self.run_grid(&self.pipelines.rmsnorm_batch, &bind, [rows as u32, 1, 1], &ob, out);
    }

    #[allow(clippy::too_many_arguments)]
    fn rope_batch(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        pos_base: usize,
        rows: usize,
        head_size: usize,
        kv_dim: usize,
        q_dim: usize,
        inv_freq: &[f32],
        mscale: f32,
    ) {
        let qb = self.storage_rw("q", f32_bytes(q));
        let kb = self.storage_rw("k", f32_bytes(k));
        let fb = self.table_buffer(inv_freq);
        let pb = self.params(&[
            head_size as u32,
            kv_dim as u32,
            pos_base as u32,
            inv_freq.len() as u32,
            mscale.to_bits(),
            q_dim as u32,
            rows as u32,
        ]);
        let bind = self.bind(&self.pipelines.rope_batch, &[&qb, &kb, &fb, &pb]);
        // 2-D grid: x over (even,odd) pairs of a row, y over batch rows.
        let grid = [ceil_div(q_dim / 2, 64), rows as u32, 1];
        self.dispatch_grid(&self.pipelines.rope_batch, &bind, grid);
        self.read_back(&qb, q);
        self.read_back(&kb, k);
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_batch(
        &self,
        out: &mut [f32],
        q: &[f32],
        key_cache: &[f32],
        value_cache: &[f32],
        att: &mut [f32],
        pos_base: usize,
        rows: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_size: usize,
        seq_len: usize,
        kv_dim: usize,
    ) {
        // The kernel's per-thread accumulator is fixed at MAX_BATCH_HEAD; wider
        // heads fall back to the looped single-token attention (still correct).
        if head_size > MAX_BATCH_HEAD {
            let dim = n_heads * head_size;
            for r in 0..rows {
                self.attention(
                    &mut out[r * dim..r * dim + dim],
                    &q[r * dim..r * dim + dim],
                    key_cache,
                    value_cache,
                    att,
                    pos_base + r,
                    n_heads,
                    n_kv_heads,
                    head_size,
                    seq_len,
                    kv_dim,
                );
            }
            return;
        }
        // Causal: row r reads keys 0..=pos_base+r, so only the first
        // (pos_base + rows) cache positions are needed.
        let used = (pos_base + rows) * kv_dim;
        let qb = self.storage_ro("q", f32_bytes(q));
        let kb = self.storage_ro("keyc", f32_bytes(&key_cache[..used]));
        let vb = self.storage_ro("valc", f32_bytes(&value_cache[..used]));
        let ob = self.alloc_out(out.len());
        let pb = self.params(&[
            head_size as u32,
            kv_dim as u32,
            (n_heads / n_kv_heads) as u32,
            pos_base as u32,
            rows as u32,
            n_heads as u32,
            (1.0 / (head_size as f32).sqrt()).to_bits(),
        ]);
        let bind = self.bind(&self.pipelines.attention_batch, &[&qb, &kb, &vb, &ob, &pb]);
        // One thread per (batch row, head).
        let grid = [ceil_div(rows * n_heads, 64), 1, 1];
        self.run_grid(&self.pipelines.attention_batch, &bind, grid, &ob, out);
        let _ = att; // GPU path keeps softmax state in registers; att unused.
    }

    fn forward_step(
        &self,
        model: &crate::model::Model,
        state: &mut crate::model::RunState,
        token: usize,
        pos: usize,
    ) {
        self.fused_step(model, state, token, pos);
    }
}

/// Compile a single-entry-point compute pipeline from WGSL source.
fn make_pipeline(device: &wgpu::Device, src: &str, label: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(src)),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// `ceil(a / b)` for laying out workgroups.
fn ceil_div(a: usize, b: usize) -> u32 {
    a.div_ceil(b) as u32
}

/// View an f32 slice as raw little-endian bytes (no copy).
///
/// SAFETY: every f32 bit pattern is a valid byte sequence; the resulting slice
/// borrows the same memory with length `4 * s.len()`. The crate is little-
/// endian only (asserted at the crate root), so the bytes match the on-device
/// layout.
fn f32_bytes(s: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

/// View a u32 slice as raw little-endian bytes (no copy). SAFETY: as above.
fn u32_bytes(s: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    /// A GPU backend for tests, or `None` (with a skip note) when no adapter is
    /// available — so the suite still passes in a headless/GPU-less CI.
    fn gpu() -> Option<GpuBackend> {
        match GpuBackend::new() {
            Ok(g) => Some(g),
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
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

    /// Assert two slices match within a tolerance generous enough for the
    /// reassociated / FMA-contracted f32 arithmetic the GPU may use.
    fn close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            let tol = 1e-3 + 1e-3 * w.abs();
            assert!((g - w).abs() <= tol, "idx {i}: gpu {g} vs cpu {w}");
        }
    }

    #[test]
    fn matmul_f32_basic() {
        let Some(g) = gpu() else { return };
        // Same hand-checked case as the CPU backend: w·[1,1] = [3, 7].
        let w = QMatrix::f32(vec![1.0, 2.0, 3.0, 4.0].into(), 2, 2).unwrap();
        let mut out = [0.0; 2];
        g.matmul(&mut out, &[1.0, 1.0], &w);
        close(&out, &[3.0, 7.0]);
    }

    #[test]
    fn matmul_f32_parity() {
        let Some(g) = gpu() else { return };
        // Rows deliberately not a multiple of the workgroup size.
        let (rows, cols) = (130usize, 96usize);
        let wf = noise(rows * cols, 1);
        let x = noise(cols, 2);
        let w = QMatrix::f32(wf.into(), rows, cols).unwrap();
        let (mut a, mut b) = (vec![0.0; rows], vec![0.0; rows]);
        g.matmul(&mut a, &x, &w);
        CpuBackend.matmul(&mut b, &x, &w);
        close(&a, &b);
    }

    #[test]
    fn matmul_quant_q8_0_matches_exact() {
        let Some(g) = gpu() else { return };
        use crate::quant::{quantize_q8_0, GgmlType};
        // Dequant-on-upload path: GPU does an exact f32 dot of the dequantized
        // weight with x, so it should match the exact f32 product of the
        // *dequantized* weight closely.
        let (rows, cols) = (3usize, 64usize);
        let wf = noise(rows * cols, 7);
        let x = noise(cols, 8);
        let bytes = quantize_q8_0(&wf);
        let deq = crate::quant::dequantize(GgmlType::Q8_0, &bytes, rows * cols).unwrap();
        let w = QMatrix::quant(GgmlType::Q8_0, bytes.into(), rows, cols).unwrap();
        let mut out = vec![0.0; rows];
        g.matmul(&mut out, &x, &w);
        let exact: Vec<f32> = (0..rows)
            .map(|r| (0..cols).map(|j| deq[r * cols + j] * x[j]).sum())
            .collect();
        close(&out, &exact);
    }

    #[test]
    fn rmsnorm_parity() {
        let Some(g) = gpu() else { return };
        for n in [4usize, 288, 769] {
            let x = noise(n, n as u64);
            let w = noise(n, n as u64 + 100);
            let (mut a, mut b) = (vec![0.0; n], vec![0.0; n]);
            g.rmsnorm(&mut a, &x, &w, 1e-5);
            CpuBackend.rmsnorm(&mut b, &x, &w, 1e-5);
            close(&a, &b);
        }
    }

    #[test]
    fn add_parity() {
        let Some(g) = gpu() else { return };
        let n = 200;
        let x = noise(n, 11);
        let mut a = noise(n, 12);
        let mut b = a.clone();
        g.add(&mut a, &x);
        CpuBackend.add(&mut b, &x);
        close(&a, &b);
    }

    #[test]
    fn swiglu_parity() {
        let Some(g) = gpu() else { return };
        let n = 200;
        let hb2 = noise(n, 21);
        let mut a = noise(n, 22);
        let mut b = a.clone();
        g.swiglu(&mut a, &hb2);
        CpuBackend.swiglu(&mut b, &hb2);
        close(&a, &b);
    }

    fn base_inv_freq(head_size: usize, theta: f32) -> Vec<f32> {
        (0..head_size / 2)
            .map(|j| (theta as f64).powf(-2.0 * j as f64 / head_size as f64) as f32)
            .collect()
    }

    #[test]
    fn rope_parity() {
        let Some(g) = gpu() else { return };
        // Full rotary, GQA widths (q = 2 heads of 8, k = 1 head of 8).
        let head_size = 8;
        let inv = base_inv_freq(head_size, 10000.0);
        for pos in [0usize, 1, 5] {
            let mut q_g = noise(2 * head_size, pos as u64 + 1);
            let mut q_c = q_g.clone();
            let mut k_g = noise(head_size, pos as u64 + 50);
            let mut k_c = k_g.clone();
            g.rope(&mut q_g, &mut k_g, pos, head_size, head_size, &inv, 1.0);
            CpuBackend.rope(&mut q_c, &mut k_c, pos, head_size, head_size, &inv, 1.0);
            close(&q_g, &q_c);
            close(&k_g, &k_c);
        }
    }

    #[test]
    fn rope_partial_and_mscale_parity() {
        let Some(g) = gpu() else { return };
        // Partial rotary (rotate 4 of 8 dims) with a YaRN-style magnitude scale.
        let head_size = 8;
        let inv = vec![1.0f32, 0.01]; // rope_dim = 4
        let mut q_g = noise(head_size, 3);
        let mut q_c = q_g.clone();
        let mut k_g = noise(head_size, 4);
        let mut k_c = k_g.clone();
        g.rope(&mut q_g, &mut k_g, 2, head_size, head_size, &inv, 1.25);
        CpuBackend.rope(&mut q_c, &mut k_c, 2, head_size, head_size, &inv, 1.25);
        close(&q_g, &q_c);
        close(&k_g, &k_c);
    }

    #[test]
    fn attention_known_values() {
        let Some(g) = gpu() else { return };
        // pos 0, one head: softmax over a single score is 1 -> output is the
        // only value vector. Matches the CPU backend's hand-checked test.
        let mut out = [0.0; 2];
        let mut att = [0.0; 1];
        g.attention(
            &mut out,
            &[0.5, 0.5],
            &[9.0, 9.0],
            &[3.0, -4.0],
            &mut att,
            0,
            1,
            1,
            2,
            1,
            2,
        );
        close(&out, &[3.0, -4.0]);
    }

    #[test]
    #[ignore = "timing benchmark; run with --release --features gpu -- --ignored --nocapture"]
    fn bench_matmul_gpu_vs_cpu() {
        use std::time::Instant;
        let Some(g) = gpu() else { return };

        let (rows, cols, iters) = (4096usize, 4096usize, 50);
        let wf = noise(rows * cols, 99);
        let w = QMatrix::f32(wf.into(), rows, cols).unwrap();
        let x = noise(cols, 100);
        let mut out = vec![0.0; rows];

        // Warm up: first GPU matmul uploads + caches the resident weight.
        g.matmul(&mut out, &x, &w);

        let t = Instant::now();
        for _ in 0..iters {
            g.matmul(&mut out, &x, &w);
        }
        let gpu = t.elapsed();

        let cpu_backend = CpuBackend::new();
        let t = Instant::now();
        for _ in 0..iters {
            cpu_backend.matmul(&mut out, &x, &w);
        }
        let cpu = t.elapsed();

        eprintln!(
            "matmul {rows}x{cols} x{iters} (weight resident): \
             gpu={:.3}ms/op cpu={:.3}ms/op  -> gpu is {:.2}x the cpu time",
            gpu.as_secs_f64() * 1e3 / iters as f64,
            cpu.as_secs_f64() * 1e3 / iters as f64,
            gpu.as_secs_f64() / cpu.as_secs_f64(),
        );
    }

    #[test]
    #[ignore = "timing benchmark; run with --release --features gpu -- --ignored --nocapture"]
    fn bench_prefill_gpu_vs_cpu() {
        use std::time::Instant;
        let Some(g) = gpu() else { return };

        // A small-but-real-shaped model; prefill a 256-token prompt.
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
        let bytes = crate::dummy::synthetic_gguf_typed(&c, crate::GgmlType::F32);
        let gguf = crate::Gguf::parse(&bytes).unwrap();
        let model = crate::Model::from_gguf(&gguf).unwrap();
        let tokens: Vec<usize> = (0..256).map(|i| (i * 7) % c.vocab_size).collect();

        let prefill = |backend: &dyn Backend| {
            let mut s = crate::RunState::new(&model.config);
            crate::forward_prefill(&model, &mut s, backend, &tokens, 0);
        };

        prefill(&g); // warm up: make weights resident
        let t = Instant::now();
        prefill(&g);
        let gpu = t.elapsed();

        let cpu_backend = CpuBackend::new();
        let t = Instant::now();
        prefill(&cpu_backend);
        let cpu = t.elapsed();

        eprintln!(
            "prefill {} tokens, dim {} x{} layers: gpu={gpu:?} cpu={cpu:?} \
             -> gpu is {:.2}x the cpu time ({:.0} vs {:.0} tok/s)",
            tokens.len(),
            c.dim,
            c.n_layers,
            gpu.as_secs_f64() / cpu.as_secs_f64(),
            tokens.len() as f64 / gpu.as_secs_f64(),
            tokens.len() as f64 / cpu.as_secs_f64(),
        );
    }

    #[test]
    #[ignore = "timing benchmark; run with --release --features gpu -- --ignored --nocapture"]
    fn bench_decode_gpu_vs_cpu() {
        use std::time::Instant;
        let Some(g) = gpu() else { return };

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
        let bytes = crate::dummy::synthetic_gguf_typed(&c, crate::GgmlType::F32);
        let gguf = crate::Gguf::parse(&bytes).unwrap();
        let model = crate::Model::from_gguf(&gguf).unwrap();
        let steps = 64usize;

        let time = |backend: &dyn Backend, warm: usize| -> std::time::Duration {
            let mut s = crate::RunState::new(&model.config);
            for pos in 0..warm {
                backend.forward_step(&model, &mut s, (pos * 13) % c.vocab_size, pos);
            }
            let t = Instant::now();
            for i in 0..steps {
                let pos = warm + i;
                backend.forward_step(&model, &mut s, (pos * 13) % c.vocab_size, pos);
            }
            t.elapsed()
        };

        let gpu = time(&g, 1); // warm: build resident state + upload weights
        let cpu = time(&CpuBackend::new(), 0);
        eprintln!(
            "decode {steps} steps, dim {} x{} layers: gpu={gpu:?} cpu={cpu:?} \
             -> gpu is {:.2}x the cpu time ({:.0} vs {:.0} tok/s)",
            c.dim,
            c.n_layers,
            gpu.as_secs_f64() / cpu.as_secs_f64(),
            steps as f64 / gpu.as_secs_f64(),
            steps as f64 / cpu.as_secs_f64(),
        );
    }

    #[test]
    #[ignore = "timing benchmark; run with --release --features gpu -- --ignored --nocapture"]
    fn bench_decode_quant_vs_f32() {
        use std::time::Instant;
        if GpuBackend::new().is_err() {
            return;
        }
        // A weights-heavy model so the weight-streaming cost is visible.
        let c = crate::Config {
            dim: 2048,
            hidden_dim: 5632,
            n_layers: 8,
            n_heads: 32,
            n_kv_heads: 32,
            vocab_size: 32000,
            seq_len: 256,
            shared_weights: true,
            ..Default::default()
        };
        let steps = 32usize;
        for ty in [crate::GgmlType::F32, crate::GgmlType::Q8_0] {
            // Fresh backend per model (same dims => the decode state wouldn't rebuild).
            let g = GpuBackend::new().unwrap();
            let bytes = crate::dummy::synthetic_gguf_typed(&c, ty);
            let gguf = crate::Gguf::parse(&bytes).unwrap();
            let model = crate::Model::from_gguf(&gguf).unwrap();
            let mut s = crate::RunState::new(&model.config);
            g.forward_step(&model, &mut s, 1, 0); // warm: build state + upload weights
            let t = Instant::now();
            for i in 0..steps {
                g.forward_step(&model, &mut s, (i * 7) % c.vocab_size, 1 + i);
            }
            let el = t.elapsed();
            let bytes_per_elem = if ty == crate::GgmlType::F32 { 4.0 } else { 34.0 / 32.0 };
            eprintln!(
                "{ty:?}: decode {steps} steps (dim {} x{}L) -> {:.0} tok/s; \
                 weights ~{:.1}x f16-element bytes ({:.2} B/elem)",
                c.dim,
                c.n_layers,
                steps as f64 / el.as_secs_f64(),
                bytes_per_elem / 4.0,
                bytes_per_elem,
            );
        }
    }

    #[test]
    fn attention_gqa_parity() {
        let Some(g) = gpu() else { return };
        // 4 query heads, 2 kv heads, head_size 4, a few timesteps populated.
        let (n_heads, n_kv_heads, head_size, seq_len) = (4usize, 2usize, 4usize, 8usize);
        let kv_dim = n_kv_heads * head_size;
        let pos = 5usize;
        let q = noise(n_heads * head_size, 1);
        let kc = noise(seq_len * kv_dim, 2);
        let vc = noise(seq_len * kv_dim, 3);
        let mut a_g = vec![0.0; n_heads * head_size];
        let mut a_c = a_g.clone();
        let mut att_g = vec![0.0; n_heads * seq_len];
        let mut att_c = att_g.clone();
        g.attention(
            &mut a_g, &q, &kc, &vc, &mut att_g, pos, n_heads, n_kv_heads, head_size, seq_len, kv_dim,
        );
        CpuBackend.attention(
            &mut a_c, &q, &kc, &vc, &mut att_c, pos, n_heads, n_kv_heads, head_size, seq_len, kv_dim,
        );
        close(&a_g, &a_c);
    }

    // --- Batched-op parity (GPU override vs CPU default loop) ------------

    #[test]
    fn matmul_batch_parity() {
        let Some(g) = gpu() else { return };
        // Output features not a multiple of the workgroup size; several rows.
        let (oc, ic, n) = (130usize, 96usize, 5usize);
        let w = QMatrix::f32(noise(oc * ic, 1).into(), oc, ic).unwrap();
        let x = noise(n * ic, 2);
        let (mut a, mut b) = (vec![0.0; n * oc], vec![0.0; n * oc]);
        g.matmul_batch(&mut a, &x, &w, n);
        CpuBackend.matmul_batch(&mut b, &x, &w, n);
        close(&a, &b);
    }

    #[test]
    fn rmsnorm_batch_parity() {
        let Some(g) = gpu() else { return };
        let (dim, n) = (288usize, 7usize);
        let x = noise(n * dim, 3);
        let w = noise(dim, 4);
        let (mut a, mut b) = (vec![0.0; n * dim], vec![0.0; n * dim]);
        g.rmsnorm_batch(&mut a, &x, &w, 1e-5, n);
        CpuBackend.rmsnorm_batch(&mut b, &x, &w, 1e-5, n);
        close(&a, &b);
    }

    #[test]
    fn rope_batch_parity() {
        let Some(g) = gpu() else { return };
        // 2 q heads, 1 kv head of size 8; each row at a different position.
        let (head_size, n) = (8usize, 6usize);
        let (q_dim, kv_dim) = (2 * head_size, head_size);
        let inv = base_inv_freq(head_size, 10000.0);
        let mut q_g = noise(n * q_dim, 5);
        let mut q_c = q_g.clone();
        let mut k_g = noise(n * kv_dim, 6);
        let mut k_c = k_g.clone();
        g.rope_batch(&mut q_g, &mut k_g, 3, n, head_size, kv_dim, q_dim, &inv, 1.0);
        CpuBackend.rope_batch(&mut q_c, &mut k_c, 3, n, head_size, kv_dim, q_dim, &inv, 1.0);
        close(&q_g, &q_c);
        close(&k_g, &k_c);
    }

    #[test]
    fn attention_batch_parity() {
        let Some(g) = gpu() else { return };
        // GQA, batch of query rows at positions 0..n, causal over a filled cache.
        let (n_heads, n_kv_heads, head_size, seq_len) = (4usize, 2usize, 8usize, 16usize);
        let kv_dim = n_kv_heads * head_size;
        let dim = n_heads * head_size;
        let n = 10usize; // rows / positions 0..10
        let q = noise(n * dim, 1);
        let kc = noise(seq_len * kv_dim, 2);
        let vc = noise(seq_len * kv_dim, 3);
        let (mut a, mut b) = (vec![0.0; n * dim], vec![0.0; n * dim]);
        let mut att = vec![0.0; n_heads * seq_len];
        g.attention_batch(
            &mut a, &q, &kc, &vc, &mut att, 0, n, n_heads, n_kv_heads, head_size, seq_len, kv_dim,
        );
        CpuBackend.attention_batch(
            &mut b, &q, &kc, &vc, &mut att, 0, n, n_heads, n_kv_heads, head_size, seq_len, kv_dim,
        );
        close(&a, &b);
    }
}
