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

// Cooperative GEMV: one workgroup per output row, 256 threads cooperatively
// reduce the dot product. Threads read consecutive weight elements (coalesced),
// unlike one-thread-per-row which strides by `cols`. Dispatch `rows` workgroups.
const WGSL_MATMUL: &str = r#"
struct P { rows: u32, cols: u32, acc: u32 };
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<storage, read> p: P;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    if (row >= p.rows) { return; }
    let tid = lid.x;
    let base = row * p.cols;
    var acc = 0.0;
    for (var j = tid; j < p.cols; j = j + 256u) {
        acc = acc + w[base + j] * x[j];
    }
    red[tid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
        workgroupBarrier();
    }
    if (tid == 0u) { outp[row] = select(red[0], outp[row] + red[0], p.acc != 0u); }
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
fn sext8(b: u32) -> i32 { return (i32(b) << 24u) >> 24u; }   // signed 8-bit
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
// Q4_K 6-bit packed scale+min for sub-block j (scales at byte `base`).
fn scale_min_k4(j: u32, base: u32) -> vec2<u32> {
    if (j < 4u) {
        return vec2<u32>(gb(base + j) & 63u, gb(base + j + 4u) & 63u);
    }
    let d = (gb(base + j + 4u) & 0x0fu) | ((gb(base + j - 4u) >> 6u) << 4u);
    let m = (gb(base + j + 4u) >> 4u) | ((gb(base + j) >> 6u) << 4u);
    return vec2<u32>(d, m);
}
"#;

// Q8_0 cooperative GEMV: one workgroup per row, threads split the 34-byte
// blocks and reduce. out[row] = sum_b sum_i d*q*x. Dispatch `rows` workgroups.
const WGSL_MATMUL_Q8_0: &str = r#"
struct P { rows: u32, cols: u32, acc: u32 };
@group(0) @binding(0) var<storage, read> wb: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<storage, read> p: P;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    if (row >= p.rows) { return; }
    let tid = lid.x;
    let nb = p.cols / 32u;
    let row_base = row * nb * 34u;
    var acc = 0.0;
    for (var b = tid; b < nb; b = b + 64u) {
        let bb = row_base + b * 34u;
        let d = f16f32(gb(bb) | (gb(bb + 1u) << 8u));
        let xb = b * 32u;
        for (var i = 0u; i < 32u; i = i + 1u) {
            let q = (i32(gb(bb + 2u + i)) << 24u) >> 24u;
            acc = acc + d * f32(q) * x[xb + i];
        }
    }
    red[tid] = acc;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
        if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
        workgroupBarrier();
    }
    if (tid == 0u) { outp[row] = select(red[0], outp[row] + red[0], p.acc != 0u); }
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

// --- int8 (DP4A) decode path -----------------------------------------------
//
// On the GPU, dequantizing to f32 and doing an f32 dot is ~2.5× slower than an
// int8 dot via `dot4I8Packed` (one DP4A = 4 i8×i8 MACs). For decode we quantize
// each f32 activation to int8 (+per-32 scale) on-device once, then matmul reads
// int8 weights (re-laid contiguous + per-block scale) and the int8 activation.

// f32 activation -> int8 (4 packed per u32) + per-32-block f32 scale. One thread
// per 32-block; dispatch ceil(nblocks/64) workgroups.
const WGSL_QUANTIZE_Q8: &str = r#"
struct P { nblocks: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> qout: array<u32>;
@group(0) @binding(2) var<storage, read_write> scout: array<f32>;
@group(0) @binding(3) var<storage, read> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= p.nblocks) { return; }
    let base = b * 32u;
    var amax = 0.0;
    for (var i = 0u; i < 32u; i = i + 1u) { amax = max(amax, abs(x[base + i])); }
    let scale = amax / 127.0;
    let inv = select(0.0, 1.0 / scale, scale > 0.0);
    scout[b] = scale;
    for (var k = 0u; k < 8u; k = k + 1u) {
        var packed: u32 = 0u;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let q = clamp(i32(round(x[base + k * 4u + j] * inv)), -128, 127);
            packed = packed | ((u32(q) & 0xffu) << (j * 8u));
        }
        qout[b * 8u + k] = packed;
    }
}
"#;

// int8 Q8_0 cooperative GEMV via dot4I8Packed. Weights are re-laid to contiguous
// int8 (`wq`, 4 per u32) + per-32 scale (`wscale`); activation is int8 (`aq`) +
// scale (`ascale`). `acc != 0` folds the residual add in (out[row] += dot).
const WGSL_MATMUL_Q8_0_I8: &str = r#"
struct P { rows: u32, cols: u32, acc: u32 };
@group(0) @binding(0) var<storage, read> wq: array<u32>;
@group(0) @binding(1) var<storage, read> wscale: array<f32>;
@group(0) @binding(2) var<storage, read> aq: array<u32>;
@group(0) @binding(3) var<storage, read> ascale: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
@group(0) @binding(5) var<storage, read> p: P;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    if (row >= p.rows) { return; }
    let tid = lid.x;
    let nb = p.cols / 32u;
    let wbase = row * (p.cols / 4u);
    let sbase = row * nb;
    var acc = 0.0;
    for (var b = tid; b < nb; b = b + 64u) {
        var dot: i32 = 0;
        let wu = wbase + b * 8u;
        let au = b * 8u;
        for (var k = 0u; k < 8u; k = k + 1u) {
            dot = dot + dot4I8Packed(wq[wu + k], aq[au + k]);
        }
        acc = acc + wscale[sbase + b] * ascale[b] * f32(dot);
    }
    red[tid] = acc;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
        if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
        workgroupBarrier();
    }
    if (tid == 0u) { outp[row] = select(red[0], outp[row] + red[0], p.acc != 0u); }
}
"#;

// Q4_K: 144-byte superblocks of 256 (f16 d, f16 dmin, 12B packed 6-bit
// scale/min, 128B of 4-bit quants). Mirrors quant::block_q4_k.
// Q4_K cooperative GEMV: one workgroup per row; threads split the 32-element
// output chunks (8 per 144-byte superblock). out-chunk oc => qs-chunk oc/2,
// low/high nibble by oc&1, sub-block scale scale_min_k4(oc). Dispatch `rows` wg.
const WGSL_MATMUL_Q4_K: &str = r#"
struct P { rows: u32, cols: u32, acc: u32 };
@group(0) @binding(0) var<storage, read> wb: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<storage, read> p: P;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    if (row >= p.rows) { return; }
    let tid = lid.x;
    let nchunks = p.cols / 32u;
    let row_base = row * (p.cols / 256u) * 144u;
    var acc = 0.0;
    for (var c = tid; c < nchunks; c = c + 64u) {
        let sb = c / 8u;
        let oc = c % 8u;
        let bb = row_base + sb * 144u;
        let d = f16f32(gb(bb) | (gb(bb + 1u) << 8u));
        let dmin = f16f32(gb(bb + 2u) | (gb(bb + 3u) << 8u));
        let sm = scale_min_k4(oc, bb + 4u);
        let sc = d * f32(sm.x);
        let mn = dmin * f32(sm.y);
        let qbase = bb + 16u + (oc / 2u) * 32u;
        let high = (oc & 1u) == 1u;
        let xbase = sb * 256u + oc * 32u;
        for (var i = 0u; i < 32u; i = i + 1u) {
            let byte = gb(qbase + i);
            let nib = select(byte & 0x0fu, byte >> 4u, high);
            acc = acc + (sc * f32(nib) - mn) * x[xbase + i];
        }
    }
    red[tid] = acc;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
        if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
        workgroupBarrier();
    }
    if (tid == 0u) { outp[row] = select(red[0], outp[row] + red[0], p.acc != 0u); }
}
"#;

const WGSL_MATMUL_Q4_K_BATCH: &str = r#"
struct P { rows: u32, cols: u32, n: u32 };
@group(0) @binding(0) var<storage, read> wb: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<storage, read> p: P;
fn dot_q4k_row(row_base: u32, nb: u32, xoff: u32) -> f32 {
    var acc = 0.0;
    for (var b = 0u; b < nb; b = b + 1u) {
        let bb = row_base + b * 144u;
        let d = f16f32(gb(bb) | (gb(bb + 1u) << 8u));
        let dmin = f16f32(gb(bb + 2u) | (gb(bb + 3u) << 8u));
        let sbase = bb + 4u;
        let qbase = bb + 16u;
        let xb = xoff + b * 256u;
        var y = 0u;
        var is = 0u;
        for (var chunk = 0u; chunk < 4u; chunk = chunk + 1u) {
            let sm1 = scale_min_k4(is, sbase);
            let sm2 = scale_min_k4(is + 1u, sbase);
            let d1 = d * f32(sm1.x); let min1 = dmin * f32(sm1.y);
            let d2 = d * f32(sm2.x); let min2 = dmin * f32(sm2.y);
            let cbase = qbase + chunk * 32u;
            for (var j = 0u; j < 32u; j = j + 1u) {
                acc = acc + (d1 * f32(gb(cbase + j) & 0x0fu) - min1) * x[xb + y];
                y = y + 1u;
            }
            for (var j = 0u; j < 32u; j = j + 1u) {
                acc = acc + (d2 * f32(gb(cbase + j) >> 4u) - min2) * x[xb + y];
                y = y + 1u;
            }
            is = is + 2u;
        }
    }
    return acc;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let r = gid.y;
    if (row >= p.rows || r >= p.n) { return; }
    let nb = p.cols / 256u;
    outp[r * p.rows + row] = dot_q4k_row(row * nb * 144u, nb, r * p.cols);
}
"#;

// Q6_K: 210-byte superblocks of 256 (128B ql, 64B qh, 16 signed-i8 scales,
// f16 d). Mirrors quant::block_q6_k.
// Q6_K cooperative GEMV: one workgroup per row; threads split the 32-element
// output chunks (8 per 210-byte superblock). out-chunk oc => half oc/4, group
// oc%4 (ql nibble + 2 qh bits + signed scale). Dispatch `rows` workgroups.
const WGSL_MATMUL_Q6_K: &str = r#"
struct P { rows: u32, cols: u32, acc: u32 };
@group(0) @binding(0) var<storage, read> wb: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<storage, read> p: P;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    if (row >= p.rows) { return; }
    let tid = lid.x;
    let nchunks = p.cols / 32u;
    let row_base = row * (p.cols / 256u) * 210u;
    var acc = 0.0;
    for (var c = tid; c < nchunks; c = c + 64u) {
        let sb = c / 8u;
        let oc = c % 8u;
        let bb = row_base + sb * 210u;
        let d = f16f32(gb(bb + 208u) | (gb(bb + 209u) << 8u));
        let half = oc / 4u;
        let group = oc % 4u;
        let qlb = bb + half * 64u;
        let qhb = bb + 128u + half * 32u;
        let scb = bb + 192u + half * 8u;
        let xbase = sb * 256u + oc * 32u;
        let lo_l32 = (group & 1u) == 1u;
        let hi_nib = group >= 2u;
        let shift = group * 2u;
        for (var l = 0u; l < 32u; l = l + 1u) {
            let is = l / 16u;
            let qlbyte = gb(qlb + select(l, l + 32u, lo_l32));
            let nib = select(qlbyte & 0x0fu, qlbyte >> 4u, hi_nib);
            let qhi = (gb(qhb + l) >> shift) & 3u;
            let q = i32(nib | (qhi << 4u)) - 32;
            let sc = sext8(gb(scb + is + group * 2u));
            acc = acc + d * f32(sc) * f32(q) * x[xbase + l];
        }
    }
    red[tid] = acc;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
        if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
        workgroupBarrier();
    }
    if (tid == 0u) { outp[row] = select(red[0], outp[row] + red[0], p.acc != 0u); }
}
"#;

const WGSL_MATMUL_Q6_K_BATCH: &str = r#"
struct P { rows: u32, cols: u32, n: u32 };
@group(0) @binding(0) var<storage, read> wb: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<storage, read> p: P;
fn dot_q6k_row(row_base: u32, nb: u32, xoff: u32) -> f32 {
    var acc = 0.0;
    for (var b = 0u; b < nb; b = b + 1u) {
        let bb = row_base + b * 210u;
        let d = f16f32(gb(bb + 208u) | (gb(bb + 209u) << 8u));
        let xb = xoff + b * 256u;
        for (var half = 0u; half < 2u; half = half + 1u) {
            let qlb = bb + half * 64u;
            let qhb = bb + 128u + half * 32u;
            let scb = bb + 192u + half * 8u;
            let y = half * 128u;
            for (var l = 0u; l < 32u; l = l + 1u) {
                let is = l / 16u;
                let qll = gb(qlb + l);
                let qll32 = gb(qlb + l + 32u);
                let qhl = gb(qhb + l);
                let q1 = i32((qll & 0x0fu) | (((qhl >> 0u) & 3u) << 4u)) - 32;
                let q2 = i32((qll32 & 0x0fu) | (((qhl >> 2u) & 3u) << 4u)) - 32;
                let q3 = i32((qll >> 4u) | (((qhl >> 4u) & 3u) << 4u)) - 32;
                let q4 = i32((qll32 >> 4u) | (((qhl >> 6u) & 3u) << 4u)) - 32;
                acc = acc + d * f32(sext8(gb(scb + is))) * f32(q1) * x[xb + y + l];
                acc = acc + d * f32(sext8(gb(scb + is + 2u))) * f32(q2) * x[xb + y + l + 32u];
                acc = acc + d * f32(sext8(gb(scb + is + 4u))) * f32(q3) * x[xb + y + l + 64u];
                acc = acc + d * f32(sext8(gb(scb + is + 6u))) * f32(q4) * x[xb + y + l + 96u];
            }
        }
    }
    return acc;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let r = gid.y;
    if (row >= p.rows || r >= p.n) { return; }
    let nb = p.cols / 256u;
    outp[r * p.rows + row] = dot_q6k_row(row * nb * 210u, nb, r * p.cols);
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
    matmul_q4_k: wgpu::ComputePipeline,
    matmul_q4_k_batch: wgpu::ComputePipeline,
    matmul_q6_k: wgpu::ComputePipeline,
    matmul_q6_k_batch: wgpu::ComputePipeline,
    // Stage-1 int8 (DP4A) decode kernels, driven by the int8 fused-decode path.
    quantize_q8: wgpu::ComputePipeline,
    matmul_q8_0_i8: wgpu::ComputePipeline,
}

/// A resident weight: its device buffer and the format it's stored in (`F32`
/// for full-precision or host-dequantized weights, or a quant type that the
/// matmul kernel dequantizes in-shader).
#[derive(Clone)]
struct GpuWeight {
    buf: Arc<wgpu::Buffer>,
    ty: GgmlType,
}

/// A resident Q8_0 weight re-laid for the int8 (DP4A) decode path: contiguous
/// int8 values (4 packed per `u32`) plus the per-32-block f32 scales — the
/// layout `matmul_q8_0_i8` expects. The int8 analogue of [`GpuWeight`]; always
/// Q8_0-derived, so it carries no format tag.
#[derive(Clone)]
struct GpuWeightI8 {
    q: Arc<wgpu::Buffer>,
    scale: Arc<wgpu::Buffer>,
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
    rms_ffn: wgpu::BindGroup,
    m1: wgpu::BindGroup,
    m3: wgpu::BindGroup,
    swiglu: wgpu::BindGroup,
    m2: wgpu::BindGroup,
    /// On-device format of each matmul weight, to pick the matmul pipeline
    /// (order: wq, wk, wv, wo, w1, w3, w2).
    mm_ty: [GgmlType; 7],
}

/// Per-layer bind groups for the int8 (DP4A) fused decode — the seven matmuls
/// run as `matmul_q8_0_i8` against relaid int8 weights and the on-device-
/// quantized activation. The rmsnorm/rope/attention/swiglu ops stay f32 and
/// reuse the matching [`LayerBinds`] groups; the quantize bind groups aren't
/// per-layer (they reference fixed resident buffers) and live on [`DecodeInt8`].
struct LayerBindsInt8 {
    mq: wgpu::BindGroup,
    mk: wgpu::BindGroup,
    mv: wgpu::BindGroup,
    mo: wgpu::BindGroup,
    m1: wgpu::BindGroup,
    m3: wgpu::BindGroup,
    m2: wgpu::BindGroup,
}

/// The int8 (DP4A) overlay for [`DecodeState`], present only when every matmul
/// weight is Q8_0 and the int8 path is enabled. Holds the two reusable quantize
/// bind groups (one per activation width) and the per-layer + classifier int8
/// matmul bind groups. The resident int8 activation buffers it writes/reads
/// (`xb_i8`/`xb_scale`/`hb_i8`/`hb_scale`) and the f32 `DecodeState` buffers it
/// shares are all kept alive by these bind groups, so they need no field here.
struct DecodeInt8 {
    /// `quantize_q8` for the dim-wide xb: reused before every xb-consuming
    /// matmul group (after attn-rmsnorm, after attention, after ffn-rmsnorm) and
    /// before the classifier.
    quant_xb: wgpu::BindGroup,
    /// `quantize_q8` for the hidden-wide swiglu output, before `m2`.
    quant_hb: wgpu::BindGroup,
    layers: Vec<LayerBindsInt8>,
    final_cls: wgpu::BindGroup,
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
    /// int8 (DP4A) overlay; `Some` when the int8 decode path is active for this
    /// model. The f32 fields above stay built either way (the Stage-1 2× weight-
    /// VRAM shortcut), so toggling the path needs no rebuild of the f32 state.
    int8: Option<DecodeInt8>,
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
    /// Resident int8-relaid Q8_0 matmul weights for the DP4A decode path, keyed
    /// by the *same* source data pointer as [`weights`] (so the raw-Q8_0 and
    /// int8 copies of one weight coexist — the Stage-1 2× weight-VRAM shortcut).
    /// Same one-backend-per-model rule as `weights`.
    weights_i8: Mutex<HashMap<usize, GpuWeightI8>>,
    /// Resident RoPE inverse-frequency tables, keyed by source data pointer.
    tables: Mutex<HashMap<usize, Arc<wgpu::Buffer>>>,
    /// The granted device limits (used to pre-check oversized weight buffers).
    limits: wgpu::Limits,
    /// Resident decode state (KV cache + activations + bind groups), built lazily
    /// on the first [`Backend::forward_step`] and reused across decode steps.
    decode: Mutex<Option<DecodeState>>,
    /// Whether the int8 (DP4A) fused-decode path may be used when the model is
    /// Q8_0-eligible. Defaults from the `RUSTY_LLAMA_NO_INT8` env var (set =>
    /// disabled); override with [`set_int8_decode`] for A/B measurement & tests.
    int8_decode: bool,
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
            matmul_q4_k: make_pipeline(
                &device,
                &format!("{WGSL_QUANT_PRELUDE}{WGSL_MATMUL_Q4_K}"),
                "matmul_q4_k",
            ),
            matmul_q4_k_batch: make_pipeline(
                &device,
                &format!("{WGSL_QUANT_PRELUDE}{WGSL_MATMUL_Q4_K_BATCH}"),
                "matmul_q4_k_batch",
            ),
            matmul_q6_k: make_pipeline(
                &device,
                &format!("{WGSL_QUANT_PRELUDE}{WGSL_MATMUL_Q6_K}"),
                "matmul_q6_k",
            ),
            matmul_q6_k_batch: make_pipeline(
                &device,
                &format!("{WGSL_QUANT_PRELUDE}{WGSL_MATMUL_Q6_K_BATCH}"),
                "matmul_q6_k_batch",
            ),
            quantize_q8: make_pipeline(&device, WGSL_QUANTIZE_Q8, "quantize_q8"),
            matmul_q8_0_i8: make_pipeline(&device, WGSL_MATMUL_Q8_0_I8, "matmul_q8_0_i8"),
        };

        Ok(GpuBackend {
            device,
            queue,
            pipelines,
            weights: Mutex::new(HashMap::new()),
            weights_i8: Mutex::new(HashMap::new()),
            tables: Mutex::new(HashMap::new()),
            limits,
            decode: Mutex::new(None),
            int8_decode: std::env::var_os("RUSTY_LLAMA_NO_INT8").is_none(),
            adapter_name,
        })
    }

    /// The selected GPU adapter's name (for logging / benchmarks).
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Force the int8 (DP4A) decode path on or off, overriding the
    /// `RUSTY_LLAMA_NO_INT8` env default. Drops any cached decode state so the
    /// next step rebuilds with the new setting — the seam A/B benchmarks and
    /// tests use to compare int8 vs f32 on one model without env races.
    pub fn set_int8_decode(&mut self, on: bool) {
        self.int8_decode = on;
        *self.decode.lock().unwrap() = None;
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
                ty: ty @ (GgmlType::Q8_0 | GgmlType::Q4_K | GgmlType::Q6_K),
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

    /// Fetch (relaying + uploading on first use) the int8 (DP4A) buffers for a
    /// Q8_0 weight `w`: contiguous int8 values + per-32-block f32 scales, both
    /// cached by source pointer in [`weights_i8`]. Only valid for
    /// `QMatrix::Quant { ty: Q8_0, .. }` — the eligibility gate guarantees the
    /// caller never asks for anything else.
    fn weight_buffer_i8(&self, w: &QMatrix) -> GpuWeightI8 {
        let (data, rows, cols) = match w {
            QMatrix::Quant {
                ty: GgmlType::Q8_0,
                data,
                rows,
                cols,
                ..
            } => (data, *rows, *cols),
            _ => panic!("weight_buffer_i8 requires a Q8_0 weight"),
        };
        let key = data.as_ptr() as usize;
        if let Some(g) = self.weights_i8.lock().unwrap().get(&key) {
            return g.clone();
        }

        let (wq, wscale) = q8_0_relayout(data, rows, cols / 32);

        // Mirror weight_buffer's per-buffer limit check; the int8 values are the
        // larger of the two buffers (rows*cols bytes vs rows*cols/32 f32).
        let nbytes = wq.len() as u64;
        let binding_cap = self.limits.max_storage_buffer_binding_size;
        if nbytes > binding_cap || nbytes > self.limits.max_buffer_size {
            panic!(
                "int8 weight matrix ({rows} rows × {cols} cols → {nbytes} bytes on device) \
                 exceeds this GPU's limits (max_storage_buffer_binding_size {binding_cap}, \
                 max_buffer_size {}); this model is too large for the int8 GPU path",
                self.limits.max_buffer_size,
            );
        }

        let gw = GpuWeightI8 {
            q: Arc::new(self.storage_ro("weight_i8", &wq)),
            scale: Arc::new(self.storage_ro("weight_i8_scale", f32_bytes(&wscale))),
        };
        self.weights_i8.lock().unwrap().insert(key, gw.clone());
        gw
    }

    /// The matmul pipeline for an on-device weight format.
    fn matmul_pipeline(&self, ty: GgmlType) -> &wgpu::ComputePipeline {
        match ty {
            GgmlType::Q8_0 => &self.pipelines.matmul_q8_0,
            GgmlType::Q4_K => &self.pipelines.matmul_q4_k,
            GgmlType::Q6_K => &self.pipelines.matmul_q6_k,
            _ => &self.pipelines.matmul, // F32 (incl. host-dequantized weights)
        }
    }

    /// The batched matmul pipeline for an on-device weight format.
    fn matmul_batch_pipeline(&self, ty: GgmlType) -> &wgpu::ComputePipeline {
        match ty {
            GgmlType::Q8_0 => &self.pipelines.matmul_q8_0_batch,
            GgmlType::Q4_K => &self.pipelines.matmul_q4_k_batch,
            GgmlType::Q6_K => &self.pipelines.matmul_q6_k_batch,
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

        // Constant matmul params: [rows, cols, acc]. acc=1 folds the residual
        // add into the matmul output (mo and m2 accumulate into x).
        let p_dimdim = self.storage_ro("p", u32_bytes(&[dim as u32, dim as u32, 0]));
        let p_dimdim_acc = self.storage_ro("p", u32_bytes(&[dim as u32, dim as u32, 1]));
        let p_kvdim = self.storage_ro("p", u32_bytes(&[kv_dim as u32, dim as u32, 0]));
        let p_hidden = self.storage_ro("p", u32_bytes(&[hidden as u32, dim as u32, 0]));
        let p_dimhidden_acc = self.storage_ro("p", u32_bytes(&[dim as u32, hidden as u32, 1]));
        let p_vocab = self.storage_ro("p", u32_bytes(&[vocab as u32, dim as u32, 0]));
        let p_rms = self.storage_ro("p", u32_bytes(&[dim as u32, p.rms_eps.to_bits()]));
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
                    // Residual add folded in: mo accumulates wo·attn into x.
                    mo: mm(&wo, &x, &p_dimdim_acc),
                    rms_ffn: self.bind(&self.pipelines.rmsnorm, &[&x, &rms_f, &xb, &p_rms]),
                    m1: mm(&w1, &hb, &p_hidden),
                    m3: mm(&w3, &hb2, &p_hidden),
                    swiglu: self.bind(&self.pipelines.swiglu, &[&hb, &hb2, &p_swiglu]),
                    // w2 reads hb (not xb) and accumulates w2·swiglu into x.
                    m2: self.bind(
                        self.matmul_pipeline(w2.ty),
                        &[&w2.buf, &hb, &x, &p_dimhidden_acc],
                    ),
                    mm_ty: [wq.ty, wk.ty, wv.ty, wo.ty, w1.ty, w3.ty, w2.ty],
                }
            })
            .collect();

        let rms_final = self.table_buffer(&w.rms_final_weight);
        let wcls = self.weight_buffer(&w.wcls);
        let final_rms = self.bind(&self.pipelines.rmsnorm, &[&x, &rms_final, &xb, &p_rms]);
        let final_cls = self.bind(self.matmul_pipeline(wcls.ty), &[&wcls.buf, &xb, &logits, &p_vocab]);
        let final_cls_ty = wcls.ty;

        // int8 (DP4A) overlay: built only when enabled and every matmul weight is
        // Q8_0. Shares the f32 resident buffers and const params; adds resident
        // int8 activations + relaid int8 weights (the Stage-1 2× weight-VRAM
        // shortcut keeps the f32 weights too). The int8 activation buffers and
        // the shared f32 buffers are kept alive by the bind groups built here, so
        // none need their own field on DecodeInt8.
        let int8 = (self.int8_decode && model_q8_0_eligible(model)).then(|| {
            let xb_i8 = self.buffer("xb_i8", dim / 4, U::STORAGE);
            let xb_scale = self.buffer("xb_scale", dim / 32, U::STORAGE);
            let hb_i8 = self.buffer("hb_i8", hidden / 4, U::STORAGE);
            let hb_scale = self.buffer("hb_scale", hidden / 32, U::STORAGE);
            let p_q_xb = self.storage_ro("p_q", u32_bytes(&[(dim / 32) as u32]));
            let p_q_hb = self.storage_ro("p_q", u32_bytes(&[(hidden / 32) as u32]));
            let quant_xb =
                self.bind(&self.pipelines.quantize_q8, &[&xb, &xb_i8, &xb_scale, &p_q_xb]);
            let quant_hb =
                self.bind(&self.pipelines.quantize_q8, &[&hb, &hb_i8, &hb_scale, &p_q_hb]);
            // int8 matmul bind order: [wq, wscale, aq, ascale, out, params].
            let mm8 = |w8: &GpuWeightI8,
                       aq: &wgpu::Buffer,
                       asc: &wgpu::Buffer,
                       out: &wgpu::Buffer,
                       params: &wgpu::Buffer| {
                self.bind(
                    &self.pipelines.matmul_q8_0_i8,
                    &[&w8.q, &w8.scale, aq, asc, out, params],
                )
            };
            let layers = (0..nl)
                .map(|l| {
                    let wq = self.weight_buffer_i8(&w.wq[l]);
                    let wk = self.weight_buffer_i8(&w.wk[l]);
                    let wv = self.weight_buffer_i8(&w.wv[l]);
                    let wo = self.weight_buffer_i8(&w.wo[l]);
                    let w1 = self.weight_buffer_i8(&w.w1[l]);
                    let w2 = self.weight_buffer_i8(&w.w2[l]);
                    let w3 = self.weight_buffer_i8(&w.w3[l]);
                    LayerBindsInt8 {
                        mq: mm8(&wq, &xb_i8, &xb_scale, &q, &p_dimdim),
                        mk: mm8(&wk, &xb_i8, &xb_scale, &k_tmp, &p_kvdim),
                        mv: mm8(&wv, &xb_i8, &xb_scale, &v_tmp, &p_kvdim),
                        // mo/m2 accumulate (acc=1) into x, folding the residual.
                        mo: mm8(&wo, &xb_i8, &xb_scale, &x, &p_dimdim_acc),
                        m1: mm8(&w1, &xb_i8, &xb_scale, &hb, &p_hidden),
                        m3: mm8(&w3, &xb_i8, &xb_scale, &hb2, &p_hidden),
                        m2: mm8(&w2, &hb_i8, &hb_scale, &x, &p_dimhidden_acc),
                    }
                })
                .collect();
            let wcls_i8 = self.weight_buffer_i8(&w.wcls);
            let final_cls = mm8(&wcls_i8, &xb_i8, &xb_scale, &logits, &p_vocab);
            DecodeInt8 {
                quant_xb,
                quant_hb,
                layers,
                final_cls,
            }
        });

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
            int8,
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
        // Cooperative matmuls dispatch one workgroup per output row; the
        // elementwise add/swiglu keep one thread per element.
        let (mm_dim, mm_kv, mm_hidden, mm_vocab) =
            (d.dim as u32, d.kv_dim as u32, d.hidden as u32, d.vocab as u32);
        let sw_wg = ceil_div(d.hidden, 64);
        let wg_rope = ceil_div(d.dim / 2, 64);
        let wg_heads = d.n_heads as u32;
        let q_xb_wg = ceil_div(d.dim / 32, 64);
        let q_hb_wg = ceil_div(d.hidden / 32, 64);
        let pos_off = (pos * d.kv_dim * 4) as u64;
        let kv_bytes = (d.kv_dim * 4) as u64;

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        if let Some(i8s) = &d.int8 {
            // int8 (DP4A) path: rmsnorm/rope/attention/swiglu stay f32 (reusing
            // the LayerBinds groups); the seven matmuls + classifier run on int8,
            // each preceded by an on-device quantize of its f32 input activation.
            // One op per pass() => wgpu inserts the read/write barriers the reused
            // xb_i8/hb_i8 buffers need; never coalesce two ops into one pass.
            let mm8 = &self.pipelines.matmul_q8_0_i8;
            let quant = &self.pipelines.quantize_q8;
            for l in 0..d.n_layers {
                let lb = &d.layers[l];
                let il = &i8s.layers[l];
                pass(&mut enc, &self.pipelines.rmsnorm, &lb.rms_att, 1);
                pass(&mut enc, quant, &i8s.quant_xb, q_xb_wg);
                pass(&mut enc, mm8, &il.mq, mm_dim);
                pass(&mut enc, mm8, &il.mk, mm_kv);
                pass(&mut enc, mm8, &il.mv, mm_kv);
                pass(&mut enc, &self.pipelines.rope, &lb.rope, wg_rope);
                enc.copy_buffer_to_buffer(&d.k_tmp, 0, &d.key[l], pos_off, kv_bytes);
                enc.copy_buffer_to_buffer(&d.v_tmp, 0, &d.value[l], pos_off, kv_bytes);
                pass(&mut enc, &self.pipelines.attention, &lb.attn, wg_heads);
                pass(&mut enc, quant, &i8s.quant_xb, q_xb_wg); // quantize attn output
                pass(&mut enc, mm8, &il.mo, mm_dim); // folds residual add into x
                pass(&mut enc, &self.pipelines.rmsnorm, &lb.rms_ffn, 1);
                pass(&mut enc, quant, &i8s.quant_xb, q_xb_wg);
                pass(&mut enc, mm8, &il.m1, mm_hidden);
                pass(&mut enc, mm8, &il.m3, mm_hidden);
                pass(&mut enc, &self.pipelines.swiglu, &lb.swiglu, sw_wg);
                pass(&mut enc, quant, &i8s.quant_hb, q_hb_wg);
                pass(&mut enc, mm8, &il.m2, mm_dim); // folds residual add into x
            }
            pass(&mut enc, &self.pipelines.rmsnorm, &d.final_rms, 1);
            pass(&mut enc, quant, &i8s.quant_xb, q_xb_wg);
            pass(&mut enc, mm8, &i8s.final_cls, mm_vocab);
        } else {
            for l in 0..d.n_layers {
                let lb = &d.layers[l];
                let mm = lb.mm_ty.map(|t| self.matmul_pipeline(t)); // [wq,wk,wv,wo,w1,w3,w2]
                pass(&mut enc, &self.pipelines.rmsnorm, &lb.rms_att, 1);
                pass(&mut enc, mm[0], &lb.mq, mm_dim);
                pass(&mut enc, mm[1], &lb.mk, mm_kv);
                pass(&mut enc, mm[2], &lb.mv, mm_kv);
                pass(&mut enc, &self.pipelines.rope, &lb.rope, wg_rope);
                // Stash the rotated K/V for this position into the resident cache.
                enc.copy_buffer_to_buffer(&d.k_tmp, 0, &d.key[l], pos_off, kv_bytes);
                enc.copy_buffer_to_buffer(&d.v_tmp, 0, &d.value[l], pos_off, kv_bytes);
                pass(&mut enc, &self.pipelines.attention, &lb.attn, wg_heads);
                pass(&mut enc, mm[3], &lb.mo, mm_dim); // folds residual add into x
                pass(&mut enc, &self.pipelines.rmsnorm, &lb.rms_ffn, 1);
                pass(&mut enc, mm[4], &lb.m1, mm_hidden);
                pass(&mut enc, mm[5], &lb.m3, mm_hidden);
                pass(&mut enc, &self.pipelines.swiglu, &lb.swiglu, sw_wg);
                pass(&mut enc, mm[6], &lb.m2, mm_dim); // folds residual add into x
            }
            pass(&mut enc, &self.pipelines.rmsnorm, &d.final_rms, 1);
            pass(&mut enc, self.matmul_pipeline(d.final_cls_ty), &d.final_cls, mm_vocab);
        }
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
        let pb = self.params(&[w.rows() as u32, w.cols() as u32, 0]); // acc=0: overwrite
        let bind = self.bind(pipe, &[&gw.buf, &xb, &ob, &pb]);
        // Cooperative GEMV: one workgroup per output row.
        self.run(pipe, &bind, w.rows() as u32, &ob, out);
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

/// Re-lay a Q8_0 weight's raw 34-byte blocks into contiguous int8 values (as
/// bytes, 4 per `u32` for `dot4I8Packed`) + per-32-block f32 scales — the layout
/// `matmul_q8_0_i8` expects. `nb` is the number of 32-blocks per row (`cols/32`).
fn q8_0_relayout(wbytes: &[u8], rows: usize, nb: usize) -> (Vec<u8>, Vec<f32>) {
    let mut wq = Vec::with_capacity(rows * nb * 32);
    let mut wscale = Vec::with_capacity(rows * nb);
    for blk in 0..rows * nb {
        let off = blk * 34;
        wscale.push(crate::quant::f16_to_f32(u16::from_le_bytes([
            wbytes[off],
            wbytes[off + 1],
        ])));
        wq.extend_from_slice(&wbytes[off + 2..off + 34]);
    }
    (wq, wscale)
}

/// Whether `model` is eligible for the int8 (DP4A) decode path: every matmul
/// weight — the seven per layer plus the classifier — is Q8_0, and the two
/// quantized activation widths (`dim` for xb, `hidden` for hb) are multiples of
/// 32. (Q8_0 already forces `cols % 32 == 0`, so the dim checks only guard a
/// pathological model. `kv_dim`/`vocab` are matmul *output* row counts, never
/// quantized activation widths, so they have no divisibility constraint.) The
/// token-embedding table is excluded: it is dequantized on the host into the
/// per-step embedding, never run through an int8 matmul.
fn model_q8_0_eligible(model: &crate::model::Model) -> bool {
    let p = &model.config;
    if !p.dim.is_multiple_of(32) || !p.hidden_dim.is_multiple_of(32) {
        return false;
    }
    let w = &model.weights;
    let q8 = |m: &QMatrix| matches!(m, QMatrix::Quant { ty: GgmlType::Q8_0, .. });
    q8(&w.wcls)
        && (0..p.n_layers).all(|l| {
            q8(&w.wq[l])
                && q8(&w.wk[l])
                && q8(&w.wv[l])
                && q8(&w.wo[l])
                && q8(&w.w1[l])
                && q8(&w.w2[l])
                && q8(&w.w3[l])
        })
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

    /// Build `rows` valid Q4_K superblocks (256 cols each), varied per row.
    fn q4_k_weight(rows: usize) -> Vec<u8> {
        use crate::quant::{f32_to_f16, pack_scales_q4_k};
        let mut bytes = Vec::new();
        for r in 0..rows as u32 {
            bytes.extend_from_slice(&f32_to_f16(0.05).to_le_bytes());
            bytes.extend_from_slice(&f32_to_f16(0.02).to_le_bytes());
            bytes.extend_from_slice(&pack_scales_q4_k(
                [10, 20, 5, 33, 41, 7, 18, 25],
                [3, 9, 14, 1, 22, 6, 30, 11],
            ));
            (0..128u32).for_each(|i| bytes.push(((i * 7 + 3 + r) % 256) as u8));
        }
        bytes
    }

    /// Build `rows` valid Q6_K superblocks (256 cols each).
    fn q6_k_weight(rows: usize) -> Vec<u8> {
        use crate::quant::f32_to_f16;
        let mut bytes = Vec::new();
        for r in 0..rows as u32 {
            (0..128u32).for_each(|i| bytes.push(((i * 5 + 1 + r) % 256) as u8));
            (0..64u32).for_each(|i| bytes.push(((i * 9 + 2) % 256) as u8));
            (0..16i32).for_each(|i| bytes.push((i - 8) as i8 as u8));
            bytes.extend_from_slice(&f32_to_f16(0.03).to_le_bytes());
        }
        bytes
    }

    #[test]
    fn matmul_q4_k_matches_exact() {
        let Some(g) = gpu() else { return };
        let (rows, cols) = (3usize, 256usize);
        let bytes = q4_k_weight(rows);
        let deq = crate::quant::dequantize(GgmlType::Q4_K, &bytes, rows * cols).unwrap();
        let x = noise(cols, 13);
        let w = QMatrix::quant(GgmlType::Q4_K, bytes.into(), rows, cols).unwrap();
        let mut out = vec![0.0; rows];
        g.matmul(&mut out, &x, &w);
        let exact: Vec<f32> = (0..rows)
            .map(|r| (0..cols).map(|j| deq[r * cols + j] * x[j]).sum())
            .collect();
        close(&out, &exact);
    }

    #[test]
    fn matmul_q6_k_matches_exact() {
        let Some(g) = gpu() else { return };
        let (rows, cols) = (2usize, 256usize);
        let bytes = q6_k_weight(rows);
        let deq = crate::quant::dequantize(GgmlType::Q6_K, &bytes, rows * cols).unwrap();
        let x = noise(cols, 14);
        let w = QMatrix::quant(GgmlType::Q6_K, bytes.into(), rows, cols).unwrap();
        let mut out = vec![0.0; rows];
        g.matmul(&mut out, &x, &w);
        let exact: Vec<f32> = (0..rows)
            .map(|r| (0..cols).map(|j| deq[r * cols + j] * x[j]).sum())
            .collect();
        close(&out, &exact);
    }

    #[test]
    fn matmul_q4_k_batch_matches_exact() {
        let Some(g) = gpu() else { return };
        // Two batch rows of x against a 3-row Q4_K weight; checks the batch
        // kernel's x/out indexing (the dequant is shared with the single kernel).
        let (rows, cols, n) = (3usize, 256usize, 2usize);
        let bytes = q4_k_weight(rows);
        let deq = crate::quant::dequantize(GgmlType::Q4_K, &bytes, rows * cols).unwrap();
        let x = noise(n * cols, 15);
        let w = QMatrix::quant(GgmlType::Q4_K, bytes.into(), rows, cols).unwrap();
        let mut out = vec![0.0; n * rows];
        g.matmul_batch(&mut out, &x, &w, n);
        let mut exact = vec![0.0f32; n * rows];
        for r in 0..n {
            for o in 0..rows {
                exact[r * rows + o] = (0..cols).map(|j| deq[o * cols + j] * x[r * cols + j]).sum();
            }
        }
        close(&out, &exact);
    }

    // --- int8 DP4A decode path (Stage 1) --------------------------------

    #[test]
    fn q8_0_relayout_roundtrips() {
        let (rows, cols) = (3usize, 64usize);
        let nb = cols / 32;
        let wbytes = crate::quant::quantize_q8_0(&noise(rows * cols, 11));
        let (wq, wscale) = q8_0_relayout(&wbytes, rows, nb);
        assert_eq!(wq.len(), rows * cols);
        assert_eq!(wscale.len(), rows * nb);
        // int8 × per-block scale must reconstruct the same values as the
        // standard Q8_0 dequant (same int8s, same f16 scale, same arithmetic).
        let deq = crate::quant::dequantize(GgmlType::Q8_0, &wbytes, rows * cols).unwrap();
        for r in 0..rows {
            for b in 0..nb {
                for i in 0..32 {
                    let got = wq[(r * nb + b) * 32 + i] as i8 as f32 * wscale[r * nb + b];
                    let want = deq[r * cols + b * 32 + i];
                    assert!((got - want).abs() <= 1e-6 * want.abs().max(1.0) + 1e-9);
                }
            }
        }
    }

    #[test]
    fn weight_buffer_i8_caches_and_sizes() {
        let Some(g) = gpu() else { return };
        let (rows, cols) = (4usize, 64usize);
        let nb = cols / 32;
        let wbytes = crate::quant::quantize_q8_0(&noise(rows * cols, 12));
        let w = QMatrix::quant(GgmlType::Q8_0, wbytes.into(), rows, cols).unwrap();
        let a = g.weight_buffer_i8(&w);
        let b = g.weight_buffer_i8(&w);
        assert_eq!(a.q.size(), (rows * cols) as u64);
        assert_eq!(a.scale.size(), (rows * nb * 4) as u64);
        assert!(Arc::ptr_eq(&a.q, &b.q), "second call should hit the cache");
    }

    #[test]
    fn matmul_q8_0_i8_matches_int_dot() {
        let Some(g) = gpu() else { return };
        // Feed a host-quantized activation so the kernel is checked against the
        // exact integer dot (no GPU-quantize round-mode ambiguity).
        let (rows, cols) = (5usize, 256usize);
        let nb = cols / 32;
        let wbytes = crate::quant::quantize_q8_0(&noise(rows * cols, 7));
        let (wq, wscale) = q8_0_relayout(&wbytes, rows, nb);
        let act = crate::quant::quantize_activation_q8(&noise(cols, 8));
        let aq: &[u8] =
            unsafe { std::slice::from_raw_parts(act.qs.as_ptr() as *const u8, act.qs.len()) };

        // Reference: exact integer dot, scaled, in the same per-block order.
        let mut want = vec![0.0f32; rows];
        for (r, w) in want.iter_mut().enumerate() {
            for b in 0..nb {
                let dot: i32 = (0..32)
                    .map(|i| wq[(r * nb + b) * 32 + i] as i8 as i32 * act.qs[b * 32 + i] as i32)
                    .sum();
                *w += wscale[r * nb + b] * act.scales[b] * dot as f32;
            }
        }

        let wqb = g.storage_ro("wq", &wq);
        let wsb = g.storage_ro("wscale", f32_bytes(&wscale));
        let aqb = g.storage_ro("aq", aq);
        let asb = g.storage_ro("ascale", f32_bytes(&act.scales));
        let ob = g.alloc_out(rows);
        let pb = g.params(&[rows as u32, cols as u32, 0]);
        let bind = g.bind(
            &g.pipelines.matmul_q8_0_i8,
            &[&wqb, &wsb, &aqb, &asb, &ob, &pb],
        );
        let mut got = vec![0.0f32; rows];
        g.run_grid(&g.pipelines.matmul_q8_0_i8, &bind, [rows as u32, 1, 1], &ob, &mut got);
        close(&got, &want);
    }

    #[test]
    fn matmul_q8_0_i8_acc_folds_residual() {
        // The acc=1 branch (out[row] += dot) is what mo/m2 use to fold the
        // residual into x; the matches_int_dot test only covers acc=0.
        let Some(g) = gpu() else { return };
        let (rows, cols) = (4usize, 256usize);
        let nb = cols / 32;
        let wbytes = crate::quant::quantize_q8_0(&noise(rows * cols, 17));
        let (wq, wscale) = q8_0_relayout(&wbytes, rows, nb);
        let act = crate::quant::quantize_activation_q8(&noise(cols, 18));
        let aq: &[u8] =
            unsafe { std::slice::from_raw_parts(act.qs.as_ptr() as *const u8, act.qs.len()) };
        let prior: Vec<f32> = (0..rows).map(|r| r as f32 * 0.5 - 1.0).collect();

        // Reference: the pre-seeded value plus the exact integer dot.
        let mut want = prior.clone();
        for (r, w) in want.iter_mut().enumerate() {
            for b in 0..nb {
                let dot: i32 = (0..32)
                    .map(|i| wq[(r * nb + b) * 32 + i] as i8 as i32 * act.qs[b * 32 + i] as i32)
                    .sum();
                *w += wscale[r * nb + b] * act.scales[b] * dot as f32;
            }
        }

        let wqb = g.storage_ro("wq", &wq);
        let wsb = g.storage_ro("wscale", f32_bytes(&wscale));
        let aqb = g.storage_ro("aq", aq);
        let asb = g.storage_ro("ascale", f32_bytes(&act.scales));
        let ob = g.storage_rw("out", f32_bytes(&prior)); // seeded + readable back
        let pb = g.params(&[rows as u32, cols as u32, 1]); // acc=1
        let bind = g.bind(&g.pipelines.matmul_q8_0_i8, &[&wqb, &wsb, &aqb, &asb, &ob, &pb]);
        let mut got = vec![0.0f32; rows];
        g.run_grid(&g.pipelines.matmul_q8_0_i8, &bind, [rows as u32, 1, 1], &ob, &mut got);
        close(&got, &want);
    }

    #[test]
    fn quantize_q8_kernel_is_valid() {
        let Some(g) = gpu() else { return };
        let n = 256usize;
        let x = noise(n, 21);
        let host = crate::quant::quantize_activation_q8(&x);
        // GPU quantize.
        let xb = g.storage_ro("x", f32_bytes(&x));
        let qb = g.buffer("q", n / 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let sb = g.buffer(
            "s",
            n / 32,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let pb = g.params(&[(n / 32) as u32]);
        let bind = g.bind(&g.pipelines.quantize_q8, &[&xb, &qb, &sb, &pb]);
        g.dispatch(&g.pipelines.quantize_q8, &bind, ceil_div(n / 32, 64));
        let mut qbytes = vec![0.0f32; n / 4]; // read u32s as f32 bits then reinterpret
        g.read_back(&qb, &mut qbytes);
        let mut scales = vec![0.0f32; n / 32];
        g.read_back(&sb, &mut scales);
        // int8 values within ±1 of host (round-to-even vs away), scales close.
        let gpu_q: Vec<i8> = qbytes
            .iter()
            .flat_map(|w| w.to_bits().to_le_bytes())
            .map(|b| b as i8)
            .collect();
        for (i, (&hq, &gq)) in host.qs.iter().zip(&gpu_q).enumerate() {
            assert!((hq as i32 - gq as i32).abs() <= 1, "idx {i}: host {hq} gpu {gq}");
        }
        for (hs, gs) in host.scales.iter().zip(&scales) {
            assert!((hs - gs).abs() <= 1e-6 * hs.abs().max(1.0) + 1e-9);
        }
    }

    #[test]
    fn int8_decode_gate_and_toggle() {
        let Some(mut g) = gpu() else { return };
        // A Model borrows its Gguf, which borrows its bytes, so keep all three
        // alive per model (can't return a Model from a helper).
        let c = crate::Config {
            dim: 32,
            hidden_dim: 64,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            vocab_size: 64,
            seq_len: 16,
            shared_weights: true,
            ..Default::default()
        };
        let q8_bytes = crate::dummy::synthetic_gguf_typed(&c, GgmlType::Q8_0);
        let q8_gguf = crate::Gguf::parse(&q8_bytes).unwrap();
        let q8 = crate::Model::from_gguf(&q8_gguf).unwrap();
        let f_bytes = crate::dummy::synthetic_gguf_typed(&c, GgmlType::F32);
        let f_gguf = crate::Gguf::parse(&f_bytes).unwrap();
        let f = crate::Model::from_gguf(&f_gguf).unwrap();

        // Q8_0 + int8 enabled (default) => the overlay is built.
        assert!(g.build_decode_state(&q8).int8.is_some(), "Q8_0 should enable int8");
        // F32 weights are ineligible => no overlay.
        assert!(g.build_decode_state(&f).int8.is_none(), "F32 is ineligible for int8");
        // Toggle off => no overlay even for an eligible Q8_0 model.
        g.set_int8_decode(false);
        assert!(g.build_decode_state(&q8).int8.is_none(), "toggle disables int8");
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

    // Prototype: does a DP4A int8 GEMV beat the f32 dequant GEMV for Q8_0?
    // Self-contained (ad-hoc pipeline); decides whether to build out roadmap #2.
    #[test]
    #[ignore = "timing benchmark; run with --release --features gpu -- --ignored --nocapture"]
    fn bench_q8_0_int8_vs_f32_gemv() {
        use std::time::Instant;
        let Some(g) = gpu() else { return };
        let (rows, cols, iters) = (4096usize, 4096usize, 50);
        let wf = noise(rows * cols, 99);
        let wbytes = crate::quant::quantize_q8_0(&wf);
        let x = noise(cols, 100);

        // f32 path: the existing cooperative dequant GEMV (matmul_q8_0).
        let w = QMatrix::quant(GgmlType::Q8_0, wbytes.clone().into(), rows, cols).unwrap();
        let mut out_f32 = vec![0.0f32; rows];
        g.matmul(&mut out_f32, &x, &w); // warm: upload resident weight
        let t = Instant::now();
        for _ in 0..iters {
            g.matmul(&mut out_f32, &x, &w);
        }
        let f32_t = t.elapsed();

        // int8 path: re-lay Q8_0 to contiguous int8 (4-aligned) + f32 scales,
        // quantize the activation to int8 + scales, dot with dot4I8Packed.
        let nb = cols / 32;
        let rb = 34 * nb;
        let mut wq = Vec::<u8>::with_capacity(rows * cols);
        let mut wscale = Vec::<f32>::with_capacity(rows * nb);
        for r in 0..rows {
            for b in 0..nb {
                let off = r * rb + b * 34;
                wscale.push(crate::quant::f16_to_f32(u16::from_le_bytes([
                    wbytes[off],
                    wbytes[off + 1],
                ])));
                wq.extend_from_slice(&wbytes[off + 2..off + 34]);
            }
        }
        let act = crate::quant::quantize_activation_q8(&x);
        let aq: &[u8] =
            unsafe { std::slice::from_raw_parts(act.qs.as_ptr() as *const u8, act.qs.len()) };

        let src = r#"
struct P { rows: u32, cols: u32 };
@group(0) @binding(0) var<storage, read> wq: array<u32>;
@group(0) @binding(1) var<storage, read> wscale: array<f32>;
@group(0) @binding(2) var<storage, read> aq: array<u32>;
@group(0) @binding(3) var<storage, read> ascale: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
@group(0) @binding(5) var<storage, read> p: P;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    if (row >= p.rows) { return; }
    let tid = lid.x;
    let nb = p.cols / 32u;
    let wbase = row * (p.cols / 4u);
    let sbase = row * nb;
    var acc = 0.0;
    for (var b = tid; b < nb; b = b + 64u) {
        var dot: i32 = 0;
        let wu = wbase + b * 8u;
        let au = b * 8u;
        for (var k = 0u; k < 8u; k = k + 1u) {
            dot = dot + dot4I8Packed(wq[wu + k], aq[au + k]);
        }
        acc = acc + wscale[sbase + b] * ascale[b] * f32(dot);
    }
    red[tid] = acc;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
        if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
        workgroupBarrier();
    }
    if (tid == 0u) { outp[row] = red[0]; }
}
"#;
        let pipe = make_pipeline(&g.device, src, "q8_0_i8");
        let wq_buf = g.storage_ro("wq", &wq);
        let wscale_buf = g.storage_ro("wscale", f32_bytes(&wscale));
        let aq_buf = g.storage_ro("aq", aq);
        let ascale_buf = g.storage_ro("ascale", f32_bytes(&act.scales));
        let out_buf = g.alloc_out(rows);
        let pb = g.params(&[rows as u32, cols as u32]);
        let bind = g.bind(
            &pipe,
            &[&wq_buf, &wscale_buf, &aq_buf, &ascale_buf, &out_buf, &pb],
        );
        let mut out_i8 = vec![0.0f32; rows];
        g.run_grid(&pipe, &bind, [rows as u32, 1, 1], &out_buf, &mut out_i8);
        let t = Instant::now();
        for _ in 0..iters {
            g.run_grid(&pipe, &bind, [rows as u32, 1, 1], &out_buf, &mut out_i8);
        }
        let i8_t = t.elapsed();

        // int8 uses the quantized activation, f32 uses exact x — expect close.
        let maxdiff = out_f32
            .iter()
            .zip(&out_i8)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "Q8_0 GEMV {rows}x{cols} x{iters}: f32={:.3}ms/op int8(DP4A)={:.3}ms/op \
             -> int8 is {:.2}x the f32 time (maxdiff={maxdiff:.4})",
            f32_t.as_secs_f64() * 1e3 / iters as f64,
            i8_t.as_secs_f64() * 1e3 / iters as f64,
            i8_t.as_secs_f64() / f32_t.as_secs_f64(),
        );
    }

    #[test]
    #[ignore = "timing benchmark; run with --release --features gpu -- --ignored --nocapture"]
    fn bench_dispatch_overhead() {
        use std::time::Instant;
        let Some(g) = gpu() else { return };
        // A trivial 1-workgroup pass, to isolate the fixed per-compute-pass cost.
        let a = g.buffer("a", 64, wgpu::BufferUsages::STORAGE);
        let b = g.buffer("b", 64, wgpu::BufferUsages::STORAGE);
        let pp = g.params(&[64u32]);
        let bind = g.bind(&g.pipelines.add, &[&a, &b, &pp]);
        let run = |k: usize| {
            let mut enc = g
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            for _ in 0..k {
                pass(&mut enc, &g.pipelines.add, &bind, 1);
            }
            g.queue.submit(Some(enc.finish()));
            g.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        };
        run(308); // warm
        let reps = 50;
        let timeit = |k: usize| {
            let t = Instant::now();
            for _ in 0..reps {
                run(k);
            }
            t.elapsed().as_secs_f64() / reps as f64
        };
        let t1 = timeit(1);
        let t308 = timeit(308);
        let per_pass_us = (t308 - t1) / 307.0 * 1e6;
        eprintln!(
            "per-pass overhead: 1 pass={:.1}us, 308 passes={:.1}us -> ~{:.1}us/pass; \
             a 308-dispatch decode token spends ~{:.2}ms here (TinyLlama decode ~40ms/token)",
            t1 * 1e6,
            t308 * 1e6,
            per_pass_us,
            per_pass_us * 308.0 / 1000.0,
        );
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
