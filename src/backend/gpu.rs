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
use crate::quant::dequantize;
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
    workgroupBarrier();

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

/// The compiled compute pipelines, one per primitive.
struct Pipelines {
    matmul: wgpu::ComputePipeline,
    rmsnorm: wgpu::ComputePipeline,
    add: wgpu::ComputePipeline,
    swiglu: wgpu::ComputePipeline,
    rope: wgpu::ComputePipeline,
    attention: wgpu::ComputePipeline,
}

/// A wgpu-backed [`Backend`].
pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Pipelines,
    /// Resident matmul weights (f32), keyed by source data pointer.
    weights: Mutex<HashMap<usize, Arc<wgpu::Buffer>>>,
    /// Resident RoPE inverse-frequency tables, keyed by source data pointer.
    tables: Mutex<HashMap<usize, Arc<wgpu::Buffer>>>,
    /// Human-readable adapter name, for logging.
    adapter_name: String,
}

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

        let pipelines = Pipelines {
            matmul: make_pipeline(&device, WGSL_MATMUL, "matmul"),
            rmsnorm: make_pipeline(&device, WGSL_RMSNORM, "rmsnorm"),
            add: make_pipeline(&device, WGSL_ADD, "add"),
            swiglu: make_pipeline(&device, WGSL_SWIGLU, "swiglu"),
            rope: make_pipeline(&device, WGSL_ROPE, "rope"),
            attention: make_pipeline(&device, WGSL_ATTENTION, "attention"),
        };

        Ok(GpuBackend {
            device,
            queue,
            pipelines,
            weights: Mutex::new(HashMap::new()),
            tables: Mutex::new(HashMap::new()),
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

    /// Dispatch `pipeline` over `workgroups` workgroups (1-D), then copy
    /// `out_buf` back into `out`. Compute + copy share a single submission.
    fn run(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind: &wgpu::BindGroup,
        workgroups: u32,
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
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        enc.copy_buffer_to_buffer(out_buf, 0, &staging, 0, byte_len);
        self.queue.submit(Some(enc.finish()));
        self.map_read(&staging, out);
    }

    /// Dispatch `pipeline` without reading anything back (the op wrote into
    /// resident buffers that are read back separately).
    fn dispatch(&self, pipeline: &wgpu::ComputePipeline, bind: &wgpu::BindGroup, workgroups: u32) {
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
            cpass.dispatch_workgroups(workgroups, 1, 1);
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

    /// Fetch (uploading on first use) the resident f32 buffer for weight `w`.
    /// Quantized weights are dequantized to f32 on the host once, here.
    fn weight_buffer(&self, w: &QMatrix) -> Arc<wgpu::Buffer> {
        let (key, f32data): (usize, Cow<[f32]>) = match w {
            QMatrix::F32 { data, .. } => (data.as_ptr() as usize, Cow::Borrowed(&data[..])),
            QMatrix::Quant {
                ty,
                data,
                rows,
                cols,
            } => {
                let key = data.as_ptr() as usize;
                // Avoid the dequant if it's already resident.
                if let Some(b) = self.weights.lock().unwrap().get(&key) {
                    return b.clone();
                }
                let f = dequantize(*ty, data, rows * cols).expect("weight dequantization");
                (key, Cow::Owned(f))
            }
        };
        let mut cache = self.weights.lock().unwrap();
        if let Some(b) = cache.get(&key) {
            return b.clone();
        }
        let buf = Arc::new(self.storage_ro("weight", f32_bytes(&f32data)));
        cache.insert(key, buf.clone());
        buf
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
        let wb = self.weight_buffer(w);
        let xb = self.storage_ro("x", f32_bytes(x));
        let ob = self.alloc_out(out.len());
        let pb = self.params(&[w.rows() as u32, w.cols() as u32]);
        let bind = self.bind(&self.pipelines.matmul, &[&wb, &xb, &ob, &pb]);
        self.run(&self.pipelines.matmul, &bind, ceil_div(w.rows(), 64), &ob, out);
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
}
