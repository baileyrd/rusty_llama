//! Compute backends.
//!
//! All of the heavy lifting in the forward pass is expressed in terms of the
//! [`Backend`] trait. The forward pass in [`crate::model`] never touches a raw
//! kernel directly, so swapping in a GPU implementation later is "just" a
//! matter of writing a new `impl Backend` — the orchestration stays the same.
//!
//! Today the only implementation is [`CpuBackend`], a multi-threaded f32
//! backend (rayon for parallelism, autovectorized inner loops for SIMD).

mod cpu;
#[cfg(feature = "gpu")]
mod gpu;

pub use cpu::CpuBackend;
#[cfg(feature = "gpu")]
pub use gpu::GpuBackend;

use crate::tensor::QMatrix;

/// The set of primitive operations the transformer is built from.
///
/// Everything operates on plain `f32` slices for now, except matmul weights
/// which are passed as a [`QMatrix`] so the kernel can dequantize on the fly.
/// Shapes are otherwise passed explicitly to keep the contract between the
/// model and the backend completely transparent.
pub trait Backend: Send + Sync {
    /// RMSNorm: `out = x / sqrt(mean(x²) + eps) * weight`.
    ///
    /// `out`, `x` and `weight` all have the same length.
    fn rmsnorm(&self, out: &mut [f32], x: &[f32], weight: &[f32], eps: f32);

    /// Matrix–vector product `out = W · x`.
    ///
    /// `w` is a `(rows, cols)` matrix (possibly quantized); `x` has length
    /// `w.cols()`, `out` has length `w.rows()`. This is *the* hot path.
    fn matmul(&self, out: &mut [f32], x: &[f32], w: &QMatrix);

    /// Apply rotary positional embeddings (RoPE) in place.
    ///
    /// Rotates the query vector `q` (length `n_heads * head_size`) and the key
    /// vector `k` (length `kv_dim`) for absolute position `pos`. `inv_freq`
    /// holds one inverse frequency per rotated `(even, odd)` pair within a head
    /// (`head_size / 2` for full rotary, fewer for partial rotary — the extra
    /// pairs are left untouched). `mscale` scales the rotation magnitude (1.0
    /// except under YaRN long-context scaling). The table is precomputed once
    /// at load time; see [`crate::config::Config::rope_table`].
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
    );

    /// Grouped-query attention for the current position.
    ///
    /// For each head, scores the query against every cached key up to and
    /// including `pos`, softmaxes them, and writes the value-weighted sum into
    /// `out` (length `n_heads * head_size`). `att` is caller-provided scratch of
    /// length `n_heads * seq_len`; its contents are unspecified after the call
    /// (the CPU backend leaves the softmaxed scores there, the GPU backend does
    /// not — callers must not rely on either).
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
    );

    /// SwiGLU gate: `hb[i] = silu(hb[i]) * hb2[i]`.
    ///
    /// Operates on the whole slice, so it doubles as the batched form: pass a
    /// `(rows, hidden_dim)` buffer flattened row-major.
    fn swiglu(&self, hb: &mut [f32], hb2: &[f32]);

    /// Residual add: `out[i] += x[i]`.
    ///
    /// Like [`Backend::swiglu`], elementwise over the whole slice, so it also
    /// serves as the batched residual add.
    fn add(&self, out: &mut [f32], x: &[f32]);

    // --- Batched (prefill) variants -------------------------------------
    //
    // These process `rows` token positions at once so a backend can fuse them
    // into a single large kernel (the GPU's sweet spot, and far fewer
    // host<->device round-trips than `rows` single-token calls). The default
    // implementations just loop the single-token op, so any backend is correct
    // for free and `forward_prefill` is exactly equivalent to `rows` sequential
    // `forward` calls; a backend overrides them only to go faster.

    /// Batched matmul: `x` is `(rows, w.cols())` row-major, `out` is
    /// `(rows, w.rows())`. Equivalent to calling [`Backend::matmul`] on each row.
    fn matmul_batch(&self, out: &mut [f32], x: &[f32], w: &QMatrix, rows: usize) {
        let (oc, ic) = (w.rows(), w.cols());
        for r in 0..rows {
            self.matmul(&mut out[r * oc..r * oc + oc], &x[r * ic..r * ic + ic], w);
        }
    }

    /// Batched RMSNorm: `out`/`x` are `(rows, dim)` row-major, normalized per
    /// row against the shared `weight` (length `dim`).
    fn rmsnorm_batch(&self, out: &mut [f32], x: &[f32], weight: &[f32], eps: f32, rows: usize) {
        let dim = weight.len();
        for r in 0..rows {
            self.rmsnorm(&mut out[r * dim..r * dim + dim], &x[r * dim..r * dim + dim], weight, eps);
        }
    }

    /// Batched RoPE: `q` is `(rows, q_dim)`, `k` is `(rows, kv_dim)`; row `r`
    /// rotates at absolute position `pos_base + r`.
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
        for r in 0..rows {
            self.rope(
                &mut q[r * q_dim..r * q_dim + q_dim],
                &mut k[r * kv_dim..r * kv_dim + kv_dim],
                pos_base + r,
                head_size,
                kv_dim,
                inv_freq,
                mscale,
            );
        }
    }

    /// Batched causal attention over `rows` query positions. Query row `r` (at
    /// absolute position `pos_base + r`) attends to cached keys `0..=pos_base+r`.
    /// `q`/`out` are `(rows, n_heads*head_size)`; `att` is single-row scratch of
    /// length `n_heads * seq_len`, reused across rows by the default impl.
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
    }
}
