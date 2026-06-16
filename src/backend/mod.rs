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

pub use cpu::CpuBackend;

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
    /// vector `k` (length `kv_dim`) for absolute position `pos`, using base
    /// frequency `theta` (10000 for Llama-2, 500000 for Llama-3, …).
    fn rope(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        pos: usize,
        head_size: usize,
        kv_dim: usize,
        theta: f32,
    );

    /// Grouped-query attention for the current position.
    ///
    /// For each head, scores the query against every cached key up to and
    /// including `pos`, softmaxes them, and writes the value-weighted sum into
    /// `out` (length `n_heads * head_size`). `att` is scratch of length
    /// `n_heads * seq_len`.
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
    fn swiglu(&self, hb: &mut [f32], hb2: &[f32]);

    /// Residual add: `out[i] += x[i]`.
    fn add(&self, out: &mut [f32], x: &[f32]);
}
