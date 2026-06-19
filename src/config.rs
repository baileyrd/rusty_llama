//! Model hyper-parameters and checkpoint header parsing.

use std::f64::consts::PI;

use crate::arch::Arch;
use crate::error::{Error, Result};

/// RoPE long-context scaling, as carried in GGUF `{arch}.rope.scaling.*`.
///
/// Each variant rewrites the per-dimension rotary frequencies (and, for YaRN,
/// an attention magnitude scale) at load time; see [`Config::rope_table`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RopeScaling {
    /// No scaling — the plain `base^(-2j/d)` frequencies.
    #[default]
    None,
    /// Linear (a.k.a. "position interpolation"): divide every frequency by
    /// `factor`, equivalently stretch positions by `factor`.
    Linear { factor: f32 },
    /// Llama-3.1 piecewise rescaling: low-frequency dims are divided by
    /// `factor`, high-frequency dims are kept, and a band in between is
    /// smoothly interpolated.
    Llama3 {
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        orig_ctx: usize,
    },
    /// YaRN (NTK-by-parts): per-dim blend of extrapolated and interpolated
    /// frequencies plus an attention magnitude scale folded into RoPE.
    Yarn {
        factor: f32,
        orig_ctx: usize,
        /// Extra attention scale (GGUF `rope.scaling.attn_factor`, usually 1).
        attn_factor: f32,
        /// NTK ramp endpoints (llama.cpp defaults: 32 / 1).
        beta_fast: f32,
        beta_slow: f32,
    },
}

/// Precomputed RoPE rotation table: one inverse frequency per rotated pair,
/// plus a scalar magnitude applied to the rotation (1.0 unless YaRN).
///
/// `inv_freq.len()` is the number of *rotated* pairs (`rope_dim / 2`), which is
/// `head_size / 2` for the usual full-rotary models and fewer for partial
/// rotary — the backend leaves the remaining pairs untouched.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeTable {
    /// Inverse frequency per rotated `(even, odd)` pair.
    pub inv_freq: Vec<f32>,
    /// Magnitude scale folded into `cos`/`sin` (YaRN's `mscale`; else 1.0).
    pub mscale: f32,
}

/// Transformer hyper-parameters.
///
/// The first eight fields mirror the llama2.c checkpoint header
/// (`dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len`, plus
/// the `shared_weights` flag that the format smuggles into the *sign* of
/// `vocab_size`). The last two are real-model knobs that GGUF carries
/// explicitly and that llama2.c leaves implicit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// Model (a.k.a. embedding / residual stream) dimension.
    pub dim: usize,
    /// Inner dimension of the feed-forward (SwiGLU) block.
    pub hidden_dim: usize,
    /// Number of transformer blocks.
    pub n_layers: usize,
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of key/value heads (`< n_heads` enables grouped-query attention).
    pub n_kv_heads: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length the KV-cache is sized for.
    pub seq_len: usize,
    /// Whether the output classifier reuses the token-embedding matrix.
    pub shared_weights: bool,
    /// RoPE base frequency θ (10000 for Llama-2; 500000 for Llama-3; 1e6 for Qwen2).
    pub rope_freq_base: f32,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
    /// Number of dimensions per head that RoPE rotates. `0` means "full
    /// rotary" (`= head_size`); a smaller value selects partial rotary, leaving
    /// the trailing dimensions of each head unrotated.
    pub rope_dim: usize,
    /// RoPE long-context frequency scaling (none for short-context models).
    pub rope_scaling: RopeScaling,
    /// Model architecture (selects the graph variant and tensor set).
    pub arch: Arch,
    /// Explicit attention head dimension. `0` means "derive from `dim / n_heads`"
    /// (Llama / Qwen2 / Phi-3); Gemma sets it independently, so `n_heads *
    /// head_dim` need not equal `dim`.
    pub head_dim: usize,
    /// Scale applied to the input token embeddings (`1.0`, except Gemma = `sqrt(dim)`).
    pub embd_scale: f32,
    /// tanh softcap on attention scores before softmax (`0.0` = off; Gemma2 = 50).
    pub attn_logit_softcap: f32,
    /// tanh softcap on the final output logits (`0.0` = off; Gemma2 = 30).
    pub final_logit_softcap: f32,
    /// Number of mixture-of-experts experts per FFN block. `0` = a dense FFN
    /// (every arch so far); `> 0` selects the routed MoE path (Mixtral).
    pub n_expert: usize,
    /// Number of experts each token is routed to (top-k). Must be in
    /// `1..=n_expert` when `n_expert > 0`; ignored for dense models.
    pub n_expert_used: usize,
    /// FFN intermediate dim of each routed expert. Defaults to `hidden_dim`
    /// (Mixtral); Qwen2-MoE sets a smaller `expert_feed_forward_length`.
    pub n_ff_exp: usize,
    /// FFN intermediate dim of the always-on shared expert. `0` = no shared
    /// expert (Mixtral); `> 0` = Qwen2-MoE's `expert_shared_feed_forward_length`.
    pub n_ff_shexp: usize,
    /// Whether the selected top-k routing weights are renormalized to sum to 1.
    /// `true` for Mixtral (`norm_w`), `false` for Qwen2-MoE.
    pub expert_weights_norm: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            dim: 0,
            hidden_dim: 0,
            n_layers: 0,
            n_heads: 0,
            n_kv_heads: 0,
            vocab_size: 0,
            seq_len: 0,
            shared_weights: true,
            rope_freq_base: 10000.0,
            rms_eps: 1e-5,
            rope_dim: 0,
            rope_scaling: RopeScaling::None,
            arch: Arch::Llama,
            head_dim: 0,
            embd_scale: 1.0,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
            n_expert: 0,
            n_expert_used: 0,
            n_ff_exp: 0,
            n_ff_shexp: 0,
            expert_weights_norm: true,
        }
    }
}

impl Config {
    /// Number of bytes occupied by the on-disk header (7 × `int32`).
    pub const HEADER_BYTES: usize = 7 * 4;

    /// Dimension of a single attention head.
    #[inline]
    pub fn head_size(&self) -> usize {
        if self.head_dim != 0 {
            self.head_dim
        } else {
            self.dim / self.n_heads
        }
    }

    /// Total width of the key/value projections (`n_kv_heads * head_size`).
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.head_size() * self.n_kv_heads
    }

    /// Total width of the query projection (`n_heads * head_size`). Equals `dim`
    /// for Llama-style models but differs when `head_dim` is set (Gemma).
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_size()
    }

    /// How many query heads share each key/value head.
    #[inline]
    pub fn kv_mul(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    /// Whether this model is eligible for the fused, device-resident
    /// decode/prefill fast path (CUDA/GPU). Only dense Llama-graph models
    /// qualify; MoE models (Mixtral is `Arch::Llama` but routed) and the breadth
    /// archs fall back to the generic per-op path.
    #[inline]
    pub fn uses_resident_decode(&self) -> bool {
        self.arch.uses_resident_decode() && self.n_expert == 0
    }

    /// Number of dimensions RoPE rotates per head (`rope_dim`, or the full
    /// `head_size` when `rope_dim == 0`).
    #[inline]
    pub fn rotary_dim(&self) -> usize {
        if self.rope_dim == 0 {
            self.head_size()
        } else {
            self.rope_dim
        }
    }

    /// Precompute the RoPE rotation table for this config (see [`RopeTable`]).
    ///
    /// The base frequency of rotated pair `j` is `θ^(-2j/rotary_dim)`; the
    /// configured [`RopeScaling`] then rewrites those frequencies (and, for
    /// YaRN, the magnitude scale). The math mirrors llama.cpp's `rope_yarn` /
    /// HF's `_compute_*_parameters`.
    pub fn rope_table(&self) -> RopeTable {
        let rot = self.rotary_dim();
        let half = rot / 2;
        let base = self.rope_freq_base as f64;
        let mut inv_freq: Vec<f32> = (0..half)
            .map(|j| base.powf(-2.0 * j as f64 / rot as f64) as f32)
            .collect();
        let mut mscale = 1.0f32;

        match self.rope_scaling {
            RopeScaling::None => {}
            RopeScaling::Linear { factor } => {
                let s = 1.0 / factor as f64;
                for f in &mut inv_freq {
                    *f = (*f as f64 * s) as f32;
                }
            }
            RopeScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                orig_ctx,
            } => {
                let (factor, lo, hi) =
                    (factor as f64, low_freq_factor as f64, high_freq_factor as f64);
                let low_wavelen = orig_ctx as f64 / lo;
                let high_wavelen = orig_ctx as f64 / hi;
                for f in &mut inv_freq {
                    let freq = *f as f64;
                    let wavelen = 2.0 * PI / freq;
                    *f = if wavelen < high_wavelen {
                        // High frequency: extrapolate (unchanged).
                        freq as f32
                    } else if wavelen > low_wavelen {
                        // Low frequency: interpolate (divide by factor).
                        (freq / factor) as f32
                    } else {
                        // Smooth blend across the medium band.
                        let smooth = (orig_ctx as f64 / wavelen - lo) / (hi - lo);
                        ((1.0 - smooth) * freq / factor + smooth * freq) as f32
                    };
                }
            }
            RopeScaling::Yarn {
                factor,
                orig_ctx,
                attn_factor,
                beta_fast,
                beta_slow,
            } => {
                let freq_scale = 1.0 / factor as f64;
                let n_dims = rot as f64;
                // Correction range: which pair indices to blend (NTK-by-parts).
                let corr = |n_rot: f64| {
                    n_dims * (orig_ctx as f64 / (n_rot * 2.0 * PI)).ln() / (2.0 * base.ln())
                };
                let low = corr(beta_fast as f64).floor().max(0.0);
                let high = corr(beta_slow as f64).ceil().min(n_dims - 1.0);
                let denom = (high - low).max(0.001);
                for (j, f) in inv_freq.iter_mut().enumerate() {
                    let base_inv = *f as f64;
                    // ramp = 1 at the high-freq end (extrapolate), 0 at the low
                    // end (interpolate by freq_scale).
                    let ramp = 1.0 - ((j as f64 - low) / denom).clamp(0.0, 1.0);
                    let mult = freq_scale * (1.0 - ramp) + ramp;
                    *f = (base_inv * mult) as f32;
                }
                // Magnitude correction for the interpolation (ext_factor != 0).
                mscale = attn_factor * (1.0 + 0.1 * (factor as f64).ln()) as f32;
            }
        }

        RopeTable { inv_freq, mscale }
    }

    /// Parse the 7-`int32` little-endian header at the start of a checkpoint.
    pub fn read_header(bytes: &[u8]) -> Result<Config> {
        if bytes.len() < Self::HEADER_BYTES {
            return Err(Error::Format(format!(
                "checkpoint is only {} bytes, need at least {} for the header",
                bytes.len(),
                Self::HEADER_BYTES
            )));
        }
        let read = |i: usize| -> i32 {
            i32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]])
        };

        let dim = read(0);
        let hidden_dim = read(4);
        let n_layers = read(8);
        let n_heads = read(12);
        let n_kv_heads = read(16);
        let vocab_raw = read(20);
        let seq_len = read(24);

        // llama2.c signals "classifier shares the embedding table" with a
        // positive vocab_size, and "separate classifier weights" with a
        // negative one.
        let shared_weights = vocab_raw > 0;

        let cfg = Config {
            dim: nonneg(dim, "dim")?,
            hidden_dim: nonneg(hidden_dim, "hidden_dim")?,
            n_layers: nonneg(n_layers, "n_layers")?,
            n_heads: nonneg(n_heads, "n_heads")?,
            n_kv_heads: nonneg(n_kv_heads, "n_kv_heads")?,
            vocab_size: vocab_raw.unsigned_abs() as usize,
            seq_len: nonneg(seq_len, "seq_len")?,
            shared_weights,
            // llama2.c implies these; use the standard Llama-2 defaults.
            ..Default::default()
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Sanity-check the invariants the inference code relies on.
    pub fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(Error::Format(m));
        if self.dim == 0 || self.n_heads == 0 || self.n_kv_heads == 0 || self.vocab_size == 0 {
            return bad("dim/n_heads/n_kv_heads/vocab_size must be non-zero".into());
        }
        // When the head dim is derived from `dim` (Llama-style) it must divide
        // evenly; an explicit `head_dim` (Gemma) decouples the two.
        if self.head_dim == 0 && !self.dim.is_multiple_of(self.n_heads) {
            return bad(format!(
                "dim ({}) must be divisible by n_heads ({})",
                self.dim, self.n_heads
            ));
        }
        if !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return bad(format!(
                "n_heads ({}) must be divisible by n_kv_heads ({})",
                self.n_heads, self.n_kv_heads
            ));
        }
        if !self.head_size().is_multiple_of(2) {
            return bad(format!(
                "head_size ({}) must be even for RoPE",
                self.head_size()
            ));
        }
        if self.n_expert > 0 && !(1..=self.n_expert).contains(&self.n_expert_used) {
            return bad(format!(
                "n_expert_used ({}) must be in 1..={} for an MoE model",
                self.n_expert_used, self.n_expert
            ));
        }
        let positive = |v: f32| v.is_finite() && v > 0.0;
        if !positive(self.rope_freq_base) || !positive(self.rms_eps) {
            return bad(format!(
                "rope_freq_base ({}) and rms_eps ({}) must be finite and positive",
                self.rope_freq_base, self.rms_eps
            ));
        }
        Ok(())
    }
}

fn nonneg(v: i32, name: &str) -> Result<usize> {
    if v <= 0 {
        Err(Error::Format(format!("{name} must be positive, got {v}")))
    } else {
        Ok(v as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A config whose head is 8-wide (one head), giving 4 rotated pairs with
    /// the tidy base frequencies `[1, 0.1, 0.01, 0.001]` at θ = 10000.
    fn cfg(rope_scaling: RopeScaling) -> Config {
        Config {
            dim: 8,
            hidden_dim: 8,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            vocab_size: 8,
            seq_len: 16,
            rope_freq_base: 10000.0,
            rope_scaling,
            ..Default::default()
        }
    }

    /// Assert two frequency tables match within f32 tolerance.
    fn approx_slice(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "length: {got:?} vs {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!(
                (g - w).abs() <= 1e-6 * w.abs().max(1.0) + 1e-9,
                "got {g}, want {w}"
            );
        }
    }

    #[test]
    fn rope_table_no_scaling_is_base_freqs() {
        let t = cfg(RopeScaling::None).rope_table();
        approx_slice(&t.inv_freq, &[1.0, 0.1, 0.01, 0.001]);
        assert_eq!(t.mscale, 1.0);
    }

    #[test]
    fn rope_table_linear_divides_freqs() {
        // factor 4 -> every frequency / 4.
        let t = cfg(RopeScaling::Linear { factor: 4.0 }).rope_table();
        approx_slice(&t.inv_freq, &[0.25, 0.025, 0.0025, 0.00025]);
        assert_eq!(t.mscale, 1.0);
    }

    #[test]
    fn rope_table_llama3_piecewise() {
        // Tuned so all three branches fire: j0/j1 kept (high freq), j2 smoothly
        // blended (medium band), j3 divided by factor (low freq). Expected
        // values cross-checked against the HF llama3 formula in Python.
        let t = cfg(RopeScaling::Llama3 {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 8.0,
            orig_ctx: 2000,
        })
        .rope_table();
        approx_slice(&t.inv_freq, &[1.0, 0.1, 0.003_978_873_6, 0.000_125]);
        assert_eq!(t.mscale, 1.0);
    }

    #[test]
    fn rope_table_yarn_ntk_and_mscale() {
        // factor 4 over a 2048 original context: j0/j1 extrapolate (kept), j2 is
        // blended, j3 fully interpolated (* freq_scale = 1/4). mscale folds in
        // the magnitude correction. Values cross-checked in Python.
        let t = cfg(RopeScaling::Yarn {
            factor: 4.0,
            orig_ctx: 2048,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        })
        .rope_table();
        approx_slice(&t.inv_freq, &[1.0, 0.1, 0.00625, 0.00025]);
        assert!((t.mscale - 1.138_629_4).abs() < 1e-6, "mscale {}", t.mscale);
    }

    #[test]
    fn rope_table_partial_rotary_shortens_table() {
        // Rotate only 4 of the 8 head dims -> 2 pairs, frequencies base^(-2j/4).
        let mut c = cfg(RopeScaling::None);
        c.rope_dim = 4;
        let t = c.rope_table();
        approx_slice(&t.inv_freq, &[1.0, 0.01]);
    }
}
