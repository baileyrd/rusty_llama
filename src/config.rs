//! Model hyper-parameters and checkpoint header parsing.

use crate::error::{Error, Result};

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
        }
    }
}

impl Config {
    /// Number of bytes occupied by the on-disk header (7 × `int32`).
    pub const HEADER_BYTES: usize = 7 * 4;

    /// Dimension of a single attention head.
    #[inline]
    pub fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }

    /// Total width of the key/value projections (`n_kv_heads * head_size`).
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.dim * self.n_kv_heads / self.n_heads
    }

    /// How many query heads share each key/value head.
    #[inline]
    pub fn kv_mul(&self) -> usize {
        self.n_heads / self.n_kv_heads
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
        if !self.dim.is_multiple_of(self.n_heads) {
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
