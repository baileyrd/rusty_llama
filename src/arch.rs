//! Architecture registry seam (Phase 0.2).
//!
//! Maps a GGUF `general.architecture` string to a known [`Arch`] and its
//! canonical GGUF tensor-name table, so the loader ([`crate::Model::from_gguf`])
//! reads tensor names through one table instead of inline string literals.
//!
//! Llama is the only architecture today; Phase 3 hangs Qwen2 / Gemma / Phi off
//! this seam — each a new [`Arch`] variant plus its own [`TensorNames`] table
//! (and, where the graph differs, a per-arch branch in the loader). Mirrors
//! llama.cpp's `LLM_ARCH_NAMES` / `LLM_TENSOR_NAMES` (`src/llama-arch.cpp`).

/// Supported model architectures. Mirrors llama.cpp's `enum llm_arch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    /// Llama-family transformer (RMSNorm, RoPE, GQA, SwiGLU). Also covers the
    /// architectures that share this exact tensor layout (Mistral, Qwen2, …).
    Llama,
}

/// Canonical GGUF tensor names for one architecture.
///
/// The `attn_*` / `ffn_*` fields are the **per-block suffixes** used as
/// `blk.{i}.{suffix}`; `token_embd` / `output` / `output_norm` are **top-level**
/// tensor names. Mirrors `LLM_TENSOR_NAMES`. All `'static` — zero allocation at
/// load except the per-block `format!`, exactly as before this seam.
pub struct TensorNames {
    pub token_embd: &'static str,
    pub output: &'static str,
    pub output_norm: &'static str,
    pub attn_norm: &'static str,
    pub attn_q: &'static str,
    pub attn_k: &'static str,
    pub attn_v: &'static str,
    pub attn_output: &'static str,
    pub ffn_norm: &'static str,
    pub ffn_gate: &'static str,
    pub ffn_up: &'static str,
    pub ffn_down: &'static str,
}

/// Llama-family tensor names (the literals previously inlined in `from_gguf`).
pub const LLAMA_NAMES: TensorNames = TensorNames {
    token_embd: "token_embd.weight",
    output: "output.weight",
    output_norm: "output_norm.weight",
    attn_norm: "attn_norm.weight",
    attn_q: "attn_q.weight",
    attn_k: "attn_k.weight",
    attn_v: "attn_v.weight",
    attn_output: "attn_output.weight",
    ffn_norm: "ffn_norm.weight",
    ffn_gate: "ffn_gate.weight",
    ffn_up: "ffn_up.weight",
    ffn_down: "ffn_down.weight",
};

impl Arch {
    /// Resolve a GGUF `general.architecture` string to an [`Arch`].
    ///
    /// Phase-0 policy: every string — recognized Llama-family or otherwise —
    /// resolves to [`Arch::Llama`], preserving today's permissive behavior (any
    /// GGUF with Llama-named tensors loads). Phase 3 adds real match arms and,
    /// once a second table exists, a rejection path for genuinely-unknown archs.
    pub fn from_name(_arch: &str) -> Self {
        Arch::Llama
    }

    /// The canonical tensor-name table for this architecture.
    pub fn tensor_names(&self) -> &'static TensorNames {
        match self {
            Arch::Llama => &LLAMA_NAMES,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_names_are_canonical() {
        // Guards a typo in the extracted table against the GGUF convention.
        assert_eq!(LLAMA_NAMES.token_embd, "token_embd.weight");
        assert_eq!(LLAMA_NAMES.output, "output.weight");
        assert_eq!(LLAMA_NAMES.output_norm, "output_norm.weight");
        assert_eq!(LLAMA_NAMES.attn_q, "attn_q.weight");
        assert_eq!(LLAMA_NAMES.attn_output, "attn_output.weight");
        assert_eq!(LLAMA_NAMES.ffn_gate, "ffn_gate.weight");
        assert_eq!(LLAMA_NAMES.ffn_down, "ffn_down.weight");
    }

    #[test]
    fn known_and_unknown_archs_resolve_to_llama() {
        // Documents the permissive Phase-0 policy.
        for a in ["llama", "qwen2", "mistral", "gemma", "totally-unknown"] {
            assert_eq!(Arch::from_name(a), Arch::Llama, "arch {a}");
        }
    }
}
