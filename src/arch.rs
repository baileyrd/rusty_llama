//! Architecture registry seam (Phase 0.2, extended for breadth in Phase 3.1).
//!
//! Maps a GGUF `general.architecture` string to a known [`Arch`], its canonical
//! GGUF tensor-name table, and the graph predicates that distinguish the
//! supported families (Llama, Qwen2, Phi-3, Gemma 2). The loader
//! ([`crate::Model::from_gguf`]) and the forward pass ([`crate::model::forward`])
//! read arch behaviour through these methods instead of inline string matches.
//!
//! Mirrors llama.cpp's `LLM_ARCH_NAMES` / `LLM_TENSOR_NAMES` (a single flat
//! name table) plus the per-arch graph branches in `llama-model.cpp`. We keep
//! only the families we validate end-to-end against `llama-cli` — explicitly not
//! llama.cpp's 132 archs.

/// FFN gate activation: SwiGLU (`silu(gate)·up`, Llama/Qwen2/Phi-3) or GeGLU
/// (`gelu(gate)·up`, Gemma).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnActivation {
    /// `silu(gate) * up`.
    SwiGlu,
    /// `gelu(gate) * up`.
    GeGlu,
}

/// Supported model architectures. Mirrors a subset of llama.cpp's `enum llm_arch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Arch {
    /// Llama-family transformer (RMSNorm, RoPE, GQA, SwiGLU). Also covers the
    /// architectures sharing this exact graph (Mistral, …) and is the permissive
    /// fallback for any unrecognized `general.architecture`.
    #[default]
    Llama,
    /// Qwen2: the Llama graph plus an additive bias on the Q/K/V projections.
    Qwen2,
    /// Phi-3: the Llama graph with a fused `attn_qkv` projection and a fused
    /// gate+up FFN tensor, both split into the standard projections at load time.
    Phi3,
    /// Gemma 2: GeGLU FFN, `(1+w)` RMSNorm, `sqrt(dim)` embedding scale, an
    /// explicit head dim, "sandwich" post-attention / post-FFN norms, and tanh
    /// logit softcapping on the attention scores and the final logits.
    Gemma2,
    /// Qwen2-MoE: the Qwen2 attention graph (QKV bias, NeoX rope) with a routed
    /// mixture-of-experts FFN plus an always-on shared expert. Selected by the
    /// `qwen2moe` arch string; routed experts use a separate `n_ff_exp`, the
    /// shared expert `n_ff_shexp`, and the top-k weights are NOT renormalized
    /// (unlike Mixtral).
    Qwen2Moe,
}

/// Canonical GGUF tensor names.
///
/// The `attn_*` / `ffn_*` fields are the **per-block suffixes** used as
/// `blk.{i}.{suffix}`; `token_embd` / `output` / `output_norm` are **top-level**
/// tensor names. A single flat table serves every [`Arch`] (mirroring modern
/// `LLM_TENSOR_NAMES`); *which* of these tensors an arch actually has is encoded
/// in the [`Arch`] predicates, not in separate per-arch tables. All `'static` —
/// zero allocation at load except the per-block `format!`.
pub struct TensorNames {
    pub token_embd: &'static str,
    pub output: &'static str,
    pub output_norm: &'static str,
    pub attn_norm: &'static str,
    pub attn_q: &'static str,
    pub attn_k: &'static str,
    pub attn_v: &'static str,
    /// Fused query+key+value projection (Phi-3), split at load.
    pub attn_qkv: &'static str,
    pub attn_output: &'static str,
    /// Q/K/V projection biases (Qwen2).
    pub attn_q_bias: &'static str,
    pub attn_k_bias: &'static str,
    pub attn_v_bias: &'static str,
    /// Post-attention "sandwich" norm (Gemma2).
    pub attn_post_norm: &'static str,
    pub ffn_norm: &'static str,
    pub ffn_gate: &'static str,
    pub ffn_up: &'static str,
    pub ffn_down: &'static str,
    /// Post-FFN "sandwich" norm (Gemma2).
    pub ffn_post_norm: &'static str,
    /// MoE router (gates tokens to experts) — a `(n_expert, dim)` projection.
    pub ffn_gate_inp: &'static str,
    /// MoE expert weight stacks — 3-D `(n_expert, *, *)` tensors split per
    /// expert at load. Present only when `config.n_expert > 0` (Mixtral).
    pub ffn_gate_exps: &'static str,
    pub ffn_up_exps: &'static str,
    pub ffn_down_exps: &'static str,
    /// Qwen2-MoE shared-expert tensors: a sigmoid gate (`ffn_gate_inp_shexp`, a
    /// length-`dim` vector → one scalar) and an always-on SwiGLU FFN at
    /// `n_ff_shexp`. Present only when the model carries a shared expert.
    pub ffn_gate_inp_shexp: &'static str,
    pub ffn_gate_shexp: &'static str,
    pub ffn_up_shexp: &'static str,
    pub ffn_down_shexp: &'static str,
}

/// The canonical GGUF tensor names (the literals previously inlined in
/// `from_gguf`, plus the bias / fused / sandwich-norm extras the breadth archs
/// reference). Names follow the `gguf-py` `TensorNameMap` convention.
pub const TENSOR_NAMES: TensorNames = TensorNames {
    token_embd: "token_embd.weight",
    output: "output.weight",
    output_norm: "output_norm.weight",
    attn_norm: "attn_norm.weight",
    attn_q: "attn_q.weight",
    attn_k: "attn_k.weight",
    attn_v: "attn_v.weight",
    attn_qkv: "attn_qkv.weight",
    attn_output: "attn_output.weight",
    attn_q_bias: "attn_q.bias",
    attn_k_bias: "attn_k.bias",
    attn_v_bias: "attn_v.bias",
    attn_post_norm: "post_attention_norm.weight",
    ffn_norm: "ffn_norm.weight",
    ffn_gate: "ffn_gate.weight",
    ffn_up: "ffn_up.weight",
    ffn_down: "ffn_down.weight",
    ffn_post_norm: "post_ffw_norm.weight",
    ffn_gate_inp: "ffn_gate_inp.weight",
    ffn_gate_exps: "ffn_gate_exps.weight",
    ffn_up_exps: "ffn_up_exps.weight",
    ffn_down_exps: "ffn_down_exps.weight",
    ffn_gate_inp_shexp: "ffn_gate_inp_shexp.weight",
    ffn_gate_shexp: "ffn_gate_shexp.weight",
    ffn_up_shexp: "ffn_up_shexp.weight",
    ffn_down_shexp: "ffn_down_shexp.weight",
};

impl Arch {
    /// Resolve a GGUF `general.architecture` string to an [`Arch`].
    ///
    /// Recognized breadth archs map to their variant; everything else falls back
    /// to [`Arch::Llama`], preserving the permissive policy that any GGUF with
    /// Llama-named tensors still loads.
    pub fn from_name(arch: &str) -> Self {
        match arch {
            "qwen2" => Arch::Qwen2,
            "phi3" => Arch::Phi3,
            "gemma2" => Arch::Gemma2,
            "qwen2moe" => Arch::Qwen2Moe,
            _ => Arch::Llama,
        }
    }

    /// The canonical tensor-name table (shared across archs).
    pub fn tensor_names(&self) -> &'static TensorNames {
        &TENSOR_NAMES
    }

    /// Additive bias on the Q/K/V projections (Qwen2 / Qwen2-MoE).
    pub fn has_qkv_bias(&self) -> bool {
        matches!(self, Arch::Qwen2 | Arch::Qwen2Moe)
    }

    /// A single fused `attn_qkv` tensor (split into Q/K/V at load) — Phi-3.
    pub fn fused_qkv(&self) -> bool {
        matches!(self, Arch::Phi3)
    }

    /// A single fused gate+up FFN tensor in `ffn_up` (split at load) — Phi-3.
    pub fn fused_gate_up(&self) -> bool {
        matches!(self, Arch::Phi3)
    }

    /// FFN gate activation for this arch.
    pub fn ffn_activation(&self) -> FfnActivation {
        match self {
            Arch::Gemma2 => FfnActivation::GeGlu,
            _ => FfnActivation::SwiGlu,
        }
    }

    /// Per-layer post-attention and post-FFN "sandwich" norms (Gemma2).
    pub fn sandwich_norm(&self) -> bool {
        matches!(self, Arch::Gemma2)
    }

    /// NeoX (half-split) RoPE, where rotary pairs are `(j, j + rot/2)` rather
    /// than the interleaved `(2j, 2j+1)` of NORM-style RoPE. Qwen2 / Phi-3 /
    /// Gemma use NeoX; Llama GGUFs use NORM (their converter pre-permutes Q/K so
    /// the interleaved kernel is correct). We keep one NORM rope kernel and
    /// permute NeoX Q/K weights at load instead (the inverse of that converter
    /// step), so no backend changes are needed.
    pub fn rope_neox(&self) -> bool {
        matches!(self, Arch::Qwen2 | Arch::Phi3 | Arch::Gemma2 | Arch::Qwen2Moe)
    }

    /// Only plain Llama uses the fused resident GPU/CUDA decode; every other arch
    /// runs the generic per-op path (which still dispatches GPU kernels per
    /// primitive — it just isn't the single fused on-device loop).
    pub fn uses_resident_decode(&self) -> bool {
        matches!(self, Arch::Llama)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_names_are_canonical() {
        // Guards a typo in the extracted table against the GGUF convention.
        assert_eq!(TENSOR_NAMES.token_embd, "token_embd.weight");
        assert_eq!(TENSOR_NAMES.output, "output.weight");
        assert_eq!(TENSOR_NAMES.output_norm, "output_norm.weight");
        assert_eq!(TENSOR_NAMES.attn_q, "attn_q.weight");
        assert_eq!(TENSOR_NAMES.attn_q_bias, "attn_q.bias");
        assert_eq!(TENSOR_NAMES.attn_qkv, "attn_qkv.weight");
        assert_eq!(TENSOR_NAMES.attn_output, "attn_output.weight");
        assert_eq!(TENSOR_NAMES.ffn_gate, "ffn_gate.weight");
        assert_eq!(TENSOR_NAMES.ffn_down, "ffn_down.weight");
        assert_eq!(TENSOR_NAMES.attn_post_norm, "post_attention_norm.weight");
        assert_eq!(TENSOR_NAMES.ffn_post_norm, "post_ffw_norm.weight");
        assert_eq!(TENSOR_NAMES.ffn_gate_inp_shexp, "ffn_gate_inp_shexp.weight");
        assert_eq!(TENSOR_NAMES.ffn_down_shexp, "ffn_down_shexp.weight");
    }

    #[test]
    fn known_archs_resolve_to_their_variant() {
        assert_eq!(Arch::from_name("llama"), Arch::Llama);
        assert_eq!(Arch::from_name("qwen2"), Arch::Qwen2);
        assert_eq!(Arch::from_name("phi3"), Arch::Phi3);
        assert_eq!(Arch::from_name("gemma2"), Arch::Gemma2);
        assert_eq!(Arch::from_name("qwen2moe"), Arch::Qwen2Moe);
        // Unknown / unsupported archs stay permissive (Llama tensor layout).
        for a in ["mistral", "gemma", "totally-unknown"] {
            assert_eq!(Arch::from_name(a), Arch::Llama, "arch {a}");
        }
    }

    #[test]
    fn arch_predicates() {
        assert!(Arch::Qwen2.has_qkv_bias() && !Arch::Llama.has_qkv_bias());
        assert!(Arch::Phi3.fused_qkv() && Arch::Phi3.fused_gate_up());
        assert_eq!(Arch::Gemma2.ffn_activation(), FfnActivation::GeGlu);
        assert_eq!(Arch::Llama.ffn_activation(), FfnActivation::SwiGlu);
        assert!(Arch::Gemma2.sandwich_norm() && !Arch::Llama.sandwich_norm());
        assert!(Arch::Llama.uses_resident_decode() && !Arch::Gemma2.uses_resident_decode());
        // Qwen2-MoE shares Qwen2 attention (QKV bias, NeoX rope).
        assert!(Arch::Qwen2Moe.has_qkv_bias() && Arch::Qwen2Moe.rope_neox());
    }
}
