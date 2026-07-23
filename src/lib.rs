//! `rusty_llama` — a small, from-scratch Llama inference engine.
//!
//! The crate reads [Karpathy llama2.c](https://github.com/karpathy/llama2.c)
//! checkpoints (`stories15M.bin` & friends) and runs them on the CPU. The
//! pieces fit together like this:
//!
//! ```text
//!  Checkpoint (mmap)  ──parse──▶  Model { Config, Weights }
//!                                      │
//!  Tokenizer.encode(prompt) ─tokens─▶  forward(token, pos) ─logits─▶ Sampler
//!                                      │  (uses a Backend)              │
//!                                      └──────────── KV cache ◀─────────┘
//! ```
//!
//! All heavy math goes through the [`Backend`] trait, so a GPU backend can be
//! added later without touching the model code.

// The on-disk format and the zero-copy weight views assume a little-endian
// host (every desktop/server CPU we target).
#[cfg(target_endian = "big")]
compile_error!("rusty_llama currently assumes a little-endian target");

pub mod arch;
pub mod adapter;
pub mod backend;
pub mod chat;
pub mod config;
pub mod dummy;
pub mod error;
pub mod gguf;
pub mod grammar;
mod math;
pub mod model;
pub mod quant;
pub mod sampler;
pub mod tensor;
pub mod tokenizer;

#[cfg(feature = "server")]
pub mod server;

mod loader;

#[cfg(test)]
mod bench_util;

pub use backend::{make_backend, Backend, CpuBackend};
pub use arch::Arch;
#[cfg(feature = "cuda")]
pub use backend::CudaBackend;
#[cfg(feature = "gpu")]
pub use backend::GpuBackend;
pub use grammar::{Grammar, GrammarStage};
pub use config::{Config, RopeScaling, RopeTable};
pub use error::{Error, Result};
pub use gguf::Gguf;
pub use loader::Checkpoint;
pub use chat::{ChatRenderer, ChatTemplate, Message, Role};
pub use adapter::{AdapterBackend, ControlVector, LoraAdapter};
pub use model::{
    forward, forward_embed, forward_prefill, generate, generate_tokens, Batch, Model, Pooling,
    RunState, Weights,
};
pub use quant::GgmlType;
pub use sampler::{SamplerChain, SamplerConfig};
pub use tensor::QMatrix;
pub use tokenizer::Tokenizer;
#[cfg(feature = "server")]
pub use server::serve;
