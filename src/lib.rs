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

pub mod backend;
pub mod config;
pub mod dummy;
pub mod error;
pub mod gguf;
mod math;
pub mod model;
pub mod quant;
pub mod sampler;
pub mod tensor;
pub mod tokenizer;

mod loader;

pub use backend::{Backend, CpuBackend};
pub use config::Config;
pub use error::{Error, Result};
pub use gguf::Gguf;
pub use loader::Checkpoint;
pub use model::{forward, generate, Model, RunState, Weights};
pub use quant::GgmlType;
pub use sampler::Sampler;
pub use tensor::QMatrix;
pub use tokenizer::Tokenizer;
