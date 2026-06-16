//! Write a small, randomly-weighted model for trying the CLI without
//! downloading real weights. The output format is chosen by the file
//! extension: `.gguf` → GGUF (with an embedded dummy tokenizer), anything else
//! → llama2.c checkpoint (needs a separate `tokenizer.bin`).
//!
//! ```sh
//! # GGUF (self-contained):
//! cargo run --release --example make_dummy -- dummy.gguf
//! cargo run --release -- dummy.gguf -i "hi" -n 16
//!
//! # llama2.c (bring your own tokenizer):
//! cargo run --release --example make_dummy -- dummy.bin
//! cargo run --release -- dummy.bin -z tokenizer.bin -i "Once upon a time" -n 32
//! ```
//!
//! The weights are noise, so the text is gibberish — this only proves the
//! pipeline runs.

use rusty_llama::dummy::{synthetic_checkpoint, synthetic_gguf};
use rusty_llama::Config;

fn main() -> std::io::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "dummy.gguf".into());

    let config = Config {
        dim: 64,
        hidden_dim: 176,
        n_layers: 2,
        n_heads: 8,
        n_kv_heads: 8,
        vocab_size: 32000,
        seq_len: 256,
        shared_weights: true,
    };

    let bytes = if path.ends_with(".gguf") {
        synthetic_gguf(&config)
    } else {
        synthetic_checkpoint(&config)
    };
    std::fs::write(&path, &bytes)?;
    println!("wrote {} ({} bytes) {config:?}", path, bytes.len());
    Ok(())
}
