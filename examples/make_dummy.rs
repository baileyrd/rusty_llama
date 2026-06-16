//! Write a small, randomly-weighted checkpoint in the llama2.c format.
//!
//! Useful for trying the CLI end-to-end without downloading real weights:
//!
//! ```sh
//! cargo run --release --example make_dummy -- dummy.bin
//! ./scripts/download_assets.sh        # just for a real tokenizer.bin
//! cargo run --release -- dummy.bin -z tokenizer.bin -i "Once upon a time" -n 32
//! ```
//!
//! The vocabulary size matches the real Llama tokenizer (32000) so the output
//! decodes to genuine sub-words — it's just gibberish because the weights are
//! noise.

use rusty_llama::dummy::synthetic_checkpoint;
use rusty_llama::Config;

fn main() -> std::io::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "dummy.bin".into());

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

    let bytes = synthetic_checkpoint(&config);
    std::fs::write(&path, &bytes)?;
    println!("wrote {} ({} bytes) {config:?}", path, bytes.len());
    Ok(())
}
