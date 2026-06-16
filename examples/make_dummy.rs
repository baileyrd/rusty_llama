//! Write a small, randomly-weighted model for trying the CLI without
//! downloading real weights. Format is chosen by the file extension:
//! `.gguf` → GGUF (with an embedded dummy tokenizer), anything else → llama2.c
//! checkpoint (needs a separate `tokenizer.bin`). An optional second argument
//! picks the GGUF matmul weight type (`f32` default, or `q8_0`).
//!
//! ```sh
//! cargo run --release --example make_dummy -- dummy.gguf          # F32 GGUF
//! cargo run --release --example make_dummy -- dummy.gguf q8_0     # quantized
//! cargo run --release -- dummy.gguf -i "hi" -n 16
//!
//! cargo run --release --example make_dummy -- dummy.bin           # llama2.c
//! cargo run --release -- dummy.bin -z tokenizer.bin -i "Once upon a time"
//! ```
//!
//! The weights are noise, so the text is gibberish — this only proves the
//! pipeline runs.

use rusty_llama::dummy::{synthetic_checkpoint, synthetic_gguf_typed};
use rusty_llama::{Config, GgmlType};

fn main() -> std::io::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "dummy.gguf".into());
    let ty = match std::env::args().nth(2).as_deref() {
        Some("q8_0") | Some("q8") => GgmlType::Q8_0,
        _ => GgmlType::F32,
    };

    // dim / hidden_dim are multiples of 32 so quantized types are representable.
    let config = Config {
        dim: 64,
        hidden_dim: 192,
        n_layers: 2,
        n_heads: 8,
        n_kv_heads: 8,
        vocab_size: 32000,
        seq_len: 256,
        shared_weights: true,
        ..Default::default()
    };

    let bytes = if path.ends_with(".gguf") {
        synthetic_gguf_typed(&config, ty)
    } else {
        synthetic_checkpoint(&config)
    };
    std::fs::write(&path, &bytes)?;
    println!("wrote {} ({} bytes) {ty:?} {config:?}", path, bytes.len());
    Ok(())
}
