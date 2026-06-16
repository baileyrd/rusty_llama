//! SentencePiece-style BPE tokenizer, matching the llama2.c `tokenizer.bin`.
//!
//! The file format is:
//! ```text
//! int32  max_token_length
//! repeat vocab_size times:
//!     float32  score
//!     int32    length
//!     u8[length] piece bytes
//! ```
//! Tokens 0/1/2 are `<unk>`/`<s>`(BOS)/`</s>`(EOS); ids 3..=258 are the raw
//! byte fallbacks `<0x00>`..=`<0xFF>`.

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Error, Result};
use crate::gguf::{Gguf, MetaValue};

/// A trained BPE vocabulary plus the merge logic to apply it.
pub struct Tokenizer {
    vocab: Vec<Vec<u8>>,
    scores: Vec<f32>,
    lookup: HashMap<Vec<u8>, usize>,
    #[allow(dead_code)]
    max_token_length: usize,
}

impl Tokenizer {
    /// Build a tokenizer directly from pieces and their merge scores.
    ///
    /// Mostly useful for tests and synthetic models; real models go through
    /// [`Tokenizer::load`].
    pub fn from_vocab(vocab: Vec<Vec<u8>>, scores: Vec<f32>) -> Self {
        assert_eq!(vocab.len(), scores.len());
        let max_token_length = vocab.iter().map(Vec::len).max().unwrap_or(0);
        let lookup = vocab
            .iter()
            .enumerate()
            .map(|(i, piece)| (piece.clone(), i))
            .collect();
        Tokenizer {
            vocab,
            scores,
            lookup,
            max_token_length,
        }
    }

    /// Load a `tokenizer.bin` file containing exactly `vocab_size` entries.
    pub fn load<P: AsRef<Path>>(path: P, vocab_size: usize) -> Result<Self> {
        let data = std::fs::read(path)?;
        let mut p = 0usize;

        let read_i32 = |data: &[u8], p: &mut usize| -> Result<i32> {
            let end = *p + 4;
            if end > data.len() {
                return Err(Error::Format("tokenizer.bin truncated".into()));
            }
            let v = i32::from_le_bytes(data[*p..end].try_into().unwrap());
            *p = end;
            Ok(v)
        };
        let read_f32 = |data: &[u8], p: &mut usize| -> Result<f32> {
            let end = *p + 4;
            if end > data.len() {
                return Err(Error::Format("tokenizer.bin truncated".into()));
            }
            let v = f32::from_le_bytes(data[*p..end].try_into().unwrap());
            *p = end;
            Ok(v)
        };

        let max_token_length = read_i32(&data, &mut p)?.max(0) as usize;
        let mut vocab = Vec::with_capacity(vocab_size);
        let mut scores = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            let score = read_f32(&data, &mut p)?;
            let len = read_i32(&data, &mut p)?;
            if len < 0 || p + len as usize > data.len() {
                return Err(Error::Format("tokenizer.bin truncated".into()));
            }
            let len = len as usize;
            vocab.push(data[p..p + len].to_vec());
            scores.push(score);
            p += len;
        }

        let mut tk = Tokenizer::from_vocab(vocab, scores);
        tk.max_token_length = max_token_length;
        Ok(tk)
    }

    /// Build a tokenizer from a GGUF file's embedded SentencePiece vocabulary.
    ///
    /// Reads `tokenizer.ggml.tokens` and `tokenizer.ggml.scores`, converting the
    /// SentencePiece whitespace marker `▁` (U+2581) to a plain space so the same
    /// encode/decode logic as the llama2.c `tokenizer.bin` path applies. This
    /// handles `llama`/SPM tokenizers (Llama-2, TinyLlama); byte-level BPE
    /// (`gpt2`) tokenizers are not yet supported.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        if let Ok(model) = gguf.meta_str("tokenizer.ggml.model") {
            if model != "llama" {
                return Err(Error::Format(format!(
                    "unsupported GGUF tokenizer model '{model}' (only 'llama'/SPM)"
                )));
            }
        }
        let tokens = match gguf.meta("tokenizer.ggml.tokens") {
            Some(MetaValue::Array(a)) => a,
            _ => return Err(Error::Format("GGUF missing tokenizer.ggml.tokens".into())),
        };
        let vocab: Vec<Vec<u8>> = tokens
            .iter()
            .map(|t| {
                t.as_str()
                    .map(normalize_spm_piece)
                    .ok_or_else(|| Error::Format("non-string token in GGUF vocab".into()))
            })
            .collect::<Result<_>>()?;

        let scores = match gguf.meta("tokenizer.ggml.scores") {
            Some(MetaValue::Array(a)) => a.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect(),
            _ => vec![0.0; vocab.len()], // some models omit scores
        };

        Ok(Tokenizer::from_vocab(vocab, scores))
    }

    /// Encode `text` into token ids, optionally bracketing with BOS/EOS.
    ///
    /// Follows SentencePiece's defaults: a dummy leading space is prepended,
    /// bytes that aren't in the vocabulary fall back to byte tokens, and pairs
    /// are greedily merged by descending score.
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<usize> {
        let mut tokens: Vec<usize> = Vec::new();

        if bos {
            tokens.push(1);
        }
        // SentencePiece prepends a space to the input (`add_dummy_prefix`).
        if !text.is_empty() {
            if let Some(&id) = self.lookup.get(b" ".as_slice()) {
                tokens.push(id);
            }
        }

        // Map each Unicode scalar to a token, or fall back to its raw bytes.
        let mut buf = [0u8; 4];
        for ch in text.chars() {
            let piece = ch.encode_utf8(&mut buf).as_bytes();
            if let Some(&id) = self.lookup.get(piece) {
                tokens.push(id);
            } else {
                // Byte fallback: byte value `b` lives at vocab id `b + 3`.
                // Real Llama vocabularies always contain those tokens; guard
                // anyway so a small/odd vocab can't index out of bounds.
                for &b in piece {
                    let id = b as usize + 3;
                    tokens.push(if id < self.vocab.len() { id } else { 0 });
                }
            }
        }

        // Greedily merge the highest-scoring adjacent pair until none remain.
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best = None;
            for i in 0..tokens.len().saturating_sub(1) {
                let mut merged = self.vocab[tokens[i]].clone();
                merged.extend_from_slice(&self.vocab[tokens[i + 1]]);
                if let Some(&id) = self.lookup.get(&merged) {
                    if self.scores[id] > best_score {
                        best_score = self.scores[id];
                        best = Some((i, id));
                    }
                }
            }
            match best {
                Some((i, id)) => {
                    tokens[i] = id;
                    tokens.remove(i + 1);
                }
                None => break,
            }
        }

        if eos {
            tokens.push(2);
        }
        tokens
    }

    /// Decode `token` (preceded by `prev_token`) into the bytes to emit.
    ///
    /// Strips the leading space SentencePiece adds after BOS, and turns raw
    /// byte tokens (`<0xXX>`) back into the single byte they represent.
    pub fn decode(&self, prev_token: usize, token: usize) -> Vec<u8> {
        let piece = &self.vocab[token];
        let mut bytes: &[u8] = piece;
        // After BOS, drop a single leading space (sentencepiece convention).
        if prev_token == 1 && bytes.first() == Some(&b' ') {
            bytes = &bytes[1..];
        }
        if let Some(byte) = parse_byte_token(bytes) {
            return vec![byte];
        }
        bytes.to_vec()
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// Convert a SentencePiece piece to the byte form this tokenizer uses,
/// replacing the `▁` whitespace marker (U+2581) with a regular space.
fn normalize_spm_piece(s: &str) -> Vec<u8> {
    s.replace('\u{2581}', " ").into_bytes()
}

/// Parse a `<0xXX>` raw-byte token into its byte value, if it is one.
fn parse_byte_token(bytes: &[u8]) -> Option<u8> {
    if bytes.len() == 6 && bytes.starts_with(b"<0x") && bytes.last() == Some(&b'>') {
        let hi = (bytes[3] as char).to_digit(16)?;
        let lo = (bytes[4] as char).to_digit(16)?;
        Some((hi * 16 + lo) as u8)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small hand-built vocabulary whose merges compose into " hello".
    fn hello_tokenizer() -> Tokenizer {
        let entries: &[(&[u8], f32)] = &[
            (b"<unk>", 0.0),    // 0
            (b"\n<s>\n", 0.0),  // 1 BOS
            (b"\n</s>\n", 0.0), // 2 EOS
            (b" ", -3.0),       // 3
            (b"h", -4.0),       // 4
            (b"e", -5.0),       // 5
            (b"l", -6.0),       // 6
            (b"o", -7.0),       // 7
            (b"he", 5.0),       // 8
            (b"ll", 4.0),       // 9
            (b"hello", 10.0),   // 10
            (b" hello", 12.0),  // 11
            (b"hell", 8.0),     // 12
        ];
        let vocab = entries.iter().map(|(p, _)| p.to_vec()).collect();
        let scores = entries.iter().map(|(_, s)| *s).collect();
        Tokenizer::from_vocab(vocab, scores)
    }

    #[test]
    fn encode_merges_into_single_token() {
        let tk = hello_tokenizer();
        // BOS, dummy-space, then "hello" greedily merges all the way up to
        // " hello" (id 11).
        assert_eq!(tk.encode("hello", true, false), vec![1, 11]);
    }

    #[test]
    fn decode_strips_space_after_bos() {
        let tk = hello_tokenizer();
        // " hello" (id 11) loses its leading space when it follows BOS.
        assert_eq!(tk.decode(1, 11), b"hello");
        // Without BOS context, the space stays.
        assert_eq!(tk.decode(7, 11), b" hello");
    }

    #[test]
    fn roundtrip_via_decode() {
        let tk = hello_tokenizer();
        let ids = tk.encode("hello", true, false);
        let mut out = Vec::new();
        let mut prev = ids[0];
        for &id in &ids[1..] {
            out.extend_from_slice(&tk.decode(prev, id));
            prev = id;
        }
        assert_eq!(out, b"hello");
    }

    #[test]
    fn spm_marker_becomes_space() {
        assert_eq!(normalize_spm_piece("\u{2581}the"), b" the");
        assert_eq!(normalize_spm_piece("of"), b"of");
    }

    #[test]
    fn byte_token_decodes_to_raw_byte() {
        let tk = Tokenizer::from_vocab(vec![b"<0x41>".to_vec()], vec![0.0]);
        assert_eq!(tk.decode(0, 0), b"A"); // 0x41 == 'A'
        assert_eq!(parse_byte_token(b"<0x0a>"), Some(b'\n'));
        assert_eq!(parse_byte_token(b"hello"), None);
    }
}
