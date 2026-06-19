//! Tokenizers.
//!
//! Two flavours, behind one [`Tokenizer`] enum:
//!
//! * [`Spm`] — the SentencePiece-style tokenizer used by llama2.c
//!   `tokenizer.bin` and `llama`-architecture GGUF files (Llama-2, TinyLlama).
//!   Greedy score-based merges with byte fallback.
//! * [`Bpe`] — GPT-2 style byte-level BPE used by `gpt2`-tokenizer GGUF files
//!   (Llama-3, Qwen2, …). Rank-based merges over a byte→unicode encoding.
//!
//! `tokenizer.bin` format (SPM):
//! ```text
//! int32  max_token_length
//! repeat vocab_size times: float32 score | int32 len | u8[len] piece
//! ```

use std::collections::{HashMap, HashSet};
use std::path::Path;

use fancy_regex::Regex;

use crate::error::{Error, Result};
use crate::gguf::{Gguf, MetaValue};

/// A tokenizer: either SentencePiece-style (SPM) or byte-level BPE.
pub enum Tokenizer {
    /// SentencePiece-style (llama2.c / `llama` GGUF). Boxed to keep the enum small.
    Spm(Box<Spm>),
    /// GPT-2 byte-level BPE (`gpt2` GGUF). Boxed — it's much larger than `Spm`.
    Bpe(Box<Bpe>),
}

impl Tokenizer {
    /// Build an SPM tokenizer directly from pieces and merge scores.
    pub fn from_vocab(vocab: Vec<Vec<u8>>, scores: Vec<f32>) -> Self {
        Tokenizer::Spm(Box::new(Spm::from_vocab(vocab, scores)))
    }

    /// Build a BPE tokenizer from encoded tokens and ordered merges.
    pub fn from_bpe(
        tokens: Vec<String>,
        merges: Vec<(String, String)>,
        bos: Option<usize>,
        eos: Option<usize>,
    ) -> Self {
        Tokenizer::Bpe(Box::new(Bpe::new(tokens, merges, bos, eos)))
    }

    /// Load an SPM `tokenizer.bin` with exactly `vocab_size` entries.
    pub fn load<P: AsRef<Path>>(path: P, vocab_size: usize) -> Result<Self> {
        Ok(Tokenizer::Spm(Box::new(Spm::load(path, vocab_size)?)))
    }

    /// Build a tokenizer from a GGUF file, dispatching on
    /// `tokenizer.ggml.model` (`llama` → SPM, `gpt2` → byte-level BPE).
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let model = gguf.meta_str("tokenizer.ggml.model").unwrap_or("llama");
        match model {
            "llama" => Ok(Tokenizer::Spm(Box::new(Spm::from_gguf(gguf)?))),
            "gpt2" => Ok(Tokenizer::Bpe(Box::new(Bpe::from_gguf(gguf)?))),
            other => Err(Error::Format(format!(
                "unsupported GGUF tokenizer model '{other}' (have 'llama'/SPM, 'gpt2'/BPE)"
            ))),
        }
    }

    /// Encode `text` into token ids, optionally bracketing with BOS/EOS.
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<usize> {
        match self {
            Tokenizer::Spm(t) => t.encode(text, bos, eos),
            Tokenizer::Bpe(t) => t.encode(text, bos, eos),
        }
    }

    /// Whether a BOS token should be prepended on encode (GGUF
    /// `tokenizer.ggml.add_bos_token`, default `true`). Llama/SPM models set
    /// `true`; Qwen2 sets `false`, so callers must not force a BOS for it.
    pub fn add_bos(&self) -> bool {
        match self {
            Tokenizer::Spm(t) => t.add_bos,
            Tokenizer::Bpe(t) => t.add_bos,
        }
    }

    /// The model's primary EOS token id (GGUF `eos_token_id`), for stopping
    /// generation. `None` if the GGUF declared none.
    pub fn eos(&self) -> Option<usize> {
        match self {
            Tokenizer::Spm(t) => t.eos,
            Tokenizer::Bpe(t) => t.eos,
        }
    }

    /// Whether `token` ends generation: the EOS or end-of-turn (`eot`) token.
    /// Used by the decode loop to stop at a turn boundary (e.g. Qwen2
    /// `<|im_end|>`, Gemma `<end_of_turn>`, Phi-3 `<|end|>`).
    pub fn is_eog(&self, token: usize) -> bool {
        let (eos, eot) = match self {
            Tokenizer::Spm(t) => (t.eos, t.eot),
            Tokenizer::Bpe(t) => (t.eos, t.eot),
        };
        eos == Some(token) || eot == Some(token)
    }

    /// The end-of-generation token ids (EOS + EOT, when present) — the tokens a
    /// grammar constraint allows once it is in an accepting state.
    pub fn eog_ids(&self) -> Vec<u32> {
        let (eos, eot) = match self {
            Tokenizer::Spm(t) => (t.eos, t.eot),
            Tokenizer::Bpe(t) => (t.eos, t.eot),
        };
        [eos, eot].into_iter().flatten().map(|i| i as u32).collect()
    }

    /// Decode `token` (preceded by `prev_token`) into the bytes to emit.
    pub fn decode(&self, prev_token: usize, token: usize) -> Vec<u8> {
        match self {
            Tokenizer::Spm(t) => t.decode(prev_token, token),
            Tokenizer::Bpe(t) => t.decode(token),
        }
    }

    /// The output bytes token `id` contributes mid-sequence, independent of the
    /// preceding token (no BOS space-strip) — the canonical "piece" used to test
    /// a candidate against a grammar. `usize::MAX` is a non-BOS `prev`.
    pub fn token_piece(&self, id: usize) -> Vec<u8> {
        match self {
            Tokenizer::Spm(t) => t.decode(usize::MAX, id),
            Tokenizer::Bpe(t) => t.decode(id),
        }
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        match self {
            Tokenizer::Spm(t) => t.vocab_size(),
            Tokenizer::Bpe(t) => t.vocab_size(),
        }
    }
}

// ===========================================================================
// Special (control / user-defined) tokens
// ===========================================================================

/// GGUF `tokenizer.ggml.token_type` value for a control token (e.g. `<s>`).
const TOKEN_TYPE_CONTROL: u64 = 3;
/// GGUF `tokenizer.ggml.token_type` value for a user-defined / added token.
const TOKEN_TYPE_USER_DEFINED: u64 = 4;

/// One piece of [`SpecialTokens::split`] output.
enum Segment<'t> {
    /// Ordinary text to run through the normal tokenizer.
    Gap(&'t str),
    /// A recognized special token, emitted as this id verbatim.
    Special(usize),
}

/// Control / user-defined tokens (`<|begin_of_text|>`, `<|im_start|>`, …) that
/// are matched verbatim during encode and emitted as literal text on decode,
/// mirroring HuggingFace's `AddedVocabulary` behaviour.
#[derive(Default)]
struct SpecialTokens {
    /// `(literal text, id)`, sorted by descending text length so overlapping
    /// tokens match longest-first.
    entries: Vec<(String, usize)>,
    /// Set of special ids, for O(1) decode-time lookup.
    ids: HashSet<usize>,
}

impl SpecialTokens {
    /// Build from `(text, id)` pairs (used by tests and the GGUF path).
    fn new(mut entries: Vec<(String, usize)>) -> Self {
        entries.sort_by_key(|e| std::cmp::Reverse(e.0.len()));
        let ids = entries.iter().map(|(_, id)| *id).collect();
        SpecialTokens { entries, ids }
    }

    /// Collect CONTROL / USER_DEFINED tokens from `tokenizer.ggml.token_type`.
    ///
    /// `tokens` are the raw (un-decoded) vocabulary strings, indexed by id;
    /// special tokens are stored literally, so their string is the text to
    /// match. Returns an empty table when the metadata is absent.
    fn from_gguf(gguf: &Gguf, tokens: &[String]) -> Self {
        let types = match gguf.meta("tokenizer.ggml.token_type") {
            Some(MetaValue::Array(a)) => a,
            _ => return SpecialTokens::default(),
        };
        let entries = types
            .iter()
            .enumerate()
            .filter_map(|(i, t)| {
                let ty = t.as_u64()?;
                let special = ty == TOKEN_TYPE_CONTROL || ty == TOKEN_TYPE_USER_DEFINED;
                match tokens.get(i) {
                    Some(s) if special && !s.is_empty() => Some((s.clone(), i)),
                    _ => None,
                }
            })
            .collect();
        SpecialTokens::new(entries)
    }

    /// Whether `id` is a special token (decode emits its literal text).
    fn contains(&self, id: usize) -> bool {
        self.ids.contains(&id)
    }

    /// Walk `text`, handing ordinary spans and special-token hits to `f` in
    /// order. With no special tokens this is a single `Gap` over all of `text`.
    fn split<'t>(&self, text: &'t str, mut f: impl FnMut(Segment<'t>)) {
        if self.entries.is_empty() {
            if !text.is_empty() {
                f(Segment::Gap(text));
            }
            return;
        }
        let mut gap_start = 0;
        let mut i = 0;
        while i < text.len() {
            let rest = &text[i..];
            // `entries` is longest-first, so the first hit is the longest match.
            if let Some((s, id)) = self.entries.iter().find(|(s, _)| rest.starts_with(s.as_str())) {
                if gap_start < i {
                    f(Segment::Gap(&text[gap_start..i]));
                }
                f(Segment::Special(*id));
                i += s.len();
                gap_start = i;
            } else {
                i += rest.chars().next().map_or(1, char::len_utf8);
            }
        }
        if gap_start < text.len() {
            f(Segment::Gap(&text[gap_start..]));
        }
    }
}

// ===========================================================================
// SentencePiece-style tokenizer
// ===========================================================================

/// A trained SentencePiece-style vocabulary plus the merge logic to apply it.
pub struct Spm {
    vocab: Vec<Vec<u8>>,
    scores: Vec<f32>,
    lookup: HashMap<Vec<u8>, usize>,
    #[allow(dead_code)]
    max_token_length: usize,
    /// Control / user-defined tokens recognized verbatim (empty unless built
    /// from a GGUF that marks them via `tokenizer.ggml.token_type`).
    specials: SpecialTokens,
    /// Whether to prepend BOS on encode (GGUF `add_bos_token`, default true).
    add_bos: bool,
    /// BOS / EOS token ids (GGUF `bos_token_id` / `eos_token_id`; default 1 / 2,
    /// the llama2.c `tokenizer.bin` convention).
    bos: usize,
    eos: Option<usize>,
    /// End-of-turn token id (GGUF `eot_token_id`), if the model declares a chat
    /// turn-end distinct from `eos` (e.g. Gemma's `<end_of_turn>`).
    eot: Option<usize>,
}

impl Spm {
    /// Build from pieces and their merge scores.
    pub fn from_vocab(vocab: Vec<Vec<u8>>, scores: Vec<f32>) -> Self {
        assert_eq!(vocab.len(), scores.len());
        let max_token_length = vocab.iter().map(Vec::len).max().unwrap_or(0);
        let lookup = vocab
            .iter()
            .enumerate()
            .map(|(i, piece)| (piece.clone(), i))
            .collect();
        Spm {
            vocab,
            scores,
            lookup,
            max_token_length,
            specials: SpecialTokens::default(),
            add_bos: true,
            bos: 1,
            eos: Some(2),
            eot: None,
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

        let mut tk = Spm::from_vocab(vocab, scores);
        tk.max_token_length = max_token_length;
        Ok(tk)
    }

    /// Build from a GGUF file's embedded SentencePiece vocabulary.
    ///
    /// Converts the whitespace marker `▁` (U+2581) to a plain space so the same
    /// encode/decode logic as the `tokenizer.bin` path applies.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let tokens = match gguf.meta("tokenizer.ggml.tokens") {
            Some(MetaValue::Array(a)) => a,
            _ => return Err(Error::Format("GGUF missing tokenizer.ggml.tokens".into())),
        };
        // Raw strings (for special-token matching) and their normalized bytes.
        let raw: Vec<String> = tokens
            .iter()
            .map(|t| {
                t.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| Error::Format("non-string token in GGUF vocab".into()))
            })
            .collect::<Result<_>>()?;
        let vocab: Vec<Vec<u8>> = raw.iter().map(|s| normalize_spm_piece(s)).collect();

        let scores = match gguf.meta("tokenizer.ggml.scores") {
            Some(MetaValue::Array(a)) => a.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect(),
            _ => vec![0.0; vocab.len()],
        };

        let mut tk = Spm::from_vocab(vocab, scores);
        tk.specials = SpecialTokens::from_gguf(gguf, &raw);
        tk.add_bos = gguf
            .meta_u64("tokenizer.ggml.add_bos_token")
            .map(|v| v != 0)
            .unwrap_or(true);
        tk.bos = gguf
            .meta_u64("tokenizer.ggml.bos_token_id")
            .map(|v| v as usize)
            .unwrap_or(1);
        tk.eos = gguf
            .meta_u64("tokenizer.ggml.eos_token_id")
            .ok()
            .map(|v| v as usize)
            .or(tk.eos);
        tk.eot = gguf
            .meta_u64("tokenizer.ggml.eot_token_id")
            .ok()
            .map(|v| v as usize);
        Ok(tk)
    }

    /// Encode following SentencePiece defaults: dummy leading space, byte
    /// fallback, and greedy highest-score adjacent merges.
    ///
    /// Any special (control / user-defined) tokens are recognized verbatim and
    /// the ordinary spans between them are encoded independently.
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<usize> {
        let mut tokens: Vec<usize> = Vec::new();
        if bos {
            tokens.push(self.bos);
        }
        // The dummy leading space is added only to the first ordinary span.
        let mut first = true;
        self.specials.split(text, |seg| match seg {
            Segment::Gap(g) => {
                self.encode_piece(g, first, &mut tokens);
                first = false;
            }
            Segment::Special(id) => {
                tokens.push(id);
                first = false;
            }
        });
        if eos {
            if let Some(e) = self.eos {
                tokens.push(e);
            }
        }
        tokens
    }

    /// Encode one ordinary span: optional dummy leading space, byte fallback,
    /// then greedy highest-score adjacent merges (which never cross spans).
    fn encode_piece(&self, text: &str, add_dummy_space: bool, out: &mut Vec<usize>) {
        if text.is_empty() {
            return;
        }
        let mut tokens: Vec<usize> = Vec::new();
        if add_dummy_space {
            if let Some(&id) = self.lookup.get(b" ".as_slice()) {
                tokens.push(id);
            }
        }

        let mut buf = [0u8; 4];
        for ch in text.chars() {
            let piece = ch.encode_utf8(&mut buf).as_bytes();
            if let Some(&id) = self.lookup.get(piece) {
                tokens.push(id);
            } else {
                // Byte fallback: byte `b` lives at vocab id `b + 3`.
                for &b in piece {
                    let id = b as usize + 3;
                    tokens.push(if id < self.vocab.len() { id } else { 0 });
                }
            }
        }

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
        out.extend(tokens);
    }

    /// Decode, stripping the leading space after BOS and expanding `<0xXX>`.
    pub fn decode(&self, prev_token: usize, token: usize) -> Vec<u8> {
        // Special tokens decode to their literal text (no space-strip / byte
        // parsing).
        if self.specials.contains(token) {
            return self.vocab[token].clone();
        }
        let piece = &self.vocab[token];
        let mut bytes: &[u8] = piece;
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

/// Convert a SentencePiece piece to bytes, replacing `▁` (U+2581) with a space.
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

// ===========================================================================
// GPT-2 byte-level BPE tokenizer
// ===========================================================================

/// A GPT-2 style byte-level BPE tokenizer.
///
/// Bytes are mapped to printable unicode (so BPE never sees control bytes),
/// the input is pre-tokenized into word-like chunks, and each chunk's symbols
/// are merged by rank. Decoding reverses the byte→unicode map.
///
/// The pre-tokenizer applies the model's exact splitting regex, selected by the
/// GGUF `tokenizer.ggml.pre` name (GPT-2 / Llama-3 / Qwen2; see [`PreTokenizer`]),
/// so token boundaries match the reference.
pub struct Bpe {
    /// Raw bytes each token id decodes to.
    id_to_bytes: Vec<Vec<u8>>,
    /// Encoded token string → id.
    token_to_id: HashMap<String, usize>,
    /// (left, right) encoded strings → merge rank (lower = earlier).
    merge_ranks: HashMap<(String, String), usize>,
    /// byte → printable unicode char.
    byte_encoder: [char; 256],
    /// Pre-tokenizer (splitting regex) selected by the model's `pre` name.
    pre: PreTokenizer,
    /// Control / user-defined tokens recognized verbatim during encode.
    specials: SpecialTokens,
    bos: Option<usize>,
    eos: Option<usize>,
    eot: Option<usize>,
    /// Whether to prepend BOS on encode (GGUF `add_bos_token`, default true).
    add_bos: bool,
}

impl Bpe {
    /// Build from encoded `tokens`, ordered `merges`, and optional BOS/EOS ids,
    /// using the default (GPT-2) pre-tokenizer.
    pub fn new(
        tokens: Vec<String>,
        merges: Vec<(String, String)>,
        bos: Option<usize>,
        eos: Option<usize>,
    ) -> Self {
        Bpe::build(tokens, merges, bos, eos, "gpt-2", SpecialTokens::default())
    }

    /// Build with an explicit pre-tokenizer name (see [`pre_pattern`]) and
    /// special-token table.
    fn build(
        tokens: Vec<String>,
        merges: Vec<(String, String)>,
        bos: Option<usize>,
        eos: Option<usize>,
        pre: &str,
        specials: SpecialTokens,
    ) -> Self {
        let byte_encoder = bytes_to_unicode();
        let byte_decoder: HashMap<char, u8> = byte_encoder
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();

        let token_to_id = tokens
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();
        let id_to_bytes = tokens
            .iter()
            .map(|s| decode_token_bytes(s, &byte_decoder))
            .collect();
        let merge_ranks = merges
            .into_iter()
            .enumerate()
            .map(|(r, pair)| (pair, r))
            .collect();

        Bpe {
            id_to_bytes,
            token_to_id,
            merge_ranks,
            byte_encoder,
            pre: PreTokenizer::new(pre),
            specials,
            bos,
            eos,
            eot: None,
            add_bos: true,
        }
    }

    /// Build from GGUF metadata (`tokenizer.ggml.tokens` / `.merges` / ids).
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let tokens = match gguf.meta("tokenizer.ggml.tokens") {
            Some(MetaValue::Array(a)) => a,
            _ => return Err(Error::Format("GGUF missing tokenizer.ggml.tokens".into())),
        };
        let tokens: Vec<String> = tokens
            .iter()
            .map(|t| {
                t.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| Error::Format("non-string token in GGUF vocab".into()))
            })
            .collect::<Result<_>>()?;

        let merges = match gguf.meta("tokenizer.ggml.merges") {
            Some(MetaValue::Array(a)) => a
                .iter()
                .filter_map(|m| m.as_str())
                .filter_map(|m| m.split_once(' '))
                .map(|(l, r)| (l.to_owned(), r.to_owned()))
                .collect(),
            _ => Vec::new(),
        };

        let bos = gguf
            .meta_u64("tokenizer.ggml.bos_token_id")
            .ok()
            .map(|v| v as usize);
        let eos = gguf
            .meta_u64("tokenizer.ggml.eos_token_id")
            .ok()
            .map(|v| v as usize);

        // The pre-tokenizer name picks the exact splitting regex (Llama-3,
        // Qwen2, …); absent or unknown falls back to GPT-2.
        let pre = gguf.meta_str("tokenizer.ggml.pre").unwrap_or("gpt-2");
        let specials = SpecialTokens::from_gguf(gguf, &tokens);

        let mut bpe = Bpe::build(tokens, merges, bos, eos, pre, specials);
        bpe.add_bos = gguf
            .meta_u64("tokenizer.ggml.add_bos_token")
            .map(|v| v != 0)
            .unwrap_or(true);
        bpe.eot = gguf
            .meta_u64("tokenizer.ggml.eot_token_id")
            .ok()
            .map(|v| v as usize);
        Ok(bpe)
    }

    /// Encode `text`, optionally bracketing with BOS/EOS.
    ///
    /// Special (control / user-defined) tokens are matched verbatim and only the
    /// ordinary spans between them are pre-tokenized and BPE-merged.
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<usize> {
        let mut ids = Vec::new();
        if bos {
            ids.extend(self.bos);
        }
        self.specials.split(text, |seg| match seg {
            Segment::Gap(gap) => self
                .pre
                .for_each_chunk(gap, |chunk| self.encode_chunk(chunk, &mut ids)),
            Segment::Special(id) => ids.push(id),
        });
        if eos {
            ids.extend(self.eos);
        }
        ids
    }

    /// BPE-encode one pre-token chunk, appending ids to `out`.
    fn encode_chunk(&self, chunk: &str, out: &mut Vec<usize>) {
        // Each byte becomes one printable-unicode symbol.
        let mut symbols: Vec<String> = chunk
            .bytes()
            .map(|b| self.byte_encoder[b as usize].to_string())
            .collect();
        if symbols.is_empty() {
            return;
        }

        // Repeatedly merge the lowest-rank adjacent pair.
        loop {
            let mut best_rank = usize::MAX;
            let mut best = None;
            for i in 0..symbols.len() - 1 {
                if let Some(&rank) = self
                    .merge_ranks
                    .get(&(symbols[i].clone(), symbols[i + 1].clone()))
                {
                    if rank < best_rank {
                        best_rank = rank;
                        best = Some(i);
                    }
                }
            }
            let Some(i) = best else { break };
            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols[i] = merged;
            symbols.remove(i + 1);
        }

        for s in &symbols {
            if let Some(&id) = self.token_to_id.get(s) {
                out.push(id);
            } else {
                // Fall back to single-char (single-byte) tokens.
                for ch in s.chars() {
                    if let Some(&id) = self.token_to_id.get(&ch.to_string()) {
                        out.push(id);
                    }
                }
            }
        }
    }

    /// Decode a token id to its raw bytes (caller concatenates).
    pub fn decode(&self, token: usize) -> Vec<u8> {
        self.id_to_bytes
            .get(token)
            .cloned()
            .unwrap_or_default()
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.id_to_bytes.len()
    }
}

/// Decode an encoded token string back to bytes via the byte→unicode inverse.
fn decode_token_bytes(s: &str, byte_decoder: &HashMap<char, u8>) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    let mut buf = [0u8; 4];
    for ch in s.chars() {
        if let Some(&b) = byte_decoder.get(&ch) {
            out.push(b);
        } else {
            // Special/added tokens stored as literal unicode (not byte-encoded).
            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        }
    }
    out
}

/// The GPT-2 reversible byte → printable-unicode mapping.
///
/// Printable byte ranges map to themselves; the rest map to U+0100.. so that
/// every byte is represented by a printable char.
pub(crate) fn bytes_to_unicode() -> [char; 256] {
    let mut printable: Vec<u32> = Vec::new();
    printable.extend(0x21..=0x7E);
    printable.extend(0xA1..=0xAC);
    printable.extend(0xAE..=0xFF);

    let mut table = ['\0'; 256];
    let mut next = 0u32;
    for b in 0..256u32 {
        let c = if printable.contains(&b) {
            b
        } else {
            let c = 256 + next;
            next += 1;
            c
        };
        table[b as usize] = char::from_u32(c).unwrap();
    }
    table
}

/// The 256 single-byte token strings (byte → its printable-unicode char).
///
/// Useful for building byte-level test/demo vocabularies.
pub(crate) fn gpt2_byte_tokens() -> Vec<String> {
    bytes_to_unicode().iter().map(|c| c.to_string()).collect()
}

// GPT-2's original pre-tokenizer regex: contractions, then runs of letters /
// digits / other (each with an optional leading space), then whitespace.
const PRE_GPT2: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

// Llama-3's pattern (from its `tokenizer.json`): case-insensitive contractions,
// letters optionally led by one non-letter/digit, digits in groups of up to 3,
// other runs that swallow trailing newlines, then whitespace.
const PRE_LLAMA3: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

// Qwen2's pattern — identical to Llama-3's except digits split one at a time
// (`\p{N}` instead of `\p{N}{1,3}`).
const PRE_QWEN2: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Map a GGUF `tokenizer.ggml.pre` name to its splitting regex.
///
/// Names follow llama.cpp's `llama-vocab.cpp`. Anything unrecognized (or a
/// missing `pre`) falls back to the GPT-2 pattern, matching llama.cpp's default.
fn pre_pattern(pre: &str) -> &'static str {
    match pre {
        "llama-bpe" | "llama3" | "llama-v3" | "llama-bpe-v3" => PRE_LLAMA3,
        "qwen2" | "qwen" => PRE_QWEN2,
        _ => PRE_GPT2,
    }
}

/// The byte-exact GPT-2-family pre-tokenizer: a compiled splitting regex chosen
/// by the model's `pre` name. The `regex` crate lacks the lookahead these
/// patterns use, so we compile them with `fancy-regex`.
struct PreTokenizer {
    re: Regex,
}

impl PreTokenizer {
    /// Compile the pattern selected by `pre` (see [`pre_pattern`]).
    fn new(pre: &str) -> Self {
        // The patterns are fixed constants, so compilation cannot fail.
        let re = Regex::new(pre_pattern(pre)).expect("built-in pretokenizer regex is valid");
        PreTokenizer { re }
    }

    /// Apply the regex, handing each pre-token chunk to `f` in order.
    ///
    /// These patterns tile the whole input (every character matches some
    /// alternative), but should a gap ever appear it is emitted verbatim so no
    /// bytes are dropped.
    fn for_each_chunk<'t>(&self, text: &'t str, mut f: impl FnMut(&'t str)) {
        let mut last = 0;
        for m in self.re.find_iter(text) {
            let m = m.expect("pretokenizer regex match");
            if m.start() > last {
                f(&text[last..m.start()]);
            }
            f(&text[m.start()..m.end()]);
            last = m.end();
        }
        if last < text.len() {
            f(&text[last..]);
        }
    }

    /// Collect the chunks into a `Vec` (convenience for callers/tests).
    fn split(&self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        self.for_each_chunk(text, |c| out.push(c.to_owned()));
        out
    }
}

/// Split `text` into GPT-2-family pre-token chunks using the regex selected by
/// the `pre` name (`"gpt-2"`, `"llama-bpe"` for Llama-3, `"qwen2"`, …).
///
/// Exposed mainly so the byte-exactness of the splitter can be checked against
/// reference tokenizer output. Compiles a regex per call, so prefer encoding
/// through a [`Tokenizer`] for hot paths.
pub fn pretokenize(pre: &str, text: &str) -> Vec<String> {
    PreTokenizer::new(pre).split(text)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn spm_encode_merges_into_single_token() {
        let tk = hello_tokenizer();
        assert_eq!(tk.encode("hello", true, false), vec![1, 11]);
    }

    #[test]
    fn spm_decode_strips_space_after_bos() {
        let tk = hello_tokenizer();
        assert_eq!(tk.decode(1, 11), b"hello");
        assert_eq!(tk.decode(7, 11), b" hello");
    }

    #[test]
    fn spm_roundtrip_via_decode() {
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
    fn spm_byte_token_decodes_to_raw_byte() {
        let tk = Tokenizer::from_vocab(vec![b"<0x41>".to_vec()], vec![0.0]);
        assert_eq!(tk.decode(0, 0), b"A");
        assert_eq!(parse_byte_token(b"<0x0a>"), Some(b'\n'));
        assert_eq!(parse_byte_token(b"hello"), None);
    }

    #[test]
    fn byte_unicode_map_is_a_bijection() {
        let table = bytes_to_unicode();
        let mut seen = std::collections::HashSet::new();
        for &c in &table {
            assert!(seen.insert(c), "duplicate char in byte map");
        }
        assert_eq!(table[b' ' as usize], 'Ġ'); // GPT-2's encoded space
        assert_eq!(table[b'a' as usize], 'a'); // printable ASCII is identity
    }

    /// A tiny byte-level vocab: all 256 byte tokens, then a few merged tokens.
    fn bpe_tokenizer() -> Tokenizer {
        let mut tokens = gpt2_byte_tokens();
        let enc = bytes_to_unicode();
        let s = |b: u8| enc[b as usize].to_string();
        // Append merged tokens and the merges that build them.
        let merged = [
            (s(b'h'), s(b'i')),                  // "hi"
            (format!("{}{}", s(b' '), s(b'y')), s(b'o')), // "Ġyo" (" yo")
        ];
        let mut merges = Vec::new();
        for (l, r) in &merged {
            tokens.push(format!("{l}{r}"));
            merges.push((l.clone(), r.clone()));
        }
        Tokenizer::from_bpe(tokens, merges, Some(1), Some(2))
    }

    #[test]
    fn bpe_merges_and_roundtrips() {
        let tk = bpe_tokenizer();
        // "hi" should merge into the single appended token (id 256).
        let ids = tk.encode("hi", false, false);
        assert_eq!(ids, vec![256]);

        // Decoding returns the original bytes.
        let mut out = Vec::new();
        for &id in &ids {
            out.extend_from_slice(&tk.decode(0, id));
        }
        assert_eq!(out, b"hi");
    }

    #[test]
    fn bpe_byte_level_roundtrip_without_merges() {
        // Arbitrary bytes (incl. UTF-8) survive encode→decode even with no
        // applicable merges, because every byte is its own token.
        let tk = bpe_tokenizer();
        let text = "Hé! 42";
        let ids = tk.encode(text, false, false);
        let mut out = Vec::new();
        for &id in &ids {
            out.extend_from_slice(&tk.decode(0, id));
        }
        assert_eq!(out, text.as_bytes());
    }

    #[test]
    fn bpe_bos_eos() {
        let tk = bpe_tokenizer();
        let ids = tk.encode("hi", true, true);
        assert_eq!(ids.first(), Some(&1)); // BOS
        assert_eq!(ids.last(), Some(&2)); // EOS
    }

    #[test]
    fn special_tokens_split_longest_first() {
        // Overlapping specials: the longer one must win at a shared prefix.
        let st = SpecialTokens::new(vec![
            ("<|a|>".to_string(), 10),
            ("<|ab|>".to_string(), 11),
        ]);
        let mut segs = Vec::new();
        st.split("x<|ab|>y<|a|>", |seg| {
            segs.push(match seg {
                Segment::Gap(g) => g.to_string(),
                Segment::Special(id) => format!("#{id}"),
            })
        });
        assert_eq!(segs, vec!["x", "#11", "y", "#10"]);
    }

    #[test]
    fn spm_special_tokens_round_trip() {
        // A tiny SPM vocab with a control token "<s2>" injected as special.
        let entries: &[(&[u8], f32)] = &[
            (b"<unk>", 0.0),    // 0
            (b"\n<s>\n", 0.0),  // 1 BOS
            (b"\n</s>\n", 0.0), // 2 EOS
            (b" ", -3.0),       // 3
            (b"h", -4.0),       // 4
            (b"i", -5.0),       // 5
            (b"<s2>", 0.0),     // 6 special (control)
        ];
        let vocab: Vec<Vec<u8>> = entries.iter().map(|(p, _)| p.to_vec()).collect();
        let scores = entries.iter().map(|(_, s)| *s).collect();
        let mut spm = Spm::from_vocab(vocab, scores);
        spm.specials = SpecialTokens::new(vec![("<s2>".to_string(), 6)]);

        // "<s2>" is recognized as id 6, "hi" still encodes around it.
        let ids = spm.encode("hi<s2>hi", false, false);
        assert!(ids.contains(&6));
        assert_eq!(ids.iter().filter(|&&i| i == 6).count(), 1);

        // Decode of the special yields its literal text.
        assert_eq!(spm.decode(0, 6), b"<s2>");
    }

    #[test]
    fn pretokenize_splits_like_gpt2() {
        assert_eq!(pretokenize("gpt-2", "hello world"), vec!["hello", " world"]);
        assert_eq!(
            pretokenize("gpt-2", "it's 42!"),
            vec!["it", "'s", " 42", "!"]
        );
        assert_eq!(pretokenize("gpt-2", ""), Vec::<String>::new());
    }

    #[test]
    fn pretokenize_digit_runs_differ_by_pre() {
        // The clearest discriminator between the three patterns: GPT-2 keeps a
        // digit run whole, Llama-3 splits it into groups of ≤3, Qwen2 one apiece.
        assert_eq!(pretokenize("gpt-2", "1234567"), vec!["1234567"]);
        assert_eq!(pretokenize("llama-bpe", "1234567"), vec!["123", "456", "7"]);
        assert_eq!(
            pretokenize("qwen2", "1234567"),
            vec!["1", "2", "3", "4", "5", "6", "7"]
        );
    }

    #[test]
    fn pretokenize_uppercase_contractions_need_case_insensitive() {
        // Llama-3/Qwen2 fold contraction case, so "'S" stays one chunk; GPT-2's
        // contractions are lowercase-only, so the apostrophe splits off.
        assert_eq!(pretokenize("gpt-2", "IT'S"), vec!["IT", "'", "S"]);
        assert_eq!(pretokenize("llama-bpe", "IT'S"), vec!["IT", "'S"]);
        assert_eq!(pretokenize("qwen2", "IT'S"), vec!["IT", "'S"]);
    }
}
