//! Byte-exactness of the GPT-2-family pre-tokenizer.
//!
//! The fixture in `fixtures/pretokenize_golden.rs` was generated offline (see
//! `scripts/gen_pretokenize_golden.py`) by running each model's documented
//! splitting regex through Python's `regex` module — a regex engine independent
//! of the `fancy-regex` crate the tokenizer uses. Reproducing those chunkings
//! exactly is strong evidence the splitter matches the reference tokenizer,
//! with no network access required at test time.

// The vendored golden table is a deeply-nested slice literal by nature.
#![allow(clippy::type_complexity)]

use rusty_llama::tokenizer::pretokenize;

// pub const PRETOKENIZE_GOLDEN: &[(&str, &[(&str, &[&str])])]
include!("fixtures/pretokenize_golden.rs");

#[test]
fn pretokenizer_matches_reference_chunking() {
    let mut cases = 0;
    for (pre, samples) in PRETOKENIZE_GOLDEN {
        for (input, expected) in *samples {
            let got = pretokenize(pre, input);
            assert_eq!(
                got, *expected,
                "pre={pre:?} input={input:?}: got {got:?}, expected {expected:?}"
            );
            cases += 1;
        }
    }
    // Guard against an empty/garbled fixture silently passing.
    assert!(cases >= 90, "expected a substantial corpus, ran {cases} cases");
}

#[test]
fn pretokenizer_chunks_reassemble_input() {
    // Whatever the splitting rule, concatenating the chunks must reproduce the
    // input byte-for-byte (no characters dropped or duplicated).
    for (pre, samples) in PRETOKENIZE_GOLDEN {
        for (input, _) in *samples {
            let joined: String = pretokenize(pre, input).concat();
            assert_eq!(&joined, input, "pre={pre:?} input={input:?} did not reassemble");
        }
    }
}
