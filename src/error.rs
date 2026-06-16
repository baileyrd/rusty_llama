//! Error types shared across the crate.

use std::fmt;

/// Errors produced while loading or running a model.
#[derive(Debug)]
pub enum Error {
    /// An underlying I/O failure (opening / mapping the checkpoint, etc.).
    Io(std::io::Error),
    /// The bytes we were handed are not a valid checkpoint / tokenizer file.
    Format(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Format(s) => write!(f, "format error: {s}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Format(_) => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
