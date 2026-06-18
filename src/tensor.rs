//! A weight matrix that may be stored full-precision or quantized.
//!
//! [`QMatrix`] is the unit the matmul kernel consumes. A quantized matrix keeps
//! its compressed bytes (typically borrowed straight from the memory-mapped
//! GGUF file) and is dequantized one block at a time inside the matmul, so the
//! full f32 expansion never has to live in RAM.

use std::borrow::Cow;

use crate::error::{Error, Result};
use crate::quant::{dequantize_into, GgmlType};

/// A 2-D weight matrix: `rows` (output features) × `cols` (input features),
/// row-major. Either raw f32 or a quantized block format.
pub enum QMatrix<'a> {
    /// Full-precision weights.
    F32 {
        data: Cow<'a, [f32]>,
        rows: usize,
        cols: usize,
    },
    /// Quantized weights, dequantized lazily.
    Quant {
        ty: GgmlType,
        data: Cow<'a, [u8]>,
        rows: usize,
        cols: usize,
    },
}

impl<'a> QMatrix<'a> {
    /// Build a full-precision matrix, validating that `data` holds `rows*cols`.
    pub fn f32(data: Cow<'a, [f32]>, rows: usize, cols: usize) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(Error::Format(format!(
                "f32 matrix expected {} elements, got {}",
                rows * cols,
                data.len()
            )));
        }
        Ok(QMatrix::F32 { data, rows, cols })
    }

    /// Build a quantized matrix, validating that `data` is long enough.
    pub fn quant(ty: GgmlType, data: Cow<'a, [u8]>, rows: usize, cols: usize) -> Result<Self> {
        if !cols.is_multiple_of(ty.block_size()) {
            return Err(Error::Format(format!(
                "{ty:?} matrix row width {cols} is not a multiple of block size {}",
                ty.block_size()
            )));
        }
        let need = ty.bytes_for(rows * cols);
        if data.len() < need {
            return Err(Error::Format(format!(
                "{ty:?} matrix data too short: need {need} bytes, have {}",
                data.len()
            )));
        }
        Ok(QMatrix::Quant {
            ty,
            data,
            rows,
            cols,
        })
    }

    /// Number of output rows.
    pub fn rows(&self) -> usize {
        match self {
            QMatrix::F32 { rows, .. } | QMatrix::Quant { rows, .. } => *rows,
        }
    }

    /// Number of input columns.
    pub fn cols(&self) -> usize {
        match self {
            QMatrix::F32 { cols, .. } | QMatrix::Quant { cols, .. } => *cols,
        }
    }

    /// Stable identity of the backing bytes — the pointer the resident weight
    /// caches (and LoRA pair lookups) key on.
    pub fn data_ptr(&self) -> usize {
        match self {
            QMatrix::F32 { data, .. } => data.as_ptr() as usize,
            QMatrix::Quant { data, .. } => data.as_ptr() as usize,
        }
    }

    /// Dequantize a single row into `out` (`out.len()` must equal `cols`).
    ///
    /// Used both for the embedding lookup and (per-row) by the matmul kernel.
    pub fn dequant_row(&self, row: usize, out: &mut [f32]) {
        match self {
            QMatrix::F32 { data, cols, .. } => {
                out.copy_from_slice(&data[row * cols..row * cols + cols]);
            }
            QMatrix::Quant {
                ty, data, cols, ..
            } => {
                let rb = ty.bytes_for(*cols);
                // Lengths were validated at construction; dequant can't fail.
                let _ = dequantize_into(*ty, &data[row * rb..row * rb + rb], out);
            }
        }
    }
}
