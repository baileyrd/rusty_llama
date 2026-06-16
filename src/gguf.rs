//! GGUF container parsing.
//!
//! GGUF is llama.cpp's model file format. The layout is:
//!
//! ```text
//! magic "GGUF" | version u32 | tensor_count u64 | metadata_kv_count u64
//! metadata key/value pairs
//! tensor infos (name, dims, type, offset)
//! padding to general.alignment
//! tensor data
//! ```
//!
//! This module parses the header, metadata, and tensor table, and hands back
//! borrowed byte slices for each tensor's (still-quantized) data. Turning that
//! into runnable weights happens in [`crate::model::Model::from_gguf`].

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::quant::GgmlType;

const MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: usize = 32;

/// A typed GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetaValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetaValue {
    /// Coerce any integer variant to `u64`.
    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            MetaValue::U8(v) => *v as u64,
            MetaValue::U16(v) => *v as u64,
            MetaValue::U32(v) => *v as u64,
            MetaValue::U64(v) => *v,
            MetaValue::I8(v) if *v >= 0 => *v as u64,
            MetaValue::I16(v) if *v >= 0 => *v as u64,
            MetaValue::I32(v) if *v >= 0 => *v as u64,
            MetaValue::I64(v) if *v >= 0 => *v as u64,
            MetaValue::Bool(b) => *b as u64,
            _ => return None,
        })
    }

    /// Coerce any float/integer variant to `f32`.
    pub fn as_f32(&self) -> Option<f32> {
        Some(match self {
            MetaValue::F32(v) => *v,
            MetaValue::F64(v) => *v as f32,
            _ => self.as_u64()? as f32,
        })
    }

    /// Borrow the string payload, if this is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::String(s) => Some(s),
            _ => None,
        }
    }
}

/// Description of one tensor in the file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name, e.g. `blk.0.attn_q.weight`.
    pub name: String,
    /// Dimensions, ggml order (`dims[0]` is the fastest-varying axis).
    pub dims: Vec<u64>,
    /// Element/quantization type.
    pub ggml_type: GgmlType,
    /// Byte offset from the start of the tensor-data section.
    pub offset: u64,
}

impl TensorInfo {
    /// Total number of elements across all dimensions.
    pub fn n_elements(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }
}

/// A parsed GGUF file, borrowing the underlying bytes.
pub struct Gguf<'a> {
    data: &'a [u8],
    /// File format version (2 or 3).
    pub version: u32,
    /// All metadata key/value pairs.
    pub metadata: HashMap<String, MetaValue>,
    /// Tensor descriptors in file order.
    pub tensors: Vec<TensorInfo>,
    index: HashMap<String, usize>,
    data_offset: usize,
}

impl<'a> Gguf<'a> {
    /// True if `data` begins with the GGUF magic.
    pub fn is_gguf(data: &[u8]) -> bool {
        data.len() >= 4 && &data[..4] == MAGIC
    }

    /// Parse a GGUF file from raw (typically memory-mapped) bytes.
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        let mut r = Reader { data, pos: 0 };

        if r.bytes(4)? != MAGIC {
            return Err(Error::Format("not a GGUF file (bad magic)".into()));
        }
        let version = r.u32()?;
        if version != 2 && version != 3 {
            return Err(Error::Format(format!(
                "unsupported GGUF version {version} (only 2 and 3 are supported)"
            )));
        }
        let tensor_count = r.u64()? as usize;
        let kv_count = r.u64()? as usize;

        let mut metadata = HashMap::with_capacity(kv_count);
        for _ in 0..kv_count {
            let key = r.string()?;
            let value_type = r.u32()?;
            let value = r.value(value_type)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count);
        let mut index = HashMap::with_capacity(tensor_count);
        for i in 0..tensor_count {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let ggml_type = GgmlType::from_u32(r.u32()?)?;
            let offset = r.u64()?;
            index.insert(name.clone(), i);
            tensors.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
            });
        }

        // Tensor data begins after the header, padded up to general.alignment.
        let alignment = metadata
            .get("general.alignment")
            .and_then(MetaValue::as_u64)
            .map(|a| a as usize)
            .unwrap_or(DEFAULT_ALIGNMENT)
            .max(1);
        let data_offset = r.pos.next_multiple_of(alignment);
        if data_offset > data.len() {
            return Err(Error::Format("GGUF tensor data section is missing".into()));
        }

        Ok(Gguf {
            data,
            version,
            metadata,
            tensors,
            index,
            data_offset,
        })
    }

    /// Look up a metadata value by key.
    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    /// Read a metadata value as `u64`, erroring if absent or wrong type.
    pub fn meta_u64(&self, key: &str) -> Result<u64> {
        self.meta(key)
            .and_then(MetaValue::as_u64)
            .ok_or_else(|| Error::Format(format!("missing/invalid u64 metadata '{key}'")))
    }

    /// Read a metadata value as `f32`, erroring if absent or wrong type.
    pub fn meta_f32(&self, key: &str) -> Result<f32> {
        self.meta(key)
            .and_then(MetaValue::as_f32)
            .ok_or_else(|| Error::Format(format!("missing/invalid f32 metadata '{key}'")))
    }

    /// Read a metadata value as `&str`, erroring if absent or wrong type.
    pub fn meta_str(&self, key: &str) -> Result<&str> {
        self.meta(key)
            .and_then(MetaValue::as_str)
            .ok_or_else(|| Error::Format(format!("missing/invalid string metadata '{key}'")))
    }

    /// Find a tensor by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    /// Borrow the raw (still-quantized) bytes for a tensor.
    pub fn tensor_bytes(&self, info: &TensorInfo) -> Result<&'a [u8]> {
        let need = info.ggml_type.bytes_for(info.n_elements());
        let start = self.data_offset + info.offset as usize;
        let end = start
            .checked_add(need)
            .ok_or_else(|| Error::Format("tensor offset overflow".into()))?;
        if end > self.data.len() {
            return Err(Error::Format(format!(
                "tensor '{}' extends past end of file",
                info.name
            )));
        }
        Ok(&self.data[start..end])
    }
}

/// A bounds-checked little-endian cursor over the file bytes.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| Error::Format("GGUF read overflow".into()))?;
        if end > self.data.len() {
            return Err(Error::Format("unexpected end of GGUF file".into()));
        }
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.bytes(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let b = self.bytes(len)?;
        Ok(String::from_utf8_lossy(b).into_owned())
    }

    /// Read a single metadata value of the given GGUF value-type tag.
    fn value(&mut self, value_type: u32) -> Result<MetaValue> {
        Ok(match value_type {
            0 => MetaValue::U8(self.u8()?),
            1 => MetaValue::I8(self.u8()? as i8),
            2 => MetaValue::U16(self.u16()?),
            3 => MetaValue::I16(self.u16()? as i16),
            4 => MetaValue::U32(self.u32()?),
            5 => MetaValue::I32(self.u32()? as i32),
            6 => MetaValue::F32(f32::from_bits(self.u32()?)),
            7 => MetaValue::Bool(self.u8()? != 0),
            8 => MetaValue::String(self.string()?),
            9 => {
                let elem_type = self.u32()?;
                let count = self.u64()? as usize;
                let mut items = Vec::with_capacity(count.min(1 << 16));
                for _ in 0..count {
                    items.push(self.value(elem_type)?);
                }
                MetaValue::Array(items)
            }
            10 => MetaValue::U64(self.u64()?),
            11 => MetaValue::I64(self.u64()? as i64),
            12 => MetaValue::F64(f64::from_bits(self.u64()?)),
            other => {
                return Err(Error::Format(format!(
                    "unknown GGUF metadata value type {other}"
                )))
            }
        })
    }
}
