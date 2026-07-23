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
    /// Index into [`Gguf`]'s per-shard data (0 for a non-split file — every
    /// tensor lives in "shard 0", the only shard).
    pub shard: usize,
}

impl TensorInfo {
    /// Total number of elements across all dimensions.
    pub fn n_elements(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }
}

/// One shard's parsed header: everything [`Gguf::parse`] produces except the
/// backing bytes themselves, so [`Gguf::parse_sharded`] can reuse the same
/// per-file parsing logic and then merge several of these.
struct ShardHeader {
    version: u32,
    metadata: HashMap<String, MetaValue>,
    /// Tensors as read from this shard's own tensor table (`shard` unset —
    /// the caller tags it after merging).
    tensors: Vec<TensorInfo>,
    data_offset: usize,
}

/// A parsed GGUF file, borrowing the underlying bytes. Ordinarily backed by a
/// single memory-mapped file; [`Gguf::parse_sharded`] backs it by several
/// (llama.cpp's split-GGUF convention — one logical model spread across
/// `model-NNNNN-of-MMMMM.gguf` files), in which case each [`TensorInfo`]
/// records which shard's bytes it lives in.
pub struct Gguf<'a> {
    /// One byte slice per shard (length 1 for a non-split file).
    data: Vec<&'a [u8]>,
    /// Per-shard tensor-data start offset, indexed the same as `data`.
    data_offset: Vec<usize>,
    /// File format version (2 or 3).
    pub version: u32,
    /// All metadata key/value pairs. For a split file this is shard 0's
    /// metadata — every shard is expected to carry the same shared keys
    /// (architecture, hyperparameters, tokenizer, ...), so this repo doesn't
    /// cross-validate them shard-to-shard.
    pub metadata: HashMap<String, MetaValue>,
    /// Tensor descriptors in file order (shard 0's tensors first, then shard
    /// 1's, ...).
    pub tensors: Vec<TensorInfo>,
    index: HashMap<String, usize>,
}

impl<'a> Gguf<'a> {
    /// True if `data` begins with the GGUF magic.
    pub fn is_gguf(data: &[u8]) -> bool {
        data.len() >= 4 && &data[..4] == MAGIC
    }

    /// Parse a single GGUF file's header (magic/version/metadata/tensor
    /// table) without tagging tensors to a shard — shared by [`Gguf::parse`]
    /// (shard 0, the only one) and [`Gguf::parse_sharded`] (one call per
    /// shard, merged afterward).
    fn parse_header(data: &[u8]) -> Result<ShardHeader> {
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

        // ponytail: cap pre-allocation at 1<<16 so a malicious count can't OOM us
        // before the read loop hits truncated data and errors out cleanly.
        let mut metadata = HashMap::with_capacity(kv_count.min(1 << 16));
        for _ in 0..kv_count {
            let key = r.string()?;
            let value_type = r.u32()?;
            let value = r.value(value_type)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count.min(1 << 16));
        let mut names = std::collections::HashSet::with_capacity(tensor_count.min(1 << 16));
        for _ in 0..tensor_count {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims.min(1 << 16));
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            // Reject dims whose element count overflows u64 — n_elements() recomputes
            // this product and feeds it to the tensor_bytes length guard; a wrap would
            // undercut that guard.
            dims.iter()
                .try_fold(1u64, |a, &d| a.checked_mul(d))
                .ok_or_else(|| Error::Format(format!("tensor '{name}' dims overflow")))?;
            let ggml_type = GgmlType::from_u32(r.u32()?)?;
            let offset = r.u64()?;
            if !names.insert(name.clone()) {
                return Err(Error::Format(format!("duplicate tensor name '{name}'")));
            }
            tensors.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
                shard: 0, // tagged by the caller once the final shard index is known
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

        Ok(ShardHeader {
            version,
            metadata,
            tensors,
            data_offset,
        })
    }

    /// Parse a GGUF file from raw (typically memory-mapped) bytes.
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        let h = Self::parse_header(data)?;
        let mut index = HashMap::with_capacity(h.tensors.len());
        for (i, t) in h.tensors.iter().enumerate() {
            index.insert(t.name.clone(), i);
        }
        Ok(Gguf {
            data: vec![data],
            data_offset: vec![h.data_offset],
            version: h.version,
            metadata: h.metadata,
            tensors: h.tensors,
            index,
        })
    }

    /// Parse a model split across several GGUF files (llama.cpp's
    /// `split.count`/`split.no` convention: each shard is a complete,
    /// independently-parseable GGUF file whose tensor table lists only the
    /// tensors physically stored in it). `shards` must be in shard order
    /// (shard 0 first); every tensor across every shard must have a unique
    /// name. Metadata is taken from `shards[0]` — see the [`Gguf::metadata`]
    /// doc.
    pub fn parse_sharded(shards: &[&'a [u8]]) -> Result<Self> {
        if shards.is_empty() {
            return Err(Error::Format("no shards given".into()));
        }
        let mut data = Vec::with_capacity(shards.len());
        let mut data_offset = Vec::with_capacity(shards.len());
        let mut tensors = Vec::new();
        let mut index = HashMap::new();
        let mut version = None;
        let mut metadata = None;
        for (shard, &bytes) in shards.iter().enumerate() {
            let h = Self::parse_header(bytes)?;
            if let Some(v) = version {
                if v != h.version {
                    return Err(Error::Format(format!(
                        "shard {shard} has GGUF version {}, expected {v} (from shard 0)",
                        h.version
                    )));
                }
            } else {
                version = Some(h.version);
            }
            data.push(bytes);
            data_offset.push(h.data_offset);
            for mut t in h.tensors {
                t.shard = shard;
                if index.insert(t.name.clone(), tensors.len()).is_some() {
                    return Err(Error::Format(format!(
                        "duplicate tensor name '{}' across shards",
                        t.name
                    )));
                }
                tensors.push(t);
            }
            if shard == 0 {
                metadata = Some(h.metadata);
            }
        }
        Ok(Gguf {
            data,
            data_offset,
            version: version.unwrap(),
            metadata: metadata.unwrap(),
            tensors,
            index,
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
        let shard_data = self.data[info.shard];
        let start = self.data_offset[info.shard] + info.offset as usize;
        let end = start
            .checked_add(need)
            .ok_or_else(|| Error::Format("tensor offset overflow".into()))?;
        if end > shard_data.len() {
            return Err(Error::Format(format!(
                "tensor '{}' extends past end of shard {}",
                info.name, info.shard
            )));
        }
        Ok(&shard_data[start..end])
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

#[cfg(test)]
mod hardening_tests {
    use super::*;

    fn str_(v: &mut Vec<u8>, s: &str) {
        v.extend_from_slice(&(s.len() as u64).to_le_bytes());
        v.extend_from_slice(s.as_bytes());
    }
    fn header(tensor_count: u64) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&tensor_count.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // kv_count
        b
    }

    #[test]
    fn rejects_duplicate_tensor_name() {
        let mut b = header(2);
        for _ in 0..2 {
            str_(&mut b, "dup");
            b.extend_from_slice(&1u32.to_le_bytes()); // n_dims
            b.extend_from_slice(&1u64.to_le_bytes()); // dims[0]
            b.extend_from_slice(&0u32.to_le_bytes()); // type F32
            b.extend_from_slice(&0u64.to_le_bytes()); // offset
        }
        b.resize(b.len() + 64, 0);
        let err = match Gguf::parse(&b) {
            Ok(_) => panic!("expected duplicate-name error"),
            Err(e) => format!("{e:?}"),
        };
        assert!(err.contains("duplicate"), "{err}");
    }

    #[test]
    fn rejects_dims_overflow() {
        let mut b = header(1);
        str_(&mut b, "big");
        b.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        b.extend_from_slice(&u64::MAX.to_le_bytes()); // dims[0]
        b.extend_from_slice(&2u64.to_le_bytes()); // dims[1] -> product overflows
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.resize(b.len() + 64, 0);
        let err = match Gguf::parse(&b) {
            Ok(_) => panic!("expected dims-overflow error"),
            Err(e) => format!("{e:?}"),
        };
        assert!(err.contains("overflow"), "{err}");
    }

    #[test]
    fn huge_count_does_not_oom() {
        // tensor_count = u64::MAX must not pre-allocate; it errors on truncated data.
        let b = header(u64::MAX);
        assert!(Gguf::parse(&b).is_err());
    }

    /// Build a minimal (no metadata) single-file GGUF holding the given named
    /// f32 tensors, packed back-to-back with no inter-tensor padding.
    fn build_f32_gguf(tensors: &[(&str, &[f32])]) -> Vec<u8> {
        let mut offsets = Vec::with_capacity(tensors.len());
        let mut offset = 0u64;
        for (_, vals) in tensors {
            offsets.push(offset);
            offset += (vals.len() * 4) as u64;
        }
        let mut b = header(tensors.len() as u64);
        for ((name, vals), off) in tensors.iter().zip(&offsets) {
            str_(&mut b, name);
            b.extend_from_slice(&1u32.to_le_bytes()); // n_dims
            b.extend_from_slice(&(vals.len() as u64).to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes()); // type F32
            b.extend_from_slice(&off.to_le_bytes());
        }
        let pad = b.len().next_multiple_of(DEFAULT_ALIGNMENT) - b.len();
        b.resize(b.len() + pad, 0);
        for (_, vals) in tensors {
            for &v in *vals {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        b
    }

    #[test]
    fn sharded_gguf_matches_equivalent_single_file() {
        let t_a = [1.0f32, 2.0, 3.0];
        let t_b = [4.0f32, 5.0];
        let t_c = [6.0f32, 7.0, 8.0, 9.0];

        // Shard 0 holds tensors a,b; shard 1 holds tensor c.
        let shard0 = build_f32_gguf(&[("a", &t_a), ("b", &t_b)]);
        let shard1 = build_f32_gguf(&[("c", &t_c)]);
        let sharded = Gguf::parse_sharded(&[&shard0, &shard1]).unwrap();

        // The equivalent single file with all three tensors together.
        let single = build_f32_gguf(&[("a", &t_a), ("b", &t_b), ("c", &t_c)]);
        let combined = Gguf::parse(&single).unwrap();

        for name in ["a", "b", "c"] {
            let s_info = sharded.tensor(name).expect("sharded lookup");
            let c_info = combined.tensor(name).expect("combined lookup");
            let s_bytes = sharded.tensor_bytes(s_info).unwrap();
            let c_bytes = combined.tensor_bytes(c_info).unwrap();
            assert_eq!(s_bytes, c_bytes, "tensor '{name}' bytes mismatch");
        }
        assert_eq!(sharded.tensor("a").unwrap().shard, 0);
        assert_eq!(sharded.tensor("b").unwrap().shard, 0);
        assert_eq!(sharded.tensor("c").unwrap().shard, 1);
    }

    #[test]
    fn sharded_gguf_rejects_duplicate_name_across_shards() {
        let shard0 = build_f32_gguf(&[("w", &[1.0f32])]);
        let shard1 = build_f32_gguf(&[("w", &[2.0f32])]);
        let err = match Gguf::parse_sharded(&[&shard0, &shard1]) {
            Ok(_) => panic!("expected duplicate-name error"),
            Err(e) => format!("{e:?}"),
        };
        assert!(err.contains("duplicate"), "{err}");
    }

    #[test]
    fn sharded_gguf_rejects_empty_shard_list() {
        assert!(Gguf::parse_sharded(&[]).is_err());
    }
}
