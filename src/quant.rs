//! GGML tensor types and dequantization.
//!
//! GGUF weights are stored in blocked quantization formats. Each format packs a
//! fixed number of values per block together with one or more scales. Here we
//! implement the *type table* plus dequantization back to `f32` for the formats
//! that show up in the common llama.cpp builds:
//!
//! | type   | values/block | bytes/block | notes                         |
//! |--------|-------------:|------------:|-------------------------------|
//! | F32    | 1            | 4           | raw                           |
//! | F16    | 1            | 2           | half precision                |
//! | Q4_0   | 32           | 18          | 4-bit, single scale           |
//! | Q8_0   | 32           | 34          | 8-bit, single scale           |
//! | Q4_K   | 256          | 144         | k-quant, 8 sub-block scales   |
//! | Q6_K   | 256          | 210         | k-quant, 16 sub-block scales  |

use crate::error::{Error, Result};

/// Number of elements in a k-quant super-block.
pub const QK_K: usize = 256;

/// The subset of `ggml_type` we can read.
///
/// Variant names follow ggml's own type names (`Q4_K`, `Q6_K`, …) so they map
/// one-to-one onto the format documentation, hence the lint allowance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q8_0,
    Q4_K,
    Q6_K,
}

impl GgmlType {
    /// Map a raw `ggml_type` discriminant to a [`GgmlType`].
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            8 => GgmlType::Q8_0,
            12 => GgmlType::Q4_K,
            14 => GgmlType::Q6_K,
            other => {
                return Err(Error::Format(format!(
                    "unsupported ggml tensor type {other}"
                )))
            }
        })
    }

    /// Number of elements stored per block.
    pub fn block_size(self) -> usize {
        match self {
            GgmlType::F32 | GgmlType::F16 => 1,
            GgmlType::Q4_0 | GgmlType::Q8_0 => 32,
            GgmlType::Q4_K | GgmlType::Q6_K => QK_K,
        }
    }

    /// Number of bytes occupied per block.
    pub fn type_size(self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            GgmlType::Q4_0 => 18,
            GgmlType::Q8_0 => 34,
            GgmlType::Q4_K => 144,
            GgmlType::Q6_K => 210,
        }
    }

    /// Bytes required to store `n` elements of this type.
    pub fn bytes_for(self, n: usize) -> usize {
        (n / self.block_size()) * self.type_size()
    }
}

/// Dequantize `n` elements of type `ty` from `src` into a fresh `f32` vector.
pub fn dequantize(ty: GgmlType, src: &[u8], n: usize) -> Result<Vec<f32>> {
    let block = ty.block_size();
    if !n.is_multiple_of(block) {
        return Err(Error::Format(format!(
            "element count {n} is not a multiple of {ty:?} block size {block}"
        )));
    }
    let need = ty.bytes_for(n);
    if src.len() < need {
        return Err(Error::Format(format!(
            "tensor data too short: need {need} bytes for {n} {ty:?} elements, have {}",
            src.len()
        )));
    }
    let src = &src[..need];
    Ok(match ty {
        GgmlType::F32 => dequant_f32(src, n),
        GgmlType::F16 => dequant_f16(src, n),
        GgmlType::Q4_0 => dequant_q4_0(src, n),
        GgmlType::Q8_0 => dequant_q8_0(src, n),
        GgmlType::Q4_K => dequant_q4_k(src, n),
        GgmlType::Q6_K => dequant_q6_k(src, n),
    })
}

// --- IEEE 754 half precision ------------------------------------------------

/// Convert an IEEE-754 binary16 value to `f32`.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let val = if exp == 0 {
        // zero or subnormal
        (mant as f32) * 2.0f32.powi(-24)
    } else if exp == 0x1f {
        if mant == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + mant as f32 / 1024.0) * 2.0f32.powi(exp as i32 - 15)
    };
    if sign == 1 {
        -val
    } else {
        val
    }
}

/// Convert an `f32` to IEEE-754 binary16 (round to nearest, ties to even).
///
/// Used to build quantized test fixtures; not on any hot path.
pub fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;

    if exp <= 0 {
        if exp < -10 {
            return sign; // underflow to zero
        }
        let m = mant | 0x80_0000;
        let shift = (14 - exp) as u32;
        let mut half = (m >> shift) as u16;
        let round_bit = 1u32 << (shift - 1);
        if m & round_bit != 0 && (m & (round_bit - 1) != 0 || half & 1 != 0) {
            half += 1;
        }
        sign | half
    } else if exp >= 0x1f {
        sign | 0x7c00 // overflow to inf
    } else {
        let mut half = ((exp as u16) << 10) | ((mant >> 13) as u16);
        let round_bit = 1u32 << 12;
        if mant & round_bit != 0 && (mant & (round_bit - 1) != 0 || half & 1 != 0) {
            half += 1; // carry into the exponent is intentional and correct
        }
        sign | half
    }
}

// --- dequantizers -----------------------------------------------------------

#[inline]
fn rd_f16(src: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([src[i], src[i + 1]]))
}

fn dequant_f32(src: &[u8], n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| f32::from_le_bytes(src[i * 4..i * 4 + 4].try_into().unwrap()))
        .collect()
}

fn dequant_f16(src: &[u8], n: usize) -> Vec<f32> {
    (0..n).map(|i| rd_f16(src, i * 2)).collect()
}

fn dequant_q8_0(src: &[u8], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for blk in src.chunks_exact(34) {
        let d = rd_f16(blk, 0);
        for &q in &blk[2..34] {
            out.push(d * (q as i8) as f32);
        }
    }
    out
}

fn dequant_q4_0(src: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0; n];
    for (b, blk) in src.chunks_exact(18).enumerate() {
        let d = rd_f16(blk, 0);
        let qs = &blk[2..18];
        let base = b * 32;
        for j in 0..16 {
            let x0 = (qs[j] & 0x0f) as i32 - 8;
            let x1 = (qs[j] >> 4) as i32 - 8;
            out[base + j] = x0 as f32 * d;
            out[base + j + 16] = x1 as f32 * d;
        }
    }
    out
}

/// Unpack the 6-bit scale and min for sub-block `j` from the packed 12 bytes.
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

fn dequant_q4_k(src: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0; n];
    for (b, blk) in src.chunks_exact(144).enumerate() {
        let d = rd_f16(blk, 0);
        let dmin = rd_f16(blk, 2);
        let scales = &blk[4..16];
        let qs = &blk[16..144];

        let mut y = b * QK_K;
        let mut is = 0;
        for q in qs.chunks_exact(32) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let (d1, min1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, min2) = (d * sc2 as f32, dmin * m2 as f32);
            for &b in q {
                out[y] = d1 * (b & 0x0f) as f32 - min1;
                y += 1;
            }
            for &b in q {
                out[y] = d2 * (b >> 4) as f32 - min2;
                y += 1;
            }
            is += 2;
        }
    }
    out
}

// `>> 0` is kept for visual symmetry with the `>> 2/4/6` shifts below.
#[allow(clippy::identity_op)]
fn dequant_q6_k(src: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0; n];
    for (b, blk) in src.chunks_exact(210).enumerate() {
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let scales = &blk[192..208];
        let d = rd_f16(blk, 208);
        let base = b * QK_K;

        for half in 0..2 {
            let ql = &ql[half * 64..];
            let qh = &qh[half * 32..];
            let sc = &scales[half * 8..];
            let y = base + half * 128;
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0f) | (((qh[l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                out[y + l] = d * sc[is] as f32 * q1 as f32;
                out[y + l + 32] = d * sc[is + 2] as f32 * q2 as f32;
                out[y + l + 64] = d * sc[is + 4] as f32 * q3 as f32;
                out[y + l + 96] = d * sc[is + 6] as f32 * q4 as f32;
            }
        }
    }
    out
}

// --- quantizers (test fixtures only) ----------------------------------------

/// Quantize one 32-element block to Q8_0, matching ggml's reference.
#[cfg(test)]
pub(crate) fn quantize_block_q8_0(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len(), 32);
    let amax = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let mut out = Vec::with_capacity(34);
    out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    for &v in x {
        out.push(((v * id).round() as i32).clamp(-128, 127) as i8 as u8);
    }
    out
}

/// Pack 8 sub-block scales and mins (each 0..=63) into the Q4_K 12-byte layout.
#[cfg(test)]
pub(crate) fn pack_scales_q4_k(sc: [u8; 8], m: [u8; 8]) -> [u8; 12] {
    let mut q = [0u8; 12];
    for i in 0..4 {
        q[i] = (sc[i] & 63) | ((sc[i + 4] >> 4) << 6);
        q[i + 4] = (m[i] & 63) | ((m[i + 4] >> 4) << 6);
        q[i + 8] = (sc[i + 4] & 0x0f) | ((m[i + 4] & 0x0f) << 4);
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_simple_values() {
        for &v in &[0.0f32, 1.0, -1.0, 0.5, -2.5, 100.0, 0.125] {
            let back = f16_to_f32(f32_to_f16(v));
            assert!((back - v).abs() < 1e-3, "{v} -> {back}");
        }
    }

    #[test]
    fn q8_0_roundtrip() {
        let x: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.3).collect();
        let block = quantize_block_q8_0(&x);
        assert_eq!(block.len(), GgmlType::Q8_0.type_size());
        let y = dequantize(GgmlType::Q8_0, &block, 32).unwrap();
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 0.05, "{a} vs {b}");
        }
    }

    #[test]
    fn q4_k_constant_block() {
        // d=1, dmin=0, every sub-block scale=1, every nibble=1 -> all ones.
        let mut blk = Vec::new();
        blk.extend_from_slice(&f32_to_f16(1.0).to_le_bytes()); // d
        blk.extend_from_slice(&f32_to_f16(0.0).to_le_bytes()); // dmin
        blk.extend_from_slice(&pack_scales_q4_k([1; 8], [0; 8]));
        blk.extend(std::iter::repeat_n(0x11u8, QK_K / 2)); // nibbles all 1
        assert_eq!(blk.len(), GgmlType::Q4_K.type_size());

        let y = dequantize(GgmlType::Q4_K, &blk, QK_K).unwrap();
        assert!(y.iter().all(|&v| (v - 1.0).abs() < 1e-4), "{:?}", &y[..4]);
    }

    #[test]
    fn q6_k_constant_block() {
        // d=1, scales=1, ql nibbles=1, qh=0 -> q = 1, y = 1*1*(1-32) = -31.
        let mut blk = Vec::new();
        blk.extend(std::iter::repeat_n(0x11u8, 128)); // ql
        blk.extend(std::iter::repeat_n(0u8, 64)); // qh
        blk.extend(std::iter::repeat_n(1u8, 16)); // scales (i8 = 1)
        blk.extend_from_slice(&f32_to_f16(1.0).to_le_bytes()); // d
        assert_eq!(blk.len(), GgmlType::Q6_K.type_size());

        let y = dequantize(GgmlType::Q6_K, &blk, QK_K).unwrap();
        assert!(y.iter().all(|&v| (v + 31.0).abs() < 1e-4), "{:?}", &y[..4]);
    }

    #[test]
    fn rejects_bad_lengths() {
        assert!(dequantize(GgmlType::Q8_0, &[0; 34], 31).is_err()); // not a multiple
        assert!(dequantize(GgmlType::Q8_0, &[0; 10], 32).is_err()); // too short
    }
}
