//! GGML tensor types and dequantization.
//!
//! GGUF weights are stored in blocked quantization formats. Each format packs a
//! fixed number of values per block together with one or more scales. We expose
//! both whole-tensor dequantization ([`dequantize`]) and per-block
//! dequantization ([`dequant_block`]) — the latter lets the matmul kernel
//! decompress one block at a time and keep the weights compressed in RAM.
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

/// Largest block size of any supported type (sizing for stack scratch buffers).
pub const MAX_BLOCK: usize = QK_K;

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

    /// The raw `ggml_type` discriminant (inverse of [`GgmlType::from_u32`]).
    pub fn to_u32(self) -> u32 {
        match self {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q4_0 => 2,
            GgmlType::Q8_0 => 8,
            GgmlType::Q4_K => 12,
            GgmlType::Q6_K => 14,
        }
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
    let mut out = vec![0.0; n];
    dequantize_into(ty, src, &mut out)?;
    Ok(out)
}

/// Dequantize into a caller-provided buffer; `out.len()` elements are written.
pub fn dequantize_into(ty: GgmlType, src: &[u8], out: &mut [f32]) -> Result<()> {
    let n = out.len();
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
    let ts = ty.type_size();
    for (b, chunk) in src[..need].chunks_exact(ts).enumerate() {
        dequant_block(ty, chunk, &mut out[b * block..b * block + block]);
    }
    Ok(())
}

/// Dequantize a single block. `chunk.len() >= ty.type_size()` and
/// `out.len() == ty.block_size()`.
pub fn dequant_block(ty: GgmlType, chunk: &[u8], out: &mut [f32]) {
    match ty {
        GgmlType::F32 => out[0] = f32::from_le_bytes(chunk[0..4].try_into().unwrap()),
        GgmlType::F16 => out[0] = rd_f16(chunk, 0),
        GgmlType::Q4_0 => block_q4_0(chunk, out),
        GgmlType::Q8_0 => block_q8_0(chunk, out),
        GgmlType::Q4_K => block_q4_k(chunk, out),
        GgmlType::Q6_K => block_q6_k(chunk, out),
    }
}

// --- IEEE 754 half precision ------------------------------------------------

/// Convert an IEEE-754 binary16 value to `f32`.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let val = if exp == 0 {
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
            return sign;
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
        sign | 0x7c00
    } else {
        let mut half = ((exp as u16) << 10) | ((mant >> 13) as u16);
        let round_bit = 1u32 << 12;
        if mant & round_bit != 0 && (mant & (round_bit - 1) != 0 || half & 1 != 0) {
            half += 1;
        }
        sign | half
    }
}

// --- per-block dequantizers -------------------------------------------------

#[inline]
fn rd_f16(src: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([src[i], src[i + 1]]))
}

fn block_q8_0(blk: &[u8], out: &mut [f32]) {
    let d = rd_f16(blk, 0);
    for (o, &q) in out.iter_mut().zip(&blk[2..34]) {
        *o = d * (q as i8) as f32;
    }
}

fn block_q4_0(blk: &[u8], out: &mut [f32]) {
    let d = rd_f16(blk, 0);
    let qs = &blk[2..18];
    for j in 0..16 {
        out[j] = ((qs[j] & 0x0f) as i32 - 8) as f32 * d;
        out[j + 16] = ((qs[j] >> 4) as i32 - 8) as f32 * d;
    }
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

fn block_q4_k(blk: &[u8], out: &mut [f32]) {
    let d = rd_f16(blk, 0);
    let dmin = rd_f16(blk, 2);
    let scales = &blk[4..16];
    let qs = &blk[16..144];

    let mut y = 0;
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

// `>> 0` is kept for visual symmetry with the `>> 2/4/6` shifts below.
#[allow(clippy::identity_op)]
fn block_q6_k(blk: &[u8], out: &mut [f32]) {
    let scales = &blk[192..208];
    let d = rd_f16(blk, 208);

    for half in 0..2 {
        let ql = &blk[half * 64..];
        let qh = &blk[128 + half * 32..];
        let sc = &scales[half * 8..];
        let y = half * 128;
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

// --- integer activation dot product -----------------------------------------

/// An activation vector quantized to signed 8-bit, one scale per 32 values.
///
/// llama.cpp quantizes the activation once per matmul and then does the row
/// dot products in integer arithmetic; this is the building block for that.
pub struct Q8Activation {
    /// Quantized values (length `n`, a multiple of 32).
    pub qs: Vec<i8>,
    /// Per-block scale (`qs.len() / 32` entries).
    pub scales: Vec<f32>,
}

impl Q8Activation {
    /// Reconstruct the (lossy) f32 values — used to check the integer kernels.
    pub fn dequantized(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.qs.len());
        for (b, chunk) in self.qs.chunks_exact(32).enumerate() {
            for &q in chunk {
                out.push(self.scales[b] * q as f32);
            }
        }
        out
    }
}

/// Quantize an activation vector (length a multiple of 32) to [`Q8Activation`].
pub fn quantize_activation_q8(x: &[f32]) -> Q8Activation {
    debug_assert!(x.len().is_multiple_of(32));
    let mut qs = Vec::with_capacity(x.len());
    let mut scales = Vec::with_capacity(x.len() / 32);
    for blk in x.chunks_exact(32) {
        let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        scales.push(d);
        for &v in blk {
            qs.push(((v * id).round() as i32).clamp(-128, 127) as i8);
        }
    }
    Q8Activation { qs, scales }
}

/// Dot product of a Q8_0 weight row with a Q8-quantized activation.
///
/// Equals `dequantize(Q8_0, weight) · act.dequantized()`, but the per-block
/// inner sum is done in integer arithmetic with a single f32 scale at the end.
pub fn vec_dot_q8_0(weight: &[u8], act: &Q8Activation) -> f32 {
    let mut acc = 0.0f32;
    for (b, blk) in weight.chunks_exact(34).enumerate() {
        let dw = rd_f16(blk, 0);
        let base = b * 32;
        let mut sum: i32 = 0;
        for i in 0..32 {
            sum += (blk[2 + i] as i8 as i32) * (act.qs[base + i] as i32);
        }
        acc += dw * act.scales[b] * sum as f32;
    }
    acc
}

/// Dot product of a Q4_0 weight row with a Q8-quantized activation.
pub fn vec_dot_q4_0(weight: &[u8], act: &Q8Activation) -> f32 {
    let mut acc = 0.0f32;
    for (b, blk) in weight.chunks_exact(18).enumerate() {
        let dw = rd_f16(blk, 0);
        let qs = &blk[2..18];
        let base = b * 32;
        let mut sum: i32 = 0;
        for (j, &q) in qs.iter().enumerate() {
            let lo = (q & 0x0f) as i32 - 8;
            let hi = (q >> 4) as i32 - 8;
            sum += lo * (act.qs[base + j] as i32);
            sum += hi * (act.qs[base + j + 16] as i32);
        }
        acc += dw * act.scales[b] * sum as f32;
    }
    acc
}

/// True if `ty`'s matmul has an integer (Q8-activation) fast path.
pub fn has_int8_path(ty: GgmlType) -> bool {
    matches!(ty, GgmlType::Q8_0 | GgmlType::Q4_0)
}

// --- quantizers -------------------------------------------------------------

/// Quantize `x` (length a multiple of 32) to Q8_0, matching ggml's reference.
///
/// Handy for building quantized test fixtures and demo GGUFs.
pub fn quantize_q8_0(x: &[f32]) -> Vec<u8> {
    assert!(x.len().is_multiple_of(32), "Q8_0 length must be a multiple of 32");
    let mut out = Vec::with_capacity(x.len() / 32 * 34);
    for blk in x.chunks_exact(32) {
        let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        for &v in blk {
            out.push(((v * id).round() as i32).clamp(-128, 127) as i8 as u8);
        }
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
        let block = quantize_q8_0(&x);
        assert_eq!(block.len(), GgmlType::Q8_0.type_size());
        let y = dequantize(GgmlType::Q8_0, &block, 32).unwrap();
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 0.05, "{a} vs {b}");
        }
    }

    #[test]
    fn q4_k_constant_block() {
        let mut blk = Vec::new();
        blk.extend_from_slice(&f32_to_f16(1.0).to_le_bytes());
        blk.extend_from_slice(&f32_to_f16(0.0).to_le_bytes());
        blk.extend_from_slice(&pack_scales_q4_k([1; 8], [0; 8]));
        blk.extend(std::iter::repeat_n(0x11u8, QK_K / 2));
        assert_eq!(blk.len(), GgmlType::Q4_K.type_size());

        let y = dequantize(GgmlType::Q4_K, &blk, QK_K).unwrap();
        assert!(y.iter().all(|&v| (v - 1.0).abs() < 1e-4), "{:?}", &y[..4]);
    }

    #[test]
    fn q6_k_constant_block() {
        let mut blk = Vec::new();
        blk.extend(std::iter::repeat_n(0x11u8, 128));
        blk.extend(std::iter::repeat_n(0u8, 64));
        blk.extend(std::iter::repeat_n(1u8, 16));
        blk.extend_from_slice(&f32_to_f16(1.0).to_le_bytes());
        assert_eq!(blk.len(), GgmlType::Q6_K.type_size());

        let y = dequantize(GgmlType::Q6_K, &blk, QK_K).unwrap();
        assert!(y.iter().all(|&v| (v + 31.0).abs() < 1e-4), "{:?}", &y[..4]);
    }

    #[test]
    fn rejects_bad_lengths() {
        assert!(dequantize(GgmlType::Q8_0, &[0; 34], 31).is_err());
        assert!(dequantize(GgmlType::Q8_0, &[0; 10], 32).is_err());
    }

    #[test]
    fn vec_dot_q8_0_equals_dequantized_dot() {
        let cols = 64; // two blocks
        let wf: Vec<f32> = (0..cols).map(|i| ((i * 13 % 17) as f32 - 8.0) * 0.3).collect();
        let wbytes = quantize_q8_0(&wf);
        let x: Vec<f32> = (0..cols).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.2).collect();
        let act = quantize_activation_q8(&x);

        let wdq = dequantize(GgmlType::Q8_0, &wbytes, cols).unwrap();
        let reference: f32 = wdq.iter().zip(&act.dequantized()).map(|(a, b)| a * b).sum();
        let got = vec_dot_q8_0(&wbytes, &act);
        assert!((got - reference).abs() < 1e-2, "{got} vs {reference}");
    }

    #[test]
    fn vec_dot_q4_0_equals_dequantized_dot() {
        let cols = 64; // two blocks of 18 bytes
        let mut wbytes = Vec::new();
        for b in 0..2 {
            wbytes.extend_from_slice(&f32_to_f16(0.05 + b as f32 * 0.01).to_le_bytes());
            for j in 0..16 {
                wbytes.push(((b * 16 + j) * 7 % 256) as u8); // arbitrary nibbles
            }
        }
        let x: Vec<f32> = (0..cols).map(|i| ((i * 5 % 9) as f32 - 4.0) * 0.1).collect();
        let act = quantize_activation_q8(&x);

        let wdq = dequantize(GgmlType::Q4_0, &wbytes, cols).unwrap();
        let reference: f32 = wdq.iter().zip(&act.dequantized()).map(|(a, b)| a * b).sum();
        let got = vec_dot_q4_0(&wbytes, &act);
        assert!((got - reference).abs() < 1e-2, "{got} vs {reference}");
    }
}
