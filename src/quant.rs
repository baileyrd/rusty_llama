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
//! | Q4_1   | 32           | 20          | 4-bit, scale + min             |
//! | Q5_0   | 32           | 22          | 5-bit (4-bit + high plane), single scale |
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
    Q4_1,
    Q5_0,
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
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
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
            GgmlType::Q4_1 => 3,
            GgmlType::Q5_0 => 6,
            GgmlType::Q8_0 => 8,
            GgmlType::Q4_K => 12,
            GgmlType::Q6_K => 14,
        }
    }

    /// Number of elements stored per block.
    pub fn block_size(self) -> usize {
        match self {
            GgmlType::F32 | GgmlType::F16 => 1,
            GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q8_0 => 32,
            GgmlType::Q4_K | GgmlType::Q6_K => QK_K,
        }
    }

    /// Number of bytes occupied per block.
    pub fn type_size(self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            GgmlType::Q4_0 => 18,
            GgmlType::Q4_1 => 20,
            GgmlType::Q5_0 => 22,
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
        GgmlType::Q4_1 => block_q4_1(chunk, out),
        GgmlType::Q5_0 => block_q5_0(chunk, out),
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

/// `Q4_1`: like [`block_q4_0`] but with an explicit `(delta, min)` pair instead
/// of an implicit `-8` zero point — `x = q*d + m`, `q` an unsigned nibble.
fn block_q4_1(blk: &[u8], out: &mut [f32]) {
    let d = rd_f16(blk, 0);
    let m = rd_f16(blk, 2);
    let qs = &blk[4..20];
    for j in 0..16 {
        out[j] = (qs[j] & 0x0f) as f32 * d + m;
        out[j + 16] = (qs[j] >> 4) as f32 * d + m;
    }
}

/// `Q5_0`: [`block_q4_0`]'s 4-bit nibble plus a 5th ("high") bit packed
/// separately in a 32-bit `qh` plane, giving a 5-bit signed value in
/// `[-16, 15]` (single scale, no min — like `Q4_0`, just one more bit).
fn block_q5_0(blk: &[u8], out: &mut [f32]) {
    let d = rd_f16(blk, 0);
    let qh = u32::from_le_bytes(blk[2..6].try_into().unwrap());
    let qs = &blk[6..22];
    for j in 0..16 {
        let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
        let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
        let x0 = ((qs[j] & 0x0f) | xh_0) as i32 - 16;
        let x1 = ((qs[j] >> 4) | xh_1) as i32 - 16;
        out[j] = x0 as f32 * d;
        out[j + 16] = x1 as f32 * d;
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
        let sc = &scales[half * 8..]; // Q6_K scales are signed int8
        let y = half * 128;
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0x0f) | (((qh[l] >> 0) & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            out[y + l] = d * (sc[is] as i8) as f32 * q1 as f32;
            out[y + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
            out[y + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
            out[y + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
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
    /// Sum of the quantized values in each 32-block (`qs.len() / 32` entries).
    ///
    /// Only the AVX-512 VNNI dot products read this: `vpdpbusd` needs an
    /// *unsigned* weight operand, so the signed Q8_0/Q4_0 weights are biased by
    /// `+128` and this exact `Σx` per block lets us subtract the bias back out.
    pub block_sums: Vec<i32>,
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
    let mut block_sums = Vec::with_capacity(x.len() / 32);
    for blk in x.chunks_exact(32) {
        let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        scales.push(d);
        let mut bs: i32 = 0;
        for &v in blk {
            let q = ((v * id).round() as i32).clamp(-128, 127) as i8;
            qs.push(q);
            bs += q as i32;
        }
        block_sums.push(bs);
    }
    Q8Activation {
        qs,
        scales,
        block_sums,
    }
}

/// Dot product of a Q8_0 weight row with a Q8-quantized activation.
///
/// Equals `dequantize(Q8_0, weight) · act.dequantized()`, but the per-block
/// inner sum is done in integer arithmetic with a single f32 scale at the end.
///
/// Dispatches to an AVX-512 VNNI implementation when the CPU supports it (see
/// the [`x86`] module); the SIMD result is *bit-identical* to the scalar one.
pub fn vec_dot_q8_0(weight: &[u8], act: &Q8Activation) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY (both arms): the required features were just feature-detected.
        if x86::vnni_supported() {
            return unsafe { x86::vec_dot_q8_0(weight, act) };
        }
        if x86::avx2_supported() {
            return unsafe { x86::vec_dot_q8_0_avx2(weight, act) };
        }
    }
    vec_dot_q8_0_scalar(weight, act)
}

/// Scalar reference for [`vec_dot_q8_0`] (portable; also the SIMD oracle).
fn vec_dot_q8_0_scalar(weight: &[u8], act: &Q8Activation) -> f32 {
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
///
/// Dispatches to AVX-512 VNNI when available (bit-identical to the scalar path).
pub fn vec_dot_q4_0(weight: &[u8], act: &Q8Activation) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if x86::vnni_supported() {
            return unsafe { x86::vec_dot_q4_0(weight, act) };
        }
        if x86::avx2_supported() {
            return unsafe { x86::vec_dot_q4_0_avx2(weight, act) };
        }
    }
    vec_dot_q4_0_scalar(weight, act)
}

/// Scalar reference for [`vec_dot_q4_0`] (portable; also the SIMD oracle).
fn vec_dot_q4_0_scalar(weight: &[u8], act: &Q8Activation) -> f32 {
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

/// An activation vector quantized to the `Q8_K` format used by the k-quant dot
/// products: one f32 scale per 256-value super-block, plus the sum of quants in
/// each group of 16 (used to fold in the k-quant per-block mins).
pub struct Q8KActivation {
    /// Quantized values (length `n`, a multiple of 256).
    pub qs: Vec<i8>,
    /// Per-super-block scale (`qs.len() / 256` entries).
    pub d: Vec<f32>,
    /// Sum of quants per group of 16 (`qs.len() / 16` entries).
    pub bsums: Vec<i16>,
}

impl Q8KActivation {
    /// Reconstruct the (lossy) f32 values — used to check the integer kernels.
    pub fn dequantized(&self) -> Vec<f32> {
        self.qs
            .iter()
            .enumerate()
            .map(|(i, &q)| self.d[i / 256] * q as f32)
            .collect()
    }
}

/// Quantize an activation vector (length a multiple of 256) to [`Q8KActivation`].
pub fn quantize_activation_q8k(x: &[f32]) -> Q8KActivation {
    debug_assert!(x.len().is_multiple_of(256));
    let mut qs = Vec::with_capacity(x.len());
    let mut d = Vec::with_capacity(x.len() / 256);
    let mut bsums = Vec::with_capacity(x.len() / 16);
    for blk in x.chunks_exact(256) {
        let (mut amax, mut max) = (0.0f32, 0.0f32);
        for &v in blk {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        if amax == 0.0 {
            d.push(0.0);
            qs.extend(std::iter::repeat_n(0i8, 256));
            bsums.extend(std::iter::repeat_n(0i16, 16));
            continue;
        }
        let iscale = -128.0 / max;
        d.push(1.0 / iscale);
        let start = qs.len();
        for &v in blk {
            qs.push(((iscale * v).round() as i32).clamp(-128, 127) as i8);
        }
        for g in 0..16 {
            let s: i32 = (0..16).map(|k| qs[start + g * 16 + k] as i32).sum();
            bsums.push(s as i16);
        }
    }
    Q8KActivation { qs, d, bsums }
}

/// Dot product of a Q4_K weight row with a Q8_K-quantized activation.
///
/// Equals `dequantize(Q4_K, weight) · act.dequantized()`: the per-sub-block
/// products run in integer arithmetic, the per-block min term is folded in via
/// the activation's `bsums`, and only the two block scales hit f32.
///
/// Dispatches to AVX-512 VNNI when available (bit-identical to the scalar path).
pub fn vec_dot_q4_k(weight: &[u8], act: &Q8KActivation) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if x86::vnni_supported() {
            return unsafe { x86::vec_dot_q4_k(weight, act) };
        }
        if x86::avx2_supported() {
            return unsafe { x86::vec_dot_q4_k_avx2(weight, act) };
        }
    }
    vec_dot_q4_k_scalar(weight, act)
}

/// Scalar reference for [`vec_dot_q4_k`] (portable; also the SIMD oracle).
fn vec_dot_q4_k_scalar(weight: &[u8], act: &Q8KActivation) -> f32 {
    let mut total = 0.0f32;
    for (sb, wblk) in weight.chunks_exact(144).enumerate() {
        let d = rd_f16(wblk, 0);
        let dmin = rd_f16(wblk, 2);
        let scales = &wblk[4..16];
        let qs = &wblk[16..144];
        let qx = &act.qs[sb * 256..sb * 256 + 256];
        let bsums = &act.bsums[sb * 16..sb * 16 + 16];
        let bigd = act.d[sb];

        let mut acc: i32 = 0;
        for j in 0..8 {
            let (sc, _) = get_scale_min_k4(j, scales);
            let chunk = j / 2;
            let hi = j % 2 == 1;
            let mut sub: i32 = 0;
            for k in 0..32 {
                let byte = qs[chunk * 32 + k];
                let nib = if hi { (byte >> 4) as i32 } else { (byte & 0x0f) as i32 };
                sub += nib * (qx[j * 32 + k] as i32);
            }
            acc += sc as i32 * sub;
        }
        let mut min_acc: i32 = 0;
        for (g, &bs) in bsums.iter().enumerate() {
            let (_, m) = get_scale_min_k4(g / 2, scales);
            min_acc += bs as i32 * m as i32;
        }
        total += d * bigd * acc as f32 - dmin * bigd * min_acc as f32;
    }
    total
}

/// Dot product of a Q6_K weight row with a Q8_K-quantized activation.
///
/// Equals `dequantize(Q6_K, weight) · act.dequantized()`. Dispatches to AVX-512
/// VNNI when available (bit-identical to the scalar path).
pub fn vec_dot_q6_k(weight: &[u8], act: &Q8KActivation) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if x86::vnni_supported() {
            return unsafe { x86::vec_dot_q6_k(weight, act) };
        }
        if x86::avx2_supported() {
            return unsafe { x86::vec_dot_q6_k_avx2(weight, act) };
        }
    }
    vec_dot_q6_k_scalar(weight, act)
}

/// Scalar reference for [`vec_dot_q6_k`] (portable; also the SIMD oracle).
// `>> 0` is kept for visual symmetry with the `>> 2/4/6` shifts.
#[allow(clippy::identity_op)]
fn vec_dot_q6_k_scalar(weight: &[u8], act: &Q8KActivation) -> f32 {
    let mut total = 0.0f32;
    for (sb, wblk) in weight.chunks_exact(210).enumerate() {
        let ql = &wblk[0..128];
        let qh = &wblk[128..192];
        let scales = &wblk[192..208];
        let d = rd_f16(wblk, 208);
        let qx = &act.qs[sb * 256..sb * 256 + 256];
        let bigd = act.d[sb];

        // Reconstruct the 6-bit signed weights (q - 32) in natural order.
        let mut a = [0i32; 256];
        for half in 0..2 {
            let ql = &ql[half * 64..];
            let qh = &qh[half * 32..];
            let y = half * 128;
            for l in 0..32 {
                a[y + l] = ((ql[l] & 0x0f) | (((qh[l] >> 0) & 3) << 4)) as i32 - 32;
                a[y + l + 32] = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                a[y + l + 64] = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                a[y + l + 96] = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            }
        }
        let mut acc: i32 = 0;
        for sub in 0..16 {
            let sc = scales[sub] as i8 as i32; // scales are signed
            let s: i32 = (0..16).map(|k| a[sub * 16 + k] * qx[sub * 16 + k] as i32).sum();
            acc += sc * s;
        }
        total += d * bigd * acc as f32;
    }
    total
}

/// True if `ty`'s matmul has an integer (Q8-activation) fast path.
pub fn has_int8_path(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::Q8_0 | GgmlType::Q4_0 | GgmlType::Q4_K | GgmlType::Q6_K
    )
}

/// Dot product of two equal-length f32 slices — the CPU flash-attention QK score.
/// Uses an 8-wide AVX2+FMA kernel when available, else a scalar reduction. This is
/// the per-key inner loop of attention, so it dominates at long context.
#[inline]
pub fn f32_dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if x86::avx2f_supported() {
            // SAFETY: gated on runtime AVX2+FMA detection.
            return unsafe { x86::f32_dot_avx2(a, b) };
        }
    }
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// `out[i] = out[i] * alpha + p * v[i]` — the flash-attention online rescale + value
/// accumulate (also per-key). AVX2+FMA when available, else scalar.
#[inline]
pub fn f32_axpy_decay(out: &mut [f32], alpha: f32, p: f32, v: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if x86::avx2f_supported() {
            // SAFETY: gated on runtime AVX2+FMA detection.
            unsafe { x86::f32_axpy_decay_avx2(out, alpha, p, v) };
            return;
        }
    }
    for (o, &vi) in out.iter_mut().zip(v) {
        *o = *o * alpha + p * vi;
    }
}

// --- AVX-512 VNNI integer dot products (x86_64) -----------------------------
//
// The scalar `vec_dot_*` kernels above already autovectorize under
// `target-cpu=native`. On CPUs with AVX-512 VNNI we can do better: a single
// `vpdpbusd` (`_mm256_dpbusd_epi32`) multiplies 32 `u8`×`i8` byte pairs and
// accumulates the products into i32 lanes — the exact integer inner product the
// scalar loop computes by hand. Because the whole reduction stays in *integer*
// arithmetic and the surrounding per-block f32 scaling is left untouched, the
// SIMD result is bit-for-bit identical to the scalar one (asserted by the
// `*_simd_matches_scalar` tests), so this is a pure speed swap that the
// dispatchers above select at run time via `is_x86_feature_detected!`.
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86 {
    use super::{get_scale_min_k4, rd_f16, Q8Activation, Q8KActivation};
    use core::arch::x86_64::*;

    /// Whether the AVX2+FMA f32 attention kernels apply (cached). The
    /// `RUSTY_LLAMA_NO_AVX2F` escape hatch forces the scalar path (A/B benchmarking).
    pub fn avx2f_supported() -> bool {
        static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *OK.get_or_init(|| {
            std::env::var_os("RUSTY_LLAMA_NO_AVX2F").is_none()
                && is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
        })
    }

    /// Horizontal sum of the eight f32 lanes of a 256-bit register.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_f32_x8(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_hadd_ps(s, s);
        let s = _mm_hadd_ps(s, s);
        _mm_cvtss_f32(s)
    }

    /// Dot product of two equal-length f32 slices, 8-wide with an FMA accumulator.
    /// # Safety
    /// AVX2+FMA must be available (see [`avx2f_supported`]).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn f32_dot_avx2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc);
            i += 8;
        }
        let mut s = hsum_f32_x8(acc);
        while i < n {
            s += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        s
    }

    /// `out[i] = out[i]*alpha + p*v[i]`, 8-wide with FMA.
    /// # Safety
    /// AVX2+FMA must be available (see [`avx2f_supported`]).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn f32_axpy_decay_avx2(out: &mut [f32], alpha: f32, p: f32, v: &[f32]) {
        debug_assert_eq!(out.len(), v.len());
        let n = out.len();
        let (va, vp) = (_mm256_set1_ps(alpha), _mm256_set1_ps(p));
        let (po, pv) = (out.as_mut_ptr(), v.as_ptr());
        let mut i = 0;
        while i + 8 <= n {
            let prod = _mm256_mul_ps(vp, _mm256_loadu_ps(pv.add(i)));
            let r = _mm256_fmadd_ps(_mm256_loadu_ps(po.add(i)), va, prod);
            _mm256_storeu_ps(po.add(i), r);
            i += 8;
        }
        while i < n {
            *po.add(i) = *po.add(i) * alpha + p * *pv.add(i);
            i += 1;
        }
    }
    use std::sync::OnceLock;

    /// Whether the VNNI kernels should be used: this CPU has the required
    /// AVX-512 subset (`vpdpbusd` plus the 256-bit byte ops around it) and the
    /// `RUSTY_LLAMA_NO_VNNI` escape hatch is unset. Cached after the first query
    /// so the hot dispatch path is a single load. The env var lets you A/B the
    /// SIMD path against the scalar fallback end to end (`RUSTY_LLAMA_NO_VNNI=1`).
    pub fn vnni_supported() -> bool {
        static OK: OnceLock<bool> = OnceLock::new();
        *OK.get_or_init(|| {
            std::env::var_os("RUSTY_LLAMA_NO_VNNI").is_none()
                && is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vl")
                && is_x86_feature_detected!("avx512vnni")
        })
    }

    /// Horizontal sum of the eight i32 lanes of a 256-bit vector.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_i32_x8(v: __m256i) -> i32 {
        let s = _mm_add_epi32(
            _mm256_castsi256_si128(v),
            _mm256_extracti128_si256::<1>(v),
        );
        let s = _mm_hadd_epi32(s, s);
        let s = _mm_hadd_epi32(s, s);
        _mm_cvtsi128_si32(s)
    }

    /// Exact `Σ_{i<32} w[i]·x[i]` for one 32-element block where both operands
    /// are signed `i8`. `vpdpbusd` needs an *unsigned* weight operand, so `w` is
    /// biased by `+128` (`^ 0x80`) and the bias removed exactly afterwards using
    /// the precomputed `xsum = Σx`: `Σ(w+128)·x = Σ w·x + 128·Σx`.
    #[inline]
    #[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
    unsafe fn dot32_signed(w: __m256i, x: __m256i, xsum: i32) -> i32 {
        let wu = _mm256_xor_si256(w, _mm256_set1_epi8(0x80u8 as i8));
        let dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), wu, x);
        hsum_i32_x8(dot) - 128 * xsum
    }

    /// AVX-512 VNNI implementation of [`super::vec_dot_q8_0`].
    ///
    /// # Safety
    /// [`vnni_supported`] must return true on this CPU.
    #[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
    pub unsafe fn vec_dot_q8_0(weight: &[u8], act: &Q8Activation) -> f32 {
        let mut acc = 0.0f32;
        for (b, blk) in weight.chunks_exact(34).enumerate() {
            let dw = rd_f16(blk, 0);
            let w = _mm256_loadu_si256(blk.as_ptr().add(2) as *const __m256i);
            let x = _mm256_loadu_si256(act.qs.as_ptr().add(b * 32) as *const __m256i);
            let sum = dot32_signed(w, x, act.block_sums[b]);
            acc += dw * act.scales[b] * sum as f32;
        }
        acc
    }

    /// AVX-512 VNNI implementation of [`super::vec_dot_q4_0`].
    ///
    /// # Safety
    /// [`vnni_supported`] must return true on this CPU.
    #[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
    pub unsafe fn vec_dot_q4_0(weight: &[u8], act: &Q8Activation) -> f32 {
        let lomask = _mm256_set1_epi8(0x0f);
        let eight = _mm256_set1_epi8(8);
        let mut acc = 0.0f32;
        for (b, blk) in weight.chunks_exact(18).enumerate() {
            let dw = rd_f16(blk, 0);
            // The 16 packed bytes hold element j's low nibble (output index j)
            // and high nibble (output index j+16). Broadcast them to both
            // halves, keep low nibbles in the lower half and high nibbles in the
            // upper half, then subtract the Q4_0 zero point of 8.
            let packed = _mm_loadu_si128(blk.as_ptr().add(2) as *const __m128i);
            let v = _mm256_broadcastsi128_si256(packed);
            let lo = _mm256_and_si256(v, lomask);
            let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), lomask);
            let nibs = _mm256_blend_epi32::<0xF0>(lo, hi);
            let w = _mm256_sub_epi8(nibs, eight);
            let x = _mm256_loadu_si256(act.qs.as_ptr().add(b * 32) as *const __m256i);
            let sum = dot32_signed(w, x, act.block_sums[b]);
            acc += dw * act.scales[b] * sum as f32;
        }
        acc
    }

    /// AVX-512 VNNI implementation of [`super::vec_dot_q4_k`].
    ///
    /// Q4_K nibbles are unsigned `[0,15]`, so they feed `vpdpbusd` directly with
    /// no bias; only the per-block min term (folded in via `bsums`) stays
    /// scalar, exactly as in the reference.
    ///
    /// # Safety
    /// [`vnni_supported`] must return true on this CPU.
    #[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
    pub unsafe fn vec_dot_q4_k(weight: &[u8], act: &Q8KActivation) -> f32 {
        let lomask = _mm256_set1_epi8(0x0f);
        let mut total = 0.0f32;
        for (sb, wblk) in weight.chunks_exact(144).enumerate() {
            let d = rd_f16(wblk, 0);
            let dmin = rd_f16(wblk, 2);
            let scales = &wblk[4..16];
            let qs = wblk[16..144].as_ptr();
            let qx = act.qs.as_ptr().add(sb * 256);
            let bigd = act.d[sb];

            // Eight sub-blocks (two per 32-byte chunk: low then high nibbles),
            // each scaled by its 6-bit `sc` and summed into `acc` in i32 — the
            // same order the scalar loop uses, so `acc` is identical.
            let mut acc: i32 = 0;
            for chunk in 0..4 {
                let v = _mm256_loadu_si256(qs.add(chunk * 32) as *const __m256i);
                let lo = _mm256_and_si256(v, lomask);
                let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), lomask);
                let xlo = _mm256_loadu_si256(qx.add((2 * chunk) * 32) as *const __m256i);
                let xhi = _mm256_loadu_si256(qx.add((2 * chunk + 1) * 32) as *const __m256i);
                let dlo = hsum_i32_x8(_mm256_dpbusd_epi32(_mm256_setzero_si256(), lo, xlo));
                let dhi = hsum_i32_x8(_mm256_dpbusd_epi32(_mm256_setzero_si256(), hi, xhi));
                let (sc_lo, _) = get_scale_min_k4(2 * chunk, scales);
                let (sc_hi, _) = get_scale_min_k4(2 * chunk + 1, scales);
                acc += sc_lo as i32 * dlo;
                acc += sc_hi as i32 * dhi;
            }
            let bsums = &act.bsums[sb * 16..sb * 16 + 16];
            let mut min_acc: i32 = 0;
            for (g, &bs) in bsums.iter().enumerate() {
                let (_, m) = get_scale_min_k4(g / 2, scales);
                min_acc += bs as i32 * m as i32;
            }
            total += d * bigd * acc as f32 - dmin * bigd * min_acc as f32;
        }
        total
    }

    /// Horizontal sum of the four i32 lanes of a 128-bit vector.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_i32_x4(v: __m128i) -> i32 {
        let s = _mm_hadd_epi32(v, v);
        let s = _mm_hadd_epi32(s, s);
        _mm_cvtsi128_si32(s)
    }

    /// AVX-512 VNNI implementation of [`super::vec_dot_q6_k`].
    ///
    /// The 6-bit weights are reconstructed (in natural order) to unsigned
    /// `[0,63]` — directly usable as `vpdpbusd`'s unsigned operand, no `+128`
    /// bias. Each reconstructed 32-element group is one `vpdpbusd`; its eight i32
    /// lanes split into the two 16-element scale sub-blocks (lanes 0–3 / 4–7).
    /// The weights' `-32` zero point is corrected exactly per sub-block via the
    /// activation's per-16 `bsums` (`Σ(q-32)·x = Σq·x − 32·Σx`), so `acc` is
    /// accumulated in the same order and is bit-identical to the scalar path.
    ///
    /// # Safety
    /// [`vnni_supported`] must return true on this CPU.
    #[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
    pub unsafe fn vec_dot_q6_k(weight: &[u8], act: &Q8KActivation) -> f32 {
        let lomask = _mm256_set1_epi8(0x0f);
        let three = _mm256_set1_epi8(3);
        let mut total = 0.0f32;
        for (sb, wblk) in weight.chunks_exact(210).enumerate() {
            let ql = wblk.as_ptr();
            let qh = wblk.as_ptr().add(128);
            let scales = &wblk[192..208];
            let d = rd_f16(wblk, 208);
            let qx = act.qs.as_ptr().add(sb * 256);
            let bsums = &act.bsums[sb * 16..sb * 16 + 16];
            let bigd = act.d[sb];

            let mut acc: i32 = 0;
            for half in 0..2usize {
                let ql_lo = _mm256_loadu_si256(ql.add(half * 64) as *const __m256i);
                let ql_hi = _mm256_loadu_si256(ql.add(half * 64 + 32) as *const __m256i);
                let qhv = _mm256_loadu_si256(qh.add(half * 32) as *const __m256i);
                // Reconstruct the four 32-element groups (natural order) of this
                // half: low/high nibble of ql OR'd with the matching 2-bit qh
                // plane shifted into bits 4–5. All u8 in [0,63].
                let hi2 = |sh: __m256i| _mm256_slli_epi16::<4>(_mm256_and_si256(sh, three));
                let q1 = _mm256_or_si256(_mm256_and_si256(ql_lo, lomask), hi2(qhv));
                let q2 = _mm256_or_si256(
                    _mm256_and_si256(ql_hi, lomask),
                    hi2(_mm256_srli_epi16::<2>(qhv)),
                );
                let q3 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16::<4>(ql_lo), lomask),
                    hi2(_mm256_srli_epi16::<4>(qhv)),
                );
                let q4 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16::<4>(ql_hi), lomask),
                    hi2(_mm256_srli_epi16::<6>(qhv)),
                );
                for (qi, qv) in [q1, q2, q3, q4].into_iter().enumerate() {
                    let nb = half * 128 + qi * 32; // natural base of this group
                    let xv = _mm256_loadu_si256(qx.add(nb) as *const __m256i);
                    let dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qv, xv);
                    let da = hsum_i32_x4(_mm256_castsi256_si128(dot)); // first 16
                    let db = hsum_i32_x4(_mm256_extracti128_si256::<1>(dot)); // next 16
                    let ga = nb / 16;
                    acc += scales[ga] as i8 as i32 * (da - 32 * bsums[ga] as i32);
                    acc += scales[ga + 1] as i8 as i32 * (db - 32 * bsums[ga + 1] as i32);
                }
            }
            total += d * bigd * acc as f32;
        }
        total
    }

    // --- AVX2 fallback (no AVX-512 VNNI) --------------------------------
    //
    // The AVX2 analog of `vpdpbusd`: `vpmaddubsw` (u8×i8 → i16, summing adjacent
    // pairs) then `vpmaddwd` against ones (i16 pairs → i32). The eight i32 lanes
    // have the same layout as `_mm256_dpbusd_epi32`, so these kernels mirror the
    // VNNI ones with the dot primitive swapped. For 4-/6-bit weights the i16
    // intermediate can't overflow, so the result is bit-identical to scalar; the
    // signed 8-bit formats widen to i16 first to stay exact. The dispatchers
    // prefer VNNI and fall back here, so this is the path most consumer x86
    // (AVX2 without AVX-512) actually runs.

    /// AVX2 available and not disabled (`RUSTY_LLAMA_NO_AVX2`). Cached.
    pub fn avx2_supported() -> bool {
        static OK: OnceLock<bool> = OnceLock::new();
        *OK.get_or_init(|| {
            std::env::var_os("RUSTY_LLAMA_NO_AVX2").is_none() && is_x86_feature_detected!("avx2")
        })
    }

    /// `Σ_{i<32} w[i]·x[i]` as eight i32 lanes (4 products each) — the same lane
    /// layout as `vpdpbusd`. `w` is unsigned `u8`, `x` signed `i8`. Exact while
    /// each `vpmaddubsw` i16 (sum of two products) stays in range — true for the
    /// 4-/6-bit weights this is used with.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn madd_u8_i8(w: __m256i, x: __m256i) -> __m256i {
        _mm256_madd_epi16(_mm256_maddubs_epi16(w, x), _mm256_set1_epi16(1))
    }

    /// Exact `Σ_{i<32} w[i]·x[i]` for two signed-`i8` operands, via i16 widening
    /// (`vpmovsxbw` + `vpmaddwd`) so nothing saturates.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn dot32_signed_avx2(w: __m256i, x: __m256i) -> i32 {
        let wlo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w));
        let whi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(w));
        let xlo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x));
        let xhi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(x));
        let p = _mm256_add_epi32(_mm256_madd_epi16(wlo, xlo), _mm256_madd_epi16(whi, xhi));
        hsum_i32_x8(p)
    }

    /// AVX2 (non-VNNI) implementation of [`super::vec_dot_q8_0`].
    /// # Safety
    /// [`avx2_supported`] must return true on this CPU.
    #[target_feature(enable = "avx2")]
    pub unsafe fn vec_dot_q8_0_avx2(weight: &[u8], act: &Q8Activation) -> f32 {
        let mut acc = 0.0f32;
        for (b, blk) in weight.chunks_exact(34).enumerate() {
            let dw = rd_f16(blk, 0);
            let w = _mm256_loadu_si256(blk.as_ptr().add(2) as *const __m256i);
            let x = _mm256_loadu_si256(act.qs.as_ptr().add(b * 32) as *const __m256i);
            acc += dw * act.scales[b] * dot32_signed_avx2(w, x) as f32;
        }
        acc
    }

    /// AVX2 (non-VNNI) implementation of [`super::vec_dot_q4_0`].
    /// # Safety
    /// [`avx2_supported`] must return true on this CPU.
    #[target_feature(enable = "avx2")]
    pub unsafe fn vec_dot_q4_0_avx2(weight: &[u8], act: &Q8Activation) -> f32 {
        let lomask = _mm256_set1_epi8(0x0f);
        let eight = _mm256_set1_epi8(8);
        let mut acc = 0.0f32;
        for (b, blk) in weight.chunks_exact(18).enumerate() {
            let dw = rd_f16(blk, 0);
            let packed = _mm_loadu_si128(blk.as_ptr().add(2) as *const __m128i);
            let v = _mm256_broadcastsi128_si256(packed);
            let lo = _mm256_and_si256(v, lomask);
            let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(v), lomask);
            let nibs = _mm256_blend_epi32::<0xF0>(lo, hi);
            let w = _mm256_sub_epi8(nibs, eight);
            let x = _mm256_loadu_si256(act.qs.as_ptr().add(b * 32) as *const __m256i);
            acc += dw * act.scales[b] * dot32_signed_avx2(w, x) as f32;
        }
        acc
    }

    /// AVX2 (non-VNNI) implementation of [`super::vec_dot_q4_k`]. Q4_K nibbles
    /// are `[0,15]`, so `vpmaddubsw` can't overflow — bit-identical to scalar.
    /// # Safety
    /// [`avx2_supported`] must return true on this CPU.
    #[target_feature(enable = "avx2")]
    pub unsafe fn vec_dot_q4_k_avx2(weight: &[u8], act: &Q8KActivation) -> f32 {
        let lomask = _mm256_set1_epi8(0x0f);
        let mut total = 0.0f32;
        for (sb, wblk) in weight.chunks_exact(144).enumerate() {
            let d = rd_f16(wblk, 0);
            let dmin = rd_f16(wblk, 2);
            let scales = &wblk[4..16];
            let qs = wblk[16..144].as_ptr();
            let qx = act.qs.as_ptr().add(sb * 256);
            let bigd = act.d[sb];
            let mut acc: i32 = 0;
            for chunk in 0..4 {
                let v = _mm256_loadu_si256(qs.add(chunk * 32) as *const __m256i);
                let lo = _mm256_and_si256(v, lomask);
                let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(v), lomask);
                let xlo = _mm256_loadu_si256(qx.add((2 * chunk) * 32) as *const __m256i);
                let xhi = _mm256_loadu_si256(qx.add((2 * chunk + 1) * 32) as *const __m256i);
                let dlo = hsum_i32_x8(madd_u8_i8(lo, xlo));
                let dhi = hsum_i32_x8(madd_u8_i8(hi, xhi));
                let (sc_lo, _) = get_scale_min_k4(2 * chunk, scales);
                let (sc_hi, _) = get_scale_min_k4(2 * chunk + 1, scales);
                acc += sc_lo as i32 * dlo;
                acc += sc_hi as i32 * dhi;
            }
            let bsums = &act.bsums[sb * 16..sb * 16 + 16];
            let mut min_acc: i32 = 0;
            for (g, &bs) in bsums.iter().enumerate() {
                let (_, m) = get_scale_min_k4(g / 2, scales);
                min_acc += bs as i32 * m as i32;
            }
            total += d * bigd * acc as f32 - dmin * bigd * min_acc as f32;
        }
        total
    }

    /// AVX2 (non-VNNI) implementation of [`super::vec_dot_q6_k`]. Reconstructed
    /// 6-bit weights are `[0,63]`, so `vpmaddubsw` can't overflow — bit-identical
    /// to scalar.
    /// # Safety
    /// [`avx2_supported`] must return true on this CPU.
    #[target_feature(enable = "avx2")]
    pub unsafe fn vec_dot_q6_k_avx2(weight: &[u8], act: &Q8KActivation) -> f32 {
        let lomask = _mm256_set1_epi8(0x0f);
        let three = _mm256_set1_epi8(3);
        let mut total = 0.0f32;
        for (sb, wblk) in weight.chunks_exact(210).enumerate() {
            let ql = wblk.as_ptr();
            let qh = wblk.as_ptr().add(128);
            let scales = &wblk[192..208];
            let d = rd_f16(wblk, 208);
            let qx = act.qs.as_ptr().add(sb * 256);
            let bsums = &act.bsums[sb * 16..sb * 16 + 16];
            let bigd = act.d[sb];
            let mut acc: i32 = 0;
            for half in 0..2usize {
                let ql_lo = _mm256_loadu_si256(ql.add(half * 64) as *const __m256i);
                let ql_hi = _mm256_loadu_si256(ql.add(half * 64 + 32) as *const __m256i);
                let qhv = _mm256_loadu_si256(qh.add(half * 32) as *const __m256i);
                let hi2 = |sh: __m256i| _mm256_slli_epi16::<4>(_mm256_and_si256(sh, three));
                let q1 = _mm256_or_si256(_mm256_and_si256(ql_lo, lomask), hi2(qhv));
                let q2 = _mm256_or_si256(
                    _mm256_and_si256(ql_hi, lomask),
                    hi2(_mm256_srli_epi16::<2>(qhv)),
                );
                let q3 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16::<4>(ql_lo), lomask),
                    hi2(_mm256_srli_epi16::<4>(qhv)),
                );
                let q4 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16::<4>(ql_hi), lomask),
                    hi2(_mm256_srli_epi16::<6>(qhv)),
                );
                for (qi, qv) in [q1, q2, q3, q4].into_iter().enumerate() {
                    let nb = half * 128 + qi * 32;
                    let xv = _mm256_loadu_si256(qx.add(nb) as *const __m256i);
                    let dot = madd_u8_i8(qv, xv);
                    let da = hsum_i32_x4(_mm256_castsi256_si128(dot));
                    let db = hsum_i32_x4(_mm256_extracti128_si256::<1>(dot));
                    let ga = nb / 16;
                    acc += scales[ga] as i8 as i32 * (da - 32 * bsums[ga] as i32);
                    acc += scales[ga + 1] as i8 as i32 * (db - 32 * bsums[ga + 1] as i32);
                }
            }
            total += d * bigd * acc as f32;
        }
        total
    }

    /// Register-tiled batched [`super::vec_dot_q4_k`]: dot `weights.len()` (≤4)
    /// weight rows against every column in `acts`, writing `out` as
    /// `(nr, acts.len())` row-major (`out[r*n + t]`). Each activation sub-block is
    /// loaded ONCE and dotted against all `nr` rows — reusing it `nr×` (the prefill
    /// GEMM's activation-bandwidth win: activations are otherwise re-read once per
    /// output feature) — and each weight super-block is unpacked once per
    /// token-block. Bit-identical to [`super::vec_dot_q4_k`] per (row, column).
    /// # Safety
    /// [`avx2_supported`] true; `weights.len() <= 4`, all rows equal length.
    #[target_feature(enable = "avx2")]
    pub unsafe fn vec_dot_q4_k_tiled_avx2(weights: &[&[u8]], acts: &[Q8KActivation], out: &mut [f32]) {
        let lomask = _mm256_set1_epi8(0x0f);
        let nr = weights.len();
        let n = acts.len();
        let nsb = weights[0].len() / 144;
        let mut bstart = 0;
        while bstart < n {
            let bs = (n - bstart).min(8);
            let mut accf = [[0.0f32; 8]; 4]; // [row][token]
            for sb in 0..nsb {
                // Unpack the nr weight rows' super-block `sb` ONCE.
                let mut nib = [[_mm256_setzero_si256(); 8]; 4];
                let mut sc = [[0i32; 8]; 4];
                let mut mn = [[0i32; 8]; 4];
                let mut dd = [0.0f32; 4];
                let mut dm = [0.0f32; 4];
                for r in 0..nr {
                    let wblk = &weights[r][sb * 144..sb * 144 + 144];
                    dd[r] = rd_f16(wblk, 0);
                    dm[r] = rd_f16(wblk, 2);
                    let scales = &wblk[4..16];
                    let qs = wblk[16..144].as_ptr();
                    for chunk in 0..4 {
                        let v = _mm256_loadu_si256(qs.add(chunk * 32) as *const __m256i);
                        nib[r][2 * chunk] = _mm256_and_si256(v, lomask);
                        nib[r][2 * chunk + 1] = _mm256_and_si256(_mm256_srli_epi16::<4>(v), lomask);
                    }
                    for j in 0..8 {
                        let (s, m) = get_scale_min_k4(j, scales);
                        sc[r][j] = s as i32;
                        mn[r][j] = m as i32;
                    }
                }
                for tt in 0..bs {
                    let act = &acts[bstart + tt];
                    let qx = act.qs.as_ptr().add(sb * 256);
                    let bigd = act.d[sb];
                    let mut vacc0 = [_mm256_setzero_si256(); 4];
                    let mut vacc1 = [_mm256_setzero_si256(); 4];
                    for chunk in 0..4 {
                        // Activation loaded ONCE, reused across all nr weight rows.
                        let xlo = _mm256_loadu_si256(qx.add((2 * chunk) * 32) as *const __m256i);
                        let xhi = _mm256_loadu_si256(qx.add((2 * chunk + 1) * 32) as *const __m256i);
                        for r in 0..nr {
                            vacc0[r] = _mm256_add_epi32(vacc0[r], _mm256_mullo_epi32(madd_u8_i8(nib[r][2 * chunk], xlo), _mm256_set1_epi32(sc[r][2 * chunk])));
                            vacc1[r] = _mm256_add_epi32(vacc1[r], _mm256_mullo_epi32(madd_u8_i8(nib[r][2 * chunk + 1], xhi), _mm256_set1_epi32(sc[r][2 * chunk + 1])));
                        }
                    }
                    let bsums = &act.bsums[sb * 16..sb * 16 + 16];
                    for r in 0..nr {
                        let acc = hsum_i32_x8(_mm256_add_epi32(vacc0[r], vacc1[r]));
                        let mut min_acc = 0i32;
                        for (g, &b) in bsums.iter().enumerate() {
                            min_acc += b as i32 * mn[r][g / 2];
                        }
                        accf[r][tt] += dd[r] * bigd * acc as f32 - dm[r] * bigd * min_acc as f32;
                    }
                }
            }
            for r in 0..nr {
                out[r * n + bstart..r * n + bstart + bs].copy_from_slice(&accf[r][..bs]);
            }
            bstart += bs;
        }
    }

    /// Register-tiled batched [`super::vec_dot_q6_k`]; see
    /// [`vec_dot_q4_k_tiled_avx2`]. The 6-bit reconstruction (the heaviest k-quant
    /// unpack) is done once per token-block; each activation sub-block is loaded
    /// once and dotted against all `nr` rows.
    /// # Safety
    /// [`avx2_supported`] true; `weights.len() <= 4`, all rows equal length.
    #[target_feature(enable = "avx2")]
    #[allow(clippy::needless_range_loop)] // k indexes qv[r][k] and the act offset/scale
    pub unsafe fn vec_dot_q6_k_tiled_avx2(weights: &[&[u8]], acts: &[Q8KActivation], out: &mut [f32]) {
        let lomask = _mm256_set1_epi8(0x0f);
        let three = _mm256_set1_epi8(3);
        let nr = weights.len();
        let n = acts.len();
        let nsb = weights[0].len() / 210;
        let mut bstart = 0;
        while bstart < n {
            let bs = (n - bstart).min(8);
            let mut accf = [[0.0f32; 8]; 4];
            for sb in 0..nsb {
                let mut qv = [[_mm256_setzero_si256(); 8]; 4];
                let mut scb = [[0i32; 16]; 4];
                let mut dd = [0.0f32; 4];
                for r in 0..nr {
                    let wblk = &weights[r][sb * 210..sb * 210 + 210];
                    let ql = wblk.as_ptr();
                    let qh = wblk.as_ptr().add(128);
                    let scales = &wblk[192..208];
                    dd[r] = rd_f16(wblk, 208);
                    for g in 0..16 {
                        scb[r][g] = scales[g] as i8 as i32;
                    }
                    for half in 0..2usize {
                        let ql_lo = _mm256_loadu_si256(ql.add(half * 64) as *const __m256i);
                        let ql_hi = _mm256_loadu_si256(ql.add(half * 64 + 32) as *const __m256i);
                        let qhv = _mm256_loadu_si256(qh.add(half * 32) as *const __m256i);
                        let hi2 = |sh: __m256i| _mm256_slli_epi16::<4>(_mm256_and_si256(sh, three));
                        qv[r][half * 4] = _mm256_or_si256(_mm256_and_si256(ql_lo, lomask), hi2(qhv));
                        qv[r][half * 4 + 1] = _mm256_or_si256(_mm256_and_si256(ql_hi, lomask), hi2(_mm256_srli_epi16::<2>(qhv)));
                        qv[r][half * 4 + 2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(ql_lo), lomask), hi2(_mm256_srli_epi16::<4>(qhv)));
                        qv[r][half * 4 + 3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(ql_hi), lomask), hi2(_mm256_srli_epi16::<6>(qhv)));
                    }
                }
                for tt in 0..bs {
                    let act = &acts[bstart + tt];
                    let qx = act.qs.as_ptr().add(sb * 256);
                    let bsums = &act.bsums[sb * 16..sb * 16 + 16];
                    let bigd = act.d[sb];
                    let mut acc = [0i32; 4];
                    for k in 0..8usize {
                        // Activation loaded ONCE, reused across all nr weight rows.
                        let xv = _mm256_loadu_si256(qx.add(k * 32) as *const __m256i);
                        let ga = k * 2;
                        let cb0 = 32 * bsums[ga] as i32;
                        let cb1 = 32 * bsums[ga + 1] as i32;
                        for r in 0..nr {
                            let dot = madd_u8_i8(qv[r][k], xv);
                            let da = hsum_i32_x4(_mm256_castsi256_si128(dot));
                            let db = hsum_i32_x4(_mm256_extracti128_si256::<1>(dot));
                            acc[r] += scb[r][ga] * (da - cb0) + scb[r][ga + 1] * (db - cb1);
                        }
                    }
                    for r in 0..nr {
                        accf[r][tt] += dd[r] * bigd * acc[r] as f32;
                    }
                }
            }
            for r in 0..nr {
                out[r * n + bstart..r * n + bstart + bs].copy_from_slice(&accf[r][..bs]);
            }
            bstart += bs;
        }
    }
}

/// Register-tiled batched [`vec_dot_q4_k`]: dot the (≤4) rows in `weights` against
/// every column in `acts`, writing `out` as `(weights.len(), acts.len())`
/// row-major. On AVX2 each activation is loaded once and reused across the rows
/// (the prefill GEMM's activation-bandwidth win); falls back to the per-(row,col)
/// dot otherwise. Bit-identical to [`vec_dot_q4_k`] per (row, column).
pub fn vec_dot_q4_k_tiled(weights: &[&[u8]], acts: &[Q8KActivation], out: &mut [f32]) {
    let n = acts.len();
    debug_assert_eq!(out.len(), weights.len() * n);
    #[cfg(target_arch = "x86_64")]
    if x86::avx2_supported() {
        unsafe { x86::vec_dot_q4_k_tiled_avx2(weights, acts, out) };
        return;
    }
    for (r, w) in weights.iter().enumerate() {
        for (t, act) in acts.iter().enumerate() {
            out[r * n + t] = vec_dot_q4_k(w, act);
        }
    }
}

/// Register-tiled batched [`vec_dot_q6_k`]; see [`vec_dot_q4_k_tiled`].
pub fn vec_dot_q6_k_tiled(weights: &[&[u8]], acts: &[Q8KActivation], out: &mut [f32]) {
    let n = acts.len();
    debug_assert_eq!(out.len(), weights.len() * n);
    #[cfg(target_arch = "x86_64")]
    if x86::avx2_supported() {
        unsafe { x86::vec_dot_q6_k_tiled_avx2(weights, acts, out) };
        return;
    }
    for (r, w) in weights.iter().enumerate() {
        for (t, act) in acts.iter().enumerate() {
            out[r * n + t] = vec_dot_q6_k(w, act);
        }
    }
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

    /// R5.2: the AVX2 f32 attention kernels must match the scalar reference at every
    /// 8-wide tile boundary (the remainder loop is the real correctness risk).
    #[test]
    fn f32_simd_kernels_match_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !x86::avx2f_supported() {
                eprintln!("no AVX2+FMA — skipping SIMD parity");
                return;
            }
            for &n in &[1usize, 7, 8, 9, 15, 16, 31, 63, 64, 65, 80, 127, 128] {
                let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7 - 3.0).sin()).collect();
                let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3 + 1.0).cos()).collect();
                // dot
                let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
                let got = unsafe { x86::f32_dot_avx2(&a, &b) };
                assert!(
                    (got - want).abs() <= 1e-3 * (1.0 + want.abs()),
                    "dot n={n}: {got} vs {want}"
                );
                // out = out*alpha + p*v
                let (alpha, p) = (0.9f32, 0.25f32);
                let mut want_v = a.clone();
                for (o, &vi) in want_v.iter_mut().zip(&b) {
                    *o = *o * alpha + p * vi;
                }
                let mut got_v = a.clone();
                unsafe { x86::f32_axpy_decay_avx2(&mut got_v, alpha, p, &b) };
                for i in 0..n {
                    assert!(
                        (got_v[i] - want_v[i]).abs() <= 1e-4 * (1.0 + want_v[i].abs()),
                        "axpy n={n} i={i}: {} vs {}",
                        got_v[i],
                        want_v[i]
                    );
                }
            }
        }
    }

    /// R5.2: measure the long-context attention inner-loop speedup (AVX2 vs scalar).
    #[test]
    #[ignore = "bench; run with --lib -- --ignored --nocapture"]
    fn bench_f32_attention_kernels() {
        #[cfg(target_arch = "x86_64")]
        {
            use std::time::Instant;
            let (hs, keys, heads, iters) = (128usize, 4096usize, 32usize, 8usize);
            let q: Vec<f32> = (0..hs).map(|i| (i as f32 * 0.01).sin()).collect();
            let kc: Vec<f32> = (0..keys * hs).map(|i| (i as f32 * 0.001).cos()).collect();
            let vc: Vec<f32> = (0..keys * hs).map(|i| (i as f32 * 0.002).sin()).collect();
            let p = 0.013f32;

            let run = |simd: bool| {
                let t0 = Instant::now();
                let mut sink = 0.0f32;
                for _ in 0..iters {
                    for _h in 0..heads {
                        let mut acc = vec![0.0f32; hs];
                        for t in 0..keys {
                            let k = &kc[t * hs..t * hs + hs];
                            let v = &vc[t * hs..t * hs + hs];
                            if simd {
                                let s = unsafe { x86::f32_dot_avx2(&q, k) };
                                unsafe { x86::f32_axpy_decay_avx2(&mut acc, 0.99, p * s, v) };
                            } else {
                                let s: f32 = q.iter().zip(k).map(|(a, b)| a * b).sum();
                                for (o, &vi) in acc.iter_mut().zip(v) {
                                    *o = *o * 0.99 + p * s * vi;
                                }
                            }
                        }
                        sink += acc[0];
                    }
                }
                (t0.elapsed(), sink)
            };
            let (ts, _) = run(false);
            let (tv, _) = if x86::avx2f_supported() { run(true) } else { (ts, 0.0) };
            eprintln!(
                "attn inner-loop (hs={hs}, keys={keys}, heads={heads}): scalar {ts:?}  avx2 {tv:?}  speedup {:.2}x",
                ts.as_secs_f64() / tv.as_secs_f64()
            );
        }
    }

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
    fn q4_1_hand_computed_block() {
        // d=0.5, m=-1.0; nibbles 0..16 packed as (hi<<4)|lo so low-nibble j
        // holds value j%16 and high-nibble j holds value 15-(j%16).
        let d = 0.5f32;
        let m = -1.0f32;
        let mut blk = Vec::new();
        blk.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        blk.extend_from_slice(&f32_to_f16(m).to_le_bytes());
        let qs: Vec<u8> = (0..16u8).map(|j| j | ((15 - j) << 4)).collect();
        blk.extend_from_slice(&qs);
        assert_eq!(blk.len(), GgmlType::Q4_1.type_size());

        let y = dequantize(GgmlType::Q4_1, &blk, 32).unwrap();
        for j in 0..16 {
            let want_lo = j as f32 * d + m;
            let want_hi = (15 - j) as f32 * d + m;
            assert!((y[j] - want_lo).abs() < 1e-4, "lo[{j}]: {} vs {want_lo}", y[j]);
            assert!(
                (y[j + 16] - want_hi).abs() < 1e-4,
                "hi[{j}]: {} vs {want_hi}",
                y[j + 16]
            );
        }
    }

    #[test]
    fn q4_1_boundary_nibbles() {
        // All-zero nibbles -> every output equals m; all-max (15) -> d*15+m.
        let d = 2.0f32;
        let m = 3.0f32;
        let mk_block = |nibble: u8| {
            let mut blk = Vec::new();
            blk.extend_from_slice(&f32_to_f16(d).to_le_bytes());
            blk.extend_from_slice(&f32_to_f16(m).to_le_bytes());
            blk.extend(std::iter::repeat_n(nibble | (nibble << 4), 16));
            blk
        };
        let zeros = dequantize(GgmlType::Q4_1, &mk_block(0), 32).unwrap();
        assert!(zeros.iter().all(|&v| (v - m).abs() < 1e-4), "{zeros:?}");
        let maxed = dequantize(GgmlType::Q4_1, &mk_block(15), 32).unwrap();
        let want = 15.0 * d + m;
        assert!(maxed.iter().all(|&v| (v - want).abs() < 1e-3), "{maxed:?}");
    }

    #[test]
    fn q5_0_hand_computed_block_no_high_bits() {
        // qh = 0: reduces to a 4-bit nibble with a -16 zero point (no 5th bit set).
        let d = 0.5f32;
        let mut blk = Vec::new();
        blk.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        blk.extend_from_slice(&0u32.to_le_bytes());
        let qs: Vec<u8> = (0..16u8).map(|j| j | ((15 - j) << 4)).collect();
        blk.extend_from_slice(&qs);
        assert_eq!(blk.len(), GgmlType::Q5_0.type_size());

        let y = dequantize(GgmlType::Q5_0, &blk, 32).unwrap();
        for j in 0..16 {
            let want_lo = (j as i32 - 16) as f32 * d;
            let want_hi = ((15 - j) as i32 - 16) as f32 * d;
            assert!((y[j] - want_lo).abs() < 1e-4, "lo[{j}]: {} vs {want_lo}", y[j]);
            assert!(
                (y[j + 16] - want_hi).abs() < 1e-4,
                "hi[{j}]: {} vs {want_hi}",
                y[j + 16]
            );
        }
    }

    #[test]
    fn q5_0_high_bit_plane() {
        // The 5th bit (packed in `qh`) adds 16 to the raw nibble before the -16
        // zero point: a zero nibble + high bit -> raw 16 -> dequantized 0; a
        // max nibble (15) + high bit -> raw 31 -> dequantized 15.
        let d = 1.0f32;
        // bit 0 = low-nibble (j=0) high bit; bit 16 = high-nibble (j=0) high bit.
        let qh: u32 = (1 << 0) | (1 << 16);

        let mut zero_blk = Vec::new();
        zero_blk.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        zero_blk.extend_from_slice(&qh.to_le_bytes());
        zero_blk.extend_from_slice(&[0u8; 16]); // both nibbles of qs[0] are 0
        assert_eq!(zero_blk.len(), GgmlType::Q5_0.type_size());
        let y0 = dequantize(GgmlType::Q5_0, &zero_blk, 32).unwrap();
        assert!((y0[0] - 0.0).abs() < 1e-4, "{}", y0[0]);
        assert!((y0[16] - 0.0).abs() < 1e-4, "{}", y0[16]);

        let mut max_blk = Vec::new();
        max_blk.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        max_blk.extend_from_slice(&qh.to_le_bytes());
        let mut qs = [0u8; 16];
        qs[0] = 0x0f | (0x0f << 4); // both nibbles of qs[0] are 15
        max_blk.extend_from_slice(&qs);
        let y1 = dequantize(GgmlType::Q5_0, &max_blk, 32).unwrap();
        assert!((y1[0] - 15.0).abs() < 1e-4, "{}", y1[0]);
        assert!((y1[16] - 15.0).abs() < 1e-4, "{}", y1[16]);
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
        blk.extend(std::iter::repeat_n(0x11u8, 128)); // ql -> nibble 1
        blk.extend(std::iter::repeat_n(0u8, 64)); // qh -> 0
        blk.extend(std::iter::repeat_n((-2i8) as u8, 16)); // signed scale -2
        blk.extend_from_slice(&f32_to_f16(1.0).to_le_bytes());
        assert_eq!(blk.len(), GgmlType::Q6_K.type_size());

        // q = 1 - 32 = -31, scale -2 -> 62 (and signed scales must be honoured).
        let y = dequantize(GgmlType::Q6_K, &blk, QK_K).unwrap();
        assert!(y.iter().all(|&v| (v - 62.0).abs() < 1e-4), "{:?}", &y[..4]);
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

    #[test]
    fn vec_dot_q4_k_equals_dequantized_dot() {
        let mut blk = Vec::new();
        blk.extend_from_slice(&f32_to_f16(0.04).to_le_bytes()); // d
        blk.extend_from_slice(&f32_to_f16(0.02).to_le_bytes()); // dmin
        blk.extend_from_slice(&pack_scales_q4_k(
            [10, 20, 5, 33, 41, 7, 18, 25],
            [3, 9, 14, 1, 22, 6, 30, 11],
        ));
        for i in 0..128u32 {
            blk.push(((i * 7 + 3) % 256) as u8); // 256 nibbles
        }
        assert_eq!(blk.len(), GgmlType::Q4_K.type_size());

        let x: Vec<f32> = (0..256).map(|i| ((i * 11 % 23) as f32 - 11.0) * 0.05).collect();
        let act = quantize_activation_q8k(&x);
        let wdq = dequantize(GgmlType::Q4_K, &blk, 256).unwrap();
        let reference: f32 = wdq.iter().zip(&act.dequantized()).map(|(a, b)| a * b).sum();
        let got = vec_dot_q4_k(&blk, &act);
        assert!((got - reference).abs() < 1e-2, "{got} vs {reference}");
    }

    #[test]
    fn vec_dot_q6_k_equals_dequantized_dot() {
        let mut blk = Vec::new();
        for i in 0..128u32 {
            blk.push(((i * 5 + 1) % 256) as u8); // ql
        }
        for i in 0..64u32 {
            blk.push(((i * 9 + 2) % 256) as u8); // qh
        }
        for i in 0..16i32 {
            blk.push((i - 8) as i8 as u8); // signed scales -8..7
        }
        blk.extend_from_slice(&f32_to_f16(0.03).to_le_bytes()); // d
        assert_eq!(blk.len(), GgmlType::Q6_K.type_size());

        let x: Vec<f32> = (0..256).map(|i| ((i * 13 % 19) as f32 - 9.0) * 0.04).collect();
        let act = quantize_activation_q8k(&x);
        let wdq = dequantize(GgmlType::Q6_K, &blk, 256).unwrap();
        let reference: f32 = wdq.iter().zip(&act.dequantized()).map(|(a, b)| a * b).sum();
        let got = vec_dot_q6_k(&blk, &act);
        assert!((got - reference).abs() < 1e-2, "{got} vs {reference}");
    }

    // --- AVX-512 VNNI: bit-identical parity + timing --------------------------

    /// `n` pseudo-random bytes from a seed (LCG) for fuzzing the SIMD kernels
    /// against their scalar oracles.
    #[cfg(target_arch = "x86_64")]
    fn prng_bytes(seed: u32, n: usize) -> Vec<u8> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 16) as u8
            })
            .collect()
    }

    // AVX2 (non-VNNI) bit-identity: these RUN on any AVX2 CPU (incl. this one,
    // which has no AVX-512), so the AVX2 kernels are run-verified, not just
    // built. 4-/6-bit weights => no vpmaddubsw saturation => exact.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn q8_0_avx2_matches_scalar() {
        if !x86::avx2_supported() {
            return;
        }
        let cols = 256;
        for seed in 0..16u32 {
            let xf: Vec<f32> = (0..cols)
                .map(|i| ((i as u32 ^ seed).wrapping_mul(2654435761) % 509) as f32 * 0.011 - 2.8)
                .collect();
            let act = quantize_activation_q8(&xf);
            let mut w = Vec::new();
            for (b, blk) in prng_bytes(seed + 1, cols).chunks_exact(32).enumerate() {
                w.extend_from_slice(&f32_to_f16(0.02 + b as f32 * 0.003).to_le_bytes());
                w.extend_from_slice(blk);
            }
            let s = vec_dot_q8_0_scalar(&w, &act);
            let v = unsafe { x86::vec_dot_q8_0_avx2(&w, &act) };
            assert_eq!(s.to_bits(), v.to_bits(), "seed {seed}: scalar {s} != avx2 {v}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn q4_0_avx2_matches_scalar() {
        if !x86::avx2_supported() {
            return;
        }
        let cols = 256;
        for seed in 0..16u32 {
            let xf: Vec<f32> = (0..cols)
                .map(|i| ((i as u32 ^ seed).wrapping_mul(40503) % 251) as f32 * 0.02 - 2.5)
                .collect();
            let act = quantize_activation_q8(&xf);
            let mut w = Vec::new();
            for (b, blk) in prng_bytes(seed + 100, (cols / 32) * 16).chunks_exact(16).enumerate() {
                w.extend_from_slice(&f32_to_f16(0.03 + b as f32 * 0.002).to_le_bytes());
                w.extend_from_slice(blk);
            }
            let s = vec_dot_q4_0_scalar(&w, &act);
            let v = unsafe { x86::vec_dot_q4_0_avx2(&w, &act) };
            assert_eq!(s.to_bits(), v.to_bits(), "seed {seed}: scalar {s} != avx2 {v}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn q4_k_avx2_matches_scalar() {
        if !x86::avx2_supported() {
            return;
        }
        let cols = 512;
        for seed in 0..16u32 {
            let xf: Vec<f32> = (0..cols)
                .map(|i| ((i as u32 ^ seed).wrapping_mul(2246822519) % 263) as f32 * 0.013 - 1.7)
                .collect();
            let act = quantize_activation_q8k(&xf);
            let mut w = Vec::new();
            for sb in 0..(cols / 256) {
                w.extend_from_slice(&f32_to_f16(0.05 + sb as f32 * 0.01).to_le_bytes());
                w.extend_from_slice(&f32_to_f16(0.02 + sb as f32 * 0.005).to_le_bytes());
                w.extend_from_slice(&prng_bytes(seed * 31 + sb as u32 + 7, 140));
            }
            let s = vec_dot_q4_k_scalar(&w, &act);
            let v = unsafe { x86::vec_dot_q4_k_avx2(&w, &act) };
            assert_eq!(s.to_bits(), v.to_bits(), "seed {seed}: scalar {s} != avx2 {v}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn q6_k_avx2_matches_scalar() {
        if !x86::avx2_supported() {
            return;
        }
        let cols = 512;
        for seed in 0..16u32 {
            let xf: Vec<f32> = (0..cols)
                .map(|i| ((i as u32 ^ seed).wrapping_mul(2246822519) % 263) as f32 * 0.013 - 1.7)
                .collect();
            let act = quantize_activation_q8k(&xf);
            let mut w = Vec::new();
            for sb in 0..(cols / 256) {
                w.extend_from_slice(&prng_bytes(seed * 41 + sb as u32 + 3, 208));
                w.extend_from_slice(&f32_to_f16(0.03 + sb as f32 * 0.01).to_le_bytes());
            }
            let s = vec_dot_q6_k_scalar(&w, &act);
            let v = unsafe { x86::vec_dot_q6_k_avx2(&w, &act) };
            assert_eq!(s.to_bits(), v.to_bits(), "seed {seed}: scalar {s} != avx2 {v}");
        }
    }

    #[test]
    fn vec_dot_q4_k_tiled_matches_per_column() {
        let cols = 512;
        let n = 13; // crosses the TB=8 token-block boundary (8 + remainder 5)
        let nr = 3; // exercises the multi-row tile + an odd row count
        let rows: Vec<Vec<u8>> = (0..nr)
            .map(|r| {
                let mut w = Vec::new();
                for sb in 0..(cols / 256) {
                    w.extend_from_slice(&f32_to_f16(0.05 + sb as f32 * 0.01 + r as f32 * 0.002).to_le_bytes());
                    w.extend_from_slice(&f32_to_f16(0.02 + sb as f32 * 0.005).to_le_bytes());
                    w.extend_from_slice(&prng_bytes(31 + sb as u32 + r as u32 * 17, 140));
                }
                w
            })
            .collect();
        let acts: Vec<Q8KActivation> = (0..n)
            .map(|t| {
                let xf: Vec<f32> = (0..cols)
                    .map(|i| {
                        ((i as u32 ^ (t as u32 * 7 + 1)).wrapping_mul(2246822519) % 263) as f32 * 0.013
                            - 1.7
                    })
                    .collect();
                quantize_activation_q8k(&xf)
            })
            .collect();
        let wrefs: Vec<&[u8]> = rows.iter().map(|w| w.as_slice()).collect();
        let mut got = vec![0.0f32; nr * n];
        vec_dot_q4_k_tiled(&wrefs, &acts, &mut got);
        for r in 0..nr {
            for (t, act) in acts.iter().enumerate() {
                let want = vec_dot_q4_k(&rows[r], act);
                assert_eq!(got[r * n + t].to_bits(), want.to_bits(), "Q4_K tiled ({r},{t}): {} != {}", got[r * n + t], want);
            }
        }
    }

    #[test]
    fn vec_dot_q6_k_tiled_matches_per_column() {
        let cols = 512;
        let n = 13;
        let nr = 3;
        let rows: Vec<Vec<u8>> = (0..nr)
            .map(|r| {
                let mut w = Vec::new();
                for sb in 0..(cols / 256) {
                    w.extend_from_slice(&prng_bytes(41 + sb as u32 + r as u32 * 23, 208));
                    w.extend_from_slice(&f32_to_f16(0.03 + sb as f32 * 0.01).to_le_bytes());
                }
                w
            })
            .collect();
        let acts: Vec<Q8KActivation> = (0..n)
            .map(|t| {
                let xf: Vec<f32> = (0..cols)
                    .map(|i| {
                        ((i as u32 ^ (t as u32 * 5 + 3)).wrapping_mul(2654435761) % 251) as f32 * 0.011
                            - 1.4
                    })
                    .collect();
                quantize_activation_q8k(&xf)
            })
            .collect();
        let wrefs: Vec<&[u8]> = rows.iter().map(|w| w.as_slice()).collect();
        let mut got = vec![0.0f32; nr * n];
        vec_dot_q6_k_tiled(&wrefs, &acts, &mut got);
        for r in 0..nr {
            for (t, act) in acts.iter().enumerate() {
                let want = vec_dot_q6_k(&rows[r], act);
                assert_eq!(got[r * n + t].to_bits(), want.to_bits(), "Q6_K tiled ({r},{t}): {} != {}", got[r * n + t], want);
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn q8_0_simd_matches_scalar() {
        if !x86::vnni_supported() {
            return; // No VNNI here; the dispatcher already equals the scalar path.
        }
        let cols = 256; // 8 blocks
        for seed in 0..16u32 {
            let xf: Vec<f32> = (0..cols)
                .map(|i| ((i as u32 ^ seed).wrapping_mul(2654435761) % 509) as f32 * 0.011 - 2.8)
                .collect();
            let act = quantize_activation_q8(&xf);
            // Random valid Q8_0 row: f16 scale + 32 i8 (full byte range) per block.
            let mut w = Vec::new();
            for (b, blk) in prng_bytes(seed + 1, cols).chunks_exact(32).enumerate() {
                w.extend_from_slice(&f32_to_f16(0.02 + b as f32 * 0.003).to_le_bytes());
                w.extend_from_slice(blk);
            }
            let s = vec_dot_q8_0_scalar(&w, &act);
            let v = unsafe { x86::vec_dot_q8_0(&w, &act) };
            assert_eq!(s.to_bits(), v.to_bits(), "seed {seed}: scalar {s} != simd {v}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn q4_0_simd_matches_scalar() {
        if !x86::vnni_supported() {
            return;
        }
        let cols = 256; // 8 blocks of 18 bytes
        for seed in 0..16u32 {
            let xf: Vec<f32> = (0..cols)
                .map(|i| ((i as u32 ^ seed).wrapping_mul(40503) % 251) as f32 * 0.02 - 2.5)
                .collect();
            let act = quantize_activation_q8(&xf);
            let mut w = Vec::new();
            for (b, blk) in prng_bytes(seed + 100, (cols / 32) * 16)
                .chunks_exact(16)
                .enumerate()
            {
                w.extend_from_slice(&f32_to_f16(0.03 + b as f32 * 0.002).to_le_bytes());
                w.extend_from_slice(blk); // 16 packed nibble bytes
            }
            let s = vec_dot_q4_0_scalar(&w, &act);
            let v = unsafe { x86::vec_dot_q4_0(&w, &act) };
            assert_eq!(s.to_bits(), v.to_bits(), "seed {seed}: scalar {s} != simd {v}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn q4_k_simd_matches_scalar() {
        if !x86::vnni_supported() {
            return;
        }
        let cols = 512; // 2 super-blocks
        for seed in 0..16u32 {
            let xf: Vec<f32> = (0..cols)
                .map(|i| ((i as u32 ^ seed).wrapping_mul(2246822519) % 263) as f32 * 0.013 - 1.7)
                .collect();
            let act = quantize_activation_q8k(&xf);
            // Valid Q4_K super-blocks: f16 d, f16 dmin, then 12 packed-scale +
            // 128 nibble bytes that can be *anything* (both paths read them
            // through `get_scale_min_k4` identically, so random maximises cover).
            let mut w = Vec::new();
            for sb in 0..(cols / 256) {
                w.extend_from_slice(&f32_to_f16(0.05 + sb as f32 * 0.01).to_le_bytes());
                w.extend_from_slice(&f32_to_f16(0.02 + sb as f32 * 0.005).to_le_bytes());
                w.extend_from_slice(&prng_bytes(seed * 31 + sb as u32 + 7, 140));
            }
            let s = vec_dot_q4_k_scalar(&w, &act);
            let v = unsafe { x86::vec_dot_q4_k(&w, &act) };
            assert_eq!(s.to_bits(), v.to_bits(), "seed {seed}: scalar {s} != simd {v}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn q6_k_simd_matches_scalar() {
        if !x86::vnni_supported() {
            return;
        }
        let cols = 512; // 2 super-blocks
        for seed in 0..16u32 {
            let xf: Vec<f32> = (0..cols)
                .map(|i| ((i as u32 ^ seed).wrapping_mul(2246822519) % 263) as f32 * 0.013 - 1.7)
                .collect();
            let act = quantize_activation_q8k(&xf);
            // Valid Q6_K super-blocks: 128 ql + 64 qh + 16 i8 scales (any bytes —
            // both paths reconstruct/read them identically) then f16 d.
            let mut w = Vec::new();
            for sb in 0..(cols / 256) {
                w.extend_from_slice(&prng_bytes(seed * 41 + sb as u32 + 3, 208));
                w.extend_from_slice(&f32_to_f16(0.03 + sb as f32 * 0.01).to_le_bytes());
            }
            let s = vec_dot_q6_k_scalar(&w, &act);
            let v = unsafe { x86::vec_dot_q6_k(&w, &act) };
            assert_eq!(s.to_bits(), v.to_bits(), "seed {seed}: scalar {s} != simd {v}");
        }
    }

    /// Scalar re-implementation of the *algorithm* the Q6_K VNNI kernel uses
    /// (unsigned 6-bit reconstruction in natural order; each 32-element group
    /// split into two 16-element scale sub-blocks; the `-32` zero point folded
    /// in via the per-16 `bsums`). Lets us verify that index/bias logic against
    /// the scalar oracle on a CPU without AVX-512 VNNI (where the real kernel
    /// can't run).
    #[allow(clippy::identity_op)]
    fn vec_dot_q6_k_vnni_emulated(weight: &[u8], act: &Q8KActivation) -> f32 {
        let mut total = 0.0f32;
        for (sb, wblk) in weight.chunks_exact(210).enumerate() {
            let scales = &wblk[192..208];
            let d = rd_f16(wblk, 208);
            let qx = &act.qs[sb * 256..sb * 256 + 256];
            let bsums = &act.bsums[sb * 16..sb * 16 + 16];
            let bigd = act.d[sb];
            // Unsigned [0,63] weights in natural order.
            let mut q6 = [0u32; 256];
            for half in 0..2 {
                let ql = &wblk[half * 64..];
                let qh = &wblk[128 + half * 32..];
                let y = half * 128;
                for l in 0..32 {
                    q6[y + l] = ((ql[l] & 0x0f) | (((qh[l] >> 0) & 3) << 4)) as u32;
                    q6[y + l + 32] = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) as u32;
                    q6[y + l + 64] = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as u32;
                    q6[y + l + 96] = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as u32;
                }
            }
            let mut acc: i32 = 0;
            for half in 0..2 {
                for qi in 0..4 {
                    let nb = half * 128 + qi * 32;
                    for sh in 0..2 {
                        let g = nb / 16 + sh;
                        let dot: i32 = (0..16)
                            .map(|k| q6[nb + sh * 16 + k] as i32 * qx[nb + sh * 16 + k] as i32)
                            .sum();
                        acc += scales[g] as i8 as i32 * (dot - 32 * bsums[g] as i32);
                    }
                }
            }
            total += d * bigd * acc as f32;
        }
        total
    }

    /// The Q6_K VNNI kernel can't run on this (AVX2-only) CPU, so check the
    /// algorithm it encodes against the scalar oracle. The real intrinsics are a
    /// mechanical translation of this, mirroring the run-verified Q4_K kernel,
    /// and `q6_k_simd_matches_scalar` checks them where VNNI is available.
    #[test]
    fn q6_k_vnni_algorithm_matches_scalar() {
        let cols = 512;
        for seed in 0..16u32 {
            let xf: Vec<f32> = (0..cols)
                .map(|i| ((i as u32 ^ seed).wrapping_mul(2246822519) % 263) as f32 * 0.013 - 1.7)
                .collect();
            let act = quantize_activation_q8k(&xf);
            let mut w = Vec::new();
            for sb in 0..(cols / 256) {
                w.extend_from_slice(&prng_bytes(seed * 41 + sb as u32 + 3, 208));
                w.extend_from_slice(&f32_to_f16(0.03 + sb as f32 * 0.01).to_le_bytes());
            }
            let s = vec_dot_q6_k_scalar(&w, &act);
            let e = vec_dot_q6_k_vnni_emulated(&w, &act);
            assert_eq!(s.to_bits(), e.to_bits(), "seed {seed}: scalar {s} != emulated {e}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    #[ignore = "timing benchmark; run with --release -- --ignored --nocapture"]
    fn bench_q8_0_simd_vs_scalar() {
        use std::time::Instant;
        if !x86::vnni_supported() {
            eprintln!("AVX-512 VNNI not available; skipping");
            return;
        }
        let (rows, cols, iters) = (2048usize, 2048usize, 10);
        let wf: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 257) as f32 - 128.0) * 0.01)
            .collect();
        let wbytes = quantize_q8_0(&wf);
        let rb = GgmlType::Q8_0.bytes_for(cols);
        let x: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let act = quantize_activation_q8(&x);
        let mut sink = 0.0f32;

        let t0 = Instant::now();
        for _ in 0..iters {
            for i in 0..rows {
                sink += vec_dot_q8_0_scalar(&wbytes[i * rb..i * rb + rb], &act);
            }
        }
        let scalar = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..iters {
            for i in 0..rows {
                sink += unsafe { x86::vec_dot_q8_0(&wbytes[i * rb..i * rb + rb], &act) };
            }
        }
        let simd = t1.elapsed();

        eprintln!(
            "Q8_0 {rows}x{cols} x{iters}: scalar={scalar:?} vnni={simd:?} \
             speedup={:.2}x (sink={sink})",
            scalar.as_secs_f64() / simd.as_secs_f64()
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    #[ignore = "timing benchmark; run with --release -- --ignored --nocapture"]
    fn bench_q4_0_simd_vs_scalar() {
        use std::time::Instant;
        if !x86::vnni_supported() {
            eprintln!("AVX-512 VNNI not available; skipping");
            return;
        }
        let (rows, cols, iters) = (2048usize, 2048usize, 10);
        let rb = GgmlType::Q4_0.bytes_for(cols);
        let wbytes = prng_bytes(1, rows * rb); // validity irrelevant for timing
        let x: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let act = quantize_activation_q8(&x);
        let mut sink = 0.0f32;

        let t0 = Instant::now();
        for _ in 0..iters {
            for i in 0..rows {
                sink += vec_dot_q4_0_scalar(&wbytes[i * rb..i * rb + rb], &act);
            }
        }
        let scalar = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..iters {
            for i in 0..rows {
                sink += unsafe { x86::vec_dot_q4_0(&wbytes[i * rb..i * rb + rb], &act) };
            }
        }
        let simd = t1.elapsed();

        eprintln!(
            "Q4_0 {rows}x{cols} x{iters}: scalar={scalar:?} vnni={simd:?} \
             speedup={:.2}x (sink={sink})",
            scalar.as_secs_f64() / simd.as_secs_f64()
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    #[ignore = "timing benchmark; run with --release -- --ignored --nocapture"]
    fn bench_q4_k_simd_vs_scalar() {
        use std::time::Instant;
        if !x86::vnni_supported() {
            eprintln!("AVX-512 VNNI not available; skipping");
            return;
        }
        let (rows, cols, iters) = (2048usize, 2048usize, 10);
        let rb = GgmlType::Q4_K.bytes_for(cols);
        let wbytes = prng_bytes(2, rows * rb); // validity irrelevant for timing
        let x: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let act = quantize_activation_q8k(&x);
        let mut sink = 0.0f32;

        let t0 = Instant::now();
        for _ in 0..iters {
            for i in 0..rows {
                sink += vec_dot_q4_k_scalar(&wbytes[i * rb..i * rb + rb], &act);
            }
        }
        let scalar = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..iters {
            for i in 0..rows {
                sink += unsafe { x86::vec_dot_q4_k(&wbytes[i * rb..i * rb + rb], &act) };
            }
        }
        let simd = t1.elapsed();

        eprintln!(
            "Q4_K {rows}x{cols} x{iters}: scalar={scalar:?} vnni={simd:?} \
             speedup={:.2}x (sink={sink})",
            scalar.as_secs_f64() / simd.as_secs_f64()
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    #[ignore = "timing benchmark; run with --release -- --ignored --nocapture"]
    fn bench_q6_k_simd_vs_scalar() {
        use std::time::Instant;
        if !x86::vnni_supported() {
            eprintln!("AVX-512 VNNI not available; skipping");
            return;
        }
        let (rows, cols, iters) = (2048usize, 2048usize, 10);
        let rb = GgmlType::Q6_K.bytes_for(cols);
        let wbytes = prng_bytes(5, rows * rb); // validity irrelevant for timing
        let x: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let act = quantize_activation_q8k(&x);
        let mut sink = 0.0f32;

        let t0 = Instant::now();
        for _ in 0..iters {
            for i in 0..rows {
                sink += vec_dot_q6_k_scalar(&wbytes[i * rb..i * rb + rb], &act);
            }
        }
        let scalar = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..iters {
            for i in 0..rows {
                sink += unsafe { x86::vec_dot_q6_k(&wbytes[i * rb..i * rb + rb], &act) };
            }
        }
        let simd = t1.elapsed();

        eprintln!(
            "Q6_K {rows}x{cols} x{iters}: scalar={scalar:?} vnni={simd:?} \
             speedup={:.2}x (sink={sink})",
            scalar.as_secs_f64() / simd.as_secs_f64()
        );
    }
}
