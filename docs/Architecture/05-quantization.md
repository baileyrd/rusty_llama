# 05. Quantization

## Summary

`rusty_llama` reads ggml's blocked quantization formats and runs matmuls directly on the
compressed bytes. The module `src/quant.rs` (≈1850 lines) owns the whole stack: the
`GgmlType` enum and its block geometry, per-block dequantizers, two activation quantizers
(`Q8Activation` block-32 and `Q8KActivation` block-256), and four integer dot kernels
(`vec_dot_q8_0/q4_0/q4_k/q6_k`) with a scalar reference, an AVX2 fast path, and an
AVX-512 VNNI fast path — all three bit-identical. Three things matter most:
1. **Only six types** are supported: F32, F16, Q4_0, Q8_0, Q4_K, Q6_K (`src/quant.rs:32`).
2. **Weights stay compressed in RAM**; the matmul activation-quantizes `x` once, then does
   each row's inner product in integer arithmetic, scaling to f32 only at block boundaries.
3. The SIMD paths are a **pure speed swap** — `vpdpbusd` / `vpmaddubsw` produce results
   bit-for-bit equal to the scalar oracle, asserted by `*_matches_scalar` tests.

`QMatrix` (`src/tensor.rs`) is the unit the matmul consumes: either a zero-copy f32 view or
a quantized block view borrowing bytes straight from the mmap'd GGUF.

llama.cpp counterpart: `docs/Research/04-quantization.md`.

─────────────────────────────────────────────────────────────────────

## `GgmlType` — the supported type set

```rust
// src/quant.rs:32
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q8_0,
    Q4_K,
    Q6_K,
}
```

`#[allow(non_camel_case_types)]` keeps the variant names one-to-one with ggml's own
(`src/quant.rs:31`). The numeric `ggml_type` discriminants are mapped by `from_u32`
(`src/quant.rs:43`) and inverted by `to_u32` (`src/quant.rs:60`); any other id is a hard
`Error::Format("unsupported ggml tensor type {other}")` (`src/quant.rs:51`).

| variant | `to_u32` id | `block_size()` | `type_size()` (bytes) | bits/weight |
|---------|------------:|---------------:|----------------------:|------------:|
| F32     | 0           | 1              | 4                     | 32          |
| F16     | 1           | 1              | 2                     | 16          |
| Q4_0    | 2           | 32             | 18                    | 4.5         |
| Q8_0    | 8           | 32             | 34                    | 8.5         |
| Q4_K    | 12          | 256            | 144                   | 4.5         |
| Q6_K    | 14          | 256            | 210                   | 6.5625      |

`block_size()` (`src/quant.rs:72`) is the elements-per-block; `type_size()`
(`src/quant.rs:81`) is the bytes-per-block. `bytes_for(n)` is just
`(n / block_size()) * type_size()` (`src/quant.rs:93`) — note integer division, so `n` must
be a multiple of the block size (callers validate this). bits/weight = `type_size()*8 /
block_size()` and matches the module's own doc-comment table at `src/quant.rs:9-16`.

Two constants frame the k-quants: `QK_K = 256` (the k-quant super-block,
`src/quant.rs:21`) and `MAX_BLOCK = QK_K` (largest block, used to size stack scratch,
`src/quant.rs:24`).

The id gaps are deliberate: ggml's Q4_1/Q5_0/Q5_1 (ids 3/6/7) and the other k-quants
(Q2_K/Q3_K/Q5_K, ids 10/11/13) are **not** handled — see Status.

─────────────────────────────────────────────────────────────────────

## Block layouts (byte-exact)

All multi-byte scalars are little-endian; the per-block `d`/`dmin` are IEEE-754 binary16
read via `rd_f16` (`src/quant.rs:202`), itself wrapping the hand-rolled `f16_to_f32`
(`src/quant.rs:144`). The block layouts are verified against the dequantizers.

### Q4_0 — 32 values, 18 bytes (`block_q4_0`, `src/quant.rs:213`)

| offset | bytes | field | meaning |
|-------:|------:|-------|---------|
| 0      | 2     | `d`   | f16 scale |
| 2      | 16    | `qs`  | 32 packed 4-bit weights |

Each byte holds two nibbles: low nibble → output `j`, high nibble → output `j+16`, with the
Q4_0 zero point of 8 subtracted: `out[j] = ((qs[j] & 0x0f) - 8) * d` (`src/quant.rs:217`).
So values 0..31 are *interleaved*, not sequential. 2 + 16 = 18 bytes.

### Q8_0 — 32 values, 34 bytes (`block_q8_0`, `src/quant.rs:206`)

| offset | bytes | field | meaning |
|-------:|------:|-------|---------|
| 0      | 2     | `d`   | f16 scale |
| 2      | 32    | `qs`  | 32 signed int8 weights |

`out[i] = d * (qs[i] as i8)` (`src/quant.rs:209`). 2 + 32 = 34 bytes.

### Q4_K — 256 values, 144 bytes (`block_q4_k`, `src/quant.rs:233`)

| offset | bytes | field    | meaning |
|-------:|------:|----------|---------|
| 0      | 2     | `d`      | f16 super-block scale |
| 2      | 2     | `dmin`   | f16 super-block min scale |
| 4      | 12    | `scales` | 8 sub-block (6-bit scale, 6-bit min) pairs, packed |
| 16     | 128   | `qs`     | 256 packed 4-bit weights |

4 + 12 + 128 = 144 bytes. The 12-byte scale block encodes eight 6-bit scales `sc[0..8]` and
eight 6-bit mins `m[0..8]`; `get_scale_min_k4(j, q)` (`src/quant.rs:223`) unpacks one pair:
the first four are `(q[j]&63, q[j+4]&63)`, the upper four steal their high two bits from
bytes 0..4 / 4..8 (`src/quant.rs:227`). Dequant: each 32-byte chunk yields two 32-value
sub-blocks, low nibbles then high nibbles, as `out = d*sc*nibble - dmin*min`
(`src/quant.rs:247-251`). `pack_scales_q4_k` (`src/quant.rs:1006`, `#[cfg(test)]`) is the
inverse used by fixtures.

### Q6_K — 256 values, 210 bytes (`block_q6_k`, `src/quant.rs:260`)

| offset | bytes | field    | meaning |
|-------:|------:|----------|---------|
| 0      | 128   | `ql`     | low 4 bits of 256 weights |
| 128    | 64    | `qh`     | high 2 bits of 256 weights |
| 192    | 16    | `scales` | 16 **signed int8** sub-block scales |
| 208    | 2     | `d`      | f16 super-block scale |

128 + 64 + 16 + 2 = 210 bytes. Each 6-bit weight is `(low4 | (hi2 << 4)) - 32`
(zero point 32), reconstructed in two halves of 128 with the four 2-bit planes peeled from
`qh` by `>>0/2/4/6` (`src/quant.rs:271-274`). The scales are signed: `sc[is] as i8`
(`src/quant.rs:275`), and a 16-element scale group covers each run of 16 weights.

─────────────────────────────────────────────────────────────────────

## Dequant paths (block → f32)

Three entry points, all routed through one match:

- `dequantize(ty, src, n) -> Result<Vec<f32>>` — allocates and fills (`src/quant.rs:99`).
- `dequantize_into(ty, src, out)` — fills a caller buffer; validates `out.len()` is a
  multiple of the block size and that `src` has `bytes_for(n)` bytes, else `Error::Format`
  (`src/quant.rs:106`). Iterates `src.chunks_exact(type_size())`.
- `dequant_block(ty, chunk, out)` — one block; `match` to `block_q4_0/q8_0/q4_k/q6_k`, with
  F32/F16 inlined (`src/quant.rs:130`).

These paths are used by the **F32 fallback** (when a backend wants a materialized f32 row),
by the embedding lookup / `QMatrix::dequant_row`, and as the **oracle** in tests
(`dequantize(..) · act.dequantized()` is the reference for every `vec_dot_*` test, e.g.
`src/quant.rs:1080-1083`). The hot matmul path does **not** use them — it uses the integer
dot kernels below.

`f32_to_f16` (`src/quant.rs:169`, round-to-nearest-ties-even) is the inverse, used only to
build test fixtures / demo GGUFs — explicitly off any hot path.

─────────────────────────────────────────────────────────────────────

## Activation quantizers

The matmul quantizes the activation vector `x` **once per matmul**, then dots it against
every weight row in integer arithmetic. Two formats, picked by the weight type.

### `Q8Activation` (block-32) — for Q8_0 / Q4_0

```rust
// src/quant.rs:289
pub struct Q8Activation {
    pub qs: Vec<i8>,        // length n (multiple of 32)
    pub scales: Vec<f32>,   // n/32 entries
    pub block_sums: Vec<i32>, // n/32 entries — Σ of the i8 quants per block
}
```

`quantize_activation_q8(x)` (`src/quant.rs:316`): per 32-block, `d = amax/127`,
`q = round(v/d).clamp(-128,127)`, and `block_sums[b] = Σ q`. `block_sums` exists **only**
for the VNNI path: `vpdpbusd` needs an *unsigned* weight operand, so signed weights are
biased `+128` and `128·Σx` is subtracted back exactly (`src/quant.rs:296-299`).

### `Q8KActivation` (block-256) — for Q4_K / Q6_K

```rust
// src/quant.rs:415
pub struct Q8KActivation {
    pub qs: Vec<i8>,     // length n (multiple of 256)
    pub d: Vec<f32>,     // n/256 super-block scales
    pub bsums: Vec<i16>, // n/16 — Σ of quants per group of 16
}
```

`quantize_activation_q8k(x)` (`src/quant.rs:436`) tracks the signed max (not just amax):
`iscale = -128/max`, `d = 1/iscale`, `q = round(iscale·v).clamp(-128,127)`, and `bsums[g]`
is the sum over each group of 16. All-zero blocks short-circuit to zeros
(`src/quant.rs:449`). `bsums` folds in the k-quant per-block **min** term (Q4_K) and the
**−32 zero-point** correction (Q6_K) without a second pass over the activation.

Both structs expose `dequantized()` (`src/quant.rs:304`, `:426`) reconstructing the lossy
f32 — used only to validate the kernels.

─────────────────────────────────────────────────────────────────────

## Integer dot kernels

`has_int8_path(ty)` (`src/quant.rs:579`) is true for Q8_0/Q4_0/Q4_K/Q6_K — the four that
have a `vec_dot_*`. Each public `vec_dot_*` is a runtime dispatcher; the scalar `_scalar`
fn is the portable reference and the SIMD oracle.

| kernel | weight type | activation | scalar ref | dispatcher |
|--------|-------------|-----------|------------|-----------|
| `vec_dot_q8_0` | Q8_0 (34 B) | `Q8Activation` | `src/quant.rs:363` | `src/quant.rs:348` |
| `vec_dot_q4_0` | Q4_0 (18 B) | `Q8Activation` | `src/quant.rs:394` | `src/quant.rs:380` |
| `vec_dot_q4_k` | Q4_K (144 B)| `Q8KActivation`| `src/quant.rs:490` | `src/quant.rs:476` |
| `vec_dot_q6_k` | Q6_K (210 B)| `Q8KActivation`| `src/quant.rs:544` | `src/quant.rs:528` |

### Scalar integer math

- **Q8_0** (`src/quant.rs:363`): per block, `sum = Σ (weight_i8 · act_i8)` in i32, then
  `acc += dw · act.scales[b] · sum`.
- **Q4_0** (`src/quant.rs:394`): unpack each nibble to `lo-8`/`hi-8` (matching the
  interleaved layout), multiply against `act.qs[base+j]` / `act.qs[base+j+16]`, same f32
  fold-out.
- **Q4_K** (`src/quant.rs:490`): 8 sub-blocks of 32; per sub-block `sub = Σ nibble·qx`,
  scaled by the 6-bit `sc` into an i32 `acc`. The min term is a *separate* accumulation
  `min_acc = Σ bsums[g]·m[g/2]`. Result: `d·bigd·acc − dmin·bigd·min_acc` — only the two
  block scales touch f32.
- **Q6_K** (`src/quant.rs:544`): reconstruct all 256 signed `(q−32)` weights into a stack
  `[i32;256]`, then 16 sub-blocks of 16 with signed `scales[sub] as i8`, summed as
  `d·bigd·acc`.

### AVX2 fast path (`src/quant.rs:811-981`)

`avx2_supported()` (`src/quant.rs:823`) gates it; disabled by `RUSTY_LLAMA_NO_AVX2`, cached
in a `OnceLock`. The AVX2 analog of `vpdpbusd` is `vpmaddubsw` (u8×i8→i16, adjacent pairs)
then `vpmaddwd` against ones (`madd_u8_i8`, `src/quant.rs:836`). For 4-/6-bit weights the
i16 intermediate cannot overflow → **bit-identical** to scalar; signed-8-bit Q8_0 widens to
i16 first via `dot32_signed_avx2` (`src/quant.rs:844`) to stay exact. Kernels:
`vec_dot_q8_0_avx2` (`:857`), `_q4_0_avx2` (`:872`), `_q4_k_avx2` (`:895`), `_q6_k_avx2`
(`:936`). **This is the path most consumer x86 (AVX2, no AVX-512) actually runs.**

### AVX-512 VNNI fast path (`src/quant.rs:597-809`)

`vnni_supported()` (`src/quant.rs:608`) requires `avx2 + avx512f + avx512bw + avx512vl +
avx512vnni`; disabled by `RUSTY_LLAMA_NO_VNNI`, cached. The primitive is
`_mm256_dpbusd_epi32` (`vpdpbusd`): 32 `u8`×`i8` products accumulated into 8 i32 lanes in
one instruction. `dot32_signed` (`src/quant.rs:639`) does the `^0x80` bias + `−128·xsum`
correction for the signed Q8_0/Q4_0 weights; Q4_K nibbles `[0,15]` and reconstructed Q6_K
weights `[0,63]` feed `vpdpbusd` unsigned directly with no bias. Q6_K corrects its `−32`
zero point per-16 via `bsums` (`src/quant.rs:802`).

### Dispatch order

Each public fn (e.g. `src/quant.rs:348`) tries **VNNI → AVX2 → scalar** under
`#[cfg(target_arch = "x86_64")]`; non-x86 builds always take the scalar arm. The SIMD
results are bit-for-bit equal to scalar (intentional design, asserted in tests), so this is
purely a speed swap.

### Consumer

`CpuBackend::matmul` (`src/backend/cpu.rs:60`) is the sole hot consumer: for
`QMatrix::Quant` it quantizes `x` once (`quantize_activation_q8` for Q8_0/Q4_0,
`quantize_activation_q8k` for Q4_K/Q6_K), picks the `dot` fn pointer, and runs
`out.par_iter_mut()` (rayon) over rows, each row a `&data[i*rb..]` slice
(`src/backend/cpu.rs:71-95`). F32 matrices take a contiguous-row autovectorized path
(`src/backend/cpu.rs:64`). See `docs/Architecture/02-backend-trait-and-cpu.md`.

─────────────────────────────────────────────────────────────────────

## Quantizers (f32 → blocks)

Only one weight quantizer exists: `quantize_q8_0(x) -> Vec<u8>` (`src/quant.rs:989`), which
mirrors ggml's reference (`d = amax/127`, write f16 `d` then 32 clamped i8). It is for test
fixtures / demo GGUFs — there is **no** quantizer for Q4_0/Q4_K/Q6_K and **no**
quantize-to-disk tooling. The activation quantizers (`quantize_activation_q8/q8k`, above)
are the only other f32→quant routines and are runtime, not persisted.

─────────────────────────────────────────────────────────────────────

## `QMatrix` (`src/tensor.rs`)

The matmul's input matrix abstraction — two variants over a `Cow` so it can borrow the mmap
or own a buffer (`src/tensor.rs:15`):

```rust
pub enum QMatrix<'a> {
    F32   { data: Cow<'a, [f32]>, rows: usize, cols: usize },
    Quant { ty: GgmlType, data: Cow<'a, [u8]>, rows: usize, cols: usize },
}
```

`rows` = output features, `cols` = input features, **row-major** (`src/tensor.rs:13`).

| constructor | validates |
|-------------|-----------|
| `QMatrix::f32(data, rows, cols)` (`src/tensor.rs:33`) | `data.len() == rows*cols` |
| `QMatrix::quant(ty, data, rows, cols)` (`src/tensor.rs:45`) | `cols` multiple of `ty.block_size()`; `data.len() >= ty.bytes_for(rows*cols)` |

`rows()` / `cols()` (`src/tensor.rs:68`, `:75`) read the field from either variant.
`dequant_row(row, out)` (`src/tensor.rs:84`) is the bridge to f32: F32 does a
`copy_from_slice` of `data[row*cols..]`; Quant computes `rb = bytes_for(cols)` and calls
`dequantize_into` on the row's bytes. Because lengths are validated at construction,
`dequant_row` discards the `Result` (`src/tensor.rs:94`). The Quant variant **keeps weights
compressed in RAM** — the full f32 expansion never materializes; the matmul decompresses one
block at a time (`src/tensor.rs:3-6`).

`QMatrix::Quant` is *zero-copy* against the GGUF mmap when constructed from a borrowed
`Cow::Borrowed`; see `docs/Architecture/06-gguf-and-loading.md` for how loader hands these
out, and `docs/Architecture/01-model-and-forward-pass.md` for where weights are matmul'd.

─────────────────────────────────────────────────────────────────────

## Testing

`#[cfg(test)] mod tests` covers, with `path:line`:
- **Round-trips**: `f16_roundtrip_simple_values` (`:1021`), `q8_0_roundtrip` (`:1029`),
  `q4_k_constant_block` (`:1040`), `q6_k_constant_block` (`:1053`, exercises signed scale).
- **Length validation**: `rejects_bad_lengths` (`:1067`).
- **Kernel correctness vs dequant oracle**: `vec_dot_q8_0/q4_0/q4_k/q6_k_equals_dequantized_dot`
  (`:1073`, `:1087`, `:1106`, `:1128`).
- **AVX2 bit-identity (runs on any AVX2 CPU)**: `q8_0/q4_0/q4_k/q6_k_avx2_matches_scalar`
  (`:1170`, `:1193`, `:1216`, `:1240`) — these are run-verified on the dev box (which has no
  AVX-512), so the AVX2 path is exercised, not just compiled.
- **VNNI bit-identity (only runs where VNNI present)**: `q8_0/q4_0/q4_k/q6_k_simd_matches_scalar`
  (`:1263`, `:1287`, `:1313`, `:1340`) — early-return when `vnni_supported()` is false.
- **VNNI algorithm cross-check**: `q6_k_vnni_algorithm_matches_scalar` (`:1414`).
- **`#[ignore]` timing benches**: `bench_q{8_0,4_0,4_k,6_k}_simd_vs_scalar` (`:1435`, `:1477`,
  `:1516`, `:1555`). CPU-backend matmul benches live in `src/backend/cpu.rs:265`, `:350`.

See `docs/Architecture/08-testing-benchmarking-parity.md`.

─────────────────────────────────────────────────────────────────────

## Status, gaps & notes

- **Type coverage is narrow**: only F32, F16, Q4_0, Q8_0, Q4_K, Q6_K. **No** IQ* (IQ2/IQ3/
  IQ4 family), **no** Q5_K / Q2_K / Q3_K, **no** Q4_1/Q5_0/Q5_1, **no** MXFP4. Any other
  `ggml_type` id is a hard `Error::Format` at load (`src/quant.rs:51`). This rules out many
  popular community quants until the dequantizer + dot kernel are added per type.
- **No quantize-to-disk tooling**: the only weight quantizer is `quantize_q8_0`
  (`src/quant.rs:989`), and it's a test/fixture helper — rusty_llama consumes pre-quantized
  GGUFs but cannot produce them. There is no Q4_K/Q6_K encoder at all.
- **VNNI status**: the AVX-512 VNNI kernels (`src/quant.rs:597-809`) are **present and wired
  into dispatch**, but on the current dev hardware (Intel Core Ultra 9 285H — no AVX-512)
  `vnni_supported()` returns false, so they are **dormant / not run-verified** there; only
  the AVX2 path and scalar oracle execute. `q*_simd_matches_scalar` tests early-return on
  such CPUs. `q6_k_vnni_algorithm_matches_scalar` (`:1414`) tests the VNNI *algorithm*
  portably as a partial guard. Escape hatches: `RUSTY_LLAMA_NO_VNNI`, `RUSTY_LLAMA_NO_AVX2`.
- **Per-block scaling stays f32**; only the inner products are integer. The min/zero-point
  corrections (`block_sums`, `bsums`) are exact integer folds, so all three paths agree to
  the bit — but accuracy is still bounded by the source quantization, not improved here.
- **`bytes_for` truncates** on non-block-multiple `n` (integer division, `src/quant.rs:93`);
  callers (`QMatrix::quant`, `dequantize_into`) validate the multiple separately, so this is
  safe only behind those guards.
- The hand-rolled `f16_to_f32` (`src/quant.rs:144`) does not use hardware F16C; it's a hot
  enough path (called per block in every dequant and SIMD kernel) to be a future SIMD/F16C
  target.

llama.cpp counterpart: `docs/Research/04-quantization.md` (k-quant block formats, `vec_dot`
families, and the much larger type matrix there).
