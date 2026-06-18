# 03. CUDA Kernels: the Performance Story

## Summary

This doc is the *why-is-llama.cpp-fast-on-NVIDIA* capture: the matmul-routing
brain in `ggml/src/ggml-cuda/ggml-cuda.cu` and the four GEMM/GEMV kernel
families it dispatches to (MMQ, MMVQ, MMF/MMVF, cuBLAS), plus the flash-attention
family (`fattn*.cu`). Three things matter most. (1) **Routing by batch size**:
quantized decode (batch ≤ 8 tokens) goes to **MMVQ**, a GEMV that reads weights
in their *packed* quant format and dot-products them with DP4A int8 — the
decode **bandwidth** win; quantized prefill (batch > 8) goes to **MMQ**, a
tile-based GEMM that runs on **int8 tensor cores** via `mma.cuh` PTX wrappers —
the prefill **compute** win, and it beats dequant→cuBLAS because the weights
never round-trip through VRAM as f16. (2) **Fused flash attention** with online
softmax and no materialized score matrix, selecting per-shape among MMA/tile/vec
kernels. (3) **Years of per-arch tuning tables** baked into every kernel. Quant
block *formats* are owned by `04-quantization.md`; the generic graph scheduler
by `02-backends-and-dispatch.md` — this doc only references them.

─────────────────────────────────────────────────────────────────────────────

## 1. Matmul routing — the decision that decides everything

`GGML_OP_MUL_MAT` lands in `ggml_cuda_mul_mat(ctx, src0, src1, dst)`
(`ggml-cuda.cu`, ~L2541). `src0` is the **weight** matrix (possibly quantized),
`src1` the **activations** (always `F32`), `dst` the `F32` result. The pivotal
quantity is `src1->ne[1]` = **`ne11` = number of tokens in the batch** (the GEMM
"N"). The function computes four candidate booleans, refines each with a
`ggml_cuda_should_use_*` predicate, then dispatches in a fixed priority order.

### 1.1 Candidate eligibility (`ggml-cuda.cu` ~L2550)

| Candidate | Base predicate | Kernel | Role |
| --- | --- | --- | --- |
| `use_mul_mat_vec_f` | src0 ∈ {F32,F16,BF16}, src1 F32, dst F32 | `mmvf` GEMV | f16 **decode** (thin src0, KQV) |
| `use_mul_mat_f` | src0 **not** quantized | `mmf` GEMM | f16 **prefill** (tensor cores) |
| `use_mul_mat_vec_q` | src0 quantized **and `ne11 ≤ MMVQ_MAX_BATCH_SIZE (8)`** | `mmvq` GEMV | quant **decode** |
| `use_mul_mat_q` | src0 quantized | `mmq` GEMM | quant **prefill** (tensor cores) |

Each is then AND-ed with its arch/shape predicate:
`use_mul_mat_q &= ggml_cuda_should_use_mmq(type, cc, ne11, n_experts)`,
`use_mul_mat_vec_q &= ggml_cuda_should_use_mmvq(type, cc, ne11)`, etc.
`any_gpus_with_slow_fp16 = !fast_fp16_hardware_available(cc)` enables a cuBLAS
f16 path only where f16 is fast.

### 1.2 Dispatch order (non-split, `ggml-cuda.cu` ~L2608)

```
if      use_mul_mat_vec_f  -> ggml_cuda_mul_mat_vec_f          // f16/f32 GEMV (decode)
else if use_mul_mat_f      -> ggml_cuda_mul_mat_f             // f16 tensor-core GEMM
else if use_mul_mat_vec_q  -> ggml_cuda_mul_mat_vec_q         // MMVQ (quant decode)
else if use_mul_mat_q      -> ggml_cuda_mul_mat_q             // MMQ (quant prefill)
else if batched cuBLAS && src1->ne[2]*ne[3] > 1 -> ggml_cuda_mul_mat_batched_cublas  // KQ/KQV
else                       -> ggml_cuda_op_mul_mat_cublas     // dequant->f16->cuBLAS fallback
```

Because **`vec_q` is checked before `q`**, a quantized matmul with `ne11 ≤ 8`
takes MMVQ; once the batch exceeds 8 the `vec_q` candidate is disabled and the
same op falls to MMQ. **This is the literal prefill/decode split.**

### 1.3 The explicit rule (NVIDIA, quantized weights)

Combining `should_use_mmvq` (`mmvq.cu` ~L300) and `should_use_mmq`
(`mmq.cu` ~L289):

- **Decode, `ne11 ≤ 8`** → **MMVQ**. On NVIDIA `ggml_cuda_should_use_mmvq`
  returns `ne11 ≤ MMVQ_MAX_BATCH_SIZE (8)` (CDNA has per-type tables).
- **Prefill, `ne11 > 8`** → **MMQ** on Turing+ (`turing_mma_available(cc)`,
  i.e. `cc ≥ 750` → int8 tensor cores) returns `true` unconditionally. On
  **Volta** (700, no int8 MMA) MMQ is used only while `ne11 < MMQ_DP4A_MAX_BATCH_SIZE (64)`,
  else the **cuBLAS dequant path**. On pre-DP4A Pascal (`< 610`) MMQ is disabled.
- `GGML_CUDA_FORCE_MMQ` / `GGML_CUDA_FORCE_CUBLAS` env vars override.

`ggml_cuda_should_use_mmq` body (NVIDIA, after the supported-type switch):
```c
if (turing_mma_available(cc)) return true;                       // int8 tensor cores -> always MMQ
if (highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) return false;  // no dp4a -> cuBLAS
return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE; // Volta: <64 tokens
```

For **non-quantized** weights the analogous split is `mmvf` (decode, `ne11 ≤
MMVF_MAX_BATCH_SIZE = 8`) vs `mmf` (tensor-core f16 GEMM) vs `batched_cublas`
(the multi-batch KQ/KQV path used when flash-attention is off).
`MUL_MAT_ID` (MoE) has its own dispatcher `ggml_cuda_mul_mat_id` (~L2633): tiny
batch → `mul_mat_vec_q`/`mul_mat_vec_f` gated by `get_mmvq_mmid_max_batch(type,
cc)`, else MMQ, else MMF.

### 1.4 Capability predicates (`common.cuh` ~L315)

| Predicate | True when (NVIDIA) | Gates |
| --- | --- | --- |
| `fast_fp16_hardware_available` | `cc ≥ Pascal (600)`, `≠ 610` | cuBLAS f16 vs f32 |
| `fp16_mma_hardware_available` | `cc ≥ Volta (700)` | f16 tensor cores |
| `turing_mma_available` | `cc ≥ Turing (750)` | **int8 MMQ + flash MMA** |
| `bf16_mma_hardware_available` | `cc ≥ Ampere (800)` | bf16 batched cuBLAS |
| `blackwell_mma_available` | `1200 ≤ cc < 1300` | native MXFP4/NVFP4 mma |

CC ladder (`common.cuh` ~L56): Pascal 600 · DP4A 610 · Volta 700 · Turing 750 ·
Ampere 800 · Ada 890 · Hopper 900 · Blackwell 1200. AMD CDNA/RDNA and Moore
Threads MUSA get parallel offsets and their own tuning everywhere.

─────────────────────────────────────────────────────────────────────────────

## 2. MMQ — quantized matmul on tensor cores (prefill)

`mmq.cuh` (~4.2k lines) + `mmq.cu`. The prefill workhorse. Pipeline
(`ggml_cuda_mul_mat_q`, `mmq.cu` ~L60):

1. **Quantize activations to q8_1 on the fly** in a transposed MMQ-specific
   layout (`quantize_mmq_q8_1_cuda`, §6) — int8 quant of the *activations*, not
   the weights. Padded to `MATRIX_ROW_PADDING = 512`.
2. Pack args into `mmq_args` and `mul_mat_q_case<type>` → `launch_mul_mat_q<type,
   mmq_x>` → the `mul_mat_q<type, mmq_x, need_check>` `__global__` kernel.

### 2.1 Tiles and shared memory

- The K-dim tile unit is `MMQ_TILE_NE_K = 32` 32-bit elements; `MMQ_NWARPS = 8`.
- `mmq_x` = batch/column tile (8…128, the loop in `mul_mat_q_case` picks the best
  multiple of the granularity that fits `smpbo` shared memory); `mmq_y` = row
  tile = **128 on Volta+**, 64 otherwise (`get_mmq_y_host`). `mmq_x` max is
  **128 with tensor cores**, else 64 (`get_mmq_x_max_host`).
- Per-type **tile loaders** `load_tiles_q4_K / q6_K / q8_0 / …` (`mmq.cuh`
  ~L2096) stream the **packed** quant weights from global → shared memory,
  unpacking the bit-layout into the int tile. Tile widths
  `MMQ_MMA_TILE_X_K_*` are padded so `% 8 == 4` to dodge shared-memory bank
  conflicts (`static_assert`s enforce it).
- The `block_q8_1_mmq` activation struct (`mmq.cuh` ~L30): 128 int8 values +
  4×`half2` scales/sums, transposed into 128-value groups + 16-byte pad —
  "a contiguous block that can be copied straight to shared memory."

### 2.2 Two compute paths per type

Each `mmq_type_traits<type>` (`mmq.cuh` ~L3270) binds three function pointers:
`load_tiles`, `vec_dot_mma` (tensor cores), `vec_dot_dp4a` (fallback). E.g.
`Q4_K` → `load_tiles_q4_K`, `vec_dot_q8_1_q8_1_mma`, `vec_dot_q4_K_q8_1_dp4a`.
The MMA path is taken when `turing_mma_available`.

### 2.3 `mma.cuh` — the tensor-core PTX wrappers

`mma.cuh` exposes nvcuda::wmma-style fragments with a defined layout:
`tile<I, J, T, data_layout>` (A row-major M×K, B col-major K×N, C col-major
M×N; J in physical 32-bit elements). Two primitives matter:

- **`load_ldmatrix`** → `ldmatrix.sync.aligned.m8n8.x2/x4.b16` (Turing+), shared
  memory only, 16-byte aligned. Cooperatively loads an 8×8/16×8 fragment across
  the warp. `load_generic` is the non-`ldmatrix` fallback (and is sometimes
  *faster* for the B tile — comments call it out).
- **`mma`** → the actual tensor-core multiply-accumulate. For MMQ's int8 path:
  ```
  Ampere+: mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32   // 16×8×32 int8 -> s32
  Turing : 4× mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 // m16n8k32 unavailable
  ```
  Also f16 (`m16n8k16.f16.f16.f16.f16`), bf16, tf32 (`m16n8k8.f32.tf32.tf32.f32`),
  and Blackwell block-scaled fp4 (`mma.sync...kind::mxf4...block_scale...m16n8k64`).
  A `vec_dot_q8_0_q8_1_mma` (`mmq.cuh` ~L1163) loads A via `load_ldmatrix`, B via
  `load_generic`, issues `mma(C, A, B)`, then folds the per-block `half2` scales.

### 2.4 Stream-K + load balancing

`use_stream_k` (Volta+ NVIDIA or CDNA) splits the K dimension across thread
blocks for SM occupancy on skewed shapes; partials land in a `tmp_fixup` pool
buffer and `mul_mat_q_stream_k_fixup` recombines them (`mmq.cu`/`mmq.cuh`
~L3788, L4006). Grid width is chosen from a `tiles_efficiency_percent ≥ 90`
heuristic.

### 2.5 Why MMQ beats dequant→cuBLAS

The fallback `ggml_cuda_op_mul_mat_cublas` (`ggml-cuda.cu` ~L1631) calls
`to_fp16_cuda = ggml_get_to_fp16_cuda(src0->type)` (from `convert.cu`) to
**dequantize the entire weight matrix into a fresh f16 VRAM buffer**, then runs
`cublasGemmEx(..., CUBLAS_GEMM_DEFAULT_TENSOR_OP)`. That f16 buffer is 4× the
size of a 4-bit weight and must be written then re-read — pure extra DRAM
traffic. **MMQ keeps weights packed in global memory, dequantizes in
shared/registers right before the int8 `mma`, and uses int8 tensor cores
(≈2× the f16 TOPS).** Net: less bandwidth **and** more math throughput — the
prefill win.

─────────────────────────────────────────────────────────────────────────────

## 3. MMVQ — the decode GEMV (bandwidth win)

`mmvq.cu` / `mmvq.cuh`. `MMVQ_MAX_BATCH_SIZE = 8`. The decode path for quantized
weights, batch ≈ 1.

### 3.1 Kernel structure (`mmvq.cu` ~L482, `mul_mat_vec_q<type, ncols_dst, …>`)

- **Activations → q8_1** once per step (`quantize_row_q8_1_cuda`, §6), giving 32
  int8 + a `half2(d, sum)` per block. Weights `vx` are read **directly in their
  packed quant byte layout** — never expanded.
- **Grid** `(nblocks_rows, nchannels_dst, n_tokens)`, **block**
  `(warp_size, nwarps)`. Each thread strides over the row's quant blocks
  (`for kbx = tid/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter`),
  calls the per-type `vec_dot_q_cuda(vx, &y[…], …)` (§4), accumulating
  `tmp[ncols_dst][rows_per_cuda_block]`.
- **Reduction**: `warp_reduce_sum` within a warp, then a `__shared__
  tmp_shared[nwarps-1][…]` cross-warp reduction; lane 0 writes `dst`.
- **Coalesced + warp-cooperative**: lanes within a warp read consecutive packed
  bytes of one row, so the GEMV streams each weight exactly once at the **packed
  bit-rate** (e.g. Q4_K ≈ 4.5 bits/weight). Since decode is bandwidth-bound and
  the weight is read once with no reuse, *bytes streamed = time* — this is the
  decisive decode lever.
- **Fusion**: a `has_fusion` template variant folds bias + GLU (SwiGLU/GeGLU/
  SwiGLU-OAI) into the GEMV epilogue (`ggml_cuda_should_fuse_mul_mat_vec_q`,
  `ggml-cuda.cu` ~L2504), saving a round-trip on the MLP.

### 3.2 Per-arch tuning tables

`mmvq.cu` is a museum of measured constants:
- `get_device_table_id` → one of GENERIC / TURING / GCN / RDNA2 / RDNA3_0 /
  RDNA4 parameter sets.
- `calc_nwarps(type, ncols_dst, table_id)` and `calc_rows_per_block(…)` pick
  threadblock shape per arch *and* per quant type (e.g. RDNA4 uses `nwarps=8`
  only for simple-`vec_dot` types; Q3_K/IQ2/IQ3 regress from register pressure
  and stay at 1).
- `get_mmvq_mmid_max_batch_{pascal,turing_plus,gcn,cdna,rdna*}` — per-type MoE
  batch ceilings, each citing the PR that measured them.
- `should_use_small_k` raises `rows_per_block` when K is small so warps aren't
  idle.

These tables are not derivable from first principles; they are accumulated
benchmark results — a large part of llama.cpp's edge.

─────────────────────────────────────────────────────────────────────────────

## 4. `vecdotq.cuh` — per-type DP4A dot products

The shared kernel of both MMVQ and the DP4A MMQ fallback. Every type provides a
`vec_dot_<type>_q8_1` that **unpacks the packed weight into int registers and
dot-products it against q8_1 activations with `ggml_cuda_dp4a`**
(`common.cuh` ~L698 → `__dp4a(a, b, c)`: four int8×int8 products summed into an
int32, CC ≥ 6.1; scalar fallback otherwise).

Idioms (all in-register, no shared memory):

- **Nibble unpack**: `(v >> 0) & 0x0F0F0F0F` / `(v >> 4) & 0x0F0F0F0F` splits a
  32-bit word into eight 4-bit weights packed as bytes for `dp4a`.
- **Offset folding** (`vec_dot_q4_0_q8_1_impl`): the symmetric "−8" is not
  applied per-element; instead `sumi * d8.x − (8·vdr/QI4_0)·d8.y`, using the
  q8_1 block **sum** `s` to subtract the constant in one multiply.
- **`vec_dot_q4_K_q8_1`** (~L869): loads two packed words `v[2]`, unpacks the
  6-bit sub-block scales/mins from `bq4_K->scales` (the `0x3f3f` / `0xc0c0`
  masking), then `impl_vmmq` does, per sub-block `i`: `dot1 = dp4a(v1i,u1,
  dp4a(v0i,u0,0))` for the scaled term and `dot2 = dp4a(0x01010101, …)` for the
  min term → `dm4.x·Σ d8·sc·dot1 − dm4.y·Σ d8·m·dot2`.
- **`vec_dot_q6_K_q8_1`** (~L961): combines `ql` (4 bits) and `qh` (2 bits) into
  6-bit values, `__vsubss4((vil|vih), 0x20202020)` subtracts 32, then `dp4a`
  with per-group int8 `scales` × `d6`.

`VDR_<type>_Q8_1_MMVQ` (1–2) is how many contiguous 32-bit words each thread
chews per call; the MMQ variant `VDR_*_MMQ` (4–8) is larger.

─────────────────────────────────────────────────────────────────────────────

## 5. Flash attention family

`fattn.cu` is a pure **selector**; the kernels live in
`fattn-mma-f16.cuh`, `fattn-tile.cu`, `fattn-vec.cuh`, `fattn-wmma-f16.cu`, with
shared machinery in `fattn-common.cuh`. Op = `GGML_OP_FLASH_ATTN_EXT`.

### 5.1 Kernel selection (`ggml_cuda_get_best_fattn_kernel`, `fattn.cu` ~L346)

Inputs: `cc`, head dim `D = Q->ne[0]`, batch `Q->ne[1]`, `gqa_ratio =
Q->ne[2]/K->ne[2]`, K/V type, and `gqa_opt_applies` (mask present, no ALiBi,
`K->ne[1] % FATTN_KQ_STRIDE(256) == 0`, 16-byte strides).

| Kernel | Chosen when | Notes |
| --- | --- | --- |
| `BEST_FATTN_KERNEL_VEC` (`fattn-vec.cuh`) | small batch / **decode** (`Q->ne[1]==1`, or `≤2` for quantized KV) | per-element, DP4A KQ dots for quantized KV |
| `BEST_FATTN_KERNEL_MMA_F16` (`fattn-mma-f16.cuh`) | `turing_mma_available` & larger batch / **prefill** | int8/f16 tensor cores; GQA-packed `ncols2` |
| `BEST_FATTN_KERNEL_TILE` (`fattn-tile.cu`) | no usable tensor cores | generic fallback |
| `BEST_FATTN_KERNEL_WMMA_F16` (`fattn-wmma-f16.cu`) | `should_use_wmma_fattn` (older/Volta) | legacy nvcuda::wmma |
| `NONE` | unsupported D / mask / type | falls back to non-fused KQ·KQV path |

The MMA path further switches on head dim (64/80/96/112/128/192/256/320/512/576
— DeepSeek-MLA 576, GLM/MiMo special cases) then on `ncols2` (how many GQA query
heads share one KV load) and `ncols1`, picking the largest tile that fits the
batch (`switch_ncols2`/`switch_ncols1`). Supported D and KV quant types are
gated up front; head dims 40/72/192 are special-cased out of the vector path.

### 5.2 Online softmax — no materialized score matrix

`fattn-common.cuh`. `FATTN_KQ_STRIDE = 256` is the KV tile length. The kernel
walks K/V in tiles, keeping a **running row-max `m` and denominator `l`** instead
of ever writing the full `Q·Kᵀ` scores: on each new tile it rescales the
accumulated `VKQ` by `exp(m_old − m_new)` and the softmax of the new scores by
`exp(s − m_new)` (the standard FlashAttention recurrence). `HALF_MAX_HALF` seeds
the max to avoid NaN; values below `SOFTMAX_FTZ_THRESHOLD (-20)` flush to zero;
`FATTN_KQ_MAX_OFFSET (3·log2)` shifts the accumulator range up by 2³ to avoid
overflow. **Scores never touch DRAM**, which is the whole point — attention is
otherwise O(seqlen²) bandwidth.

### 5.3 Split-K combine

When one (row, head) is split across `parallel_blocks` thread blocks (chosen by
an occupancy/efficiency search, ~L1117), each writes a partial `VKQ` plus a
`float2 dst_meta = (max, denom)`. `flash_attn_combine_results` (~L919) reloads
all partials, takes the global max, rescales each by `expf(meta[l].x − kqmax)`,
and emits `Σ scale·VKQ / Σ scale·denom`. Stream-K tail tiles are fixed up by
`flash_attn_stream_k_fixup_{uniform,general}` with the same `exp`-rescale logic.

### 5.4 Quantized KV cache

`fattn-common.cuh` provides `vec_dot_fattn_vec_KQ_{f16,bf16,q4_0,q4_1,q5_0,
q5_1,q8_0}`. For a quantized KV cache the **query is quantized to q8** and the
K·Q dot uses `ggml_cuda_dp4a` over the packed K bytes — the same packed-weight
trick as MMVQ, applied to the cache. So KV can live in f16 *or* Q4_0/Q5_0/Q8_0,
halving/quartering KV bandwidth and VRAM.

─────────────────────────────────────────────────────────────────────────────

## 6. `quantize.cu` — activations → q8_1

Both quant matmul kernels feed on **int8 activations**, produced here.

- **`quantize_q8_1`** (MMVQ path): one `block_q8_1` per `QK8_1 = 32` activations.
  Per block: `amax = warp_reduce_max(|x|)`, `d = amax/127`, `q = round(x/d)`,
  and `ds = half2(d, sum)` where `sum = Σx` is kept so asymmetric weight formats
  (Q4_1/Q5_1/Q4_K min terms) can subtract their constant via the §4 fold.
- **`quantize_mmq_q8_1`** (MMQ path): writes the transposed `block_q8_1_mmq`
  layout (128 vals/block, vectorized `float4` loads, `char4` stores) so the MMQ
  tile loader can `memcpy` straight to shared memory. The `ds_layout`
  (`D4`/`DS4`/`D2S6`, chosen by `mmq_get_q8_1_ds_layout(type)`) controls whether
  per-32 sums are stored, matching what each weight type's dot needs.
- Blackwell adds `quantize_mmq_nvfp4` / `quantize_mmq_mxfp4` (FP4 activations
  with E8M0/UE4M3 shared exponents) for native FP4 tensor cores.

Host launchers: `quantize_row_q8_1_cuda`, `quantize_mmq_q8_1_cuda`,
`quantize_mmq_fp4_cuda`.

─────────────────────────────────────────────────────────────────────────────

## Relevance to rusty_llama

rusty_llama already has tensor-core CUDA prefill (cuBLASLt TF32→f16) and a
resident f16 decode; measured (TinyLlama-1.1B Q4_K_M, RTX 5070 Ti) ~4.4×
behind on prefill (`pp512` 4,450 vs 19,637) and ~3.4× behind on decode
(`tg128` ~123 vs 419 tok/s) per `PERFORMANCE.md`. This subsystem maps directly
onto both walls.

- **Decode bandwidth gap is exactly the MMVQ design.** rusty_llama streams
  **f16** weights at ~2 B/weight; llama.cpp's MMVQ reads weights in **packed
  Q4_K** (~0.56 B/weight) and dequantizes in-register with `__dp4a` — the
  ~3.5× ratio that *is* the decode gap. The actionable port is a dedicated
  batch-1 GEMV that reads the packed Q4_K/Q6_K/Q8_0 bytes directly (à la
  `mul_mat_vec_q` + `vec_dot_q4_K_q8_1`), **not** the f16 GEMM at n=1.
  Medium effort, high ROI — it attacks the dominant decode cost.
- **Note rusty_llama's own DP4A dead-end was a different comparison.** Its
  Stage-2 microbench compared int8-DP4A vs an *in-shader-dequant GEMV that
  already read packed bytes* (so neither saved bandwidth → break-even). The win
  here is replacing the **f16-weight** decode path with a **packed-weight** one;
  the bandwidth saving is real precisely because today's shipped decode expands
  to f16 first.
- **Prefill "cuBLASLt wall" is the MMQ insight.** rusty_llama caches
  **dequantized f16 weights** and runs cuBLASLt — structurally the same as
  llama.cpp's *fallback* `ggml_cuda_op_mul_mat_cublas`, which llama.cpp avoids.
  MMQ keeps weights packed and feeds int8 tensor cores via `mma.cuh`. Porting a
  real MMQ-style tiled kernel is large effort (per-type tile loaders + `ldmatrix`
  + `mma.sync` PTX + shared-mem padding), but it is the only way past the cuBLASLt
  ceiling rusty_llama already identified.
- **`mma.cuh` is a portable blueprint for tensor cores.** rusty_llama's CUDA
  backend is cudarc/nvrtc; the `ldmatrix.sync.aligned.m8n8.x4.b16` +
  `mma.sync.aligned.m16n8k32.s32.s8.s8.s32` (int8) / `m16n8k16.f16...` (f16) PTX
  pairs are exactly what an nvrtc kernel would emit. Lifting just the f16
  `m16n8k16` wrapper would let rusty_llama's fused prefill stop calling cuBLASLt.
- **Fused flash attention.** rusty_llama already has a cooperative
  block-per-(row,head) attention with shared-memory softmax (no global scratch) —
  conceptually the `fattn-vec`/`tile` design. Adding the **online-softmax
  running-max/denom recurrence** (`fattn-common.cuh`) removes the score-matrix
  entirely and is the prerequisite for long context; the split-K
  `flash_attn_combine_results` combine is a clean, low-risk pattern to copy.
- **Activation quantization is cheap and reusable.** `quantize_q8_1` (per-32
  `amax/127`, keep the sum) is a ~20-line kernel that unlocks *both* MMVQ decode
  and MMQ prefill. rusty_llama would need it once; it already has the CPU
  integer-dot oracles (`vec_dot_q4_k`/`q6_k`) to parity-check it against.
- **Per-arch tuning is the long tail, not the entry fee.** The `calc_nwarps` /
  `get_mmvq_mmid_max_batch` / fattn `ncols2` tables are years of measured
  constants. rusty_llama can ship a single sensible tile config and still
  capture most of the win; matching the tables is diminishing returns and not
  worth chasing for a legible reference engine.
- **Scope check vs feature breadth.** MMQ/MMVQ/fattn support ~20 quant types and
  dozens of head-dim/GQA shapes; rusty_llama needs only Q4_K/Q6_K/Q8_0 and
  Llama's D=64 (GQA 32/4). The kernels above can be ported as **narrow
  single-type specializations**, sidestepping the dispatch combinatorics — far
  smaller effort than the upstream breadth implies. See `04-quantization.md` for
  the block formats these kernels consume and `02-backends-and-dispatch.md` for
  how the scheduler reaches `ggml_cuda_mul_mat`.
