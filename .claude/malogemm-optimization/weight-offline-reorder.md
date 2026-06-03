# Weight Offline Reorder

Date: 2026-05-14

Scope: the 4-bit weight preprocessing used by
`examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_fp8_grouped_gemm.cu`.
This note covers the CUTLASS tree only.

## Executive Summary

The grouped mixed-dtype example does not feed the original packed weight buffer
to the GEMM kernel. It creates a second buffer, `block_B_interleaved`, and passes
that pointer array to CUTLASS:

```text
original B          -> block_B              -> reference path
offline reordered B -> block_B_interleaved  -> GEMM mainloop
```

The reorder is size-preserving and happens before the benchmark timing loop.
In a production setting this should be treated as an offline weight preparation
step performed when weights are loaded or when a layer shape is initialized.

The goal is to make the runtime mainloop cheap:

- TMA can load regular 2D tiles.
- LDSM can move sub-byte data from shared memory to registers in the shape the
  GMMA A-side fragment expects.
- Runtime conversion can operate on compact 32-bit registers:
  - INT4 x FP8 uses a LUT/PRMT converter on normal int4 nibbles.
  - FP4 x BF16 uses a converter that expects pre-interleaved FP4 bit fields.

## Call Path

In `69_hopper_int4_fp8_grouped_gemm.cu`:

```cpp
ptr_B_host[i] = block_B_interleaved.get() + offset_B[i];
```

After random initialization of `block_B`, the example chooses one reorder path:

```cpp
if QuantType == float_e2m1_t && MmaType == bfloat16_t:
  interleave_fp4xbf16_Hopper(block_B, block_B_interleaved, N, K)

else if QuantType == int4b_t && MmaType == float_e4m3_t:
  interleave_int4xfp8_Hopper(block_B, block_B_interleaved, N, K)

else:
  block_B_interleaved = block_B
```

The reference kernel still consumes `block_B`, not `block_B_interleaved`.

The wrapper functions launch:

```cpp
dim3 block(16, 32);
interleave_fp4xbf16_Hopper_kernel_combine_triton<<<1024, block>>>(...);
interleave_int4xfp8_Hopper_kernel<<<1024, block>>>(...);
```

## Source Layout

For each group, the source weight matrix is logically:

```text
B[N, K]
```

where `N` is the original output-channel dimension and `K` is the original
reduction dimension. The storage is row-major packed 4-bit:

```text
row r, element k -> byte/word inside row r
```

For FP4 code, the kernel views the source as `uint8_t*`, one byte holding two
4-bit values.

For INT4 x FP8, the kernel views the source as `uint16_t*`, one word holding
four 4-bit values.

The destination has the same total byte size and is addressed with the same
logical strides, but the bytes/words within each 16-row by 64-K-element tile
are rearranged.

## Common Tile Pattern

Both Hopper reorder kernels operate on a macro-tile:

```text
16 rows x 64 K-elements
```

Rows are paired across the two 8-row halves of a 16-row tile:

```text
(row 0, row 8), (row 1, row 9), ..., (row 7, row 15)
```

The row-pair formula is:

```cpp
row_id = block_id / 8 * 16 + block_id % 8;
paired_row_id = row_id + 8;
```

This means `rows` must be a multiple of 16 for the current kernels to avoid
out-of-bounds row pairs. `cols` must be a multiple of 64 because the loops use
`cols / 64`.

The existing option parser does not enforce all of these conditions, so callers
should treat them as required invariants.

## INT4 x FP8 Mapping

Function:

```text
interleave_int4xfp8_Hopper_kernel
```

Data view:

```text
uint16 word = 4 int4 values
64 K-elements = 16 uint16 words per row
```

For one `partition_id` and one `lane_id` in `0..15`:

```cpp
row_id     = block_id / 8 * 16 + block_id % 8;
row_pair   = row_id + 8;

src_col    = partition_id * 16 + lane_id;  // uint16 column
src_a      = src(row_id,   src_col);
src_b      = src(row_pair, src_col);

dst_row    = row_id + (lane_id % 8) / 4 * 8;
mma_id     = lane_id / 8;
dst_col    = partition_id * 16 + mma_id * 8 + (lane_id % 4) * 2;

dst(dst_row, dst_col)     = src_a;
dst(dst_row, dst_col + 1) = src_b;
```

So each destination row interleaves adjacent 16-bit words from the two source
rows. Lanes `0..3` and `8..11` land in the lower row; lanes `4..7` and
`12..15` land in the upper row.

For a row pair `(r, r+8)` and a 64-element K partition:

```text
dst row r:
  words 0,2,4,6    <- src row r   words 0..3
  words 1,3,5,7    <- src row r+8 words 0..3
  words 8,10,12,14 <- src row r   words 8..11
  words 9,11,13,15 <- src row r+8 words 8..11

dst row r+8:
  words 0,2,4,6    <- src row r   words 4..7
  words 1,3,5,7    <- src row r+8 words 4..7
  words 8,10,12,14 <- src row r   words 12..15
  words 9,11,13,15 <- src row r+8 words 12..15
```

No int4 nibble bit manipulation is done in the offline INT4 path. The kernel
moves whole `uint16_t` words. Runtime conversion later reinterprets 32-bit
chunks as `int4x8` and uses `psx_cvt_lut_prmt_int4x8_to_fp8x8()`.

## FP4 x BF16 Mapping

Function:

```text
interleave_fp4xbf16_Hopper_kernel_combine_triton
```

Data view:

```text
uint8 byte = 2 fp4 values
64 K-elements = 32 bytes per row
```

For one `partition_id` and one `lane_id` in `0..15`:

```cpp
row_id     = block_id / 8 * 16 + block_id % 8;
row_pair   = row_id + 8;

mma_id     = lane_id / 4;
dst_row    = row_id + (mma_id % 2) * 8;

src_col    = partition_id * 32 + mma_id * 8 + lane_id % 4; // byte column
src0       = src(row_id,   src_col);
src1       = src(row_pair, src_col);
src2       = src(row_id,   src_col + 4);
src3       = src(row_pair, src_col + 4);

dst_col    = partition_id * 32 + (lane_id / 8) * 16 + (lane_id % 4) * 4;
dst(dst_row, dst_col..dst_col+3) = bit_interleave(src0, src1, src2, src3);
```

This path combines two operations:

1. Row/K lane permutation similar in spirit to the INT4 path.
2. FP4 bit-field spreading for the BF16 converter.

The runtime converter `psx_cvt_triton_fp4x8_to_bf16x8_interleaved()` expects a
32-bit `fp4x8` input whose bits are already arranged so masks and shifts can
form BF16 sign/exponent/mantissa fields cheaply. The offline kernel therefore
does not just move nibbles; it constructs a special interleaved 32-bit layout
from eight FP4 values.

The comment in `mixed_input_utils.hpp` describes the expected bit placement as
an interleaved version of four consecutive FP4 values. The converter then uses
masks such as `0x81C081C0`, shifts, `lop3`, and `hmul2` to materialize BF16x8.

## Why This Matches The Mainloop

The mixed-input mainloop uses GMMA with the quantized/scaled operand on the
register-sourced A side:

```text
TMA global -> shared memory -> LDSM -> register fragment -> convert -> GMMA A
```

Key runtime pieces:

- `SmemCopyAtomA_LDSM = Copy_Atom<SM75_U32x4_LDSM_N, ElementB>`
- `sA_LDSM = recast<ElementB>(sA)`
- LDSM loads a register fragment with the same byte width as the converted
  MMA element.
- The code then reinterprets that register fragment back to the real 4-bit
  type:

```cpp
ptr = recast_ptr<RealSwappedElementA>(tCrA_load_LDSM.data());
tCrA_load_4b_packed = make_tensor(ptr, ... doubled/expanded K shape ...);
Utils::convert_A_kblock(tCrA_load_4b_packed, tCrA_mma, k_block);
```

For INT4 x FP8, `convert_A_kblock()` groups the loaded 4-bit values into
32-bit `int4x8` registers and calls the LUT/PRMT converter.

For FP4 x BF16, `convert_A_kblock()` groups the loaded 4-bit values into
32-bit `fp4x8` registers and calls the interleaved FP4-to-BF16 converter.

The offline reorder is what makes each register group contain the values and
bit fields in the order these converters expect, without doing expensive
cross-lane or shared-memory rearrangement inside the timed GEMM mainloop.

## Practical Constraints

Treat these as required for the current reorder kernels:

- `N` multiple of 16.
- `K` multiple of 64.
- Source B is row-major logical `[N, K]` packed 4-bit.
- Destination buffer has the same byte size as source.
- Reorder must complete before `ptr_B` is copied to or consumed by GEMM.
- The original non-reordered B must be kept if using the current reference
  verifier.

Current gaps:

- The example does not check `N % 16 == 0`.
- The option parser checks only a weaker K alignment condition.
- The wrappers do not report CUDA errors after the interleave launch.
- The kernels are written as GPU preprocessing kernels, not CPU offline tools.
  They are offline relative to GEMM timing, but still run on the GPU.

## Optimization Notes

The reorder is outside the timed `grouped_mixed_dtype_profiling()` loop, so it
does not explain steady-state GEMM runtime. It still matters for end-to-end
serving if weights are reloaded frequently.

Potential future cleanup:

- Add explicit shape validation before launching the reorder kernels.
- Add CUDA error checks after reorder launches.
- Split INT4 and FP4 reorder docs/tests because FP4 includes bit-level packing
  while INT4 is mostly word permutation.
- Build a tiny deterministic reorder test that labels source coordinates and
  checks destination coordinates for one `16 x 64` tile.
- Consider precomputing reordered weights on CPU or during model conversion if
  model load time or GPU memory bandwidth during initialization becomes an
  issue.
