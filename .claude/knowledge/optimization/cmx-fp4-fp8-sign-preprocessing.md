# CMX FP4 to FP8 Sign Preprocessing

**Last Updated**: 2026-05-26
**Status**: Implemented for CMX `mxfp4 x fp8` and `mxfp4 x mxfp8`
**Related cache**:
`cutlass_mixed_gemm/.claude/knowledge/optimization/cmx-best-config-cache.md`

## Background

The CMX FP4(E2M1) to FP8(E4M3) converter used two PRMT instructions only to
gather FP4 sign bits:

```text
l4b_sign = prmt(hb_sign, lb_sign, 0x5140)
h4b_sign = prmt(hb_sign, lb_sign, 0x7362)
```

The FP4 EM selector bits are only the low 3 bits of every nibble. Since the
mainloop masks sign bits before LUT indexing, the offline weight layout can
reuse each nibble's sign-bit slot to carry a better packed sign layout.

## Implementation

The offline W4A8 interleave path now has an optional FP4 sign preprocessing
mode:

```text
examples/69_hopper_mixed_dtype_grouped_gemm/BF16_MXFP4_Test.h
  preprocess_fp4x8_signs_for_fp8()
  interleave_w4a8_Hopper<T, true>()
```

For each 32-bit FP4x8 word:

```text
low 3 EM bits: preserved in place
signs 0..3:    moved to byte bit7 positions
signs 4..7:    moved to bit3 of each byte
```

The mainloop converter variant is:

```text
include/cutlass/detail/collective/mixed_input_utils.hpp
  psx_cvt_lut_prmt_fp4x8_to_fp8x8_preprocessed_signs()
```

Runtime sign extraction becomes:

```text
fp8x4_raw[0] = LUT(fp4x8 & 0x77777777) | (fp4x8 & 0x80808080)
fp8x4_raw[1] = LUT((fp4x8 & 0x77777777) >> 16) |
               ((fp4x8 << 4) & 0x80808080)
```

This removes the two sign-gather PRMT instructions. The remaining PRMT
instructions are the two LUT lookups for FP4 EM to positive FP8 E4M3 bytes.

The optimized converter is selected only when:

```text
CUTLASS_MIXED_GEMM_FP4_FP8_PREPROCESSED_SIGNS=1
```

That macro is enabled for the CMX `mxfp4 x fp8` and `mxfp4 x mxfp8` CMake
targets. INT4 x FP8 and MXFP4 x BF16 are unchanged.

## Validation

Standalone bit-level converter test:

```bash
cmake --build build -j --target test_fp4_to_fp8_convert
CUDA_VISIBLE_DEVICES=3 ./build/fp4_to_fp8_study/test_fp4_to_fp8_convert
```

Result:

```text
FP4 -> FP8 converter test PASSED: 404 cases
```

SASS spot check:

```bash
cmake --build build -j --target fp4_to_fp8_sass
```

The optimized standalone path shows two PRMT instructions for LUT lookup only.
The old path still shows four PRMT instructions: two LUT PRMTs plus the two
sign-gather PRMTs.

Detailed SASS check from 2026-05-26:

```bash
cmake --build build -j --target test_fp4_to_fp8_convert
CUDA_VISIBLE_DEVICES=3 ./build/fp4_to_fp8_study/test_fp4_to_fp8_convert
cuobjdump --dump-sass ./build/fp4_to_fp8_study/test_fp4_to_fp8_convert \
  > /tmp/test_fp4_to_fp8_convert.sass
```

```text
kernel                       non-NOP static   PRMT   sign PRMT   LOP3
convert_kernel                         32      4          2        5
convert_preprocessed_kernel            28      2          0        3
```

For just the 8xFP4 to 8xFP8 conversion compute sequence, excluding input load,
output store, bounds check, and index/address arithmetic:

```text
kernel                       convert compute   PRMT   sign PRMT   LOP3
convert_kernel                            14      4          2        5
convert_preprocessed_kernel               10      2          0        3
```

If the shared LUT constant materialization is also excluded, the pure
data-transform sequence is 11 instructions before preprocessing and 7
instructions after preprocessing.

The removed sign PRMT selectors are `0x5140` and `0x7362`. The optimized path
still has the two LUT PRMTs and folds sign merge into `LOP3.LUT` using
`0x80808080`.

CMX production SASS spot check:

```bash
cmake --build cutlass_mixed_gemm/build -j --target \
  69_hopper_mxfp4_fp8_grouped_gemm_fc1_m8_best
cuobjdump --dump-sass \
  cutlass_mixed_gemm/build/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_mxfp4_fp8_grouped_gemm_fc1_m8_best \
  > /tmp/cmx_mxfp4_fp8_fc1_m8_best.sass
```

For the main GEMM device kernel only:

```text
main_gemm_instr             4096
PRMT                         201
sign_PRMT_5140_7362            0
em_mask_77777777              96
sign_merge_80808080          192
LDC_c3                         2
fp4_lut_PRMT_R161_R162       192
fp4_lut_literal                0
```

Whole-binary `0x5140`/`0x7362` hits are from helper/reference kernels, including
`fp4_to_fp8_kernel`, not from the main CMX GEMM kernel. The hot GEMM converter
uses two `LDC c[0x3]` LUT words as GPR operands to `PRMT`, which preserves the
constant-memory LUT codegen fix documented in
`../../../../.claude/notes/prmt-lut-constant-codegen.md`.

CMX fixed-target build:

```bash
cmake -S cutlass_mixed_gemm -B cutlass_mixed_gemm/build
cmake --build cutlass_mixed_gemm/build -j --target \
  69_hopper_mxfp4_mxfp8_grouped_gemm_fc1_m8_best \
  69_hopper_mxfp4_mxfp8_grouped_gemm_fc2_m8_best
```

FC1 exact-shape correctness smoke:

```bash
CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
  ./cutlass_mixed_gemm/build/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_mxfp4_mxfp8_grouped_gemm_fc1_m8_best \
  --m=8 --n=1024 --k=4096 --groups=128 --iterations=0 --warmup=0
```

The command returned success. The harness still prints non-fatal FP8 reference
differences and P99/P98/P95 error counts, as it did for approximate MXFP8
comparison paths.

## Performance

All performance runs used GPU 3, fixed best-config targets, and no sweep:

```bash
CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
  ./cutlass_mixed_gemm/build/examples/69_hopper_mixed_dtype_grouped_gemm/<target> \
  --m=8 --n=<N> --k=<K> --groups=128 \
  --iterations=300 --warmup=50 --compare=false
```

```text
Shape                    Baseline us   New runs us          Best delta
FC1 8x1024x4096 x128     225.690       207.509, 205.697     -19.993 us / -8.86%
FC2 8x4096x512  x128     219.341       210.723, 212.503     -8.618 us / -3.93%
```

For `mxfp4 x fp8`, the same preprocessed converter path was checked on
2026-05-26 with dedicated fixed targets and no sweep:

```text
Shape                    Source sweep us   New us    Delta vs sweep
FC1 8x1024x4096 x128     194.890           182.091   -12.799 us / -6.57%
FC2 8x4096x512  x128     203.388           195.765   -7.623 us / -3.75%
```

The cached best config did not change:

```text
FC1 m=8: k512, Coop C1x1x1, Tile 128x16x512, Stages=5
FC2 m=8: k128, Coop C1x1x1, Tile 128x16x128, Stages=19
```

## Current Decision

Keep the sign-preprocessed offline weight layout for FP4 x FP8/MXFP8 CMX
targets. It is a local instruction-count reduction in the mainloop converter,
does not change FP4 EM selector semantics, and improves both m=8 focus shapes
without requiring a new config sweep.

Follow-up profiling, if needed, should use NCU with explicit:

```bash
--clock-control none --cache-control all
```
