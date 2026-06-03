# Implementation Map

Date: 2026-05-14

## Entry Points

Primary entry:

```text
examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_fp8_grouped_gemm.cu
```

Supporting headers and generated-profiler path:

```text
examples/69_hopper_mixed_dtype_grouped_gemm/kernel_profiler_shared.hpp
examples/69_hopper_mixed_dtype_grouped_gemm/kernel_profiler.h
examples/69_hopper_mixed_dtype_grouped_gemm/profiler_part*.cu
examples/69_hopper_mixed_dtype_grouped_gemm/grouped_mixed_dtype_utils.hpp
examples/69_hopper_mixed_dtype_grouped_gemm/host_validation.hpp
examples/69_hopper_mixed_dtype_grouped_gemm/BF16_MXFP4_Test.h
```

Related examples:

```text
examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_bf16_grouped_gemm.cu
examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_mixed_dtype_grouped_gemm.cu
```

Core CUTLASS hooks:

```text
include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl
include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp
include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_prescale.hpp
include/cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_cooperative.hpp
include/cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp
```

## Build Target

Target name:

```bash
cmake --build build -j --target 69_hopper_int4_fp8_grouped_gemm
```

Binary:

```text
build/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_fp8_grouped_gemm
```

Local build note: this checkout needed `examples/CMakeLists.txt` to include
`${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}` for the example utility headers. That patch
is inside `cutlass_mixed_gemm` only and does not connect this tree to MaloGEMM.

## Dtype Selection

The active dtype mode is chosen by editing `kernel_profiler_shared.hpp`.

Active mode:

```cpp
using MmaType = cutlass::float_e4m3_t;
using QuantType = cutlass::int4b_t;
#define GROUP_SIZE 128
using ElementScale = cutlass::bfloat16_t;
inline constexpr int TileShapeM = 128;
inline constexpr int TileShapeN = 16;
inline constexpr int TileShapeK = 8192 / TileShapeN; // 512
```

Commented alternate mode:

```cpp
using MmaType = cutlass::bfloat16_t;
using QuantType = cutlass::float_e2m1_t;
#define GROUP_SIZE 32
using ElementScale = cutlass::float_ue8m0_t;
inline constexpr int TileShapeM = 128;
inline constexpr int TileShapeN = 32;
inline constexpr int TileShapeK = 128;
```

`ElementScalePacked` is `Array<ElementScale, TileShapeK / GROUP_SIZE>`.
For INT4 x FP8 this is 4 BF16 scale values per K tile.

## Host Flow

Normal run path:

1. `main()` checks CUDA version and SM90 capability.
2. `Options::parse()` reads CLI arguments and forces scale-only mode.
3. `run<GemmScaleOnly>(options, false)` is selected unless `--explore` is used.
4. `allocate()` computes per-group offsets and stride arrays.
5. `initialize()` fills inputs, interleaves weights, packs scales, creates the
   reference output, and writes swapped problem shapes to the device.
6. `args_from_options()` creates grouped GEMM arguments.
7. `gemm.can_implement()`, `gemm.initialize()`, and `gemm.run()` are called.
8. `verify()` compares device output against `block_ref_D`.
9. `grouped_mixed_dtype_profiling()` runs timed iterations using CUDA events.

Setup, random initialization, weight interleave, scale packing, and reference
generation are not included in the timed loop.

## Runtime Arguments

Useful flags:

```text
--groups=<int>
--m=<int>
--n=<int>
--k=<int>
--expert_counts=<path>
--benchmark=<path>
--iterations=<int>
--warmup=<int>
--alpha=<float>
--beta=<float>
--swizzle=<1|2|4|8>
--compare=<true|false>
--explore=<true|false>
```

`--expert_counts` reads per-group original M values. The file may contain
whitespace-separated integers or log-style lines with `=`; only the right-hand
side is parsed for such lines. The global `--n` and `--k` are used for every
group, and `--groups` is overridden by the number of parsed counts.

## Argument Layout

The default grouped GEMM arguments are:

```cpp
{
  GemmUniversalMode::kGrouped,
  {groups, problem_sizes_device, nullptr},
  {ptr_B, stride_B, ptr_A, stride_A, ptr_scale_packed, stride_S, GROUP_SIZE},
  {fusion_args, ptr_C, stride_C, ptr_D, stride_D},
  hw_info
}
```

The operand order is intentionally `B, A` because the example applies a
SwapAB-style arrangement at the host/type level. Device problem shapes are also
transposed from original `(M, N, K)` to `(N, M, K)` before launch. The host copy
is restored afterward for validation and reporting.

`args_from_options()` currently sets `hw_info.device_id = 0` and queries SM
count for logical device 0. This is correct when `CUDA_VISIBLE_DEVICES=3`
exposes physical GPU 3 as logical device 0, but it is fragile if multiple GPUs
are visible.

## Allocation And Strides

Per group, original shapes are `(M, N, K)`.

- A: `M x K`, row-major, stored contiguously.
- B: logical `N x K`, 4-bit packed. Offsets are byte offsets because the type is
  sub-byte.
- C/D internal: shaped as swapped `N x M` for SwapAB.
- Reference C/D: shaped as original `M x N`.
- Scale: `N x (K / TileShapeK)`.

The current INT4 x FP8 allocation uses integer division:

```cpp
scale_k = K / TileShapeK;
```

This makes `K < TileShapeK` allocate zero scale elements and is not valid for
the current default INT4 x FP8 path.

## Weight Interleave

The entry supports two interleave paths:

- MXFP4 x BF16:
  `interleave_fp4xbf16_Hopper()`
- INT4 x FP8:
  `interleave_int4xfp8_Hopper()`

Both use kernels in `BF16_MXFP4_Test.h` with `<<<1024, dim3(16, 32)>>>`.
They assume regular Hopper WGMMA-friendly packing. For practical use, treat
the B matrix as requiring even row groups and K aligned to the interleave tile
width, currently multiples of 64 columns in the interleave loops.

## Verification

`groupwise_verify()` creates a device reference from original A, original
non-interleaved B, and packed scales. It uses the original shape order.

`verify()` then compares each group. It skips zero-M groups in the explicit
`BlockCompareRelativelyEqual()` loop, but `compare_device()` still uses the
swapped device problem-shape array and walks the flattened output.

Observed limitation: a target192 expert-count file with zero-M groups triggered
an illegal memory access in the CUTLASS FC1 run. Until fixed, treat zero-M
groups as unsupported for the CUTLASS benchmark path even though validation
contains partial zero-M handling.

## Profiler Path

`kernel_profiler_shared.hpp` has a disabled `#define PROFILE`. When enabled,
`--explore=true` uses `best_config_finder()` and the `profiler_part*.cu`
translation units to compile many `(schedule, cluster, tile)` choices.

Covered search families include:

- Pingpong schedules for smaller M tiles.
- Cooperative schedules for `128 x N x K` tiles.
- Cluster shapes such as `1 x 1 x 1`, `2 x 1 x 1`, `1 x 2 x 1`, and `2 x 2 x 1`.

The normal non-profiler default remains `128 x 16 x 512`, cluster `2 x 1 x 1`,
cooperative ptr-array schedule.
