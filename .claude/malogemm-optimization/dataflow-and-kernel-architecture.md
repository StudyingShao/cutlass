# Dataflow And Kernel Architecture

Date: 2026-05-14

## Why SwapAB Exists

Hopper WGMMA mixed-input instructions constrain the MMA tile shape. The grouped
MoE workload has fixed original `N` and `K`, but variable original `M` across
experts. This implementation uses SwapAB so the variable dimension maps to the
WGMMA N axis instead of the WGMMA M axis.

Original operation:

```text
D[M, N] = A[M, K] * B[N, K]^T
```

Kernel operation after SwapAB:

```text
D^T[N, M] = B[N, K] * A[M, K]^T
```

Practical consequences:

- The device problem shape is written as `(N, M, K)`.
- The mainloop receives B as its logical A-side operand and A as its logical
  B-side operand.
- C/D internal strides are created for `N x M`.
- Host validation and reporting restore the original `(M, N, K)` shape.
- Scale is associated with the quantized weight side, and the scale-MN
  dimension follows the original output-channel dimension.

Implementation nuance: in the CUTLASS collective, the internal `SwapAB` boolean
is `!IsATransformed`. This example usually puts the scaled/quantized weight as
the builder A operand (`tuple<ElementB, scale>`), so the internal boolean can be
false even though the host code is still using the high-level SwapAB-style
transposed operand and problem-shape arrangement.

## Builder Selection

`sm90_gmma_builder.inl` selects the array mixed-input collective when the
problem is pointer-array/grouped mixed input. It also chooses between post-scale
and pre-scale collectives:

```text
RealElementA == float_e2m1_t -> PreScale collective
otherwise                    -> PostScale collective
```

Therefore:

- MXFP4 x BF16 routes to
  `sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_prescale.hpp`.
- INT4 x FP8 routes to
  `sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp`.

The active INT4 x FP8 benchmark is post-scale.

## TMA Usage

A and B use descriptor-based TMA loads. The TMA descriptors encode the global
tensor shape, byte strides, swizzle, and shared-memory box shape. The
cooperative schedule uses multicast masks derived from the cluster coordinates:

- A is multicast along the cluster N axis.
- B is multicast along the cluster M axis.

Scale and zero-point tensors do not use TMA descriptors in the current mixed
input collectives. They are loaded with descriptor-less `SM90_BULK_COPY_G2S`.
That avoids 128-byte descriptor overhead for small scale/zero tiles.

The pipeline barrier transaction byte count includes A, B, and the extra
scale/zero bulk-copy bytes.

## Post-Scale Mainloop

The current INT4 x FP8 path uses post-scale accumulation:

1. Load A/B tiles with TMA and scale with bulk copy.
2. Convert packed 4-bit weight fragments into MMA fragments.
3. Run WGMMA for a scale group chunk into an intermediate accumulator.
4. Apply groupwise scale to the previous chunk while the next chunk is in
   flight.
5. Accumulate scaled results into the final accumulator.

For INT4 x FP8:

```text
ScalingGroupSize = 128
WGMMA K step      = 32
NumMMAsPerChunk   = 4
TileK             = 512
NumChunksPerTileK = 4
```

For MXFP4 x BF16:

```text
ScalingGroupSize = 32
WGMMA K step      = 16
NumMMAsPerChunk   = 2
TileK             = 128
NumChunksPerTileK = 4
```

The overlap uses `warpgroup_wait<1>()` for previous chunks and `warpgroup_wait<0>()`
for the final chunk. The final chunk's scale application cannot be overlapped.

## Pre-Scale Path

The pre-scale collective exists and is active for MXFP4 x BF16 through the
builder. It contains additional TMA descriptor pre-baking infrastructure:

- A device helper can replace TMA descriptor addresses, dimensions, and strides
  in global memory.
- `initialize_workspace()` launches a small `test_kernel` that creates one
  fully baked A descriptor and one fully baked B descriptor per group.
- The main kernel can then update descriptor pointers by group index instead of
  rewriting descriptor fields on each group switch.
- `#define ORIGINAL_TMA` is commented out, so the pre-baked path is the intended
  current path in that file.

Important distinction: this pre-baking machinery is in the pre-scale file.
The active INT4 x FP8 post-scale path allocates A/B descriptors per SM and still
uses the normal descriptor update model.

## Scheduler

The cooperative grouped kernel uses `PersistentTileSchedulerSm90Group`.

The grid is persistent and grid-stride based. Each CTA starts from a linear work
index derived from `blockIdx`, then advances by the total physical grid size.
When the linear index crosses a group's tile count, the scheduler moves to the
next group and recomputes per-group tile counts.

Current optimization:

- `problem_blocks_m_fixed` is cached in `GroupInfo`.
- This relies on SwapAB and fixed original `N`.
- After SwapAB, shape<0> is original `N`, so `tiles_m` is constant across all
  groups in the intended MoE workload.
- Only `tiles_n`, which corresponds to variable original `M`, is recomputed on
  group switches.

This is valid only when original `N` is fixed across groups. If original `N`
varies, this optimization can mis-schedule work.

The existing scheduler does not implement work stealing. Prior notes propose it,
but the later optimization log concludes scheduler ROI is low after the fixed-N
optimization.

## Cooperative Kernel Structure

`sm90_gemm_array_tma_warpspecialized_cooperative.hpp` specializes the grouped
ptr-array cooperative kernel:

- 1 producer warpgroup.
- 2 MMA warp groups.
- `LoadRegisterRequirement = 40`.
- `MmaRegisterRequirement = 232`.
- Mainloop producer, epilogue load producer, and epilogue store pipeline are
  separated.
- `scheduler.fetch_next_work()` is used around producer/consumer transitions.

The cooperative kernel requires `size(TiledMma{}) == 256`, tile M at least 128,
and `NumMmaWarpGroups == 2`.

## Current Invariants And Hazards

Treat these as hard constraints before starting optimization:

- Build for SM90a/Hopper and CUDA 12.3 or newer.
- Original `N` and `K` should be fixed across groups.
- Original `M` may vary across groups for MoE.
- Zero-M groups are not currently safe in the observed CUTLASS benchmark path.
- INT4 x FP8 default `TileShapeK` is 512. Current scale allocation requires
  `K >= 512` and effectively assumes `K` is a multiple of 512.
- `grouped_mixed_dtype_utils.hpp` only checks `K` against 128-bit sub-byte
  alignment; this is weaker than the current INT4 x FP8 tile/scale requirement.
- Interleave kernels assume regular row and column divisibility. Practically,
  use even `N` and `K` multiples of 64 for the current B interleave kernels.
- `args_from_options()` passes no host problem-shape pointer to the device
  arguments, so some host-side `can_implement()` shape checks are bypassed.
- `hw_info.device_id = 0` relies on `CUDA_VISIBLE_DEVICES` mapping the intended
  physical GPU to logical device 0.
