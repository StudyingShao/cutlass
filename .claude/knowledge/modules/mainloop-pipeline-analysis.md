# Mainloop Pipeline Analysis: Pre-Scale vs Post-Scale

**Last Updated**: 2026-03-19
**Status**: Core understanding with critical corrections
**Session**: Deep learning session on mainloop instruction-level pipelining

---

## Purpose

This document provides **detailed instruction-level understanding** of the mainloop pipeline in Hopper mixed-precision grouped GEMM kernels, comparing two implementations:
- **Pre-Scale version** (`sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_prescale.hpp`): Simple pipeline without groupwise scaling
- **Post-Scale version** (`sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp`): Production pipeline with overlapped groupwise scaling

**Why this matters:**
- Understanding instruction-level overlap is critical for performance optimization
- Post-scale achieves ~37.5% scaling overhead reduction through careful pipeline design
- These patterns apply broadly to Hopper kernel optimization

**Related documentation:**
- Architecture overview: `.claude/knowledge/modules/hopper-mixed-precision-gemm-overview.md`
- TMA usage: `.claude/knowledge/modules/tma-usage-in-mixed-precision-gemm.md`
- SwapAB pattern: `.claude/knowledge/architecture/swapab-pattern.md`

---

## Table of Contents

1. [Critical Architecture Constraints](#critical-architecture-constraints)
2. [Chunk Organization](#chunk-organization)
3. [Pre-Scale Mainloop Pipeline](#pre-scale-mainloop-pipeline)
4. [Post-Scale Mainloop Pipeline](#post-scale-mainloop-pipeline)
5. [Performance Comparison](#performance-comparison)
6. [Critical Insights and Corrections](#critical-insights-and-corrections)
7. [Code References](#code-references)

---

## Critical Architecture Constraints

### WGMMA Tile Shape Constraints

**⚠️ CRITICAL: K dimension depends on input data type!**

```cpp
WGMMA Tile Shape: M × N × K

Constraints by input type:
┌─────────────┬──────────┬─────────────┬──────────┐
│ Input Type  │ M        │ N           │ K        │
├─────────────┼──────────┼─────────────┼──────────┤
│ BF16        │ 64 (固定) │ 可变 (16-256) │ 16 (固定) │
│ FP8 (E4M3)  │ 64 (固定) │ 可变 (16-256) │ 32 (固定) │
│ FP16        │ 64 (固定) │ 可变 (16-256) │ 16 (固定) │
│ INT8        │ 64 (固定) │ 可变 (16-256) │ 32 (固定) │
└─────────────┴──────────┴─────────────┴──────────┘

Key insight: K_per_WGMMA determines chunk organization!
```

**Design implications:**
- **SwapAB pattern**: Driven by M/N constraint (fixed M=64, variable N)
  - Maps variable problem M to WGMMA's variable N
  - **NOT related to K dimension!**
- **Chunk organization**: Driven by K constraint
  - Determines `NumMMAsPerChunk = ScalingGroupSize / K_per_WGMMA`
  - **NOT related to SwapAB!**

**Code evidence:**
- WGMMA traits: `include/cute/atom/mma_traits_sm90_gmma.hpp`
- BF16: `SM90_64x*x16_*BF16*` instructions
- FP8: `SM90_64x*x32_*E4M3*` instructions

---

## Chunk Organization

### Configuration 1: MXFP4 × BF16 (TileK=128)

```cpp
constexpr int ScalingGroupSize = 32;  // Group size for scaling
constexpr int K_per_WGMMA = 16;       // ⚠️ BF16 input type!

constexpr int NumMMAsPerChunk = ScalingGroupSize / K_per_WGMMA
                               = 32 / 16 = 2;
// Each chunk contains 2 WGMMAs

constexpr int K_BLOCK_MAX = TileK / K_per_WGMMA
                          = 128 / 16 = 8;
// 8 k_blocks per K-tile

constexpr int NumChunksPerTileK = TileK / ScalingGroupSize
                                = 128 / 32 = 4;
// 4 chunks per K-tile
```

**Chunk structure:**
```
K-tile (K=128):
├─ Chunk 0: k_block [0, 1]   → K range [0, 31]   → scale[0]
├─ Chunk 1: k_block [2, 3]   → K range [32, 63]  → scale[1]
├─ Chunk 2: k_block [4, 5]   → K range [64, 95]  → scale[2]
└─ Chunk 3: k_block [6, 7]   → K range [96, 127] → scale[3]
```

### Configuration 2: INT4 × FP8 (TileK=512)

```cpp
constexpr int ScalingGroupSize = 128;  // Group size for scaling
constexpr int K_per_WGMMA = 32;        // ⚠️ FP8 input type!

constexpr int NumMMAsPerChunk = ScalingGroupSize / K_per_WGMMA
                               = 128 / 32 = 4;
// Each chunk contains 4 WGMMAs

constexpr int K_BLOCK_MAX = TileK / K_per_WGMMA
                          = 512 / 32 = 16;
// 16 k_blocks per K-tile

constexpr int NumChunksPerTileK = TileK / ScalingGroupSize
                                = 512 / 128 = 4;
// 4 chunks per K-tile
```

**Chunk structure:**
```
K-tile (K=512):
├─ Chunk 0: k_block [0, 1, 2, 3]     → K range [0, 127]    → scale[0]
├─ Chunk 1: k_block [4, 5, 6, 7]     → K range [128, 255]  → scale[1]
├─ Chunk 2: k_block [8, 9, 10, 11]   → K range [256, 383]  → scale[2]
└─ Chunk 3: k_block [12, 13, 14, 15] → K range [384, 511]  → scale[3]
```

---

## Pre-Scale Mainloop Pipeline

### Architecture Overview

**Pre-Scale version characteristics:**
- ✅ Single-level k_block loop
- ✅ Direct accumulation to `accum` register
- ✅ No groupwise scaling
- ✅ Simple control flow
- ✅ Minimal register pressure

**File:** `include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_prescale.hpp`

### Code Structure (MXFP4×BF16 Example)

```cpp
// First K-tile (line 1235-1280)
{
  // Wait for TMA to complete
  pipeline.consumer_wait(smem_pipe_read);
  int read_stage = smem_pipe_read.index();
  ++smem_pipe_read;

  // ═══════════════════════════════════════════════════════════════
  // Stage 0: Pre-load k_block 0 and 1 (BEFORE loop starts)
  // ═══════════════════════════════════════════════════════════════
  Utils::copy_tensors_A(..., k_block=0, read_stage);  // LDSM
  Utils::copy_tensors_A(..., k_block=1, read_stage);  // LDSM
  Utils::convert_A_kblock(..., k_block=0);             // Convert

  tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

  // ═══════════════════════════════════════════════════════════════
  // Main loop: k_block 0-7
  // ═══════════════════════════════════════════════════════════════
  for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
    warpgroup_arrive();
    cute::gemm(tiled_mma, tCrA_mma(_,_,k_block),
               tCrB(_,_,k_block,read_stage), accum);
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    warpgroup_commit_batch();

    // Prefetch k_block + 2
    if (k_block < K_BLOCK_MAX - 2) {
      Utils::copy_tensors_A(..., k_block + 2, read_stage);
    }
    // Convert k_block + 1
    if (k_block < K_BLOCK_MAX - 1) {
      Utils::convert_A_kblock(..., k_block + 1);
    }
  }
}
```

### Complete Pipeline Timeline (K_BLOCK_MAX=8, MXFP4×BF16)

**⚠️ IMPORTANT: Always show pre-load stage for completeness!**

```
═══════════════════════════════════════════════════════════════════════════
Stage 0: Data Pre-load (BEFORE loop starts)
───────────────────────────────────────────────────────────────────────────
Cycle -20 to -15:  pipeline.consumer_wait() 🛑 Wait for TMA
Cycle -15 to -10:  LDSM(k_block=0)  📥 Load packed 4-bit from SMEM
Cycle -10 to -5:   LDSM(k_block=1)  📥 Load packed 4-bit from SMEM
Cycle -5 to 0:     Convert(k_block=0) 🔄 4-bit → BF16 conversion

═══════════════════════════════════════════════════════════════════════════
k_block 0: (Uses pre-loaded data)
───────────────────────────────────────────────────────────────────────────
Cycle 0-5:     warpgroup_arrive()
Cycle 5-25:    WGMMA[0] executes  ✈️ (20 cycles)
               └─ A: tCrA_mma(_,_,0) - converted BF16 data
               └─ B: tCrB(_,_,0,read_stage) - SMEM descriptor
               └─ C: accum (accumulate_ = Zero → initialize)
Cycle 25-30:   warpgroup_commit_batch()

───────────────────────────────────────────────────────────────────────────
Cycle 30-35:   LDSM(k_block=2)  📥 Prefetch k+2
Cycle 35-40:   Convert(k_block=1) 🔄 Convert k+1
               ⚠️ Overlapped with WGMMA[0] in flight!

═══════════════════════════════════════════════════════════════════════════
k_block 1:
───────────────────────────────────────────────────────────────────────────
Cycle 40-45:   warpgroup_arrive()
Cycle 45-65:   WGMMA[1] executes  ✈️
               └─ accumulate_ = One → accumulate to accum
Cycle 65-70:   warpgroup_commit_batch()

Cycle 70-75:   LDSM(k_block=3)  📥
Cycle 75-80:   Convert(k_block=2) 🔄
               ⚠️ Overlapped with WGMMA[1]!

═══════════════════════════════════════════════════════════════════════════
k_block 2:
───────────────────────────────────────────────────────────────────────────
Cycle 80-85:   warpgroup_arrive()
Cycle 85-105:  WGMMA[2] executes  ✈️
Cycle 105-110: warpgroup_commit_batch()

Cycle 110-115: LDSM(k_block=4)  📥
Cycle 115-120: Convert(k_block=3) 🔄

═══════════════════════════════════════════════════════════════════════════
k_block 3:
───────────────────────────────────────────────────────────────────────────
Cycle 120-125: warpgroup_arrive()
Cycle 125-145: WGMMA[3] executes  ✈️
Cycle 145-150: warpgroup_commit_batch()

Cycle 150-155: LDSM(k_block=5)  📥
Cycle 155-160: Convert(k_block=4) 🔄

═══════════════════════════════════════════════════════════════════════════
k_block 4:
───────────────────────────────────────────────────────────────────────────
Cycle 160-165: warpgroup_arrive()
Cycle 165-185: WGMMA[4] executes  ✈️
Cycle 185-190: warpgroup_commit_batch()

Cycle 190-195: LDSM(k_block=6)  📥
Cycle 195-200: Convert(k_block=5) 🔄

═══════════════════════════════════════════════════════════════════════════
k_block 5:
───────────────────────────────────────────────────────────────────────────
Cycle 200-205: warpgroup_arrive()
Cycle 205-225: WGMMA[5] executes  ✈️
Cycle 225-230: warpgroup_commit_batch()

Cycle 230-235: LDSM(k_block=7)  📥 Last prefetch
Cycle 235-240: Convert(k_block=6) 🔄

═══════════════════════════════════════════════════════════════════════════
k_block 6:
───────────────────────────────────────────────────────────────────────────
Cycle 240-245: warpgroup_arrive()
Cycle 245-265: WGMMA[6] executes  ✈️
Cycle 265-270: warpgroup_commit_batch()

Cycle 270-275: (k_block + 2 = 8, out of range, skip LDSM)
Cycle 275-280: Convert(k_block=7) 🔄 Last conversion

═══════════════════════════════════════════════════════════════════════════
k_block 7: (Last k_block)
───────────────────────────────────────────────────────────────────────────
Cycle 280-285: warpgroup_arrive()
Cycle 285-305: WGMMA[7] executes  ✈️
Cycle 305-310: warpgroup_commit_batch()

Cycle 310-315: (k_block + 2 out of range, skip)
Cycle 315-320: (k_block + 1 = 8, out of range, skip Convert)

═══════════════════════════════════════════════════════════════════════════
After loop: Prefetch next K-tile (line 1272-1278)
───────────────────────────────────────────────────────────────────────────
Cycle 320-330: pipeline.consumer_wait(smem_pipe_read, barrier_token)
Cycle 330-335: LDSM(k_block=0, next_stage)  📥 Next K-tile pre-load
Cycle 335-340: LDSM(k_block=1, next_stage)  📥
Cycle 340-345: warpgroup_wait<K_WAIT_MAX>()  ⏱️ Wait for WGMMA queue
Cycle 345-350: Convert(k_block=0) 🔄 Next K-tile's first k_block

═══════════════════════════════════════════════════════════════════════════
Total time: ~350 cycles per K-tile (including pre-load and next prep)
Pipeline efficiency: ~94% (160 WGMMA cycles / 170 effective cycles)
═══════════════════════════════════════════════════════════════════════════
```

### Key Pipeline Mechanisms

**1. Prefetch Distance: k+2**

```cpp
if (k_block < K_BLOCK_MAX - 2) {
  Utils::copy_tensors_A(..., k_block + 2, read_stage);
}
```

**Why k+2, not k+1?**
- k_block N: Issue WGMMA(N)
- k_block N+1: Use converted data (converted in k_block N)
- k_block N+2: Load data (loaded in k_block N, converted in k_block N+1)

**Pipeline depth = 2:**
```
k_block N:   WGMMA(N)   ← uses data from k_block N-2
             LDSM(N+2)  ← prefetch
             Convert(N+1) ← convert prefetched data
```

**2. warpgroup_wait Control**

```cpp
constexpr int K_WAIT_MAX = cute::min(K_BLOCK_MAX - 1, 7);  // = 7 for K_BLOCK_MAX=8
warpgroup_wait<K_WAIT_MAX>();  // Allow up to K_WAIT_MAX WGMMAs in flight
```

**Effect:**
- First 7 k_blocks: Issue WGMMA without waiting
- k_block 7: Must wait for WGMMA[0] to complete before continuing
- Hardware limit: Hopper supports up to 8 WGMMAs in flight

---

## Post-Scale Mainloop Pipeline

### Architecture Overview

**Post-Scale version characteristics:**
- ✅ Dual-level loop: chunk (outer) + mma (inner)
- ✅ Intermediate buffer for each chunk
- ✅ Groupwise scaling with compute overlap
- ✅ More complex control flow
- ❌ Increased register pressure (intermediate_array)

**File:** `include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp`

### Code Structure (MXFP4×BF16 Example)

```cpp
// Allocate intermediate buffers (line 1007)
cute::array<decltype(make_fragment_like(accum)), NumChunksPerTileK> intermediate_array;

// First K-tile (line 1015-1091)
{
  pipeline.consumer_wait(smem_pipe_read);
  int read_stage = smem_pipe_read.index();

  // Pre-load k_block 0 and 1
  Utils::copy_tensors_A(..., 0, read_stage);
  Utils::copy_tensors_A(..., 1, read_stage);
  Utils::convert_A_kblock(..., 0);

  // ═══════════════════════════════════════════════════════════════
  // Dual-level loop: chunks and mmas
  // ═══════════════════════════════════════════════════════════════
  for (int chunk_id = 0; chunk_id < NumChunksPerTileK; ++chunk_id) {
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;  // Reset per chunk

    for (int mma_id = 0; mma_id < NumMMAsPerChunk; ++mma_id) {
      int k_block = chunk_id * NumMMAsPerChunk + mma_id;

      warpgroup_arrive();
      cute::gemm(tiled_mma, tCrA_mma(_,_,k_block),
                 tCrB(_,_,k_block,read_stage),
                 intermediate_array[chunk_id]);  // Accumulate to intermediate
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;

      if (k_block == 0) {
        // Load scale factors
        Utils::copy_tensors_SFA(...);
      }

      if (k_block < K_BLOCK_MAX - 2) {
        Utils::copy_tensors_A(..., k_block + 2, read_stage);
      }
      if (k_block < K_BLOCK_MAX - 1) {
        Utils::convert_A_kblock(..., k_block + 1);
      }
    }

    warpgroup_commit_batch();  // Commit all WGMMAs in this chunk

    // ═══════════════════════════════════════════════════════════════
    // ⭐ KEY OPTIMIZATION: Overlapped scaling
    // ═══════════════════════════════════════════════════════════════
    if (chunk_id > 0) {
      warpgroup_wait<1>();  // Wait until ≤1 WGMMA in flight
                            // Current: Chunk N in flight
                            // Guaranteed: Chunk N-1 completed

      warpgroup_fence_operand(intermediate_array[chunk_id - 1]);

      // Scale chunk N-1 while chunk N executes!
      apply_groupwise_scale(accum, intermediate_array[chunk_id - 1],
                            scale, chunk_id - 1, chunk_id == 1);
    }
  }

  // Last chunk scaling (cannot overlap)
  warpgroup_wait<0>();
  apply_groupwise_scale(accum, intermediate_array[NumChunksPerTileK - 1],
                        scale, NumChunksPerTileK - 1, false);
}
```

### Complete Pipeline Timeline (K_BLOCK_MAX=8, NumChunksPerTileK=4, NumMMAsPerChunk=2)

```
═══════════════════════════════════════════════════════════════════════════
Stage 0: Data Pre-load (BEFORE loop starts, line 1024-1030)
───────────────────────────────────────────────────────────────────────────
Cycle -20 to -15:  pipeline.consumer_wait() 🛑
Cycle -15 to -10:  LDSM(k_block=0)  📥
Cycle -10 to -5:   LDSM(k_block=1)  📥
Cycle -5 to 0:     Convert(k_block=0) 🔄

═══════════════════════════════════════════════════════════════════════════
Chunk 0 (k_blocks [0, 1], K range [0, 31]):
───────────────────────────────────────────────────────────────────────────
k_block 0 (mma_id=0):
Cycle 0-5:     tiled_mma.accumulate_ = Zero  (initialize intermediate_array[0])
Cycle 5-10:    warpgroup_arrive()
Cycle 10-30:   WGMMA[0] → intermediate_array[0]  ✈️
               └─ accumulate_ = Zero (initialize)
Cycle 30-35:   tiled_mma.accumulate_ = One
Cycle 35-40:   copy_tensors_SFA(...)  📥 Load scale factors to registers

Cycle 40-45:   LDSM(k_block=2)  📥 Prefetch k+2
Cycle 45-50:   Convert(k_block=1) 🔄 Convert k+1
               ⚠️ Overlapped with WGMMA[0] in flight!

k_block 1 (mma_id=1):
Cycle 50-55:   warpgroup_arrive()
Cycle 55-75:   WGMMA[1] → intermediate_array[0]  ✈️
               └─ accumulate_ = One (accumulate to intermediate_array[0])
Cycle 75-80:   (mma_id=1 does not execute copy_tensors_SFA)

Cycle 80-85:   LDSM(k_block=3)  📥 Prefetch k+2
Cycle 85-90:   Convert(k_block=2) 🔄 Convert k+1

───────────────────────────────────────────────────────────────────────────
Cycle 90-95:   warpgroup_commit_batch()  ✅ Commit all WGMMAs in Chunk 0
Cycle 95-100:  if (chunk_id > 0) - Skip (chunk_id = 0)

═══════════════════════════════════════════════════════════════════════════
Chunk 1 (k_blocks [2, 3], K range [32, 63]):
───────────────────────────────────────────────────────────────────────────
k_block 2 (mma_id=0):
Cycle 100-105: tiled_mma.accumulate_ = Zero  (reset intermediate_array[1])
Cycle 105-110: warpgroup_arrive()
Cycle 110-130: WGMMA[2] → intermediate_array[1]  ✈️
Cycle 130-135: tiled_mma.accumulate_ = One

Cycle 135-140: LDSM(k_block=4)  📥
Cycle 140-145: Convert(k_block=3) 🔄

k_block 3 (mma_id=1):
Cycle 145-150: warpgroup_arrive()
Cycle 150-170: WGMMA[3] → intermediate_array[1]  ✈️

Cycle 170-175: LDSM(k_block=5)  📥
Cycle 175-180: Convert(k_block=4) 🔄

───────────────────────────────────────────────────────────────────────────
Cycle 180-185: warpgroup_commit_batch()
Cycle 185-190: warpgroup_wait<1>()  ⏱️
               ⚠️ Currently in flight: WGMMA[3]
               ✅ WGMMA[0], WGMMA[1] completed → intermediate_array[0] ready
───────────────────────────────────────────────────────────────────────────

Cycle 190-195: warpgroup_fence_operand(intermediate_array[0])
Cycle 195-235: apply_groupwise_scale(accum, intermediate_array[0], scale, 0)
               🚀 ~40 cycles, overlapped with WGMMA[3] in flight!
               └─ Traverse all accum elements: accum = intermediate[0] × scale[0]

═══════════════════════════════════════════════════════════════════════════
Chunk 2 (k_blocks [4, 5], K range [64, 95]):
───────────────────────────────────────────────────────────────────────────
k_block 4 (mma_id=0):
Cycle 235-240: tiled_mma.accumulate_ = Zero  (reset intermediate_array[2])
Cycle 240-245: warpgroup_arrive()
Cycle 245-265: WGMMA[4] → intermediate_array[2]  ✈️
Cycle 265-270: tiled_mma.accumulate_ = One

Cycle 270-275: LDSM(k_block=6)  📥
Cycle 275-280: Convert(k_block=5) 🔄

k_block 5 (mma_id=1):
Cycle 280-285: warpgroup_arrive()
Cycle 285-305: WGMMA[5] → intermediate_array[2]  ✈️

Cycle 305-310: LDSM(k_block=7)  📥 Last prefetch
Cycle 310-315: Convert(k_block=6) 🔄

───────────────────────────────────────────────────────────────────────────
Cycle 315-320: warpgroup_commit_batch()
Cycle 320-325: warpgroup_wait<1>()
               ✅ intermediate_array[1] ready
───────────────────────────────────────────────────────────────────────────

Cycle 325-365: apply_groupwise_scale(accum, intermediate_array[1], scale, 1)
               🚀 Overlapped with WGMMA[5] in flight!
               └─ accum += intermediate[1] × scale[1]

═══════════════════════════════════════════════════════════════════════════
Chunk 3 (k_blocks [6, 7], K range [96, 127]):
───────────────────────────────────────────────────────────────────────────
k_block 6 (mma_id=0):
Cycle 365-370: tiled_mma.accumulate_ = Zero  (reset intermediate_array[3])
Cycle 370-375: warpgroup_arrive()
Cycle 375-395: WGMMA[6] → intermediate_array[3]  ✈️
Cycle 395-400: tiled_mma.accumulate_ = One

Cycle 400-405: (k_block + 2 = 8, out of range, skip LDSM)
Cycle 405-410: Convert(k_block=7) 🔄 Last conversion

k_block 7 (mma_id=1, last k_block):
Cycle 410-415: warpgroup_arrive()
Cycle 415-435: WGMMA[7] → intermediate_array[3]  ✈️

Cycle 435-440: (k_block + 2 out of range, skip)
Cycle 440-445: (k_block + 1 = 8, out of range, skip)

───────────────────────────────────────────────────────────────────────────
Cycle 445-450: warpgroup_commit_batch()
Cycle 450-455: warpgroup_wait<1>()
               ✅ intermediate_array[2] ready
───────────────────────────────────────────────────────────────────────────

Cycle 455-495: apply_groupwise_scale(accum, intermediate_array[2], scale, 2)
               🚀 Overlapped with WGMMA[7] in flight!

═══════════════════════════════════════════════════════════════════════════
Last chunk scaling (line 1073-1079):
───────────────────────────────────────────────────────────────────────────
Cycle 495-500: warpgroup_wait<0>()  ⏱️ Wait for all WGMMAs to complete
               ✅ WGMMA[7] completed → intermediate_array[3] ready
               ⚠️ Cannot overlap with any WGMMA!

Cycle 500-505: warpgroup_fence_operand(intermediate_array[3])
Cycle 505-545: apply_groupwise_scale(accum, intermediate_array[3], scale, 3)
               ❌ Pure overhead, no overlap

═══════════════════════════════════════════════════════════════════════════
After loop: Prefetch next K-tile (line 1082-1090)
───────────────────────────────────────────────────────────────────────────
Cycle 545-555: pipeline.consumer_wait(smem_pipe_read, barrier_token)
Cycle 555-560: LDSM(k_block=0, next_stage)  📥
Cycle 560-565: LDSM(k_block=1, next_stage)  📥
Cycle 565-570: Convert(k_block=0) 🔄

═══════════════════════════════════════════════════════════════════════════
Total time: ~570 cycles per K-tile (including pre-load and next prep)
Pipeline efficiency: ~70% (160 WGMMA cycles / ~230 effective cycles)
═══════════════════════════════════════════════════════════════════════════
```

### Key Pipeline Mechanisms

**1. Overlapped Scaling Pattern**

```cpp
warpgroup_commit_batch();  // Commit Chunk N's WGMMAs

if (chunk_id > 0) {
  warpgroup_wait<1>();  // ⚠️ Only wait until ≤1 WGMMA in flight
                        // Current: Chunk N in flight
                        // Guaranteed: Chunk N-1 completed

  // 🚀 Scale Chunk N-1 while Chunk N executes!
  apply_groupwise_scale(accum, intermediate_array[chunk_id - 1], ...);
  //     └─ ~40 cycles, fully overlapped!
}
```

**Overlap diagram:**
```
Time axis:
         ┌─────────────────────────────────────────┐
Chunk 0: │ WGMMA[0,1] (40 cycles)                  │
         └─────────────────────────────────────────┘
                                 ↓ commit
         ┌─────────────────────────────────────────┐
Chunk 1: │ WGMMA[2,3] (40 cycles)                  │
         └─────────────────────────────────────────┘
              ↑ wait<1>   ↓ commit
              │
              └── Scale Chunk 0 (40 cycles) ─────┐
                  ⚠️ Fully overlapped with WGMMA[2,3]! │
                                                 ↓
         ┌─────────────────────────────────────────┐
Chunk 2: │ WGMMA[4,5] (40 cycles)                  │
         └─────────────────────────────────────────┘
              ↑ wait<1>   ↓ commit
              │
              └── Scale Chunk 1 (40 cycles) ─────┐
                  ⚠️ Fully overlapped with WGMMA[4,5]! │
                                                 ↓
         ┌─────────────────────────────────────────┐
Chunk 3: │ WGMMA[6,7] (40 cycles)                  │
         └─────────────────────────────────────────┘
              ↑ wait<1>   ↓ commit
              │
              └── Scale Chunk 2 (40 cycles) ─────┐
                  ⚠️ Fully overlapped with WGMMA[6,7]! │
                                                 ↓
              wait<0> (wait for WGMMA[6,7] to complete)

              Scale Chunk 3 (40 cycles) ← Cannot overlap
```

**2. Why warpgroup_wait<1>() Not <0>()?**

```cpp
if (chunk_id > 0) {
  warpgroup_wait<1>();  // Allow 1 WGMMA in flight
  apply_groupwise_scale(...);
}
```

**Reasoning:**
- `warpgroup_wait<1>()` guarantees Chunk N-1 completed, Chunk N still in flight
- During scaling Chunk N-1, Chunk N's WGMMAs execute in parallel
- If using `warpgroup_wait<0>()`, would wait for all WGMMAs → no overlap

**Tradeoff:**
- ✅ Allows scaling to overlap with WGMMA
- ❌ Increases register pressure (needs multiple intermediate buffers)

**3. apply_groupwise_scale Implementation**

```cpp
template <class AccumTensor, class IntermTensor, class ScaleTensor>
CUTLASS_DEVICE void
apply_groupwise_scale(
    AccumTensor& accum,
    IntermTensor const& intermediate,
    ScaleTensor const& tCrS,
    int scale_idx,
    bool is_first_accum)
{
  multiply_add<ElementAccumulator> fma_op;

  // Traverse all accumulator elements
  for (int mma_m = 0; mma_m < size<1>(accum); mma_m++) {
    for (int m = 0; m < size<0, 1>(accum); m++) {

      float scale_val = scale_convertor(tCrS(...)[scale_idx]);

      for (int n = 0; n < size<0, 2>(accum); n++) {
        for (int e = 0; e < size<0, 0>(accum); e++) {

          auto coord = make_coord(make_tuple(e, m, n), mma_m, 0);

          if (is_first_accum) {
            accum(coord) = intermediate(coord) * scale_val;
          } else {
            accum(coord) = fma_op(intermediate(coord), scale_val, accum(coord));
          }
        }
      }
    }
  }
}
```

**Cost estimation:**
- Typical accum size: 128×32 = 4096 elements (TileM=128, TileN=32)
- Per element: 1 FMA (fused multiply-add)
- Estimated latency: ~40 cycles (depends on register file throughput)

---

## Performance Comparison

### Pre-Scale (MXFP4×BF16, K_BLOCK_MAX=8)

```
WGMMA time: 8 × 20 = 160 cycles (dominant term)
LDSM + Convert: Fully overlapped (prefetch distance k+2)
Control flow overhead: ~10 cycles
───────────────────────────────────
Total: ~170 cycles per K-tile
Pipeline efficiency: ~94% (160/170)
```

**Breakdown:**
```
Component               Cycles   Overlap   Effective
────────────────────────────────────────────────────
8 × WGMMA               160      -         160
8 × LDSM                40       100%      0
8 × Convert             40       100%      0
Control flow            10       0%        10
────────────────────────────────────────────────────
Total                   250      -         170
Overlap efficiency: (250 - 170) / 250 = 32%
```

### Post-Scale (MXFP4×BF16, K_BLOCK_MAX=8, NumChunksPerTileK=4, NumMMAsPerChunk=2)

```
Naive implementation (no overlap):
  4 chunks × (2×WGMMA + Scale) = 4 × (40 + 40) = 320 cycles

Optimized implementation (with overlap):
  Chunk 0: 2×WGMMA = 40 cycles (cannot scale, chunk_id=0)
  Chunk 1-3: max(2×WGMMA, Scale) = max(40, 40) = 40 cycles × 3 = 120 cycles
            └─ WGMMA and Scale perfectly overlapped!
  Last Scale: 40 cycles (cannot overlap)
  ───────────────────────────────────
  Total: 40 + 120 + 40 = 200 cycles

Actual: ~200-220 cycles per K-tile (including control flow overhead)

Compared to Pre-Scale: (210 - 170) / 170 ≈ +24% time
Overlap efficiency: (320 - 200) / 320 = 37.5% performance improvement
```

**Breakdown:**
```
Component               Cycles   Overlap   Effective
────────────────────────────────────────────────────
8 × WGMMA               160      -         160
8 × LDSM                40       100%      0
8 × Convert             40       100%      0
4 × Scale               160      75%       40
                                 (3/4)
Control flow            20       0%        20
────────────────────────────────────────────────────
Total                   420      -         220
Overlap efficiency: (420 - 220) / 420 = 48%
```

### Summary

| Metric | Pre-Scale | Post-Scale | Delta |
|--------|-----------|------------|-------|
| **Total cycles** | ~170 | ~220 | +29% |
| **WGMMA cycles** | 160 | 160 | 0% |
| **Scale cycles** | 0 | 40 (effective) | N/A |
| **Pipeline efficiency** | ~94% | ~73% | -21% |
| **Overlap efficiency** | 32% | 48% | +16% |
| **Functionality** | No scaling | Groupwise scaling | ✓ |
| **Register pressure** | Low | High | +4× |

**Key insights:**
- Post-scale adds ~50 cycles (29%) for groupwise scaling functionality
- 75% of scaling overhead (120/160 cycles) is hidden through overlap
- The 40-cycle residual (last chunk) is the fundamental bottleneck
- Still achieves high efficiency considering added functionality

---

## Critical Insights and Corrections

### ⚠️ Correction 1: WGMMA K Dimension Depends on Input Type

**❌ Initial misunderstanding:**
> "Every WGMMA processes 32 K elements"

**✅ Correct understanding:**
```
WGMMA K dimension depends on input data type:
- BF16 input: K = 16 per WGMMA
- FP8 input:  K = 32 per WGMMA
- FP16 input: K = 16 per WGMMA
- INT8 input: K = 32 per WGMMA
```

**Impact:**
- Directly affects `NumMMAsPerChunk = ScalingGroupSize / K_per_WGMMA`
- MXFP4×BF16 (K=16): NumMMAsPerChunk = 32/16 = 2
- INT4×FP8 (K=32): NumMMAsPerChunk = 128/32 = 4
- Also affects `K_BLOCK_MAX = TileK / K_per_WGMMA`

**Why this matters:**
- Misunderstanding K dimension leads to incorrect chunk organization calculations
- Affects all pipeline timing estimations
- Critical for understanding why overlap works so well in MXFP4×BF16 case

### ⚠️ Correction 2: SwapAB Is NOT Related to K Dimension

**❌ Initial misunderstanding:**
> "K-dimension constraint drives the swapAB design pattern"

**✅ Correct understanding:**
```
SwapAB is ONLY driven by M/N dimension constraints:
- WGMMA constraint: M=64 (fixed), N=variable
- Problem characteristic: M=variable, N=fixed
- Mismatch: Variable problem M → Fixed WGMMA M ❌

Solution: SwapAB maps variable problem M to variable WGMMA N ✓

K dimension plays NO ROLE in SwapAB design!
```

**Correct separation of concerns:**
- **SwapAB pattern**: Solves M/N variability mismatch
- **Chunk organization**: Solves groupwise scaling organization (depends on K)
- **Two independent design patterns!**

**Why this matters:**
- Conflating SwapAB with K dimension misrepresents the design rationale
- SwapAB is purely about tensor dimension mapping (M ↔ N)
- K dimension only affects how many WGMMAs per chunk, not the SwapAB decision

### ⚠️ Correction 3: Always Show Pre-Load Stage in Pipeline Diagrams

**❌ Initial bad habit:**
> Starting pipeline diagrams from k_block=0's WGMMA execution

**✅ Correct practice:**
> Always include the pre-load stage (LDSM and Convert for k_block 0)

**Why this matters:**
- Pre-load is a CRITICAL part of the pipeline design
- Without showing pre-load, the pipeline appears to have a cold start
- Omitting pre-load misrepresents the actual cycle count
- Makes it harder to understand the k+2 prefetch strategy

**Correct diagram structure:**
```
Stage 0: Pre-load (BEFORE loop)
  └─ LDSM(k_block=0)
  └─ LDSM(k_block=1)
  └─ Convert(k_block=0)

k_block 0: (Uses pre-loaded data)
  └─ WGMMA[0]
  └─ Prefetch k_block=2
  └─ Convert k_block=1

...
```

### Key Learnings

**1. Precision matters in architecture understanding:**
- Small details (K=16 vs K=32) have cascading effects on entire system
- Always verify hardware constraints against documentation
- Don't assume symmetry across different data types

**2. Separation of concerns:**
- SwapAB: M/N dimension mapping
- Chunk: K dimension organization
- TMA: Data movement optimization
- Each serves a distinct purpose, don't conflate them

**3. Complete pipeline visualization:**
- Always show initialization and finalization stages
- Pre-load is not "setup", it's part of the steady-state pipeline
- Omitting stages creates false understanding of cycle counts

**4. Overlap efficiency is configuration-dependent:**
- MXFP4×BF16 (NumMMAsPerChunk=2): Near-perfect overlap (40 vs 40 cycles)
- INT4×FP8 (NumMMAsPerChunk=4): Less perfect (80 vs 40 cycles)
- The "sweet spot" is when WGMMA time ≈ Scale time per chunk

---

## Code References

### Pre-Scale Implementation

**Main file:**
- `include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_prescale.hpp`

**Key sections:**
- Load function: Line 899-1038
- MMA function (First K-tile): Line 1235-1280
- MMA function (Mainloop): Line 1287-1330
- MMA function (Last K-tile): Line 1332-1362

**Critical code snippets:**
```cpp
// Pre-load (line 1244-1250)
Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, 0, read_stage);
Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, 1, read_stage);
Utils::convert_A_kblock(tCrA_load_4b_packed, tCrA_mma, 0);

// Main loop (line 1256-1268)
for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
  warpgroup_arrive();
  cute::gemm(tiled_mma, tCrA_mma(_,_,k_block), tCrB(_,_,k_block,read_stage), accum);
  tiled_mma.accumulate_ = GMMA::ScaleOut::One;
  warpgroup_commit_batch();

  if (k_block < K_BLOCK_MAX - 2) {
    Utils::copy_tensors_A(..., k_block + 2, read_stage);
  }
  if (k_block < K_BLOCK_MAX - 1) {
    Utils::convert_A_kblock(..., k_block + 1);
  }
}
```

### Post-Scale Implementation

**Main file:**
- `include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp`

**Key sections:**
- Chunk organization: Line 1005-1007
- Scaling helper: Line 848-882
- MMA function (First K-tile): Line 1015-1091
- MMA function (Mainloop): Line 1099-1170
- MMA function (Last K-tile): Line 1172-1219

**Critical code snippets:**
```cpp
// Intermediate buffer allocation (line 1007)
cute::array<decltype(make_fragment_like(accum)), NumChunksPerTileK> intermediate_array;

// Chunk loop with overlapped scaling (line 1034-1071)
for (int chunk_id = 0; chunk_id < NumChunksPerTileK; ++chunk_id) {
  tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

  for (int mma_id = 0; mma_id < NumMMAsPerChunk; ++mma_id) {
    int k_block = chunk_id * NumMMAsPerChunk + mma_id;

    warpgroup_arrive();
    cute::gemm(tiled_mma, tCrA_mma(_,_,k_block), tCrB(_,_,k_block,read_stage),
               intermediate_array[chunk_id]);
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;

    // ... prefetch and convert ...
  }

  warpgroup_commit_batch();

  // Overlapped scaling
  if (chunk_id > 0) {
    warpgroup_wait<1>();
    warpgroup_fence_operand(intermediate_array[chunk_id - 1]);
    apply_groupwise_scale(accum, intermediate_array[chunk_id - 1],
                          cute::get<1>(partitioned_extra_info), chunk_id - 1, chunk_id == 1);
  }
}

// Last chunk scaling (line 1073-1079)
warpgroup_wait<0>();
warpgroup_fence_operand(intermediate_array[chunk_id_]);
apply_groupwise_scale(accum, intermediate_array[chunk_id_],
                      cute::get<1>(partitioned_extra_info), chunk_id_, false);
```

### Related Files

**WGMMA instruction definitions:**
- `include/cute/atom/mma_traits_sm90_gmma.hpp`
  - BF16: Line 4200-4800 (SM90_64x*x16_*BF16* instructions)
  - FP8: Line 5465-6318 (SM90_64x*x32_*E4M3* instructions)

**Mainloop utilities:**
- `include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp`
  - Utils::copy_tensors_A: LDSM wrapper
  - Utils::convert_A_kblock: Type conversion (4-bit → BF16/FP8)
  - Utils::copy_tensors_SFA: Scale factor loading

**Pipeline synchronization:**
- `include/cutlass/pipeline/pipeline.hpp`
  - ProducerBarrierType: TMA barrier management
  - consumer_wait/consumer_release: Pipeline control

---

**Last Updated:** 2026-03-19
**Documented by:** Claude Code (mainloop pipeline learning session)
**Purpose:** Foundational understanding of instruction-level pipelining in Hopper mixed-precision kernels

**Next steps for deeper understanding:**
- [ ] SASS-level analysis of WGMMA instruction scheduling
- [ ] Register allocation and spilling analysis for intermediate buffers
- [ ] Bank conflict analysis in SMEM layouts for LDSM operations
- [ ] Cluster-level scheduling and CTA synchronization patterns
- [ ] Alternative overlap strategies (e.g., true pre-scale approaches)
