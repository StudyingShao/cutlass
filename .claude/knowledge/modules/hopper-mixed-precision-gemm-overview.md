# Hopper Mixed-Precision Grouped GEMM - Architecture Overview

**Last Updated**: 2026-03-18
**Status**: Evolving (active learning)

## Background

This document captures the architecture and implementation of CUTLASS's Hopper-based mixed-precision grouped GEMM kernels. These kernels enable efficient matrix multiplication with low-precision weights (MXFP4/INT4) and higher-precision activations (BF16/FP8), critical for LLM inference optimization.

**Why this matters:**
- Core kernel for quantized LLM inference (4-bit weights)
- Leverages Hopper's specialized hardware (TMA, WGMMA)
- Complex design patterns (SwapAB) that require deep understanding
- Performance-critical: every microsecond matters at scale

## Supported Configurations

The implementation supports two mixed-precision modes:

### Configuration 1: MXFP4 × BF16

| Component | Type | Details |
|-----------|------|---------|
| **Weights (A)** | `float_e2m1_t` | MXFP4 (4-bit floating point, E2M1 format) |
| **Activations (B)** | `bfloat16_t` | BF16 (16-bit brain float) |
| **Scale** | `float_ue8m0_t` | 8-bit exponent-only format |
| **Group Size** | 32 elements | Each scale applies to 32 weight elements |
| **Tile Shape** | 128×32×128 | TileM=128, TileN=32, **TileK=128 (fixed)** |
| **Accumulator** | `float` | 32-bit floating point |

### Configuration 2: INT4 × FP8

| Component | Type | Details |
|-----------|------|---------|
| **Weights (A)** | `int4b_t` | INT4 (4-bit signed integer) |
| **Activations (B)** | `float_e4m3_t` | FP8 E4M3 format |
| **Scale** | `bfloat16_t` | BF16 |
| **Group Size** | 128 elements | Each scale applies to 128 weight elements |
| **Tile Shape** | 128×16×512 | TileM=128, TileN=16/32/64, **TileK=128/256/512** |
| **Accumulator** | `float` | 32-bit floating point |

**Key difference**: INT4×FP8 has more flexible TileK (128/256/512), while MXFP4×BF16 fixes TileK=128.

## Core Technical Components

### 1. SwapAB Design Pattern

**Problem**: Hopper WGMMA instruction for FP8/INT4 has fixed M=64, K=32, but **variable N**.
Our grouped GEMM problem has **variable M** (batch dimension) but fixed N and K.

**Solution**: SwapAB transposes the operation to map variable M to WGMMA's variable N:
- Original: `C[M,N] = A[M,K] × B[K,N]` (M variable, N/K fixed)
- After SwapAB: `C^T[N,M] = B^T[N,K] × A^T[K,M]` (N/M swapped)
- **Result**: Problem's variable M now maps to TileN, which maps to WGMMA's variable N

**Why it works**: Maximizes Tensor Core utilization across different batch sizes.

**Implementation location**: `69_hopper_int4_fp8_grouped_gemm.cu:678-688`

**For deeper understanding**: See dedicated SwapAB analysis document (to be created).

### 2. TMA (Tensor Memory Accelerator)

**Purpose**: Hardware-accelerated data movement from Global Memory to Shared Memory.

**Our kernel uses TWO forms of TMA:**

1. **Descriptor-based TMA** (for A and B matrices)
   - Large 2D data (2-32 KB per tile)
   - Uses 128-byte descriptors (created via `cuTensorMapEncodeTiled`)
   - Supports multicast across cluster
   - PTX: `cp.async.bulk.tensor.2d...`

2. **Descriptor-less bulk copy TMA** (for scale and zero)
   - Small 1D data (128-256 bytes per tile)
   - No descriptor overhead
   - Direct GMEM address
   - PTX: `cp.async.bulk.shared::cluster.global...`
   - Wrapper: `SM90_BULK_COPY_G2S`

**Data loading summary:**

| Data | Size | TMA Type | Multicast |
|------|------|----------|-----------|
| A matrix (weights) | 2-32 KB | Descriptor-based | ✅ Along N dimension |
| B matrix (activations) | 2-32 KB | Descriptor-based | ✅ Along M dimension |
| Scale factors | 128-256 B | Descriptor-less | ❌ Too small |
| Zero points (optional) | 128-256 B | Descriptor-less | ❌ Too small |

**Critical requirement**: Compute capability 90a (not 90) - the "a" suffix enables TMA hardware.

**For detailed code-level understanding:**
- TMA concepts: `.claude/knowledge/architecture/tma-fundamentals.md`
- **Detailed TMA usage in this kernel**: `.claude/knowledge/modules/tma-usage-in-mixed-precision-gemm.md`

### 3. WGMMA (Warpgroup Matrix Multiply-Accumulate)

**Purpose**: Warp-group-level Tensor Core operations (128 threads = 4 warps).

**Data sourcing**:
- A operand: from registers (rmem)
- B operand: from shared memory descriptors (smem_desc)
- Accumulator: in registers

**Key feature**: Dynamic scaling control
- `ScaleOut::Zero` - initialize accumulator
- `ScaleOut::One` - accumulate to existing value

**Tile shape constraints** (for FP8/INT4):
- M: 64 (fixed for FP8), 64/128/256 (for other types)
- K: 32 (fixed)
- N: variable (16, 32, 64, 128, 256)

**Code location**: `sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp:1042-1045`

### 4. Groupwise Scaling

**Purpose**: Apply per-group scale factors to low-precision weights during computation.

**Two approaches**:
- **Post-scale** (current implementation): Scale after WGMMA accumulation
- **Pre-scale** (future optimization): Scale before WGMMA

**Key optimization - Overlapped Scaling**:
```
Chunk 0: GMMA compute
Chunk 1: GMMA compute  ← overlaps with → Chunk 0: Scale application
Chunk 2: GMMA compute  ← overlaps with → Chunk 1: Scale application
...
```

**Mechanism**:
1. Divide K dimension into chunks (chunk_size = group_size)
2. Each chunk: multiple GMMA operations → intermediate result
3. Wait for chunk N's GMMA to complete
4. Apply scale to chunk N-1 (while chunk N+1 starts)
5. Result: scaling latency hidden by GMMA compute

**Code location**: `sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp:848-882` (helper function)
**Usage**: Lines 1057-1069 (overlap pattern)

### 5. Data Type Conversion

**MXFP4 → BF16 Conversion**:
- **Old approach**: Lookup table (LUT)
- **Current approach**: Bit manipulation (ported from Triton)
- **Advantage**: Faster, no memory lookup overhead

**INT4 → FP8 Conversion**:
- Direct conversion via hardware instructions

**Load pattern**: LDSM (Load Shared Memory) → packed 4-bit registers → conversion → BF16/FP8 registers

**Code location**:
- LDSM load: `sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp:978-995`
- Conversion: `Utils::convert_A_kblock()`

## Data Flow

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ Host: Initialize A, B, Scale                            │
│  - Interleave weights for Hopper layout                 │
│  - Create TMA descriptors                               │
│  - Pack scales (group_size elements → 1 scale)          │
└─────────────────────────┬───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Producer Warp (load)                                     │
│  - TMA: GMEM → SMEM (A, B)                              │
│  - BULK_COPY: GMEM → SMEM (Scale)                       │
│  - Pipeline barrier sync                                 │
└─────────────────────────┬───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Consumer Warps (mma)                                     │
│  - LDSM: SMEM → RMEM (A, packed 4-bit)                  │
│  - Convert: FP4/INT4 → BF16/FP8                          │
│  - WGMMA: RMEM(A) × SMEM_DESC(B) → RMEM(intermediate)   │
│  - Scale: intermediate × scale → accum                   │
└─────────────────────────┬───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Epilogue: RMEM(accum) → GMEM(C)                         │
│  - Apply α and β (C = α·AB + β·C)                       │
└─────────────────────────────────────────────────────────┘
```

### Detailed Mainloop Flow

**Stage 1: TMA Load (Producer Warp)**
```cpp
// Load A and B via TMA
copy(tma_load_a.with(...), tAgA, tAsA);  // GMEM → SMEM
copy(tma_load_b.with(...), tBgB, tBsB);  // GMEM → SMEM

// Load Scale via BULK_COPY (lightweight, no TMA descriptor)
SM90_BULK_COPY_G2S::copy(scale_gmem_addr, tma_barrier, scale_smem_addr, bytes);
```

**Stage 2: LDSM Load (Consumer Warps)**
```cpp
// Load packed 4-bit weights from SMEM to registers
smem_thr_copy_A_LDSM.copy(tCsA_LDSM, tCrA_copy_view_LDSM);
```

**Stage 3: Conversion**
```cpp
// Unpack and convert: 4-bit → BF16/FP8
Utils::convert_A_kblock(tCrA_load_4b_packed, tCrA_mma, k_block);
```

**Stage 4: WGMMA Compute**
```cpp
// Chunk-based computation with overlapped scaling
for (chunk_id : chunks) {
  tiled_mma.accumulate_ = ScaleOut::Zero;  // Initialize

  for (mma_id : chunk) {
    gemm(tiled_mma, tCrA_mma, tCrB, intermediate[chunk_id]);
    tiled_mma.accumulate_ = ScaleOut::One;  // Accumulate
  }

  warpgroup_commit_batch();

  // While waiting for current chunk, scale previous chunk
  if (chunk_id > 0) {
    warpgroup_wait<1>();
    apply_groupwise_scale(accum, intermediate[chunk_id-1], scale);
  }
}
```

**Stage 5: Final Scaling**
```cpp
// Scale the last chunk
warpgroup_wait<0>();
apply_groupwise_scale(accum, intermediate[last_chunk], scale);
```

## Software Pipeline

**Multi-stage TMA Pipeline:**
- Typically 3-4 stages (controlled by `Stages` template parameter)
- Producer and consumer overlap via barriers
- `producer_acquire` → load → `producer_release`
- `consumer_wait` → compute → `consumer_release`

**K-block Pipeline within MMA:**
- Prefetch A data for k_block+2 while computing k_block
- Convert k_block+1 while computing k_block
- Minimizes bubble time between GMMA operations

## File Structure

### Main Entry Point
**File**: `examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_fp8_grouped_gemm.cu`

**Responsibilities**:
- Define data types for MXFP4×BF16 or INT4×FP8 (lines 115-135)
- Problem size configuration
- Input tensor initialization (with interleaving)
- TMA descriptor creation
- Kernel launch and verification

**Key configuration switch** (lines 115-135):
```cpp
// MXFP4 x BF16 (uncommented = active)
using MmaType = cutlass::bfloat16_t;
using QuantType = cutlass::float_e2m1_t;
#define GROUP_SIZE 32

// INT4 x FP8 (commented = inactive)
// using MmaType = cutlass::float_e4m3_t;
// using QuantType = cutlass::int4b_t;
// #define GROUP_SIZE 128
```

### Core Collective Implementation
**File**: `include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp`

**Responsibilities**:
- Post-scale mainloop collective (current production version)
- TMA and WGMMA orchestration
- Software pipelining logic
- Groupwise scaling with compute overlap
- Type conversions (FP4→BF16, INT4→FP8)

**Key sections**:
- `load()`: Producer warp TMA/BULK_COPY operations (lines 674-814)
- `mma()`: Consumer warp WGMMA compute and scaling (lines 886-1220)
- `apply_groupwise_scale()`: Scaling helper (lines 848-882)

### Pre-Scale Scaffold (Future Optimization)
**File**: `include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_prescale.hpp`

**Purpose**: Experimental scaffold for applying scales before WGMMA instead of after.

**Status**: Not yet complete, reserved for future optimization work.

## Recent Optimizations

### 1. FP4→BF16 Conversion (Bit Manipulation)
**Date**: Recent (before this session)
**Change**: Replaced LUT-based conversion with bit manipulation
**Impact**: Faster conversion, no memory lookup overhead
**Source**: Ported from Triton implementation

### 2. Overlapped Groupwise Scaling
**Date**: Recent (before this session)
**Change**: Scale previous chunk while computing current chunk
**Impact**: Hides scaling latency behind GMMA compute
**Mechanism**: `warpgroup_wait<1>()` + interleaved scaling

### 3. Scale Loading via BULK_COPY
**Date**: Recent (before this session)
**Change**: Replaced TMA load for scales with `SM90_BULK_COPY_G2S`
**Impact**:
- Lighter weight (no TMA descriptor needed)
- Simpler descriptor management
- Still participates in TMA barrier synchronization

### 4. Extracted Scaling Helper
**Date**: Recent (before this session)
**Change**: Created `apply_groupwise_scale()` function to deduplicate code
**Impact**: Shared between post-scale and future pre-scale implementations

## Performance Characteristics

### Typical Performance Numbers
From recent benchmarks (MXFP4×BF16):
- **Configuration**: groups=32, m=32, n=5888, k=2944, c=32
- **Tile**: 128×32×128
- **Duration**: ~157 μs (average across 5 runs)
- **Variation**: 1-2% across runs (stable)

### Key Performance Factors

**1. Tile Shape Selection**
- K=512 better for small M (problem M, which maps to TileN after SwapAB)
- Scheduler choice affects available tile shapes
- See CLAUDE.md for tile tuning guidelines

**2. SwapAB Efficiency**
- 30-40% speedup when dimension mapping matches WGMMA constraints
- Critical for variable batch size workloads

**3. Scaling Overhead**
- Post-scale overlapped with compute: minimal overhead
- Pre-scale (future): may enable further optimizations

## Practical Guidelines

### When to Use This Kernel

✅ **Good fit:**
- LLM inference with 4-bit quantized weights
- Grouped GEMM workloads (e.g., MoE, per-layer batching)
- Hopper GPUs (H100, H200)
- Variable batch sizes

❌ **Not suitable:**
- Non-Hopper GPUs (requires SM90a architecture)
- Full precision operations (use standard GEMM)
- Fixed, uniform problem sizes (SwapAB overhead unnecessary)

### Debugging Tips

**Common issues:**
- **TMA errors**: Check cmake used `-DCUTLASS_NVCC_ARCHS=90a` (not `90`)
- **Wrong results**: Run with `--debug_input_weight=true` to verify data layout
- **Performance regression**: Compare Duration metric across 5 runs minimum

**Debug flags** (in `69_hopper_int4_fp8_grouped_gemm.cu`):
- `--enable_print=true` - print intermediate results
- `--debug_input_weight=true` - debug weight values
- `--compare=true` - compare with reference implementation

### Build and Run

```bash
# Build
./shao_rebuild.sh  # Select task 0

# Run with profiling
./shao_rebuild.sh  # Select task 2 (NSight Compute)

# Debug
./shao_rebuild.sh  # Select task 4 (compute-sanitizer + SASS)
```

## Code References

**Main files:**
- Entry: `examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_fp8_grouped_gemm.cu`
- Collective: `include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp`
- Pre-scale: `include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_prescale.hpp`

**CuTe primitives:**
- MMA atoms: `include/cute/atom/mma_traits_sm90.hpp`
- TMA copy: `include/cute/arch/copy_sm90.hpp`
- Layouts: `include/cute/layout.hpp`

**Utilities:**
- Benchmark script: `.claude/skills/benchmark/SKILL.md`
- Development script: `shao_rebuild.sh`

## Next Steps / Open Questions

### Areas for deeper investigation:
- [ ] Pre-scale implementation details and performance comparison
- [ ] TMA descriptor creation and update mechanisms for grouped GEMM
- [ ] Detailed SASS analysis of WGMMA instruction utilization
- [ ] Memory bank conflict analysis in shared memory layout
- [ ] Cluster-level multicast patterns and efficiency

### Potential optimizations to explore:
- [ ] Pre-scale approach (vs current post-scale)
- [ ] Alternative tile shapes for different problem characteristics
- [ ] Different scheduler choices (Cooperative vs Pingpong)
- [ ] Further overlap opportunities (e.g., epilogue with next tile load)

## Related Knowledge

**To be created:**
- @.claude/knowledge/architecture/swapab-deep-dive.md
- @.claude/knowledge/architecture/tma-wgmma-integration.md
- @.claude/knowledge/optimization/tile-shape-tuning.md
- @.claude/knowledge/modules/mainloop-pipeline-analysis.md

**External references:**
- CUTLASS 3.x documentation
- NVIDIA Hopper architecture whitepaper
- CuTe layout system guide

---

**Note**: This document represents initial learning from code analysis. Understanding will deepen through hands-on optimization work and debugging sessions. Expect updates as knowledge evolves.
