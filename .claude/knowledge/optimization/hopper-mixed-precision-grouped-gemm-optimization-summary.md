# Hopper Mixed-Precision Grouped GEMM Optimization Summary

**Branch**: `jiangs/mxfp4xfp8`
**Base Commit**: `df18f5e4` (main branch)
**Last Updated**: 2026-03-23
**Total Commits**: 33

---

## Overview

This document summarizes all optimization work on Hopper mixed-precision grouped GEMM kernels, organized by kernel type and optimization category. The work spans three major kernel variants:

1. **INT4 × FP8 (w4a8)** - INT4 weights × FP8 activations, group_size=128
2. **MXFP4 × BF16** - MXFP4 weights × BF16 activations, group_size=32
3. **Shared Infrastructure** - Optimizations and features benefiting both kernels

---

## Table of Contents

- [1. INT4 × FP8 (w4a8) Kernel Optimizations](#1-int4--fp8-w4a8-kernel-optimizations)
- [2. MXFP4 × BF16 Kernel Optimizations](#2-mxfp4--bf16-kernel-optimizations)
- [3. Cross-Kernel Infrastructure Optimizations](#3-cross-kernel-infrastructure-optimizations)
- [4. Scheduler and Work Distribution Optimizations](#4-scheduler-and-work-distribution-optimizations)
- [5. Debugging and Tooling Improvements](#5-debugging-and-tooling-improvements)
- [6. Code Quality and Maintenance](#6-code-quality-and-maintenance)
- [7. Performance Summary](#7-performance-summary)

---

## 1. INT4 × FP8 (w4a8) Kernel Optimizations

### 1.1 Initial Implementation
**Commits**:
- `f84d8a47` - Hopper w4a8 per-group per-tensor grouped gemm
- `428b76bf` - Hopper w4a8 grouped gemm: sub k tile scaling part I

**Description**:
- Initial implementation of INT4 weights × FP8 activations grouped GEMM
- Per-group per-tensor scaling support
- Sub-K-tile scaling (part I)
- Leverages Hopper TMA and WGMMA instructions

**Key Features**:
- Input types: `int4b_t` (weights), `float_e4m3_t` (activations)
- Scale type: `bfloat16_t`
- Group size: 128 elements per scale factor

---

### 1.2 Configuration and Tuning
**Commits**:
- `5da4e8f3` - Hopper w4a8 grouped gemm: config auto-tuner
- `91659b36` - Hopper w4a8 grouped gemm: fix a bug and add more profiling configs
- `5e8c8588` - Hopper w4a8 grouped gemm: fix kernel profiler
- `8a32ad55` - INT4 x FP8: Update empirical CUTLASS config

**Changes**:
- **Auto-tuner configuration** - Set up tile shape search space and profiling infrastructure
- **Kernel profiler fixes** - Fixed bugs in kernel profiler for accurate benchmarking
- **Additional profiling configs** - Added more tile shape configurations for comprehensive tuning
- **Empirical CUTLASS config updates** - Updated configuration based on benchmark results

**Impact**: Established systematic performance tuning methodology

---

### 1.3 Performance Optimizations
**Commits**:
- `ebe9885e` - Hopper w4a8 grouped gemm: perf optimization
- `1139831c` - Hopper w4a8 grouped gemm: add warpgroup_wait before accum scaling
- `ffe36996` - Remove some useless warpgroup_wait<K_WAIT_MAX>()

**Changes**:
1. **General performance optimization** (`ebe9885e`)
   - Low-level SASS optimizations for compute efficiency

2. **Warpgroup synchronization fixes** (`1139831c`)
   - Added `warpgroup_wait` before accumulator scaling
   - Ensures correctness of accumulator operations

3. **Removed unnecessary synchronization** (`ffe36996`)
   - Removed useless `warpgroup_wait<K_WAIT_MAX>()` calls
   - Reduced unnecessary pipeline stalls

**Impact**: Improved kernel throughput and reduced latency

---

### 1.4 Type Conversion Optimization
**Commits**:
- `c7ef82f0` - INT4 x FP8: Optimize int4 to FP8 convert SASS

**Description**:
- **Optimized INT4 → FP8 conversion SASS**
- Hand-optimized PTX/SASS for efficient type conversion
- Reduced register pressure and instruction count

**Impact**: Faster data conversion in mainloop

---

### 1.5 Bug Fixes
**Commits**:
- `15b636db` - INT4 x FP8 fix

**Description**: Critical bug fixes for INT4 × FP8 kernel correctness

---

## 2. MXFP4 × BF16 Kernel Optimizations

### 2.1 Initial Implementation
**Commits**:
- `7b7f5a81` - MXFP4 x BF16 Grouped GEMM support
- `2d9c777d` - Update groupwise_verify function
- `e2a96c11` - MXFP4 x BF16 Dense and Grouped GEMM finished

**Description**:
- Initial implementation of MXFP4 × BF16 grouped GEMM
- Dense and grouped GEMM variants
- Updated groupwise verification functions

**Key Features**:
- Input types: `float_e2m1_t` (MXFP4 weights), `bfloat16_t` (activations)
- Scale type: `float_ue8m0_t` (8-bit exponent-only format)
- Group size: 32 elements per scale factor

---

### 2.2 Critical Performance Fixes
**Commits**:
- `09aca1b4` - Fix a bug from CUTLASS itself
- `034d0078` - Fix a perf regression caused by fp8->bf16 scale factor conversion for MXFP4 x BF16 Dense and Grouped GEMM

**Changes**:
1. **Fixed CUTLASS bug** (`09aca1b4`)
   - Identified and fixed upstream CUTLASS bug
   - Reference: https://github.com/NVIDIA/TensorRT-LLM/pull/7072

2. **Fixed scale conversion performance regression** (`034d0078`)
   - Resolved FP8 → BF16 scale factor conversion performance issue
   - Prevented unexpected performance degradation in both dense and grouped GEMM

**Impact**: Restored optimal performance baseline

---

### 2.3 FP4 → BF16 Conversion Optimizations

#### 2.3.1 Interleaved Format Approach
**Commit**:
- `00d02341` - MXFP4 x BF16 Introduce an interleaved format to accelerate the FP4->BF16 conversion

**Description**:
- Introduced interleaved memory format for FP4 weights
- Accelerated FP4 → BF16 conversion through improved memory access pattern

---

#### 2.3.2 Bit-Manipulation Conversion (Major Optimization)
**Commit**:
- `6d36a517` - MXFP4 x BF16: Replace LUT-based with bit-manipulation FP4->BF16 conversion ported from Triton

**Description**:
- **Replaced LUT-based with bit-manipulation FP4 → BF16 conversion**
- Ported efficient conversion algorithm from Triton
- Eliminated shared memory LUT overhead
- Direct register-based bit manipulation

**Impact**: Significant reduction in shared memory usage and conversion latency

---

#### 2.3.3 SASS-Level Optimization
**Commit**:
- `9d496805` - MXFP4 x BF16: Optimize FP4 to BF16 convert SASS

**Description**:
- Hand-optimized FP4 → BF16 conversion SASS
- Low-level instruction scheduling and register allocation
- Minimized instruction count and pipeline bubbles

**Impact**: Further improved conversion throughput beyond algorithmic change

---

### 2.4 Architecture Exploration - Pre-Scale Design
**Commits**:
- `66be1e08` - MXFP4 x BF16: Introduce pre-scale mainloop scaffold for future development
- `353884e0` - MXFP4 x BF16: Remove post-scale process for the new created scaffold

**Description**:
1. **Introduced pre-scale mainloop scaffold** (`66be1e08`)
   - Created alternative architecture for future development
   - Pre-scale weights before WGMMA instead of post-scale accumulators

2. **Removed post-scale from new scaffold** (`353884e0`)
   - Clean separation between pre-scale and post-scale approaches
   - Prepared for performance comparison experiments

**Status**: Work-in-progress exploration for potential future optimization

---

### 2.5 TMA Descriptor Experiments
**Commit**:
- `3e8e1cac` - [WIP] MXFP4 x BF16 & INT4 x FP8: grouped TMA descriptor experiment

**Description**:
- [WIP] Grouped TMA descriptor experiments for both MXFP4 × BF16 and INT4 × FP8
- Exploring alternative TMA configuration strategies
- Potential for improved memory access efficiency

**Status**: Work-in-progress

---

## 3. Cross-Kernel Infrastructure Optimizations

These optimizations benefit **both INT4 × FP8 and MXFP4 × BF16 kernels**.

### 3.1 Memory Access Optimizations

#### 3.1.1 Shared Memory Access Pattern
**Commit**:
- `f1ca160c` - MXFP4 x BF16 & INT4 x FP8: Replace LDS with LDSM

**Description**:
- **Replaced LDS with LDSM** (Load Shared Matrix)
- Used matrix-oriented shared memory load instruction
- Improved memory coalescing and reduced bank conflicts

**Impact**: Better shared memory bandwidth utilization for both kernels

---

#### 3.1.2 TMA Descriptor Optimization
**Commit**:
- `e166a053` - MXFP4 x BF16 & INT4 x FP8: TMA Desc opt

**Description**:
- Optimized TMA descriptor configurations
- Improved multidimensional tensor access patterns
- Reduced TMA overhead in mainloop

**Impact**: Faster global → shared memory transfers

---

#### 3.1.3 Scale/Zero Loading Optimization
**Commit**:
- `1c34631d` - MXFP4 x BF16 & INT4 x FP8: Replace scale/zero TMA load with SM90 bulk copy

**Description**:
- **Replaced scale/zero TMA load with SM90 bulk copy**
- Switched from TMA instructions to optimized bulk copy for scale factors
- Reduced TMA descriptor overhead for small tensors
- Better suited for scale tensor access patterns

**Impact**: Faster scale factor loading, reduced TMA resource contention

---

### 3.2 Compute-Memory Overlap
**Commit**:
- `5a10875d` - MXFP4 x BF16 & INT4 x FP8: Overlap groupwise scaling with GMMA compute

**Description**:
- **Overlap groupwise scaling with GMMA compute**
- Pipelined scaling operations with Tensor Core computation
- Hides scaling latency behind WGMMA execution

**Impact**: Improved overall kernel throughput, better compute utilization

---

### 3.3 Code Refactoring and Deduplication
**Commit**:
- `03b9c945` - MXFP4 x BF16 & INT4 x FP8: Extract apply_groupwise_scale helper to deduplicate post-scale code

**Description**:
- **Extracted `apply_groupwise_scale` helper function**
- Deduplicated post-scale code between MXFP4 × BF16 and INT4 × FP8 kernels
- Unified scaling interface for maintainability

**Impact**: Improved code maintainability, easier to optimize both kernels together

---

### 3.4 Work Distribution and Scheduling
**Commit**:
- `8872d972` - MXFP4 x BF16 & INT4 x FP8: Add ThreadBlockSwizzle option

**Description**:
- Added `ThreadBlockSwizzle` option
- Flexible thread block scheduling for different problem shapes
- Improves load balancing across SMs

**Impact**: Better GPU utilization for irregular workloads

---

## 4. Scheduler and Work Distribution Optimizations

### 4.1 Fixed-N Optimization
**Commit**:
- `6222dcc1` - Scheduler: Pre-compute tiles_m for fixed N optimization

**Description**:
- **Pre-compute `tiles_m` for fixed N optimization**
- Leveraged the constraint that N (output_dim) is **fixed across all groups** in LLM inference workloads
- Cached `problem_blocks_m` computation to avoid redundant `ceil_div` and `round_up` in work-stealing loop

**Changes**:
- Added `problem_blocks_m_fixed` to `GroupInfo` struct
- Computed and cached in constructor and first call to `get_work_idx_m_and_n()`
- Reused cached value in while loop, only compute variable `ctas_along_n`

**Performance Impact**:
- Baseline: **158.05 μs**
- After optimization: **153.25 μs**
- Improvement: **-3.04%** (4.8 μs reduction)

**Key Insight**:
In grouped GEMM for LLM inference:
- **N and K are FIXED** across all groups (same value for every group)
- **M is VARIABLE** per group (different groups have different M values)
- After SwapAB transformation, kernel sees: M'=N (fixed), N'=M (variable), K'=K (fixed)
- **Implication**: `tiles_m = ceil(N/TileM)` is constant and can be pre-computed

**Reference**: `.claude/optimization-logs/2026-03-19-scheduler-work-switching.md`

---

## 5. Debugging and Tooling Improvements

### 5.1 Runtime Debug Infrastructure
**Commits**:
- `7aafe950` - MXFP4 x BF16 & INT4 x FP8: Add runtime debug options and improve error stats
- `7c660459` - MXFP4 x BF16 & INT4 x FP8: Improve runtime debug options

**Changes**:
1. **Initial debug options** (`7aafe950`)
   - Added runtime debug flags for intermediate result printing
   - Improved error statistics reporting
   - Better numerical error tracking

2. **Enhanced debug options** (`7c660459`)
   - Expanded debug flag coverage
   - More granular control over debug outputs
   - Improved debugging workflow

**Available Debug Flags**:
```bash
--enable_print=true              # Print intermediate results
--enable_print_weight=true       # Print weight values
--debug_input_act=true           # Debug activation inputs
--debug_input_weight=true        # Debug weight inputs
--debug_input_scale=true         # Debug scale factors
--compare=true                   # Compare with reference
```

**Impact**: Faster debugging, easier correctness validation

---

### 5.2 Example Cleanup
**Commit**:
- `d59d6369` - Ex55: Remove shuffle instance in example

**Description**:
- Removed shuffle instance in Example 55
- Cleaned up example code for clarity

---

## 6. Code Quality and Maintenance

### 6.1 Code Cleanup
**Commit**:
- `28ca70b7` - MXFP4 x BF16 & INT4 x FP8: Code Clean

**Description**:
- General code cleanup for both MXFP4 × BF16 and INT4 × FP8 kernels
- Removed dead code and commented-out experiments
- Improved code readability

---

## 7. Performance Summary

### 7.1 Key Performance Wins

| Optimization | Kernel(s) | Impact |
|-------------|-----------|---------|
| Bit-manipulation FP4 → BF16 conversion | MXFP4 × BF16 | Major: Eliminated LUT overhead |
| Optimized INT4 → FP8 conversion SASS | INT4 × FP8 | Significant: Reduced conversion latency |
| Overlap scaling with GMMA compute | Both | High: Better compute utilization |
| Replace TMA with bulk copy for scales | Both | Medium: Reduced TMA contention |
| Replace LDS with LDSM | Both | Medium: Better SMEM bandwidth |
| Fixed-N scheduler optimization | Both | **3.04%**: 158.05 μs → 153.25 μs |
| TMA descriptor optimization | Both | Medium: Improved memory access |
| Removed unnecessary warpgroup_wait | INT4 × FP8 | Small: Reduced pipeline stalls |

### 7.2 Optimization Timeline

**Phase 1: Initial Implementation (Early commits)**
- `f84d8a47` (Hopper w4a8 per-group per-tensor grouped gemm) - `7b7f5a81` (MXFP4 x BF16 Grouped GEMM support): Basic kernel implementations

**Phase 2: Performance Tuning (Middle commits)**
- `00d02341` (MXFP4 x BF16 Introduce an interleaved format to accelerate the FP4->BF16 conversion) - `5a10875d` (MXFP4 x BF16 & INT4 x FP8: Overlap groupwise scaling with GMMA compute): Memory access and conversion optimizations

**Phase 3: Infrastructure and Refinement (Recent commits)**
- `1c34631d` (MXFP4 x BF16 & INT4 x FP8: Replace scale/zero TMA load with SM90 bulk copy) - `03b9c945` (MXFP4 x BF16 & INT4 x FP8: Extract apply_groupwise_scale helper to deduplicate post-scale code): Code quality and shared infrastructure

**Phase 4: Advanced Optimization (Latest commits)**
- `66be1e08` (MXFP4 x BF16: Introduce pre-scale mainloop scaffold for future development) - `6222dcc1` (Scheduler: Pre-compute tiles_m for fixed N optimization): Architecture exploration and scheduler optimization

### 7.3 Latest Performance Baseline

**Configuration**:
- Problem size: `groups=32, m=32, n=5888, k=2944, c=32` (MXFP4×BF16)
- Tile shape: 128×16×128 (TileM × TileN × TileK after SwapAB)
- Scheduler: `KernelPtrArrayTmaWarpSpecializedCooperative`

**Current Performance** (as of `6222dcc1` - Scheduler: Pre-compute tiles_m for fixed N optimization):
- **Duration**: 153.25 μs
- **Previous baseline**: 158.05 μs
- **Total improvement from scheduler optimization**: 3.04%

---

## 8. Open Experiments and Future Work

### 8.1 In-Progress Experiments

1. **Grouped TMA Descriptor**
   - `3e8e1cac` - [WIP] MXFP4 x BF16 & INT4 x FP8: grouped TMA descriptor experiment
   - Exploring alternative TMA configuration
   - Potential for memory access improvement
   - Status: WIP, needs performance validation

2. **Pre-Scale Mainloop**
   - `66be1e08` - MXFP4 x BF16: Introduce pre-scale mainloop scaffold for future development
   - `353884e0` - MXFP4 x BF16: Remove post-scale process for the new created scaffold
   - Alternative architecture: scale before WGMMA instead of after
   - May reduce accumulator pressure
   - Status: Scaffold ready, needs implementation and benchmarking

### 8.2 Future Optimization Directions

1. **Further SASS optimization**
   - Continue hand-tuning critical conversion and scaling code paths

2. **Pipeline optimization**
   - Explore additional opportunities for compute-memory overlap

3. **SwapAB and tile shape tuning**
   - Systematic evaluation of tile shapes for different problem sizes
   - Leverage SwapAB design for optimal WGMMA utilization

4. **Activation scaling exploration** (from other branches)
   - Activation block scaling (branch: `jiangs/Hopper_moe_w4group_a8tensor`)
   - K-major ActScale with cp.async (branch: `jiangs/mxfp4xfp8_actScaling`)

---

## 9. Key Technical Insights

### 9.1 SwapAB Design Pattern
- **Problem**: Variable M per group, fixed N and K
- **WGMMA constraint**: M=64 (fixed), N=variable, K=16/32 (fixed, type-dependent)
- **Solution**: SwapAB maps variable problem M → variable WGMMA N
- **Result**: Maximized Tensor Core utilization across irregular workloads

### 9.2 Conversion Strategy Evolution
- **LUT approach**: Simple but consumes shared memory
- **Bit-manipulation**: Ported from Triton, eliminates SMEM overhead
- **SASS optimization**: Hand-tuned for minimal instruction count
- **Lesson**: Algorithmic change first, then low-level tuning

### 9.3 Scaling Overlap Strategy
- **Key insight**: Scaling operations are memory-bound
- **Solution**: Pipeline scaling loads/computes with WGMMA execution
- **Result**: Hide scaling latency behind compute

### 9.4 Scheduler Optimization Pattern
- **Key insight**: LLM inference has fixed N/K, variable M per group
- **Solution**: Pre-compute constant expressions (tiles_m) once
- **Result**: Eliminate redundant arithmetic in hot loop
- **Lesson**: Domain-specific constraints enable targeted optimizations

---

## 10. References

### Code Locations
- **Main kernel**: `examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_fp8_grouped_gemm.cu`
- **Mainloop (post-scale)**: `include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp`
- **Mainloop (pre-scale scaffold)**: `include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_prescale.hpp`
- **Scheduler**: Work-stealing scheduler in kernel implementation

### External References
- CUTLASS bug fix: https://github.com/NVIDIA/TensorRT-LLM/pull/7072
- Triton FP4 conversion reference: (algorithm ported)
- Optimization log: `.claude/optimization-logs/2026-03-19-scheduler-work-switching.md`

### Related Branches
- `jiangs/mxfp4xfp8` - Main development branch (current)
- `jiangs/mxfp4xfp8_actScaling` - Activation scaling experiments
- `jiangs/Hopper_moe_w4group_a8tensor` - Alternative activation scaling approach
- `jiangs/Hopper_moe_w4a8_context_opt` - Context-specific optimizations

---

## Document History

- **Created**: 2026-03-23 by Claude Sonnet 4.5
- **Based on**: Git history from `df18f5e4` (main branch merge point) to `6222dcc1` (HEAD - Scheduler: Pre-compute tiles_m for fixed N optimization)
- **Scope**: 33 commits spanning INT4 × FP8 and MXFP4 × BF16 kernel development
- **Purpose**: Comprehensive optimization summary for future reference and knowledge transfer
- **Last Updated**: 2026-03-23 (added commit messages to all commit IDs)
