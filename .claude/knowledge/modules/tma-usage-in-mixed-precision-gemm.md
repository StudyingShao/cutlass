# TMA Usage in Mixed-Precision Grouped GEMM

**Last Updated**: 2026-03-18
**Status**: Code understanding documentation
**Target Kernel**: `sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_prescale.hpp`

---

## Purpose

This document provides **detailed code-level understanding** of how TMA is used in our Hopper mixed-precision grouped GEMM kernel to load different types of input data.

**Kernel configurations:**
- MXFP4×BF16: MXFP4 weights with BF16 activations
- INT4×FP8: INT4 weights with FP8 activations

**Related documentation:**
- General TMA concepts: `.claude/knowledge/architecture/tma-fundamentals.md`
- Kernel overview: `.claude/knowledge/modules/hopper-mixed-precision-gemm-overview.md`

---

## Table of Contents

1. [TMA Usage Overview](#tma-usage-overview)
2. [TMA for A Matrix (Activations)](#tma-for-a-matrix-activations)
3. [TMA for B Matrix (Quantized Weights)](#tma-for-b-matrix-quantized-weights)
4. [Descriptor-Less TMA for Scale](#descriptor-less-tma-for-scale)
5. [Descriptor-Less TMA for Zero](#descriptor-less-tma-for-zero)
6. [Unified Mbarrier Design](#unified-mbarrier-design)
7. [Performance Characteristics](#performance-characteristics)

---

## TMA Usage Overview

### Three TMA Operation Types

Our kernel uses **all three TMA operation types** provided by Hopper:

| Data | TMA Type | PTX Instruction | Reason |
|------|----------|-----------------|--------|
| **A (activations)** | Descriptor-based LOAD or MULTICAST | `cp.async.bulk.tensor.2d...` | Large 2D tile, multicast benefit |
| **B (quantized weights)** | Descriptor-based LOAD or MULTICAST | `cp.async.bulk.tensor.2d...` | Large 2D tile, multicast benefit |
| **Scale factors** | Descriptor-less bulk copy | `cp.async.bulk.shared::cluster.global...` | Small 1D data, descriptor overhead too high |
| **Zero points** (optional) | Descriptor-less bulk copy | `cp.async.bulk.shared::cluster.global...` | Small 1D data, descriptor overhead too high |

### Data Size Comparison (MXFP4×BF16 Configuration)

```
Per K-tile data transfer:

A matrix (BF16):        128 × 128 × 2 bytes = 32,768 bytes  (94.1%)
B matrix (MXFP4):       32 × 128 × 0.5 bytes = 2,048 bytes  (5.9%)
Scale (UE8M0):          128 × 1 byte = 128 bytes            (0.37%)
Zero (UE8M0):           128 × 1 byte = 128 bytes            (0.37%)
────────────────────────────────────────────────────────────
Total:                                          34,072 bytes (100%)
```

**Design principle:** Use descriptor-based TMA for 99.9% of bandwidth (A+B), use descriptor-less TMA for 0.74% (scale+zero).

---

## TMA for A Matrix (Activations)

### Element Type

**File:** `sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_prescale.hpp`

**Type determination** (line 333, 343):
```cpp
using SwappedElementA = cute::conditional_t<!SwapAB, ConvertedElementA, ConvertedElementB>;
using TmaElementA = cute::conditional_t<IsSubbyteA, uint8_t, SwappedElementA>;
```

**For our configurations (SwapAB = true):**
- **MXFP4×BF16**: `SwappedElementA = bfloat16_t` (2 bytes)
  - Activations are BF16, after SwapAB they become matrix A
  - `TmaElementA = bfloat16_t` (non-subbyte)

- **INT4×FP8**: `SwappedElementA = float_e4m3_t` (1 byte)
  - Activations are FP8, after SwapAB they become matrix A
  - `TmaElementA = float_e4m3_t` (non-subbyte)

### TMA Type Selection

**Type assertion** (line 382-383):
```cpp
static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> ||
              cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
              "GmemTiledCopyA must be SM90_TMA_LOAD or SM90_TMA_LOAD_MULTICAST");
```

**Decision logic:**
- If `size<1>(ClusterShape{})` > 1 → Use `SM90_TMA_LOAD_MULTICAST`
- If `size<1>(ClusterShape{})` = 1 → Use `SM90_TMA_LOAD`

### Multicast Configuration

**Mask computation** (line 961-966):
```cpp
uint16_t mcast_mask_a = 0;
if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
  // Multicast A across N dimension
  for (int n = 0; n < size<1>(block_layout); ++n) {
    mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x, n, Int<0>{}));
  }
}
```

**Why multicast along N dimension:**
- After SwapAB, A matrix has shape `(M, K)` where M is the original problem's N dimension
- Cluster is organized as `(ClusterM, ClusterN, 1)`
- CTAs along the N dimension compute different N-tiles but need the same A data (same M, K tile)
- Multicast A to all CTAs along N axis → save `ClusterN` × bandwidth

**Example:**
```
ClusterShape = (2, 2, 1)
CTA grid:
  N →
M ┌─────┬─────┐
↓ │(0,0)│(0,1)│  ← These two need same A tile (same M-tile)
  ├─────┼─────┤
  │(1,0)│(1,1)│  ← These two need same A tile (different M-tile)
  └─────┴─────┘

Multicast mask for CTA(0,0): 0b0011 → broadcast to (0,0) and (0,1)
Multicast mask for CTA(1,0): 0b1100 → broadcast to (1,0) and (1,1)
```

### Descriptor Creation

**Host-side creation** (line 715-720):
```cpp
typename Params::TMA_A tma_load_a = make_tma_copy<TmaElementA>(
    GmemTiledCopyA{},                                           // SM90_TMA_LOAD or MULTICAST
    tensor_a,                                                   // GMEM tensor (M, K, L)
    SmemLayoutA{}(_,_,cute::Int<0>{}),                          // SMEM layout with swizzle
    make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),  // Box: (TileM, TileK)
    size<1>(ClusterShape{}));                                   // Multicast size along N
```

**Parameters explained:**
- `GmemTiledCopyA{}`: Copy operation type (LOAD or LOAD_MULTICAST)
- `tensor_a`: GMEM tensor with shape `(init_M, init_K, mock_L)`, where `init_M = get<0>(init_shape)` after SwapAB
- `SmemLayoutA{}(...)`: SMEM layout, includes swizzle pattern (e.g., `Swizzle<3,3,3>` for 128B swizzle)
- `make_shape(...)`: TMA box dimensions
  - **MXFP4×BF16**: `(128, 128)` → load 128×128 BF16 elements per TMA op
  - **INT4×FP8**: `(128, 512)` → load 128×512 FP8 elements per TMA op
- `size<1>(ClusterShape{})`: Multicast dimension size (number of CTAs to broadcast to)

**TileShape values:**
- **MXFP4×BF16**: `TileShape = (128, 32, 128)` → TileM=128, TileN=32, TileK=128
- **INT4×FP8**: `TileShape = (128, 32, 512)` → TileM=128, TileN=32, TileK=512

### Initialization Kernel

**Pre-baking descriptors** (line 765-773):
```cpp
test_kernel<SwapAB, ElementA, ElementB, SwappedElementA, SwappedElementB,
            UnderlyingProblemShape,
            typename Params::TMA_A, typename Params::TMA_B,
            InternalSwappedStrideA, InternalSwappedStrideB>
  <<<group_count, 1, 0, stream>>>(
    tma_load_a, tma_load_b,
    workspace, args.ptr_A, args.ptr_B,
    problem_shapes_device,
    ptr_dA, ptr_dB);
```

**What it does:**
- Launches with grid size = group_count (one block per group)
- Each block creates complete TMA descriptors for its group in global memory
- Descriptors stored at offset: `&workspace[sm_count * 2 + group_idx]`

**See:** `.claude/knowledge/optimization/tma-descriptor-prebaking.md` for optimization details.

### Main Kernel Invocation

**TMA copy** (line 1004):
```cpp
if (cute::elect_one_sync()) {
  copy(mainloop_params.tma_load_a.with(get<2>(input_tensormaps), *tma_barrier, mcast_mask_a),
       tAgA(_,_,_,*k_tile_iter),    // GMEM source: tile coordinates (m_tile, k_tile)
       tAsA(_,_,_,write_stage));    // SMEM destination: pipeline stage
}
```

**Parameters:**
- `get<2>(input_tensormaps)`: Pointer to pre-baked descriptor for current group
  - Base pointer: `&gmem_tensormap[sm_count * 2]`
  - Current group: `&gmem_tensormap[sm_count * 2 + group_idx]`
- `*tma_barrier`: Mbarrier address for transaction tracking
- `mcast_mask_a`: Multicast mask (0 if not using multicast)
- `*k_tile_iter`: K-dimension tile index (coordinates for TMA)

**What happens:**
1. Elect one thread per warp issues TMA instruction
2. Hardware reads descriptor from `get<2>(input_tensormaps)`
3. Hardware computes GMEM address: `base + stride * coordinates`
4. Hardware copies data from GMEM to SMEM at `tAsA` location
5. Hardware increments mbarrier transaction count
6. All asynchronous - issuing thread continues immediately

---

## TMA for B Matrix (Quantized Weights)

### Element Type

**Type determination** (line 334):
```cpp
using SwappedElementB = cute::conditional_t<!SwapAB, ConvertedElementB, ConvertedElementA>;
```

**For our configurations (SwapAB = true):**
- **MXFP4×BF16**: `SwappedElementB = float_e2m1_t` (4-bit MXFP4)
  - Weights are MXFP4, after SwapAB they become matrix B
  - Stored as **packed uint8**: 2 MXFP4 elements per byte

- **INT4×FP8**: `SwappedElementB = int4b_t` (4-bit INT4)
  - Weights are INT4, after SwapAB they become matrix B
  - Stored as **packed uint8**: 2 INT4 elements per byte

**TMA loads packed bytes:**
- TMA sees these as contiguous bytes
- Unpacking happens later in SMEM during type conversion

### TMA Type Selection

**Type assertion** (line 384-385):
```cpp
static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> ||
              cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
              "GmemTiledCopyB must be SM90_TMA_LOAD or SM90_TMA_LOAD_MULTICAST");
```

**Decision logic:**
- If `size<0>(ClusterShape{})` > 1 → Use `SM90_TMA_LOAD_MULTICAST`
- If `size<0>(ClusterShape{})` = 1 → Use `SM90_TMA_LOAD`

### Multicast Configuration

**Mask computation** (line 968-973):
```cpp
uint16_t mcast_mask_b = 0;
if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
  // Multicast B across M dimension
  for (int m = 0; m < size<0>(block_layout); ++m) {
    mcast_mask_b |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, Int<0>{}));
  }
}
```

**Why multicast along M dimension:**
- After SwapAB, B matrix has shape `(N, K)` where N is the original problem's M dimension
- CTAs along the M dimension compute different M-tiles but need the same B data (same N, K tile)
- Multicast B to all CTAs along M axis → save `ClusterM` × bandwidth

**Example:**
```
ClusterShape = (2, 2, 1)
CTA grid:
  N →
M ┌─────┬─────┐
↓ │(0,0)│(0,1)│  ← These two need different B tiles
  ├─────┼─────┤
  │(1,0)│(1,1)│
  └─────┴─────┘
        ↑
  These two need same B tile (same N-tile)

Multicast mask for CTA(0,0): 0b0101 → broadcast to (0,0) and (1,0)
Multicast mask for CTA(0,1): 0b1010 → broadcast to (0,1) and (1,1)
```

### Descriptor Creation

**Host-side creation** (line 721-726):
```cpp
typename Params::TMA_B tma_load_b = make_tma_copy(
    GmemTiledCopyB{},                                           // SM90_TMA_LOAD or MULTICAST
    tensor_b,                                                   // GMEM tensor (N, K, L)
    SmemLayoutB{}(_,_,cute::Int<0>{}),                          // SMEM layout with swizzle
    make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),  // Box: (TileN, TileK)
    size<0>(ClusterShape{}));                                   // Multicast size along M
```

**TMA box dimensions:**
- **MXFP4×BF16**: `(32, 128)` → load 32×128 packed MXFP4 elements
  - Actual bytes: 32×128×0.5 = 2,048 bytes
- **INT4×FP8**: `(32, 512)` → load 32×512 packed INT4 elements
  - Actual bytes: 32×512×0.5 = 8,192 bytes

### Main Kernel Invocation

**TMA copy** (line 1005):
```cpp
if (cute::elect_one_sync()) {
  copy(mainloop_params.tma_load_b.with(get<3>(input_tensormaps), *tma_barrier, mcast_mask_b),
       tBgB(_,_,_,*k_tile_iter),    // GMEM source: tile coordinates (n_tile, k_tile)
       tBsB(_,_,_,write_stage));    // SMEM destination: pipeline stage
}
```

**Parameters:**
- `get<3>(input_tensormaps)`: Pointer to pre-baked descriptor for current group
  - Base pointer: `&gmem_tensormap[sm_count * 2 + group_count]`
  - Current group: `&gmem_tensormap[sm_count * 2 + group_count + group_idx]`

### Combined Multicast Benefit

**Example calculation:**

Without multicast:
```
ClusterShape = (2, 2, 1) = 4 CTAs
Each CTA loads its own A and B tiles
Total bandwidth = 4 × (A_size + B_size)
```

With multicast:
```
A multicast to 2 CTAs along N → 2 loads instead of 4
B multicast to 2 CTAs along M → 2 loads instead of 4
Total bandwidth = 2 × A_size + 2 × B_size = 0.5 × (4 × (A_size + B_size))
Savings: 50% bandwidth
```

**For MXFP4×BF16 per K-tile:**
- Without multicast: 4 × (32 KB + 2 KB) = 136 KB
- With multicast: 2 × 32 KB + 2 × 2 KB = 68 KB
- **Bandwidth savings: 50%**

---

## Descriptor-Less TMA for Scale

### Why Not Descriptor-Based TMA?

**Overhead analysis:**

| Component | Size | Notes |
|-----------|------|-------|
| **TMA descriptor** | 128 bytes | Per-group overhead |
| **Scale data (MXFP4×BF16)** | 128 bytes | BLK_M × sizeof(float_ue8m0_t) = 128 × 1 |
| **Scale data (INT4×FP8)** | 256 bytes | BLK_M × sizeof(bfloat16_t) = 128 × 2 |

**Ratio:**
- MXFP4×BF16: Descriptor size = data size → **1:1 overhead!**
- INT4×FP8: Descriptor size = 0.5 × data size → **Still 50% overhead**

**Conclusion:** Descriptor overhead unjustifiable for such small data.

### Element Type

**Type determination:**
- **MXFP4×BF16**: `ElementScale = float_ue8m0_t` (1 byte, unsigned E8M0 format)
- **INT4×FP8**: `ElementScale = bfloat16_t` (2 bytes)

**Data layout:**
```
Scale shape: (scale_mn, scale_k)
where:
  scale_mn = SwapAB ? N : M  (line 811)
  scale_k = (K + chunk_size - 1) / chunk_size

For MXFP4×BF16 with chunk_size=32:
  scale_mn = N (original problem dimension)
  scale_k = K / 32

Each K-tile loads: BLK_M × 1 scale value
```

### Address Calculation

**Software-computed address** (line 1015-1016):
```cpp
auto scale_ptr = get<2>(load_inputs);            // Base pointer
auto scale_stride_k = get<3>(load_inputs);       // K-dimension stride
const int scale_load_k = *k_tile_iter / 1;       // K-tile index

constexpr int BLK_M = size<0>(TileShape{});
constexpr int scale_load_bytes = BLK_M * sizeof(NonVoidElementScale);

auto* scale_gmem_addr = reinterpret_cast<void const*>(
    scale_ptr + m_coord * BLK_M + scale_load_k * scale_stride_k);
//  ^^^^^^^^^   ^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//  Base        M offset            K offset
```

**Example calculation (MXFP4×BF16):**
```
m_coord = 2 (CTA's M coordinate)
BLK_M = 128
k_tile_iter = 1 (second K-tile)
scale_stride_k = get<1>(dS[group_idx])

scale_gmem_addr = scale_ptr + 2 × 128 + 1 × scale_stride_k
                = scale_ptr + 256 + scale_stride_k
```

### Descriptor-Less Bulk Copy Invocation

**PTX instruction wrapper** (line 1018-1019):
```cpp
auto* scale_smem_addr = static_cast<void*>(&sS(0, 0, write_stage));

cute::SM90_BULK_COPY_G2S::copy(
    scale_gmem_addr,                              // GMEM source (software computed)
    reinterpret_cast<uint64_t*>(tma_barrier),     // Mbarrier (shared with A/B!)
    scale_smem_addr,                              // SMEM destination
    scale_load_bytes);                            // Number of bytes (128 or 256)
```

**What happens:**
1. Elect one thread issues `cp.async.bulk.shared::cluster.global...` instruction
2. Hardware performs asynchronous copy from GMEM to SMEM
3. Hardware increments mbarrier transaction count (same mbarrier as A/B TMA)
4. No descriptor involved - direct addressing

**PTX instruction (actual):**
```ptx
cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
    [smem_addr], [gmem_addr], bytes, [mbar_addr];
```

### Key Benefits

1. **Zero descriptor overhead:**
   - No 128-byte descriptor to create
   - No descriptor storage in workspace
   - No descriptor modification on group transition

2. **Still hardware-accelerated:**
   - Asynchronous execution (non-blocking)
   - Dedicated TMA hardware units
   - Same performance characteristics as descriptor-based TMA for simple 1D copy

3. **Mbarrier integration:**
   - Shares mbarrier with A/B TMA loads
   - Consumer waits once for all data
   - Clean producer-consumer pipeline

4. **Simple addressing:**
   - Software computes address explicitly
   - No multidimensional coordinate calculation
   - Straightforward for 1D contiguous data

---

## Descriptor-Less TMA for Zero

### Usage (Optional)

**Enabled when** (line 1021-1030):
```cpp
if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
  auto zero_ptr = get<4>(load_inputs);
  constexpr int zero_load_bytes = BLK_M * sizeof(NonVoidElementZero);

  auto* zero_gmem_addr = reinterpret_cast<void const*>(
      zero_ptr + m_coord * BLK_M + scale_load_k * scale_stride_k);
  auto* zero_smem_addr = static_cast<void*>(&sZ(0, 0, write_stage));

  cute::SM90_BULK_COPY_G2S::copy(zero_gmem_addr,
      reinterpret_cast<uint64_t*>(tma_barrier),
      zero_smem_addr,
      zero_load_bytes);
}
```

**Configuration:**
- Kernel configurations can use scale-only or scale+zero
- Zero-point has same layout as scale: `(scale_mn, scale_k)`
- Same addressing calculation as scale
- Same descriptor-less bulk copy TMA

**Why optional:**
- Some quantization schemes use zero-point (asymmetric quantization)
- Some use only scale (symmetric quantization)
- Current configurations (MXFP4×BF16, INT4×FP8) use scale-only

---

## Unified Mbarrier Design

### Single Mbarrier for All Transfers

**Key insight:** All TMA operations (descriptor-based and descriptor-less) share one mbarrier per pipeline stage.

### Transaction Bytes Calculation

**Total bytes** (line 847):
```cpp
static constexpr uint32_t TmaTransactionBytes =
    TmaTransactionBytesMK +      // A matrix TMA
    TmaTransactionBytesNK +      // B matrix TMA
    TmaTransactionBytesExtra;    // Scale + Zero bulk copy
```

**Component calculation** (line 844-846):
```cpp
static constexpr uint32_t TmaTransactionBytesMK = Utils::compute_tma_transaction_bytes_mk();
static constexpr uint32_t TmaTransactionBytesNK = Utils::compute_tma_transaction_bytes_nk();
static constexpr uint32_t TmaTransactionBytesExtra = Utils::compute_tma_transaction_bytes_extra();
```

**Example values (MXFP4×BF16, TileShape = 128×32×128):**
```
TmaTransactionBytesMK = 128 × 128 × sizeof(bfloat16_t) = 32,768 bytes
TmaTransactionBytesNK = 32 × 128 × 0.5 (MXFP4 packed) = 2,048 bytes
TmaTransactionBytesExtra = 128 × 1 (scale) + 0 (no zero) = 128 bytes
────────────────────────────────────────────────────────────────────
TmaTransactionBytes = 34,944 bytes
```

### Producer-Consumer Flow

**Producer warp** (mainloop, line 997-1032):
```cpp
// Initialize barrier for this stage
using BarrierType = typename MainloopPipeline::ProducerBarrierType;
BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

// Set expected transaction bytes
// (Done once during initialization, not shown here)

int write_stage = smem_pipe_write.index();
if (cute::elect_one_sync()) {
  // Issue all TMA operations in parallel

  // 1. Descriptor-based TMA for A
  copy(mainloop_params.tma_load_a.with(get<2>(input_tensormaps), *tma_barrier, mcast_mask_a),
       tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));

  // 2. Descriptor-based TMA for B
  copy(mainloop_params.tma_load_b.with(get<3>(input_tensormaps), *tma_barrier, mcast_mask_b),
       tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));

  // 3. Descriptor-less bulk copy for scale
  if constexpr (ModeHasScales) {
    SM90_BULK_COPY_G2S::copy(scale_gmem_addr, tma_barrier, scale_smem_addr, scale_load_bytes);

    // 4. Descriptor-less bulk copy for zero (if needed)
    if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      SM90_BULK_COPY_G2S::copy(zero_gmem_addr, tma_barrier, zero_smem_addr, zero_load_bytes);
    }
  }
}

// Producer continues to next stage
++k_tile_iter;
++smem_pipe_write;
```

**Consumer warp** (WGMMA compute):
```cpp
// Wait for data to arrive
pipeline.consumer_wait(smem_pipe_read);  // Internally calls wait_barrier()

// Data now available in shared memory at read_stage
int read_stage = smem_pipe_read.index();

// Access data
Tensor sA = tAsA(_,_,_,read_stage);  // A matrix in SMEM
Tensor sB = tBsB(_,_,_,read_stage);  // B matrix in SMEM
Tensor sS = sS(_,_,read_stage);      // Scale in SMEM
Tensor sZ = sZ(_,_,read_stage);      // Zero in SMEM (if applicable)

// Perform WGMMA
wgmma(...);

// Consumer done with this stage
pipeline.consumer_release(smem_pipe_read);
++smem_pipe_read;
```

### Mbarrier State Machine

**Phase bit tracking:**
```
Initial state:
  mbarrier.state = 0
  mbarrier.phase = 0
  mbarrier.expected_bytes = TmaTransactionBytes

After set_barrier_transaction_bytes():
  mbarrier.state = TmaTransactionBytes
  mbarrier.arrived_bytes = 0

As TMA operations complete:
  mbarrier.arrived_bytes += bytes_transferred

When mbarrier.arrived_bytes == mbarrier.state:
  mbarrier.phase flips (0 → 1)
  wait_barrier() returns
  Data ready for consumer
```

**Key property:** All TMA operations (descriptor-based A/B, descriptor-less scale/zero) contribute to the same arrived_bytes counter → single synchronization point.

---

## Performance Characteristics

### Bandwidth Distribution

**Per K-tile transfer (MXFP4×BF16):**

```
Component               Bytes       % of Total   TMA Type
───────────────────────────────────────────────────────────────
A (BF16)               32,768       94.12%       Descriptor-based
B (MXFP4 packed)        2,048        5.88%       Descriptor-based
Scale (UE8M0)             128        0.368%      Descriptor-less
Zero (UE8M0)              128        0.368%      Descriptor-less (if used)
───────────────────────────────────────────────────────────────
Total                  34,072       100%         Mixed TMA
```

### Multicast Efficiency

**Without multicast** (ClusterShape = (2, 2, 1)):
```
Total loads per K-tile:
  4 CTAs × (32 KB A + 2 KB B) = 136 KB
```

**With multicast:**
```
A loads: 2 (multicast to 2 CTAs each) × 32 KB = 64 KB
B loads: 2 (multicast to 2 CTAs each) × 2 KB = 4 KB
Scale loads: 4 (no multicast for small data) × 128 B = 512 B
───────────────────────────────────────────────────────────
Total: 68.5 KB
```

**Bandwidth savings:**
```
Without multicast: 136 KB
With multicast:     68.5 KB
Savings:            49.6% ≈ 50%
```

**Why scale doesn't use multicast:**
- Data too small (128 bytes) to justify multicast complexity
- Each CTA loads its own scale (different M-tile → different scale values)
- Total scale bandwidth (4×128 = 512 bytes) is negligible compared to A+B (68 KB)

### Design Rationale Summary

| Decision | Reason |
|----------|--------|
| **A/B use descriptor-based TMA** | Large 2D data (2-32 KB), multicast saves 50% bandwidth, descriptor overhead < 1% |
| **A/B use multicast** | ClusterM×ClusterN CTAs share data along orthogonal axes → bandwidth savings = ~50% |
| **Scale/Zero use descriptor-less TMA** | Small 1D data (128-256 B), descriptor overhead = 50-100%, no multicast needed |
| **Single mbarrier** | Clean producer-consumer sync, hardware tracks all TMA ops automatically |
| **Pre-baking A/B descriptors** | Amortize descriptor creation cost across all K-tiles, only applies to descriptor-based TMA |

### Latency Analysis

**TMA operation latency (approximate):**
- First byte arrival: ~200-300 cycles
- Full tile transfer: ~200-500 cycles (depends on size and bandwidth)

**Overlap opportunities:**
- Producer issues TMA → immediately continues to next stage (async)
- Consumer waits only when needed (pipeline hides latency)
- With 3+ stages: TMA latency fully hidden by compute

**Critical path:**
```
Without TMA:
  Load A (blocking) → Load B (blocking) → Compute → Repeat

With TMA:
  Issue TMA A/B/Scale → Continue → Compute (wait if needed) → Repeat
  └─ TMA happens in parallel with previous stage compute
```

**Result:** TMA enables near-perfect overlap of data movement and computation.

---

## Code Location Reference

### Key Files

**Main implementation:**
- `include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_prescale.hpp`

### Critical Line Numbers

**Type definitions:**
- Line 333-334: `SwappedElementA`, `SwappedElementB`
- Line 343: `TmaElementA`
- Line 382-385: TMA type assertions

**Multicast configuration:**
- Line 961-966: A matrix multicast mask
- Line 968-973: B matrix multicast mask

**Descriptor creation:**
- Line 715-720: `tma_load_a` creation
- Line 721-726: `tma_load_b` creation

**Initialization kernel:**
- Line 107-191: `test_kernel` definition
- Line 765-773: `test_kernel` launch

**Main kernel TMA invocations:**
- Line 1004: A matrix TMA copy
- Line 1005: B matrix TMA copy
- Line 1018-1019: Scale descriptor-less bulk copy
- Line 1028-1029: Zero descriptor-less bulk copy (if enabled)

**Transaction bytes:**
- Line 844-847: Transaction bytes calculation

---

## Related Documentation

**Architecture concepts:**
- `.claude/knowledge/architecture/tma-fundamentals.md` - General TMA concepts, two forms, PTX instructions

**Optimization techniques:**
- `.claude/knowledge/optimization/tma-descriptor-prebaking.md` - Pre-baking optimization for A/B descriptors

**Kernel overview:**
- `.claude/knowledge/modules/hopper-mixed-precision-gemm-overview.md` - High-level kernel architecture

**Design patterns:**
- `.claude/knowledge/architecture/swapab-pattern.md` - Why dimensions are swapped, affects TMA setup

---

**Last Updated:** 2026-03-18
**Documented by:** Claude Code (knowledge management system)
**Purpose:** Code understanding for maintenance and development of this specific kernel

