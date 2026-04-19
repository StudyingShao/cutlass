# TMA (Tensor Memory Accelerator) Fundamentals for Hopper Architecture

**Last Updated**: 2026-03-18
**Status**: Stable foundational knowledge
**Architecture**: NVIDIA Hopper (SM90)

---

## Table of Contents

1. [What is TMA?](#what-is-tma)
2. [Two Forms of TMA](#two-forms-of-tma)
3. [TMA Descriptor Structure](#tma-descriptor-structure)
4. [TMA PTX Instructions](#tma-ptx-instructions)
5. [CuTe TMA Abstraction](#cute-tma-abstraction)
6. [TMA Usage Workflow](#tma-usage-workflow)
7. [Advanced Topics](#advanced-topics)
8. [Code References](#code-references)

---

## What is TMA?

### Overview

**TMA (Tensor Memory Accelerator)** is a hardware feature introduced in NVIDIA Hopper (SM90) architecture that provides **hardware-accelerated multidimensional data movement** between global memory (GMEM) and shared memory (SMEM).

**Historical context:**
- **Pre-Hopper** (Ampere, Turing): `cp.async` instructions for asynchronous GMEM→SMEM copies
  - Single-dimensional, requires manual address calculation
  - Software-managed pipelines for overlapping copy and compute
- **Hopper (SM90)**: TMA hardware units
  - Multi-dimensional tensor awareness (1D-5D)
  - Hardware understands shapes, strides, and layouts
  - Integrated with cluster-wide operations (multicast)

### Key Benefits

| Feature | Benefit | Impact |
|---------|---------|--------|
| **Hardware acceleration** | Dedicated TMA units | Frees compute threads, reduces latency |
| **Multidimensional** | Understands tensor shapes/strides | Single instruction moves entire tiles |
| **Descriptor-based** | Encodes all transfer parameters | Reduces instruction overhead |
| **Asynchronous** | Non-blocking for compute threads | Overlaps copy with compute |
| **Cluster multicast** | Broadcasts data to multiple CTAs | Reduces memory bandwidth 2-4x |
| **Swizzle support** | Built-in shared memory layout patterns | Avoids bank conflicts automatically |

### When to Use TMA

✅ **TMA is ideal for:**
- Moving large data tiles (128×128, 256×128, etc.)
- Regular tensor layouts (row-major, column-major with constant strides)
- Kernels with producer-consumer patterns (warp specialization)
- Cluster-level parallelism (multiple CTAs cooperating)

❌ **TMA is not suitable for:**
- Small, irregular data accesses (use bulk copy or direct loads)
- Dynamic, data-dependent addressing (TMA requires static shapes)
- Very small tiles (<16 elements) where overhead dominates

---

## Two Forms of TMA

**Critical understanding:** Hopper TMA actually provides **two distinct instruction families**, both hardware-accelerated but serving different use cases.

### Form 1: Descriptor-Based TMA

**Purpose:** Multidimensional tensor transfers with complex addressing.

**PTX instruction pattern:**
```ptx
cp.async.bulk.tensor.{1d,2d,3d,4d,5d}.shared::cluster.global.mbarrier::complete_tx::bytes...
    [smem_addr], [descriptor_addr, {coordinates}], [mbar_addr], cache_hint;
```

**Key characteristics:**
- **Requires:** 128-byte TMA descriptor
- **Addressing:** Hardware computes address from descriptor + coordinates
- **Dimensions:** Supports 1D-5D tensors
- **Swizzle:** Built-in shared memory swizzle patterns
- **Typical use:** Large matrices (A, B in GEMM)

### Form 2: Descriptor-Less Bulk Copy TMA

**Purpose:** Simple 1D contiguous memory transfers without descriptor overhead.

**PTX instruction:**
```ptx
cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
    [smem_addr], [gmem_addr], bytes, [mbar_addr];
```

**Key characteristics:**
- **No descriptor needed:** Direct GMEM address provided
- **Addressing:** Software computes address explicitly
- **Dimensions:** 1D contiguous copy only
- **Swizzle:** Not supported
- **Typical use:** Small auxiliary data (scales, biases, metadata)

**CuTe wrapper:** `SM90_BULK_COPY_G2S`

**Code example** (`copy_sm90_tma.hpp:1365-1381`):
```cpp
struct SM90_BULK_COPY_G2S {
  CUTE_HOST_DEVICE static void
  copy(void const* gmem_ptr, uint64_t* mbar_ptr,
       void* smem_ptr, int32_t load_bytes)
  {
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
      :: "r"(smem_int_ptr), "l"(gmem_ptr), "r"(load_bytes), "r"(smem_int_mbar)
      : "memory");
  }
};
```

### Common Features (Both Forms)

Both forms share these TMA benefits:

| Feature | Descriptor-Based | Descriptor-Less |
|---------|-----------------|-----------------|
| **Asynchronous execution** | ✅ Yes | ✅ Yes |
| **Hardware acceleration** | ✅ Yes | ✅ Yes |
| **Mbarrier integration** | ✅ Yes | ✅ Yes |
| **Cluster-wide operations** | ✅ Yes (multicast) | ✅ Yes |
| **Frees compute threads** | ✅ Yes | ✅ Yes |
| **Pipeline friendly** | ✅ Yes | ✅ Yes |

### When to Use Each Form

**Use Descriptor-Based TMA when:**
- ✅ Transferring large tiles (> 1 KB)
- ✅ Multidimensional access patterns (2D matrices, 3D volumes)
- ✅ Need swizzle for bank conflict avoidance
- ✅ Multicast benefits justify descriptor overhead
- ✅ Examples: Weight matrices, activation tiles in GEMM

**Use Descriptor-Less Bulk Copy when:**
- ✅ Transferring small data (< 512 bytes)
- ✅ Simple 1D contiguous arrays
- ✅ Descriptor overhead (128 bytes) too high relative to data size
- ✅ Software address calculation is trivial
- ✅ Examples: Scale factors, zero points, bias vectors

**Real-world example:** In mixed-precision GEMM:
- A matrix (32 KB): Descriptor-based TMA → justify descriptor cost
- B matrix (2 KB): Descriptor-based TMA → justify descriptor cost
- Scale (128 bytes): Descriptor-less bulk copy → avoid descriptor overhead
- **All three use TMA!** Just different variants.

---

## TMA Descriptor Structure

### The 128-Byte Opaque Descriptor

A TMA descriptor (`cute::TmaDescriptor`) is a **128-byte opaque structure** created on the **CPU** and passed to GPU kernels.

**Type definition** (`include/cute/arch/copy_sm90_desc.hpp:293-298`):
```cpp
#if (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)
  using TmaDescriptor = CUtensorMap;              // CUDA driver type
  using Im2ColTmaDescriptor = CUtensorMap;
#else
  using TmaDescriptor = struct alignas(64) { char bytes[128]; };  // Opaque 128 bytes
  using Im2ColTmaDescriptor = struct alignas(64) { char bytes[128]; };
#endif
```

**Alignment requirement:** 64-byte aligned (hardware constraint).

### Descriptor Fields (Logical View)

While the descriptor is opaque, it logically encodes:

```
TMA Descriptor (128 bytes):
┌─────────────────────────────────────────────────────────────────────┐
│ Global Memory Properties                                            │
├─────────────────────────────────────────────────────────────────────┤
│ • Base address (64-bit pointer)          : Tensor start in GMEM    │
│ • Global dimensions (5× uint32)          : Shape [D0, D1, D2, D3, D4]│
│ • Global strides (4× uint64, in bytes)   : Strides [S1, S2, S3, S4]│
│   (S0 is implicitly 1 element)                                      │
├─────────────────────────────────────────────────────────────────────┤
│ Shared Memory Properties                                            │
├─────────────────────────────────────────────────────────────────────┤
│ • Box dimensions (5× uint32)             : Tile size per TMA op    │
│ • Box strides (5× uint32)                : Element traversal order │
│ • Swizzle mode (enum)                    : B32, B64, B128, or NONE │
│ • Swizzle base (enum)                    : Base alignment (16B-64B)│
├─────────────────────────────────────────────────────────────────────┤
│ Data Type and Metadata                                              │
├─────────────────────────────────────────────────────────────────────┤
│ • Element type (enum)                    : FP8, BF16, INT4, etc.   │
│ • Interleave mode (enum)                 : None or Im2Col          │
│ • L2 promotion hint (enum)               : Cache policy            │
│ • OOB fill mode (enum)                   : Out-of-bounds behavior  │
└─────────────────────────────────────────────────────────────────────┘
```

**Key constraints:**
- **Address**: Must be 16-byte aligned
- **Dimensions**: 1 ≤ size ≤ 2³² per dimension
- **Strides**: Must be multiple of 16 bytes (128 bits)
- **Box shape**: 1 ≤ box_dim ≤ 256 per dimension
- **Box stride**: 1 ≤ box_stride ≤ 8

### Descriptor Creation: `cuTensorMapEncodeTiled`

Descriptors are created on **CPU** using CUDA driver API:

**Function signature** (CUDA driver API):
```cpp
CUresult cuTensorMapEncodeTiled(
    CUtensorMap*            tensorMap,        // Output: 128-byte descriptor
    CUtensorMapDataType     tensorDataType,   // Element type
    cuuint32_t              tensorRank,       // 1-5 dimensions
    void*                   globalAddress,    // GMEM tensor base
    const cuuint64_t*       globalDim,        // [D0, D1, D2, D3, D4]
    const cuuint64_t*       globalStrides,    // [S1, S2, S3, S4] (S0 implicit)
    const cuuint32_t*       boxDim,           // Tile size per TMA op
    const cuuint32_t*       elementStrides,   // Element traversal order
    CUtensorMapInterleave   interleave,       // None or Im2Col
    CUtensorMapSwizzle      swizzle,          // Swizzle mode
    CUtensorMapL2promotion  l2Promotion,      // L2 cache hint
    CUtensorMapFloatOOBfill oobFill           // OOB behavior
);
```

**Example** (from `copy_traits_sm90_tma.hpp:1007-1019`):
```cpp
CUresult result = cuTensorMapEncodeTiled(
    &tma_desc,                              // Output descriptor
    TMA::to_CUtensorMapDataType<TmaInternalType>(),  // e.g., UINT8 for FP8
    tma_dim,                                // e.g., 2 for 2D tensor
    gmem_address,                           // ptr_A or ptr_B
    gmem_prob_shape.data(),                 // [M, K, 1, 1, 1]
    gmem_prob_stride.data() + 1,            // [stride_K, 0, 0, 0] (stride_M=1 implicit)
    smem_box_shape.data(),                  // [128, 32, 1, 1, 1] (tile MxK)
    smem_box_stride.data(),                 // [1, 1, 1, 1, 1]
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    TMA::to_CUtensorMapSwizzle(swizzle_bits, swizzle_base),  // e.g., 128B swizzle
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,     // Prefer L2 cache
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE       // Zero-fill out-of-bounds
);
```

**Constraints checked** (asserts in `make_tma_copy_desc`, lines 915-984):
- Address alignment (16B)
- Shape bounds (1 ≤ size ≤ 2³²)
- Stride alignment (multiples of 16B)
- Box dimension limits (≤ 256)
- Stride[0] must be 1 (contiguous innermost dimension)

---

## TMA PTX Instructions

### Load Operations: GMEM → SMEM

**Basic pattern:**
```ptx
cp.async.bulk.tensor.<dim>d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint
    [smem_addr], [gmem_desc, {coordinates}], [mbarrier_addr], cache_hint;
```

**Parameters:**
- `smem_addr`: Shared memory destination address
- `gmem_desc`: TMA descriptor address in GMEM (64-bit)
- `coordinates`: Tile coordinates (1-5 integers, depending on tensor rank)
- `mbarrier_addr`: Mbarrier for transaction tracking
- `cache_hint`: L2 cache eviction policy

**Dimensions:**

**1D Load** (`copy_sm90_tma.hpp:46-69`):
```cpp
struct SM90_TMA_LOAD_1D {
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void* smem_ptr, int32_t const& crd0)
  {
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);

    asm volatile (
      "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3}], [%2], %4;"
      :: "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
         "r"(crd0), "l"(cache_hint)
      : "memory");
  }
};
```

**2D Load** (line 92-114):
```ptx
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint
    [%0], [%1, {%3, %4}], [%2], %5;
```
Coordinates: `{crd0, crd1}` (e.g., M-tile index, K-tile index)

**3D, 4D, 5D Loads:** Similar pattern with 3, 4, or 5 coordinates.

### Multicast Loads

**Purpose:** Broadcast data to multiple CTAs in a cluster simultaneously.

**Pattern:**
```ptx
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint
    [smem_addr], [gmem_desc, {coordinates}], [mbarrier_addr], multicast_mask, cache_hint;
```

**Additional parameter:**
- `multicast_mask`: 16-bit mask indicating which CTAs in cluster receive data

**Example** (`copy_sm90_tma.hpp:602-615`):
```cpp
struct SM90_TMA_LOAD_MULTICAST_2D {
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void* smem_ptr, uint16_t multicast_mask,
       int32_t const& crd0, int32_t const& crd1)
  {
    asm volatile (
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5, %6;"
      :: "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
         "r"(crd0), "r"(crd1), "h"(multicast_mask), "l"(cache_hint)
      : "memory");
  }
};
```

**Benefit:** If 4 CTAs need the same data, multicast loads it once instead of 4 times → **4× bandwidth savings**.

### Store Operations: SMEM → GMEM

**Pattern:**
```ptx
cp.async.bulk.tensor.<dim>d.global.shared::cta.bulk_group
    [gmem_desc, {coordinates}], [smem_addr];
```

**Example** (`copy_sm90_tma.hpp:883-895`):
```cpp
struct SM90_TMA_STORE_1D {
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, void* smem_ptr, int32_t const& crd0)
  {
    asm volatile (
      "cp.async.bulk.tensor.1d.global.shared::cta.bulk_group [%0, {%2}], [%1];"
      :: "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0)
      : "memory");
  }
};
```

**Note:** Stores are less common than loads in GEMM kernels (epilogue often uses non-TMA stores for flexibility).

### Descriptor-Less Bulk Copy Operations

**Purpose:** Lightweight TMA for small 1D data without descriptor overhead.

#### Load: GMEM → SMEM

**PTX instruction:**
```ptx
cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
    [smem_addr], [gmem_addr], bytes, [mbar_addr];
```

**CuTe wrapper** (`copy_sm90_tma.hpp:1365-1381`):
```cpp
struct SM90_BULK_COPY_G2S {
  CUTE_HOST_DEVICE static void
  copy(void const* gmem_ptr,     // Global memory source
       uint64_t* mbar_ptr,       // Mbarrier for tracking
       void* smem_ptr,           // Shared memory destination
       int32_t load_bytes)       // Number of bytes to copy
  {
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
      :: "r"(smem_int_ptr), "l"(gmem_ptr), "r"(load_bytes), "r"(smem_int_mbar)
      : "memory");
  }
};
```

**Usage example** (from prescale collective, line 1018-1019):
```cpp
// Load scale factors (small 1D array)
constexpr int scale_load_bytes = BLK_M * sizeof(ElementScale);  // e.g., 128-256 bytes

auto* scale_gmem_addr = reinterpret_cast<void const*>(
    scale_ptr + m_coord * BLK_M + scale_load_k * scale_stride_k);
auto* scale_smem_addr = static_cast<void*>(&sS(0, 0, write_stage));

cute::SM90_BULK_COPY_G2S::copy(scale_gmem_addr,
                               reinterpret_cast<uint64_t*>(tma_barrier),
                               scale_smem_addr,
                               scale_load_bytes);
```

**Key benefits:**
- ✅ No 128-byte descriptor needed
- ✅ Still asynchronous and hardware-accelerated
- ✅ Integrates with mbarrier (same as descriptor-based TMA)
- ✅ Can share mbarrier with descriptor-based TMA loads
- ✅ Ideal for auxiliary data (scales, biases, < 512 bytes)

**Comparison with descriptor-based TMA:**

| Aspect | Descriptor-Based | Descriptor-Less Bulk Copy |
|--------|------------------|---------------------------|
| **Overhead** | 128-byte descriptor + creation | Zero (direct addressing) |
| **Addressing** | Hardware (descriptor + coords) | Software (explicit address) |
| **Data size sweet spot** | > 1 KB | < 512 bytes |
| **Dimensions** | 1D-5D | 1D only |
| **Async + mbarrier** | ✅ Yes | ✅ Yes |
| **Hardware accelerated** | ✅ Yes | ✅ Yes |

#### Store: SMEM → GMEM

**PTX instruction:**
```ptx
cp.async.bulk.global.shared::cta.bulk_group
    [gmem_addr], [smem_addr], bytes;
```

**Note:** Descriptor-less bulk stores exist but are less common in practice.

### Descriptor Modification PTX

**Why modify descriptors?**
- Grouped GEMM: Each group has different problem size and base address
- Batched operations: Different batch has different tensor pointer
- Dynamic shapes: Problem dimensions unknown at compile time

**Operations available:**

**1. Replace Global Address** (`copy_sm90_desc.hpp:327-339`):
```cpp
// Modify descriptor in GMEM
void tma_descriptor_replace_addr_in_global_mem(TmaDescriptor const* desc_ptr,
                                               void const* new_tensor_ptr)
{
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint64_t new_desc_addr = reinterpret_cast<uint64_t>(new_tensor_ptr);

  asm volatile (
    "tensormap.replace.tile.global_address.global.b1024.b64 [%0], %1;"
    :: "l"(gmem_int_desc), "l"(new_desc_addr));
}
```

**PTX:** `tensormap.replace.tile.global_address.{space}.b1024.b64`
- `{space}`: `.global` or `.shared::cta`
- Updates only the 64-bit base pointer field

**2. Replace Dimensions and Strides** (`copy_sm90_desc.hpp:359-418`):
```cpp
// Modify descriptor in shared memory (typical usage)
void tma_descriptor_replace_dims_strides_in_shared_mem(
    TmaDescriptor& smem_desc,
    cute::array<uint32_t, 5> const& prob_shape,
    cute::array<uint64_t, 5> const& prob_stride)
{
  uint32_t smem_int_desc = cast_smem_ptr_to_uint(&smem_desc);

  // Replace dimensions
  asm volatile (
    "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0, %1;"
    :: "l"(smem_int64_desc), "r"(prob_shape[0]));
  asm volatile (
    "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 1, %1;"
    :: "l"(smem_int64_desc), "r"(prob_shape[1]));
  // ... dimensions 2, 3, 4

  // Replace strides (in bytes, stride[0] is implicit)
  asm volatile (
    "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0, %1;"
    :: "l"(smem_int64_desc), "l"(prob_stride[1]));
  // ... strides 1, 2, 3 (indices map to stride[1-4])
}
```

**PTX patterns:**
- `tensormap.replace.tile.global_dim.{space}.b1024.b32 [desc], dim_idx, value;`
- `tensormap.replace.tile.global_stride.{space}.b1024.b64 [desc], stride_idx, value;`

**Stride encoding caveat:** Before CUDA 12.5, strides must be right-shifted by 4 bits (`stride >> 4`).

**3. Fence Operations**

After modifying a descriptor, **fence operations** ensure modifications are visible to TMA hardware:

**Copy-fence-release** (`copy_sm90_desc.hpp:424-437`):
```cpp
// Copy descriptor from smem to gmem with release fence
void tma_descriptor_cp_fence_release(TmaDescriptor const* gmem_desc_ptr,
                                     TmaDescriptor& smem_desc)
{
  asm volatile (
    "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned"
    " [%0], [%1], 128;"
    :: "l"(gmem_int_desc), "r"(smem_int_desc));
}
```

**Release fence (for GMEM modifications)** (`copy_sm90_desc.hpp:443-452`):
```cpp
void tma_descriptor_fence_release()
{
  asm volatile ("fence.proxy.tensormap::generic.release.gpu;");
}
```

**Acquire fence (before TMA load)** (`copy_sm90_desc.hpp:458-472`):
```cpp
void tma_descriptor_fence_acquire(TmaDescriptor const* desc_ptr)
{
  asm volatile (
    "fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"
    :: "l"(gmem_int_desc) : "memory");
}
```

**Fence semantics:**
- **Release**: Make descriptor modifications visible to TMA hardware
- **Acquire**: Ensure TMA sees the latest descriptor version before issuing load

---

## CuTe TMA Abstraction

CuTe wraps raw TMA PTX instructions in high-level C++ abstractions.

### `make_tma_copy`: Descriptor Creation

**Primary interface** (`copy_traits_sm90_tma.hpp:1267-1302`):
```cpp
template <class TmaType = void, class CopyOp, class GEngine, class GLayout,
          class SLayout, class TShape, class CTA_Tiler>
CUTE_HOST_RTC
auto
make_tma_copy(CopyOp        const& copy_op,      // SM90_TMA_LOAD or SM90_TMA_STORE
              Tensor<GEngine,GLayout> gtensor,   // GMEM tensor
              SLayout       const& slayout,      // SMEM layout
              TShape        const& cta_tile,     // CTA tile shape
              CTA_Tiler     const& cluster_size) // Cluster size
{
  // Calls detail::make_tma_copy_tiled<TmaType>(...)
  // Returns TiledCopy object with embedded TmaDescriptor
}
```

**Overloads:**
- `make_tma_copy(copy_op, gtensor, slayout)` - No CTA tiling (infers from slayout)
- `make_tma_copy(copy_op, gtensor, slayout, cta_tile)` - With CTA tile, cluster_size=1
- `make_tma_copy(copy_op, gtensor, slayout, cta_tile, cluster_size)` - Full specification

**Typical usage** (`sm90_mma_array_*_prescale.hpp:715-726`):
```cpp
Tensor tensor_a = make_tensor(ptr_A_first_batch,
                              detail::get_gmem_layout(make_shape(M, K, L), dA));

typename Params::TMA_A tma_load_a = make_tma_copy<TmaElementA>(
    GmemTiledCopyA{},                        // Copy operation (typically SM90_TMA_LOAD)
    tensor_a,                                // GMEM tensor
    SmemLayoutA{}(_,_,cute::Int<0>{}),       // SMEM layout
    make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),  // CTA tile (M, K)
    size<1>(ClusterShape{}));                // Cluster size along N dimension
```

**Return type:** `TiledCopy` object containing:
- `TmaDescriptor` (created via `cuTensorMapEncodeTiled`)
- Auxiliary parameters (coordinate mapping, swizzle info)
- `get_tma_descriptor()` method to access descriptor pointer

### Internal: `make_tma_copy_desc`

**Core function** (`copy_traits_sm90_tma.hpp:890-1079`):

**Steps:**

**1. Recast tensor for shape/stride inspection:**
```cpp
Tensor gtensor_T = recast<TmaInternalType>(gtensor);
void* gmem_address = raw_pointer_cast(gtensor_T.data());
```

**2. Fill GMEM shape and stride arrays:**
```cpp
cute::array<uint64_t, 5> gmem_prob_shape  = {1,1,1,1,1};
cute::array<uint64_t, 5> gmem_prob_stride = {0,0,0,0,0};

fill_tma_gmem_shape_stride(gtensor_T, stride(tma_gbasis), gmem_prob_shape, gmem_prob_stride);

// Convert strides to bytes
for(uint64_t& stride : gmem_prob_stride) {
  stride = (stride * sizeof_bits_v<TmaInternalType>) / 8;
}
```

**3. Compute SMEM box shape:**
```cpp
cute::array<uint32_t, 5> smem_box_shape  = {1,1,1,1,1};
for_each(make_seq<tma_dim>{}, [&](auto i) {
  smem_box_shape[i] *= size<i>(tma_gbasis);
});

// Adjust for multicast
for (uint32_t i = tma_dim-1, multicast = num_multicast; multicast > 1; --i) {
  smem_box_shape[i] = ceil_div(smem_box_shape[i], multicast);
  multicast = ceil_div(multicast, smem_box_shape[i]);
}
```

**4. Call `cuTensorMapEncodeTiled`:**
```cpp
CUresult result = cuTensorMapEncodeTiled(
    &tma_desc,
    tma_format,                           // Element type
    tma_dim,                              // 1-5
    gmem_address,
    gmem_prob_shape.data(),
    gmem_prob_stride.data() + 1,          // Skip stride[0] (implicit 1)
    smem_box_shape.data(),
    smem_box_stride.data(),
    tma_interleave,
    smem_swizzle,
    tma_l2Promotion,
    tma_oobFill);
```

**5. Return descriptor and auxiliary params:**
```cpp
return cute::make_tuple(tma_desc, AuxParams{gmem_tma_basis_stride});
```

### TiledCopy Object

**Usage in kernel:**
```cpp
// At kernel launch, tma_load_a contains the descriptor
typename Params::TMA_A tma_load_a = mainloop_params.tma_load_a;

// Get descriptor pointer
TmaDescriptor const* tma_desc_a = tma_load_a.get_tma_descriptor();

// Invoke TMA copy with .with() modifier
copy(tma_load_a.with(tma_desc_ptr, *tma_barrier, multicast_mask),
     src_tensor, dst_tensor);
```

**`.with()` method:** Binds descriptor pointer, mbarrier, and multicast mask to the copy operation.

---

## TMA Usage Workflow

### Standard TMA Pattern in CUTLASS Kernels

**Phase 1: Host-side Descriptor Creation**

```cpp
// 1. Create GMEM tensor
Tensor tensor_a = make_tensor(ptr_A, detail::get_gmem_layout(make_shape(M, K), stride_A));

// 2. Create TMA copy object (embeds descriptor)
auto tma_load_a = make_tma_copy(
    SM90_TMA_LOAD{},
    tensor_a,
    SmemLayoutA{}(_,_,Int<0>{}),
    make_shape(TileM, TileK),
    cluster_size);

// 3. Pass tma_load_a to kernel
```

**Phase 2: Kernel-side Initialization**

```cpp
__global__ void kernel(TMA_A tma_load_a, ...) {
  // 1. Get descriptor pointer
  TmaDescriptor const* tma_desc_a = tma_load_a.get_tma_descriptor();

  // 2. Optionally copy to shared memory (if modification needed)
  __shared__ TmaDescriptor smem_desc_a;
  if (threadIdx.x == 0) {
    smem_desc_a = *tma_desc_a;  // Copy to smem
  }
  __syncthreads();

  // 3. Modify if needed (e.g., for grouped GEMM)
  if (threadIdx.x == 0) {
    tma_descriptor_replace_addr_in_shared_mem(smem_desc_a, new_ptr);
    // Replace dims/strides if problem size changes
  }
  __syncthreads();

  // 4. Fence operations
  if (threadIdx.x == 0) {
    tma_descriptor_cp_fence_release(tma_desc_a, smem_desc_a);
  }
  __syncthreads();

  tma_descriptor_fence_acquire(tma_desc_a);
}
```

**Phase 3: TMA Load in Mainloop**

```cpp
// Producer warp issues TMA loads
if (warp_role == ProducerWarp) {
  // Initialize mbarrier
  uint64_t* mbar_ptr = &smem_barrier[stage];
  initialize_barrier(*mbar_ptr, 1);
  set_barrier_transaction_bytes(*mbar_ptr, transaction_bytes);

  // Compute tile coordinates
  int tile_m = blockIdx.y;
  int tile_k = k_iter;

  // Issue TMA load (elect one thread per warp)
  if (elect_one_sync()) {
    copy(tma_load_a.with(tma_desc_a, mbar_ptr, multicast_mask),
         make_coord(tile_m, tile_k),  // Coordinates
         make_tensor(smem_ptr, smem_layout));
  }
}
```

**Phase 4: Consumer Synchronization**

```cpp
// Consumer warp waits for data
if (warp_role == ConsumerWarp) {
  wait_barrier(smem_barrier[stage], phase_bit);

  // Data now available in shared memory
  // Proceed with WGMMA or other compute
}
```

### Multicast Pattern

**When to use:** Multiple CTAs in a cluster need the same data (e.g., weight matrix in GEMM).

**Multicast mask computation:**
```cpp
// For cluster shape (ClusterM, ClusterN, 1):
// If tile is along N dimension and all M-clusters need it:
uint16_t multicast_mask = (1 << ClusterM) - 1;  // e.g., 0b1111 for ClusterM=4
```

**Invocation:**
```cpp
copy(tma_load_b.with(tma_desc_b, mbar_ptr, multicast_mask),
     make_coord(tile_n, tile_k),
     smem_tensor);
```

**Result:** Data loaded once, visible to all CTAs in cluster → **Bandwidth savings = ClusterM**.

---

## Example: Using Both TMA Forms Together

**Scenario:** A mixed-precision GEMM kernel needs to load multiple types of data with different characteristics.

### Data Types and TMA Selection

| Data | Size | Dimensions | TMA Choice | Reason |
|------|------|------------|------------|--------|
| **Matrix A** | 32 KB | 2D (M×K) | Descriptor-based MULTICAST | Large, regular 2D access, multicast saves bandwidth |
| **Matrix B** | 2 KB | 2D (N×K) | Descriptor-based MULTICAST | Large, regular 2D access, multicast saves bandwidth |
| **Scale factors** | 128 bytes | 1D array | Descriptor-less bulk copy | Small, simple 1D, descriptor overhead too high |

### Code Pattern (Abstract)

**Descriptor-based TMA for large matrices:**
```cpp
// Host: Create TMA descriptor
auto tma_load_a = make_tma_copy(
    SM90_TMA_LOAD_MULTICAST{},
    gmem_tensor_a,                  // Shape: (M, K, L)
    smem_layout_a,                  // With swizzle
    make_shape(TileM, TileK),       // Box dimensions
    cluster_size_n);                // Multicast to N CTAs

// Kernel: Issue TMA load
copy(tma_load_a.with(tma_desc_ptr, mbar, mcast_mask),
     gmem_coords, smem_dest);
```

**Descriptor-less TMA for small auxiliary data:**
```cpp
// Kernel: Compute address and issue bulk copy
auto* scale_gmem = scale_base + m_offset + k_offset;
auto* scale_smem = &smem_scale[stage];

SM90_BULK_COPY_G2S::copy(
    scale_gmem,      // Direct GMEM address
    mbar,            // Same mbarrier as matrices!
    scale_smem,      // SMEM destination
    128);            // Bytes
```

**Unified synchronization:**
```cpp
// Set total expected bytes (all TMA ops combined)
set_barrier_transaction_bytes(mbar, 
    sizeof_A + sizeof_B + sizeof_scale);

// Issue all TMA operations (parallel execution)
copy(tma_load_a.with(...), ...);           // Descriptor-based
copy(tma_load_b.with(...), ...);           // Descriptor-based
SM90_BULK_COPY_G2S::copy(...);             // Descriptor-less

// Consumer waits once for all data
wait_barrier(mbar, phase_bit);  // All TMA complete
```

### Key Takeaways from Example

1. **Coexistence:** Both TMA forms used in same kernel, same pipeline
2. **Selection criteria:** Data size and complexity determine TMA type
3. **Unified mbarrier:** All TMA operations contribute to single synchronization point
4. **Optimal trade-off:** Use complexity (descriptor) where it matters (A, B), simplicity (direct addressing) where it doesn't (scale)

**For detailed code-level analysis of a real implementation:**
- See: `.claude/knowledge/modules/tma-usage-in-mixed-precision-gemm.md`

---

## Advanced Topics

### TMA and Warp Specialization

**Pattern:** Different warps have different roles.

**Producer warp:**
- Issues TMA loads
- Manages pipeline stages
- Advances tile iterators

**Consumer warps:**
- Wait on mbarriers
- Perform WGMMA compute
- Consume data from shared memory

**Benefit:** TMA operates independently of compute → **high overlap**.

### TMA Transaction Bytes

**Purpose:** Tell mbarrier how many bytes to expect.

**Calculation:**
```cpp
constexpr int TMA_transaction_bytes =
    size<0>(cta_tile) * size<2>(cta_tile) * sizeof(ElementA);  // M × K × element_size
```

**Usage:**
```cpp
set_barrier_transaction_bytes(smem_barrier, TMA_transaction_bytes);
```

**Effect:** Mbarrier tracks TMA progress, flips phase bit when all bytes arrive.

### Swizzle Modes

**Purpose:** Avoid shared memory bank conflicts.

**Common patterns:**
- `Swizzle<3,3,3>`: 128B swizzle (typical for mixed-precision GEMM)
- `Swizzle<2,3,3>`: 64B swizzle
- `Swizzle<1,3,3>`: 32B swizzle
- `Swizzle<0,0,0>`: No swizzle (bank conflicts possible)

**Encoding in descriptor:**
```cpp
TMA::SmemSwizzleBits swizzle_bits = get_tma_swizzle_bits(swizzle);  // e.g., B128
TMA::SmemSwizzleBase swizzle_base = get_tma_swizzle_base(swizzle);  // e.g., SWIZZLE_BASE_16B
CUtensorMapSwizzle smem_swizzle = TMA::to_CUtensorMapSwizzle(swizzle_bits, swizzle_base);
```

**Effect:** TMA hardware automatically applies swizzle pattern when writing to shared memory → **optimal bank access**.

### Im2Col TMA

**Purpose:** Efficiently load image patches for convolution.

**Descriptor type:** `Im2ColTmaDescriptor` (same size as `TmaDescriptor`)

**PTX suffix:** `im2col` in instruction name
```ptx
cp.async.bulk.tensor.3d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes
```

**Use case:** Convolution kernels where sliding window access pattern is needed.

### L2 Promotion and Cache Hints

**L2 Promotion** (descriptor field):
- `CU_TENSOR_MAP_L2_PROMOTION_L2_128B`: Prefer 128B L2 cache lines
- `CU_TENSOR_MAP_L2_PROMOTION_L2_256B`: Prefer 256B L2 cache lines
- `CU_TENSOR_MAP_L2_PROMOTION_NONE`: No preference

**Cache hints** (per-load parameter):
- `CacheHintSm90::EVICT_NORMAL`: Normal eviction policy
- `CacheHintSm90::EVICT_FIRST`: Evict after first use (streaming data)
- `CacheHintSm90::EVICT_LAST`: Keep in cache (reused data)

**Usage:**
```cpp
uint64_t cache_hint = static_cast<uint64_t>(TMA::CacheHintSm90::EVICT_FIRST);
copy(tma_load_a.with(tma_desc_a, mbar_ptr, multicast_mask, cache_hint), ...);
```

---

## Code References

### Key Files

**TMA PTX wrappers:**
- `include/cute/arch/copy_sm90_tma.hpp` (1422 lines)
  - **Descriptor-based TMA:**
    - `SM90_TMA_LOAD_{1D,2D,3D,4D,5D}`: Load operations (lines 46-275)
    - `SM90_TMA_LOAD_MULTICAST_{1D,2D,3D,4D,5D}`: Multicast loads (lines 576-838)
    - `SM90_TMA_STORE_{1D,2D,3D,4D,5D}`: Store operations (lines 883-996)
    - `SM90_TMA_LOAD_IM2COL_*`: Im2Col loads (lines 354-505)
  - **Descriptor-less bulk copy TMA:**
    - `SM90_BULK_COPY_G2S`: GMEM→SMEM bulk copy (lines 1365-1381)
    - Uses PTX: `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes`

**TMA descriptor and utilities:**
- `include/cute/arch/copy_sm90_desc.hpp` (477 lines)
  - `TmaDescriptor` type definition (lines 293-298)
  - Descriptor modification functions:
    - `tma_descriptor_replace_addr_in_global_mem()` (lines 327-339)
    - `tma_descriptor_replace_addr_in_shared_mem()` (lines 343-356)
    - `tma_descriptor_replace_dims_strides_in_shared_mem()` (lines 359-418)
  - Fence operations:
    - `tma_descriptor_cp_fence_release()` (lines 424-437)
    - `tma_descriptor_fence_release()` (lines 443-452)
    - `tma_descriptor_fence_acquire()` (lines 458-472)
  - Mbarrier operations:
    - `initialize_barrier()` (lines 62-73)
    - `set_barrier_transaction_bytes()` (lines 76-87)
    - `wait_barrier()` (lines 90-110)
    - `arrive_barrier()` (lines 113-126)

**TMA abstraction (CuTe):**
- `include/cute/atom/copy_traits_sm90_tma.hpp` (1546 lines)
  - `make_tma_copy()` overloads (lines 1267-1316)
  - `make_tma_copy_desc()` - core descriptor creation (lines 890-1079)
  - `fill_tma_gmem_shape_stride()` - helper for shape/stride computation
  - TMA data type mappings (lines 204-236)

**Usage in CUTLASS kernels:**
- `include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp`
  - Host-side descriptor creation (line ~412-420)
  - Kernel-side TMA copy invocation (mainloop `load()` function)
  - Descriptor-based TMA for A/B matrices
- `include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_prescale.hpp`
  - Initialization kernel with per-group descriptors (lines 107-191)
  - Descriptor-based TMA for A/B matrices (lines 1004-1005)
  - **Descriptor-less bulk copy TMA for scale/zero** (lines 1018-1029)
    - `SM90_BULK_COPY_G2S::copy()` for small auxiliary data
  - Pre-baking optimization (see `.claude/knowledge/optimization/tma-descriptor-prebaking.md`)

### Related CUDA Documentation

**CUDA Toolkit Programming Guide:**
- Section 11.1.4: Asynchronous Data Copies using TMA
- PTX ISA: Section 9.7.13 (Tensor Memory Accelerator operations)

**Driver API:**
- `cuTensorMapEncodeTiled`: Descriptor creation
- `cuTensorMapEncodeIm2col`: Im2Col descriptor creation

---

## Debugging TMA Issues

### Common Errors

**1. "TMA descriptor creation failed"**
- **Cause:** Invalid shape/stride parameters
- **Fix:** Check assertions in `make_tma_copy_desc` (lines 915-984)
  - Address must be 16B-aligned
  - Strides must be multiples of 16B
  - Dimensions must be ≤ 2³²

**2. "TMA load hangs (mbarrier never completes)"**
- **Cause:** Transaction bytes mismatch or missing fence
- **Fix:**
  - Verify `set_barrier_transaction_bytes()` matches actual transfer size
  - Ensure `tma_descriptor_fence_acquire()` called before load
  - Check multicast mask if using cluster

**3. "Incorrect data loaded"**
- **Cause:** Descriptor not updated correctly for new problem
- **Fix:**
  - For grouped GEMM, ensure descriptor updated per group
  - Check fence operations after descriptor modification
  - Use `print_tma_descriptor()` helper to debug (in prescale file, line 85-93)

**4. "Shared memory bank conflicts despite swizzle"**
- **Cause:** Swizzle mode mismatch between descriptor and smem layout
- **Fix:** Ensure `SmemLayoutA` swizzle matches TMA descriptor swizzle

### Debug Helpers

**Print TMA descriptor** (from prescale file):
```cpp
CUTE_DEVICE void print_tma_descriptor(const char* label, TmaDescriptor const& desc) {
  auto const* raw = reinterpret_cast<uint32_t const*>(&desc);
  printf("%s:\n", label);
  for (int i = 0; i < 128 / 4; i += 4) {
    printf("[%3d]: %08x %08x %08x %08x\n", i*4, raw[i+3], raw[i+2], raw[i+1], raw[i]);
  }
}
```

**Verify descriptor fields:**
```cpp
// Use during development to check descriptor contents match expectations
if (threadIdx.x == 0 && blockIdx.x == 0) {
  print_tma_descriptor("TMA Desc A", *tma_desc_a);
}
```

---

## Performance Considerations

### TMA Overhead

**Descriptor creation:** CPU-side, one-time cost (~1-10 μs)
- Negligible for kernels launched many times
- Significant if creating descriptors per-launch with varying shapes

**Descriptor modification:** Device-side, per-group/batch cost (~50-100 cycles)
- See optimization in `.claude/knowledge/optimization/tma-descriptor-prebaking.md`

**TMA latency:** ~200-300 cycles for first byte arrival
- Amortized over large tiles (128×128 or larger)
- Overlap with compute using warp specialization

### Bandwidth Efficiency

**Without TMA:**
- Manual pointer arithmetic and `ldgsts` instructions
- Each CTA loads its own copy

**With TMA:**
- Hardware handles addressing
- Multicast reduces redundant loads by ClusterSize factor

**Example:**
- Cluster = 2×2 = 4 CTAs
- Without multicast: 4× bandwidth cost
- With multicast: 1× bandwidth cost → **4× improvement**

### When TMA is Not Optimal

- **Very small tiles** (<16 elements): Overhead dominates
- **Irregular access patterns**: TMA requires regular strides
- **Dynamic shapes per-warp**: TMA is per-CTA, not per-warp
- **High descriptor update frequency**: Consider pre-baking (see optimization doc)

---

## Related Knowledge

**Dependencies** (read first):
- `.claude/knowledge/modules/hopper-mixed-precision-gemm-overview.md` - Overall kernel context
- CuTe layout algebra basics (shapes, strides, tensors)
- CUDA mbarriers and asynchronous copy concepts

**Related optimizations:**
- `.claude/knowledge/optimization/tma-descriptor-prebaking.md` - Pre-creating descriptors for grouped GEMM
  - **Note:** This optimization applies only to **descriptor-based TMA**, not descriptor-less bulk copy
- `.claude/knowledge/architecture/swapab-pattern.md` - Dimension mapping affects TMA descriptor setup

**Advanced topics** (future documentation):
- TMA descriptor caching across kernel launches
- TMA performance tuning (box size, multicast patterns)
- Detailed SASS analysis of TMA instructions
- TMA in multi-GPU scenarios (NVLink, multi-node)

---

**Last Updated:** 2026-03-18 (Revised: corrected SM90_BULK_COPY_G2S as descriptor-less TMA)
**Documented by:** Claude Code (knowledge management system)
**Next steps:** Read optimization document for pre-baking technique (applies to descriptor-based TMA only)

---

## Key Takeaways

1. **TMA has two forms:** Descriptor-based (for large multidimensional tensors) and descriptor-less bulk copy (for small 1D data)
2. **Both forms are hardware-accelerated TMA** - asynchronous, mbarrier-integrated, cluster-aware
3. **Choose based on data characteristics:** Large/complex → descriptor-based; Small/simple → descriptor-less
4. **Real kernels use both:** e.g., descriptor-based for A/B matrices, descriptor-less for scales/biases
5. **Pre-baking optimization** only applies to descriptor-based TMA (where descriptor overhead matters)

