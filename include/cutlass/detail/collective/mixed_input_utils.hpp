/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"

#include "cute/util/type_traits.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

// The universal converter
template <
  class SrcType,
  class DstType,
  class LayoutIn,
  class LayoutOut
>
struct LayoutAwareConvertImpl {
  template<class EngineIn, class EngineOut>
  CUTLASS_DEVICE
  static void convert(
    cute::Tensor<EngineIn, LayoutIn> const& src,
    cute::Tensor<EngineOut, LayoutOut>    & dst) {

    static_assert(cute::is_same_v<SrcType, typename EngineIn::value_type> &&
                  cute::is_same_v<DstType, typename EngineOut::value_type>);
    static_assert(cute::cosize_v<LayoutIn> == cute::cosize_v<LayoutOut>);
    constexpr int N = decltype(cute::max_common_vector(LayoutIn{}, LayoutOut{})){};
    using SrcArray = cutlass::Array<SrcType, N>;
    using DstArray = cutlass::Array<DstType, N>;
    using Converter = cutlass::NumericArrayConverter<DstType,
                                                     SrcType,
                                                     N,
                                                     cutlass::FloatRoundStyle::round_to_nearest>;
    auto&& src_vm = cute::recast<SrcArray>(src);
    auto&& dst_vm = cute::recast<DstArray>(dst);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < src_vm.size(); ++i) {
      dst_vm(i) = Converter::convert(src_vm(i));
    }
  }
};

// Specialization for INT4 -> BF16 with [02461357] value order
template <>
struct LayoutAwareConvertImpl<
  cutlass::int4b_t,
  cutlass::bfloat16_t,
  cute::Layout<cute::Shape<_2,_4>, cute::Stride<_4,_1>>,
  cute::Layout<_8>
> {
  template<class EngineIn, class EngineOut>
  CUTLASS_DEVICE
  static void convert(
    cute::Tensor<EngineIn,
                 cute::Layout<cute::Shape<_2,_4>, cute::Stride<_4,_1>>
                > const& src, 
    cute::Tensor<EngineOut,
                 cute::Layout<_8>
                >& dst) {

    static_assert(cute::is_same_v<cutlass::int4b_t, typename EngineIn::value_type> &&
                  cute::is_same_v<cutlass::bfloat16_t, typename EngineOut::value_type>);
    using SrcArray = cutlass::Array<cutlass::int4b_t, 8>;
    using DstArray = cutlass::Array<cutlass::bfloat16_t, 8>;
    using RegArray = cutlass::AlignedArray<uint32_t, 4, sizeof(DstArray)>;

    auto&& src_reg = cute::recast<uint32_t>(src)(0);
    auto&& r       = cute::recast<RegArray>(dst)(0);
    CUTLASS_PRAGMA_UNROLL
    for (size_t ii = 0; ii < RegArray::kElements; ++ii) {
      r[ii] = src_reg >> (4 * (ii));
      static constexpr uint32_t xor_mask = 0x43084308;
      static constexpr uint32_t lo_mask  = 0x000F000F;
      static constexpr uint32_t immLut   = (0xf0 & 0xcc) ^ 0xaa;
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii])
          : "n"(lo_mask), "n"(xor_mask), "n"(immLut));
      static constexpr uint32_t lo_bias = xor_mask; // 0x43084308, {136, 136}
      {
        __nv_bfloat162& bf16x2_val = reinterpret_cast<__nv_bfloat162&>(r[ii]);
        bf16x2_val = __hsub2(bf16x2_val,
                              reinterpret_cast<const __nv_bfloat162&>(lo_bias));
      }
    }
  }
};

// Specialization for UINT4 -> BF16 with [02461357] value order
template <>
struct LayoutAwareConvertImpl<
  cutlass::uint4b_t,
  cutlass::bfloat16_t,
  cute::Layout<cute::Shape<_2,_4>, cute::Stride<_4,_1>>,
  cute::Layout<_8>
> {
  template<class EngineIn, class EngineOut>
  CUTLASS_DEVICE
  static void convert(
    cute::Tensor<EngineIn,
                cute::Layout<cute::Shape<_2,_4>, cute::Stride<_4,_1>>
                > const& src, 
    cute::Tensor<EngineOut,
                 cute::Layout<_8>
                >& dst) {

    static_assert(cute::is_same_v<cutlass::uint4b_t, typename EngineIn::value_type> &&
                  cute::is_same_v<cutlass::bfloat16_t, typename EngineOut::value_type>);
    using SrcArray = cutlass::Array<cutlass::uint4b_t, 8>;
    using DstArray = cutlass::Array<cutlass::bfloat16_t, 8>;
    using RegArray = cutlass::AlignedArray<uint32_t, 4, sizeof(DstArray)>;

    auto&& src_reg = cute::recast<uint32_t>(src)(0);
    auto&& r       = cute::recast<RegArray>(dst)(0);
    CUTLASS_PRAGMA_UNROLL
    for (size_t ii = 0; ii < RegArray::kElements; ++ii) {
      r[ii] = src_reg >> (4 * (ii));
      static constexpr uint32_t or_mask = 0x43004300;
      static constexpr uint32_t lo_mask = 0x000F000F;
      static constexpr uint32_t immLut  = (0xf0 & 0xcc) | 0xaa;
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii])
          : "n"(lo_mask), "n"(or_mask), "n"(immLut));
      static constexpr uint32_t lo_bias = or_mask; // 0x43004300, {128, 128}
      {
        __nv_bfloat162& bf16x2_val = reinterpret_cast<__nv_bfloat162&>(r[ii]);
        bf16x2_val = __hsub2(bf16x2_val,
                             reinterpret_cast<const __nv_bfloat162&>(lo_bias));
      }
    }
  }
};

// Specialization for INT4 -> FP16 with [02461357] value order
template <>
struct LayoutAwareConvertImpl<
  cutlass::int4b_t,
  cutlass::half_t,
  cute::Layout<cute::Shape<_2,_4>, cute::Stride<_4,_1>>,
  cute::Layout<_8>
> {
  template<class EngineIn, class EngineOut>
  CUTLASS_DEVICE
  static void convert(
    cute::Tensor<EngineIn,
                cute::Layout<cute::Shape<_2,_4>, cute::Stride<_4,_1>>
                > const& src, 
    cute::Tensor<EngineOut,
                cute::Layout<_8>
                >& dst) {

    static_assert(cute::is_same_v<cutlass::int4b_t, typename EngineIn::value_type> &&
                  cute::is_same_v<cutlass::half_t, typename EngineOut::value_type>);
    using SrcArray = cutlass::Array<cutlass::int4b_t, 8>;
    using DstArray = cutlass::Array<cutlass::half_t, 8>;
    using RegArray = cutlass::AlignedArray<uint32_t, 4, sizeof(DstArray)>;

    auto&& src_reg = cute::recast<uint32_t>(src)(0);
    auto&& r       = cute::recast<RegArray>(dst)(0);
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ii += 2) {
      auto src_ = src_reg >> (4 * (ii));
      r[ii + 0] = src_;
      r[ii + 1] = src_;
      static constexpr uint32_t lo_xor_mask = 0x64086408;
      static constexpr uint32_t hi_xor_mask = 0x64806480;
      static constexpr uint32_t lo_mask     = 0x000F000F;
      static constexpr uint32_t hi_mask     = 0x00F000F0;
      static constexpr uint32_t immLut      = (0xf0 & 0xcc) ^ 0xaa;
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii + 0])
          : "n"(lo_mask), "n"(lo_xor_mask), "n"(immLut));
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii + 1])
          : "n"(hi_mask), "n"(hi_xor_mask), "n"(immLut));
      static constexpr uint32_t lo_bias  = 0x64086408; // {1032, 1032}
      static constexpr uint32_t hi_bias  = 0xD480D480; // {-72, -72}
      static constexpr uint32_t hi_scale = 0x2C002C00; // {1/16, 1/16}
      {
        half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii + 0]);
        fp16x2_val = __hsub2(fp16x2_val,
                             reinterpret_cast<const half2&>(lo_bias));
      }
      {
        half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii + 1]);
        fp16x2_val = __hfma2(fp16x2_val,
                              reinterpret_cast<const half2&>(hi_scale),
                              reinterpret_cast<const half2&>(hi_bias));
      }
    }
  }
};

// Specialization for UINT4 -> FP16 with [02461357] value order
template <>
struct LayoutAwareConvertImpl<
  cutlass::uint4b_t,
  cutlass::half_t,
  cute::Layout<cute::Shape<_2,_4>, cute::Stride<_4,_1>>,
  cute::Layout<_8>
> {
  template<class EngineIn, class EngineOut>
  CUTLASS_DEVICE
  static void convert(
    cute::Tensor<EngineIn,
                cute::Layout<cute::Shape<_2,_4>, cute::Stride<_4,_1>>
                > const& src, 
    cute::Tensor<EngineOut,
                cute::Layout<_8>
                >& dst) {

    static_assert(cute::is_same_v<cutlass::uint4b_t, typename EngineIn::value_type> &&
                  cute::is_same_v<cutlass::half_t, typename EngineOut::value_type>);
    using SrcArray = cutlass::Array<cutlass::uint4b_t, 8>;
    using DstArray = cutlass::Array<cutlass::half_t, 8>;
    using RegArray = cutlass::AlignedArray<uint32_t, 4, sizeof(DstArray)>;

    auto&& src_reg = cute::recast<uint32_t>(src)(0);
    auto&& r       = cute::recast<RegArray>(dst)(0);
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ii += 2) {
      auto src_ = src_reg >> (4 * (ii));
      r[ii + 0] = src_;
      r[ii + 1] = src_;
      static constexpr uint32_t or_mask = 0x64006400;
      static constexpr uint32_t lo_mask = 0x000F000F;
      static constexpr uint32_t hi_mask = 0x00F000F0;
      static constexpr uint32_t immLut  = (0xf0 & 0xcc) | 0xaa;
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii])
          : "n"(lo_mask), "n"(or_mask), "n"(immLut));
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii + 1])
          : "n"(hi_mask), "n"(or_mask), "n"(immLut));
      static constexpr uint32_t lo_bias  = or_mask;    // 0x64006400, {1024, 1024}
      static constexpr uint32_t hi_bias  = 0xD400D400; // {-64, -64}
      static constexpr uint32_t hi_scale = 0x2C002C00; // {1/16, 1/16}
      {
        half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii + 0]);
        fp16x2_val = __hsub2(fp16x2_val,
                             reinterpret_cast<const half2&>(lo_bias));
      }
      {
        half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii + 1]);
        fp16x2_val = __hfma2(fp16x2_val,
                             reinterpret_cast<const half2&>(hi_scale),
                             reinterpret_cast<const half2&>(hi_bias));
      }
    }
  }
};
/*
// Specialization for E5M2 -> FP16 with [3120] value order
template <>
struct LayoutAwareConvertImpl<
  cutlass::float_e5m2_t,
  cutlass::half_t,
  cute::Layout<cute::Shape<_2,_2>, cute::Stride<_2,_1>>,
  cute::Layout<_4>
> {
  template<class EngineIn, class EngineOut>
  CUTLASS_DEVICE
  static void convert(
    cute::Tensor<EngineIn,
                cute::Layout<cute::Shape<_2,_2>, cute::Stride<_2,_1>>
                > const& src,
    cute::Tensor<EngineOut,
                cute::Layout<_4>
                >& dst) {

    static_assert(cute::is_same_v<cutlass::float_e5m2_t, typename EngineIn::value_type> &&
                  cute::is_same_v<cutlass::half_t, typename EngineOut::value_type>);
    using SrcArray = cutlass::Array<cutlass::float_e5m2_t, 8>;
    using DstArray = cutlass::Array<cutlass::half_t, 8>;
    using RegArray = cutlass::AlignedArray<uint32_t, 4, sizeof(DstArray)>;

    auto&& src_reg = cute::recast<uint32_t>(src)(0);
    auto&& r       = cute::recast<RegArray>(dst)(0);
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      // in registers: a3, a1, a2, a0
      r[RegArray::kElements - ii - 1] = src_reg << (8 * (ii));

      static constexpr uint32_t and_mask = 0xFF00FF00;
      asm volatile(
          "{\n"
          "  and.b32 %0, %0, %1;\n"
          "}\n"
          : "+r"(r[ii])
          : "n"(and_mask));
    }
  }
};
*/
// Specialization for INT8 -> BF16 with [3120] value order
template <>
struct LayoutAwareConvertImpl<
  cutlass::int8_t,
  cutlass::bfloat16_t,
  cute::Layout<cute::Shape<_2,_2>, cute::Stride<_2,_1>>,
  cute::Layout<_4>
> {
  template<class EngineIn, class EngineOut>
  CUTLASS_DEVICE
  static void convert(
    cute::Tensor<EngineIn,
                cute::Layout<cute::Shape<_2,_2>, cute::Stride<_2,_1>>
                > const& src,
    cute::Tensor<EngineOut,
                cute::Layout<_4>
                >& dst) {

    static_assert(cute::is_same_v<cutlass::int8_t, typename EngineIn::value_type> &&
                  cute::is_same_v<cutlass::bfloat16_t, typename EngineOut::value_type>);
    using SrcArray = cutlass::Array<cutlass::int8_t, 8>;
    using DstArray = cutlass::Array<cutlass::bfloat16_t, 8>;
    using RegArray = cutlass::AlignedArray<uint32_t, 4, sizeof(DstArray)>;

    auto&& src_reg = cute::recast<uint32_t>(src)(0);
    auto&& r       = cute::recast<RegArray>(dst)(0);
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      uint32_t tmp0, tmp1;
      r[ii] = src_reg >> (8 * (ii));
      static constexpr uint32_t or_mask    = 0x43004300;
      static constexpr uint32_t and_mask_0 = 0x007F007F;
      static constexpr uint32_t and_mask_1 = 0x00800080;
      static constexpr uint32_t immLut     = (0xf0 & 0xcc) | 0xaa;
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %1, %2, %3, %4;\n"
          "}\n"
          : "=r"(tmp0)
          : "r"(r[ii]), "n"(and_mask_0), "n"(or_mask), "n"(immLut));
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %1, %2, %3, %4;\n"
          "}\n"
          : "=r"(tmp1)
          : "r"(r[ii]), "n"(and_mask_1), "n"(or_mask), "n"(immLut));
      {
        __nv_bfloat162& bf16x2_val = reinterpret_cast<__nv_bfloat162&>(r[ii]);
        bf16x2_val = __hsub2(reinterpret_cast<__nv_bfloat162 const&>(tmp0),
                             reinterpret_cast<__nv_bfloat162 const&>(tmp1));
      }
    }
  }
};

// Specialization for INT8 -> FP16 with [3120] value order
template <>
struct LayoutAwareConvertImpl<
  cutlass::int8_t,
  cutlass::half_t,
  cute::Layout<cute::Shape<_2,_2>, cute::Stride<_2,_1>>,
  cute::Layout<_4>
> {
  template<class EngineIn, class EngineOut>
  CUTLASS_DEVICE
  static void convert(
    cute::Tensor<EngineIn,
                cute::Layout<cute::Shape<_2,_2>, cute::Stride<_2,_1>>
                > const& src,
    cute::Tensor<EngineOut,
                cute::Layout<_4>
                >& dst) {

    static_assert(cute::is_same_v<cutlass::int8_t, typename EngineIn::value_type> &&
                  cute::is_same_v<cutlass::half_t, typename EngineOut::value_type>);
    using SrcArray = cutlass::Array<cutlass::int8_t, 8>;
    using DstArray = cutlass::Array<cutlass::half_t, 8>;
    using RegArray = cutlass::AlignedArray<uint32_t, 4, sizeof(DstArray)>;

    auto&& src_reg = cute::recast<uint32_t>(src)(0);
    auto&& r       = cute::recast<RegArray>(dst)(0);
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      r[ii] = src_reg >> (8 * (ii));
      static constexpr uint32_t xor_mask = 0x64806480;
      static constexpr uint32_t and_mask = 0x00FF00FF;
      static constexpr uint32_t immLut   = (0xf0 & 0xcc) ^ 0xaa;
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii])
          : "n"(and_mask), "n"(xor_mask), "n"(immLut));
      {
        static constexpr uint32_t bias = 0x64806480;
        __half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii]);
        fp16x2_val = __hsub2(fp16x2_val,
                             reinterpret_cast<__half2 const&>(bias));
      }
    }
  }
};

template <
  class EngineIn,
  class EngineOut,
  class LayoutIn,
  class LayoutOut
>
CUTLASS_DEVICE
void LayoutAwareConvert( // Accept mutable temporaries
  cute::Tensor<EngineIn, LayoutIn>   const& src,
  cute::Tensor<EngineOut, LayoutOut>     && dst) {

  LayoutAwareConvert(src, dst);
}
template <
  class EngineIn,
  class EngineOut,
  class LayoutIn,
  class LayoutOut
>
CUTLASS_DEVICE
void LayoutAwareConvert(
  cute::Tensor<EngineIn, LayoutIn>   const& src,
  cute::Tensor<EngineOut, LayoutOut>      & dst) {

  using SrcType = typename EngineIn::value_type;
  using DstType = typename EngineOut::value_type;
  Tensor src_vm = coalesce(src);
  Tensor dst_vm = coalesce(dst);
  Layout src_layout = src_vm.layout();
  Layout dst_layout = dst_vm.layout();
  LayoutAwareConvertImpl<SrcType, 
                         DstType,
                         decltype(src_layout),
                         decltype(dst_layout)>::convert(src_vm, dst_vm);
}

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective::detail {

template <class PointerType>
static constexpr
CUTLASS_HOST_DEVICE
auto get_logical_ptr(PointerType const* ptr) {
  if constexpr (cute::sizeof_bits_v<PointerType> < 8) {
    return subbyte_iterator<PointerType const>(ptr);
  }
  else {  
    return ptr;
  }
}
template<int Stages, class LayoutAtom, class TileShape, class Stride>
static constexpr
CUTLASS_HOST_DEVICE
auto get_smem_layout(LayoutAtom layout_atom, TileShape const& tile_shape, Stride const& stride) {
  if constexpr (not cute::is_layout<Stride>::value) {
    return tile_to_shape(
      layout_atom,
      append(tile_shape, Int<Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,Stride>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{});
  }
  else {
    auto gmem_tile = composition(stride, tile_shape);
    return make_layout_like(append(gmem_tile, make_layout(Int<Stages>{}, 0)));
  }
}
template<class Shape, class Stride>
static constexpr
CUTLASS_HOST_DEVICE
auto get_gmem_layout(Shape const& shape, Stride const& stride) {
  if constexpr (not cute::is_layout<Stride>::value) {
    return make_layout(shape, stride);
  }
  else {
    return stride;
  }
}


typedef uint32_t            __nv_fp4x8_storage_t;
typedef uint32_t            __nv_bf16x2_storage_t;
typedef uint32_t            __nv_int4x8_storage_t;
typedef uint64_t            __nv_fp8x8_storage_t;
typedef cutlass::uint128_t  __nv_bf16x8_storage_t;


// -----------------------------------------------------------------------
// Interleaved version of the bits of four consecutive fp4 values (i.e. 16-bits):
//     s000000eem000000         (1st fp4)
//        s000000eem000000      (2nd fp4)
//           s000000eem000000   (3rd fp4)
//     0sm0ee0000000000         (4th fp4)
// -----------------------------------------------------------------------

__device__ __inline__
__nv_bf16x8_storage_t
psx_cvt_triton_fp4x8_to_bf16x8_interleaved
(
    const __nv_fp4x8_storage_t fp4x8
)
{
  __nv_bf16x8_storage_t bf16x8_raw;
  __nv_bfloat162 *bf16x2_raw = reinterpret_cast<__nv_bfloat162 *>(&bf16x8_raw);

  // 0x7e807e80 -> BF16 [126, 126]
  uint32_t bias_raw = 0x7e807e80U;
  __nv_bfloat162 bias = reinterpret_cast<__nv_bfloat162&>(bias_raw);

  __nv_fp4x8_storage_t first_fp4 = fp4x8 & 0x81C081C0U;
  bf16x2_raw[0] = __hmul2(reinterpret_cast<__nv_bfloat162&>(first_fp4), bias);

  __nv_fp4x8_storage_t second_fp4 = (fp4x8 << 3) & 0x81C081C0U;
  bf16x2_raw[1] = __hmul2(reinterpret_cast<__nv_bfloat162&>(second_fp4), bias);

  __nv_fp4x8_storage_t third_fp4 = (fp4x8 << 6) & 0x81C081C0U;
  bf16x2_raw[2] = __hmul2(reinterpret_cast<__nv_bfloat162&>(third_fp4), bias);

  __nv_fp4x8_storage_t fourth_fp4;
  __nv_fp4x8_storage_t fourth_fp4_s = (fp4x8 << 1) & 0x80008000U;
  __nv_fp4x8_storage_t fourth_fp4_e = fp4x8 >> 3;

  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  asm volatile(
    "{\n"
    "  lop3.b32 %0, %0, %1, %2, %3;\n"
    "}\n"
    : "+r"(fourth_fp4_e)
    : "n"(0x01800180U), "r"(fourth_fp4_s), "n"(immLut));

  __nv_fp4x8_storage_t fourth_fp4_m = fp4x8 >> 7;

  asm volatile(
    "{\n"
    "  lop3.b32 %0, %1, %2, %3, %4;\n"
    "}\n"
    : "=r"(fourth_fp4)
    : "r"(fourth_fp4_m), "n"(0x00400040U), "r"(fourth_fp4_e), "n"(immLut));

  bf16x2_raw[3] = __hmul2(reinterpret_cast<__nv_bfloat162&>(fourth_fp4), bias);

  return bf16x8_raw;
}



inline __device__ unsigned
prmt(unsigned hi, unsigned lo, unsigned select_code)
{
  unsigned res = 0;

  asm volatile(
    "{\n"							\
    "prmt.b32 %0, %1, %2, %3;\n"				\
    "}\n"							\
    : "=r"(res) : "r"(lo) , "r"(hi), "r"(select_code));

  return res;
}


__constant__ static __nv_fp8x4_storage_t HIGH_E4M3s_LUT_[2] = {0x03020100U, 0x03020100U};
__constant__ static __nv_fp8x4_storage_t LOW_E4M3s_LUT_[2] = {0xFFFEFC00U, 0xFFFEFC00U};


__device__ __inline__
__nv_fp8x4_storage_t
cvt_lut_fp4_to_bf16
(
  const unsigned index
)
{

  auto lane_id = threadIdx.x & 0x1;
  __nv_fp8x4_storage_t h4b_lut = HIGH_E4M3s_LUT_[lane_id];
  __nv_fp8x4_storage_t l4b_lut = LOW_E4M3s_LUT_[lane_id];

  __nv_fp8x4_storage_t lut_res = prmt(h4b_lut, l4b_lut, index);

  return lut_res;
}

__device__ __inline__
__nv_bf16x8_storage_t
psx_cvt_lut_prmt_fp4x8_to_bf16x8_interleaved
(
    const __nv_fp4x8_storage_t fp4x8
)
{
    // interleaved version
    // input fp4x8: 7564 3120
    // output bf16x8: 7654 3210

    __nv_bf16x8_storage_t bf16x8_raw;
    __nv_bf16x2_storage_t *bf16x2_raw = reinterpret_cast<__nv_bf16x2_storage_t *>(&bf16x8_raw);

    __nv_fp8x4_storage_t h_fp8x4_0to1_bits = (fp4x8 & 0xC0C0C0C0U) >> 6; // 7632
    __nv_fp8x4_storage_t l_fp8x4_0to1_bits = (fp4x8 & 0x0C0C0C0CU) >> 2; // 5410
    
    unsigned h4b_em_fp4x4 = (fp4x8 & 0x77770000U) >> 16U;
    unsigned l4b_em_fp4x4 = (fp4x8 & 0x00007777U);

    __nv_fp8x4_storage_t h4b_2to9_bits = cvt_lut_fp4_to_bf16(h4b_em_fp4x4); // 7564
    __nv_fp8x4_storage_t l4b_2to9_bits = cvt_lut_fp4_to_bf16(l4b_em_fp4x4); // 3120

    bf16x2_raw[0] = prmt(l_fp8x4_0to1_bits, l4b_2to9_bits, 0x5240U) << 6U; // 1 0
    bf16x2_raw[1] = prmt(h_fp8x4_0to1_bits, l4b_2to9_bits, 0x5341U) << 6U; // 3 2

    bf16x2_raw[2] = prmt(l_fp8x4_0to1_bits, h4b_2to9_bits, 0x7260U) << 6U; // 5 4
    bf16x2_raw[3] = prmt(h_fp8x4_0to1_bits, h4b_2to9_bits, 0x7361U) << 6U; // 7 6

    return bf16x8_raw;
}

// FP4 E2M1 [0, 0.5, 1, 1.5] encoded as FP8 E4M3.
__constant__ static uint32_t FP4_POS_E4M3s_REG1_[2] = {0x3C383000, 0x3C383000};
// FP4 E2M1 [2, 3, 4, 6] encoded as FP8 E4M3.
__constant__ static uint32_t FP4_POS_E4M3s_REG2_[2] = {0x4C484440, 0x4C484440};


__device__ __inline__
__nv_fp8x8_storage_t
psx_cvt_lut_prmt_fp4x8_to_fp8x8
(
    const __nv_fp4x8_storage_t fp4x8
)
{
    __nv_fp8x8_storage_t fp8x8_raw;
    __nv_fp8x4_storage_t *fp8x4_raw = reinterpret_cast<__nv_fp8x4_storage_t *>(&fp8x8_raw);

    __nv_fp8x4_storage_t hb_sign_fp8x4 = (fp4x8 & 0x80808080U);
    __nv_fp8x4_storage_t lb_sign_fp8x4 = (fp4x8 & 0x08080808U) << 4U;

    __nv_fp8x4_storage_t h4b_sign_fp8x4 = prmt(hb_sign_fp8x4, lb_sign_fp8x4, 0x7362U);
    __nv_fp8x4_storage_t l4b_sign_fp8x4 = prmt(hb_sign_fp8x4, lb_sign_fp8x4, 0x5140U);

    // PRMT consumes only the low 16 bits of its selector in generic mode, so
    // the low half does not need its high selector half cleared.
    unsigned l4b_em_fp4x4 = fp4x8 & 0x77777777U;
    unsigned h4b_em_fp4x4 = l4b_em_fp4x4 >> 16U;

    auto lane_id = threadIdx.x & 0x1;
    uint32_t h4b_lut = FP4_POS_E4M3s_REG2_[lane_id];
    uint32_t l4b_lut = FP4_POS_E4M3s_REG1_[lane_id];
    __nv_fp8x4_storage_t h4b_em_fp8x4 = prmt(h4b_lut, l4b_lut, h4b_em_fp4x4);
    __nv_fp8x4_storage_t l4b_em_fp8x4 = prmt(h4b_lut, l4b_lut, l4b_em_fp4x4);

    fp8x4_raw[0] = l4b_sign_fp8x4 | l4b_em_fp8x4;
    fp8x4_raw[1] = h4b_sign_fp8x4 | h4b_em_fp8x4;

    return fp8x8_raw;
}

__device__ __inline__
__nv_fp8x8_storage_t
psx_cvt_lut_prmt_fp4x8_to_fp8x8_preprocessed_signs
(
    const __nv_fp4x8_storage_t fp4x8
)
{
    __nv_fp8x8_storage_t fp8x8_raw;
    __nv_fp8x4_storage_t *fp8x4_raw = reinterpret_cast<__nv_fp8x4_storage_t *>(&fp8x8_raw);

    // Offline preprocessing keeps each nibble's low 3 EM bits in place, but
    // repacks signs so outputs 0..3 are already in byte bit7 and outputs 4..7
    // are in bit3 of each byte.  That removes the runtime sign-gather PRMTs.
    // PRMT consumes only the low 16 bits of its selector in generic mode, so
    // the low half does not need its high selector half cleared.
    unsigned l4b_em_fp4x4 = fp4x8 & 0x77777777U;
    unsigned h4b_em_fp4x4 = l4b_em_fp4x4 >> 16U;

    auto lane_id = threadIdx.x & 0x1;
    uint32_t h4b_lut = FP4_POS_E4M3s_REG2_[lane_id];
    uint32_t l4b_lut = FP4_POS_E4M3s_REG1_[lane_id];
    __nv_fp8x4_storage_t h4b_em_fp8x4 = prmt(h4b_lut, l4b_lut, h4b_em_fp4x4);
    __nv_fp8x4_storage_t l4b_em_fp8x4 = prmt(h4b_lut, l4b_lut, l4b_em_fp4x4);

    fp8x4_raw[0] = (fp4x8 & 0x80808080U) | l4b_em_fp8x4;
    fp8x4_raw[1] = ((fp4x8 << 4U) & 0x80808080U) | h4b_em_fp8x4;

    return fp8x8_raw;
}


// [ 0,  1,  2,  3] encoded as FP8
__constant__ static uint32_t POS_E4M3s_REG1_[2] = {0x44403800, 0x44403800};
// [ 4,  5,  6,  7] encoded as FP8
__constant__ static uint32_t POS_E4M3s_REG2_[2] = {0x4E4C4A48, 0x4E4C4A48};
// [-8, -7, -6, -5] encoded as FP8
__constant__ static uint32_t NEG_E4M3s_REG1_[2] = {0xCACCCED0, 0xCACCCED0};
// [-4, -3, -2, -1] encoded as FP8
__constant__ static uint32_t NEG_E4M3s_REG2_[2] = {0xB8C0C4C8, 0xB8C0C4C8};


__device__ __inline__
__nv_fp8x8_storage_t
psx_cvt_lut_prmt_int4x8_to_fp8x8
(
    const __nv_int4x8_storage_t int4x8
)
{
    __nv_fp8x8_storage_t fp8x8_raw;
    __nv_fp8x4_storage_t *fp8x4_raw = reinterpret_cast<__nv_fp8x4_storage_t *>(&fp8x8_raw);

    // View the input as reg
    uint32_t reg = reinterpret_cast<const uint32_t&>(int4x8);

    // Determines if to get from the signed or unsigned candidates
    uint32_t sign = (reg & 0x88888888) >> 1;

    // Ignore sign bit when indexing into LUT
    uint32_t lut_idx = (reg & 0x77777777);

    // Signed is OR'd with 0x32103210 to find the correct value in the LUT
    const uint32_t final_prmt_base = 0x32103210;

    auto lane_id = threadIdx.x & 0x1;
    uint32_t POS_E4M3s_REG1 = POS_E4M3s_REG1_[lane_id];
    uint32_t POS_E4M3s_REG2 = POS_E4M3s_REG2_[lane_id];
    uint32_t NEG_E4M3s_REG1 = NEG_E4M3s_REG1_[lane_id];
    uint32_t NEG_E4M3s_REG2 = NEG_E4M3s_REG2_[lane_id];

    asm volatile(
      "{\n"
      "  .reg .b32 pos_f8s, neg_f8s;\n"
      "  .reg .b32 lut1, sign1, prmt0, prmt1;\n"
      "  or.b32 prmt0, %4, %3;\n"
      "  prmt.b32 pos_f8s, %5, %6, %2;\n"
      "  prmt.b32 neg_f8s, %7, %8, %2;\n"
      "  prmt.b32 %0, pos_f8s, neg_f8s, prmt0;\n"
      "  shr.u32 lut1, %2, 16;\n"
      "  shr.u32 sign1, %3, 16;\n"
      "  or.b32 prmt1, %4, sign1;\n"
      "  prmt.b32 pos_f8s, %5, %6, lut1;\n"
      "  prmt.b32 neg_f8s, %7, %8, lut1;\n"
      "  prmt.b32 %1, pos_f8s, neg_f8s, prmt1;\n"
      "}\n"
      : "=r"(fp8x4_raw[0]), "=r"(fp8x4_raw[1])
      : "r"(lut_idx), "r"(sign), "r"(final_prmt_base),
        "r"(POS_E4M3s_REG1), "r"(POS_E4M3s_REG2),
        "r"(NEG_E4M3s_REG1), "r"(NEG_E4M3s_REG2)
    );

    return fp8x8_raw;
}

template<class...>
using MixedInputVoid = void;

template<class Collective, class = void>
struct MixedInputFusedE8M0PreMmaScale {
  static constexpr bool value = false;
};

template<class Collective>
struct MixedInputFusedE8M0PreMmaScale<
    Collective,
    MixedInputVoid<decltype(Collective::FusedE8M0PreMmaScale)>> {
  static constexpr bool value = Collective::FusedE8M0PreMmaScale;
};

template<class Collective>
struct MixedInputUtils {
private:
  using KernelSchedule = typename Collective::KernelSchedule;
  using ConversionMode = typename Collective::ConversionMode;
  using SmemLayoutA = typename Collective::SmemLayoutA;
  using SmemLayoutB = typename Collective::SmemLayoutB;
  using SmemLayoutScale = typename Collective::SmemLayoutScale;
  using SmemLayoutActivationScale = typename Collective::SmemLayoutActivationScale;
  using SwappedElementA = typename Collective::SwappedElementA;
  using SwappedElementB = typename Collective::SwappedElementB;
  using RealSwappedElementA = typename Collective::RealSwappedElementA;
  using RealSwappedElementB = typename Collective::RealSwappedElementB;
  using ElementScale = typename Collective::ElementScale;
  using ElementZero = typename Collective::ElementZero;
  using NonVoidElementActivationScale = typename Collective::NonVoidElementActivationScale;
  using SmemCopyAtomScale = typename Collective::SmemCopyAtomScale;
  static constexpr auto KernelConversionMode = Collective::KernelConversionMode;
  static constexpr auto ModeHasScales = Collective::ModeHasScales;
  static constexpr auto UseScaleLookupTable = Collective::UseScaleLookupTable;
  static constexpr auto UseFP4ToBF16LookupTable = Collective::UseFP4ToBF16LookupTable;
  static constexpr auto UseFP4ToFP8LookupTable = Collective::UseFP4ToFP8LookupTable;
  static constexpr auto UseInt4ToFP8LookupTable = Collective::UseInt4ToFP8LookupTable;
  static constexpr auto HasActivationScale = Collective::HasActivationScale;
  static constexpr bool FusedE8M0PreMmaScale =
      MixedInputFusedE8M0PreMmaScale<Collective>::value;

public:
  static constexpr auto
  elements_per_smem_scale() {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return 0;
    } 
    else if constexpr (ModeHasScales) {
      return cute::cosize_v<SmemLayoutScale>;
    } 
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Type not handled in scale smem allocation.");
    }
  }

  static constexpr auto
  elements_per_smem_zero() {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert ||
                  KernelConversionMode == ConversionMode::ConvertAndScale ) {
      return 0;
    } 
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      return cute::cosize_v<SmemLayoutScale>;
    } 
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Type not handled in scale smem allocation.");
    }
  }

  // These methods use some the public members of the class. For that reason, we define them after the public section.
  static constexpr uint32_t
  compute_tma_transaction_bytes_mk() {
    return cutlass::bits_to_bytes(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(cute::sizeof_bits_v<SwappedElementA>));
  }

  static constexpr uint32_t
  compute_tma_transaction_bytes_nk() {
    return cutlass::bits_to_bytes(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(cute::sizeof_bits_v<SwappedElementB>));
  }

  static constexpr uint32_t
  compute_tma_transaction_bytes_extra() {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return 0;
    }
    else if constexpr (ModeHasScales) {
      constexpr uint32_t scale_tx_bytes = cutlass::bits_to_bytes(size<0>(SmemLayoutScale{}) * size<1>(SmemLayoutScale{}) * static_cast<uint32_t>(cute::sizeof_bits_v<ElementScale>));
      static_assert(scale_tx_bytes % 128 == 0, "Each scale stage must be 128B aligned."); // required by TMA
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        if constexpr (HasActivationScale) {
          constexpr uint32_t activation_scale_tx_bytes = cutlass::bits_to_bytes(
              size<0>(SmemLayoutActivationScale{}) * size<1>(SmemLayoutActivationScale{}) *
              static_cast<uint32_t>(cute::sizeof_bits_v<NonVoidElementActivationScale>));
          return scale_tx_bytes + activation_scale_tx_bytes;
        }
        else {
          return scale_tx_bytes;
        }
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        // Scale and zero share smem layout
        constexpr uint32_t zero_tx_bytes = cutlass::bits_to_bytes(size<0>(SmemLayoutScale{}) * size<1>(SmemLayoutScale{}) * static_cast<uint32_t>(cute::sizeof_bits_v<ElementZero>));
        static_assert(zero_tx_bytes % 128 == 0, "Each zero stage must be 128B aligned."); // required by TMA
        return scale_tx_bytes + zero_tx_bytes;
      }
      else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Type not handled in tma transaction bytes computation.");
      }
    }
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Type not handled in tma transaction bytes computation.");
    }
  }

  /// Utilities to copy A from smem to RF
  template <class SmemTiledCopyA,
            class TensorASmemView,
            class TensorACopyView
            >
  CUTLASS_DEVICE
  static void copy_tensors_A(
    SmemTiledCopyA const& smem_tiled_copy_A,
    TensorASmemView const& tCsA,
    TensorACopyView& tCrA_copy_view,
    int k_block,
    int read_stage) {

    if (k_block < size<2>(tCsA.shape())) {
      copy(smem_tiled_copy_A, tCsA(_,_,k_block,read_stage), tCrA_copy_view(_,_,k_block));
    }
  }


  /// Utilities to copy Scales for A from smem to RF
  template <class... Ts,
            class... Us
            >
  CUTLASS_DEVICE
  static void copy_tensors_SFA(
    cute::tuple<Ts...> const& partitioned_mma_extra_info,
    cute::tuple<Us...> const& tiled_copy_and_views,
    int k_block,
    int read_stage) {

    // We are starting a new k-tile so copy the scale
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // nothing to do
    }
    else if constexpr (ModeHasScales) {
      auto smem_tiled_copy_S = cute::get<0>(tiled_copy_and_views);
      auto tCrS_copy_view    = cute::get<1>(tiled_copy_and_views);
      auto tCsS              = cute::get<0>(partitioned_mma_extra_info);
      copy(smem_tiled_copy_S, tCsS(_,_,k_block,read_stage), tCrS_copy_view(_,_,k_block));
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        // Nothing extra to do
      } else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        auto tCsZ              = cute::get<2>(partitioned_mma_extra_info);
        auto tCrZ_copy_view    = cute::get<2>(tiled_copy_and_views);
        copy(smem_tiled_copy_S, tCsZ(_,_,k_block,read_stage), tCrZ_copy_view(_,_,k_block));
      } else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
      }
    } 
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
    }
  }


  /// Utilities to copy A and extra inputs from smem to RF
  template <class SmemTiledCopyA,
            class TensorASmemView,
            class TensorACopyView,
            class... Ts,
            class... Us
            >
  CUTLASS_DEVICE
  static void copy_tensors_MK(
    SmemTiledCopyA const& smem_tiled_copy_A,
    TensorASmemView const& tCsA,
    TensorACopyView& tCrA_copy_view,
    cute::tuple<Ts...> const& partitioned_mma_extra_info,
    cute::tuple<Us...> const& tiled_copy_and_views,
    int k_block,
    int read_stage) {

    if (k_block < size<2>(tCsA.shape())) {
      copy(smem_tiled_copy_A, tCsA(_,_,k_block,read_stage), tCrA_copy_view(_,_,k_block));
    }

    if (k_block == 0) {
      // We are starting a new k-tile so copy the scale
      if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
        // nothing to do
      } 
      else if constexpr (ModeHasScales) {
        auto smem_tiled_copy_S = cute::get<0>(tiled_copy_and_views);
        auto tCrS_copy_view    = cute::get<1>(tiled_copy_and_views);
        auto tCsS              = cute::get<0>(partitioned_mma_extra_info);
        copy(smem_tiled_copy_S, tCsS(_,_,k_block,read_stage), tCrS_copy_view(_,_,k_block));
        if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
          // Nothing extra to do
        } else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
          auto tCsZ              = cute::get<2>(partitioned_mma_extra_info);
          auto tCrZ_copy_view    = cute::get<2>(tiled_copy_and_views);
          copy(smem_tiled_copy_S, tCsZ(_,_,k_block,read_stage), tCrZ_copy_view(_,_,k_block));
        } else {
          static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
        }
      } 
      else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
      }
    }
  }

  // The core converter uses a lookup table to converts i4 -> 8 bit value.
  template <class EngineIn,
            class LayoutIn,
            class EngineOut,
            class LayoutOut,
            class EngineScale,
            class LayoutScale>
  CUTLASS_DEVICE
  static void lookup_table_convert( // Accept mutable temporaries
    Tensor<EngineIn, LayoutIn>       const& src,
    Tensor<EngineOut, LayoutOut>         && dst,
    Tensor<EngineScale, LayoutScale> const& scales_neg,
    Tensor<EngineScale, LayoutScale> const& scales_pos) {
    
    lookup_table_convert(src, dst, scales_neg, scales_pos);
  }
  template <class EngineIn,
            class LayoutIn,
            class EngineOut,
            class LayoutOut,
            class EngineScale,
            class LayoutScale>
  CUTLASS_DEVICE
  static void lookup_table_convert(
    Tensor<EngineIn, LayoutIn>       const& src,
    Tensor<EngineOut, LayoutOut>          & dst,
    Tensor<EngineScale, LayoutScale> const& scales_neg,
    Tensor<EngineScale, LayoutScale> const& scales_pos) {

    constexpr int N = cute::cosize(LayoutIn{});
    static_assert(N == 4 || N == 8);
    static_assert(cosize(LayoutScale{}) <= N / 4, 
                  "at least 4 consecutive weights must share the same scale.");
    using SrcArray = cutlass::Array<cutlass::int4b_t, 8>;
    using DstArray = cutlass::Array<RealSwappedElementB, 8>;
    using RegArray = cutlass::AlignedArray<uint32_t, N / 4, sizeof(DstArray)>;

    // View the input as reg
    auto&& src_reg = cute::recast<uint32_t>(src)(0);
    auto&& r       = cute::recast<RegArray>(dst)(0);

    // Determines if to get from the signed or unsigned candidates
    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    uint32_t sign; // ((reg & 0x88888888) | 0x64206420) >> 1 
    asm volatile(
      "{\n"
      "  lop3.b32 %0, %1, %2, %3, %4;\n" \
      "}\n"
      : "=r"(sign)
      : "r"(src_reg), "n"(0x88888888), "n"(0x64206420), "n"(immLut)
    );
    sign = sign >> 1;

    // Ignore sign bit when indexing into LUT
    uint32_t lut_idx = src_reg & 0x77777777;
    Tensor scales_neg_ = cute::filter(scales_neg);
    Tensor scales_pos_ = cute::filter(scales_pos);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 4; ++i, lut_idx >>=16, sign >>=16) {
      auto&& scale_neg_ = reinterpret_cast<cutlass::Array<uint32_t, 2> const&>(scales_neg_(i));
      auto&& scale_pos_ = reinterpret_cast<cutlass::Array<uint32_t, 2> const&>(scales_pos_(i));
      asm volatile(
        "{\n"
        "  .reg .b32 pos, neg                    ;\n" \
        "  prmt .b32 neg, %3, %4, %1             ;\n" \
        "  prmt .b32 pos, %5, %6, %1             ;\n" \
        "  prmt .b32 %0, pos, neg, %2            ;\n" \
        "}\n"
        : "=r"(r[i])
        : "r"(lut_idx), "r"(sign), "r"(scale_neg_[0]), "r"(scale_neg_[1]), "r"(scale_pos_[0]), "r"(scale_pos_[1])
      );
    }
  }


  template <class EngineIn,
            class LayoutIn,
            class EngineOut,
            class LayoutOut>
  CUTLASS_DEVICE
  static void fp4tobf16_lookup_table_convert( // Accept mutable temporaries
    Tensor<EngineIn, LayoutIn>       const& src,
    Tensor<EngineOut, LayoutOut>         && dst) {
    fp4tobf16_lookup_table_convert(src, dst);
  }

  template <class EngineIn,
            class LayoutIn,
            class EngineOut,
            class LayoutOut>
  CUTLASS_DEVICE
  static void fp4tobf16_lookup_table_convert(
    Tensor<EngineIn, LayoutIn>       const& src,
    Tensor<EngineOut, LayoutOut>          & dst) {

    // View the input as reg
    auto&& src_ = cute::recast<__nv_fp4x8_storage_t>(src)(0);
    auto&& dst_ = cute::recast<__nv_bf16x8_storage_t>(dst)(0);
    
    // dst_ = psx_cvt_lut_prmt_fp4x8_to_bf16x8_interleaved(src_);
    dst_ = psx_cvt_triton_fp4x8_to_bf16x8_interleaved(src_);
  }


  template <class EngineIn,
            class LayoutIn,
            class EngineOut,
            class LayoutOut>
  CUTLASS_DEVICE
  static void int4tofp8_lookup_table_convert( // Accept mutable temporaries
    Tensor<EngineIn, LayoutIn>       const& src,
    Tensor<EngineOut, LayoutOut>         && dst) {
    int4tofp8_lookup_table_convert(src, dst);
  }

  template <class EngineIn,
            class LayoutIn,
            class EngineOut,
            class LayoutOut>
  CUTLASS_DEVICE
  static void int4tofp8_lookup_table_convert(
    Tensor<EngineIn, LayoutIn>       const& src,
    Tensor<EngineOut, LayoutOut>          & dst) {

    // View the input as reg
    auto&& src_ = cute::recast<__nv_int4x8_storage_t>(src)(0);
    auto&& dst_ = cute::recast<__nv_fp8x8_storage_t>(dst)(0);
    
    dst_ = psx_cvt_lut_prmt_int4x8_to_fp8x8(src_);
  }

  template <class EngineIn,
            class LayoutIn,
            class EngineOut,
            class LayoutOut>
  CUTLASS_DEVICE
  static void fp4tofp8_lookup_table_convert( // Accept mutable temporaries
    Tensor<EngineIn, LayoutIn>       const& src,
    Tensor<EngineOut, LayoutOut>         && dst) {
    fp4tofp8_lookup_table_convert(src, dst);
  }

  template <class EngineIn,
            class LayoutIn,
            class EngineOut,
            class LayoutOut>
  CUTLASS_DEVICE
  static void fp4tofp8_lookup_table_convert(
    Tensor<EngineIn, LayoutIn>       const& src,
    Tensor<EngineOut, LayoutOut>          & dst) {

    auto&& src_ = cute::recast<__nv_fp4x8_storage_t>(src)(0);
    auto&& dst_ = cute::recast<__nv_fp8x8_storage_t>(dst)(0);

#if defined(CUTLASS_MIXED_GEMM_FP4_FP8_PREPROCESSED_SIGNS)
    dst_ = psx_cvt_lut_prmt_fp4x8_to_fp8x8_preprocessed_signs(src_);
#else
    dst_ = psx_cvt_lut_prmt_fp4x8_to_fp8x8(src_);
#endif
  }

  __device__ __inline__
  static void fp4tofp8_fused_e8m0_pre_mma_convert_pair(
      __nv_fp4x8_storage_t fp4x8_0,
      __nv_fp4x8_storage_t fp4x8_1,
      __nv_fp8x8_storage_t& fp8x8_raw_0,
      __nv_fp8x8_storage_t& fp8x8_raw_1,
      uint32_t lo_exp_offset,
      uint32_t hi_exp_offset) {
    // One WGMMA A operand lane contributes two fp4x8 registers whose low
    // fp8x4 chunks share one row scale, and high chunks share the other.
    __nv_fp8x4_storage_t *fp8x4_raw_0 =
        reinterpret_cast<__nv_fp8x4_storage_t *>(&fp8x8_raw_0);
    __nv_fp8x4_storage_t *fp8x4_raw_1 =
        reinterpret_cast<__nv_fp8x4_storage_t *>(&fp8x8_raw_1);

    uint32_t const fp4_raw_0 = reinterpret_cast<uint32_t const&>(fp4x8_0);
    uint32_t const fp4_raw_1 = reinterpret_cast<uint32_t const&>(fp4x8_1);
    uint32_t const em_selector_0 = fp4_raw_0 & 0x77777777U;
    uint32_t const em_selector_1 = fp4_raw_1 & 0x77777777U;
    constexpr uint32_t fp4_codes_0_to_3_em_bias = 0x0c080000U;
    constexpr uint32_t fp4_codes_4_to_7_em_bias = 0x1c181410U;
    uint32_t const lo_l4b_exp_offseted_lut =
        (lo_exp_offset * 0x08080800U) + fp4_codes_0_to_3_em_bias;
    uint32_t const lo_h4b_exp_offseted_lut =
        (lo_exp_offset * 0x08080808U) + fp4_codes_4_to_7_em_bias;
    uint32_t const hi_l4b_exp_offseted_lut =
        (hi_exp_offset * 0x08080800U) + fp4_codes_0_to_3_em_bias;
    uint32_t const hi_h4b_exp_offseted_lut =
        (hi_exp_offset * 0x08080808U) + fp4_codes_4_to_7_em_bias;

    uint32_t const lo_em_fp8x4_0 =
        prmt(lo_h4b_exp_offseted_lut, lo_l4b_exp_offseted_lut, em_selector_0);
    uint32_t const lo_em_fp8x4_1 =
        prmt(lo_h4b_exp_offseted_lut, lo_l4b_exp_offseted_lut, em_selector_1);

#if defined(CUTLASS_MIXED_GEMM_FP4_FP8_PREPROCESSED_SIGNS)
    fp8x4_raw_0[0] = (fp4_raw_0 & 0x80808080U) | lo_em_fp8x4_0;
    fp8x4_raw_1[0] = (fp4_raw_1 & 0x80808080U) | lo_em_fp8x4_1;
#else
    uint32_t const hb_sign_fp8x4_0 = fp4_raw_0 & 0x80808080U;
    uint32_t const hb_sign_fp8x4_1 = fp4_raw_1 & 0x80808080U;
    uint32_t const lb_sign_fp8x4_0 = (fp4_raw_0 & 0x08080808U) << 4U;
    uint32_t const lb_sign_fp8x4_1 = (fp4_raw_1 & 0x08080808U) << 4U;
    uint32_t const l4b_sign_fp8x4_0 = prmt(hb_sign_fp8x4_0, lb_sign_fp8x4_0, 0x5140U);
    uint32_t const l4b_sign_fp8x4_1 = prmt(hb_sign_fp8x4_1, lb_sign_fp8x4_1, 0x5140U);
    uint32_t const h4b_sign_fp8x4_0 = prmt(hb_sign_fp8x4_0, lb_sign_fp8x4_0, 0x7362U);
    uint32_t const h4b_sign_fp8x4_1 = prmt(hb_sign_fp8x4_1, lb_sign_fp8x4_1, 0x7362U);

    fp8x4_raw_0[0] = l4b_sign_fp8x4_0 | lo_em_fp8x4_0;
    fp8x4_raw_1[0] = l4b_sign_fp8x4_1 | lo_em_fp8x4_1;
#endif

    uint32_t const hi_em_fp8x4_0 =
        prmt(hi_h4b_exp_offseted_lut, hi_l4b_exp_offseted_lut, em_selector_0 >> 16U);
    uint32_t const hi_em_fp8x4_1 =
        prmt(hi_h4b_exp_offseted_lut, hi_l4b_exp_offseted_lut, em_selector_1 >> 16U);

#if defined(CUTLASS_MIXED_GEMM_FP4_FP8_PREPROCESSED_SIGNS)
    fp8x4_raw_0[1] = ((fp4_raw_0 << 4U) & 0x80808080U) | hi_em_fp8x4_0;
    fp8x4_raw_1[1] = ((fp4_raw_1 << 4U) & 0x80808080U) | hi_em_fp8x4_1;
#else
    fp8x4_raw_0[1] = h4b_sign_fp8x4_0 | hi_em_fp8x4_0;
    fp8x4_raw_1[1] = h4b_sign_fp8x4_1 | hi_em_fp8x4_1;
#endif
  }

  template <class EngineIn,
            class LayoutIn,
            class EngineOut,
            class LayoutOut>
  CUTLASS_DEVICE
  static void fp4tofp8_fused_e8m0_pre_mma_convert_pair(
    Tensor<EngineIn, LayoutIn>       const& src0,
    Tensor<EngineIn, LayoutIn>       const& src1,
    Tensor<EngineOut, LayoutOut>          & dst0,
    Tensor<EngineOut, LayoutOut>          & dst1,
    uint32_t lo_exp_offset,
    uint32_t hi_exp_offset) {

    auto&& src0_ = cute::recast<__nv_fp4x8_storage_t>(src0)(0);
    auto&& src1_ = cute::recast<__nv_fp4x8_storage_t>(src1)(0);
    auto&& dst0_ = cute::recast<__nv_fp8x8_storage_t>(dst0)(0);
    auto&& dst1_ = cute::recast<__nv_fp8x8_storage_t>(dst1)(0);

    fp4tofp8_fused_e8m0_pre_mma_convert_pair(
        src0_, src1_, dst0_, dst1_, lo_exp_offset, hi_exp_offset);
  }

  /// Utilities to dequantize A.
  template <class Layout>
  CUTLASS_DEVICE
  static void static_check_scale(Layout const& tensor) {
    static_assert(shape<0>(Layout{}) >= 4 && stride<0>(Layout{}) == 0, "At least 4 adjacent weights in a thread must share the same scale.");
  }
  template <class Engine,
            class Layout>
  CUTLASS_DEVICE
  static void static_check_scale(Tensor<Engine, Layout> const& tensor) {
    static_check_scale(flatten(Layout{}));
  }

  template <class EngineIn,
            class EngineOut, 
            class LayoutIn,
            class LayoutOut,
            class... Ts>
  CUTLASS_DEVICE
  static void dequantize_A_kblock(
    Tensor<EngineIn, LayoutIn> const& tCrA_load, 
    Tensor<EngineOut, LayoutOut>& tCrA_mma,
    cute::tuple<Ts...>& partitioned_extra_info,
    int const k_block) {

    static_assert(is_rmem<EngineIn>::value, "Input tensor for A conversion must come from registers");
    static_assert(is_rmem<EngineOut>::value, "Output tensor for A conversion must come from registers");
    static_assert(cosize_v<LayoutIn> == cosize_v<LayoutOut>);
    static_assert(size_v<LayoutIn> == cosize_v<LayoutIn>);
    static_assert(size_v<LayoutOut> == cosize_v<LayoutOut>);
    using SrcType = typename EngineIn::value_type;
    using DstType = typename EngineOut::value_type;

    Tensor src = tCrA_load(_, _, k_block);
    Tensor dst = tCrA_mma(_, _, k_block);
    
    CUTE_STATIC_ASSERT_V(size(src(_, 0)) == cosize(src(_, 0).layout()),
                         "The first mode of tensor src must be contiguous in memory");
    // try to make the size of the first mode equal to 32bit
    int constexpr NumValPerSrcReg = cute::min(decltype(size(src(_, 0)))::value,
                                              ceil_div(32, sizeof_bits_v<SrcType>));
    Tensor src_vm = cute::group_modes<1,-1>(cute::zipped_divide(src, Int<NumValPerSrcReg>{}));
    Tensor dst_vm = cute::group_modes<1,-1>(cute::zipped_divide(dst, Int<NumValPerSrcReg>{}));

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<1>(dst_vm); ++i) {
        LayoutAwareConvert(src_vm(_, i), dst_vm(_, i));
      }
    } 
    else if constexpr (UseScaleLookupTable) {
      constexpr int num_elements = decltype(size(src))::value;
      static_assert(is_same_v<RealSwappedElementA, cutlass::int4b_t>, "Lookup table only supports int4 being the quant type now.");
      static_assert(sizeof_bits_v<ElementScale> == 64, "Lookup table only supports 8 8bit scale values now.");
      static_assert(num_elements % 4 == 0 && num_elements >= 4, "Lookup table requires a vector size of 4x when converting.");

      Tensor tCrS_neg = cute::get<1>(partitioned_extra_info);
      auto&& tCrS_pos = cute::get<2>(partitioned_extra_info); // modification to its value is needed
      Tensor scales_neg = tCrS_neg(_, _, k_block);
      Tensor scales_pos = tCrS_pos(_, _, k_block);
      CUTE_STATIC_ASSERT_V(cute::size(src) == cute::size(scales_neg));

      static_check_scale(scales_neg);
      static_check_scale(scales_pos);
      Tensor scales_neg_vm = cute::group_modes<1,-1>(cute::zipped_divide(scales_neg, Int<NumValPerSrcReg>{}));
      Tensor scales_pos_vm = cute::group_modes<1,-1>(cute::zipped_divide(scales_pos, Int<NumValPerSrcReg>{}));

      if (k_block == 0) {
        Tensor scales_neg_vm_ = filter(scales_neg_vm);
        Tensor scales_pos_vm_ = filter(scales_pos_vm);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(scales_neg_vm_.layout()); ++i)
        {
          auto&& scale_neg_ = reinterpret_cast<cutlass::Array<uint32_t, 2> const&>(scales_neg_vm_(i));
          auto&& scale_pos_ = reinterpret_cast<cutlass::Array<uint32_t, 2>      &>(scales_pos_vm_(i));
          constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;
          asm volatile(
              "{\n"
              "  lop3 .b32 %0, %2, %4, %5, %6;\n" \
              "  xor  .b32 %1, %3, %5;        \n" \
              "}\n"
              : "=r"(scale_pos_[0]), "=r"(scale_pos_[1])
              : "r"(scale_neg_[0]), "r"(scale_neg_[1]), "n"(0xFFFFFF00), "n"(0x80808080), "n"(immLut)
            );
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<1>(dst_vm); ++i) {
        lookup_table_convert(src_vm(_, i), dst_vm(_, i), scales_neg_vm(_, i), scales_pos_vm(_, i));
      }
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      Tensor scales = cute::get<1>(partitioned_extra_info)(_, _, k_block);
      CUTE_STATIC_ASSERT_V(size(src) == size(scales));
      Tensor scales_vm = cute::group_modes<1,-1>(cute::zipped_divide(scales, Int<NumValPerSrcReg>{}));

      if constexpr (is_same_v<DstType, ElementScale>) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(dst_vm); ++i) {
          LayoutAwareConvert(src_vm(_, i), dst_vm(_, i));
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<0>(dst_vm); ++j) {
            dst_vm(j, i) *= scales_vm(j, i);
          }
        }
      }
      else {
        auto stage = make_tensor_like<ElementScale>(src_vm(_, 0));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(dst_vm); ++i) {
          LayoutAwareConvert(src_vm(_, i), stage);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<0>(dst_vm); ++j) {
            stage(j) *= scales_vm(j, i);
          }
          LayoutAwareConvert(stage, dst_vm(_, i));
        }
      }
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      static_assert(is_same_v<ElementScale, ElementZero>, "ElementScale and ElementZero must be the same.");
      Tensor scales = cute::get<1>(partitioned_extra_info)(_, _, k_block);
      Tensor zeros  = cute::get<3>(partitioned_extra_info)(_, _, k_block);
      CUTE_STATIC_ASSERT_V(size(src) == size(scales));
      CUTE_STATIC_ASSERT_V(size(src) == size(zeros));
      Tensor scales_vm = cute::group_modes<1,-1>(cute::zipped_divide(scales, Int<NumValPerSrcReg>{}));
      Tensor zeros_vm = cute::group_modes<1,-1>(cute::zipped_divide(zeros, Int<NumValPerSrcReg>{}));
      
      if constexpr (is_same_v<DstType, ElementScale>) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(dst_vm); ++i) {
          LayoutAwareConvert(src_vm(_, i), dst_vm(_, i));
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<0>(dst_vm); ++j) {
            dst_vm(j, i) = dst_vm(j, i) * scales_vm(j, i) + zeros_vm(j, i);
          }
        }
      }
      else {
        auto stage = make_tensor_like<ElementScale>(src_vm(_, 0));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(dst_vm); ++i) {
          LayoutAwareConvert(src_vm(_, i), stage);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<0>(dst_vm); ++j) {
            stage(j) = stage(j) * scales_vm(j, i) + zeros_vm(j, i);
          }
          LayoutAwareConvert(stage, dst_vm(_, i));
        }
      }
    }
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "No A data is loaded.");
    }
  }

  template <class EngineIn,
            class EngineOut, 
            class LayoutIn,
            class LayoutOut>
  CUTLASS_DEVICE
  static void convert_A_slot(
    Tensor<EngineIn, LayoutIn> const& src,
    Tensor<EngineOut, LayoutOut>& dst) {

    static_assert(is_rmem<EngineIn>::value, "Input tensor for A conversion must come from registers");
    static_assert(is_rmem<EngineOut>::value, "Output tensor for A conversion must come from registers");
    using SrcType = typename EngineIn::value_type;

    CUTE_STATIC_ASSERT_V(size(src(_, 0)) == cosize(src(_, 0).layout()),
                         "The first mode of tensor src must be contiguous in memory");
    CUTE_STATIC_ASSERT_V(size(src) == size(dst));
    // try to make the size of the first mode equal to 32bit
    int constexpr NumValPerSrcReg = cute::min(decltype(size(src(_, 0)))::value,
                                              ceil_div(32, sizeof_bits_v<SrcType>));
    Tensor src_vm = cute::group_modes<1,-1>(cute::zipped_divide(src, Int<NumValPerSrcReg>{}));
    Tensor dst_vm = cute::group_modes<1,-1>(cute::zipped_divide(dst, Int<NumValPerSrcReg>{}));

    // KernelConversionMode == ConversionMode::DirectConvert
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(dst_vm); ++i) {
      if constexpr (UseFP4ToBF16LookupTable) {
        fp4tobf16_lookup_table_convert(src_vm(_, i), dst_vm(_, i));
      }
      else if constexpr (UseFP4ToFP8LookupTable) {
        fp4tofp8_lookup_table_convert(src_vm(_, i), dst_vm(_, i));
      }
      else if constexpr (UseInt4ToFP8LookupTable) {
        int4tofp8_lookup_table_convert(src_vm(_, i), dst_vm(_, i));
      }
      else {
        LayoutAwareConvert(src_vm(_, i), dst_vm(_, i));
      }
    }
  }

  template <class EngineIn,
            class EngineOut,
            class LayoutIn,
            class LayoutOut>
  CUTLASS_DEVICE
  static void convert_A_kblock(
    Tensor<EngineIn, LayoutIn> const& tCrA_load,
    Tensor<EngineOut, LayoutOut>& tCrA_mma,
    int const k_block) {

    Tensor src = tCrA_load(_, _, k_block);
    Tensor dst = tCrA_mma(_, _, k_block);
    convert_A_slot(src, dst);
  }

  template <int KBlock,
            class EngineIn,
            class EngineOut,
            class LayoutIn,
            class LayoutOut>
  CUTLASS_DEVICE
  static void convert_A_kblock(
    Tensor<EngineIn, LayoutIn> const& tCrA_load,
    Tensor<EngineOut, LayoutOut>& tCrA_mma,
    cute::Int<KBlock> k_block) {

    Tensor src = tCrA_load(_, _, k_block);
    Tensor dst = tCrA_mma(_, _, k_block);
    convert_A_slot(src, dst);
  }

  template <class EngineIn,
            class EngineOut,
            class LayoutIn,
            class LayoutOut,
            class EngineScale,
            class LayoutScale>
  CUTLASS_DEVICE
  static void convert_A_kblock_fused_e8m0_pre_mma_to_slot(
    Tensor<EngineIn, LayoutIn> const& tCrA_load,
    Tensor<EngineOut, LayoutOut>& tCrA_mma_slot,
    Tensor<EngineScale, LayoutScale>& scale_packs,
    int const k_block,
    int const scale_idx) {

    static_assert(FusedE8M0PreMmaScale, "This helper is only for fused e8m0 pre-MMA scale.");
    static_assert(UseFP4ToFP8LookupTable, "Fused e8m0 pre-MMA scale currently supports MXFP4 x FP8 only.");
    static_assert(cutlass::detail::is_Array_v<ElementScale>,
        "Fused e8m0 pre-MMA scale expects TileK-packed e8m0 scale arrays.");
    static_assert(is_rmem<EngineIn>::value, "Input tensor for A conversion must come from registers");
    static_assert(is_rmem<EngineOut>::value, "Output tensor for A conversion must come from registers");
    static_assert(is_rmem<EngineScale>::value, "Scale tensor for A conversion must come from registers");
    using SrcType = typename EngineIn::value_type;

    Tensor src = tCrA_load(_, _, k_block);
    Tensor dst = tCrA_mma_slot;

    CUTE_STATIC_ASSERT_V(size(src(_, 0)) == cosize(src(_, 0).layout()),
                         "The first mode of tensor src must be contiguous in memory");
    CUTE_STATIC_ASSERT_V(size(src) == size(dst));
    CUTE_STATIC_ASSERT_V(size(src) == size(scale_packs));

    int constexpr NumValPerSrcReg = cute::min(decltype(size(src(_, 0)))::value,
                                              ceil_div(32, sizeof_bits_v<SrcType>));
    Tensor src_vm = cute::group_modes<1,-1>(cute::zipped_divide(src, Int<NumValPerSrcReg>{}));
    Tensor dst_vm = cute::group_modes<1,-1>(cute::zipped_divide(dst, Int<NumValPerSrcReg>{}));
    Tensor scale_packs_vm = cute::group_modes<1,-1>(
        cute::zipped_divide(scale_packs, Int<NumValPerSrcReg>{}));

    auto scale_pack_values_0 = cute::filter(scale_packs_vm(_, Int<0>{}));
    constexpr int ScalePackCount = decltype(size(scale_pack_values_0))::value;
    constexpr int DstVecCount = decltype(size<1>(dst_vm))::value;
    static_assert(ScalePackCount == 2,
        "Fused e8m0 pre-MMA scale expects exactly two row-scale packs per paired A operand.");
    static_assert((DstVecCount % 2) == 0,
        "Fused e8m0 pre-MMA pair conversion expects an even number of fp4x8 operands.");
    using ScaleScalar = typename ElementScale::Element;

    // Make the row-scale pattern explicit: chunks 0/2 use scale 0, chunks
    // 1/3 use scale 1. Do not rely on ptxas to rediscover this pairing.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < DstVecCount; i += 2) {
      auto scale_pack_values = cute::filter(scale_packs_vm(_, i));
      ScaleScalar const lo_scale = scale_pack_values(0)[scale_idx];
      ScaleScalar const hi_scale = scale_pack_values(1)[scale_idx];
      uint32_t const lo_exp_offset = static_cast<uint32_t>(lo_scale.storage);
      uint32_t const hi_exp_offset = static_cast<uint32_t>(hi_scale.storage);
      auto src_vec0 = src_vm(_, i);
      auto src_vec1 = src_vm(_, i + 1);
      auto dst_vec0 = dst_vm(_, i);
      auto dst_vec1 = dst_vm(_, i + 1);
      fp4tofp8_fused_e8m0_pre_mma_convert_pair(
          src_vec0, src_vec1, dst_vec0, dst_vec1, lo_exp_offset, hi_exp_offset);
    }
  }

  template <int KBlock, int ScaleIdx,
            class EngineIn,
            class EngineOut,
            class LayoutIn,
            class LayoutOut,
            class EngineScale,
            class LayoutScale>
  CUTLASS_DEVICE
  static void convert_A_kblock_fused_e8m0_pre_mma_to_slot(
    Tensor<EngineIn, LayoutIn> const& tCrA_load,
    Tensor<EngineOut, LayoutOut>& tCrA_mma_slot,
    Tensor<EngineScale, LayoutScale>& scale_packs,
    cute::Int<KBlock>,
    cute::Int<ScaleIdx>) {

    static_assert(FusedE8M0PreMmaScale, "This helper is only for fused e8m0 pre-MMA scale.");
    static_assert(UseFP4ToFP8LookupTable, "Fused e8m0 pre-MMA scale currently supports MXFP4 x FP8 only.");
    static_assert(cutlass::detail::is_Array_v<ElementScale>,
        "Fused e8m0 pre-MMA scale expects TileK-packed e8m0 scale arrays.");
    static_assert(is_rmem<EngineIn>::value, "Input tensor for A conversion must come from registers");
    static_assert(is_rmem<EngineOut>::value, "Output tensor for A conversion must come from registers");
    static_assert(is_rmem<EngineScale>::value, "Scale tensor for A conversion must come from registers");
    using SrcType = typename EngineIn::value_type;

    Tensor src = tCrA_load(_, _, cute::Int<KBlock>{});
    Tensor dst = tCrA_mma_slot;

    CUTE_STATIC_ASSERT_V(size(src(_, 0)) == cosize(src(_, 0).layout()),
                         "The first mode of tensor src must be contiguous in memory");
    CUTE_STATIC_ASSERT_V(size(src) == size(dst));
    CUTE_STATIC_ASSERT_V(size(src) == size(scale_packs));

    int constexpr NumValPerSrcReg = cute::min(decltype(size(src(_, 0)))::value,
                                              ceil_div(32, sizeof_bits_v<SrcType>));
    Tensor src_vm = cute::group_modes<1,-1>(cute::zipped_divide(src, Int<NumValPerSrcReg>{}));
    Tensor dst_vm = cute::group_modes<1,-1>(cute::zipped_divide(dst, Int<NumValPerSrcReg>{}));
    Tensor scale_packs_vm = cute::group_modes<1,-1>(
        cute::zipped_divide(scale_packs, Int<NumValPerSrcReg>{}));

    auto scale_pack_values_0 = cute::filter(scale_packs_vm(_, Int<0>{}));
    constexpr int ScalePackCount = decltype(size(scale_pack_values_0))::value;
    constexpr int DstVecCount = decltype(size<1>(dst_vm))::value;
    static_assert(ScalePackCount == 2,
        "Fused e8m0 pre-MMA scale expects exactly two row-scale packs per paired A operand.");
    static_assert((DstVecCount % 2) == 0,
        "Fused e8m0 pre-MMA pair conversion expects an even number of fp4x8 operands.");
    using ScaleScalar = typename ElementScale::Element;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < DstVecCount; i += 2) {
      auto scale_pack_values = cute::filter(scale_packs_vm(_, i));
      ScaleScalar const lo_scale = scale_pack_values(0)[ScaleIdx];
      ScaleScalar const hi_scale = scale_pack_values(1)[ScaleIdx];
      uint32_t const lo_exp_offset = static_cast<uint32_t>(lo_scale.storage);
      uint32_t const hi_exp_offset = static_cast<uint32_t>(hi_scale.storage);
      auto src_vec0 = src_vm(_, i);
      auto src_vec1 = src_vm(_, i + 1);
      auto dst_vec0 = dst_vm(_, i);
      auto dst_vec1 = dst_vm(_, i + 1);
      fp4tofp8_fused_e8m0_pre_mma_convert_pair(
          src_vec0, src_vec1, dst_vec0, dst_vec1, lo_exp_offset, hi_exp_offset);
    }
  }

  template <class EngineIn,
            class EngineOut,
            class LayoutIn,
            class LayoutOut,
            class... Ts>
  CUTLASS_DEVICE
  static void convert_A_kblock_fused_e8m0_pre_mma_to_slot(
    Tensor<EngineIn, LayoutIn> const& tCrA_load,
    Tensor<EngineOut, LayoutOut>& tCrA_mma_slot,
    cute::tuple<Ts...>& partitioned_extra_info,
    int const k_block,
    int const scale_idx) {

    Tensor scale_packs = cute::get<1>(partitioned_extra_info)(_, _, Int<0>{});
    convert_A_kblock_fused_e8m0_pre_mma_to_slot(
        tCrA_load, tCrA_mma_slot, scale_packs, k_block, scale_idx);
  }

  template <int KBlock, int ScaleIdx,
            class EngineIn,
            class EngineOut,
            class LayoutIn,
            class LayoutOut,
            class... Ts>
  CUTLASS_DEVICE
  static void convert_A_kblock_fused_e8m0_pre_mma_to_slot(
    Tensor<EngineIn, LayoutIn> const& tCrA_load,
    Tensor<EngineOut, LayoutOut>& tCrA_mma_slot,
    cute::tuple<Ts...>& partitioned_extra_info,
    cute::Int<KBlock> k_block,
    cute::Int<ScaleIdx> scale_idx) {

    Tensor scale_packs = cute::get<1>(partitioned_extra_info)(_, _, Int<0>{});
    convert_A_kblock_fused_e8m0_pre_mma_to_slot(
        tCrA_load, tCrA_mma_slot, scale_packs, k_block, scale_idx);
  }

  /// Utilities for any additional inputs inside of the TMA load
  template <
    class Params,
    class TensorStorage,
    class... Ts
  >
  CUTLASS_DEVICE
  static auto partition_extra_tma_inputs(
    Params const& mainloop_params,
    cute::tuple<Ts...> const& load_inputs,
    TensorStorage& shared_tensors,
    uint2 const& cluster_local_block_id,
    int const m_coord, 
    int const l_coord) {

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return cute::make_tuple();
    } 
    else if constexpr (ModeHasScales) {
      Tensor sS  = make_tensor(make_smem_ptr(shared_tensors.smem_scale.begin()), SmemLayoutScale{}); // (BLK_M,BLK_K,PIPE)
      Tensor gS_mkl = get<2>(load_inputs);
      auto block_tma_s = mainloop_params.tma_load_scale.get_slice(cluster_local_block_id.y);
      Tensor gS = gS_mkl(_,_,m_coord,_,l_coord);                                                  // (BLK_M,BLK_K,k)

      Tensor tSgS = block_tma_s.partition_S(gS);                                              // (TMA,TMA_M,TMA_K,k)
      Tensor tSsS = block_tma_s.partition_D(sS);                                              // (TMA,TMA_M,TMA_K,PIPE)
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(tSgS, tSsS);
      } 
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor sZ  = make_tensor(make_smem_ptr(shared_tensors.smem_zero.begin()), SmemLayoutScale{}); // (BLK_M,BLK_K,PIPE)
        Tensor gZ_mkl = get<3>(load_inputs);
        auto block_tma_z = mainloop_params.tma_load_zero.get_slice(cluster_local_block_id.y);
        Tensor gZ = gZ_mkl(_,_,m_coord,_,l_coord);                                            // (BLK_M,BLK_K,k)

        Tensor tZgZ = block_tma_z.partition_S(gZ);                                            // (TMA,TMA_M,TMA_K,k)
        Tensor tZsZ = block_tma_z.partition_D(sZ);                                            // (TMA,TMA_M,TMA_K,PIPE)
        return cute::make_tuple(tSgS, tSsS, tZgZ, tZsZ);          
      }
      else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled for input partitioning.");      
      }
    }
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled for input partitioning.");      
    }
  }

  /// Utilities for partitioning extra inputs for loading from smem in the mainloop.
  template <
    class ThreadMma,
    class TensorStorage
  >
  CUTLASS_DEVICE 
  static auto partition_extra_mma_info(
    ThreadMma const& mma_thread_slice,
    TensorStorage& shared_tensors) {

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // nothing to do
      return cute::make_tuple();
    }
    else if constexpr (UseScaleLookupTable) {
      Tensor sS = make_tensor(make_smem_ptr(shared_tensors.smem_scale.begin()), SmemLayoutScale{});// (BLK_M,BLK_SCALE_K,PIPE)
      Tensor tCsS = mma_thread_slice.partition_A(sS);
      Tensor tCrS = make_tensor<ElementScale>(mma_thread_slice.partition_fragment_A(sS(_,_,Int<0>{})).layout()); 

      return cute::make_tuple(tCsS, tCrS);
    }
    else if constexpr (ModeHasScales) {
      Tensor sS = make_tensor(make_smem_ptr(shared_tensors.smem_scale.begin()), SmemLayoutScale{});// (BLK_M,BLK_SCALE_K,PIPE)
      Tensor tCsS = mma_thread_slice.partition_A(sS);
      Tensor tCrS = make_tensor<ElementScale>(mma_thread_slice.partition_fragment_A(sS(_,_,Int<0>{})).layout()); 

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(tCsS, tCrS);
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor sZ = make_tensor(make_smem_ptr(shared_tensors.smem_zero.begin()), SmemLayoutScale{});// (BLK_M,BLK_SCALE_K,PIPE)
        Tensor tCsZ = mma_thread_slice.partition_A(sZ);
        Tensor tCrZ = make_tensor<ElementZero>(mma_thread_slice.partition_fragment_A(sZ(_,_,Int<0>{})).layout()); 
        return cute::make_tuple(tCsS, tCrS, tCsZ, tCrZ);
      }
      else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
      }
    } 
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
    }
  }

  /// Returns the tiled copy and copy views for the extra inputs.
  template <class TiledMma, class... Ts>
  CUTLASS_DEVICE
  static auto retile_extra_mma_info(
    TiledMma const& tiled_mma,
    cute::tuple<Ts...>& partitioned_extra_info,
    int const warp_group_thread_idx) {

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // nothing to do
      return cute::make_tuple();
    }
    else if constexpr (ModeHasScales) {
      auto smem_tiled_copy_S = make_tiled_copy_A(SmemCopyAtomScale{}, tiled_mma);
      auto smem_thr_copy_S   = smem_tiled_copy_S.get_thread_slice(warp_group_thread_idx);
      Tensor tCrS_copy_view  = smem_thr_copy_S.retile_D(cute::get<1>(partitioned_extra_info));        // (CPY,CPY_M,CPY_K)
      
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(smem_tiled_copy_S, tCrS_copy_view);
      } 
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor tCrZ_copy_view  = smem_thr_copy_S.retile_D(cute::get<3>(partitioned_extra_info));      // (CPY,CPY_M,CPY_K)
        return cute::make_tuple(smem_tiled_copy_S, tCrS_copy_view, tCrZ_copy_view);
      } 
      else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
      }
    } 
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
    }
  }
};

} // cutlass::gemm::collective::detail
