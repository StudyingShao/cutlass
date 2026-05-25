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
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"
#include "cutlass/cuda_host_adapter.hpp"
#include "cutlass/detail/collective/mixed_input_utils.hpp"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
template <
  int Stages,
  class ClusterShape,
  class KernelSchedule_,
  class TileShape_,
  class ElementAOptionalTuple,
  class StrideA_,
  class ElementBOptionalTuple,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInput<Stages, ClusterShape, KernelSchedule_>,
    TileShape_,
    ElementAOptionalTuple,
    StrideA_,
    ElementBOptionalTuple,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
public:
  enum class ConversionMode {
    DirectConvert,
    ConvertAndScale,
    ConvertAndScaleWithZero
  };
  
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInput<Stages, ClusterShape, KernelSchedule_>;
  using TileShape = TileShape_;
  using KernelSchedule = KernelSchedule_;

private:
  template<class T> friend struct detail::MixedInputUtils;
  using CollectiveType = CollectiveMma<DispatchPolicy, TileShape_, 
                                       ElementAOptionalTuple, StrideA_, 
                                       ElementBOptionalTuple, StrideB_,
                                       TiledMma_, 
                                       GmemTiledCopyA_, SmemLayoutAtomA_, SmemCopyAtomA_,
                                       TransformA_,
                                       GmemTiledCopyB_, SmemLayoutAtomB_, SmemCopyAtomB_,
                                       TransformB_>;
  using Utils = detail::MixedInputUtils<CollectiveType>;

  //
  // Type Aliases
  //
  using ScaleA = detail::deduce_mixed_width_dtype_t<1, ElementAOptionalTuple>;
  using ScaleB = detail::deduce_mixed_width_dtype_t<1, ElementBOptionalTuple>;
  using ZeroA = detail::deduce_mixed_width_dtype_t<2, ElementAOptionalTuple>;
  using ZeroB = detail::deduce_mixed_width_dtype_t<2, ElementBOptionalTuple>;

public:
  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementAOptionalTuple>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementBOptionalTuple>;
  static constexpr bool IsANarrow = sizeof_bits<ElementA>::value < sizeof_bits<ElementB>::value;
  static constexpr bool HasWeightScale = !cute::is_void_v<ScaleA>;
  static constexpr bool HasZeroB = !cute::is_void_v<ZeroB>;
  static_assert(IsANarrow, "CMX mixed-input mainloop expects the first operand to be the narrow transformed weight.");
  static_assert(HasWeightScale, "The transformed weight operand must carry mixed-input scale.");
  static_assert(!HasZeroB, "Activation operand must not carry zero-point.");
  static constexpr bool IsATransformed = true;
  using ElementScale = ScaleA;
  using ElementZero = ZeroA;

  using StrideA = StrideA_;
  using InternalStrideA = cute::remove_pointer_t<StrideA>;
  using StrideB = StrideB_;
  using InternalStrideB = cute::remove_pointer_t<StrideB>;

  static constexpr bool IsMXFP4 = cute::is_same_v<ElementA, cutlass::float_e2m1_t>;
  static constexpr bool HasActivationScale =
      !cute::is_void_v<ScaleB> &&
      cute::is_same_v<ElementA, cutlass::float_e2m1_t> &&
      cute::is_same_v<ElementB, cutlass::float_e4m3_t>;
  using ElementActivationScale = cute::conditional_t<HasActivationScale, ScaleB, void>;
  // For cases where we can't have a void type, we can use this to allow the code to compile when the scale / zero is void.
  using NonVoidElementScale = cute::conditional_t<cute::is_void_v<ElementScale>, float, ElementScale>;
  using NonVoidElementZero = cute::conditional_t<cute::is_void_v<ElementZero>, float, ElementZero>;
  using NonVoidElementActivationScale =
      cute::conditional_t<cute::is_void_v<ElementActivationScale>, cutlass::float_ue8m0_t, ElementActivationScale>;
  // The GEMM kernel consumes weight scales in Ktile-major, MN-contiguous form.
  // MXFP8 activation scales stay in their raw M-major, K-contiguous form and
  // are loaded with a separate TMA descriptor.
  using StrideScale = cute::Stride<cute::Int<1>, int64_t, int64_t>;
  using NonVoidStrideScale = cute::conditional_t<cute::is_void_v<StrideScale>, cute::Stride<_1, int64_t, int64_t>, StrideScale>;
  using StrideActivationScale = cute::Stride<int64_t, cute::Int<1>, int64_t>;

  static_assert(( IsATransformed && (cutlass::gemm::detail::is_k_major<StrideA>() || is_layout<StrideA>::value || is_layout<InternalStrideA>::value)) || 
                (!IsATransformed && (cutlass::gemm::detail::is_k_major<StrideB>() || is_layout<StrideB>::value || is_layout<InternalStrideB>::value)),
                "The transformed type must be K-major.");

  static_assert(( IsATransformed && (sizeof(ElementB) == 2)) ||
                (!IsATransformed && (sizeof(ElementA) == 2)) ||
                ((cutlass::gemm::detail::is_k_major<StrideA>() || is_layout<StrideA>::value || is_layout<InternalStrideA>::value) && 
                 (cutlass::gemm::detail::is_k_major<StrideB>() || is_layout<StrideB>::value || is_layout<InternalStrideB>::value)), 
                "The unscaled element must be 2 bytes OR both inputs must be K-major");

  static_assert(cutlass::gemm::detail::is_mn_major<NonVoidStrideScale>(),
    "Scale tensor consumed by the GEMM kernel must be MN major.");

  // Group size 128 for int4 weights
  // Group size 32 for mxfp4 weights
  static constexpr int ScalingGroupSize = IsMXFP4? 32 : 128;

  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using SmemCopyAtomScale = Copy_Atom<cute::AutoVectorizingCopy, NonVoidElementScale>;

  // We must ensure the type to be scaled goes to RF
  static constexpr bool SwapAB = !IsATransformed;
  using SwappedStrideA = cute::conditional_t<!SwapAB, StrideA, StrideB>;
  using SwappedStrideB = cute::conditional_t<!SwapAB, StrideB, StrideA>;
  using InternalSwappedStrideA = cute::conditional_t<!SwapAB, InternalStrideA, InternalStrideB>;
  using InternalSwappedStrideB = cute::conditional_t<!SwapAB, InternalStrideB, InternalStrideA>;
  using SwappedSmemLayoutAtomA = cute::conditional_t<!SwapAB, SmemLayoutAtomA, SmemLayoutAtomB>;
  using SwappedSmemLayoutAtomB = cute::conditional_t<!SwapAB, SmemLayoutAtomB, SmemLayoutAtomA>;
  using SwappedSmemCopyAtomA   = cute::conditional_t<!SwapAB, SmemCopyAtomA, SmemCopyAtomB>;
  using SwappedSmemCopyAtomB   = cute::conditional_t<!SwapAB, SmemCopyAtomB, SmemCopyAtomA>;
  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  using ConvertedElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  using ConvertedElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;
  using RealSwappedElementA = cute::conditional_t<!SwapAB, ElementA, ElementB>;
  using RealSwappedElementB = cute::conditional_t<!SwapAB, ElementB, ElementA>;
  using SwappedElementA = cute::conditional_t<!SwapAB, ConvertedElementA, ConvertedElementB>;
  using SwappedElementB = cute::conditional_t<!SwapAB, ConvertedElementB, ConvertedElementA>;

  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using SwappedTransformA  = cute::conditional_t<!SwapAB, TransformA, TransformB>;
  using SwappedTransformB  = cute::conditional_t<!SwapAB, TransformB, TransformA>;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static constexpr int IsSubbyteA = cute::sizeof_bits_v<SwappedElementA> < 8;
  using TmaElementA = cute::conditional_t<IsSubbyteA, uint8_t, SwappedElementA>;
  // TmaElementScale removed: scale no longer uses TMA
  // Scale loaded via SM90_BULK_COPY_G2S (lightweight, no TMA descriptor needed)

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;
  using PipelineParams = typename MainloopPipeline::Params;

  static constexpr int NumProducerThreadEvents = 1;

  static constexpr int NumScaleChunksPerTileK = size<2>(TileShape{}) / ScalingGroupSize;
  static constexpr int ActScaleMinTmaChunks =
      128 / cutlass::sizeof_bits<NonVoidElementActivationScale>::value;
  static constexpr int ActScaleTmaChunks =
      (NumScaleChunksPerTileK > ActScaleMinTmaChunks) ? NumScaleChunksPerTileK : ActScaleMinTmaChunks;
  static_assert(ActScaleTmaChunks % NumScaleChunksPerTileK == 0,
      "Activation scale TMA window must cover an integer number of compute Ktiles.");
  static constexpr int ScaleNRawElementsPerStage = size<1>(TileShape{}) * ActScaleTmaChunks;
  static constexpr int ScaleNElementsPerStage = size<1>(TileShape{});

  using SmemLayoutAtomScale = Layout<Shape<decltype(cute::shape<0>(SwappedSmemLayoutAtomA{})), cute::Int<1>>>;
  using ScaleTileShape = decltype(make_shape(shape<0>(TileShape{}), shape<1>(SmemLayoutAtomScale{})));

  static_assert(cute::rank(SwappedSmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SwappedSmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SwappedSmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SwappedSmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SwappedSmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SwappedSmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomScale{}) == 2, "SmemLayoutAtomScale must be rank 2");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomScale{})) == 0, "SmemLayoutAtomScale must equal the tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomScale{})) == 0, "SmemLayoutAtomScale must evenly divide tile k shape.");

  /// Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(detail::get_smem_layout<DispatchPolicy::Stages>(SwappedSmemLayoutAtomA{}, select<0,2>(TileShape{}), InternalSwappedStrideA{}));
  using SmemLayoutB = decltype(detail::get_smem_layout<DispatchPolicy::Stages>(SwappedSmemLayoutAtomB{}, select<1,2>(TileShape{}), InternalSwappedStrideB{}));
  
  // It is assumed that weight scales and zero-points share the same smem layout.
  using SmemLayoutScale = decltype(tile_to_shape(
      SmemLayoutAtomScale{},
      make_shape(shape<0>(ScaleTileShape{}), shape<1>(ScaleTileShape{}), Int<Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,NonVoidStrideScale>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  // MXFP8 activation scales are independent from MXFP4 weight scales.  They are
  // stored in raw M-major, K-contiguous form and TMA-loaded into this raw scale
  // layout: (BLK_N, ActScaleTmaChunks, PIPE).  The TMA window is at least
  // 16B wide for ue8m0 scales; smaller compute Ktiles reuse a subrange.
  using SmemLayoutActivationScale = Layout<
      Shape<decltype(shape<1>(TileShape{})), Int<ActScaleTmaChunks>, Int<Stages>>,
      Stride<Int<ActScaleTmaChunks>, _1, Int<ScaleNRawElementsPerStage>>>;

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                    cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source A from rmem and B operand from smem_desc for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  // To relax them, we need to handle loading more than 1 row of scales for every main loop iteration.
  // We must also handle updating the pipeline transaction bytes on the fly.
  static_assert(size<1>(SmemLayoutAtomScale{}) == 1, "size<1>(SmemLayoutAtomScale) must be 1.");

private:
  static constexpr ConversionMode 
  get_conversion_mode() {
    if constexpr (cute::is_void_v<ElementScale>) {
      return ConversionMode::DirectConvert;
    } 
    else if constexpr (cute::is_void_v<ElementZero>) {
      return ConversionMode::ConvertAndScale;
    }
    else {
      return ConversionMode::ConvertAndScaleWithZero;
    }
  }  

  bool TensormapUpdateShapesStridesForAandScale = true;
  int current_group_idx_ = 0;

public:
  static constexpr ConversionMode KernelConversionMode = get_conversion_mode();
  static constexpr bool ModeHasScales = KernelConversionMode == ConversionMode::ConvertAndScale ||
                                        KernelConversionMode == ConversionMode::ConvertAndScaleWithZero;
  static constexpr bool UseScaleLookupTable = KernelConversionMode == ConversionMode::ConvertAndScale &&
                                              cutlass::detail::is_Array_v<ElementScale>;
  static constexpr bool UseFP4ToBF16LookupTable = KernelConversionMode == ConversionMode::ConvertAndScale &&
                                                  cute::is_same_v<ElementA, cutlass::float_e2m1_t> &&
                                                  cute::is_same_v<ElementB, cutlass::bfloat16_t>;
  static constexpr bool UseFP4ToFP8LookupTable = KernelConversionMode == ConversionMode::ConvertAndScale &&
                                                 cute::is_same_v<ElementA, cutlass::float_e2m1_t> &&
                                                 cute::is_same_v<ElementB, cutlass::float_e4m3_t>;
  static constexpr bool UseInt4ToFP8LookupTable = KernelConversionMode == ConversionMode::ConvertAndScale &&
                                                  cute::is_same_v<ElementA, cutlass::int4_t> &&
                                                  cute::is_same_v<ElementB, cutlass::float_e4m3_t>;
  static constexpr size_t SmemAlignmentA = cutlass::detail::alignment_for_swizzle(SmemLayoutA{}); 
  static constexpr size_t SmemAlignmentB = cutlass::detail::alignment_for_swizzle(SmemLayoutB{});
  static constexpr size_t SmemAlignmentScale = cute::max(SmemAlignmentA, SmemAlignmentB);

  static_assert(SmemAlignmentA >= 128 and SmemAlignmentB >= 128, "Require at least 128B alignment");

  struct SharedStorage {
    static constexpr int scale_elements = Utils::elements_per_smem_scale();
    static constexpr int zero_elements = Utils::elements_per_smem_zero();
    static constexpr int activation_scale_elements = HasActivationScale ? cute::cosize_v<SmemLayoutActivationScale> : 0;
    struct TensorStorage {
      CUTE_ALIGNAS(SmemAlignmentA) cute::ArrayEngine<RealSwappedElementA, cute::cosize_v<SmemLayoutA>> smem_A;
      CUTE_ALIGNAS(SmemAlignmentB) cute::ArrayEngine<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
      cute::ArrayEngine<NonVoidElementScale, scale_elements> smem_scale;
      cute::ArrayEngine<NonVoidElementActivationScale, activation_scale_elements> smem_activation_scale;
      cute::ArrayEngine<NonVoidElementZero, zero_elements> smem_zero;
    } tensors;

    struct TensorMapStorage {
      cute::TmaDescriptor smem_tensormap_A;
      cute::TmaDescriptor smem_tensormap_B;
      using ActivationScaleTmaDescriptor = cute::conditional_t<HasActivationScale, cute::TmaDescriptor, cute::tuple<>>;
      ActivationScaleTmaDescriptor smem_tensormap_activation_scale;
    };

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using TensorMapStorage = typename SharedStorage::TensorMapStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideA, StrideA>;

  // kernel Arguments
  // Host side kernel arguments
  struct Arguments {
    ElementA const** ptr_A;
    StrideA dA;
    ElementB const** ptr_B;
    StrideB dB;
    ElementScale const** ptr_S = nullptr;
    NonVoidStrideScale const* dS{};
    int chunk_size = 0;
    ElementZero const** ptr_Z = nullptr;
    NonVoidElementActivationScale const** ptr_ActivationScale = nullptr;
    StrideActivationScale const* dActivationScale{};
  };

  // Device side kernel params
  struct Params {
    // For grouped GEMM with non-layout stride: replace static-zero L stride (_0) with
    // a static non-zero value so the TMA descriptor includes the L dimension at creation.
    // Int<32> is the minimum static value that after subbyte upcast<2> (FP4→uint8_t)
    // produces Int<16> = 16 bytes, satisfying cuTensorMapEncodeTiled's 16-byte alignment.
    // Being fully static, all CuTe coordinate computations remain compile-time optimizable.
    using TmaStrideA = cute::conditional_t<
        IsGroupedGemmKernel && !cute::is_layout<InternalSwappedStrideA>::value,
        decltype(cute::make_stride(
            cute::get<0>(InternalSwappedStrideA{}),
            cute::get<1>(InternalSwappedStrideA{}),
            cute::Int<32>{})),
        InternalSwappedStrideA>;
    using LayoutA = decltype(detail::get_gmem_layout(repeat_like(TmaStrideA{}, int32_t(0)), TmaStrideA{}));
    using TmaStrideB = cute::conditional_t<
        IsGroupedGemmKernel && !cute::is_layout<InternalSwappedStrideB>::value,
        decltype(cute::make_stride(
            cute::get<0>(InternalSwappedStrideB{}),
            cute::get<1>(InternalSwappedStrideB{}),
            cute::Int<16>{})),
        InternalSwappedStrideB>;
    using LayoutB = decltype(detail::get_gmem_layout(repeat_like(TmaStrideB{}, int32_t(0)), TmaStrideB{}));

    using TMA_A = decltype(make_tma_copy<TmaElementA>(
        GmemTiledCopyA{},
        make_tensor(detail::get_logical_ptr(static_cast<SwappedElementA const*>(nullptr)), LayoutA{}),
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{})));  // mcast along N mode for this M load, if any
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy(
        GmemTiledCopyB{},
        make_tensor(detail::get_logical_ptr(static_cast<SwappedElementB const*>(nullptr)), LayoutB{}),
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any
    using LayoutActivationScale = decltype(detail::get_gmem_layout(repeat_like(StrideActivationScale{}, int32_t(0)), StrideActivationScale{}));
    using TMA_ActivationScale_ = decltype(make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(detail::get_logical_ptr(static_cast<NonVoidElementActivationScale const*>(nullptr)), LayoutActivationScale{}),
        SmemLayoutActivationScale{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), Int<ActScaleTmaChunks>{}),
        Int<1>{}));
    using TMA_ActivationScale = cute::conditional_t<HasActivationScale, TMA_ActivationScale_, cute::tuple<>>;

    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_ActivationScale tma_load_activation_scale;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    void* tensormaps;
    SwappedElementA const** ptr_A;
    SwappedStrideA ptr_dA;
    SwappedElementB const** ptr_B;
    SwappedStrideB ptr_dB;
    NonVoidElementScale const** ptr_S;
    NonVoidStrideScale const* dS;
    NonVoidElementActivationScale const** ptr_ActivationScale;
    StrideActivationScale const* dActivationScale;
    NonVoidElementZero const** ptr_Z;
    int64_t scale_k;
    int chunk_size;
    int reload_factor = (chunk_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{});
    InternalSwappedStrideA dA;
    InternalSwappedStrideB dB;
    int num_groups;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape problem_shapes,
      Arguments const& args,
      void* workspace) {

    // These tensor shapes (only applicable for grouped gemm) and pointers are only used to create tensormap/tma desc.
    // These will be replaced with correct values before the initial tma load.
    auto init_shape = repeat_like(typename ProblemShape::UnderlyingProblemShape{}, int32_t(1));
    auto init_M = get<0>(init_shape);
    auto init_N = get<1>(init_shape);
    auto init_K = get<2>(init_shape);

    if constexpr (SwapAB) {
      init_M = get<1>(init_shape);
      init_N = get<0>(init_shape);
    }
    // Batches/Groups are managed by using appropriate pointers to input matrices
    const uint32_t mock_L = 1;
    SwappedElementA const* ptr_A_first_batch;
    SwappedElementB const* ptr_B_first_batch;
    NonVoidElementActivationScale const* ptr_activation_scale_first_batch =
        reinterpret_cast<NonVoidElementActivationScale const*>(args.ptr_ActivationScale);
    SwappedStrideA ptr_dA;
    SwappedStrideB ptr_dB;
    InternalSwappedStrideA dA;
    InternalSwappedStrideB dB;

    if constexpr (not SwapAB) {
      ptr_A_first_batch = reinterpret_cast<SwappedElementA const*>(args.ptr_A);
      ptr_B_first_batch = reinterpret_cast<SwappedElementB const*>(args.ptr_B);
    }
    else {
      ptr_A_first_batch = reinterpret_cast<SwappedElementA const*>(args.ptr_B);
      ptr_B_first_batch = reinterpret_cast<SwappedElementB const*>(args.ptr_A);
    }

    if constexpr (IsGroupedGemmKernel) {
      // Strides for Grouped Gemm will be replaced prior to the first access regardless.
      if constexpr (not SwapAB) {
        ptr_dA = args.dA;
        ptr_dB = args.dB;
      }
      else {
        ptr_dA = args.dB;
        ptr_dB = args.dA;
      }
      dA = InternalSwappedStrideA{};
      if constexpr (is_layout<InternalSwappedStrideA>::value) {
        dA = make_layout(
          transform_leaf(dA.shape(), [](auto x){ 
            if constexpr (not is_static_v<decltype(x)>) {
              return static_cast<decltype(x)>(1);
            } else {
              return x;
            }
          }),
          dA.stride());
      }
      dB = InternalSwappedStrideB{};
    }
    else {
      // Tensor shapes for Ptr-Array are initialized correctly only here.
      auto problem_shape_MNK = problem_shapes.get_host_problem_shape(0);
      init_M = get<0>(problem_shape_MNK);
      init_N = get<1>(problem_shape_MNK);
      init_K = get<2>(problem_shape_MNK);

      if constexpr (not SwapAB) {
        dA = args.dA;
        dB = args.dB;
      }
      else {
        dA = args.dB;
        dB = args.dA;
      }
      ptr_dA = SwappedStrideA{};
      ptr_dB = SwappedStrideB{};
    }
    // For grouped GEMM: use TmaStrideA (with static _1 L stride) so the TMA descriptor
    // is created as 3D, enabling coordinate-based group selection.
    typename Params::TmaStrideA tma_dA;
    if constexpr (!IsGroupedGemmKernel || cute::is_layout<InternalSwappedStrideA>::value) {
      tma_dA = dA;
    }
    Tensor tensor_a = make_tensor(ptr_A_first_batch, detail::get_gmem_layout(make_shape(init_M,init_K,mock_L), tma_dA));
    typename Params::TmaStrideB tma_dB;
    if constexpr (!IsGroupedGemmKernel || cute::is_layout<InternalSwappedStrideB>::value) {
      tma_dB = dB;
    }
    Tensor tensor_b = make_tensor(ptr_B_first_batch, detail::get_gmem_layout(make_shape(init_N,init_K,mock_L), tma_dB));

    typename Params::TMA_A tma_load_a = make_tma_copy<TmaElementA>(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{})); // mcast along N mode for this M load, if any
    typename Params::TMA_B tma_load_b = make_tma_copy(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{})); // mcast along M mode for this N load, if any
    typename Params::TMA_ActivationScale tma_load_activation_scale{};
    if constexpr (HasActivationScale) {
      Tensor tensor_activation_scale = make_tensor(
          ptr_activation_scale_first_batch,
          detail::get_gmem_layout(make_shape(init_N, int32_t(ActScaleTmaChunks), mock_L), StrideActivationScale{}));
      tma_load_activation_scale = make_tma_copy(
          SM90_TMA_LOAD{},
          tensor_activation_scale,
          SmemLayoutActivationScale{}(_,_,cute::Int<0>{}),
          make_shape(shape<1>(TileShape{}), Int<ActScaleTmaChunks>{}),
          Int<1>{});
    }

    void* tensormaps = workspace;
    int num_groups_val = 1;
    if constexpr (IsGroupedGemmKernel) {
      num_groups_val = problem_shapes.groups();
    }
    auto args_setup = [&](auto ptr_A, auto ptr_B, int64_t scale_k = 0, int chunk_size = 0, int reload_factor = 1) -> Params {
      return {
          tma_load_a,
          tma_load_b,
          tma_load_activation_scale,
          TmaTransactionBytes,
          tensormaps,
          reinterpret_cast<SwappedElementA const**>(ptr_A),
          ptr_dA,
          reinterpret_cast<SwappedElementB const**>(ptr_B),
          ptr_dB,
          reinterpret_cast<NonVoidElementScale const**>(args.ptr_S),
          args.dS,
          args.ptr_ActivationScale,
          args.dActivationScale,
          reinterpret_cast<NonVoidElementZero const**>(args.ptr_Z),
          scale_k,
          chunk_size,
          reload_factor,
          dA,
          dB,
          num_groups_val
      };
    };

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return SwapAB ? args_setup(args.ptr_B, args.ptr_A)
                    : args_setup(args.ptr_A, args.ptr_B);
    }
    else if constexpr (ModeHasScales) {
      auto fake_scale_k = 1;
      return SwapAB ? args_setup(args.ptr_B, args.ptr_A, fake_scale_k, args.chunk_size, (args.chunk_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{}))
                    : args_setup(args.ptr_A, args.ptr_B, fake_scale_k, args.chunk_size, (args.chunk_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{}));
    } 
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in to_underlying_arguments.");
    } 
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int sm_count) {
    constexpr size_t SizeOfCuTensorMap = sizeof(cute::TmaDescriptor);

    // Calculating workspace size
    auto calculate_workspace_size = [SizeOfCuTensorMap, sm_count](uint32_t num_input_tensors) {
        return num_input_tensors * SizeOfCuTensorMap * sm_count;
    };

    // A/B always use TMA; MXFP8 activation scale adds a third descriptor.
    return calculate_workspace_size(HasActivationScale ? 3 : 2);
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream, CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }


  template<class ProblemShape>
  CUTLASS_HOST_DEVICE static bool
  can_implement(
      ProblemShape problem_shapes,
      Arguments const& args) {
    constexpr int tma_alignment_bits = 128;
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;

    bool implementable = true;
    if (problem_shapes.is_host_problem_shape_available()) {
      // Check alignment for all problem sizes
      for (int i = 0; i < problem_shapes.groups(); i++) {
        auto problem_shape_MNKL = append<4>(problem_shapes.get_host_problem_shape(i), 1);
        auto [M,N,K,L] = problem_shape_MNKL;
        auto get_stride = [](auto stride) {
          if constexpr (cute::is_pointer_v<cute::decay_t<decltype(stride)>>) {
            return *stride;
          }
          else {
            return stride;
          }
        };
        auto dA = get_stride(args.dA);
        auto dB = get_stride(args.dB);
        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(detail::get_gmem_layout(cute::make_shape(M,K,L), dA));
        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(detail::get_gmem_layout(cute::make_shape(N,K,L), dB));
        if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
          implementable = implementable && (args.ptr_S == nullptr);
          implementable = implementable && (args.ptr_Z == nullptr);
        }
        else if constexpr (ModeHasScales) {
          const int scale_mn = SwapAB ? N : M;
          const int scale_k = (K + args.chunk_size - 1) / args.chunk_size;
          constexpr int min_tma_aligned_elements_scale = tma_alignment_bits / cutlass::sizeof_bits<ElementScale>::value;
          implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_scale>(cute::make_shape(scale_mn,scale_k,L), StrideScale{});
          implementable = implementable && (args.chunk_size == K || ((args.chunk_size % size<2>(TileShape{})) == 0));
          implementable = implementable && args.chunk_size != 0;
          implementable = implementable && (args.ptr_S != nullptr);
          if constexpr (HasActivationScale) {
            implementable = implementable && (args.ptr_ActivationScale != nullptr);
          }
          if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
            implementable = implementable && (args.ptr_Z == nullptr);
          }
          else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
            constexpr int min_tma_aligned_elements_zero = tma_alignment_bits / cutlass::sizeof_bits<ElementZero>::value;
            implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_zero>(cute::make_shape(scale_mn,scale_k,L), StrideScale{});
            implementable = implementable && (args.ptr_Z != nullptr);
          } 
          else {
            static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in can_implement.");
          }
        }
        else {
          static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in can_implement.");
        }
      }
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr int K_PIPE_MMAS = 1;
  static constexpr uint32_t TmaTransactionBytesMK = Utils::compute_tma_transaction_bytes_mk();
  static constexpr uint32_t TmaTransactionBytesNK = Utils::compute_tma_transaction_bytes_nk();
  static constexpr uint32_t TmaTransactionBytesExtra = Utils::compute_tma_transaction_bytes_extra();
  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK + TmaTransactionBytesExtra;

  // Set up the data needed by this collective for load and mma.
  // Returns a tuple of tensors. The collective and the kernel layer have the contract that the
  // returned tuple must contain at least two elements, with the first two elements being:
  // gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
  // gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
  // The rest of the tensors can be specified as needed by this collective.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;
    const int32_t mock_L = 1;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    // For grouped GEMM: A uses L=num_groups so all groups are accessible via TMA L coordinate
    auto A_L = IsGroupedGemmKernel ? mainloop_params.num_groups : mock_L;
    auto B_L = IsGroupedGemmKernel ? mainloop_params.num_groups : mock_L;
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(shape(detail::get_gmem_layout(make_shape(M,K,A_L), mainloop_params.dA))); // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(shape(detail::get_gmem_layout(make_shape(N,K,B_L), mainloop_params.dB))); // (n,k,l)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});  // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});  // (BLK_N,BLK_K,n,k,l)

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return cute::make_tuple(gA_mkl, gB_nkl);
    } 
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      // Scale ptr/stride placeholders — set correctly by tensors_perform_update before first load()
      if constexpr (HasActivationScale) {
        Tensor mActS_nkl = mainloop_params.tma_load_activation_scale.get_tma_tensor(
            shape(detail::get_gmem_layout(make_shape(N, K / ScalingGroupSize, B_L), StrideActivationScale{}))); // (n,k_scale,l)
        Tensor gActS_nkl = local_tile(
            mActS_nkl,
            make_shape(shape<1>(TileShape{}), Int<ActScaleTmaChunks>{}, Int<1>{}),
            make_coord(_,_,_),
            Step<_1,_1, X>{});  // (BLK_N, TMA_SCALE_K, n, scale_window, l)
        return cute::make_tuple(gA_mkl, gB_nkl,
            static_cast<NonVoidElementScale const*>(nullptr), int64_t(0),
            gActS_nkl);
      }
      else {
        return cute::make_tuple(gA_mkl, gB_nkl,
            static_cast<NonVoidElementScale const*>(nullptr), int64_t(0));
      }
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      return cute::make_tuple(gA_mkl, gB_nkl,
          static_cast<NonVoidElementScale const*>(nullptr), int64_t(0),
          static_cast<NonVoidElementZero const*>(nullptr));
    }
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in load_init.");
    }
  }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Perform a collective-scoped matrix multiply-accumulate
  // Producer Perspective
  template <
    class... Ts,
    class... TMs,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      MainloopPipeline pipeline, 
      PipelineState smem_pipe_write,
      cute::tuple<Ts...> const& load_inputs,
      cute::tuple<TMs...> const& input_tensormaps,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      static_assert(sizeof... (Ts) == 2, "Direct convert needs two inputs");
    } 
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      if constexpr (HasActivationScale) {
        static_assert(sizeof... (Ts) == 5, "MXFP8 activation scale needs five inputs (gA, gB, weight_scale_ptr, weight_stride_k, gActS)");
      }
      else {
        static_assert(sizeof... (Ts) == 4, "Scaled convert needs four inputs (gA, gB, scale_ptr, stride_k)");
      }
    } 
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      static_assert(sizeof... (Ts) == 5, "Scaled+zero convert needs five inputs (gA, gB, scale_ptr, stride_k, zero_ptr)");
    } 
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in load.");
    }
    if constexpr (HasActivationScale) {
      static_assert(sizeof... (TMs) == 3, "A, B, and activation scale tensormaps needed");
    }
    else {
      static_assert(sizeof... (TMs) == 2, "Only A and B tensormaps needed");
    }

    Tensor sA_ = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB_ = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)
    Tensor sA  = as_position_independent_swizzle_tensor(sA_);                                       // (BLK_M,BLK_K,PIPE)
    Tensor sB  = as_position_independent_swizzle_tensor(sB_);                                       // (BLK_N,BLK_K,PIPE)

    //
    // Prepare the TMA loads for A and B
    //

    constexpr uint32_t cluster_shape_x = get<0>(typename DispatchPolicy::ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
    auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

    // Partition the inputs based on the current block coordinates.
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    // For grouped GEMM: A selects group via L coordinate instead of descriptor update
    auto a_l_coord = IsGroupedGemmKernel ? current_group_idx_ : l_coord;
    auto b_l_coord = IsGroupedGemmKernel ? current_group_idx_ : l_coord;
    Tensor gA = gA_mkl(_,_,m_coord,_,a_l_coord);                                                   // (BLK_M,BLK_K,k)
    Tensor gB = gB_nkl(_,_,n_coord,_,b_l_coord);                                                   // (BLK_N,BLK_K,k)

    // Applies the mapping from block_tma_a
    Tensor tAgA = block_tma_a.partition_S(gA);                                                 // (TMA,TMA_M,TMA_K,k)
    Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

    Tensor tBgB = block_tma_b.partition_S(gB);                                                 // (TMA,TMA_N,TMA_K,k)
    Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

    uint16_t mcast_mask_a = 0;
    uint16_t mcast_mask_b = 0;

    // Issue TmaLoads
    // Maps the tile -> block, value
    if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
      for (int n = 0; n < size<1>(block_layout); ++n) {
        mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x,n,Int<0>{}));
      }
    }

    if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
      for (int m = 0; m < size<0>(block_layout); ++m) {
        mcast_mask_b |= (uint16_t(1) << block_layout(m,cluster_local_block_id.y,Int<0>{}));
      }
    }

    // Prepare SMEM tensors for scale/zero bulk copy (if needed)
    [[maybe_unused]] Tensor sS = make_tensor(make_smem_ptr(shared_tensors.smem_scale.begin()), SmemLayoutScale{});
    [[maybe_unused]] Tensor sZ = [&]() {
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        return make_tensor(make_smem_ptr(shared_tensors.smem_zero.begin()), SmemLayoutScale{});
      } else {
        return make_tensor(make_smem_ptr(shared_tensors.smem_scale.begin()), SmemLayoutScale{}); // dummy
      }
    }();

    // Mainloop
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count)
    {
      // LOCK smem_pipe_write for _writing_
      pipeline.producer_acquire(smem_pipe_write);

      //
      // Copy gmem to smem for *k_tile_iter
      //

      using BarrierType = typename MainloopPipeline::ProducerBarrierType;
      BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

      int write_stage = smem_pipe_write.index();
      if (cute::elect_one_sync()) {
        // TMA for A and B
        copy(mainloop_params.tma_load_a.with(get<0>(input_tensormaps), *tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        copy(mainloop_params.tma_load_b.with(get<1>(input_tensormaps), *tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));

        // Bulk copy for scale/zero (lightweight, no TMA descriptor)
        if constexpr (ModeHasScales) {
          auto scale_ptr = get<2>(load_inputs);
          auto scale_stride_k = get<3>(load_inputs);
          const int scale_k_tile = *k_tile_iter;
          constexpr int BLK_M = size<0>(TileShape{});
          constexpr int scale_load_bytes = BLK_M * sizeof(NonVoidElementScale);

          auto* scale_gmem_addr = reinterpret_cast<void const*>(
              scale_ptr + m_coord * BLK_M + scale_k_tile * scale_stride_k);
          auto* scale_smem_addr = static_cast<void*>(&sS(0, 0, write_stage));
          cute::SM90_BULK_COPY_G2S::copy(scale_gmem_addr,
              reinterpret_cast<uint64_t*>(tma_barrier), scale_smem_addr, scale_load_bytes);

          if constexpr (HasActivationScale) {
            Tensor sActS = make_tensor(
                make_smem_ptr(shared_tensors.smem_activation_scale.begin()), SmemLayoutActivationScale{});
            Tensor gActS_nkl = get<4>(load_inputs);
            auto block_tma_activation_scale = mainloop_params.tma_load_activation_scale.get_slice(Int<0>{});
            auto act_l_coord = IsGroupedGemmKernel ? current_group_idx_ : l_coord;
            int act_scale_window = (scale_k_tile * NumScaleChunksPerTileK) / ActScaleTmaChunks;
            Tensor gActS = gActS_nkl(_,_,n_coord,act_scale_window,act_l_coord);
            Tensor tActSgActS = block_tma_activation_scale.partition_S(gActS);
            Tensor tActSsActS = block_tma_activation_scale.partition_D(sActS);
            copy(mainloop_params.tma_load_activation_scale.with(get<2>(input_tensormaps), *tma_barrier),
                 tActSgActS,
                 tActSsActS(_,_,_,write_stage));
          }

          if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
            auto zero_ptr = get<4>(load_inputs);
            constexpr int zero_load_bytes = BLK_M * sizeof(NonVoidElementZero);

            auto* zero_gmem_addr = reinterpret_cast<void const*>(
                zero_ptr + m_coord * BLK_M + scale_k_tile * scale_stride_k);
            auto* zero_smem_addr = static_cast<void*>(&sZ(0, 0, write_stage));
            cute::SM90_BULK_COPY_G2S::copy(zero_gmem_addr,
                reinterpret_cast<uint64_t*>(tma_barrier), zero_smem_addr, zero_load_bytes);
          }
        }
      }
      ++k_tile_iter;

      // Advance smem_pipe_write
      ++smem_pipe_write;
    }
  }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      // This helps avoid early exit of blocks in Cluster.
      // Waits for all stages to either be released (all 
      // Consumer UNLOCKs), or if the stage was never used
      // then it would just be acquired since the phase was 
      // still inverted from make_producer_start_state.
      pipeline.producer_tail(smem_pipe_write);
    }
  }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  template <class T>
  CUTLASS_DEVICE float 
  scale_convertor(T scale) {
    if constexpr (cute::is_same_v<ElementA, cutlass::float_e2m1_t>) {

      cutlass::float_ue8m0_t scale_ue8m0 = scale;

      uint32_t temp = 0;
      temp = (temp | *reinterpret_cast<uint8_t*>(&scale_ue8m0)) << 23;
      return *reinterpret_cast<float*>(&temp);
    }
    else {
      return static_cast<float>(scale);
    }
  }

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
    
    CUTLASS_PRAGMA_UNROLL
    for (int mma_m = 0; mma_m < size<1>(accum); mma_m++) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < size<0, 1>(accum); m++) {
        
        float scale_val = scale_convertor(tCrS(make_coord(make_tuple(0, m, 0), mma_m, 0))[scale_idx]);
        
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < size<0, 2>(accum); n++) {
          CUTLASS_PRAGMA_UNROLL
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

  template <class TiledMma, class TensorStorage, class ScaleTensor>
  CUTLASS_DEVICE void
  copy_activation_scale_for_chunk(
      TiledMma const& tiled_mma,
      int thread_idx,
      TensorStorage& shared_tensors,
      ScaleTensor& tCrScaleN,
      int scale_idx,
      int scale_window_offset,
      int read_stage)
  {
    using ElementScaleN = NonVoidElementActivationScale;
    auto scale_smem_ptr = make_smem_ptr(
        reinterpret_cast<ElementScaleN *>(shared_tensors.smem_activation_scale.begin()) +
        scale_window_offset + scale_idx);
    Tensor sScaleNViewAsC = make_tensor(
        scale_smem_ptr,
        Layout<
            Shape<decltype(shape<0>(TileShape{})), decltype(shape<1>(TileShape{})), Int<DispatchPolicy::Stages>>,
            Stride<_0, Int<ActScaleTmaChunks>, Int<ScaleNRawElementsPerStage>>>{});
    Tensor tCsScaleNViewAsC = tiled_mma.get_slice(thread_idx).partition_C(sScaleNViewAsC);
    copy(tCsScaleNViewAsC(_, _, _, read_stage), tCrScaleN);
  }

  template <class AccumTensor, class IntermTensor, class ScaleTensorM, class ScaleTensorN>
  CUTLASS_DEVICE void
  apply_groupwise_scale_mn(
      AccumTensor& accum,
      IntermTensor const& intermediate,
      ScaleTensorM const& tCrScaleM,
      ScaleTensorN const& tCrScaleN,
      int scale_idx,
      bool is_first_accum)
  {
    multiply_add<ElementAccumulator> fma_op;

    CUTLASS_PRAGMA_UNROLL
    for (int mma_m = 0; mma_m < size<1>(accum); mma_m++) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < size<0, 1>(accum); m++) {
        float scale_m = scale_convertor(tCrScaleM(make_coord(make_tuple(0, m, 0), mma_m, 0))[scale_idx]);
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < size<0, 2>(accum); n++) {
          CUTLASS_PRAGMA_UNROLL
          for (int e = 0; e < size<0, 0>(accum); e++) {
            auto coord = make_coord(make_tuple(e, m, n), mma_m, 0);
            float scale_val = scale_m * scale_convertor(tCrScaleN(coord));
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

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgTensorC
  >
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& mainloop_params) {

    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SwappedSmemLayoutAtomA{}) == 2, "SwappedSmemLayoutAtomA must be rank 2.");
    static_assert(cute::rank(SwappedSmemLayoutAtomB{}) == 2, "SwappedSmemLayoutAtomB must be rank 2.");
    static_assert(!cute::is_void_v<SwappedSmemCopyAtomA>,
      "SM90 GMMA mainloops must specify a non-void copy atom for smem sourced instructions.");
    static_assert(cute::is_void_v<SwappedSmemCopyAtomB>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    // Obtain warp index
    int warp_idx = canonical_warp_idx_sync();
    [[maybe_unused]] int warp_group_thread_idx = thread_idx % 128;

    
    Tensor sA_ = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
    Tensor sA = as_position_independent_swizzle_tensor(sA_);                                      // (BLK_M,BLK_K,PIPE)

    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});         // (BLK_N,BLK_K,PIPE)

    //
    // Define C accumulators and A/B partitioning
    //

    // Layout of warp group to thread mapping

    static_assert(stride<0>(typename TiledMma::BLayout{}) == 0 and
                  size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup, 
                  "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");

    constexpr int MmaWarpGroups = size(TiledMma{}) / NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{},
                                                  Int<NumThreadsPerWarpGroup>{});

    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);

    TiledMma tiled_mma;
    auto mma_thread_slice = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCsA = mma_thread_slice.partition_A(sA);
    auto mma_warpgroup_slice = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

    // Allocate fragments and descriptors
    Tensor tCrA_mma = mma_thread_slice.partition_fragment_A(sA(_,_,Int<0>{}));                // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrA_load = [&]{
      if constexpr (not is_layout<InternalSwappedStrideA>::value) {
        // Make register tensor with MMA layout
        return make_fragment_like<RealSwappedElementA>(tCrA_mma);
      }
      else {
        // Make register tensor matching smem layout, converter will take care of de-swizzling
        return make_tensor_like<RealSwappedElementA>(tCsA(_,_,_,Int<0>{}));
      }
    }();
    Tensor tCsB = mma_warpgroup_slice.partition_B(sB);                                        // (MMA,MMA_N,MMA_K,PIPE)
    // tCrB is just a view of the tensor tCsB
    Tensor tCrB = mma_warpgroup_slice.make_fragment_B(tCsB);                                  // (MMA,MMA_N,MMA_K,PIPE)

    //
    // Copy Atom A retiling
    //
    auto smem_tiled_copy_A = make_tiled_copy_A(SwappedSmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(warp_group_thread_idx);

    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA_load);                                  // (CPY,CPY_M,CPY_K)

    // Partition of thread -> shared and thread -> RF
    auto partitioned_extra_info = Utils::partition_extra_mma_info(mma_thread_slice, shared_tensors);
    auto copy_partitions_extra_info = Utils::retile_extra_mma_info(tiled_mma, partitioned_extra_info, warp_group_thread_idx);
    auto tCrScaleN = [&] {
      if constexpr (HasActivationScale) {
        using ElementScaleN = NonVoidElementActivationScale;
        Tensor sScaleNViewAsC = make_tensor(
            make_smem_ptr(reinterpret_cast<ElementScaleN *>(shared_tensors.smem_activation_scale.begin())),
            Layout<
                Shape<decltype(shape<0>(TileShape{})), decltype(shape<1>(TileShape{})), Int<DispatchPolicy::Stages>>,
                Stride<_0, Int<ActScaleTmaChunks>, Int<ScaleNRawElementsPerStage>>>{});
        Tensor tCsScaleNViewAsC = tiled_mma.get_slice(thread_idx).partition_C(sScaleNViewAsC);
        return make_tensor_like<ElementScaleN>(tCsScaleNViewAsC(_, _, _, Int<0>{}));
      }
      else {
        return make_tensor<ElementScale>(make_shape(Int<1>{}));
      }
    }();
    auto copy_scale_for_tile = [&](int read_stage) {
      Utils::copy_tensors_SFA(partitioned_extra_info, copy_partitions_extra_info, 0, read_stage);
    };
    cute::array<decltype(tCrScaleN), 2> tCrScaleN_pipe;
    auto copy_activation_scale_for_pipeline = [&](int scale_idx, int read_stage, int current_k_tile) {
      if constexpr (HasActivationScale) {
        int scale_window_offset =
            (current_k_tile * NumScaleChunksPerTileK) % ActScaleTmaChunks;
        copy_activation_scale_for_chunk(
            tiled_mma, thread_idx, shared_tensors, tCrScaleN_pipe[scale_idx & 1],
            scale_idx, scale_window_offset, read_stage);
      }
    };
    auto scale_intermediate = [&](auto const& intermediate, int scale_idx, bool is_first_accum) {
      if constexpr (HasActivationScale) {
        apply_groupwise_scale_mn(accum, intermediate,
            cute::get<1>(partitioned_extra_info), tCrScaleN_pipe[scale_idx & 1], scale_idx, is_first_accum);
      }
      else {
        apply_groupwise_scale(accum, intermediate,
            cute::get<1>(partitioned_extra_info), scale_idx, is_first_accum);
      }
    };

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));                                            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));                                            // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCrA_mma) == size<1>(accum));                                                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                                                         // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                                          // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                                       // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                         // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                         // PIPE

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    using SmemCopyAtomA_LDSM = Copy_Atom<SM75_U32x4_LDSM_N, ElementB>;
  
    auto smem_tiled_copy_A_LDSM = make_tiled_copy_A(SmemCopyAtomA_LDSM{}, tiled_mma);
    auto smem_thr_copy_A_LDSM   = smem_tiled_copy_A_LDSM.get_thread_slice(thread_idx);
    
    Tensor sA_LDSM = recast<ElementB>(sA);
    auto tCsA_LDSM   = smem_thr_copy_A_LDSM.partition_S(sA_LDSM);

    using ABBitWidthRatio = Int<sizeof_bits_v<ElementB> / sizeof_bits_v<ElementA>>;
    auto tCrA_load_LDSM_shape = replace<2>(tCrA_mma.shape(), size(get<2>(tCrA_mma.shape())) / ABBitWidthRatio{});
    Tensor tCrA_load_LDSM = make_fragment_like<ElementB>(tCrA_load_LDSM_shape);
    Tensor tCrA_copy_view_LDSM  = smem_thr_copy_A_LDSM.retile_D(tCrA_load_LDSM); // (CPY,CPY_M,CPY_K)
    
    auto ptr = recast_ptr<RealSwappedElementA>(tCrA_load_LDSM.data());
    auto old_shape = tCrA_load_LDSM.shape();
    auto new_shape = make_shape(size<0>(old_shape), get<1>(old_shape), size<2>(old_shape) * ABBitWidthRatio{});
    Tensor tCrA_load_4b_packed = make_tensor(ptr, make_layout(new_shape));

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    //
    // PIPELINED MAIN LOOP
    //

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    constexpr int NumMMAsPerChunk = ScalingGroupSize / cute::get<0, 1>(tCsB.shape())();
    constexpr int NumChunksPerTileK = cute::size<1>(sA.shape())() / ScalingGroupSize;
    cute::array<decltype(make_fragment_like(accum)) , NumChunksPerTileK> intermediate_array;

    constexpr int K_BLOCK_MAX = size<2>(tCrA_load);
    constexpr int K_WAIT_MAX = cute::min(K_BLOCK_MAX - 1, 7);
    static_assert(K_BLOCK_MAX >= 4, "Consider increasing TileShapeK");

    ConsumerToken barrier_token = {BarrierStatus::WaitAgain};
    int const k_tile_count_initial = k_tile_count;
    // First k tile
    {
      barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      int read_stage = smem_pipe_read.index();
      int const current_k_tile = k_tile_count_initial - k_tile_count;

      ++smem_pipe_read;
      barrier_token = pipeline.consumer_try_wait(smem_pipe_read);

      Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, 0, read_stage);
      if (K_BLOCK_MAX > 1) {
        Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, 1, read_stage);
      }
      
      // src: tCrA_load, dst: tCrA_mma
      Utils::convert_A_kblock(tCrA_load_4b_packed, tCrA_mma, 0);

      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int chunk_id = 0; chunk_id < NumChunksPerTileK; ++chunk_id) {
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

        CUTLASS_PRAGMA_UNROLL
        for (int mma_id = 0; mma_id < NumMMAsPerChunk; ++mma_id) {
          int k_block = chunk_id * NumMMAsPerChunk + mma_id;

          warpgroup_arrive();
            
          // (V,M) x (V,N) => (V,M,N)
          cute::gemm(tiled_mma, tCrA_mma(_,_,k_block), tCrB(_,_,k_block,read_stage), intermediate_array[chunk_id]);
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;

          if (k_block == 0) {
            copy_scale_for_tile(read_stage);
          }
          if (mma_id == 0) {
            copy_activation_scale_for_pipeline(chunk_id, read_stage, current_k_tile);
          }

          if (k_block < K_BLOCK_MAX - 2) {
            Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, k_block + 2, read_stage);
          }
          if (k_block < K_BLOCK_MAX - 1) {
            Utils::convert_A_kblock(tCrA_load_4b_packed, tCrA_mma, k_block + 1);
          }
        }

        warpgroup_commit_batch();

        if (chunk_id > 0) {
          warpgroup_wait<1>();

          int chunk_id_ = chunk_id - 1;
          warpgroup_fence_operand(intermediate_array[chunk_id_]);

          scale_intermediate(intermediate_array[chunk_id_], chunk_id_, chunk_id_ == 0);
        }

      }

      warpgroup_wait<0>();

      int chunk_id_ = NumChunksPerTileK - 1;
      warpgroup_fence_operand(intermediate_array[chunk_id_]);

      scale_intermediate(intermediate_array[chunk_id_], chunk_id_, false);

      --k_tile_count;
      if (k_tile_count > 0) {
        // Wait for K_BLOCK_MAX - 1 to be in flight to ensure that it is safe to overwrite the A registers for the first mma. 
        pipeline.consumer_wait(smem_pipe_read, barrier_token);

        Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, 0, smem_pipe_read.index());
        Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, 1, smem_pipe_read.index());
        
        Utils::convert_A_kblock(tCrA_load_4b_packed, tCrA_mma, 0);
      }
    }

    if (k_tile_count == 0) {
      return;
    }

    // Mainloop GMMAs
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 1; --k_tile_count) {

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();
      int const current_k_tile = k_tile_count_initial - k_tile_count;
      ++smem_pipe_read;

      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int chunk_id = 0; chunk_id < NumChunksPerTileK; ++chunk_id) {
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

        CUTLASS_PRAGMA_UNROLL
        for (int mma_id = 0; mma_id < NumMMAsPerChunk; ++mma_id) {
          int k_block = chunk_id * NumMMAsPerChunk + mma_id;

          warpgroup_arrive();
          // (V,M) x (V,N) => (V,M,N)
          cute::gemm(tiled_mma, tCrA_mma(_,_,k_block), tCrB(_,_,k_block,read_stage), intermediate_array[chunk_id]);
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;

          if (k_block == K_BLOCK_MAX - 1) {
            pipeline.consumer_release(smem_pipe_release);             // UNLOCK smem_pipe_release, done _computing_ on it
            ++smem_pipe_release;
          }

          if (k_block == 0) {
            barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            copy_scale_for_tile(read_stage);
          }
          if (mma_id == 0) {
            copy_activation_scale_for_pipeline(chunk_id, read_stage, current_k_tile);
          }

          if (k_block == K_BLOCK_MAX - 1) {
            // The last k_block

            pipeline.consumer_wait(smem_pipe_read, barrier_token);
            Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, 0, smem_pipe_read.index());
            Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, 1, smem_pipe_read.index());

            warpgroup_commit_batch();
            warpgroup_wait<0>();

            warpgroup_fence_operand(intermediate_array[chunk_id]);

            scale_intermediate(intermediate_array[chunk_id], chunk_id, false);

            Utils::convert_A_kblock(tCrA_load_4b_packed, tCrA_mma, 0);
          }
          else {
            if (k_block < K_BLOCK_MAX - 2) {
              Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, k_block + 2, read_stage);
            }
            Utils::convert_A_kblock(tCrA_load_4b_packed, tCrA_mma, k_block + 1);
          }
        }

        warpgroup_commit_batch();

        if (chunk_id > 0) {
          warpgroup_wait<1>();

          int chunk_id_ = chunk_id - 1;          
          warpgroup_fence_operand(intermediate_array[chunk_id_]);

          scale_intermediate(intermediate_array[chunk_id_], chunk_id_, false);
        }

      }
    }

    {
      //
      // Last k tile
      //
      Tensor intermediate = make_fragment_like(accum);

      int read_stage = smem_pipe_read.index();
      int const current_k_tile = k_tile_count_initial - k_tile_count;
      
      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {

        warpgroup_arrive();
        // (V,M) x (V,N) => (V,M,N)
        cute::gemm(tiled_mma, tCrA_mma(_,_,k_block), tCrB(_,_,k_block,read_stage), intermediate);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;

        if (k_block == 0) {
          copy_scale_for_tile(read_stage);
        }
        if (k_block % NumMMAsPerChunk == 0) {
          copy_activation_scale_for_pipeline(k_block / NumMMAsPerChunk, read_stage, current_k_tile);
        }

        if (k_block == K_BLOCK_MAX - 1) {
          // release prior barrier
          pipeline.consumer_release(smem_pipe_release);             // UNLOCK smem_pipe_release, done _computing_ on it
          ++smem_pipe_release;
        }

        if (k_block < K_BLOCK_MAX - 2) {
          Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, k_block + 2, read_stage);
        }
        if (k_block < K_BLOCK_MAX - 1) {
          Utils::convert_A_kblock(tCrA_load_4b_packed, tCrA_mma, k_block + 1);
        }

        if ((k_block + 1) % NumMMAsPerChunk == 0) {
          tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

          warpgroup_commit_batch();
          warpgroup_wait<0>();
          warpgroup_fence_operand(intermediate);

          scale_intermediate(intermediate, k_block / NumMMAsPerChunk, false);
        }
      }
    }
  }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = 1;
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(smem_pipe_release);                 // UNLOCK smem_pipe_release, done _computing_ on it
      ++smem_pipe_release;
    }
  }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  // Methods to perform different parts of TMA/Tensormap modifications
  //
  CUTLASS_DEVICE auto
  tensormaps_init(
      Params const& mainloop_params,
      TensorMapStorage& shared_tensormaps,
      int32_t sm_count,
      int32_t sm_idx) {
    cute::TmaDescriptor* gmem_tensormap = reinterpret_cast<cute::TmaDescriptor*>(mainloop_params.tensormaps);

    cute::TmaDescriptor* tma_desc_a = &gmem_tensormap[sm_idx];
    cute::TmaDescriptor* tma_desc_b = &gmem_tensormap[sm_idx + sm_count];
    Tensor pA_tensormap = make_tensor(mainloop_params.tma_load_a.get_tma_descriptor(), Int<1>{}, Int<1>{});
    Tensor sA_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_A), Int<1>{}, Int<1>{});
    Tensor pB_tensormap = make_tensor(mainloop_params.tma_load_b.get_tma_descriptor(), Int<1>{}, Int<1>{});
    Tensor sB_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_B), Int<1>{}, Int<1>{});

    if (cute::elect_one_sync()) {
      copy(recast<uint128_t>(pA_tensormap), recast<uint128_t>(sA_tensormap));
      copy(recast<uint128_t>(pB_tensormap), recast<uint128_t>(sB_tensormap));
      if constexpr (HasActivationScale) {
        Tensor pActivationScale_tensormap =
            make_tensor(mainloop_params.tma_load_activation_scale.get_tma_descriptor(), Int<1>{}, Int<1>{});
        Tensor sActivationScale_tensormap =
            make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_activation_scale), Int<1>{}, Int<1>{});
        copy(recast<uint128_t>(pActivationScale_tensormap), recast<uint128_t>(sActivationScale_tensormap));
      }
    }

    __syncwarp();

    if constexpr (HasActivationScale) {
      cute::TmaDescriptor* tma_desc_activation_scale = &gmem_tensormap[sm_idx + 2 * sm_count];
      return cute::make_tuple(tma_desc_a, tma_desc_b, tma_desc_activation_scale);
    }
    else {
      return cute::make_tuple(tma_desc_a, tma_desc_b);
    }
  }

  // Replace address for the global tensor (to be done by single thread)
  template <class... TMs>
  CUTLASS_DEVICE
  void
  tensormaps_replace_global_address(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      cute::tuple<TMs...> const& input_tensormaps,
      int32_t next_batch) {
    // Only A and B use TMA descriptors; scale/zero addresses are passed via load_inputs
    if constexpr (IsGroupedGemmKernel) {
      // Grouped GEMM: set base addresses once; groups are accessed via TMA L coordinate
      if (TensormapUpdateShapesStridesForAandScale) {
        cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_A,
                                                        mainloop_params.ptr_A[0]);
        cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                        mainloop_params.ptr_B[0]);
        if constexpr (HasActivationScale) {
          cute::tma_descriptor_replace_addr_in_shared_mem(
              shared_tensormaps.smem_tensormap_activation_scale,
              mainloop_params.ptr_ActivationScale[0]);
        }
      }
    } else {
      // Ptr-Array: update B address per batch
      cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                      mainloop_params.ptr_B[next_batch]);
      // Ptr-Array: update A address per batch
      if (TensormapUpdateShapesStridesForAandScale) {
        cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_A,
                                                        mainloop_params.ptr_A[next_batch]);
        if constexpr (HasActivationScale) {
          cute::tma_descriptor_replace_addr_in_shared_mem(
              shared_tensormaps.smem_tensormap_activation_scale,
              mainloop_params.ptr_ActivationScale[next_batch]);
        }
      }
      else {
        cute::tma_descriptor_replace_addr_in_global_mem(get<0>(input_tensormaps),
                                                        mainloop_params.ptr_A[next_batch]);
        if constexpr (HasActivationScale) {
          cute::tma_descriptor_replace_addr_in_global_mem(
              get<2>(input_tensormaps),
              mainloop_params.ptr_ActivationScale[next_batch]);
        }
      }
    }

  }

  // Replace dim and strides for the global tensor - used only for Grouped GEMM (to be done by single thread)
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE
  void
  tensormaps_replace_global_tensor_properties(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      int32_t next_group,
      ProblemShape_MNKL problem_shape_mnkl) {
    if (!TensormapUpdateShapesStridesForAandScale) return;

    const uint32_t M = get<0>(problem_shape_mnkl);
    const uint32_t N = get<1>(problem_shape_mnkl);
    const uint32_t K = get<2>(problem_shape_mnkl);
    
    constexpr int MaxTensorRank = 5;
    cute::array<uint32_t, MaxTensorRank> prob_shape_A  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_A = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_B  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_B = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_activation_scale  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_activation_scale = {0,0,0,0,0};

    // B: build 3D layout (N, K, num_groups) for coordinate-based group selection
    SwappedElementB const* ptr_B = nullptr;
    if constexpr (!cute::is_layout<InternalSwappedStrideB>::value) {
      auto dB_group = mainloop_params.ptr_dB[next_group];
      auto stride_n = cute::get<0>(dB_group);
      auto stride_k = cute::get<1>(dB_group);
      int64_t term_n = static_cast<int64_t>(N) * static_cast<int64_t>(stride_n);
      int64_t term_k = static_cast<int64_t>(K) * static_cast<int64_t>(stride_k);
      int64_t stride_l = term_n > term_k ? term_n : term_k;
      auto full_layout = make_layout(
          make_shape(N, K, static_cast<uint32_t>(mainloop_params.num_groups)),
          cute::make_stride(stride_n, stride_k, stride_l));
      Tensor tensor_b = make_tensor(ptr_B, full_layout);
      cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_b, tensor_b,
                                              prob_shape_B, prob_stride_B);
    } else {
      Tensor tensor_b = make_tensor(ptr_B, detail::get_gmem_layout(make_shape(N,K,Int<1>{}), mainloop_params.ptr_dB[next_group]));
      cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_b, tensor_b, 
                                              prob_shape_B, prob_stride_B);
    }

    for (uint64_t& stride : prob_stride_B) {
      stride = (stride * sizeof_bits_v<SwappedElementB>) / 8;
    }

    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                            prob_shape_B,
                                                            prob_stride_B);

    // A: build 3D layout (M, K, num_groups) for coordinate-based group selection
    SwappedElementA const* ptr_A = nullptr;
    if constexpr (!cute::is_layout<InternalSwappedStrideA>::value) {
      auto dA_group = mainloop_params.ptr_dA[next_group];
      auto stride_m = cute::get<0>(dA_group);
      auto stride_k = cute::get<1>(dA_group);
      int64_t term_m = static_cast<int64_t>(M) * static_cast<int64_t>(stride_m);
      int64_t term_k = static_cast<int64_t>(K) * static_cast<int64_t>(stride_k);
      int64_t stride_l = term_m > term_k ? term_m : term_k;
      auto full_layout = make_layout(
          make_shape(M, K, static_cast<uint32_t>(mainloop_params.num_groups)),
          cute::make_stride(stride_m, stride_k, stride_l));
      Tensor tensor_a = make_tensor(ptr_A, full_layout);
      cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_a, tensor_a,
                                              prob_shape_A, prob_stride_A);
    } else {
      Tensor tensor_a = make_tensor(ptr_A, detail::get_gmem_layout(make_shape(M,K,Int<1>{}), mainloop_params.ptr_dA[next_group]));
      cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_a, tensor_a,
                                              prob_shape_A, prob_stride_A);
    }

    for (uint64_t& stride : prob_stride_A) {
      stride = (stride * sizeof_bits_v<SwappedElementA>) / 8;
    }
    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_A,
                                                            prob_shape_A,
                                                            prob_stride_A);

    if constexpr (HasActivationScale) {
      NonVoidElementActivationScale const* ptr_activation_scale = nullptr;
      const uint32_t scale_groups = K / ScalingGroupSize;
      auto full_layout = make_layout(
          make_shape(N, scale_groups, static_cast<uint32_t>(mainloop_params.num_groups)),
          cute::make_stride(
              static_cast<int64_t>(scale_groups),
              cute::Int<1>{},
              static_cast<int64_t>(N) * static_cast<int64_t>(scale_groups)));
      Tensor tensor_activation_scale = make_tensor(ptr_activation_scale, full_layout);
      cute::detail::fill_tma_gmem_shape_stride(
          mainloop_params.tma_load_activation_scale,
          tensor_activation_scale,
          prob_shape_activation_scale,
          prob_stride_activation_scale);

      for (uint64_t& stride : prob_stride_activation_scale) {
        stride = (stride * sizeof_bits_v<NonVoidElementActivationScale>) / 8;
      }
      cute::tma_descriptor_replace_dims_strides_in_shared_mem(
          shared_tensormaps.smem_tensormap_activation_scale,
          prob_shape_activation_scale,
          prob_stride_activation_scale);
    }
  }

  template <class... TMs, class ProblemShape_MNKL>
  CUTLASS_DEVICE
  void
  tensormaps_perform_update(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      cute::tuple<TMs...> const& input_tensormaps,
      ProblemShape_MNKL problem_shape_mnkl,
      int32_t next_batch) {
    if (cute::elect_one_sync()) {
      // Replacing global_address for the next batch
      tensormaps_replace_global_address(shared_tensormaps, mainloop_params, input_tensormaps, next_batch);

      if constexpr (IsGroupedGemmKernel) {
        // Replacing global dims and strides for the next batch
        tensormaps_replace_global_tensor_properties(shared_tensormaps,
          mainloop_params, next_batch, problem_shape_mnkl);
      }
    }
  }

  template <class... TMs>
  CUTLASS_DEVICE
  void
  tensormaps_cp_fence_release (
      TensorMapStorage& shared_tensormaps,
      cute::tuple<TMs...> const& input_tensormaps) {

    // [None][fix] Fix W4A8 MoE kernel issue
    // https://github.com/NVIDIA/TensorRT-LLM/pull/7072
    if (cute::elect_one_sync())
    {
        cute::tma_desc_commit_group();
        cute::tma_desc_wait_group();
    }

    // Entire warp must do this (i.e. it's aligned)
    if (TensormapUpdateShapesStridesForAandScale) {
      TensormapUpdateShapesStridesForAandScale = false;

      tma_descriptor_cp_fence_release(get<0>(input_tensormaps), shared_tensormaps.smem_tensormap_A);
      tma_descriptor_cp_fence_release(get<1>(input_tensormaps), shared_tensormaps.smem_tensormap_B);
      if constexpr (HasActivationScale) {
        tma_descriptor_cp_fence_release(
            get<2>(input_tensormaps),
            shared_tensormaps.smem_tensormap_activation_scale);
      }
    }
    else if constexpr (!IsGroupedGemmKernel) {
      // Ptr-Array: B address updated in shared mem, A address updated in gmem
      tma_descriptor_cp_fence_release(get<1>(input_tensormaps), shared_tensormaps.smem_tensormap_B);
      if constexpr (HasActivationScale) {
        tma_descriptor_cp_fence_release(
            get<2>(input_tensormaps),
            shared_tensormaps.smem_tensormap_activation_scale);
      }
      tma_descriptor_fence_release();
    }
  }

  // The entire warp must call this function collectively (that is, the instructions are aligned)
  template <class... TMs>
  CUTLASS_DEVICE
  void
  tensormaps_fence_acquire(cute::tuple<TMs...> const& input_tensormaps) {
    cute::tma_descriptor_fence_acquire(get<0>(input_tensormaps));
    cute::tma_descriptor_fence_acquire(get<1>(input_tensormaps));
    if constexpr (HasActivationScale) {
      cute::tma_descriptor_fence_acquire(get<2>(input_tensormaps));
    }
  }

  template <class InputTensors, class ProblemShape_MNKL>
  CUTLASS_DEVICE
  InputTensors
  tensors_perform_update(
      InputTensors const& input_tensors,
      [[maybe_unused]] Params const& mainloop_params,
      [[maybe_unused]] ProblemShape_MNKL problem_shape_mnkl,
      [[maybe_unused]] int32_t next_batch) {
    if constexpr (IsGroupedGemmKernel) {
      current_group_idx_ = next_batch;
    }
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return input_tensors;
    }
    else if constexpr (ModeHasScales) {
      auto new_scale_ptr = mainloop_params.ptr_S[next_batch];
      int64_t new_stride_k;
      if constexpr (IsGroupedGemmKernel) {
        new_stride_k = get<1>(mainloop_params.dS[next_batch]);
      } else {
        new_stride_k = get<1>(mainloop_params.dS[0]);
      }
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        if constexpr (HasActivationScale) {
          return cute::make_tuple(get<0>(input_tensors), get<1>(input_tensors),
                                  new_scale_ptr, new_stride_k,
                                  get<4>(input_tensors));
        }
        else {
          return cute::make_tuple(get<0>(input_tensors), get<1>(input_tensors),
                                  new_scale_ptr, new_stride_k);
        }
      }
      else {
        auto new_zero_ptr = mainloop_params.ptr_Z[next_batch];
        return cute::make_tuple(get<0>(input_tensors), get<1>(input_tensors),
                                new_scale_ptr, new_stride_k, new_zero_ptr);
      }
    }
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in tensors_perform_update.");
    }
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
