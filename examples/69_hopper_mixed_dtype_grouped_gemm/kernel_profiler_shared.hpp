#pragma once

// Dedicated CMake targets enable the kernel config profiler across 8 TUs, e.g.
// 69_hopper_int4_fp8_grouped_gemm_k512.
// Keep this local toggle for ad-hoc builds only.
// #define PROFILE

/***************************************************************************************************
 * Shared types, extern declarations, and template implementations for the kernel profiler.
 * Included by the main .cu and by all profiler part files to enable parallel compilation.
 **************************************************************************************************/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <numeric>
#include <typeinfo>
#include <float.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#if defined(CUTLASS_MIXED_GEMM_SINGLE_WG_SMEM_EPILOGUE)
#include "cutlass/epilogue/collective/default_epilogue_array_per_token_scale.hpp"
#endif
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_ptr_array_scale_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#if defined(CUTLASS_MIXED_GEMM_SINGLE_WG_CTAS_PER_SM)
#include "cutlass/gemm/kernel/sm90_gemm_array_tma_single_warpgroup_persistent.hpp"
#endif
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/mixed_dtype_utils.hpp"

#include "helper.h"
#include "grouped_mixed_dtype_utils.hpp"
// Note: host_validation.hpp and BF16_MXFP4_Test.h are included only in
// 69_hopper_int4_fp8_grouped_gemm.cu to avoid multiple-definition linker
// errors when compiling across parallel TUs.

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;

//--------------------------------------------------------------------------------------------
// Kernel type configuration — MXFP4 x BF16
//--------------------------------------------------------------------------------------------
// using MmaType   = cutlass::bfloat16_t;     // activations
// using QuantType = cutlass::float_e2m1_t;   // weights
// #define GROUP_SIZE 32
// using ElementScale = cutlass::float_ue8m0_t;
// inline constexpr int TileShapeM = 128;
// inline constexpr int TileShapeN = 32;
// inline constexpr int TileShapeK = 8192 / TileShapeN;

//--------------------------------------------------------------------------------------------

#ifndef CUTLASS_MIXED_GEMM_TILE_SHAPE_K
#define CUTLASS_MIXED_GEMM_TILE_SHAPE_K (8192 / TileShapeN)
#endif

#ifndef CUTLASS_MIXED_GEMM_TILE_SHAPE_N
#define CUTLASS_MIXED_GEMM_TILE_SHAPE_N 16
#endif

#ifndef CUTLASS_MIXED_GEMM_TILE_SHAPE_M
#define CUTLASS_MIXED_GEMM_TILE_SHAPE_M 128
#endif

#if defined(CUTLASS_MIXED_GEMM_MXFP4_BF16)
using MmaType = cutlass::bfloat16_t;     // activations
using QuantType = cutlass::float_e2m1_t; // weights
#define GROUP_SIZE 32
using ElementScale = cutlass::float_ue8m0_t;
inline constexpr int TileShapeM = CUTLASS_MIXED_GEMM_TILE_SHAPE_M;
inline constexpr int TileShapeN = CUTLASS_MIXED_GEMM_TILE_SHAPE_N;
inline constexpr int TileShapeK = CUTLASS_MIXED_GEMM_TILE_SHAPE_K;
#elif defined(CUTLASS_MIXED_GEMM_MXFP4_FP8)
// MXFP4 x FP8 path. This keeps the same MXFP4 weight semantics
// as the MXFP4 x BF16 path: e2m1 payload, UE8M0 block scale, group size 32.
// The offline weight layout still follows the W4A8 INT4xFP8 path.
using MmaType = cutlass::float_e4m3_t;      // activations
using QuantType = cutlass::float_e2m1_t;    // weights
#define GROUP_SIZE 32
using ElementScale = cutlass::float_ue8m0_t;
inline constexpr int TileShapeM = CUTLASS_MIXED_GEMM_TILE_SHAPE_M;
inline constexpr int TileShapeN = CUTLASS_MIXED_GEMM_TILE_SHAPE_N;
inline constexpr int TileShapeK = CUTLASS_MIXED_GEMM_TILE_SHAPE_K;
#elif defined(CUTLASS_MIXED_GEMM_MXFP4_MXFP8)
// MXFP4 x MXFP8 keeps the activation payload as plain FP8 and reads the
// activation UE8M0 scale from its original M x K/32, K-contiguous layout.
using MmaType = cutlass::float_e4m3_t;      // activation payload
using QuantType = cutlass::float_e2m1_t;    // weights
#define GROUP_SIZE 32
using ElementScale = cutlass::float_ue8m0_t;
inline constexpr int TileShapeM = CUTLASS_MIXED_GEMM_TILE_SHAPE_M;
inline constexpr int TileShapeN = CUTLASS_MIXED_GEMM_TILE_SHAPE_N;
inline constexpr int TileShapeK = CUTLASS_MIXED_GEMM_TILE_SHAPE_K;
#else
// INT4 x FP8
using MmaType = cutlass::float_e4m3_t;      // activations
using QuantType = cutlass::int4b_t;         // weights
#define GROUP_SIZE 128
using ElementScale = cutlass::bfloat16_t;
inline constexpr int TileShapeM = CUTLASS_MIXED_GEMM_TILE_SHAPE_M;
inline constexpr int TileShapeN = CUTLASS_MIXED_GEMM_TILE_SHAPE_N;
inline constexpr int TileShapeK = CUTLASS_MIXED_GEMM_TILE_SHAPE_K;
#endif

//--------------------------------------------------------------------------------------------

// CMX kernels consume folded scalar weight scales so scale storage is
// independent from the profiled TileK.
using MainloopWeightScale = ElementScale;
using ElementWeightScaleStorage = ElementScale;
inline constexpr bool ScaleAppliesToActivation =
#if defined(CUTLASS_MIXED_GEMM_MXFP4_MXFP8)
    true;
#else
    false;
#endif
using ElementActivationScale = cutlass::float_ue8m0_t;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Options struct (defined here so all part files share the same type)
/////////////////////////////////////////////////////////////////////////////////////////////////

struct Options : GroupedMixedDtypeOptions<QuantType> {
  using Base = GroupedMixedDtypeOptions<QuantType>;

  int swizzle = 2;
  bool enable_print = false;
  bool enable_print_weight = false;
  bool debug_input_act = false;
  bool debug_input_weight = false;
  bool debug_input_scale = false;
  bool split_timing = false;
  int64_t total_routed_tokens = -1;

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("explore", explore);
    bool const compare_was_set = cmd.check_cmd_line_flag("compare");
    cmd.get_cmd_line_argument("compare", compare);
    if (explore && !compare_was_set) {
      compare = false;
    }
    cmd.get_cmd_line_argument("enable_print", enable_print);
    cmd.get_cmd_line_argument("enable_print_weight", enable_print_weight);
    cmd.get_cmd_line_argument("debug_input_act", debug_input_act);
    cmd.get_cmd_line_argument("debug_input_weight", debug_input_weight);
    cmd.get_cmd_line_argument("debug_input_scale", debug_input_scale);
    cmd.get_cmd_line_argument("split_timing", split_timing);
    cmd.get_cmd_line_argument("swizzle", swizzle);
    cmd.get_cmd_line_argument("total_routed_tokens", total_routed_tokens);
    this->Base::parse(argc, args);
    mode = 1;
  }

  std::ostream & print_usage(std::ostream &out) const {
    out << "69_hopper_int4_fp8_grouped_gemm\n\n"
      << "  Hopper Mixed Dtype Grouped GEMM using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM for all groups\n"
      << "  --n=<int>                   Sets the N extent of the GEMM for all groups\n"
      << "  --k=<int>                   Sets the K extent of the GEMM for all groups\n"
      << "  --groups=<int>              Sets the number of individual GEMM problems for Grouped GEMM\n"
      << "  --c=<int>                   The size of each chunk for the scales and zeros.\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform\n\n"
      << "  --warmup=<int>              Number of warmup iterations to perform\n\n"
      << "  --compare=<bool>            Print per-element mismatch details. Default false for --explore=true.\n"
      << "  --swizzle=<int>             Tile scheduler swizzle size (1, 2, 4, or 8). Default: 2\n"
      << "  --split_timing=<bool>       Print diagnostic builder/GEMM event timing split.\n"
      << "  --total_routed_tokens=<int> Override total token count for scheduler capacity sizing.\n"
      << "  --benchmark=<str>           Executes a benchmark problem size.\n";

    out << "\n\nExamples:\n\n"
      << "$ " << "69_hopper_int4_fp8_grouped_gemm"
      << " --m=1024 --n=512 --k=1024 --groups=10 --alpha=1 --beta=0 \n\n";

    return out;
  }
};

template <
    int TileShapeM,
    int TileShapeN,
    int ClusterShapeM,
    int ClusterShapeN>
void prepare_precomputed_work_tile_map(Options const& options);

template <
    int TileShapeM,
    int TileShapeN,
    int ClusterShapeM,
    int ClusterShapeN,
    class MainloopParams>
void build_precomputed_work_tile_map(
    Options const& options,
    MainloopParams const& mainloop_params,
    cudaStream_t stream);

namespace precomputed_scheduler {

uint64_t const* work_tiles_data();
#if defined(CUTLASS_MIXED_GEMM_SINGLE_WG_CHUNK_MAJOR_WORK_MAP)
uint32_t work_tiles_per_worker();
#endif
cute::TmaDescriptor const* prebuilt_tma_desc_A_data();
cute::TmaDescriptor const* prebuilt_tma_desc_B_data();
cute::TmaDescriptor const* prebuilt_tma_desc_activation_scale_data();

}  // namespace precomputed_scheduler

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM type chain
/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

// A matrix
using         ElementA    = MmaType;
using         LayoutA     = cutlass::layout::RowMajor;
inline constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

// B matrix
using         ElementB    = QuantType;
using         LayoutB     = cutlass::layout::ColumnMajor;
inline constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using MainloopElementA = cute::conditional_t<
    ScaleAppliesToActivation,
    cute::tuple<ElementA, ElementActivationScale>,
    ElementA>;
using MainloopElementB = cute::tuple<ElementB, MainloopWeightScale>;

using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

using StrideA = cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
using StrideB = cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;

using ElementZero = cutlass::float_e4m3_t;
using LayoutScale = cutlass::layout::RowMajor;

// C/D matrix
using         ElementC    = cutlass::bfloat16_t;
using         LayoutC     = cutlass::layout::RowMajor;
inline constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
using         ElementD    = ElementC;
using         LayoutD     = LayoutC;
inline constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

// Core
using ElementAccumulator = float;
using ElementEpilogueTokenScale = ElementAccumulator;
using ArchTag            = cutlass::arch::Sm90;
using OperatorClass      = cutlass::arch::OpClassTensorOp;
using StageCountType     = cutlass::gemm::collective::StageCountAuto;

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::
    PersistentTileSchedulerSm90GroupParams<Shape<int,int,int>>::RasterOrderOptions;

// Default configuration (used for the normal non-profiler run path, and to derive stride types)
#ifndef CUTLASS_MIXED_GEMM_DEFAULT_CLUSTER_M
#define CUTLASS_MIXED_GEMM_DEFAULT_CLUSTER_M 2
#endif
#ifndef CUTLASS_MIXED_GEMM_DEFAULT_CLUSTER_N
#define CUTLASS_MIXED_GEMM_DEFAULT_CLUSTER_N 1
#endif
#ifndef CUTLASS_MIXED_GEMM_DEFAULT_CLUSTER_K
#define CUTLASS_MIXED_GEMM_DEFAULT_CLUSTER_K 1
#endif

using DefaultTileShape    = Shape<cute::Int<TileShapeM>, cute::Int<TileShapeN>, cute::Int<TileShapeK>>;
using DefaultClusterShape = Shape<
    cute::Int<CUTLASS_MIXED_GEMM_DEFAULT_CLUSTER_M>,
    cute::Int<CUTLASS_MIXED_GEMM_DEFAULT_CLUSTER_N>,
    cute::Int<CUTLASS_MIXED_GEMM_DEFAULT_CLUSTER_K>>;
#if defined(CUTLASS_MIXED_GEMM_DEFAULT_SCHEDULE_PINGPONG)
using DefaultKernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
using DefaultEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
#else
using DefaultKernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using DefaultEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
#endif

#if defined(CUTLASS_MIXED_GEMM_EPILOGUE_TOKEN_SCALE)
using DefaultFusionOperation = cutlass::epilogue::fusion::PtrArrayPerTokenScaledAcc<
    ElementD,
    ElementAccumulator,
    ElementEpilogueTokenScale>;
#else
using DefaultFusionOperation = cutlass::epilogue::fusion::LinearCombination<
    ElementD,
    ElementAccumulator,
    ElementC,
    ElementAccumulator>;
#endif

#if defined(CUTLASS_MIXED_GEMM_SINGLE_WG_SMEM_EPILOGUE)
using DefaultEpilogueLayoutC = typename cutlass::layout::LayoutTranspose<LayoutC>::type;
using DefaultEpilogueLayoutD = typename cutlass::layout::LayoutTranspose<LayoutD>::type;
using DefaultSmallKEpilogue = cutlass::epilogue::collective::SmemEpilogueArrayPerTokenScale<
    DefaultTileShape,
    ElementC,
    cutlass::detail::TagToStrideC_t<DefaultEpilogueLayoutC *>,
    ElementD,
    cutlass::detail::TagToStrideC_t<DefaultEpilogueLayoutD *>,
    ElementAccumulator,
    ElementEpilogueTokenScale>;
using DefaultCollectiveEpilogue =
    cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<DefaultSmallKEpilogue>;
#else
using DefaultCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    DefaultTileShape, DefaultClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type *, AlignmentC,
    ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type *, AlignmentD,
    DefaultEpilogueSchedule,
    DefaultFusionOperation
>::CollectiveOp;
#endif

#if defined(CUTLASS_MIXED_GEMM_MANUAL_STAGE_COUNT)
using DefaultMainloopStageCount = cutlass::gemm::collective::StageCount<
    CUTLASS_MIXED_GEMM_MANUAL_STAGE_COUNT>;
#else
using DefaultMainloopStageCount = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename DefaultCollectiveEpilogue::SharedStorage))>;
#endif

using DefaultCollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MainloopElementB, LayoutB_Transpose *, AlignmentB,
    MainloopElementA, LayoutA_Transpose *, AlignmentA,
    ElementAccumulator,
    DefaultTileShape, DefaultClusterShape,
    DefaultMainloopStageCount,
    DefaultKernelSchedule
>::CollectiveOp;

#if defined(CUTLASS_MIXED_GEMM_SINGLE_WG_CTAS_PER_SM)
#if defined(CUTLASS_MIXED_GEMM_SINGLE_WG_ROLLING_REFILL)
using DefaultGemmKernel = cutlass::gemm::kernel::SingleWarpgroupPersistentGemm<
    ProblemShape,
    DefaultCollectiveMainloop,
    DefaultCollectiveEpilogue,
    CUTLASS_MIXED_GEMM_SINGLE_WG_CTAS_PER_SM,
    CUTLASS_MIXED_GEMM_SINGLE_WG_PREFETCH_NEXT_TILE,
    cutlass::gemm::kernel::SingleWarpgroupPipelineMode::RollingRefill>;
#else
using DefaultGemmKernel = cutlass::gemm::kernel::SingleWarpgroupPersistentGemm<
    ProblemShape,
    DefaultCollectiveMainloop,
    DefaultCollectiveEpilogue,
    CUTLASS_MIXED_GEMM_SINGLE_WG_CTAS_PER_SM,
    CUTLASS_MIXED_GEMM_SINGLE_WG_PREFETCH_NEXT_TILE>;
#endif
#else
using DefaultGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    DefaultCollectiveMainloop,
    DefaultCollectiveEpilogue>;
#endif

using DefaultGemm    = cutlass::gemm::device::GemmUniversalAdapter<DefaultGemmKernel>;
using GemmScaleOnly  = DefaultGemm;  // alias for backward compatibility with main .cu

// Stride types (stable for any config using the same element types and layouts)
using StrideC     = typename DefaultGemmKernel::InternalStrideC;
using StrideD     = typename DefaultGemmKernel::InternalStrideD;
using StrideC_ref = cutlass::detail::TagToStrideC_t<LayoutC>;
using StrideD_ref = cutlass::detail::TagToStrideC_t<LayoutD>;
using StrideS     = typename DefaultCollectiveMainloop::StrideScale;
using StrideActivationScale = typename DefaultCollectiveMainloop::StrideActivationScale;
using StrideS_ref = cutlass::detail::TagToStrideB_t<LayoutScale>;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// gemm_constructor: builds a complete Gemm type from (Schedule, Cluster, Tile) triple
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    typename KernelSchedule,
    typename ClusterShape,
    typename TileShape
>
class gemm_constructor {
public:
    using EpilogueSchedule = typename std::conditional<
        std::is_same<KernelSchedule, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>::value,
        cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong,
        cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative
    >::type;

#if defined(CUTLASS_MIXED_GEMM_EPILOGUE_TOKEN_SCALE)
    using FusionOperation = cutlass::epilogue::fusion::PtrArrayPerTokenScaledAcc<
        ElementD,
        ElementAccumulator,
        ElementEpilogueTokenScale>;
#else
    using FusionOperation = cutlass::epilogue::fusion::LinearCombination<
        ElementD,
        ElementAccumulator,
        ElementC,
        ElementAccumulator>;
#endif

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type *, AlignmentC,
        ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type *, AlignmentD,
        EpilogueSchedule,
        FusionOperation
    >::CollectiveOp;

    using CollectiveMainloopScaleOnly = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        MainloopElementB, LayoutB_Transpose *, AlignmentB,
        MainloopElementA, LayoutA_Transpose *, AlignmentA,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule
    >::CollectiveOp;

    using GemmKernelScaleOnly = cutlass::gemm::kernel::GemmUniversal<
        ProblemShape,
        CollectiveMainloopScaleOnly,
        CollectiveEpilogue
    >;

    using GemmScaleOnly = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleOnly>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Extern declarations — defined in 69_hopper_int4_fp8_grouped_gemm.cu
/////////////////////////////////////////////////////////////////////////////////////////////////

extern std::vector<StrideA>     stride_A_host;
extern std::vector<StrideB>     stride_B_host;
extern std::vector<StrideC>     stride_C_host;
extern std::vector<StrideD>     stride_D_host;
extern std::vector<StrideC_ref> stride_C_host_ref;
extern std::vector<StrideD_ref> stride_D_host_ref;
extern std::vector<StrideS>     stride_weight_scale_host;
extern std::vector<StrideActivationScale> stride_activation_scale_host;

extern std::vector<ElementAccumulator> alpha_host;
extern std::vector<ElementAccumulator> beta_host;

extern uint64_t seed;
extern bool setup;

extern cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

extern cutlass::DeviceAllocation<MmaType>                                               block_A;
extern cutlass::DeviceAllocation<QuantType>                                             block_B;
extern cutlass::DeviceAllocation<QuantType>                                             block_B_interleaved;
extern cutlass::DeviceAllocation<ElementScale>                                          block_weight_scale;
extern cutlass::DeviceAllocation<ElementWeightScaleStorage>                             block_weight_scale_folded;
extern cutlass::DeviceAllocation<ElementActivationScale>                                block_activation_scale;
extern cutlass::DeviceAllocation<ElementEpilogueTokenScale>                             block_epilogue_token_scale;
extern cutlass::DeviceAllocation<ElementZero>                                           block_zero;
extern cutlass::DeviceAllocation<ElementC>                                              block_C;
extern cutlass::DeviceAllocation<typename DefaultGemm::EpilogueOutputOp::ElementOutput> block_D;
extern cutlass::DeviceAllocation<typename DefaultGemm::EpilogueOutputOp::ElementOutput> block_ref_D;

extern cutlass::DeviceAllocation<const MmaType *>                    ptr_A;
extern cutlass::DeviceAllocation<const QuantType *>                  ptr_B;
extern cutlass::DeviceAllocation<const ElementActivationScale *>     ptr_activation_scale;
extern cutlass::DeviceAllocation<const ElementEpilogueTokenScale *>  ptr_epilogue_token_scale;
extern cutlass::DeviceAllocation<const MainloopWeightScale *>        ptr_weight_scale_folded;
extern cutlass::DeviceAllocation<const ElementZero *>                ptr_zero;
extern cutlass::DeviceAllocation<const ElementC *>                   ptr_C;
extern cutlass::DeviceAllocation<typename DefaultGemm::EpilogueOutputOp::ElementOutput *> ptr_D;

extern cutlass::DeviceAllocation<StrideA>     stride_A;
extern cutlass::DeviceAllocation<StrideB>     stride_B;
extern cutlass::DeviceAllocation<StrideC>     stride_C;
extern cutlass::DeviceAllocation<StrideD>     stride_D;
extern cutlass::DeviceAllocation<StrideC_ref> stride_C_ref;
extern cutlass::DeviceAllocation<StrideD_ref> stride_D_ref;
extern cutlass::DeviceAllocation<StrideS>     stride_weight_scale;
extern cutlass::DeviceAllocation<StrideActivationScale> stride_activation_scale;

extern cutlass::DeviceAllocation<ElementAccumulator*> alpha_device;
extern cutlass::DeviceAllocation<ElementAccumulator*> beta_device;
extern cutlass::DeviceAllocation<ElementAccumulator>  block_alpha;
extern cutlass::DeviceAllocation<ElementAccumulator>  block_beta;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Forward declarations of non-template functions (defined in main .cu)
/////////////////////////////////////////////////////////////////////////////////////////////////

void allocate(Options const& options);
void initialize(Options& options);
bool verify(Options const& options);

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Template implementations (must live in header for cross-TU instantiation)
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
typename Gemm::Arguments args_from_options(Options const& options)
{
  using Args = typename Gemm::Arguments;
  auto dB = stride_B.get();
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
#if defined(CUTLASS_MIXED_GEMM_SINGLE_WG_CTAS_PER_SM)
  hw_info.sm_count *= CUTLASS_MIXED_GEMM_SINGLE_WG_CTAS_PER_SM;
#endif

  Args arguments;
  decltype(arguments.epilogue.thread) fusion_args;

#if defined(CUTLASS_MIXED_GEMM_EPILOGUE_TOKEN_SCALE)
  fusion_args.token_scale_default = ElementAccumulator(1);
  fusion_args.token_scale_ptr_array = ptr_epilogue_token_scale.get();
#else
  if (options.alpha != FLT_MAX && options.beta != FLT_MAX) {
    fusion_args.alpha = options.alpha;
    fusion_args.beta  = options.beta;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr  = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array  = nullptr;
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta  = {cute::_0{}, cute::_0{}, 0};
  }
  else {
    fusion_args.alpha = 0;
    fusion_args.beta  = 0;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr  = nullptr;
    fusion_args.alpha_ptr_array = alpha_device.get();
    fusion_args.beta_ptr_array  = beta_device.get();
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
    fusion_args.dBeta  = {cute::_0{}, cute::_0{}, 1};
  }
#endif

  decltype(arguments.mainloop) mainloop_args{
    ptr_B.get(), dB, ptr_A.get(), stride_A.get(), ptr_weight_scale_folded.get(), stride_weight_scale.get(), GROUP_SIZE
  };
  mainloop_args.ptr_B_prebuilt_tma_descs =
      precomputed_scheduler::prebuilt_tma_desc_B_data();
  mainloop_args.ptr_A_prebuilt_tma_desc =
      precomputed_scheduler::prebuilt_tma_desc_A_data();
  if constexpr (ScaleAppliesToActivation) {
    mainloop_args.ptr_ActivationScale = ptr_activation_scale.get();
    mainloop_args.dActivationScale = stride_activation_scale.get();
    mainloop_args.ptr_ActivationScale_prebuilt_tma_descs =
        precomputed_scheduler::prebuilt_tma_desc_activation_scale_data();
  }

#if defined(CUTLASS_MIXED_GEMM_SINGLE_WG_SMEM_EPILOGUE)
  ElementAccumulator compact_epilogue_beta = ElementAccumulator(0);
  if (options.beta != FLT_MAX) {
    compact_epilogue_beta = options.beta;
  }
  else {
    for (ElementAccumulator beta : beta_host) {
      if (beta != ElementAccumulator(0)) {
        compact_epilogue_beta = beta;
        break;
      }
    }
  }

  int64_t compact_output_channel_extent = 0;
  if (!options.problem_sizes_host.empty()) {
    compact_output_channel_extent =
        int64_t(cute::get<1>(options.problem_sizes_host.front()));
    for (auto const& problem : options.problem_sizes_host) {
      if (int64_t(cute::get<1>(problem)) != compact_output_channel_extent) {
        compact_output_channel_extent = 0;
        break;
      }
    }
  }

  int64_t compact_output_row_stride = 0;
  if (!stride_D_host.empty()) {
    compact_output_row_stride = int64_t(cute::get<1>(stride_D_host.front()));
    for (auto const& stride : stride_D_host) {
      if (int64_t(cute::get<1>(stride)) != compact_output_row_stride) {
        compact_output_row_stride = 0;
        break;
      }
    }
  }
#endif

  arguments = Args {
    cutlass::gemm::GemmUniversalMode::kGrouped,
    {options.groups, problem_sizes.get(), nullptr},
    mainloop_args,
#if defined(CUTLASS_MIXED_GEMM_SINGLE_WG_SMEM_EPILOGUE)
    {
      fusion_args,
      nullptr,
      stride_C.get(),
      ptr_D.get(),
      stride_D.get(),
      block_D.get(),
      compact_output_channel_extent,
      compact_output_row_stride,
      compact_epilogue_beta
    },
#else
    {fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
#endif
    hw_info
  };

  arguments.scheduler.max_swizzle_size = options.swizzle;
  arguments.scheduler.raster_order     = RasterOrderOptions::AlongM;
#if defined(CUTLASS_MIXED_GEMM_PRECOMPUTED_GROUP_OFFSETS)
  arguments.scheduler.precomputed_work_tiles =
      precomputed_scheduler::work_tiles_data();
#if defined(CUTLASS_MIXED_GEMM_SINGLE_WG_CHUNK_MAJOR_WORK_MAP)
  arguments.scheduler.precomputed_work_tiles_per_worker =
      precomputed_scheduler::work_tiles_per_worker();
#endif
#endif

  return arguments;
}

template <class Gemm, class TileShape = DefaultTileShape, class ClusterShape = DefaultClusterShape>
void profile_grouped_mixed_dtype(
    Gemm& gemm,
    const Options& options,
    MixedDtypeResult& result,
    const std::vector<ElementAccumulator>& alpha_host,
    const std::vector<ElementAccumulator>& beta_host) {

  if (options.iterations <= 0) return;

  cudaEvent_t timing_start, timing_done;
  cudaEventCreate(&timing_start);
  cudaEventCreate(&timing_done);

  float split_builder_ms = 0.0f;
  cudaStream_t stream = nullptr;
  constexpr int CurrentTileShapeM = cute::size<0>(TileShape{});
  constexpr int CurrentTileShapeN = cute::size<1>(TileShape{});
  constexpr int CurrentClusterShapeM = cute::size<0>(ClusterShape{});
  constexpr int CurrentClusterShapeN = cute::size<1>(ClusterShape{});
  auto build_work_map = [&]() {
    build_precomputed_work_tile_map<
        CurrentTileShapeM,
        CurrentTileShapeN,
        CurrentClusterShapeM,
        CurrentClusterShapeN>(
        options,
        gemm.params().mainloop,
        stream);
  };

  auto run_iteration = [&]() {
    build_work_map();
    return gemm.run(stream);
  };

  for (int iter = 0; iter < options.warmup; ++iter) {
    result.status = run_iteration();
    if (result.status != cutlass::Status::kSuccess) {
      result.passed = false;
      cudaEventDestroy(timing_start);
      cudaEventDestroy(timing_done);
      return;
    }
  }

  cudaEventRecord(timing_start, stream);
  for (int iter = 0; iter < options.iterations; ++iter) {
    result.status = run_iteration();
    if (result.status != cutlass::Status::kSuccess) {
      result.passed = false;
      cudaEventDestroy(timing_start);
      cudaEventDestroy(timing_done);
      return;
    }
  }
  cudaEventRecord(timing_done, stream);
  cudaEventSynchronize(timing_done);

  float total_runtime_ms = 0.0f;
  cudaEventElapsedTime(&total_runtime_ms, timing_start, timing_done);

  if (options.split_timing) {
    cudaEventRecord(timing_start, stream);
    for (int iter = 0; iter < options.iterations; ++iter) {
      build_work_map();
    }
    cudaEventRecord(timing_done, stream);
    cudaEventSynchronize(timing_done);
    cudaEventElapsedTime(&split_builder_ms, timing_start, timing_done);
  }

  cudaEventDestroy(timing_start);
  cudaEventDestroy(timing_done);

  result.avg_runtime_ms = total_runtime_ms / static_cast<float>(options.iterations);
  result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

  if (!options.explore) {
    int64_t total_m = 0;
    int min_m = std::numeric_limits<int>::max();
    int max_m = 0;
    int zero_m_groups = 0;
    for (int i = 0; i < options.groups; ++i) {
      int m = cute::get<0>(options.problem_sizes_host[i]);
      total_m += m;
      min_m = std::min(min_m, m);
      max_m = std::max(max_m, m);
      if (m == 0) ++zero_m_groups;
    }
    const auto fixed_n = cute::get<1>(options.problem_sizes_host[0]);
    const auto fixed_k = cute::get<2>(options.problem_sizes_host[0]);

    std::cout << "  Problem Sizes G x (M, N, K), Alpha, Beta\n";
    if (min_m == max_m) {
      std::cout << "    " << options.groups << " x " << options.problem_sizes_host[0]
                << ", " << alpha_host[0] << ", " << beta_host[0] << '\n';
    }
    else {
      std::cout << "    " << options.groups << " x (M_var, " << fixed_n << ", " << fixed_k << ")"
                << ", " << alpha_host[0] << ", " << beta_host[0] << '\n'
                << "    M: total=" << total_m
                << " avg=" << (total_m / std::max(1, options.groups))
                << " min=" << min_m
                << " max=" << max_m
                << " zero_groups=" << zero_m_groups << '\n';
    }
    std::cout << "  Avg runtime : " << result.avg_runtime_ms * 1000.0 << " us\n"
              << "  GFLOPS      : " << result.gflops << '\n';
    if (options.split_timing) {
      double const builder_avg_ms =
          static_cast<double>(split_builder_ms) / static_cast<double>(options.iterations);
      double const gemm_residual_ms = std::max(0.0, result.avg_runtime_ms - builder_avg_ms);
      std::cout << "  Split timing: builder-only " << builder_avg_ms * 1000.0
                << " us, GEMM residual " << gemm_residual_ms * 1000.0 << " us\n";
    }
  }
}

template <
    typename Gemm,
    typename TileShape = DefaultTileShape,
    typename ClusterShape = DefaultClusterShape>
MixedDtypeResult run(Options &options)
{
  if (!setup) {
    printf("Setup input tensors.\n");
    allocate(options);
    initialize(options);
    setup = true;
  }

  MixedDtypeResult result;
  constexpr int CurrentTileShapeM = cute::size<0>(TileShape{});
  constexpr int CurrentTileShapeN = cute::size<1>(TileShape{});
  constexpr int CurrentClusterShapeM = cute::size<0>(ClusterShape{});
  constexpr int CurrentClusterShapeN = cute::size<1>(ClusterShape{});

  prepare_precomputed_work_tile_map<
      CurrentTileShapeM,
      CurrentTileShapeN,
      CurrentClusterShapeM,
      CurrentClusterShapeN>(options);

  Gemm gemm;
  auto arguments    = args_from_options<Gemm>(options);
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Soft-fail status checks: write status into result and return early on failure,
  // so the profiler (capture_results) can skip this config and continue with others.
  result.status = gemm.can_implement(arguments);
  if (result.status != cutlass::Status::kSuccess) {
    result.passed = false;
    return result;
  }

  result.status = gemm.initialize(arguments, workspace.get());
  if (result.status != cutlass::Status::kSuccess) {
    result.passed = false;
    return result;
  }

  build_precomputed_work_tile_map<
      CurrentTileShapeM,
      CurrentTileShapeN,
      CurrentClusterShapeM,
      CurrentClusterShapeN>(
      options,
      gemm.params().mainloop,
      nullptr);
  result.status = gemm.run();
  if (result.status != cutlass::Status::kSuccess) {
    result.passed = false;
    return result;
  }

  result.passed = verify(options);
  if (!result.passed) {
    return result;
  }
  profile_grouped_mixed_dtype<Gemm, TileShape, ClusterShape>(
      gemm, options, result, alpha_host, beta_host);

  return result;
}

template <
    typename KernelSchedule,
    typename ClusterShape,
    typename TileShape
>
void capture_results(Options options, std::vector<MixedDtypeResult> &results,
                     std::vector<std::string> &configs, std::string config) {
    using Ctor = gemm_constructor<KernelSchedule, ClusterShape, TileShape>;
    using gemm = typename Ctor::GemmScaleOnly;
    // Resolve the auto-picked pipeline stage count from the mainloop DispatchPolicy
    // and append it to the config label so every log line shows the real Stages.
    constexpr int resolved_stages =
        Ctor::CollectiveMainloopScaleOnly::DispatchPolicy::Stages;
    const std::string labeled_config =
        config + " Stages=" + std::to_string(resolved_stages);

    MixedDtypeResult result = run<gemm, TileShape, ClusterShape>(options);
    if (result.status != cutlass::Status::kSuccess) {
        // Do not push a failing config into the ranking vectors — it cannot be the "best".
        std::cout << "[SKIPPED] " << cutlassGetStatusString(result.status)
                  << "     "     << labeled_config << std::endl;
        return;
    }
    if (!result.passed) {
        std::cout << "[FAILED] correctness"
                  << "     " << labeled_config << std::endl;
        return;
    }
    std::cout << "Avg runtime : " << result.avg_runtime_ms << " ms  "
              << "GFLOPS : "      << result.gflops         << "     "
              << labeled_config   << std::endl;
    results.push_back(result);
    configs.push_back(labeled_config);
}

#include "precomputed_scheduler_work_map.hpp"

#endif // defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
