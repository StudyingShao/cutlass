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
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
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

#if defined(CUTLASS_MIXED_GEMM_MXFP4_BF16)
using MmaType = cutlass::bfloat16_t;     // activations
using QuantType = cutlass::float_e2m1_t; // weights
#define GROUP_SIZE 32
using ElementScale = cutlass::float_ue8m0_t;
inline constexpr int TileShapeM = 128;
inline constexpr int TileShapeN = 16;
inline constexpr int TileShapeK = CUTLASS_MIXED_GEMM_TILE_SHAPE_K;
#elif defined(CUTLASS_MIXED_GEMM_MXFP4_FP8)
// MXFP4 x FP8 experimental path.  This keeps the same MXFP4 weight semantics
// as the MXFP4 x BF16 path: e2m1 payload, UE8M0 block scale, group size 32.
// The offline weight layout still follows the W4A8 INT4xFP8 path.
using MmaType = cutlass::float_e4m3_t;      // activations
using QuantType = cutlass::float_e2m1_t;    // weights
#define GROUP_SIZE 32
using ElementScale = cutlass::float_ue8m0_t;
inline constexpr int TileShapeM = 128;
inline constexpr int TileShapeN = 16;
inline constexpr int TileShapeK = CUTLASS_MIXED_GEMM_TILE_SHAPE_K;
#elif defined(CUTLASS_MIXED_GEMM_MXFP4_MXFP8)
// MXFP4 x MXFP8 keeps the activation payload as plain FP8 and reads the
// activation UE8M0 scale from its original M x K/32, K-contiguous layout.
using MmaType = cutlass::float_e4m3_t;      // activation payload
using QuantType = cutlass::float_e2m1_t;    // weights
#define GROUP_SIZE 32
using ElementScale = cutlass::float_ue8m0_t;
inline constexpr int TileShapeM = 128;
inline constexpr int TileShapeN = 16;
inline constexpr int TileShapeK = CUTLASS_MIXED_GEMM_TILE_SHAPE_K;
#else
// INT4 x FP8
using MmaType = cutlass::float_e4m3_t;      // activations
using QuantType = cutlass::int4b_t;         // weights
#define GROUP_SIZE 128
using ElementScale = cutlass::bfloat16_t;
inline constexpr int TileShapeM = 128;
inline constexpr int TileShapeN = 16;
inline constexpr int TileShapeK = CUTLASS_MIXED_GEMM_TILE_SHAPE_K;
#endif

//--------------------------------------------------------------------------------------------

// Weight scales are the normal mixed-input scale attached to the 4-bit weight operand.
using ElementScalePacked = cutlass::Array<ElementScale, TileShapeK / GROUP_SIZE>;
inline constexpr bool ScaleAppliesToActivation =
#if defined(CUTLASS_MIXED_GEMM_MXFP4_MXFP8)
    true;
#else
    false;
#endif
using ElementActivationScale = cutlass::float_ue8m0_t;
using ElementActivationScalePacked = cutlass::Array<ElementActivationScale, TileShapeK / GROUP_SIZE>;

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

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("explore", explore);
    cmd.get_cmd_line_argument("compare", compare);
    cmd.get_cmd_line_argument("enable_print", enable_print);
    cmd.get_cmd_line_argument("enable_print_weight", enable_print_weight);
    cmd.get_cmd_line_argument("debug_input_act", debug_input_act);
    cmd.get_cmd_line_argument("debug_input_weight", debug_input_weight);
    cmd.get_cmd_line_argument("debug_input_scale", debug_input_scale);
    cmd.get_cmd_line_argument("swizzle", swizzle);
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
      << "  --swizzle=<int>             Tile scheduler swizzle size (1, 2, 4, or 8). Default: 2\n"
      << "  --benchmark=<str>           Executes a benchmark problem size.\n";

    out << "\n\nExamples:\n\n"
      << "$ " << "69_hopper_int4_fp8_grouped_gemm"
      << " --m=1024 --n=512 --k=1024 --groups=10 --alpha=1 --beta=0 \n\n";

    return out;
  }
};

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
using ArchTag            = cutlass::arch::Sm90;
using OperatorClass      = cutlass::arch::OpClassTensorOp;
using StageCountType     = cutlass::gemm::collective::StageCountAuto;

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::
    PersistentTileSchedulerSm90GroupParams<Shape<int,int,int>>::RasterOrderOptions;

// Default configuration (used for the normal non-profiler run path, and to derive stride types)
using DefaultTileShape    = Shape<cute::Int<TileShapeM>, cute::Int<TileShapeN>, cute::Int<TileShapeK>>;
using DefaultClusterShape = Shape<_2,_1,_1>;
using DefaultKernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using DefaultEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;

using DefaultCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    DefaultTileShape, DefaultClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type *, AlignmentC,
    ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type *, AlignmentD,
    DefaultEpilogueSchedule
>::CollectiveOp;

using DefaultCollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    cute::tuple<ElementB, ElementScalePacked>, LayoutB_Transpose *, AlignmentB,
    ElementA, LayoutA_Transpose *, AlignmentA,
    ElementAccumulator,
    DefaultTileShape, DefaultClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename DefaultCollectiveEpilogue::SharedStorage))>,
    DefaultKernelSchedule
>::CollectiveOp;

using DefaultGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    DefaultCollectiveMainloop,
    DefaultCollectiveEpilogue
>;

using DefaultGemm    = cutlass::gemm::device::GemmUniversalAdapter<DefaultGemmKernel>;
using GemmScaleOnly  = DefaultGemm;  // alias for backward compatibility with main .cu

// Stride types (stable for any config using the same element types and layouts)
using StrideC     = typename DefaultGemmKernel::InternalStrideC;
using StrideD     = typename DefaultGemmKernel::InternalStrideD;
using StrideC_ref = cutlass::detail::TagToStrideC_t<LayoutC>;
using StrideD_ref = cutlass::detail::TagToStrideC_t<LayoutD>;
using StrideS     = typename DefaultCollectiveMainloop::StrideScale;
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

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type *, AlignmentC,
        ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type *, AlignmentD,
        EpilogueSchedule
    >::CollectiveOp;

    using CollectiveMainloopScaleOnly = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        cute::tuple<ElementB, ElementScalePacked>, LayoutB_Transpose *, AlignmentB,
        ElementA, LayoutA_Transpose *, AlignmentA,
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
extern std::vector<StrideS>     stride_activation_scale_host;

extern std::vector<ElementAccumulator> alpha_host;
extern std::vector<ElementAccumulator> beta_host;

extern uint64_t seed;
extern bool setup;

extern cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

extern cutlass::DeviceAllocation<MmaType>                                               block_A;
extern cutlass::DeviceAllocation<QuantType>                                             block_B;
extern cutlass::DeviceAllocation<QuantType>                                             block_B_interleaved;
extern cutlass::DeviceAllocation<ElementScale>                                          block_weight_scale;
extern cutlass::DeviceAllocation<ElementScalePacked>                                    block_weight_scale_packed;
extern cutlass::DeviceAllocation<ElementActivationScale>                                block_activation_scale;
extern cutlass::DeviceAllocation<ElementActivationScalePacked>                          block_activation_scale_packed;
extern cutlass::DeviceAllocation<ElementZero>                                           block_zero;
extern cutlass::DeviceAllocation<ElementC>                                              block_C;
extern cutlass::DeviceAllocation<typename DefaultGemm::EpilogueOutputOp::ElementOutput> block_D;
extern cutlass::DeviceAllocation<typename DefaultGemm::EpilogueOutputOp::ElementOutput> block_ref_D;

extern cutlass::DeviceAllocation<const MmaType *>                    ptr_A;
extern cutlass::DeviceAllocation<const QuantType *>                  ptr_B;
extern cutlass::DeviceAllocation<const ElementActivationScale *>     ptr_activation_scale;
extern cutlass::DeviceAllocation<ElementActivationScalePacked *>     ptr_activation_scale_packed;
extern cutlass::DeviceAllocation<const ElementScalePacked *>         ptr_weight_scale_packed;
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
extern cutlass::DeviceAllocation<StrideS>     stride_activation_scale;

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
void prepare_activation_scale_tensor(Options const& options);

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Template implementations (must live in header for cross-TU instantiation)
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
typename Gemm::Arguments args_from_options(Options const& options, bool host_problem_shapes_available = true)
{
  using Args = typename Gemm::Arguments;
  auto dB = stride_B.get();
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  Args arguments;
  decltype(arguments.epilogue.thread) fusion_args;

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

  decltype(arguments.mainloop) mainloop_args{
    ptr_B.get(), dB, ptr_A.get(), stride_A.get(), ptr_weight_scale_packed.get(), stride_weight_scale.get(), GROUP_SIZE
  };
  if constexpr (ScaleAppliesToActivation) {
    // The repack kernel writes through this pointer array; the GEMM mainloop only reads it.
    mainloop_args.ptr_ActivationScale =
        const_cast<decltype(mainloop_args.ptr_ActivationScale)>(ptr_activation_scale_packed.get());
    mainloop_args.dActivationScale = stride_activation_scale.get();
  }

  arguments = Args {
    cutlass::gemm::GemmUniversalMode::kGrouped,
    {options.groups, problem_sizes.get(), nullptr},
    mainloop_args,
    {fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
    hw_info
  };

  arguments.scheduler.max_swizzle_size = options.swizzle;
  arguments.scheduler.raster_order     = RasterOrderOptions::Heuristic;

  return arguments;
}

template <class Gemm>
void profile_grouped_mixed_dtype(
    Gemm& gemm,
    const Options& options,
    MixedDtypeResult& result,
    const std::vector<ElementAccumulator>& alpha_host,
    const std::vector<ElementAccumulator>& beta_host) {

  if (options.iterations <= 0) return;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::vector<float> runtimes;
  runtimes.reserve(options.iterations);

  for (int iter = 0; iter < options.warmup + options.iterations; ++iter) {
    cudaEventRecord(start);
    if constexpr (ScaleAppliesToActivation) {
      prepare_activation_scale_tensor(options);
    }
    result.status = gemm.run();
    if (result.status != cutlass::Status::kSuccess) {
      result.passed = false;
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    if (iter >= options.warmup) {
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      runtimes.push_back(milliseconds);
    }
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  if (runtimes.empty()) return;
  result.avg_runtime_ms = std::accumulate(runtimes.begin(), runtimes.end(), 0.0f) / runtimes.size();
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
  }
}

template <typename Gemm>
MixedDtypeResult run(Options &options, bool host_problem_shapes_available = true)
{
  if (!setup) {
    printf("Setup input tensors.\n");
    allocate(options);
    initialize(options);
    setup = true;
  }

  MixedDtypeResult result;

  Gemm gemm;
  auto arguments    = args_from_options<Gemm>(options, host_problem_shapes_available);
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

  result.status = gemm.run();
  if (result.status != cutlass::Status::kSuccess) {
    result.passed = false;
    return result;
  }

  result.passed = verify(options);
  profile_grouped_mixed_dtype(gemm, options, result, alpha_host, beta_host);

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

    MixedDtypeResult result = run<gemm>(options, false);
    if (result.status != cutlass::Status::kSuccess) {
        // Do not push a failing config into the ranking vectors — it cannot be the "best".
        std::cout << "[SKIPPED] " << cutlassGetStatusString(result.status)
                  << "     "     << labeled_config << std::endl;
        return;
    }
    std::cout << "Avg runtime : " << result.avg_runtime_ms << " ms  "
              << "GFLOPS : "      << result.gflops         << "     "
              << labeled_config   << std::endl;
    results.push_back(result);
    configs.push_back(labeled_config);
}

#endif // defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
