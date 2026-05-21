#include "kernel_profiler_shared.hpp"
#ifdef PROFILE

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

void profile_coop_2x2x1(Options& options, std::vector<MixedDtypeResult>& results, std::vector<std::string>& configs) {
    using Schedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
    using Cluster  = Shape<_2,_2,_1>;
    const std::string K  = "_" + std::to_string(TileShapeK);
    const std::string PX = "KernelPtrArrayTmaWarpSpecializedCooperative Shape<_2,_2,_1>";

    capture_results<Schedule, Cluster, Shape<_128, _16, cute::Int<TileShapeK>>>(options, results, configs, PX + " Shape<_128, _16, " + K + ">");
    capture_results<Schedule, Cluster, Shape<_128, _32, cute::Int<TileShapeK>>>(options, results, configs, PX + " Shape<_128, _32, " + K + ">");
    capture_results<Schedule, Cluster, Shape<_128, _64, cute::Int<TileShapeK>>>(options, results, configs, PX + " Shape<_128, _64, " + K + ">");

    if constexpr (TileShapeK <= 256) {
        capture_results<Schedule, Cluster, Shape<_128,_128, cute::Int<TileShapeK>>>(options, results, configs, PX + " Shape<_128,_128, " + K + ">");
        constexpr bool skip_mxfp4_bf16_k256_large_n =
            TileShapeK == 256 &&
            cute::is_same_v<MmaType, cutlass::bfloat16_t> &&
            cute::is_same_v<QuantType, cutlass::float_e2m1_t>;
        if constexpr (!skip_mxfp4_bf16_k256_large_n) {
            capture_results<Schedule, Cluster, Shape<_128,_256, cute::Int<TileShapeK>>>(options, results, configs, PX + " Shape<_128,_256, " + K + ">");
            capture_results<Schedule, Cluster, Shape<_256,_128, cute::Int<TileShapeK>>>(options, results, configs, PX + " Shape<_256,_128, " + K + ">");
        }
    }
    if constexpr (TileShapeK == 128) {
        capture_results<Schedule, Cluster, Shape<_256,_256, cute::Int<TileShapeK>>>(options, results, configs, PX + " Shape<_256,_256, " + K + ">");
    }
}

#endif // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED
#endif // PROFILE
