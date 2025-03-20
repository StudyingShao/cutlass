#pragma once

using namespace cute;

template <
    typename KernelSchedule,
    typename ClusterShape,
    typename TileShape
>
class gemm_constructor {
public:

    using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>; // <M,N,K> per group
    using MmaType = cutlass::float_e4m3_t;
    using QuantType = cutlass::int4b_t;
    // static const int TileShapeK = 128 * 8 / sizeof_bits<MmaType>::value;

    // A matrix configuration
    using         ElementA    = MmaType;
    using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
    static const int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Alignment of A matrix in units of elements (up to 16 bytes)

    // B matrix configuration
    using         ElementB    = QuantType;                                      // Element type for B matrix operand
    using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
    static const int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

    // This example manually swaps and transposes, so keep transpose of input layouts
    using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
    using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

    // Need to pass a pointer type to make the 3rd dimension of Stride be _0
    using StrideA = cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
    using StrideB = cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;

    // using ElementScale = cutlass::float_e4m3_t;
    // using ElementScale = float;
    // using ElementScale = cutlass::half_t;
    using ElementScalePacked = cutlass::Array<ElementScale, 1>;
    using LayoutScale = cutlass::layout::RowMajor;

    // C/D matrix configuration
    using         ElementC    = cutlass::half_t;                                // Element type for C and D matrix operands
    using         LayoutC     = cutlass::layout::RowMajor;                      // Layout type for C and D matrix operands
    static const int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

    // D matrix configuration
    using         ElementD    = ElementC;
    using         LayoutD     = LayoutC;
    static const int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;

    // Core kernel configurations
    using ElementAccumulator  = float;                                          // Element type for internal accumulation
    using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
    using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
    // using TileShape           = Shape<_128,_16,cute::Int<TileShapeK>>;          // Threadblock-level tile size
    // using ClusterShape        = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster
    using StageCountType = cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size
    // using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
    // using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;

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


    // =========================================================== MIXED INPUT WITH SCALES ===========================================================================
    // The Scale information must get paired with the operand that will be scaled. In this example, B is scaled so we make a tuple of B's information and the scale information.
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

template <
    typename KernelSchedule,
    typename ClusterShape,
    typename TileShape
>
void capture_results(Options options, std::vector<MixedDtypeResult> &results, std::vector<std::string> &configs, std::string config) {
    using gemm = typename gemm_constructor<KernelSchedule, ClusterShape, TileShape>::GemmScaleOnly;

    MixedDtypeResult result = run<gemm>(options, false);

    results.push_back(result);
    configs.push_back(config);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    typename KernelSchedule,
    typename ClusterShape
>
void dispatch_TileShape(Options options, std::vector<MixedDtypeResult> &results, std::vector<std::string> &configs, std::string config) {

    static const int TileShapeK = 128 * 8 / sizeof_bits<MmaType>::value;

    if constexpr (std::is_same_v<KernelSchedule, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>) {
        
        capture_results<KernelSchedule, ClusterShape, Shape< _64, _16, cute::Int<TileShapeK>>>(options, results, configs, config + " Shape< _64, _16, _" + std::to_string(TileShapeK) + ">");
        capture_results<KernelSchedule, ClusterShape, Shape< _64, _32, cute::Int<TileShapeK>>>(options, results, configs, config + " Shape< _64, _32, _" + std::to_string(TileShapeK) + ">");
        capture_results<KernelSchedule, ClusterShape, Shape< _64, _64, cute::Int<TileShapeK>>>(options, results, configs, config + " Shape< _64, _64, _" + std::to_string(TileShapeK) + ">");
        capture_results<KernelSchedule, ClusterShape, Shape< _64,_128, cute::Int<TileShapeK>>>(options, results, configs, config + " Shape< _64,_128, _" + std::to_string(TileShapeK) + ">");
    }

    capture_results<KernelSchedule, ClusterShape, Shape<_128, _16, cute::Int<TileShapeK>>>(options, results, configs, config + " Shape<_128, _16, _" + std::to_string(TileShapeK) + ">");
    capture_results<KernelSchedule, ClusterShape, Shape<_128, _32, cute::Int<TileShapeK>>>(options, results, configs, config + " Shape<_128, _32, _" + std::to_string(TileShapeK) + ">");
    capture_results<KernelSchedule, ClusterShape, Shape<_128, _64, cute::Int<TileShapeK>>>(options, results, configs, config + " Shape<_128, _64, _" + std::to_string(TileShapeK) + ">");
    capture_results<KernelSchedule, ClusterShape, Shape<_128,_128, cute::Int<TileShapeK>>>(options, results, configs, config + " Shape<_128,_128, _" + std::to_string(TileShapeK) + ">");
    
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    typename KernelSchedule
>
void dispatch_ClusterShape(Options options, std::vector<MixedDtypeResult> &results, std::vector<std::string> &configs, std::string config) {

    // Dispatch KernelSchedule
    dispatch_TileShape<KernelSchedule, Shape<_1,_1,_1>>(options, results, configs, config + " Shape<_1,_1,_1>");
    dispatch_TileShape<KernelSchedule, Shape<_2,_1,_1>>(options, results, configs, config + " Shape<_2,_1,_1>");
    dispatch_TileShape<KernelSchedule, Shape<_1,_2,_1>>(options, results, configs, config + " Shape<_1,_2,_1>");
    dispatch_TileShape<KernelSchedule, Shape<_2,_2,_1>>(options, results, configs, config + " Shape<_2,_2,_1>");
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void best_config_finder(Options options) {

    std::vector<MixedDtypeResult> results;
    std::vector<std::string> configs;

    // Dispatch KernelSchedule
    dispatch_ClusterShape<   cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>(options, results, configs, "   KernelPtrArrayTmaWarpSpecializedPingpong");
    dispatch_ClusterShape<cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>(options, results, configs, "KernelPtrArrayTmaWarpSpecializedCooperative");

    double max_gflops = 0.0;
    int best_config_id = -1;

    for (int i = 0; i < results.size(); i++)
    {
        std::cout << "Avg runtime : " << results[i].avg_runtime_ms << " ms"
                  << "GFLOPS : "      << results[i].gflops         << "   "
                  << configs[i]       << std::endl;
        
        if (results[i].gflops > max_gflops) {
            best_config_id = i;
            max_gflops = results[i].gflops;
        }
    }
    
    auto [M, N, K] = options.problem_sizes_host[0];
    printf("\nProblem %d x (%d, %d, %d)\n", options.groups, M, N, K);

    std::cout << "Best CUTLASS Config:"  << std::endl
              << "Avg Runtime : "        << results[best_config_id].avg_runtime_ms << " ms  "
              << "GFLOPS : "             << results[best_config_id].gflops         << "     "
              << configs[best_config_id] << std::endl;

}