#pragma once

/*
 * kernel_profiler.h — coordinator for parallel-compiled profiler.
 *
 * Each (KernelSchedule, ClusterShape) pair is compiled in its own .cu file
 * so that ninja/make can build all 8 translation units in parallel.
 *
 * To enable: uncomment `#define PROFILE` at the top of kernel_profiler_shared.hpp
 * and rebuild. When disabled, all profiler part files compile to empty TUs.
 */

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

// Forward declarations — each is defined in a separate profiler_*.cu file
void profile_pp_1x1x1  (Options&, std::vector<MixedDtypeResult>&, std::vector<std::string>&);
void profile_pp_2x1x1  (Options&, std::vector<MixedDtypeResult>&, std::vector<std::string>&);
void profile_pp_1x2x1  (Options&, std::vector<MixedDtypeResult>&, std::vector<std::string>&);
void profile_pp_2x2x1  (Options&, std::vector<MixedDtypeResult>&, std::vector<std::string>&);
void profile_coop_1x1x1(Options&, std::vector<MixedDtypeResult>&, std::vector<std::string>&);
void profile_coop_2x1x1(Options&, std::vector<MixedDtypeResult>&, std::vector<std::string>&);
void profile_coop_1x2x1(Options&, std::vector<MixedDtypeResult>&, std::vector<std::string>&);
void profile_coop_2x2x1(Options&, std::vector<MixedDtypeResult>&, std::vector<std::string>&);

inline void best_config_finder(Options options) {
    std::vector<MixedDtypeResult> results;
    std::vector<std::string>      configs;

    // Pingpong — 4 cluster shapes
    profile_pp_1x1x1(options, results, configs);
    profile_pp_2x1x1(options, results, configs);
    profile_pp_1x2x1(options, results, configs);
    profile_pp_2x2x1(options, results, configs);

    // Cooperative — 4 cluster shapes
    profile_coop_1x1x1(options, results, configs);
    profile_coop_2x1x1(options, results, configs);
    profile_coop_1x2x1(options, results, configs);
    profile_coop_2x2x1(options, results, configs);

    // Find best
    double max_gflops   = 0.0;
    int    best_config_id = -1;

    for (int i = 0; i < static_cast<int>(results.size()); i++) {
        std::cout << "Avg runtime : " << results[i].avg_runtime_ms << " ms  "
                  << "GFLOPS : "      << results[i].gflops         << "     "
                  << configs[i]       << std::endl;

        if (results[i].gflops > max_gflops) {
            best_config_id = i;
            max_gflops     = results[i].gflops;
        }
    }

    auto [M, N, K] = options.problem_sizes_host[0];
    printf("\nProblem %d x (%d, %d, %d)\n", options.groups, M, N, K);

    std::cout << "Best CUTLASS Config:"  << std::endl
              << "Avg Runtime : "        << results[best_config_id].avg_runtime_ms << " ms  "
              << "GFLOPS : "             << results[best_config_id].gflops         << "     "
              << configs[best_config_id] << std::endl;
}

#endif // defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
