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

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../55_hopper_mixed_dtype_gemm/mixed_dtype_utils.hpp"

template<class QuantType>
class GroupedMixedDtypeOptions : public MixedDtypeOptions {
public:
    using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int,int,int>>;
    using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

    bool explore = false;
    bool compare = true;
    int groups = 6;
    int c = 512;
    std::string benchmark_path;
    std::string expert_counts_path;
    std::vector<UnderlyingProblemShape> problem_sizes_host;

    GroupedMixedDtypeOptions() : MixedDtypeOptions()
    {
      m = 1024;
      n = 2048;
      k = 512;
    };

    void parse(int argc, char const **args) {
        cutlass::CommandLine cmd(argc, args);
        cmd.get_cmd_line_argument("groups", groups);
        MixedDtypeOptions::parse(argc, args);
        c = k;
        cmd.get_cmd_line_argument("c", c);
        cmd.get_cmd_line_argument("benchmark", benchmark_path);
        cmd.get_cmd_line_argument("expert_counts", expert_counts_path);

        if (!expert_counts_path.empty()) {
            problem_sizes_host = load_expert_counts_problems();
        } else if (!benchmark_path.empty()) {
            problem_sizes_host = load_benchmark_problems();
        } else {
            problem_sizes_host = randomize_problems(cmd);
        }
    }

    std::ostream& print_usage(std::ostream& out) const {
        out << "69_hopper_mixed_dtype_grouped_gemm\n\n"
            << "Options:\n"
            << "  --help                      Display this usage statement\n"
            << "  --m=<int>                   Sets the M extent of the GEMM for all groups\n"
            << "  --n=<int>                   Sets the N extent of the GEMM for all groups\n"
            << "  --k=<int>                   Sets the K extent of the GEMM for all groups\n"
            << "  --groups=<int>              Sets the number of individual GEMM problems\n"
            << "  --c=<int>                   The size of each chunk for the scales and zeros. To broadcast a vector of scales or zeros, set the group size to K.\n"
            << "  --mode=<int>                The mode to run the gemm\n"
            << "  --alpha=<f32>               Epilogue scalar alpha\n"
            << "  --beta=<f32>                Epilogue scalar beta\n"
            << "  --iterations=<int>          Number of profiling iterations\n"
            << "  --warmup=<int>              Number of warmup iterations\n"
            << "  --benchmark=<str>           Executes a benchmark problem size\n"
            << "  --expert_counts=<str>       Path to a file with per-expert M (token) counts.\n"
            << "                              One integer per entry (whitespace/newline separated,\n"
            << "                              lines starting with '#' are ignored).\n"
            << "                              --n and --k are used for fixed N/K across all groups;\n"
            << "                              --groups is overridden by the file length.\n";
        return out;
    }

    double gflops(double runtime_s) const {
        uint64_t fmas = std::accumulate(problem_sizes_host.begin(), problem_sizes_host.end(), 0ULL,
            [](uint64_t sum, const UnderlyingProblemShape& problem) {
                return sum + static_cast<uint64_t>(cute::get<0>(problem)) *
                             static_cast<uint64_t>(cute::get<1>(problem)) *
                             static_cast<uint64_t>(cute::get<2>(problem));
            });
        return (2.0 * fmas) / (runtime_s * 1e9);
    }

private:
    static constexpr int tma_alignment_bits = 128;
    const int alignment = tma_alignment_bits / cutlass::sizeof_bits<QuantType>::value;

    std::vector<UnderlyingProblemShape> randomize_problems(cutlass::CommandLine& cmd) {
        std::vector<UnderlyingProblemShape> problems;
        problems.reserve(groups);

        int cmd_line_m = -1, cmd_line_n = -1, cmd_line_k = -1;
        cmd.get_cmd_line_argument("m", cmd_line_m);
        cmd.get_cmd_line_argument("n", cmd_line_n);
        cmd.get_cmd_line_argument("k", cmd_line_k);

        for (int i = 0; i < groups; ++i) {
            int m = (cmd_line_m >= 0) ? cmd_line_m : alignment * ((rand() % 64) + 1);
            int n = (cmd_line_n >= 0) ? cmd_line_n : this->n;
            int k = (cmd_line_k >= 0) ? cmd_line_k : this->k;

            if (k % alignment != 0) {
                throw std::runtime_error("Error: k dimension must be a multiple of " + std::to_string(alignment));
            }
            problems.push_back({m, n, k});
        }
        return problems;
    }

    bool try_parse_count_token(std::string const& tok, int& value) const {
        try {
            size_t consumed = 0;
            int parsed = std::stoi(tok, &consumed);
            if (consumed != tok.size() || parsed < 0) return false;
            value = parsed;
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    void append_count_tokens(std::vector<UnderlyingProblemShape>& problems,
                             std::string const& text) const {
        std::istringstream iss(text);
        std::string tok;
        while (iss >> tok) {
            int m = 0;
            if (try_parse_count_token(tok, m)) {
                problems.push_back({m, this->n, this->k});
            }
        }
    }

    // Load per-expert M counts from a text file. Uses the global --n and --k as the
    // (fixed) N/K for every group; groups is overridden by the number of entries found.
    //
    // File format:
    //   - One non-negative integer per entry (whitespace or newline separated).
    //   - Lines containing '=' are treated as logs like "expert[i] = count"; only
    //     the right-hand side is parsed so expert indices are never mistaken for M.
    //   - Lines starting with '#' are treated as comments.
    std::vector<UnderlyingProblemShape> load_expert_counts_problems() {
        std::ifstream file(expert_counts_path);
        if (!file) {
            throw std::runtime_error("Failed to open expert_counts file: " + expert_counts_path);
        }

        if (this->k % alignment != 0) {
            throw std::runtime_error("Error: k dimension must be a multiple of " +
                                     std::to_string(alignment));
        }

        std::vector<UnderlyingProblemShape> problems;
        std::string line;
        while (std::getline(file, line)) {
            auto hash = line.find('#');
            if (hash != std::string::npos) line.erase(hash);

            auto equals = line.find('=');
            if (equals != std::string::npos) {
                append_count_tokens(problems, line.substr(equals + 1));
            } else {
                append_count_tokens(problems, line);
            }
        }

        if (problems.empty()) {
            throw std::runtime_error("No expert counts parsed from: " + expert_counts_path);
        }

        groups = static_cast<int>(problems.size());
        return problems;
    }

    std::vector<UnderlyingProblemShape> load_benchmark_problems() {
        std::ifstream file(benchmark_path);
        if (!file) {
            throw std::runtime_error("Failed to open benchmark file: " + benchmark_path);
        }

        std::vector<UnderlyingProblemShape> problems;
        int idx;
        std::string extent_str;

        while (file >> idx >> extent_str) {
            if (idx < 0 || extent_str.empty()) break;

            std::vector<std::string> tokens;
            cutlass::CommandLine::tokenize(tokens, extent_str, 'x');
            
            cutlass::gemm::GemmCoord extent;
            for (int i = 0; i < std::min(3, static_cast<int>(tokens.size())); ++i) {
                int x = std::stoi(tokens[i]);
                extent.at(i) = (x % alignment) ? x + (alignment - (x % alignment)) : x;
            }

            if (extent.product()) {
                problems.push_back({extent.m(), extent.n(), extent.k()});
            }
        }
        if (problems.empty()) {
            throw std::runtime_error("No benchmark problems parsed from: " + benchmark_path);
        }
        groups = static_cast<int>(problems.size());
        return problems;
    }
};

template <class QuantType, class Gemm, class ElementAccumulator>
void grouped_mixed_dtype_profiling(
    Gemm& gemm,
    const GroupedMixedDtypeOptions<QuantType>& options,
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
        // Soft-fail: write status to `result` and return early instead of exit(),
        // so the profiler explore loop can skip this config and continue.
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
        // Compute aggregate M stats so variable-M (e.g. MoE expert_counts) runs
        // are printed accurately instead of just showing the first group's M.
        int64_t total_m = 0;
        int     min_m   = std::numeric_limits<int>::max();
        int     max_m   = 0;
        int     zero_m_groups = 0;
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
        } else {
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
