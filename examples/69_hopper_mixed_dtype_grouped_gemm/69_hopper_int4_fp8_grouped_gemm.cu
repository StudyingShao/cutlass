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


/*! \file
    \brief
    Hopper Mixed-input Grouped GEMM example using CUTLASS 3 APIs for NVIDIA Hopper architecture.
    See 55_hopper_int4_fp8_gemm.cu for more details about W4A8 GEMMs with lookup table.

    Limitations:
      1) Only support row-wise scaling. Zero-points and block-wise scaling is currently not supported.

    To run this example:

      $ ./examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_fp8_grouped_gemm --m=2048 --n=2048 --k=2048 --mode=1 --groups=10

    To enable the kernel config profiler (parallel compilation, ~8x faster build):
      1) Uncomment `#define PROFILE` at the top of kernel_profiler_shared.hpp
      2) make -j 69_hopper_int4_fp8_grouped_gemm
      3) ./69_hopper_int4_fp8_grouped_gemm --explore=true ...
*/

#include "kernel_profiler_shared.hpp"
#include "host_validation.hpp"
#include "BF16_MXFP4_Test.h"

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Global variable definitions (extern-declared in kernel_profiler_shared.hpp)
/////////////////////////////////////////////////////////////////////////////////////////////////

// Host-side allocations
std::vector<int64_t> offset_A;
std::vector<int64_t> offset_B;
std::vector<int64_t> offset_C;
std::vector<int64_t> offset_D;
std::vector<int64_t> offset_scale;
std::vector<int64_t> offset_zero;

std::vector<StrideA>     stride_A_host;
std::vector<StrideB>     stride_B_host;
std::vector<StrideC>     stride_C_host;
std::vector<StrideD>     stride_D_host;
std::vector<StrideC_ref> stride_C_host_ref;
std::vector<StrideD_ref> stride_D_host_ref;
std::vector<StrideS>     stride_S_host;
std::vector<StrideS_ref> stride_S_host_ref;

std::vector<ElementAccumulator> alpha_host;
std::vector<ElementAccumulator> beta_host;

uint64_t seed = 2020;

// Device-side allocations
cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

cutlass::DeviceAllocation<MmaType>            block_A;
cutlass::DeviceAllocation<QuantType>          block_B;
cutlass::DeviceAllocation<QuantType>          block_B_interleaved;
cutlass::DeviceAllocation<ElementScale>       block_scale;
cutlass::DeviceAllocation<ElementScalePacked> block_scale_packed;
cutlass::DeviceAllocation<ElementZero>        block_zero;
cutlass::DeviceAllocation<ElementC>           block_C;
cutlass::DeviceAllocation<typename DefaultGemm::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename DefaultGemm::EpilogueOutputOp::ElementOutput> block_ref_D;

cutlass::DeviceAllocation<const MmaType *>            ptr_A;
cutlass::DeviceAllocation<const QuantType *>          ptr_B;
cutlass::DeviceAllocation<const ElementScalePacked *> ptr_scale_packed;
cutlass::DeviceAllocation<const ElementZero *>        ptr_zero;
cutlass::DeviceAllocation<const ElementC *>           ptr_C;
cutlass::DeviceAllocation<typename DefaultGemm::EpilogueOutputOp::ElementOutput *> ptr_D;

cutlass::DeviceAllocation<StrideA>     stride_A;
cutlass::DeviceAllocation<StrideB>     stride_B;
cutlass::DeviceAllocation<StrideC>     stride_C;
cutlass::DeviceAllocation<StrideD>     stride_D;
cutlass::DeviceAllocation<StrideC_ref> stride_C_ref;
cutlass::DeviceAllocation<StrideD_ref> stride_D_ref;
cutlass::DeviceAllocation<StrideS_ref> stride_S_ref;
cutlass::DeviceAllocation<StrideS>     stride_S;

cutlass::DeviceAllocation<ElementAccumulator*> alpha_device;
cutlass::DeviceAllocation<ElementAccumulator*> beta_device;
cutlass::DeviceAllocation<ElementAccumulator>  block_alpha;
cutlass::DeviceAllocation<ElementAccumulator>  block_beta;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed functions
/////////////////////////////////////////////////////////////////////////////////////////////////

void allocate(Options const& options) {
  int64_t total_elements_A = 0;
  int64_t total_elements_B = 0;
  int64_t total_elements_C = 0;
  int64_t total_elements_D = 0;
  int64_t total_elements_scale = 0;
  int64_t total_elements_zero = 0;

  for (int32_t i = 0; i < options.groups; ++i) {

    auto problem = options.problem_sizes_host.at(i);
    auto M = get<0>(problem);
    auto N = get<1>(problem);
    auto K = get<2>(problem);

    const int scale_k = K / TileShapeK;

    offset_A.push_back(total_elements_A);
    offset_B.push_back(total_elements_B * cutlass::sizeof_bits<QuantType>::value / 8);
    offset_C.push_back(total_elements_C);
    offset_D.push_back(total_elements_D);
    offset_scale.push_back(total_elements_scale);
    offset_zero.push_back(total_elements_zero);

    int64_t elements_A     = M * K;
    int64_t elements_B     = K * N;
    int64_t elements_C     = M * N;
    int64_t elements_D     = M * N;
    int64_t elements_scale = scale_k * N;
    int64_t elements_zero  = scale_k * N;

    total_elements_A     += elements_A;
    total_elements_B     += elements_B;
    total_elements_C     += elements_C;
    total_elements_D     += elements_D;
    total_elements_scale += elements_scale;
    total_elements_zero  += elements_zero;

    stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
    stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
    stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {N, M, 1}));
    stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {N, M, 1}));
    stride_C_host_ref.push_back(cutlass::make_cute_packed_stride(StrideC_ref{}, {M, N, 1}));
    stride_D_host_ref.push_back(cutlass::make_cute_packed_stride(StrideD_ref{}, {M, N, 1}));
    stride_S_host_ref.push_back(cutlass::make_cute_packed_stride(StrideS_ref{}, {N, scale_k, 1}));
    stride_S_host.push_back(cutlass::make_cute_packed_stride(StrideS{}, {N, scale_k, 1}));
  }

  block_A.reset(total_elements_A);
  block_B.reset(total_elements_B);
  block_B_interleaved.reset(total_elements_B);
  block_C.reset(total_elements_C);
  block_D.reset(total_elements_D);
  block_ref_D.reset(total_elements_D);
  block_scale.reset(total_elements_scale);
  block_scale_packed.reset(total_elements_scale);
  block_zero.reset(total_elements_zero);

  block_alpha.reset(options.groups);
  block_beta.reset(options.groups);
}

void initialize(Options& options) {

  [[maybe_unused]]uint64_t seed = 2020;

  problem_sizes.reset(options.groups);
  problem_sizes.copy_from_host(options.problem_sizes_host.data());

  std::vector<MmaType *>            ptr_A_host(options.groups);
  std::vector<QuantType *>          ptr_B_host(options.groups);
  std::vector<ElementC *>           ptr_C_host(options.groups);
  std::vector<ElementC *>           ptr_D_host(options.groups);
  std::vector<ElementScalePacked *> ptr_scale_packed_host(options.groups);
  std::vector<ElementZero *>        ptr_zero_host(options.groups);
  std::vector<ElementAccumulator *> ptr_alpha_host(options.groups);
  std::vector<ElementAccumulator *> ptr_beta_host(options.groups);

  for (int32_t i = 0; i < options.groups; ++i) {
    ptr_A_host.at(i)            = block_A.get() + offset_A.at(i);
    ptr_B_host.at(i)            = block_B_interleaved.get() + offset_B.at(i);
    ptr_C_host.at(i)            = block_C.get() + offset_C.at(i);
    ptr_D_host.at(i)            = block_D.get() + offset_D.at(i);
    ptr_scale_packed_host.at(i) = block_scale_packed.get() + offset_scale.at(i);
    ptr_zero_host.at(i)         = block_zero.get() + offset_zero.at(i);
    alpha_host.push_back((options.alpha == FLT_MAX) ? static_cast<ElementAccumulator>((rand() % 5) + 1) : options.alpha);
    beta_host.push_back( (options.beta  == FLT_MAX) ? static_cast<ElementAccumulator>(rand() % 5)       : options.beta);
    ptr_alpha_host.at(i) = block_alpha.get() + i;
    ptr_beta_host.at(i)  = block_beta.get() + i;
  }

  ptr_A.reset(options.groups);  ptr_A.copy_from_host(ptr_A_host.data());
  ptr_B.reset(options.groups);  ptr_B.copy_from_host(ptr_B_host.data());
  ptr_C.reset(options.groups);  ptr_C.copy_from_host(ptr_C_host.data());
  ptr_D.reset(options.groups);  ptr_D.copy_from_host(ptr_D_host.data());

  ptr_scale_packed.reset(options.groups);
  ptr_scale_packed.copy_from_host(ptr_scale_packed_host.data());

  ptr_zero.reset(options.groups);
  ptr_zero.copy_from_host(ptr_zero_host.data());

  stride_A.reset(options.groups);   stride_A.copy_from_host(stride_A_host.data());
  stride_B.reset(options.groups);   stride_B.copy_from_host(stride_B_host.data());
  stride_C.reset(options.groups);   stride_C.copy_from_host(stride_C_host.data());
  stride_D.reset(options.groups);   stride_D.copy_from_host(stride_D_host.data());
  stride_C_ref.reset(options.groups); stride_C_ref.copy_from_host(stride_C_host_ref.data());
  stride_D_ref.reset(options.groups); stride_D_ref.copy_from_host(stride_D_host_ref.data());
  stride_S_ref.reset(options.groups); stride_S_ref.copy_from_host(stride_S_host_ref.data());
  stride_S.reset(options.groups);   stride_S.copy_from_host(stride_S_host.data());

  alpha_device.reset(options.groups); alpha_device.copy_from_host(ptr_alpha_host.data());
  beta_device.reset(options.groups);  beta_device.copy_from_host(ptr_beta_host.data());

  /////////////////////////////////////////////////////////////////////////////////////////////////////////

  initialize_tensor(block_A, seed + 2023);
  set_device<<<1, 1>>>(options.debug_input_act, block_A.get(), block_A.size(), 1);
  print_device<<<1, 1>>>(options.enable_print, block_A.get(), block_A.size(), options.groups, 'A');
  cudaDeviceSynchronize();

  /////////////////////////////////////////////////////////////////////////////////////////////////////////

  initialize_quant_tensor(block_B, seed + 2022);

  set_device_int4<<<1, 1>>>(options.debug_input_weight, block_B.get(), block_B.size(), 1);
  print_device_4b<<<1, 1>>>(options.enable_print_weight, block_B.get(), options.groups * options.n, options.k, 'B');

  if constexpr (cute::is_same_v<QuantType, cutlass::float_e2m1_t> &&
      cute::is_same_v<MmaType, cutlass::bfloat16_t>)
  {
    for (int32_t i = 0; i < options.groups; ++i) {
      auto problem = options.problem_sizes_host.at(i);
      auto N = get<1>(problem);
      auto K = get<2>(problem);
      interleave_fp4xbf16_Hopper<QuantType>(
        block_B.get() + offset_B.at(i), block_B_interleaved.get() + offset_B.at(i),
        N, K);
    }
  }
  else if constexpr (cute::is_same_v<QuantType, cutlass::int4b_t> &&
    cute::is_same_v<MmaType, cutlass::float_e4m3_t>)
  {
    for (int32_t i = 0; i < options.groups; ++i) {
      auto problem = options.problem_sizes_host.at(i);
      auto N = get<1>(problem);
      auto K = get<2>(problem);
      interleave_int4xfp8_Hopper<QuantType>(
        block_B.get() + offset_B.at(i), block_B_interleaved.get() + offset_B.at(i),
        N, K);
    }
  }
  else
  {
    block_B_interleaved.copy_from_device(block_B.get());
  }
  print_device_4b<<<1, 1>>>(options.enable_print_weight, block_B_interleaved.get(), options.groups * options.n, options.k, 'B');

  /////////////////////////////////////////////////////////////////////////////////////////////////////////

  initialize_tensor(block_C, seed + 2021);
  print_device<<<1, 1>>>(options.enable_print, block_C.get(), block_C.size(), options.groups, 'C');
  cudaDeviceSynchronize();

  /////////////////////////////////////////////////////////////////////////////////////////////////////////

  if constexpr (cute::is_same_v<ElementScale, cutlass::float_ue8m0_t>) {
    set_device_ue8m0<<<1, 1>>>(options.debug_input_scale, block_scale.get(), block_scale.size());
  }
  else {
    set_device_sequential<<<1, 1>>>(block_scale.get(), block_scale.size(), 5678);
  }
  print_device<<<1, 1>>>(options.enable_print, block_scale.get(), block_scale.size(), options.groups, 'S');
  cudaDeviceSynchronize();

  cutlass::pack_scale_fp32(options.debug_input_scale, block_scale.get(), block_scale_packed.get(), block_scale.size(), ElementScalePacked::kElements);
  print_device_packed<<<1, 1>>>(options.enable_print, block_scale_packed.get(), block_scale.size(), 'P');
  /////////////////////////////////////////////////////////////////////////////////////////////////////////

  groupwise_verify(
    problem_sizes.get(),
    options.groups,
    block_A.get(), block_B.get(), block_scale_packed.get(), block_ref_D.get(),
    TileShapeK,
    GROUP_SIZE,
    stride_A.get(), stride_B.get()
  );

  print_device<<<1,1>>>(options.enable_print, block_ref_D.get(), block_ref_D.size(), options.groups, 'R');

  initialize_zero(block_zero, options);
  block_alpha.copy_from_host(alpha_host.data());
  block_beta.copy_from_host(beta_host.data());

  // Swap MN -> NM for SwapAB (device only; restore host copy below)
  for (int32_t i = 0; i < options.groups; ++i) {
    auto [M, N, K] = options.problem_sizes_host[i];
    options.problem_sizes_host[i] = make_tuple(N, M, K);
  }
  problem_sizes.reset(options.groups);
  problem_sizes.copy_from_host(options.problem_sizes_host.data());
  // Restore host copy to original MN order
  for (int32_t i = 0; i < options.groups; ++i) {
    auto [M, N, K] = options.problem_sizes_host[i];
    options.problem_sizes_host[i] = make_tuple(N, M, K);
  }
}

bool verify(Options const& options) {
  bool passed = true;

  const ElementD epsilon(1e-2f);
  const ElementD non_zero_floor(1e-4f);

  for (int32_t i = 0; i < options.groups; ++i) {
    auto problem = options.problem_sizes_host.at(i);
    auto M = get<0>(problem);  // original M (after swap-back)
    auto N = get<1>(problem);  // original N
    if (M == 0) {
      continue;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    passed &= cutlass::reference::device::BlockCompareRelativelyEqual(
      block_ref_D.get() + offset_D.at(i), block_D.get() + offset_D.at(i), M * N, epsilon, non_zero_floor);
  }

  compare_device<<<1,1>>>(options.compare, block_D.get(), block_ref_D.get(), block_D.size(), problem_sizes.get(), options.groups);
  print_device<<<1,1>>>(options.enable_print, block_ref_D.get(), block_ref_D.size(), options.groups, 'R');
  print_device<<<1,1>>>(options.enable_print, block_D.get(), block_D.size(), options.groups, 'D');

  return passed;
}

bool setup = false;

#ifdef PROFILE
#include "kernel_profiler.h"
#endif

#endif // defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.3 Toolkit to run this example
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 3)) {
    std::cerr << "This example requires CUDA 12.3 or newer.\n";
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  if (props.major != 9 || props.minor != 0) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Hopper Architecture (compute capability 90).\n";
    return 0;
  }

  int sm_count = props.multiProcessorCount;
  std::cout << "Device: " << props.name << ", SM count: " << sm_count << std::endl;

  Options options;
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.explore) {
    #ifdef PROFILE
    best_config_finder(options);
    #endif
  }
  else {
    #if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
      auto result = run<GemmScaleOnly>(options, false);
      if (result.status != cutlass::Status::kSuccess) {
        std::cerr << "Kernel failed: " << cutlassGetStatusString(result.status) << std::endl;
        return -1;
      }
    #endif
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
