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
std::vector<int64_t> offset_weight_scale_folded;
std::vector<int64_t> offset_activation_scale;
std::vector<int64_t> offset_epilogue_token_scale;
std::vector<int64_t> offset_zero;

std::vector<StrideA>     stride_A_host;
std::vector<StrideB>     stride_B_host;
std::vector<StrideC>     stride_C_host;
std::vector<StrideD>     stride_D_host;
std::vector<StrideC_ref> stride_C_host_ref;
std::vector<StrideD_ref> stride_D_host_ref;
std::vector<StrideS>     stride_weight_scale_host;
std::vector<StrideActivationScale> stride_activation_scale_host;

std::vector<ElementAccumulator> alpha_host;
std::vector<ElementAccumulator> beta_host;

uint64_t seed = 2020;

// Device-side allocations
cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

cutlass::DeviceAllocation<MmaType>                                               block_A;
cutlass::DeviceAllocation<QuantType>                                             block_B;
cutlass::DeviceAllocation<QuantType>                                             block_B_interleaved;
cutlass::DeviceAllocation<ElementScale>                                          block_weight_scale;
cutlass::DeviceAllocation<ElementWeightScaleStorage>                             block_weight_scale_folded;
cutlass::DeviceAllocation<ElementActivationScale>                                block_activation_scale;
cutlass::DeviceAllocation<ElementEpilogueTokenScale>                             block_epilogue_token_scale;
cutlass::DeviceAllocation<ElementZero>                                           block_zero;
cutlass::DeviceAllocation<ElementC>                                              block_C;
cutlass::DeviceAllocation<typename DefaultGemm::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename DefaultGemm::EpilogueOutputOp::ElementOutput> block_ref_D;
cutlass::DeviceAllocation<float>                                                 block_ref_abs_error_bound;

cutlass::DeviceAllocation<const MmaType *>                                         ptr_A;
cutlass::DeviceAllocation<const QuantType *>                                       ptr_B;
cutlass::DeviceAllocation<const ElementActivationScale *>                          ptr_activation_scale;
cutlass::DeviceAllocation<const ElementEpilogueTokenScale *>                       ptr_epilogue_token_scale;
cutlass::DeviceAllocation<const MainloopWeightScale *>                             ptr_weight_scale_folded;
cutlass::DeviceAllocation<const ElementZero *>                                     ptr_zero;
cutlass::DeviceAllocation<const ElementC *>                                        ptr_C;
cutlass::DeviceAllocation<typename DefaultGemm::EpilogueOutputOp::ElementOutput *> ptr_D;

cutlass::DeviceAllocation<StrideA>     stride_A;
cutlass::DeviceAllocation<StrideB>     stride_B;
cutlass::DeviceAllocation<StrideC>     stride_C;
cutlass::DeviceAllocation<StrideD>     stride_D;
cutlass::DeviceAllocation<StrideC_ref> stride_C_ref;
cutlass::DeviceAllocation<StrideD_ref> stride_D_ref;
cutlass::DeviceAllocation<StrideS>     stride_weight_scale;
cutlass::DeviceAllocation<StrideActivationScale> stride_activation_scale;

cutlass::DeviceAllocation<ElementAccumulator*> alpha_device;
cutlass::DeviceAllocation<ElementAccumulator*> beta_device;
cutlass::DeviceAllocation<ElementAccumulator>  block_alpha;
cutlass::DeviceAllocation<ElementAccumulator>  block_beta;

#if defined(CUTLASS_MIXED_GEMM_FUSED_E8M0_PRE_MMA_SCALE)
static bool pack_fused_e8m0_offset_scale(
    ElementScale *logical_out,
    ElementWeightScaleStorage *folded_out,
    size_t block_size,
    Options const& options,
    bool debug_input_scale,
    float *residual_scale_out) {
  static_assert(cute::is_same_v<ElementScale, cutlass::float_ue8m0_t>,
      "Fused e8m0 pre-MMA scale expects raw ue8m0 offset bytes.");

  std::vector<ElementScale> logical_data(block_size);
  std::vector<ElementWeightScaleStorage> folded_data(block_size);
  float residual_scale = 1.0f;

  // Host preprocessing only: fold each logical 64x128 scale tile into a 16x512
  // physical tile so the kernel can bulk-copy 16B scale rows for any Ktile.
  auto folded_scale_index = [](int scale_row, int k_group, int K) {
    return
        ((scale_row / 64) * (K / 128) + (k_group / 4)) * 256 + // folded 16x16 block
        (scale_row % 16) * 16 +                                // row inside the 16-row warp slice
        ((scale_row % 64) / 16) * 4 +                          // which 16-row warp slice
        (k_group % 4);                                         // scale group inside K128
  };

  if (!debug_input_scale) {
    uint8_t scale_min = 0xff;
    uint8_t scale_max = 0;
    size_t logical_idx = 0;
    for (int32_t group = 0; group < options.groups; ++group) {
      auto problem = options.problem_sizes_host.at(group);
      int const N = get<1>(problem);
      int const K = get<2>(problem);
      int const scale_groups = K / GROUP_SIZE;
      for (int kg = 0; kg < scale_groups; ++kg) {
        for (int n = 0; n < N; ++n, ++logical_idx) {
          uint8_t const raw_scale = static_cast<uint8_t>(114 + (logical_idx % (131 - 114)));
          scale_min = std::min(scale_min, raw_scale);
          scale_max = std::max(scale_max, raw_scale);
        }
      }
    }

    uint8_t const scale_range = std::min<uint8_t>(
        static_cast<uint8_t>(scale_max - scale_min), uint8_t(11));
    uint8_t const scale_min_new = static_cast<uint8_t>(scale_max - scale_range);
    int residual_exp = int(scale_min_new) - 127;
    residual_scale = 0.5f;
    while (residual_exp > 0) {
      residual_scale *= 2.0f;
      --residual_exp;
    }
    while (residual_exp < 0) {
      residual_scale *= 0.5f;
      ++residual_exp;
    }

    logical_idx = 0;
    for (int32_t group = 0; group < options.groups; ++group) {
      auto problem = options.problem_sizes_host.at(group);
      int const N = get<1>(problem);
      int const K = get<2>(problem);
      int const scale_groups = K / GROUP_SIZE;
      int64_t const group_base = offset_weight_scale_folded.at(group);
      for (int kg = 0; kg < scale_groups; ++kg) {
        for (int n = 0; n < N; ++n, ++logical_idx) {
          uint8_t const raw_scale = static_cast<uint8_t>(114 + (logical_idx % (131 - 114)));
          uint8_t const clamped_scale = std::max(raw_scale, scale_min_new);
          uint8_t const raw_offset = static_cast<uint8_t>(clamped_scale - scale_min_new + 1);
          int const folded_idx = folded_scale_index(n, kg, K);
          logical_data[group_base + kg * N + n] = ElementScale::bitcast(raw_offset);
          folded_data[group_base + folded_idx] = ElementScale::bitcast(raw_offset);
        }
      }
    }
  }
  else {
    for (size_t i = 0; i < block_size; ++i) {
      logical_data[i] = ElementScale::bitcast(uint8_t(1));
      folded_data[i] = ElementScale::bitcast(uint8_t(1));
    }
  }

  try {
    cutlass::device_memory::copy_to_device(logical_out, logical_data.data(), block_size);
    cutlass::device_memory::copy_to_device(folded_out, folded_data.data(), block_size);
  }
  catch (cutlass::cuda_exception const& e) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(e.cudaError()) << std::endl;
    return false;
  }
  *residual_scale_out = residual_scale;
  return true;
}
#endif

template <class ProblemSizes, class ElementScaleRaw, class ElementScaleStorage>
__global__ void fold_weight_scale_ktile_independent_kernel(
    ProblemSizes problem_sizes,
    int group_num,
    ElementScaleRaw const* logical_scale,
    ElementScaleStorage* folded_scale,
    int group_size) {

  static_assert(cute::is_same_v<ElementScaleRaw, ElementScaleStorage>,
      "Folded weight scale storage is scalar raw scale storage.");

  int const group = blockIdx.x;
  if (group >= group_num) {
    return;
  }

  int64_t group_base = 0;
  for (int g = 0; g < group; ++g) {
    int const prev_n = get<0>(problem_sizes[g]);
    int const prev_k = get<2>(problem_sizes[g]);
    group_base += int64_t(prev_n) * int64_t(prev_k / group_size);
  }

  int const N = get<0>(problem_sizes[group]);
  int const K = get<2>(problem_sizes[group]);
  int const scale_groups = K / group_size;
  int const scale_groups_per_k128 = 128 / group_size;
  int const physical_cols = 16 / int(sizeof(ElementScaleRaw));
  int const m_slices_per_m64 = physical_cols / scale_groups_per_k128;
  int const folded_m = 64 / m_slices_per_m64;
  int const k128_blocks = K / 128;
  int const fold_block_elems = 64 * scale_groups_per_k128;

  for (int idx = threadIdx.x; idx < N * scale_groups; idx += blockDim.x) {
    int const kg = idx / N;
    int const n = idx % N;
    int const k128 = kg / scale_groups_per_k128;
    int const kg_in_k128 = kg % scale_groups_per_k128;
    int const folded_idx =
        ((n / 64) * k128_blocks + k128) * fold_block_elems +
        (n % folded_m) * physical_cols +
        ((n % 64) / folded_m) * scale_groups_per_k128 +
        kg_in_k128;

    folded_scale[group_base + folded_idx] = logical_scale[group_base + idx];
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed functions
/////////////////////////////////////////////////////////////////////////////////////////////////

void allocate(Options const& options) {
  int64_t total_elements_A = 0;
  int64_t total_elements_B = 0;
  int64_t total_elements_C = 0;
  int64_t total_elements_D = 0;
  int64_t total_elements_weight_scale = 0;
  int64_t total_elements_weight_scale_folded = 0;
  int64_t total_elements_activation_scale = 0;
  int64_t total_elements_epilogue_token_scale = 0;
  int64_t total_elements_zero = 0;

  for (int32_t i = 0; i < options.groups; ++i) {

    auto problem = options.problem_sizes_host.at(i);
    auto M = get<0>(problem);
    auto N = get<1>(problem);
    auto K = get<2>(problem);

    const int scale_k = K / TileShapeK;
    const int scale_groups = K / GROUP_SIZE;
    const int weight_scale_storage_k = scale_groups;

    offset_A.push_back(total_elements_A);
    offset_B.push_back(total_elements_B * cutlass::sizeof_bits<QuantType>::value / 8);
    offset_C.push_back(total_elements_C);
    offset_D.push_back(total_elements_D);
    offset_weight_scale_folded.push_back(total_elements_weight_scale_folded);
    offset_activation_scale.push_back(total_elements_activation_scale);
    offset_epilogue_token_scale.push_back(total_elements_epilogue_token_scale);
    offset_zero.push_back(total_elements_zero);

    int64_t elements_A = M * K;
    int64_t elements_B = K * N;
    int64_t elements_C = M * N;
    int64_t elements_D = M * N;
    int64_t elements_weight_scale = int64_t(weight_scale_storage_k) * N;
    int64_t elements_activation_scale =
        ScaleAppliesToActivation ? int64_t(M) * scale_groups : 0;
    int64_t elements_epilogue_token_scale =
#if defined(CUTLASS_MIXED_GEMM_EPILOGUE_TOKEN_SCALE)
        M;
#else
        0;
#endif
    int64_t elements_zero = scale_k * N;

    total_elements_A                       += elements_A;
    total_elements_B                       += elements_B;
    total_elements_C                       += elements_C;
    total_elements_D                       += elements_D;
    total_elements_weight_scale            += elements_weight_scale;
    total_elements_weight_scale_folded     += elements_weight_scale;
    total_elements_activation_scale        += elements_activation_scale;
    total_elements_epilogue_token_scale    += elements_epilogue_token_scale;
    total_elements_zero                    += elements_zero;

    stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
    stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
    stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {N, M, 1}));
    stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {N, M, 1}));
    stride_C_host_ref.push_back(cutlass::make_cute_packed_stride(StrideC_ref{}, {M, N, 1}));
    stride_D_host_ref.push_back(cutlass::make_cute_packed_stride(StrideD_ref{}, {M, N, 1}));
    stride_weight_scale_host.push_back(cutlass::make_cute_packed_stride(
        StrideS{}, {N, weight_scale_storage_k, 1}));
    stride_activation_scale_host.push_back(cutlass::make_cute_packed_stride(
        StrideActivationScale{}, {M, scale_groups, 1}));
  }

  block_A.reset(total_elements_A);
  block_B.reset(total_elements_B);
  block_B_interleaved.reset(total_elements_B);
  block_C.reset(total_elements_C);
  block_D.reset(total_elements_D);
  block_ref_D.reset(total_elements_D);
  block_ref_abs_error_bound.reset(total_elements_D);
  block_weight_scale.reset(total_elements_weight_scale);
  block_weight_scale_folded.reset(total_elements_weight_scale_folded);
  block_activation_scale.reset(total_elements_activation_scale);
  block_epilogue_token_scale.reset(total_elements_epilogue_token_scale);
  block_zero.reset(total_elements_zero);

  block_alpha.reset(options.groups);
  block_beta.reset(options.groups);
}

void initialize(Options& options) {

  [[maybe_unused]]uint64_t seed = 2020;

  problem_sizes.reset(options.groups);
  // The CMX mixed mainloop uses swapped problem shapes on device: (N, M, K).
  // Keep options.problem_sizes_host in original (M, N, K) form for host setup.
  for (int32_t i = 0; i < options.groups; ++i) {
    auto [M, N, K] = options.problem_sizes_host[i];
    options.problem_sizes_host[i] = make_tuple(N, M, K);
  }
  problem_sizes.copy_from_host(options.problem_sizes_host.data());
  for (int32_t i = 0; i < options.groups; ++i) {
    auto [N, M, K] = options.problem_sizes_host[i];
    options.problem_sizes_host[i] = make_tuple(M, N, K);
  }

  std::vector<MmaType *>                         ptr_A_host(options.groups);
  std::vector<QuantType *>                       ptr_B_host(options.groups);
  std::vector<ElementC *>                        ptr_C_host(options.groups);
  std::vector<ElementC *>                        ptr_D_host(options.groups);
  std::vector<MainloopWeightScale *>             ptr_weight_scale_folded_host(options.groups);
  std::vector<ElementActivationScale const *>    ptr_activation_scale_host(options.groups);
  std::vector<ElementEpilogueTokenScale const *> ptr_epilogue_token_scale_host(options.groups);
  std::vector<ElementZero *>                     ptr_zero_host(options.groups);
  std::vector<ElementAccumulator *>              ptr_alpha_host(options.groups);
  std::vector<ElementAccumulator *>              ptr_beta_host(options.groups);

  for (int32_t i = 0; i < options.groups; ++i) {
    ptr_A_host.at(i)                       = block_A.get() + offset_A.at(i);
    ptr_B_host.at(i)                       = block_B_interleaved.get() + offset_B.at(i);
    ptr_C_host.at(i)                       = block_C.get() + offset_C.at(i);
    ptr_D_host.at(i)                       = block_D.get() + offset_D.at(i);
    ptr_weight_scale_folded_host.at(i)     =
        reinterpret_cast<MainloopWeightScale*>(
            block_weight_scale_folded.get() + offset_weight_scale_folded.at(i));
    if constexpr (ScaleAppliesToActivation) {
      ptr_activation_scale_host.at(i)        =
          block_activation_scale.get() + offset_activation_scale.at(i);
    }
#if defined(CUTLASS_MIXED_GEMM_EPILOGUE_TOKEN_SCALE)
    ptr_epilogue_token_scale_host.at(i) =
        block_epilogue_token_scale.get() + offset_epilogue_token_scale.at(i);
#endif
    ptr_zero_host.at(i)                    = block_zero.get() + offset_zero.at(i);
    alpha_host.push_back((options.alpha == FLT_MAX) ? static_cast<ElementAccumulator>((rand() % 5) + 1) : options.alpha);
    beta_host.push_back( (options.beta  == FLT_MAX) ? static_cast<ElementAccumulator>(rand() % 5)       : options.beta);
    ptr_alpha_host.at(i)                   = block_alpha.get() + i;
    ptr_beta_host.at(i)                    = block_beta.get() + i;
  }

  ptr_A.reset(options.groups);                       ptr_A.copy_from_host(ptr_A_host.data());
  ptr_B.reset(options.groups);                       ptr_B.copy_from_host(ptr_B_host.data());
  ptr_C.reset(options.groups);                       ptr_C.copy_from_host(ptr_C_host.data());
  ptr_D.reset(options.groups);                       ptr_D.copy_from_host(ptr_D_host.data());
  ptr_activation_scale.reset(options.groups);        ptr_activation_scale.copy_from_host(ptr_activation_scale_host.data());
  ptr_epilogue_token_scale.reset(options.groups);    ptr_epilogue_token_scale.copy_from_host(ptr_epilogue_token_scale_host.data());
  ptr_weight_scale_folded.reset(options.groups);     ptr_weight_scale_folded.copy_from_host(ptr_weight_scale_folded_host.data());
  ptr_zero.reset(options.groups);                    ptr_zero.copy_from_host(ptr_zero_host.data());

  stride_A.reset(options.groups);                    stride_A.copy_from_host(stride_A_host.data());
  stride_B.reset(options.groups);                    stride_B.copy_from_host(stride_B_host.data());
  stride_C.reset(options.groups);                    stride_C.copy_from_host(stride_C_host.data());
  stride_D.reset(options.groups);                    stride_D.copy_from_host(stride_D_host.data());
  stride_C_ref.reset(options.groups);                stride_C_ref.copy_from_host(stride_C_host_ref.data());
  stride_D_ref.reset(options.groups);                stride_D_ref.copy_from_host(stride_D_host_ref.data());
  stride_weight_scale.reset(options.groups);         stride_weight_scale.copy_from_host(stride_weight_scale_host.data());
  stride_activation_scale.reset(options.groups);     stride_activation_scale.copy_from_host(stride_activation_scale_host.data());

  alpha_device.reset(options.groups);                alpha_device.copy_from_host(ptr_alpha_host.data());
  beta_device.reset(options.groups);                 beta_device.copy_from_host(ptr_beta_host.data());

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
  else if constexpr (cute::is_same_v<MmaType, cutlass::float_e4m3_t> &&
    (cute::is_same_v<QuantType, cutlass::int4b_t> ||
     cute::is_same_v<QuantType, cutlass::float_e2m1_t>))
  {
    // Both 4-bit weight formats use the W4A8 Hopper offline layout.
    for (int32_t i = 0; i < options.groups; ++i) {
      auto problem = options.problem_sizes_host.at(i);
      auto N = get<1>(problem);
      auto K = get<2>(problem);
#if defined(CUTLASS_MIXED_GEMM_FP4_FP8_PREPROCESSED_SIGNS)
      static constexpr bool PreprocessFp4SignsForFp8 =
        cute::is_same_v<QuantType, cutlass::float_e2m1_t>;
#else
      static constexpr bool PreprocessFp4SignsForFp8 = false;
#endif
      interleave_w4a8_Hopper<QuantType, PreprocessFp4SignsForFp8>(
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
    set_device_ue8m0<<<1, 1>>>(options.debug_input_scale, block_weight_scale.get(), block_weight_scale.size());
  }
  else {
    set_device_sequential<<<1, 1>>>(block_weight_scale.get(), block_weight_scale.size(), 5678);
  }
  print_device<<<1, 1>>>(options.enable_print, block_weight_scale.get(), block_weight_scale.size(), options.groups, 'S');
  cudaDeviceSynchronize();

#if defined(CUTLASS_MIXED_GEMM_FUSED_E8M0_PRE_MMA_SCALE)
  float fused_e8m0_residual_scale = 1.0f;
  if (!pack_fused_e8m0_offset_scale(
      block_weight_scale.get(),
      block_weight_scale_folded.get(),
      block_weight_scale_folded.size(),
      options,
      options.debug_input_scale,
      &fused_e8m0_residual_scale)) {
    return;
  }
#else
  fold_weight_scale_ktile_independent_kernel<<<options.groups, 256>>>(
      problem_sizes.get(),
      options.groups,
      block_weight_scale.get(),
      block_weight_scale_folded.get(),
      GROUP_SIZE);
  cudaDeviceSynchronize();
#endif
  print_device<<<1, 1>>>(
      options.enable_print, block_weight_scale_folded.get(), block_weight_scale_folded.size(), options.groups, 'W');

  if constexpr (ScaleAppliesToActivation) {
    set_device_ue8m0<<<1, 1>>>(
        options.debug_input_scale, block_activation_scale.get(), block_activation_scale.size());
    print_device<<<1, 1>>>(
        options.enable_print, block_activation_scale.get(), block_activation_scale.size(), options.groups, 's');
    cudaDeviceSynchronize();
  }
#if defined(CUTLASS_MIXED_GEMM_EPILOGUE_TOKEN_SCALE)
  set_device_sequential<<<1, 1>>>(
      block_epilogue_token_scale.get(), block_epilogue_token_scale.size(), 6789);
#if defined(CUTLASS_MIXED_GEMM_FUSED_E8M0_PRE_MMA_SCALE)
  scale_device<<<1, 1>>>(
      block_epilogue_token_scale.get(),
      block_epilogue_token_scale.size(),
      fused_e8m0_residual_scale);
#endif
  cudaDeviceSynchronize();
#endif
  /////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_MIXED_GEMM_FUSED_E8M0_PRE_MMA_SCALE)
#if defined(CUTLASS_MIXED_GEMM_EPILOGUE_TOKEN_SCALE)
  groupwise_verify_fused_e8m0_pre_mma<true>(
#else
  groupwise_verify_fused_e8m0_pre_mma<false>(
#endif
    problem_sizes.get(),
    options.groups,
    block_A.get(), block_B.get(),
    block_weight_scale.get(), block_epilogue_token_scale.get(), block_ref_D.get(),
    GROUP_SIZE,
    stride_A.get(), stride_B.get()
  );
#else
  groupwise_verify(
    problem_sizes.get(),
    options.groups,
    block_A.get(), block_B.get(),
    block_weight_scale.get(), block_activation_scale.get(), block_ref_D.get(),
    block_ref_abs_error_bound.get(),
    TileShapeK,
    GROUP_SIZE,
    stride_A.get(), stride_B.get()
  );
#endif

  print_device<<<1,1>>>(options.enable_print, block_ref_D.get(), block_ref_D.size(), options.groups, 'R');

  initialize_zero(block_zero, options);
  block_alpha.copy_from_host(alpha_host.data());
  block_beta.copy_from_host(beta_host.data());
}

bool verify(Options const& options) {
  bool passed = true;

  cutlass::DeviceAllocation<int> error_counts;
  error_counts.reset(kMixedGemmValidationCounterCount);
  CUDA_CHECK(cudaMemset(
      error_counts.get(), 0, sizeof(int) * kMixedGemmValidationCounterCount));

  // Standard mixed low-precision paths validate against a K/input-dependent
  // absolute reduction bound. Fused pre-MMA scaling uses a separate reference
  // path and keeps its stricter gate below.
  float const *reduction_abs_error_bound = nullptr;
#if !defined(CUTLASS_MIXED_GEMM_FUSED_E8M0_PRE_MMA_SCALE)
  reduction_abs_error_bound = block_ref_abs_error_bound.get();
#endif

#if defined(CUTLASS_MIXED_GEMM_FUSED_E8M0_PRE_MMA_SCALE)
  compare_device<true><<<1,1>>>(
#else
  compare_device<<<1,1>>>(
#endif
      options.compare, block_D.get(), block_ref_D.get(), block_D.size(),
      problem_sizes.get(), options.groups, error_counts.get(), reduction_abs_error_bound);

  int error_counts_host[kMixedGemmValidationCounterCount] = {};
  error_counts.copy_to_host(error_counts_host);
#if defined(CUTLASS_MIXED_GEMM_FUSED_E8M0_PRE_MMA_SCALE)
  passed &= (error_counts_host[kP99ErrorCount] == 0);
  passed &= (error_counts_host[kP98ErrorCount] == 0);
  passed &= (error_counts_host[kP95ErrorCount] == 0);
  passed &= (error_counts_host[kHummingToleranceErrorCount] == 0);
#else
  passed &= (error_counts_host[kReductionBoundErrorCount] == 0);
#endif
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
    if (!best_config_finder(options)) {
      return -1;
    }
    #endif
  }
  else {
    #if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
      auto result = run<GemmScaleOnly>(options);
      if (result.status != cutlass::Status::kSuccess) {
        std::cerr << "Kernel failed: " << cutlassGetStatusString(result.status) << std::endl;
        return -1;
      }
      if (!result.passed) {
        std::cerr << "Kernel failed correctness verification." << std::endl;
        return -1;
      }
    #endif
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
