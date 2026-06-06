
#pragma once

#include <curand_kernel.h>

using namespace cute;


template <typename T>
__global__ void print_device(bool enable_print, T *ptr, int count, int Groups, const char str = ' ') {
if (enable_print) {
  if (thread0()) {
    printf("print_device %c ", str);
    
    for (int g = 0; g < Groups; g++) {
      for (int i = 0; i < count / Groups; i++)
      {
        int idx = g * count / Groups + i;
        printf("(g-%d  %d):%f ", g, i, float(ptr[idx]));
      }
      printf("\n");
    }
  }
}
}

template <typename T>
__global__ void print_device_packed(bool enable_print, T *ptr, int count, const char str = ' ') {
if (enable_print) {
  if (thread0()) {
    printf("print_device %c ", str);
    for (int i = 0; i < count; i++)
    {
      printf("(%d):%f", i, float(ptr[i][0]));
    }
    printf("\n");
  }
}
}


template <typename T>
__global__ void print_device_4b(bool enable_print, T *ptr_, int rows, int cols, const char str = ' ') {
if (enable_print) {

  float lut[16];

  if constexpr (cute::is_same_v<T, cutlass::float_e2m1_t>) {
    // fp4 e2m1
    float tmp[] = {0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0, 
                   0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};
    memcpy(lut, tmp, sizeof(lut));
  }
  else {
    // int4
    float tmp[] = { 0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
                   -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0};
    memcpy(lut, tmp, sizeof(lut));
  }

  if (thread0()) {
    printf("print_device(int4) %c ", str);

    for (int i = 0; i < rows; i++)
    {
      printf("(%d):  ", i);
      for (int j = 0; j < cols / 2; j++)
      {
        uint8_t *ptr = reinterpret_cast<uint8_t *>(&ptr_[i * cols / 2 + j]);
        uint8_t low = *ptr & 0x0F;
        uint8_t high = (*ptr & 0xF0) >> 4;
    
        // printf("(%d):%d ", 2 * i, int(low));
        // printf("(%d):%d ", 2 * i + 1, int(high));
        printf("(%d):%f ", 2 * j, lut[low]);
        printf("(%d):%f ", 2 * j + 1, lut[high]);
      }
      printf("\n");
    }
  }
}
}


template<typename T>
__global__ void set_device(bool debug_input_act, T *ptr, int count, int value = 0) {
if (debug_input_act) {
  if (thread0())
    for (int i = 0; i < count; i++)
    {
      if (value == 0)
        ptr[i] = static_cast<T>(i / (16*128) + 1);
      else
        ptr[i] = static_cast<T>(value);
    }
}
}


template<typename T>
__global__ void set_device_sequential(T *ptr, int count, int seed, int value = 0) {
  if (thread0()) {
    // Initialize curand state
    curandState state;
    curand_init(seed, 0, 0, &state);
    
    for (int i = 0; i < count; i++)
    {
      if (value == 0) {
        float rand_val = curand_uniform(&state) * 2.0f;
        ptr[i] = static_cast<T>(rand_val);
      } 
      else {
        ptr[i] = static_cast<T>(value);
      }
    }
  }
}

template<typename T>
__global__ void scale_device(T *ptr, int count, float scale) {
  if (thread0()) {
    for (int i = 0; i < count; ++i) {
      ptr[i] = static_cast<T>(static_cast<float>(ptr[i]) * scale);
    }
  }
}

__global__ void set_device_ue8m0(bool debug_input_scale, void *ptr_, int count, int default_val = 1) {

  cutlass::float_ue8m0_t *ptr = reinterpret_cast<cutlass::float_ue8m0_t *>(ptr_);

  if (thread0())
    for (int i = 0; i < count; i++)
    {
      if (!debug_input_scale)
      {
        // 114 -> 0.000122
        // 130 -> 8.000000
        uint8_t value = 114 + (i % (131 - 114));
        ptr[i] = *reinterpret_cast<cutlass::float_ue8m0_t *>(&value);
  
        // printf("set_device_ue8m0 %d %f\n", int(value), static_cast<float>(ptr[i]));
      }
      else
      {
        ptr[i] = static_cast<cutlass::float_ue8m0_t>(default_val);
      }
    }
}


template<typename T>
__global__ void set_device_int4(bool debug_input_weight, T *ptr_, int count, uint8_t value = 0) {
if (debug_input_weight) {
  // N x K
  // 16 x 256
  if (thread0())
  {
    for (int i = 0; i < count / 2; i++)
    {
      uint8_t low = 0;

      if (value == 0)
        low = (i / (64 * 128)) % 15 + 1; // 1~15
      else
        low = value; // 0~14

      uint8_t high = low; // 1~15
      // uint8_t high = low + 1; // 1~15
      uint8_t *ptr = reinterpret_cast<uint8_t *>(ptr_);
      ptr[i] = (high << 4) | low;
    }

    // // fp4:    1 2 3 4
    // // bits:   0010 0100 0101 0110
    // // 16-bit: 0x2456U
    // curandState state;
    // curand_init(0, 0, 0, &state);
    
    // for (int i = 0; i < 16 / 4; i++)
    // {
    //   uint16_t value = static_cast<uint16_t>(curand(&state));
    //   uint16_t *ptr = reinterpret_cast<uint16_t *>(ptr_);
    //   ptr[i] = value;
    // }

    // uint32_t *ptr = reinterpret_cast<uint32_t *>(ptr_);
    // // mma0
    // ptr[0] = 0x76543210U;   // 0-7
    // ptr[1] = 0xFEDCBA98U;   // 8-15
    // ptr[128] = 0xFEDCBA98U; // 1024-1031
    // ptr[129] = 0x76543210U; // 1032-1040
    
    // // mma1
    // ptr[2] = 0x76543210U;   // 16-23
    // ptr[3] = 0xFEDCBA98U;   // 24-31
    // ptr[130] = 0xFEDCBA98U; // 1040-1047
    // ptr[131] = 0x76543210U; // 1048-1055
    
    // // mma4
    // ptr[8] = 0x76543210U;   // 64-71
    // ptr[9] = 0xFEDCBA98U;
    // ptr[136] = 0xFEDCBA98U;
    // ptr[137] = 0x76543210U;
  }
}
}


// Walks the packed D buffer per-group using each group's actual (M, N) from
// `problem_sizes`. This is mandatory for variable-M (e.g. MoE) tests — using a
// single uniform M from `options.m` will overrun `block_D` and corrupt the CUDA
// context (subsequent kernel launches then fail with "Error Internal").
//
// NOTE: at the point this kernel is launched (from verify()), the device-side
// `problem_sizes` array has already been transposed for SwapAB by initialize(),
// so each entry is laid out as (N, M, K) — original M lives at index 1.
enum MixedGemmValidationCounter : int {
  kP99ErrorCount = 0,
  kP98ErrorCount = 1,
  kP95ErrorCount = 2,
  kHummingToleranceErrorCount = 3,
  kReductionBoundErrorCount = 4,
  // Sentinel used as the device counter array length; not a real counter slot.
  kMixedGemmValidationCounterCount = 5
};

__device__ float bf16_ulp_from_abs(float ref_abs) {
  if (!(ref_abs > 0.0f)) {
    return 0x1p-133f;
  }

  uint32_t const bits = __float_as_uint(ref_abs);
  int const exponent = int((bits >> 23) & 0xff);
  if (exponent <= 7) {
    return 0x1p-133f;
  }

  return __uint_as_float(uint32_t(exponent - 7) << 23);
}

template <typename T>
__device__ float output_round_budget(float ref_abs) {
  if constexpr (cute::is_same_v<T, cutlass::bfloat16_t>) {
    return 2.0f * bf16_ulp_from_abs(ref_abs);
  }
  else {
    return 0.0f;
  }
}

template <bool UseHummingTolerance = false, typename T, typename ProblemSizes>
__global__ void compare_device(
    bool compare_print, T *out, T *ref, int count,
    ProblemSizes problem_sizes_swapped, int Groups,
    int *error_counts = nullptr,
    float const *abs_error_bound = nullptr)
{
    if (!thread0()) return;

    if (count == 0) {
      printf("P99_error_count 0 0.00%%\n");
      printf("P98_error_count 0 0.00%%\n");
      printf("P95_error_count 0 0.00%%\n");
      return;
    }

    int P99_error_count = 0;
    int P98_error_count = 0;
    int P95_error_count = 0;
    int humming_tolerance_error_count = 0;
    int reduction_bound_error_count = 0;
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;

    int64_t base = 0;
    for (int g = 0; g < Groups; g++) {
      // problem_sizes_swapped[g] = (N_orig, M_orig, K)
      int N = get<0>(problem_sizes_swapped[g]);
      int M = get<1>(problem_sizes_swapped[g]);
      int K = get<2>(problem_sizes_swapped[g]);
      float const fp32_unit_roundoff = 0x1p-24f;
      float const k_roundoff = float(K) * fp32_unit_roundoff;
      float const gamma_k = k_roundoff / max(1.0f - k_roundoff, fp32_unit_roundoff);
      for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
          int64_t idx = base + int64_t(m) * N + n;
          float ref_abs = abs(float(ref[idx]));
          float abs_error = abs(float(out[idx]) - float(ref[idx]));
          float rel_error = abs_error / max(ref_abs, 1.0e-20f);
          max_abs_error = max(max_abs_error, abs_error);
          max_rel_error = max(max_rel_error, rel_error);

          if constexpr (UseHummingTolerance) {
            if (abs_error > (0.5f + 0.05f * ref_abs)) {
              humming_tolerance_error_count++;
            }
          }
          if (abs_error_bound != nullptr) {
            float const reduction_budget = 2.0f * gamma_k * abs_error_bound[idx];
            float const bf16_output_budget = output_round_budget<T>(ref_abs);
            float const abs_limit = reduction_budget + bf16_output_budget;
            if (abs_error > abs_limit) {
              reduction_bound_error_count++;
            }
          }

          bool p99_error = false;
          bool p98_error = false;
          bool p95_error = false;
          if constexpr (UseHummingTolerance) {
            // Direct/fused pre-MMA paths accumulate a larger dynamic range
            // before bf16 output rounding. Use combined tolerance so near-zero
            // references do not dominate the relative-error count.
            float p99_abs_limit = max(0.75f, 0.03f * ref_abs);
            float p98_abs_limit = max(1.50f, 0.06f * ref_abs);
            float p95_abs_limit = max(3.00f, 0.12f * ref_abs);
            p99_error = abs_error > p99_abs_limit;
            p98_error = abs_error > p98_abs_limit;
            p95_error = abs_error > p95_abs_limit;
          }
          else {
            p99_error = rel_error > 0.01f;
            p98_error = rel_error > 0.02f;
            p95_error = rel_error > 0.05f;
          }

          if (p99_error) {
            P99_error_count++;
            if (compare_print)
              printf("(g=%d, m=%d, n=%d): out %f ref %f abs_error %f rel_error %f\n",
                  g, m, n, float(out[idx]), float(ref[idx]), abs_error, rel_error);
            if (p98_error) P98_error_count++;
            if (p95_error) P95_error_count++;
          }
        }
      }
      base += int64_t(M) * N;
    }
    printf("P99_error_count %d %.2f%%\n", P99_error_count, float(P99_error_count) / count * 100.0f);
    printf("P98_error_count %d %.2f%%\n", P98_error_count, float(P98_error_count) / count * 100.0f);
    printf("P95_error_count %d %.2f%%\n", P95_error_count, float(P95_error_count) / count * 100.0f);
    if constexpr (UseHummingTolerance) {
      printf("Humming_tol_error_count %d %.2f%% max_abs_error %f max_rel_error %f\n",
          humming_tolerance_error_count, float(humming_tolerance_error_count) / count * 100.0f,
          max_abs_error, max_rel_error);
    }
    if (abs_error_bound != nullptr) {
      printf("Reduction_bound_error_count %d %.2f%%\n",
          reduction_bound_error_count, float(reduction_bound_error_count) / count * 100.0f);
    }
    if (error_counts != nullptr) {
      error_counts[kP99ErrorCount] = P99_error_count;
      error_counts[kP98ErrorCount] = P98_error_count;
      error_counts[kP95ErrorCount] = P95_error_count;
      if constexpr (UseHummingTolerance) {
        error_counts[kHummingToleranceErrorCount] = humming_tolerance_error_count;
      }
      error_counts[kReductionBoundErrorCount] = reduction_bound_error_count;
    }
}


template <
    typename ElementA,
    typename ElementB,
    typename ElementWeightScalePacked,
    typename ElementActivationScaleRaw,
    typename ElementD
>
__device__ void single_gemm_varify(
  int bid, int tid,
  int block_tile_k, int group_size,
  int M, int N, int K,
  ElementA *A_ptr,
  ElementB *B_ptr,
  ElementWeightScalePacked *weight_scale_ptr,
  ElementActivationScaleRaw *activation_scale_ptr,
  ElementD *D_ptr,
  float *abs_error_bound_ptr = nullptr) {

  float lut[16];

  if constexpr (cute::is_same_v<ElementB, cutlass::float_e2m1_t>) {
    // fp4 e2m1
    float tmp[] = {0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0, 
                   0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};
    memcpy(lut, tmp, sizeof(lut));
  }
  else {
    // int4
    float tmp[] = { 0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
                   -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0};
    memcpy(lut, tmp, sizeof(lut));
  }


  for (int m = bid; m < M; m += gridDim.x) {
    for (int n = tid; n < N; n += blockDim.x) {

        float accum = 0.0f;
        float abs_error_bound = 0.0f;

        for (int k_group = 0; k_group < K; k_group += group_size) {
            float group_accum = 0.0f;
            float group_abs_bound = 0.0f;

            for (int k = k_group; k < k_group + group_size; k += 2) {
                ElementA *local_A_ptr = A_ptr + m * K + k;
                uint8_t *local_B_ptr = reinterpret_cast<uint8_t *>(B_ptr) + n * K / 2 + k / 2;

                float elem_A_0 = local_A_ptr[0];
                float elem_A_1 = local_A_ptr[1];
                uint8_t elem_B_low_ = (*local_B_ptr) & 0xF;
                uint8_t elem_B_high_  = ((*local_B_ptr) & 0xF0) >> 4;
                // float elem_B_low = (elem_B_low_ < 8) ? elem_B_low_ : (float)elem_B_low_ - 16;
                // float elem_B_high = (elem_B_high_ < 8) ? elem_B_high_ : (float)elem_B_high_ - 16;
                float elem_B_low = lut[elem_B_low_];
                float elem_B_high = lut[elem_B_high_];

                group_accum += elem_A_0 * elem_B_low + elem_A_1 * elem_B_high;
                group_abs_bound +=
                    abs(elem_A_0) * abs(elem_B_low) + abs(elem_A_1) * abs(elem_B_high);
            }

            ElementWeightScalePacked *local_weight_scale_ptr =
                weight_scale_ptr + (k_group / block_tile_k) * N + n;
            int scale_idx = (k_group % block_tile_k) / group_size;
            float scale = static_cast<float>((*local_weight_scale_ptr)[scale_idx]);

            if constexpr (ScaleAppliesToActivation) {
              ElementActivationScaleRaw *local_activation_scale_ptr =
                  activation_scale_ptr + m * (K / group_size) + k_group / group_size;
              scale *= static_cast<float>(*local_activation_scale_ptr);
            }

            accum += group_accum * scale;
            abs_error_bound += group_abs_bound * abs(scale);

            // if (group_id == 0 && bid == 0 && tid == 0)
            //     printf("A %f %f B %f %f scale %f accum %f\n",
            //         elem_A_0,
            //         elem_A_1,
            //         elem_B_low,
            //         elem_B_high,
            //         scale,
            //         accum
            //     );
        }

        ElementD *local_D_ptr = D_ptr + m * N + n;
        *local_D_ptr = static_cast<ElementD>(accum);
        if (abs_error_bound_ptr != nullptr) {
          abs_error_bound_ptr[m * N + n] = abs_error_bound;
        }
    }
  }
}

__device__ uint8_t fused_e8m0_fp4_to_e4m3_raw(uint8_t fp4_code, uint8_t exp_offset) {
  uint8_t const sign = (fp4_code & 0x8) ? 0x80 : 0x00;
  uint8_t const em_code = fp4_code & 0x7;

  uint8_t em = 0;
  if (em_code == 0) {
    em = 0;
  }
  else if (em_code == 1) {
    em = static_cast<uint8_t>(exp_offset * 8);
  }
  else if (em_code == 2) {
    em = static_cast<uint8_t>(exp_offset * 8 + 0x08);
  }
  else if (em_code == 3) {
    em = static_cast<uint8_t>(exp_offset * 8 + 0x0c);
  }
  else {
    em = static_cast<uint8_t>(exp_offset * 8 + 0x10 + (em_code - 4) * 4);
  }

  return sign | em;
}

__device__ float fused_e8m0_fp4_to_float(uint8_t fp4_code, uint8_t exp_offset) {
  cutlass::float_e4m3_t fp8 =
      cutlass::float_e4m3_t::bitcast(fused_e8m0_fp4_to_e4m3_raw(fp4_code, exp_offset));
  return cutlass::float_e4m3_t::to_float(fp8);
}

template <
    bool ApplyTokenScale,
    typename ElementA,
    typename ElementB,
    typename ElementWeightScalePacked,
    typename ElementTokenScale,
    typename ElementD
>
__device__ void single_gemm_verify_fused_e8m0_pre_mma(
  int bid, int tid,
  int block_tile_k, int group_size,
  int M, int N, int K,
  ElementA *A_ptr,
  ElementB *B_ptr,
  ElementWeightScalePacked *weight_scale_ptr,
  ElementTokenScale *token_scale_ptr,
  ElementD *D_ptr) {

  for (int m = bid; m < M; m += gridDim.x) {
    for (int n = tid; n < N; n += blockDim.x) {
      float accum = 0.0f;

      for (int k_group = 0; k_group < K; k_group += group_size) {
        ElementWeightScalePacked *local_weight_scale_ptr =
            weight_scale_ptr + (k_group / block_tile_k) * N + n;
        int const scale_idx = (k_group % block_tile_k) / group_size;
        using ScaleScalar = typename ElementWeightScalePacked::Element;
        ScaleScalar const scale = (*local_weight_scale_ptr)[scale_idx];
        uint8_t const exp_offset = scale.storage;

        for (int k = k_group; k < k_group + group_size; k += 2) {
          ElementA *local_A_ptr = A_ptr + m * K + k;
          uint8_t *local_B_ptr = reinterpret_cast<uint8_t *>(B_ptr) + n * K / 2 + k / 2;

          float const elem_A_0 = static_cast<float>(local_A_ptr[0]);
          float const elem_A_1 = static_cast<float>(local_A_ptr[1]);
          uint8_t const elem_B_low = (*local_B_ptr) & 0x0f;
          uint8_t const elem_B_high = ((*local_B_ptr) & 0xf0) >> 4;

          float const elem_B_0 = fused_e8m0_fp4_to_float(elem_B_low, exp_offset);
          float const elem_B_1 = fused_e8m0_fp4_to_float(elem_B_high, exp_offset);
          accum += elem_A_0 * elem_B_0 + elem_A_1 * elem_B_1;
        }
      }

      if constexpr (ApplyTokenScale) {
        accum *= static_cast<float>(token_scale_ptr[m]);
      }

      ElementD *local_D_ptr = D_ptr + m * N + n;
      *local_D_ptr = static_cast<ElementD>(accum);
    }
  }
}

template <
    bool ApplyTokenScale,
    typename ProblemSizes,
    typename ElementA,
    typename ElementB,
    typename ElementWeightScalePacked,
    typename ElementTokenScale,
    typename ElementD,
    typename StrideA,
    typename StrideB
>
__global__ void groupwise_verify_fused_e8m0_pre_mma_kernel(
    ProblemSizes problem_sizes,
    int group_num,
    ElementA *A,
    ElementB *B,
    ElementWeightScalePacked *weight_scale,
    ElementTokenScale *token_scale,
    ElementD *D,
    int block_tile_k, int group_size,
    StrideA stride_A, StrideB stride_B
) {
    ElementA *A_ptr = A;
    ElementB *B_ptr = B;
    ElementWeightScalePacked *weight_scale_ptr = weight_scale;
    ElementTokenScale *token_scale_ptr = token_scale;
    ElementD *D_ptr = D;

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    for (int group_id = 0; group_id < group_num; group_id++) {
        int N = get<0>(problem_sizes[group_id]);
        int M = get<1>(problem_sizes[group_id]);
        int K = get<2>(problem_sizes[group_id]);

        single_gemm_verify_fused_e8m0_pre_mma<ApplyTokenScale>(
          bid, tid,
          block_tile_k, group_size,
          M, N, K,
          A_ptr, B_ptr, weight_scale_ptr, token_scale_ptr, D_ptr
        );

        A_ptr += M * K;
        B_ptr += N * K / 2;
        weight_scale_ptr += N * K / block_tile_k;
        if constexpr (ApplyTokenScale) {
          token_scale_ptr += M;
        }
        D_ptr += M * N;
    }
}

template <
    bool ApplyTokenScale,
    typename ProblemSizes,
    typename ElementA,
    typename ElementB,
    typename ElementWeightScalePacked,
    typename ElementTokenScale,
    typename ElementD,
    typename StrideA,
    typename StrideB
>
void groupwise_verify_fused_e8m0_pre_mma(
    ProblemSizes problem_sizes,
    int group_num,
    ElementA *A,
    ElementB *B,
    ElementWeightScalePacked *weight_scale,
    ElementTokenScale *token_scale,
    ElementD *D,
    int block_tile_k, int group_size,
    StrideA stride_A, StrideB stride_B
) {
    groupwise_verify_fused_e8m0_pre_mma_kernel<ApplyTokenScale><<<1024, 1024>>>(
        problem_sizes,
        group_num,
        A, B, weight_scale, token_scale, D,
        block_tile_k, group_size,
        stride_A, stride_B);
    cudaDeviceSynchronize();
}



template <
    typename ProblemSizes,
    typename ElementA, // fp8
    typename ElementB, // int4
    typename ElementWeightScalePacked,
    typename ElementActivationScaleRaw,
    typename ElementD,
    typename StrideA,
    typename StrideB
>
__global__ void groupwise_verify_kernel(
    ProblemSizes problem_sizes,
    int group_num,
    ElementA *A,
    ElementB *B,
    ElementWeightScalePacked *weight_scale,
    ElementActivationScaleRaw *activation_scale,
    ElementD *D,
    float *abs_error_bound,
    int block_tile_k, int group_size,
    StrideA stride_A, StrideB stride_B
) {
    // if (thread0()) {
    //     printf("group num = %d\n", group_num);
    //     printf("stride = %d\n", group_num);
        
    //     for (int g = 0; g < group_num; g++) {
    //         printf("group %d, problem size (%d, %d, %d)\n", 
    //             g,
    //             get<0>(problem_sizes[g]),
    //             get<1>(problem_sizes[g]),
    //             get<2>(problem_sizes[g])
    //         );
    //     }

    //     print(problem_sizes[0]);
    //     print(problem_sizes[1]);
    //     print(stride_B[0]);
    //     print(stride_B[1]);
    //     printf("\n");

    //     uint8_t low = reinterpret_cast<uint8_t *>(B)[0] & 0xF;
    //     uint8_t high  = (reinterpret_cast<uint8_t *>(B)[0] & 0xF0) >> 4;

    //     printf("A     %f %f\n", float(A[0]), float(A[1]));
    //     printf("B     %f %f\n", float(low), float(high));
    //     printf("scale %f %f\n", float(scale[0]), float(scale[1]));
    // }

    ElementA *A_ptr = A;
    ElementB *B_ptr = B;
    ElementWeightScalePacked *weight_scale_ptr = weight_scale;
    ElementActivationScaleRaw *activation_scale_ptr = activation_scale;
    ElementD *D_ptr = D;

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    for (int group_id = 0; group_id < group_num; group_id++) {
        int N = get<0>(problem_sizes[group_id]);
        int M = get<1>(problem_sizes[group_id]);
        int K = get<2>(problem_sizes[group_id]);

        single_gemm_varify(
          bid, tid,
          block_tile_k, group_size,
          M, N, K,
          A_ptr, B_ptr, weight_scale_ptr, activation_scale_ptr, D_ptr,
          abs_error_bound
        );

        A_ptr += M * K;
        B_ptr += N * K / 2;
        weight_scale_ptr += N * K / block_tile_k;
        if constexpr (ScaleAppliesToActivation) {
          activation_scale_ptr += M * K / group_size;
        }
        D_ptr += M * N;
        if (abs_error_bound != nullptr) {
          abs_error_bound += M * N;
        }
    }
}


template <
    typename ProblemSizes,
    typename ElementA,
    typename ElementB,
    typename ElementWeightScalePacked,
    typename ElementActivationScaleRaw,
    typename ElementD,
    typename StrideA,
    typename StrideB
>
void groupwise_verify(
    ProblemSizes problem_sizes,
    int group_num,
    ElementA *A,
    ElementB *B,
    ElementWeightScalePacked *weight_scale,
    ElementActivationScaleRaw *activation_scale,
    ElementD *D,
    float *abs_error_bound,
    int block_tile_k, int group_size,
    StrideA stride_A, StrideB stride_B
) {
    groupwise_verify_kernel<<<1024, 1024>>>(
        problem_sizes, 
        group_num, 
        A, B, weight_scale, activation_scale, D,
        abs_error_bound,
        block_tile_k, group_size,
        stride_A, stride_B);
    cudaDeviceSynchronize();
}
