
#pragma once

using namespace cute;

// #define ENABLE_PRINT
// #define DEBUG_INPUT


template <typename T>
__global__ void print_device(T *ptr, int count, const char str = ' ') {
#ifdef ENABLE_PRINT
  if (thread0()) {
    printf("print_device %c ", str);
    for (int i = 0; i < count; i++)
    {
      printf("(%d):%f ", i, float(ptr[i]));
    }
    printf("\n");
  }
#endif
}

template <typename T>
__global__ void print_device_packed(T *ptr, int count, const char str = ' ') {
#ifdef ENABLE_PRINT
  if (thread0()) {
    printf("print_device %c ", str);
    for (int i = 0; i < count; i++)
    {
      printf("(%d):%f", i, float(ptr[i][0]));
    }
    printf("\n");
  }
#endif
}


template <typename T>
__global__ void print_device_int4(T *ptr_, int count, const char str = ' ') {
#ifdef ENABLE_PRINT
  if (thread0()) {
    printf("print_device(int4) %c ", str);
    for (int i = 0; i < count / 2; i++)
    {
      uint8_t *ptr = reinterpret_cast<uint8_t *>(&ptr_[i]);
      uint8_t low = *ptr & 0x0F;
      uint8_t high = (*ptr & 0xF0) >> 4;
  
      printf("(%d):%d ", 2 * i, int(low));
      printf("(%d):%d ", 2 * i + 1, int(high));
    }
    printf("\n");
  }
#endif
}


template<typename T>
__global__ void set_device(T *ptr, int count, int value = 0) {
#ifdef DEBUG_INPUT
  if (thread0())
    for (int i = 0; i < count; i++)
    {
      if (value == 0)
        ptr[i] = static_cast<T>(i / (16*128) + 1);
      else
        ptr[i] = static_cast<T>(value);
    }
#endif
}


template<typename T>
__global__ void set_device_sequential(T *ptr, int count) {
  if (thread0())
    for (int i = 0; i < count; i++)
    {
      ptr[i] = static_cast<T>((i + 1) % 50) * 0.1f;
    }
}

__global__ void set_device_ue8m0(cutlass::float_ue8m0_t *ptr, int count) {
  if (thread0())
    for (int i = 0; i < count; i++)
    {
      // 114 -> 0.000122
      // 130 -> 8.000000
      uint8_t value = 114 + (i % (131 - 114));
      ptr[i] = *reinterpret_cast<cutlass::float_ue8m0_t *>(&value);

      // printf("set_device_ue8m0 %d %f\n", int(value), static_cast<float>(ptr[i]));
    }
}


template<typename T>
__global__ void set_device_int4(T *ptr_, int count, uint8_t value = 0) {
#ifdef DEBUG_INPUT
  // N x K
  // 16 x 256
  if (thread0())
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
#endif
}


template <typename T>
__global__ void compare_device(T *out, T *ref, int count) {
    
    if (thread0())
    {
        for (int i = 0; i < count; i++)
        {
            float abs_error = abs(float(out[i]) - float(ref[i]));
            if (abs_error > 1e-1)
                printf("(%d): out %f ref %f abs_error %f\n",
                    i, float(out[i]), float(ref[i]), abs_error);
        }
        printf("\n");
    }
}


template <
    typename ProblemSizes,
    typename ElementA, // fp8
    typename ElementB, // int4
    typename ElementScalePacked,
    typename ElementD,
    typename StrideA,
    typename StrideB
>
__global__ void groupwise_verify_kernel(
    ProblemSizes problem_sizes,
    int group_num,
    ElementA A, ElementB B, ElementScalePacked scale, ElementD D,
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

    float fp4_lut[] = {0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0, 
                       0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};

    ElementA A_ptr = A;
    uint8_t * B_ptr = reinterpret_cast<uint8_t *>(B);
    ElementScalePacked scale_ptr = scale;
    ElementD D_ptr = D;

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    for (int group_id = 0; group_id < group_num; group_id++) {
        int M = get<0>(problem_sizes[group_id]);
        int N = get<1>(problem_sizes[group_id]);
        int K = get<2>(problem_sizes[group_id]);

        for (int m = bid; m < M; m += gridDim.x) {
            for (int n = tid; n < N; n += blockDim.x) {

                float accum = 0.0f;

                for (int k = 0; k < K; k += 2) {

                    ElementA local_A_ptr = A_ptr + m * K + k;
                    uint8_t * local_B_ptr = B_ptr + n * K / 2 + k / 2;
                    ElementScalePacked local_scale_ptr = scale_ptr + (k / block_tile_k) * N + n;

                    float elem_A_0 = local_A_ptr[0];
                    float elem_A_1 = local_A_ptr[1];
                    uint8_t elem_B_low_ = (*local_B_ptr) & 0xF;
                    uint8_t elem_B_high_  = ((*local_B_ptr) & 0xF0) >> 4;
                    // float elem_B_low = (elem_B_low_ < 8) ? elem_B_low_ : (float)elem_B_low_ - 16;
                    // float elem_B_high = (elem_B_high_ < 8) ? elem_B_high_ : (float)elem_B_high_ - 16;
                    float elem_B_low = fp4_lut[elem_B_low_];
                    float elem_B_high = fp4_lut[elem_B_high_];

                    int scale_idx = (k % block_tile_k) / group_size;
                    float scale = static_cast<float>((*local_scale_ptr)[scale_idx]);

                    accum += elem_A_0 * elem_B_low * scale + elem_A_1 * elem_B_high * scale;

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

                ElementD local_D_ptr = D_ptr + m * N + n;
                *local_D_ptr = accum;
            }
        }

        A_ptr += M * K;
        B_ptr += N * K / 2;
        scale_ptr += N * K / block_tile_k;
        D_ptr += M * N;
    }
}


template <
    typename ProblemSizes,
    typename ElementA,
    typename ElementB,
    typename ElementScalePacked,
    typename ElementD,
    typename StrideA,
    typename StrideB
>
void groupwise_verify(
    ProblemSizes problem_sizes,
    int group_num,
    ElementA A, ElementB B, ElementScalePacked scale, ElementD D,
    int block_tile_k, int group_size,
    StrideA stride_A, StrideB stride_B
) {
    groupwise_verify_kernel<<<1024, 1024>>>(
        problem_sizes, 
        group_num, 
        A, B, scale, D,
        block_tile_k, group_size,
        stride_A, stride_B);
    cudaDeviceSynchronize();
}
