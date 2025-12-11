/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/arch/memory.h"
#include "cutlass/arch/cache_operation.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/numeric_conversion.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA_,
  typename LayoutA_,
  typename ElementB_,
  typename ElementC_,
  typename ElementAccumulator_,
  typename EpilogueOutputOp_,
  int kElementsPerAccess_ = 1,            ///< Number of elements involved in a global access.
  int kThreadCount_ = 0,                  ///< Number of threads in the thread block.
                                          ///  It will be calculated automatically if set to 0.
  int kThreadsPerRow_ = 0,                ///< Number of threads in the k dimension.
                                          ///  It will be calculated automatically if set to 0.
  typename ElementSF_ = float,
  int kSFVecSize_ = 16
>
struct Gemv;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
CUTLASS_GLOBAL
void matrix_A_interleave_kernel(T *A_interleaved_, T *A_, int B, int M, int K) {

  int kElementsAccess = 128 / cutlass::sizeof_bits<T>::value;
  int interleave_block_k = blockDim.x * kElementsAccess;

  for (size_t b = blockIdx.y; b < B; b+= gridDim.y) {
    for (size_t m = blockIdx.x; m < M; m += gridDim.x) {
      for (size_t k = threadIdx.y * interleave_block_k; k < K; k+= blockDim.y * interleave_block_k) {
        
        if (k + threadIdx.x * kElementsAccess >= K) {
          break;
        }

        T *A = A_;
        T *A_interleaved = A_interleaved_;

        // move in the B dimension
        A += b * M * K;
        A_interleaved += b * M * K;

        // move in the M dimension
        A += m * K;
        A_interleaved += (m / 2) * K * 2 + (m % 2) * interleave_block_k;

        // move in the K dimension
        A += k + threadIdx.x * kElementsAccess;
        A_interleaved += k * 2 + threadIdx.x * kElementsAccess;

        float4 *A_128b = reinterpret_cast<float4 *>(A);
        float4 *A_interleaved_128b = reinterpret_cast<float4 *>(A_interleaved);

        float4 temp1 = *A_128b;

        *A_interleaved_128b = temp1;
      }
    }
  }
}


// GEMV for row-major A matrix
template <
    typename ElementA_,
    typename ElementB_,
    typename ElementC_,
    typename ElementAccumulator_,
    typename EpilogueOutputOp_,
    int kElementsPerAccess_,
    int kThreadCount_,
    int kThreadsPerRow_,
    typename ElementSF_,
    int kSFVecSize_
>
struct Gemv <
    ElementA_,            
    layout::RowMajor,
    ElementB_,            
    ElementC_,
    ElementAccumulator_,
    EpilogueOutputOp_,
    kElementsPerAccess_,
    kThreadCount_,
    kThreadsPerRow_,
    ElementSF_,
    kSFVecSize_
>{
public:

  using ElementA = ElementA_;
  using LayoutA = layout::RowMajor;
  using TensorRefA = TensorRef<ElementA, LayoutA>;

  using ElementB = ElementB_;
  using ElementC = ElementC_;

  using ElementAccumulator = ElementAccumulator_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ElementSF = ElementSF_;

  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  static FloatRoundStyle const Round = cutlass::FloatRoundStyle::round_to_nearest;

  // number of return elements in a global access
  static int const kElementsPerAccess = kElementsPerAccess_;
  static int const kSFVecSize = kSFVecSize_;
  static_assert(kSFVecSize == 16, 
    "Only SFVecSize = 16 is supported");
  static int const kSFPerAccess = std::max(1, kElementsPerAccess / kSFVecSize);
  static_assert(kSFPerAccess <= 4, 
    "kElementsPerAccess cannot exceed 64");
  
  static int const kPackedElementsA = cutlass::sizeof_bits<ElementA>::value == 4 ? 2 : 1;

  using FragmentA = Array<ElementA, kElementsPerAccess>;
  using FragmentB = Array<ElementB, kElementsPerAccess>;

  static int const kUnroll = 2;

  using FragmentArrayA = Array<FragmentA, kUnroll>;
  using FragmentArrayB = Array<FragmentB, kUnroll>;
  using FragmentArrayC = Array<float, 4>;

  // using FragmentCompute = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = Array<cutlass::half_t, kElementsPerAccess>;
  using FragmentSF = Array<ElementSF, kSFPerAccess>;

  // thread block shape (kThreadsPerRow, kThreadCount / kThreadsPerRow, 1)
  static int const kThreadCount = (kThreadCount_ <= 0) ? 128 : kThreadCount_;
  static int const kThreadsPerRow = 8; // fixed to 4 for mma.sync.aligned.m16n8k32. changed to 8 for interleaved format

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {
    // MatrixCoord      problem_size;
    int32_t         M;
    int32_t        *N;
    int32_t         K;
    int32_t         max_N;

    int32_t         batch_count;
    typename EpilogueOutputOp::Params output_op;

    TensorRefA      ref_A;

    ElementB const *ptr_B;
    ElementC const *ptr_C;
    ElementC       *ptr_D;

    int64_t         batch_stride_A;
    int64_t         batch_stride_B;
    int64_t         batch_stride_C;
    int64_t         batch_stride_D;

    ElementSF const *ptr_SF_A;
    ElementSF const *ptr_SF_B;

    //
    // Methods
    //

    Arguments(): batch_count(0) { }

    Arguments(
      // MatrixCoord      problem_size,
      int32_t          M,
      int32_t         *N,
      int32_t          K,
      int32_t          max_N,
      int32_t          batch_count,
      typename EpilogueOutputOp::Params output_op,
      TensorRefA       ref_A,
      void const      *ptr_B,
      void const      *ptr_C,
      void            *ptr_D,
      int64_t          batch_stride_A,
      int64_t          batch_stride_B,
      int64_t          batch_stride_C,
      int64_t          batch_stride_D,
      ElementSF const *ptr_SF_A = nullptr,
      ElementSF const *ptr_SF_B = nullptr
    ):
      // problem_size(problem_size),
      M(M),
      N(N),
      K(K),
      max_N(max_N),
      batch_count(batch_count),
      output_op(output_op),
      ref_A(ref_A),
      ptr_B(static_cast<ElementB const *>(ptr_B)),
      ptr_C(static_cast<ElementC const *>(ptr_C)),
      ptr_D(static_cast<ElementC       *>(ptr_D)),
      batch_stride_A(batch_stride_A),
      batch_stride_B(batch_stride_B),
      batch_stride_C(batch_stride_C),
      batch_stride_D(batch_stride_D),
      ptr_SF_A(ptr_SF_A),
      ptr_SF_B(ptr_SF_B)
    { }

    Arguments(
      // MatrixCoord problem_size,
      int32_t  M,
      int32_t *N,
      int32_t  K,
      int32_t  max_N,
      typename EpilogueOutputOp::Params output_op,
      TensorRefA  ref_A,
      void const *ptr_B,
      void const *ptr_C,
      void       *ptr_D
    ):
      Arguments(
        // problem_size,
        M,
        N,
        K,
        max_N,
        1,
        output_op,
        ref_A,
        ptr_B,
        ptr_C,
        ptr_D,
        1,
        1,
        1,
        1)
    { }

    Status update(Arguments const &args) {
      // problem_size = args.problem_size;
      M = args.M;
      N = args.N;
      K = args.K;
      max_N = args.max_N;
      batch_count = args.batch_count;
      output_op = args.output_op;
      ref_A = ref_A;
      ptr_B = args.ptr_B;
      ptr_C = args.ptr_C;
      ptr_D = args.ptr_D;
      batch_stride_A = args.batch_stride_A;
      batch_stride_B = args.batch_stride_B;
      batch_stride_C = args.batch_stride_C;
      batch_stride_D = args.batch_stride_D;

      return Status::kSuccess;
    }
  };

  using Params = Arguments;

  /// Shared memory storage structure
  union SharedStorage {

  };

public:

  template <typename T>
  static void matrix_A_interleave(T *A_interleaved, T* A, int B, int M, int K, CUstream_st *stream = 0) {
    dim3 grid(1024, 1024, 1);
    dim3 block(4, 64, 1);
    matrix_A_interleave_kernel<<<grid, block, 0, stream>>>(A_interleaved, A, B, M, K);
    cudaStreamSynchronize(stream);
  }

  //
  // Methods
  //

  CUTLASS_DEVICE
  Gemv() {}

  /// Determines whether kernel satisfies alignment
  static Status can_implement(int32_t K) {
    if (K % (4 * kElementsPerAccess * kUnroll) != 0) {
      return Status::kErrorMisalignedOperand;
    }
    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return can_implement(args.K);
  }

  using MMA_16x8x16_F32F16F16 = cutlass::arch::Mma<
    cutlass::gemm::GemmShape<16, 8, 16>,    // MMA Shape
    32,                                     // Number of threads participating
    cutlass::half_t,                        // A type
    cutlass::layout::RowMajor,              // A layout
    cutlass::half_t,                        // B type
    cutlass::layout::ColumnMajor,           // B layout
    float,                                  // accum type
    cutlass::layout::RowMajor,              // accum layout
    cutlass::arch::OpMultiplyAdd>;          // operator


  /// Executes one GEMV
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    
    // Loop over batch indices
    for (int batch_idx = blockIdx.z; batch_idx < params.batch_count; batch_idx += gridDim.z) {
      int idx_col_k = threadIdx.x;
      int idx_row_m = 4 * (blockIdx.x * blockDim.y + threadIdx.y);
      int N = params.N[batch_idx];

      int n_tile = blockIdx.y;

      if (n_tile >= (N + 7) / 8)
        return;

      // if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
      //   for(int b = 0; b < params.batch_count; b++) {
      //     printf("batch_idx %d, N %d\n", b, params.N[b]);
      //   }
      // }

      if (idx_row_m < params.M) {
        // problem_size (row = m, column = k)
        // matrix A (batch, m, k)
        // vector B (batch, k, n)
        // vector C (batch, m, n)
        // vector D (batch, m, n)

        // move in the batch dimension
        ElementA const *ptr_A = params.ref_A.data() + batch_idx * params.batch_stride_A / kPackedElementsA;
        ElementB const *ptr_B = params.ptr_B + batch_idx * params.batch_stride_B;

        ElementC const *ptr_C = params.ptr_C + batch_idx * params.batch_stride_C;
        ElementC *ptr_D = params.ptr_D + batch_idx * params.batch_stride_D;

        // move in the k dimension
        ptr_A += idx_col_k * kElementsPerAccess / kPackedElementsA;
        ptr_B += (idx_col_k % 4) * kElementsPerAccess;

        // move in the m dimension
        ptr_A += idx_row_m * params.K / kPackedElementsA;
        ptr_C += idx_row_m + idx_col_k / 4;
        ptr_D += idx_row_m + idx_col_k / 4;

        // move in the n dimension
        int n_B = (threadIdx.y % 4) * 2 + idx_col_k / 4 + 8 * n_tile;
        if (n_B < N) {
          ptr_B +=  n_B * params.K;
        }

        int n_CD = (idx_col_k % 4) * 2 + 8 * n_tile;
        ptr_C += n_CD * params.M;
        ptr_D += n_CD * params.M;

        FragmentArrayC frag_mma_c;
        frag_mma_c.clear();

        FragmentArrayA frag_array_A_row0;
        FragmentArrayA frag_array_A_row1;
        FragmentArrayB frag_array_B;

        FragmentSF fragSFA;
        FragmentSF fragSFB;

        int unroll_col_k = 0;

        // cols of the rolling tile
        int const tileA_k = kThreadsPerRow * kElementsPerAccess;
        int unroll_tile_k = kUnroll * tileA_k;
        int unroll_cols = params.K * 2 / unroll_tile_k * unroll_tile_k;

        for (; unroll_col_k < unroll_cols; unroll_col_k += unroll_tile_k) {

          for (int unroll_idx = 0; unroll_idx < kUnroll; unroll_idx++) {

            int unroll_col_k_ = unroll_col_k + unroll_idx * tileA_k;

            // fetch from matrix A
            arch::global_load<FragmentA,
                              sizeof(FragmentA),
                              arch::CacheOperation::LastUse>(
                                frag_array_A_row0[unroll_idx],
                                (ptr_A + unroll_col_k_ / kPackedElementsA), true);
            arch::global_load<FragmentA,
                              sizeof(FragmentA),
                              arch::CacheOperation::LastUse>(
                                frag_array_A_row1[unroll_idx],
                                (ptr_A + unroll_col_k_ / kPackedElementsA + params.K * 2 / kPackedElementsA), true);
  
            // fetch from vector B
            arch::global_load<FragmentB,
                              sizeof(FragmentB),
                              arch::CacheOperation::Always>(frag_array_B[unroll_idx], (ptr_B + unroll_col_k_ / 2), true);
          }

          NumericArrayConverter<cutlass::half_t, ElementA, kElementsPerAccess, Round> srcA_converter;
          NumericArrayConverter<cutlass::half_t, ElementB, kElementsPerAccess, Round> srcB_converter;
          MMA_16x8x16_F32F16F16 mma_op;

          for (int unroll_idx = 0; unroll_idx < kUnroll; unroll_idx++) {
  
            FragmentCompute fragA_compute_row0 = srcA_converter(frag_array_A_row0[unroll_idx]);
            FragmentCompute fragA_compute_row1 = srcA_converter(frag_array_A_row1[unroll_idx]);
            FragmentCompute fragB_compute = srcB_converter(frag_array_B[unroll_idx]);

            for (int e = 0; e < kElementsPerAccess; e+=4) {

              Array<cutlass::half_t, 8> frag_mma_a;
              Array<cutlass::half_t, 4> frag_mma_b;

              uint32_t *mma_2xfp16_A = reinterpret_cast<uint32_t *>(&frag_mma_a);
              uint32_t *mma_2xfp16_B = reinterpret_cast<uint32_t *>(&frag_mma_b);

              uint32_t const *frag_2xfp16_A_row0 = reinterpret_cast<uint32_t const *>(&(fragA_compute_row0.data()[e]));
              uint32_t const *frag_2xfp16_A_row1 = reinterpret_cast<uint32_t const *>(&(fragA_compute_row1.data()[e]));
              uint32_t const *frag_2xfp16_B = reinterpret_cast<uint32_t const *>(&(fragB_compute.data()[e]));

              mma_2xfp16_A[0] = frag_2xfp16_A_row0[0];
              mma_2xfp16_A[1] = frag_2xfp16_A_row1[0];
              mma_2xfp16_A[2] = frag_2xfp16_A_row0[1];
              mma_2xfp16_A[3] = frag_2xfp16_A_row1[1];

              mma_2xfp16_B[0] = frag_2xfp16_B[0];
              mma_2xfp16_B[1] = frag_2xfp16_B[1];

              mma_op(frag_mma_c, frag_mma_a, frag_mma_b, frag_mma_c);
            }
          }
        }

        EpilogueOutputOp output_op(params.output_op);
        typename EpilogueOutputOp::FragmentOutput source_fragment;

        if (n_CD < N) {
          // prefetch from source matrix C
          if (output_op.is_source_needed()) {         
            source_fragment[0] = *(ptr_C);
            source_fragment[2] = *(ptr_C + 2);
            if (n_CD + 1 < N) {
              source_fragment[1] = *(ptr_C + params.M);
              source_fragment[3] = *(ptr_C + params.M + 2);
            }
          }

          typename EpilogueOutputOp::FragmentOutput output_fragment;

          if (output_op.is_source_needed()) {
            output_fragment = output_op(frag_mma_c, source_fragment);
          }
          else {
            output_fragment = output_op(frag_mma_c);
          }

          *ptr_D = output_fragment[0];
          *(ptr_D + 2) = output_fragment[2];
          if (n_CD + 1 < N) {
            *(ptr_D + params.M) = output_fragment[1];
            *(ptr_D + params.M + 2) = output_fragment[3];
          }
        }
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
