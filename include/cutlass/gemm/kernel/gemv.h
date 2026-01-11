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
  int kSplitKSlices_ = 1,
  typename ElementSF_ = float,
  int kSFVecSize_ = 16
>
struct Gemv;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ElementA, 
          typename ElementSF, 
          int kElementsPerAccess,
          int kSFVecSize>
CUTLASS_GLOBAL
void matrix_A_interleave_kernel(
  ElementA *A_interleaved_, 
  ElementA *A_, 
  ElementSF *SFA_padded_,
  ElementSF *SFA_,
  int B, 
  int M, 
  int K) {

    int kPackedElements = 8 / cutlass::sizeof_bits<ElementA>::value;
    int interleave_block_k = blockDim.x * kElementsPerAccess;

    for (size_t b = blockIdx.y; b < B; b+= gridDim.y) {
      for (size_t m = blockIdx.x; m < M; m += gridDim.x) {
        for (size_t k = threadIdx.y * interleave_block_k; k < K; k+= blockDim.y * interleave_block_k) {
          
          if (k + threadIdx.x * kElementsPerAccess >= K) {
            break;
          }

          uint8_t *A = reinterpret_cast<uint8_t *>(A_);
          uint8_t *A_interleaved = reinterpret_cast<uint8_t *>(A_interleaved_);

          // move in the B dimension
          A += b * M * K / kPackedElements;
          A_interleaved += b * M * K / kPackedElements;

          // move in the M dimension
          A += m * K / kPackedElements;
          A_interleaved += (m / 2) * K * 2 / kPackedElements + (m % 2) * interleave_block_k / kPackedElements;

          // move in the K dimension
          A += k / kPackedElements + threadIdx.x * kElementsPerAccess / kPackedElements;
          A_interleaved += k * 2 / kPackedElements + threadIdx.x * kElementsPerAccess / kPackedElements;

          using ElementAccess = cutlass::Array<ElementA, kElementsPerAccess>;

          if constexpr (cutlass::sizeof_bits<ElementA>::value == 4) {
            // INT4 x FP8

            const int num_uint32_t = cutlass::sizeof_bits<ElementAccess>::value / 32;

            Array<uint32_t, num_uint32_t> results;
            results.fill(0);

            uint32_t *ptr = reinterpret_cast<uint32_t *>(A);
            const int interleave_map[8] = {0, 4, 1, 5, 2, 6, 3, 7}; // {7, 3, 6, 2, 5, 1, 4, 0};
            
            for (int i = 0; i < num_uint32_t; i++) {
              uint32_t value = ptr[i];
          
              for (int j = 0; j < 8; j++) {
                  uint32_t int4_value = (value >> (j * 4)) & 0xF;                
                  results[i] |= (int4_value << (interleave_map[j] * 4));
              }
            }

            *reinterpret_cast<ElementAccess *>(A_interleaved) = *reinterpret_cast<ElementAccess *>(&results);

            if (k % kSFVecSize == 0) {
              int SFsPerRow = K / kSFVecSize;
              int PaddedSFsPerRow = (K / kSFVecSize + 7) / 8 * 8;

              // move in the B dimension
              ElementSF *SFA = SFA_ + b * M * SFsPerRow;
              ElementSF *SFA_padded = SFA_padded_ + b * M * PaddedSFsPerRow;

              // move in the M dimension
              SFA += m * SFsPerRow;
              SFA_padded += m * PaddedSFsPerRow;

              // move in the K dimension
              SFA += k / kSFVecSize;
              SFA_padded += k / kSFVecSize;

              *SFA_padded = *SFA;
            }
          }
          else {
            // FP8 x FP8
            *reinterpret_cast<ElementAccess *>(A_interleaved) = *reinterpret_cast<ElementAccess *>(A);
          }
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
    int kSplitKSlices_,
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
    kSplitKSlices_,
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
  
  static bool const kDequantizeA = cutlass::sizeof_bits<ElementA>::value == 4;
  static int const kPackedElementsA = cutlass::sizeof_bits<ElementA>::value == 4 ? 2 : 1;
  static int const kSFsPerAccess = 128 / cutlass::sizeof_bits<ElementSF>::value;

  using FragmentA = Array<ElementA, kElementsPerAccess>;
  using FragmentB = Array<ElementB, kElementsPerAccess>;
  using FragmentSF = Array<ElementSF, kSFsPerAccess>;

  static int const kUnroll = 2;
  static int const kSplitKSlices = kSplitKSlices_;

  using FragmentArrayA = Array<FragmentA, kUnroll>;
  using FragmentArrayB = Array<FragmentB, kUnroll>;
  using FragmentArrayC = Array<float, 4>;

  // using FragmentCompute = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = Array<cutlass::half_t, kElementsPerAccess>;

  static_assert(!kDequantizeA || kSFVecSize == 128, "Only kSFVecSize = 128 is supported");
  static_assert(!kDequantizeA || cutlass::sizeof_bits<ElementSF>::value == 16, "Only ElementSF of type BF16/FP16 is supported");
  static_assert(!kDequantizeA || kUnroll * 4 * kElementsPerAccess == kSFVecSize);

  // thread block shape (kThreadsPerRow, kThreadCount / kThreadsPerRow, 1)
  static int const kThreadCount = (kThreadCount_ <= 0) ? 128 : kThreadCount_;
  static int const kThreadsPerRow = 8; // fixed to 4 for mma.sync.aligned.m16n8k32. changed to 8 for interleaved format

  static constexpr uint32_t MaxThreadsPerBlock = kThreadCount;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 8;

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

    ElementSF const *ptr_SFA;

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
      ElementSF const *ptr_SFA = nullptr
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
      ptr_SFA(ptr_SFA)
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
    public:
    //
    // Type definitions
    //

    using ShapeSFA = MatrixShape<kThreadsPerRow + 1, kThreadCount / kThreadsPerRow * 2>;

    //
    // Data members
    //

    /// Buffer for SFA operand
    AlignedBuffer<float4, ShapeSFA::kCount> SFA;
  };

public:

  static void matrix_A_interleave(
    ElementA *A_interleaved, 
    ElementA *A,
    ElementSF *SFA_padded,
    ElementSF *SFA,
    int B, 
    int M, 
    int K, 
    CUstream_st *stream = 0) {
      dim3 grid(1024, 1024, 1);
      dim3 block(4, 64, 1);
      matrix_A_interleave_kernel<ElementA, ElementSF, kElementsPerAccess, kSFVecSize><<<grid, block, 0, stream>>>(
        A_interleaved, A, SFA_padded, SFA, B, M, K);
      cudaStreamSynchronize(stream);
  }

  static unsigned int get_SF_mem_size(int B, int M, int K) {
    return B * M * ((K / kSFVecSize * sizeof(ElementSF) + 15) / 16 * 16);
  }

  //
  // Methods
  //

  CUTLASS_DEVICE
  Gemv() {}

  /// Determines whether kernel satisfies alignment
  static Status can_implement(int32_t K) {
    if (K % (4 * kElementsPerAccess * kUnroll * kSplitKSlices) != 0) {
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
  // CUTLASS_DEVICE
  // void operator()(Params const &params, char* smem_buf) {
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    
    uint8_t *smem = reinterpret_cast<uint8_t *>(&shared_storage) + threadIdx.y * 16 * (kThreadsPerRow + 1);

    // Loop over batch indices
    int batch_idx = blockIdx.z / kSplitKSlices;
    int split_k_idx = blockIdx.z % kSplitKSlices;

    for (; batch_idx < params.batch_count; batch_idx += gridDim.z) {
      int idx_col_k = threadIdx.x;
      int idx_row_m = 4 * (blockIdx.x * blockDim.y + threadIdx.y);
      int N = params.N[batch_idx];
      int K_A_split = params.K * 2 / kSplitKSlices;

      int n_tile = blockIdx.y;

      if (n_tile >= (N + 7) / 8)
        return;

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

        ElementSF *s_SFA_row0 = (ElementSF *)(smem + threadIdx.x / 4 * 64);
        ElementSF *s_SFA_row1 = (ElementSF *)(smem + threadIdx.x / 4 * 64 + SharedStorage::ShapeSFA::kCount * 16 / 2);

        // move in the k dimension
        ptr_A += idx_col_k * kElementsPerAccess / kPackedElementsA + split_k_idx * K_A_split / kPackedElementsA;
        ptr_B += (idx_col_k % 4) * kElementsPerAccess + split_k_idx * params.K / kSplitKSlices;

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

        ElementSF const *ptr_SFA;
        if constexpr (kDequantizeA) {
          if ((idx_col_k % 4) * kSFsPerAccess * kSFVecSize * 2 < K_A_split) {
            
            int PaddedSFsPerRow = (params.K / kSFVecSize + 7) / 8 * 8;

            // move in the batch dimension
            ptr_SFA = params.ptr_SFA + batch_idx * params.M * PaddedSFsPerRow;

            // move in the k dimension
            ptr_SFA += split_k_idx * params.K / kSplitKSlices / kSFVecSize;

            // move in the m dimension
            ptr_SFA += (idx_row_m + idx_col_k / 4) * PaddedSFsPerRow;

            ptr_SFA += (idx_col_k % 4) * kSFsPerAccess;

            smem += threadIdx.x * 16;

            arch::cp_async<16, arch::CacheOperation::Global>(
                smem,
                reinterpret_cast<uint8_t const *>(ptr_SFA));
            arch::cp_async<16, arch::CacheOperation::Global>(
                smem + SharedStorage::ShapeSFA::kCount * 16 / 2,
                reinterpret_cast<uint8_t const *>(ptr_SFA + 2 * PaddedSFsPerRow));
          }
        }

        FragmentArrayC frag_mma_c;
        frag_mma_c.clear();

        FragmentArrayA frag_array_A_row0;
        FragmentArrayA frag_array_A_row1;
        FragmentArrayB frag_array_B;

        int unroll_col_k = 0;

        // cols of the rolling tile
        int const tileA_k = kThreadsPerRow * kElementsPerAccess;
        int unroll_tile_k = kUnroll * tileA_k;
        // int unroll_cols = params.K * 2 / unroll_tile_k * unroll_tile_k;

        for (; unroll_col_k < K_A_split; unroll_col_k += unroll_tile_k) {

          FragmentArrayC frag_mma_accum;
          frag_mma_accum.clear();

          CUTLASS_PRAGMA_UNROLL
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

          CUTLASS_PRAGMA_UNROLL
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

              if constexpr (kDequantizeA) {
                mma_op(frag_mma_accum, frag_mma_a, frag_mma_b, frag_mma_accum);
              }
              else {
                mma_op(frag_mma_c, frag_mma_a, frag_mma_b, frag_mma_c);
              }
            }
          }

          if constexpr (kDequantizeA) {

            cutlass::arch::cp_async_fence();
            cutlass::arch::cp_async_wait<0>();
            // __syncthreads();

            ElementSF SFA_row0 = *(s_SFA_row0);
            ElementSF SFA_row1 = *(s_SFA_row1);

            frag_mma_c[0] += frag_mma_accum[0] * float(SFA_row0);
            frag_mma_c[1] += frag_mma_accum[1] * float(SFA_row0);
            frag_mma_c[2] += frag_mma_accum[2] * float(SFA_row1);
            frag_mma_c[3] += frag_mma_accum[3] * float(SFA_row1);

            s_SFA_row0++;
            s_SFA_row1++;
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

          if (kSplitKSlices > 1)
          {
            atomic_add<ElementC> atom_add;
  
            atom_add(ptr_D, output_fragment[0]);
            atom_add(ptr_D + 2, output_fragment[2]);
  
            if (n_CD + 1 < N) {
              atom_add(ptr_D + params.M, output_fragment[1]);
              atom_add(ptr_D + params.M + 2, output_fragment[3]);
            }
          }
          else {
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
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
