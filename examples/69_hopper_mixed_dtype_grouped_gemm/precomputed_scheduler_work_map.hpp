#pragma once

#include <algorithm>
#include <stdexcept>
#include <type_traits>

#include "kernel_profiler_shared.hpp"

namespace precomputed_scheduler {

using WorkTileCodec = cutlass::gemm::kernel::detail::PrecomputedGroupWorkTile;

struct State {
  cutlass::DeviceAllocation<uint64_t> work_tiles;
  cutlass::DeviceAllocation<cute::TmaDescriptor> prebuilt_tma_desc_A;
  cutlass::DeviceAllocation<cute::TmaDescriptor> prebuilt_tma_desc_B;
  cutlass::DeviceAllocation<cute::TmaDescriptor> prebuilt_tma_desc_activation_scale;
  uint64_t max_work_tile_count = 0;
  uint64_t work_tile_capacity = 0;
  uint32_t work_tiles_per_worker = 0;
  uint64_t prebuilt_tma_desc_A_capacity = 0;
  uint64_t prebuilt_tma_desc_B_capacity = 0;
  uint64_t prebuilt_tma_desc_activation_scale_capacity = 0;
  dim3 gemm_grid_shape = dim3(1, 1, 1);
};

inline State scheduler_state;

inline uint64_t const* work_tiles_data() {
  return scheduler_state.work_tiles.get();
}

inline uint32_t work_tiles_per_worker() {
  return scheduler_state.work_tiles_per_worker;
}

inline cute::TmaDescriptor const* prebuilt_tma_desc_A_data() {
  return scheduler_state.prebuilt_tma_desc_A.get();
}

inline cute::TmaDescriptor const* prebuilt_tma_desc_B_data() {
  return scheduler_state.prebuilt_tma_desc_B.get();
}

inline cute::TmaDescriptor const* prebuilt_tma_desc_activation_scale_data() {
  return scheduler_state.prebuilt_tma_desc_activation_scale.get();
}

inline int log_swizzle_size(
    Options const& options,
    uint64_t problem_blocks_m,
    uint64_t problem_blocks_n) {
  using SchedulerParams = cutlass::gemm::kernel::detail::
      PersistentTileSchedulerSm90GroupParams<Shape<int,int,int>>;
  return SchedulerParams::get_log_swizzle_size(
      static_cast<int>(problem_blocks_m),
      static_cast<int>(problem_blocks_n),
      options.swizzle);
}

inline uint64_t total_tokens_host(Options const& options) {
  if (options.total_routed_tokens >= 0) {
    return uint64_t(options.total_routed_tokens);
  }

  uint64_t total_tokens = 0;
  for (int32_t i = 0; i < options.groups; ++i) {
    auto const problem = options.problem_sizes_host.at(i);
    total_tokens += uint64_t(cute::get<0>(problem));
  }
  return total_tokens;
}

template <
    int TileShapeM,
    int TileShapeN,
    int ClusterShapeM,
    int ClusterShapeN>
inline uint64_t max_tiles_from_total_tokens_host(Options const& options) {
  if (options.groups <= 0) {
    return 0;
  }

  uint64_t const total_tokens = total_tokens_host(options);
  if (total_tokens == 0) {
    return 0;
  }

  // The precomputed grouped scheduler presents all group tiles as a flattened
  // logical problem of shape (total_tiles, 1, 1), so swizzle is computed on
  // that flattened scheduler topology.
  int const swizzle_log = log_swizzle_size(options, total_tokens, 1);
  uint64_t const swizzle = uint64_t(1) << swizzle_log;
  uint64_t const channel_multiple =
      swizzle * uint64_t(ClusterShapeM);
  uint64_t const token_tile_group =
      swizzle * uint64_t(ClusterShapeN);
  uint64_t const tokens_per_padded_group =
      uint64_t(TileShapeN) * token_tile_group;

  uint64_t channel_tiles = 0;
  for (int32_t i = 0; i < options.groups; ++i) {
    auto const problem = options.problem_sizes_host.at(i);
    uint64_t const original_n = uint64_t(cute::get<1>(problem));
    uint64_t const ctas_along_channel =
        (original_n + uint64_t(TileShapeM) - 1) / uint64_t(TileShapeM);
    uint64_t const padded_channel_tiles =
        ((ctas_along_channel + channel_multiple - 1) / channel_multiple) *
        channel_multiple;
    channel_tiles = std::max(channel_tiles, padded_channel_tiles);
  }

  uint64_t const nonempty_experts =
      std::min(total_tokens, uint64_t(options.groups));
  uint64_t const extra_tokens = total_tokens - nonempty_experts;
  uint64_t const max_token_tile_rows =
      token_tile_group *
      (nonempty_experts + extra_tokens / tokens_per_padded_group);

  return channel_tiles * max_token_tile_rows;
}

template <int ClusterShapeM, int ClusterShapeN>
inline dim3 gemm_grid_shape(Options const& options) {
  using SchedulerParams = cutlass::gemm::kernel::detail::
      PersistentTileSchedulerSm90GroupParams<Shape<int,int,int>>;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
#if defined(CUTLASS_MIXED_GEMM_SINGLE_WG_CTAS_PER_SM)
  hw_info.sm_count *= CUTLASS_MIXED_GEMM_SINGLE_WG_CTAS_PER_SM;
#endif

  cutlass::gemm::GemmCoord cluster_shape(
      ClusterShapeM,
      ClusterShapeN,
      1);
  // The production path does not provide host problem shapes to the CUTLASS
  // grouped scheduler, so the GEMM launch grid is derived from SM count.
  dim3 problem_blocks = SchedulerParams::get_tiled_cta_shape_mnl(
      cluster_shape,
      static_cast<uint32_t>(hw_info.sm_count),
      1);
  return SchedulerParams::get_grid_shape(
      problem_blocks,
      cluster_shape,
      hw_info,
      options.swizzle,
      SchedulerParams::RasterOrderOptions::AlongM,
      true);
}

template <
    int TileShapeM,
    int TileShapeN,
    int ClusterShapeM,
    int ClusterShapeN>
inline void validate_limits_host(Options const& options) {
  if (options.groups > int32_t(WorkTileCodec::ExpertMask + 1)) {
    throw std::runtime_error(
        "Precomputed scheduler supports at most 524288 experts in the packed work-map format");
  }

  uint64_t const total_tokens = total_tokens_host(options);
  int const swizzle_log = log_swizzle_size(options, total_tokens, 1);
  uint64_t const swizzle = uint64_t(1) << swizzle_log;
  uint64_t const channel_tile_multiple =
      swizzle * uint64_t(ClusterShapeM);
  uint64_t const token_tile_multiple =
      swizzle * uint64_t(ClusterShapeN);

  for (int32_t group = 0; group < options.groups; ++group) {
    auto const problem = options.problem_sizes_host.at(group);
    uint64_t const tokens = uint64_t(cute::get<0>(problem));
    uint64_t const channels = uint64_t(cute::get<1>(problem));
    uint64_t const channel_tiles =
        (channels + uint64_t(TileShapeM) - 1) / uint64_t(TileShapeM);
    uint64_t const token_tiles =
        (tokens + uint64_t(TileShapeN) - 1) / uint64_t(TileShapeN);
    uint64_t const padded_channel_tiles =
        ((channel_tiles + channel_tile_multiple - 1) / channel_tile_multiple) *
        channel_tile_multiple;
    uint64_t const padded_token_tiles =
        ((token_tiles + token_tile_multiple - 1) / token_tile_multiple) *
        token_tile_multiple;

    if (padded_channel_tiles > uint64_t(WorkTileCodec::ChannelMask + 1) ||
        padded_token_tiles > uint64_t(WorkTileCodec::TokenMask + 1)) {
      throw std::runtime_error(
          "Precomputed scheduler packed work-map limit exceeded: "
          "requires padded channel tile count <= 1048576 and padded token tile count <= 16777216");
    }
  }
}

template <
    int TileShapeM,
    int TileShapeN,
    int ClusterShapeM,
    int ClusterShapeN>
inline void prepare_work_tiles(Options const& options) {
#if defined(CUTLASS_MIXED_GEMM_PRECOMPUTED_GROUP_OFFSETS)
  auto& state = scheduler_state;
  validate_limits_host<
      TileShapeM,
      TileShapeN,
      ClusterShapeM,
      ClusterShapeN>(options);
  state.max_work_tile_count = max_tiles_from_total_tokens_host<
      TileShapeM,
      TileShapeN,
      ClusterShapeM,
      ClusterShapeN>(options);
  state.gemm_grid_shape = gemm_grid_shape<ClusterShapeM, ClusterShapeN>(options);
  uint64_t const scheduler_sentinel_count =
      uint64_t(state.gemm_grid_shape.x) *
      uint64_t(state.gemm_grid_shape.y) *
      uint64_t(state.gemm_grid_shape.z);
#if defined(CUTLASS_MIXED_GEMM_SINGLE_WG_CHUNK_MAJOR_WORK_MAP)
  state.work_tiles_per_worker = static_cast<uint32_t>(
      (state.max_work_tile_count + scheduler_sentinel_count - 1) /
          scheduler_sentinel_count +
      1);
  uint64_t const required_work_tile_capacity =
      scheduler_sentinel_count * uint64_t(state.work_tiles_per_worker);
#else
  state.work_tiles_per_worker = 0;
  uint64_t const required_work_tile_capacity =
      state.max_work_tile_count + scheduler_sentinel_count;
#endif
  if (required_work_tile_capacity > uint64_t(0xffffffffu)) {
    throw std::runtime_error("Precomputed scheduler work-map exceeds uint32_t index range");
  }
  if (required_work_tile_capacity > state.work_tile_capacity) {
    state.work_tile_capacity = required_work_tile_capacity;
    state.work_tiles.reset(state.work_tile_capacity == 0 ?
        1 : state.work_tile_capacity);
  }
  if (state.prebuilt_tma_desc_A_capacity == 0) {
    state.prebuilt_tma_desc_A_capacity = 1;
    state.prebuilt_tma_desc_A.reset(state.prebuilt_tma_desc_A_capacity);
  }
  uint64_t const required_prebuilt_desc_capacity =
      options.groups > 0 ? uint64_t(options.groups) : uint64_t(1);
  if (required_prebuilt_desc_capacity > state.prebuilt_tma_desc_B_capacity) {
    state.prebuilt_tma_desc_B_capacity = required_prebuilt_desc_capacity;
    state.prebuilt_tma_desc_B.reset(state.prebuilt_tma_desc_B_capacity);
  }
  uint64_t const required_prebuilt_activation_scale_desc_capacity =
      options.groups > 0 ? uint64_t(options.groups) : uint64_t(1);
  if (required_prebuilt_activation_scale_desc_capacity >
      state.prebuilt_tma_desc_activation_scale_capacity) {
    state.prebuilt_tma_desc_activation_scale_capacity =
        required_prebuilt_activation_scale_desc_capacity;
    state.prebuilt_tma_desc_activation_scale.reset(
        state.prebuilt_tma_desc_activation_scale_capacity);
  }
#else
  (void)options;
#endif
}

template <int ClusterShapeM, int ClusterShapeN>
__device__ __forceinline__
uint64_t make_work_tile_static(
    uint64_t global_linear_idx,
    uint64_t local_linear_idx,
    int group_idx,
    uint64_t problem_blocks_m,
    int swizzle_log,
    int gemm_grid_x,
    int gemm_grid_y) {
  uint64_t const cluster_shape_major = uint64_t(ClusterShapeM);
  uint64_t const cluster_shape_minor = uint64_t(ClusterShapeN);
  uint64_t const total_grid_size = uint64_t(gemm_grid_x) * uint64_t(gemm_grid_y);
  uint64_t const worker_id = total_grid_size == 0 ? 0 : global_linear_idx % total_grid_size;
  uint64_t const cluster_minor_offset = worker_id % uint64_t(gemm_grid_y);

  uint64_t const blk_per_grid_dim = local_linear_idx / cluster_shape_minor;
  uint64_t const cluster_id = blk_per_grid_dim / cluster_shape_major;
  uint64_t const cluster_major_offset = blk_per_grid_dim % cluster_shape_major;

  uint64_t const swizzle = uint64_t(1) << swizzle_log;
  uint64_t const offset = cluster_id & (swizzle - 1);
  uint64_t const extra = cluster_id >> swizzle_log;
  uint64_t const curr_group_cluster_blk_major =
      problem_blocks_m / cluster_shape_major;
  uint64_t const cluster_idx_minor_div_swizzle = extra / curr_group_cluster_blk_major;
  uint64_t const cluster_idx_major = extra % curr_group_cluster_blk_major;
  uint64_t const cluster_idx_minor = cluster_idx_minor_div_swizzle * swizzle + offset;

  uint64_t const minor_work_idx = cluster_idx_minor * cluster_shape_minor + cluster_minor_offset;
  uint64_t const major_work_idx = cluster_idx_major * cluster_shape_major + cluster_major_offset;

  return WorkTileCodec::pack(major_work_idx, minor_work_idx, uint64_t(group_idx));
}

struct GroupInfo {
  uint64_t problem_blocks_m = 0;
  uint64_t problem_blocks_n = 0;
  uint64_t group_tiles = 0;
};

template <
    int TileShapeM,
    int TileShapeN,
    int ClusterShapeM,
    int ClusterShapeN>
__device__ __forceinline__
GroupInfo get_group_info_static(
    typename ProblemShape::UnderlyingProblemShape const& problem,
    int swizzle_log) {
  uint64_t const ctas_along_m =
      (uint64_t(cute::get<0>(problem)) + uint64_t(TileShapeM) - 1) / uint64_t(TileShapeM);
  uint64_t const ctas_along_n =
      (uint64_t(cute::get<1>(problem)) + uint64_t(TileShapeN) - 1) / uint64_t(TileShapeN);
  uint64_t const swizzle = uint64_t(1) << swizzle_log;
  uint64_t const m_multiple = swizzle * uint64_t(ClusterShapeM);
  uint64_t const n_multiple = swizzle * uint64_t(ClusterShapeN);
  uint64_t const problem_blocks_m =
      ((ctas_along_m + m_multiple - 1) / m_multiple) * m_multiple;
  uint64_t const problem_blocks_n =
      ((ctas_along_n + n_multiple - 1) / n_multiple) * n_multiple;

  if (problem_blocks_m > 0xffffffffULL || problem_blocks_n > 0xffffffffULL) {
    asm volatile("trap;");
  }

  return {problem_blocks_m, problem_blocks_n, problem_blocks_m * problem_blocks_n};
}

// The builder CTA stages TMA descriptors in dynamic shared memory before
// publishing them to the global prebuilt descriptor arrays. Slot and warp
// assignments are fixed so A, B, and activation-scale descriptors can be
// constructed independently inside one CTA.
static constexpr int PrebuiltTmaDescriptorScratchCount = 3;
static constexpr size_t PrebuiltTmaDescriptorScratchBytes =
    PrebuiltTmaDescriptorScratchCount * sizeof(cute::TmaDescriptor);
// Descriptor scratch slots inside the dynamic shared-memory prefix.
static constexpr int PrebuiltTmaDescriptorSlotA = 0;
static constexpr int PrebuiltTmaDescriptorSlotB = 1;
static constexpr int PrebuiltTmaDescriptorSlotActivationScale = 2;
// Publisher warp assignments for descriptor release to global memory.
static constexpr int PrebuiltTmaDescriptorWarpA = 0;
static constexpr int PrebuiltTmaDescriptorWarpB = 1;
static constexpr int PrebuiltTmaDescriptorWarpActivationScale = 2;

template <class MainloopParams>
static constexpr bool HasPrebuiltActivationScaleTmaDesc =
    !cute::is_same_v<typename MainloopParams::TMA_ActivationScale, cute::tuple<>>;

CUTE_DEVICE void
publish_prebuilt_tma_descriptor(
    cute::TmaDescriptor const* gmem_desc_ptr,
    cute::TmaDescriptor& smem_desc,
    int publisher_warp) {
  if ((threadIdx.x >> 5) == publisher_warp) {
    __syncwarp();
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    cute::tma_descriptor_cp_fence_release(gmem_desc_ptr, smem_desc);
    __syncwarp();
  }
}

template <class MainloopParams>
__device__ __forceinline__
void build_prebuilt_tma_descriptors(
    MainloopParams const& mainloop_params,
    typename ProblemShape::UnderlyingProblemShape const& problem,
    int group,
    cute::TmaDescriptor* smem_tma_desc,
    cute::TmaDescriptor* prebuilt_tma_desc_A,
    cute::TmaDescriptor* prebuilt_tma_desc_B,
    cute::TmaDescriptor* prebuilt_tma_desc_activation_scale) {
  if (group == 0) {
    cute::TmaDescriptor& smem_desc = smem_tma_desc[PrebuiltTmaDescriptorSlotA];
    if (threadIdx.x == PrebuiltTmaDescriptorWarpA * 32) {
      constexpr int MaxTensorRank = 5;
      cute::array<uint32_t, MaxTensorRank> prob_shape_A  = {1,1,1,1,1};
      cute::array<uint64_t, MaxTensorRank> prob_stride_A = {0,0,0,0,0};
      using PtrA = std::remove_reference_t<decltype(mainloop_params.ptr_A[group])>;
      PtrA ptr_A = nullptr;
      uint32_t const M = static_cast<uint32_t>(cute::get<0>(problem));
      uint32_t const K = static_cast<uint32_t>(cute::get<2>(problem));
      auto dA_group = mainloop_params.ptr_dA[group];
      auto stride_m = cute::get<0>(dA_group);
      auto stride_k = cute::get<1>(dA_group);
      int64_t const term_m = static_cast<int64_t>(M) * static_cast<int64_t>(stride_m);
      int64_t const term_k = static_cast<int64_t>(K) * static_cast<int64_t>(stride_k);
      int64_t const stride_l = term_m > term_k ? term_m : term_k;
      auto full_layout = make_layout(
          make_shape(M, K, static_cast<uint32_t>(mainloop_params.num_groups)),
          cute::make_stride(stride_m, stride_k, stride_l));
      Tensor tensor_a = make_tensor(ptr_A, full_layout);

      smem_desc = *mainloop_params.tma_load_a.get_tma_descriptor();
      cute::tma_descriptor_replace_addr_in_shared_mem(
          smem_desc,
          mainloop_params.ptr_A[0]);
      cute::detail::fill_tma_gmem_shape_stride(
          mainloop_params.tma_load_a,
          tensor_a,
          prob_shape_A,
          prob_stride_A);

      using ElementA = std::remove_cv_t<std::remove_pointer_t<PtrA>>;
      for (uint64_t& stride : prob_stride_A) {
        stride = (stride * cutlass::sizeof_bits<ElementA>::value) / 8;
      }
      cute::tma_descriptor_replace_dims_strides_in_shared_mem(
          smem_desc,
          prob_shape_A,
          prob_stride_A);
    }
    publish_prebuilt_tma_descriptor(
        &prebuilt_tma_desc_A[0],
        smem_desc,
        PrebuiltTmaDescriptorWarpA);
  }
  {
    cute::TmaDescriptor& smem_desc = smem_tma_desc[PrebuiltTmaDescriptorSlotB];
    if (threadIdx.x == PrebuiltTmaDescriptorWarpB * 32) {
      constexpr int MaxTensorRank = 5;
      cute::array<uint32_t, MaxTensorRank> prob_shape_B  = {1,1,1,1,1};
      cute::array<uint64_t, MaxTensorRank> prob_stride_B = {0,0,0,0,0};
      using PtrB = std::remove_reference_t<decltype(mainloop_params.ptr_B[group])>;
      PtrB ptr_B = nullptr;
      uint32_t const N = static_cast<uint32_t>(cute::get<1>(problem));
      uint32_t const K = static_cast<uint32_t>(cute::get<2>(problem));
      auto dB_group = mainloop_params.ptr_dB[group];
      auto stride_n = cute::get<0>(dB_group);
      auto stride_k = cute::get<1>(dB_group);
      auto full_layout = make_layout(
          make_shape(N, K, uint32_t(1)),
          cute::make_stride(stride_n, stride_k, int64_t(0)));
      Tensor tensor_b = make_tensor(ptr_B, full_layout);

      smem_desc = *mainloop_params.tma_load_b.get_tma_descriptor();
      cute::tma_descriptor_replace_addr_in_shared_mem(
          smem_desc,
          mainloop_params.ptr_B[group]);
      cute::detail::fill_tma_gmem_shape_stride(
          mainloop_params.tma_load_b,
          tensor_b,
          prob_shape_B,
          prob_stride_B);

      using ElementB = std::remove_cv_t<std::remove_pointer_t<PtrB>>;
      for (uint64_t& stride : prob_stride_B) {
        stride = (stride * cutlass::sizeof_bits<ElementB>::value) / 8;
      }
      cute::tma_descriptor_replace_dims_strides_in_shared_mem(
          smem_desc,
          prob_shape_B,
          prob_stride_B);
    }
    publish_prebuilt_tma_descriptor(
        &prebuilt_tma_desc_B[group],
        smem_desc,
        PrebuiltTmaDescriptorWarpB);
  }
  if constexpr (HasPrebuiltActivationScaleTmaDesc<MainloopParams>) {
    cute::TmaDescriptor& smem_desc =
        smem_tma_desc[PrebuiltTmaDescriptorSlotActivationScale];
    if (threadIdx.x == PrebuiltTmaDescriptorWarpActivationScale * 32) {
      constexpr int MaxTensorRank = 5;
      cute::array<uint32_t, MaxTensorRank> prob_shape_activation_scale  = {1,1,1,1,1};
      cute::array<uint64_t, MaxTensorRank> prob_stride_activation_scale = {0,0,0,0,0};
      using PtrActivationScale =
          std::remove_reference_t<decltype(mainloop_params.ptr_ActivationScale[group])>;
      PtrActivationScale ptr_activation_scale = nullptr;
      uint32_t const N = static_cast<uint32_t>(cute::get<1>(problem));
      uint32_t const K = static_cast<uint32_t>(cute::get<2>(problem));
      uint32_t const scale_groups = K / GROUP_SIZE;
      auto full_layout = make_layout(
          make_shape(N, scale_groups, uint32_t(1)),
          cute::make_stride(
              static_cast<int64_t>(scale_groups),
              cute::Int<1>{},
              int64_t(0)));
      Tensor tensor_activation_scale =
          make_tensor(ptr_activation_scale, full_layout);

      smem_desc = *mainloop_params.tma_load_activation_scale.get_tma_descriptor();
      cute::tma_descriptor_replace_addr_in_shared_mem(
          smem_desc,
          mainloop_params.ptr_ActivationScale[group]);
      cute::detail::fill_tma_gmem_shape_stride(
          mainloop_params.tma_load_activation_scale,
          tensor_activation_scale,
          prob_shape_activation_scale,
          prob_stride_activation_scale);

      using ElementActivationScale = std::remove_cv_t<std::remove_pointer_t<PtrActivationScale>>;
      for (uint64_t& stride : prob_stride_activation_scale) {
        stride = (stride * cutlass::sizeof_bits<ElementActivationScale>::value) / 8;
      }
      cute::tma_descriptor_replace_dims_strides_in_shared_mem(
          smem_desc,
          prob_shape_activation_scale,
          prob_stride_activation_scale);
    }
    publish_prebuilt_tma_descriptor(
        &prebuilt_tma_desc_activation_scale[group],
        smem_desc,
        PrebuiltTmaDescriptorWarpActivationScale);
  }
}

template <
    int StaticTileShapeM,
    int StaticTileShapeN,
    int StaticClusterShapeM,
    int StaticClusterShapeN,
    class MainloopParams>
__global__
void build_work_tile_map_kernel(
    typename ProblemShape::UnderlyingProblemShape const* problem_shapes,
    int groups,
    int swizzle_log,
    int gemm_grid_x,
    int gemm_grid_y,
    uint32_t work_tiles_per_worker,
    uint64_t* work_tiles,
    MainloopParams mainloop_params,
    cute::TmaDescriptor* prebuilt_tma_desc_A,
    cute::TmaDescriptor* prebuilt_tma_desc_B,
    cute::TmaDescriptor* prebuilt_tma_desc_activation_scale) {
  int const tid = threadIdx.x;
  uint64_t const total_grid_size = uint64_t(gemm_grid_x) * uint64_t(gemm_grid_y);

  if (groups <= 0) {
    if (blockIdx.x == 0) {
      for (uint64_t i = uint64_t(tid); i < total_grid_size; i += uint64_t(blockDim.x)) {
#if defined(CUTLASS_MIXED_GEMM_SINGLE_WG_CHUNK_MAJOR_WORK_MAP)
        work_tiles[i * uint64_t(work_tiles_per_worker)] = WorkTileCodec::Invalid;
#else
        work_tiles[i] = WorkTileCodec::Invalid;
#endif
      }
    }
    return;
  }

  int const group = int(blockIdx.x);
  if (group >= groups) {
    return;
  }

  extern __shared__ __align__(64) unsigned char shared_storage[];
  cute::TmaDescriptor* smem_tma_desc =
      reinterpret_cast<cute::TmaDescriptor*>(shared_storage);
  unsigned long long* prefix_partials =
      reinterpret_cast<unsigned long long*>(shared_storage + PrebuiltTmaDescriptorScratchBytes);
#if defined(CUTLASS_MIXED_GEMM_SINGLE_WG_CHUNK_MAJOR_WORK_MAP)
  unsigned long long* total_partials = prefix_partials + blockDim.x;
  unsigned long long* group_info_storage = total_partials + blockDim.x;
#else
  unsigned long long* group_info_storage = prefix_partials + blockDim.x;
#endif

  if (tid == 0) {
    GroupInfo const info = get_group_info_static<
        StaticTileShapeM,
        StaticTileShapeN,
        StaticClusterShapeM,
        StaticClusterShapeN>(
        problem_shapes[group],
        swizzle_log);
    group_info_storage[0] = static_cast<unsigned long long>(info.problem_blocks_m);
    group_info_storage[1] = static_cast<unsigned long long>(info.group_tiles);
  }

  build_prebuilt_tma_descriptors(
      mainloop_params,
      problem_shapes[group],
      group,
      smem_tma_desc,
      prebuilt_tma_desc_A,
      prebuilt_tma_desc_B,
      prebuilt_tma_desc_activation_scale);

  uint64_t prefix_sum = 0;
#if defined(CUTLASS_MIXED_GEMM_SINGLE_WG_CHUNK_MAJOR_WORK_MAP)
  uint64_t total_sum = 0;
  for (int scan_group = tid; scan_group < groups; scan_group += blockDim.x) {
    GroupInfo const info = get_group_info_static<
        StaticTileShapeM,
        StaticTileShapeN,
        StaticClusterShapeM,
        StaticClusterShapeN>(
        problem_shapes[scan_group],
        swizzle_log);
    total_sum += info.group_tiles;
    if (scan_group < group) {
      prefix_sum += info.group_tiles;
    }
  }
  total_partials[tid] = static_cast<unsigned long long>(total_sum);
#else
  for (int prefix_group = tid; prefix_group < group; prefix_group += blockDim.x) {
    GroupInfo const info = get_group_info_static<
        StaticTileShapeM,
        StaticTileShapeN,
        StaticClusterShapeM,
        StaticClusterShapeN>(
        problem_shapes[prefix_group],
        swizzle_log);
    prefix_sum += info.group_tiles;
  }
#endif

  prefix_partials[tid] = static_cast<unsigned long long>(prefix_sum);
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      prefix_partials[tid] += prefix_partials[tid + offset];
#if defined(CUTLASS_MIXED_GEMM_SINGLE_WG_CHUNK_MAJOR_WORK_MAP)
      total_partials[tid] += total_partials[tid + offset];
#endif
    }
    __syncthreads();
  }

  uint64_t const group_start = static_cast<uint64_t>(prefix_partials[0]);
  uint64_t const problem_blocks_m = static_cast<uint64_t>(group_info_storage[0]);
  uint64_t const group_tiles = static_cast<uint64_t>(group_info_storage[1]);
#if defined(CUTLASS_MIXED_GEMM_SINGLE_WG_CHUNK_MAJOR_WORK_MAP)
  uint64_t const total_tiles = static_cast<uint64_t>(total_partials[0]);
  uint64_t const tiles_per_worker = total_tiles == 0
      ? 1
      : (total_tiles + total_grid_size - 1) / total_grid_size;
#endif

  for (uint64_t local_tile = uint64_t(tid);
       local_tile < group_tiles;
       local_tile += uint64_t(blockDim.x)) {
    uint64_t const global_tile = group_start + local_tile;
#if defined(CUTLASS_MIXED_GEMM_SINGLE_WG_CHUNK_MAJOR_WORK_MAP)
    uint64_t const worker_idx = global_tile / tiles_per_worker;
    uint64_t const worker_tile_idx = global_tile % tiles_per_worker;
    uint64_t const storage_idx =
        worker_idx * uint64_t(work_tiles_per_worker) + worker_tile_idx;
#else
    uint64_t const storage_idx = global_tile;
#endif
    work_tiles[storage_idx] = make_work_tile_static<
        StaticClusterShapeM,
        StaticClusterShapeN>(
        global_tile,
        local_tile,
        group,
        problem_blocks_m,
        swizzle_log,
        gemm_grid_x,
        gemm_grid_y);
  }

  if (group == groups - 1) {
    [[maybe_unused]] uint64_t const sentinel_start = group_start + group_tiles;
    for (uint64_t i = uint64_t(tid); i < total_grid_size; i += uint64_t(blockDim.x)) {
#if defined(CUTLASS_MIXED_GEMM_SINGLE_WG_CHUNK_MAJOR_WORK_MAP)
      uint64_t const worker_start = i * tiles_per_worker;
      uint64_t const worker_tile_count = worker_start < total_tiles
          ? ((total_tiles - worker_start < tiles_per_worker)
                 ? total_tiles - worker_start
                 : tiles_per_worker)
          : 0;
      work_tiles[i * uint64_t(work_tiles_per_worker) + worker_tile_count] =
          WorkTileCodec::Invalid;
#else
      work_tiles[sentinel_start + i] = WorkTileCodec::Invalid;
#endif
    }
  }
}

template <
    int TileShapeM,
    int TileShapeN,
    int ClusterShapeM,
    int ClusterShapeN,
    class MainloopParams>
inline void build_work_tile_map(
    Options const& options,
    MainloopParams const& mainloop_params,
    cudaStream_t stream) {
#if defined(CUTLASS_MIXED_GEMM_PRECOMPUTED_GROUP_OFFSETS)
  auto& state = scheduler_state;
  int const scheduler_threads = 128;
  dim3 const scheduler_grid(options.groups > 0 ? options.groups : 1);
  size_t const scheduler_smem =
      PrebuiltTmaDescriptorScratchBytes +
      size_t(
#if defined(CUTLASS_MIXED_GEMM_SINGLE_WG_CHUNK_MAJOR_WORK_MAP)
          scheduler_threads * 2 + 2
#else
          scheduler_threads + 2
#endif
          ) * sizeof(unsigned long long);
  build_work_tile_map_kernel<
      TileShapeM,
      TileShapeN,
      ClusterShapeM,
      ClusterShapeN,
      MainloopParams><<<scheduler_grid, scheduler_threads, scheduler_smem, stream>>>(
      problem_sizes.get(),
      options.groups,
      log_swizzle_size(options, state.max_work_tile_count, 1),
      state.gemm_grid_shape.x,
      state.gemm_grid_shape.y,
      state.work_tiles_per_worker,
      state.work_tiles.get(),
      mainloop_params,
      state.prebuilt_tma_desc_A.get(),
      state.prebuilt_tma_desc_B.get(),
      state.prebuilt_tma_desc_activation_scale.get());
  CUDA_CHECK(cudaPeekAtLastError());
#else
  (void)options;
  (void)mainloop_params;
  (void)stream;
#endif
}

}  // namespace precomputed_scheduler

template <
    int TileShapeM,
    int TileShapeN,
    int ClusterShapeM,
    int ClusterShapeN>
void prepare_precomputed_work_tile_map(Options const& options) {
  precomputed_scheduler::prepare_work_tiles<
      TileShapeM,
      TileShapeN,
      ClusterShapeM,
      ClusterShapeN>(options);
}

template <
    int TileShapeM,
    int TileShapeN,
    int ClusterShapeM,
    int ClusterShapeN,
    class MainloopParams>
void build_precomputed_work_tile_map(
    Options const& options,
    MainloopParams const& mainloop_params,
    cudaStream_t stream) {
  precomputed_scheduler::build_work_tile_map<
      TileShapeM,
      TileShapeN,
      ClusterShapeM,
      ClusterShapeN>(
      options,
      mainloop_params,
      stream);
}
