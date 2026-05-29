#pragma once

#include <algorithm>
#include <stdexcept>

#include "kernel_profiler_shared.hpp"

namespace precomputed_scheduler {

using WorkTileCodec = cutlass::gemm::kernel::detail::PrecomputedGroupWorkTile;

struct State {
  cutlass::DeviceAllocation<uint64_t> work_tiles;
  uint64_t max_work_tile_count = 0;
  uint64_t work_tile_capacity = 0;
  dim3 gemm_grid_shape = dim3(1, 1, 1);
};

State scheduler_state;

uint64_t const* work_tiles_data() {
  return scheduler_state.work_tiles.get();
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

inline uint64_t max_tiles_from_total_tokens_host(
    Options const& options,
    int tile_shape_m,
    int tile_shape_n,
    int cluster_shape_m,
    int cluster_shape_n) {
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
      swizzle * uint64_t(cluster_shape_m);
  uint64_t const token_tile_group =
      swizzle * uint64_t(cluster_shape_n);
  uint64_t const tokens_per_padded_group =
      uint64_t(tile_shape_n) * token_tile_group;

  uint64_t channel_tiles = 0;
  for (int32_t i = 0; i < options.groups; ++i) {
    auto const problem = options.problem_sizes_host.at(i);
    uint64_t const original_n = uint64_t(cute::get<1>(problem));
    uint64_t const ctas_along_channel =
        (original_n + uint64_t(tile_shape_m) - 1) / uint64_t(tile_shape_m);
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

inline dim3 gemm_grid_shape(
    Options const& options,
    int cluster_shape_m,
    int cluster_shape_n) {
  using SchedulerParams = cutlass::gemm::kernel::detail::
      PersistentTileSchedulerSm90GroupParams<Shape<int,int,int>>;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  cutlass::gemm::GemmCoord cluster_shape(
      cluster_shape_m,
      cluster_shape_n,
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

inline void validate_limits_host(
    Options const& options,
    int tile_shape_m,
    int tile_shape_n,
    int cluster_shape_m,
    int cluster_shape_n) {
  if (options.groups > int32_t(WorkTileCodec::ExpertMask + 1)) {
    throw std::runtime_error(
        "Precomputed scheduler supports at most 524288 experts in the packed work-map format");
  }

  uint64_t const total_tokens = total_tokens_host(options);
  int const swizzle_log = log_swizzle_size(options, total_tokens, 1);
  uint64_t const swizzle = uint64_t(1) << swizzle_log;
  uint64_t const channel_tile_multiple =
      swizzle * uint64_t(cluster_shape_m);
  uint64_t const token_tile_multiple =
      swizzle * uint64_t(cluster_shape_n);

  for (int32_t group = 0; group < options.groups; ++group) {
    auto const problem = options.problem_sizes_host.at(group);
    uint64_t const tokens = uint64_t(cute::get<0>(problem));
    uint64_t const channels = uint64_t(cute::get<1>(problem));
    uint64_t const channel_tiles =
        (channels + uint64_t(tile_shape_m) - 1) / uint64_t(tile_shape_m);
    uint64_t const token_tiles =
        (tokens + uint64_t(tile_shape_n) - 1) / uint64_t(tile_shape_n);
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

inline void prepare_work_tiles(
    Options const& options,
    int tile_shape_m,
    int tile_shape_n,
    int cluster_shape_m,
    int cluster_shape_n) {
#if defined(CUTLASS_MIXED_GEMM_PRECOMPUTED_GROUP_OFFSETS)
  auto& state = scheduler_state;
  validate_limits_host(options, tile_shape_m, tile_shape_n, cluster_shape_m, cluster_shape_n);
  state.max_work_tile_count = max_tiles_from_total_tokens_host(
      options,
      tile_shape_m,
      tile_shape_n,
      cluster_shape_m,
      cluster_shape_n);
  state.gemm_grid_shape = gemm_grid_shape(options, cluster_shape_m, cluster_shape_n);
  uint64_t const scheduler_sentinel_count =
      uint64_t(state.gemm_grid_shape.x) *
      uint64_t(state.gemm_grid_shape.y) *
      uint64_t(state.gemm_grid_shape.z);
  uint64_t const required_work_tile_capacity =
      state.max_work_tile_count + scheduler_sentinel_count;
  if (required_work_tile_capacity > uint64_t(0xffffffffu)) {
    throw std::runtime_error("Precomputed scheduler work-map exceeds uint32_t index range");
  }
  if (required_work_tile_capacity > state.work_tile_capacity) {
    state.work_tile_capacity = required_work_tile_capacity;
    state.work_tiles.reset(state.work_tile_capacity == 0 ?
        1 : state.work_tile_capacity);
  }
#else
  (void)options;
  (void)tile_shape_m;
  (void)tile_shape_n;
  (void)cluster_shape_m;
  (void)cluster_shape_n;
#endif
}

__device__ __forceinline__
uint64_t make_work_tile(
    uint64_t global_linear_idx,
    uint64_t local_linear_idx,
    int group_idx,
    uint64_t problem_blocks_m,
    int cluster_shape_m,
    int cluster_shape_n,
    int swizzle_log,
    int gemm_grid_x,
    int gemm_grid_y) {
  uint64_t const cluster_shape_major = uint64_t(cluster_shape_m);
  uint64_t const cluster_shape_minor = uint64_t(cluster_shape_n);
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

__device__ __forceinline__
GroupInfo get_group_info(
    typename ProblemShape::UnderlyingProblemShape const& problem,
    int tile_shape_m,
    int tile_shape_n,
    int cluster_shape_m,
    int cluster_shape_n,
    int swizzle_log) {
  uint64_t const ctas_along_m =
      (uint64_t(cute::get<0>(problem)) + uint64_t(tile_shape_m) - 1) / uint64_t(tile_shape_m);
  uint64_t const ctas_along_n =
      (uint64_t(cute::get<1>(problem)) + uint64_t(tile_shape_n) - 1) / uint64_t(tile_shape_n);
  uint64_t const swizzle = uint64_t(1) << swizzle_log;
  uint64_t const m_multiple = swizzle * uint64_t(cluster_shape_m);
  uint64_t const n_multiple = swizzle * uint64_t(cluster_shape_n);
  uint64_t const problem_blocks_m =
      ((ctas_along_m + m_multiple - 1) / m_multiple) * m_multiple;
  uint64_t const problem_blocks_n =
      ((ctas_along_n + n_multiple - 1) / n_multiple) * n_multiple;

  if (problem_blocks_m > 0xffffffffULL || problem_blocks_n > 0xffffffffULL) {
    asm volatile("trap;");
  }

  return {problem_blocks_m, problem_blocks_n, problem_blocks_m * problem_blocks_n};
}

__global__
void build_work_tile_map_kernel(
    typename ProblemShape::UnderlyingProblemShape const* problem_shapes,
    int groups,
    int tile_shape_m,
    int tile_shape_n,
    int cluster_shape_m,
    int cluster_shape_n,
    int swizzle_log,
    int gemm_grid_x,
    int gemm_grid_y,
    uint64_t* work_tiles) {
  int const tid = threadIdx.x;
  uint64_t const total_grid_size = uint64_t(gemm_grid_x) * uint64_t(gemm_grid_y);

  if (groups <= 0) {
    if (blockIdx.x == 0) {
      for (uint64_t i = uint64_t(tid); i < total_grid_size; i += uint64_t(blockDim.x)) {
        work_tiles[i] = WorkTileCodec::Invalid;
      }
    }
    return;
  }

  int const group = int(blockIdx.x);
  if (group >= groups) {
    return;
  }

  extern __shared__ unsigned long long shared_storage[];
  unsigned long long* prefix_partials = shared_storage;
  unsigned long long* group_info_storage = shared_storage + blockDim.x;

  if (tid == 0) {
    GroupInfo const info =
        get_group_info(
            problem_shapes[group],
            tile_shape_m,
            tile_shape_n,
            cluster_shape_m,
            cluster_shape_n,
            swizzle_log);
    group_info_storage[0] = static_cast<unsigned long long>(info.problem_blocks_m);
    group_info_storage[1] = static_cast<unsigned long long>(info.group_tiles);
  }
  __syncthreads();

  uint64_t prefix_sum = 0;
  for (int prefix_group = tid; prefix_group < group; prefix_group += blockDim.x) {
    GroupInfo const info =
        get_group_info(
            problem_shapes[prefix_group],
            tile_shape_m,
            tile_shape_n,
            cluster_shape_m,
            cluster_shape_n,
            swizzle_log);
    prefix_sum += info.group_tiles;
  }

  prefix_partials[tid] = static_cast<unsigned long long>(prefix_sum);
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      prefix_partials[tid] += prefix_partials[tid + offset];
    }
    __syncthreads();
  }

  uint64_t const group_start = static_cast<uint64_t>(prefix_partials[0]);
  uint64_t const problem_blocks_m = static_cast<uint64_t>(group_info_storage[0]);
  uint64_t const group_tiles = static_cast<uint64_t>(group_info_storage[1]);

  for (uint64_t local_tile = uint64_t(tid);
       local_tile < group_tiles;
       local_tile += uint64_t(blockDim.x)) {
    uint64_t const global_tile = group_start + local_tile;
    work_tiles[global_tile] = make_work_tile(
        global_tile,
        local_tile,
        group,
        problem_blocks_m,
        cluster_shape_m,
        cluster_shape_n,
        swizzle_log,
        gemm_grid_x,
        gemm_grid_y);
  }

  if (group == groups - 1) {
    uint64_t const sentinel_start = group_start + group_tiles;
    for (uint64_t i = uint64_t(tid); i < total_grid_size; i += uint64_t(blockDim.x)) {
      work_tiles[sentinel_start + i] =
          WorkTileCodec::Invalid;
    }
  }
}

inline void build_work_tile_map(
    Options const& options,
    int tile_shape_m,
    int tile_shape_n,
    int cluster_shape_m,
    int cluster_shape_n,
    cudaStream_t stream) {
#if defined(CUTLASS_MIXED_GEMM_PRECOMPUTED_GROUP_OFFSETS)
  auto& state = scheduler_state;
  int const scheduler_threads = 128;
  dim3 const scheduler_grid(options.groups > 0 ? options.groups : 1);
  size_t const scheduler_smem =
      size_t(scheduler_threads + 2) * sizeof(unsigned long long);
  build_work_tile_map_kernel<<<scheduler_grid, scheduler_threads, scheduler_smem, stream>>>(
      problem_sizes.get(),
      options.groups,
      tile_shape_m,
      tile_shape_n,
      cluster_shape_m,
      cluster_shape_n,
      log_swizzle_size(options, state.max_work_tile_count, 1),
      state.gemm_grid_shape.x,
      state.gemm_grid_shape.y,
      state.work_tiles.get());
  CUDA_CHECK(cudaPeekAtLastError());
#else
  (void)options;
  (void)tile_shape_m;
  (void)tile_shape_n;
  (void)cluster_shape_m;
  (void)cluster_shape_n;
  (void)stream;
#endif
}

}  // namespace precomputed_scheduler

void prepare_precomputed_work_tile_map(
    Options const& options,
    int tile_shape_m,
    int tile_shape_n,
    int cluster_shape_m,
    int cluster_shape_n) {
  precomputed_scheduler::prepare_work_tiles(
      options,
      tile_shape_m,
      tile_shape_n,
      cluster_shape_m,
      cluster_shape_n);
}

void build_precomputed_work_tile_map(
    Options const& options,
    int tile_shape_m,
    int tile_shape_n,
    int cluster_shape_m,
    int cluster_shape_n,
    cudaStream_t stream) {
  precomputed_scheduler::build_work_tile_map(
      options,
      tile_shape_m,
      tile_shape_n,
      cluster_shape_m,
      cluster_shape_n,
      stream);
}
