# Optimization Roadmap

Date: 2026-05-14

This roadmap is for the independent `cutlass_mixed_gemm` implementation only.
It must not create dependencies on MaloGEMM low-latency code.

## Optimization Rules

- Keep all CUTLASS-specific notes in
  `cutlass_mixed_gemm/.claude/malogemm-optimization/`.
- Benchmark CUTLASS and MaloGEMM low-latency serially.
- Use physical GPU 3 by default:

```bash
CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3
```

- Every final NCU command must explicitly include:

```bash
--clock-control none --cache-control all
```

- Treat NCU runs without both flags as exploratory only.
- Preserve negative results in docs before reverting or removing patches.

## Phase 0: Make The Benchmark Harness Safe

Before optimizing the kernel, fix or document harness constraints that can
invalidate measurements.

Recommended tasks:

1. Add explicit argument validation for the active INT4 x FP8 mode:
   - `K >= TileShapeK`
   - `K % TileShapeK == 0`
   - interleave-compatible `N` and `K`
   - no zero-M groups unless a safe skip path is implemented
2. Make `hw_info.device_id` follow `cudaGetDevice()` instead of hardcoding 0,
   then query SM count for that logical device.
3. Decide whether `host_problem_shapes_available` should be passed to CUTLASS
   for better `can_implement()` validation, or keep the current device-only
   grouped shape path and add equivalent explicit checks in the example.
4. Keep the build fix for `${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}` local to this
   independent CUTLASS tree.

Success criteria:

- Invalid target192-style inputs fail cleanly.
- Uniform 128-expert FC1/FC2 cases still pass correctness and benchmark.
- No changes are made to `low_latency_grouped_gemm`.

## Phase 1: Refresh Profiling Evidence

The previous scheduler log points to memory/occupancy pressure, but before
optimizing we should capture current evidence for the active binary and target
shapes.

Run:

- Large-M case where CUTLASS wins, such as `m/expert = 64` or `128`.
- Near-crossover case, such as `m/expert = 22`.
- FC1 and FC2 separately.

Collect:

- Runtime distribution.
- Register count and spill warnings from ptxas.
- NCU sections for memory throughput, eligible warps, local memory, global load
  efficiency, L2, shared-memory usage, and tensor pipe utilization.
- SASS evidence for conversion and scaling loops when needed.

Do not run low-latency and CUTLASS profiling concurrently.

## Phase 2: Memory And Occupancy First

The best documented ROI is memory/occupancy, not scheduler.

Candidate work:

- Reduce register pressure in the post-scale INT4 x FP8 mainloop.
- Inspect `intermediate_array` lifetime and whether chunk scaling can be
  restructured to reduce live accumulators.
- Check for local-memory spills caused by scale fragments, intermediate
  accumulators, or conversion helpers.
- Revisit conversion helpers and scale application SASS.
- Measure whether `TileShapeN = 16` is still best for FC1 and FC2 at large M.
- Examine SMEM usage and stage count from `StageCountAutoCarveout`.

Success criteria:

- Same correctness threshold as baseline.
- Improvement on both FC1 and FC2 or a documented shape-specific tradeoff.
- No regression at the near-crossover case unless explicitly accepted.

## Phase 3: TMA Descriptor Pre-Baking For INT4 x FP8

The pre-scale file already has a pre-baked descriptor path. The active INT4 x
FP8 post-scale file does not.

Experiment:

1. Port or factor the pre-baked descriptor mechanism into the post-scale
   collective, within `cutlass_mixed_gemm` only.
2. Keep the original path behind a compile-time switch for A/B testing.
3. Measure group counts and M distributions that stress group switching.
4. Include zero-M behavior in tests only after the harness is fixed.

Risks:

- Extra initialization kernel can dominate small group counts.
- Workspace layout changes can conflict with epilogue/scheduler workspace if
  alignment is wrong.
- Descriptor pointer switching must remain correct under SwapAB.

Expected use case:

- Many groups.
- Frequent group switches.
- Enough work per kernel that the initialization overhead is amortized.

## Phase 4: Config Search And Shape Specialization

The current default is fixed:

```text
Tile = 128 x 16 x 512
Cluster = 2 x 1 x 1
Schedule = cooperative ptr-array
```

Use the `PROFILE` path to re-run config search for MaloGEMM-relevant shapes.
Do not assume the current default is optimal for:

- FC1: `N = 1024, K = 4096`
- FC2: `N = 4096, K = 512`
- Near-crossover M values.
- Very large per-expert M values.

Watch for:

- Tile shape improving FC1 but hurting FC2.
- Cluster shapes that improve occupancy but worsen memory locality.
- Configs skipped by `can_implement()`.
- Profiler robustness if every config is skipped.

## Phase 5: Scheduler Only After Memory Work

The fixed-N scheduler optimization is already kept. Further scheduler ideas
should be lower priority unless fresh profiling shows group-switch overhead is
dominant.

Possible later experiments:

- Work stealing for severe group imbalance.
- A safer metadata prefetch strategy with lower register pressure.
- Prefix metadata only if problem shapes are host-known and stable.

Use these only after Phase 2/3 evidence shows memory and descriptor paths are
not the bottleneck.

## Phase 6: MXFP4 x BF16 Track

MXFP4 x BF16 is architecturally different because it routes to pre-scale.

Recommended approach:

- Keep INT4 x FP8 and MXFP4 x BF16 benchmark results separate.
- Validate the pre-scale route independently before applying INT4 x FP8
  conclusions.
- Measure whether descriptor pre-baking is already helping MXFP4.
- Check FP4 interleave cost and correctness separately from kernel runtime.

Do not mix MXFP4 findings into INT4 x FP8 decisions without a direct benchmark.
