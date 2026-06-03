# MaloGEMM CUTLASS Mixed GEMM Notes

Date: 2026-05-14

This directory is the MaloGEMM-specific optimization notebook for the independent
`cutlass_mixed_gemm` tree. It intentionally does not update or share documents
with the MaloGEMM low-latency implementation.

Status: historical notebook. These files preserve the May 14 investigation path
and may contain wording such as "active INT4 x FP8" that was accurate at that
time. For current CMX scope, active dtype paths, benchmark rules, and commit
message rules, use `AGENTS.md`, `.claude/knowledge/`, and
`.claude/optimization-logs/`.

## Boundary Rule

`cutlass_mixed_gemm` is an independent library and benchmark target.

- Do not add includes, CMake links, source dependencies, or shared helper code
  between `cutlass_mixed_gemm` and `low_latency_grouped_gemm`.
- Use this tree only as a reference implementation, comparison target, and
  separate optimization target.
- Keep future CUTLASS-specific optimization logs and notes in this directory.
- Keep low-latency MaloGEMM design and production notes in the existing
  low-latency documentation, not here.

## Index

- `implementation-map.md`: entry points, build targets, dtype modes, host flow,
  runtime arguments, verification, and profiler path.
- `dataflow-and-kernel-architecture.md`: SwapAB, layouts, TMA, mainloop,
  scheduler, and correctness constraints.
- `weight-offline-reorder.md`: offline 4-bit weight interleave/reorder path,
  INT4 x FP8 and FP4 x BF16 mapping formulas, and mainloop consumption.
- `prior-optimization-record.md`: distilled knowledge from the existing
  `.claude/knowledge` and `.claude/optimization-logs` documents in this
  CUTLASS tree.
- `baseline-comparison-2026-05-14.md`: current CUTLASS vs MaloGEMM
  low-latency benchmark baseline.
- `target-workloads.md`: target MoE shapes, token distribution implications,
  and benchmark gaps for upcoming optimization.
- `optimization-roadmap.md`: recommended optimization order and experiment
  rules for upcoming work.

## Source Coverage

Read for this note set:

- `examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_fp8_grouped_gemm.cu`
- `examples/69_hopper_mixed_dtype_grouped_gemm/kernel_profiler_shared.hpp`
- `examples/69_hopper_mixed_dtype_grouped_gemm/grouped_mixed_dtype_utils.hpp`
- `examples/69_hopper_mixed_dtype_grouped_gemm/host_validation.hpp`
- `examples/69_hopper_mixed_dtype_grouped_gemm/BF16_MXFP4_Test.h`
- `examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_bf16_grouped_gemm.cu`
- `examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_mixed_dtype_grouped_gemm.cu`
- `include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp`
- `include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_prescale.hpp`
- `include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl`
- `include/cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp`
- `include/cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_cooperative.hpp`
- Existing CUTLASS-tree notes under `.claude/knowledge/`
- Existing CUTLASS-tree optimization log
  `.claude/optimization-logs/2026-03-19-scheduler-work-switching.md`

## Historical Quick Status

The active `69_hopper_int4_fp8_grouped_gemm` build currently selects the
INT4 x FP8 mode in `kernel_profiler_shared.hpp`:

- Activation: `cutlass::float_e4m3_t`
- Weight: `cutlass::int4b_t`
- Scale: `cutlass::bfloat16_t`
- Group size: `128`
- Default tile: `128 x 16 x 512`
- Default cluster: `2 x 1 x 1`
- Schedule: `KernelPtrArrayTmaWarpSpecializedCooperative`

The same code path has a commented MXFP4 x BF16 configuration:

- Activation: `cutlass::bfloat16_t`
- Weight: `cutlass::float_e2m1_t`
- Scale: `cutlass::float_ue8m0_t`
- Group size: `32`
- Default tile: `128 x 32 x 128`

The kernel is not intended for low-latency small-M cases. The baseline run on
2026-05-14 shows MaloGEMM low-latency wins at small M, while this CUTLASS path
wins once per-expert M is large enough.
