# Prior Optimization Record

Date: 2026-05-14

This summarizes the existing Claude Code knowledge already stored in the
independent `cutlass_mixed_gemm` tree. It is a distilled working record for
future MaloGEMM-side optimization sessions.

## Documents Learned

Architecture:

- `.claude/knowledge/architecture/swapab-pattern.md`
- `.claude/knowledge/architecture/tma-fundamentals.md`
- `.claude/knowledge/architecture/persistent-kernel-work-scheduling.md`

Module deep dives:

- `.claude/knowledge/modules/hopper-mixed-precision-gemm-overview.md`
- `.claude/knowledge/modules/mainloop-pipeline-analysis.md`
- `.claude/knowledge/modules/tma-usage-in-mixed-precision-gemm.md`

Optimization:

- `.claude/knowledge/optimization/hopper-mixed-precision-grouped-gemm-optimization-summary.md`
- `.claude/knowledge/optimization/tma-descriptor-prebaking.md`
- `.claude/optimization-logs/2026-03-19-scheduler-work-switching.md`

## Durable Lessons

### SwapAB Is Central

SwapAB is not a cosmetic transpose. It is the reason variable original `M` can
be handled efficiently by Hopper WGMMA mixed-input kernels. It moves fixed
original `N` to the WGMMA M dimension and variable original `M` to the WGMMA N
dimension.

Optimization implications:

- Scheduler optimizations may exploit fixed original `N`, because after SwapAB
  it becomes fixed shape<0>.
- Scale layout must still match the quantized weight/output-channel dimension.
- TMA descriptors must be generated for the swapped tensor views.

### Scale/Zero Should Avoid Descriptor TMA

Previous work moved scale/zero from descriptor-based TMA to
`SM90_BULK_COPY_G2S`. This is a good design choice because scale/zero tiles are
small and descriptor setup would be disproportionately expensive.

Do not regress this without an explicit benchmark showing a win.

### Post-Scale Overlap Was Valuable

The post-scale mainloop intentionally overlaps groupwise scaling for chunk
`i - 1` with WGMMA for chunk `i`. The extracted `apply_groupwise_scale()`
helper made that logic easier to share and reason about.

Future changes to the mainloop should preserve this overlap unless the new path
reduces register pressure or instruction count enough to compensate.

### Fixed-N Scheduler Optimization Was Kept

The scheduler now caches `problem_blocks_m_fixed`. Existing record:

```text
Baseline:    158.05 us
Optimized:   153.25 us
Improvement: -4.80 us (-3.04%)
Decision:    keep
```

The real insight is narrow but useful: arithmetic and divmod work inside the
group-switch path can matter, but the bigger scheduler cost was still memory
latency for problem metadata.

### Scheduler Experiments After That Regressed

Two attempted scheduler optimizations were documented and reverted:

- SMEM cache for `problem_shapes`: setup cost exceeded benefit for CTAs that
  process few groups.
- Double-buffer scheduler prefetch: added register pressure and complexity
  without addressing the real bottleneck.

The optimization log concluded that more scheduler work likely has low ROI
unless memory and occupancy bottlenecks are first addressed.

### Memory/Occupancy Is The Next Serious Area

The scheduler log recorded NCU-style conclusions:

```text
Memory throughput pressure: primary bottleneck
Compute throughput: secondary bottleneck
Potential areas: local-memory spilling, global load coalescing, L2 compression,
SMEM/register usage
```

The exact NCU report was historical, so rerun profiling before using it as
final evidence. Per MaloGEMM rules, final NCU evidence must explicitly include:

```bash
--clock-control none --cache-control all
```

## TMA Descriptor Pre-Baking Status

The pre-scale file contains a concrete pre-baking implementation:

- Global-memory TMA descriptor field replacement helpers.
- A one-block-per-group initialization kernel.
- Workspace size of `2 * sizeof(TmaDescriptor) * (sm_count + group_count)`.
- Runtime use of per-group descriptor pointers (`get<2>`, `get<3>`).
- Original descriptor update/fence logic guarded by `ORIGINAL_TMA`.

Expected benefit in the prior note was 5-15% for many small groups, but
validation was marked pending.

Current code reality:

- MXFP4 x BF16 selects pre-scale, and the pre-scale collective contains
  pre-baked descriptor machinery.
- INT4 x FP8 selects the post-scale collective, which does not use this
  pre-baked per-group descriptor path.

Therefore, for the active INT4 x FP8 benchmark, TMA pre-baking is still a real
future experiment, not an already-measured production optimization.

## INT4 x FP8 History

The existing optimization summary records these major pieces:

- Initial W4A8 grouped implementation.
- Config profiler/auto-tuner infrastructure.
- Warpgroup wait cleanup.
- INT4 to FP8 conversion SASS optimization.
- Shared TMA and scheduler changes.

Current active defaults are still the hand-picked default config, not a fresh
auto-tuned result for the MaloGEMM shapes.

## MXFP4 x BF16 History

The existing summary records these major pieces:

- Initial grouped support.
- TensorRT-LLM/CUTLASS correctness fixes.
- FP8-to-BF16 scale conversion performance fix.
- FP4-to-BF16 conversion improvements including interleaved format and
  bit-manipulation/SASS work.
- Pre-scale scaffold and TMA descriptor pre-baking path.

The builder now routes MXFP4 to pre-scale based on `RealElementA ==
float_e2m1_t`.

## Negative-Result Discipline

The prior logs are useful because they preserved failed attempts with measured
outcomes. Keep this practice for upcoming work:

- Record baseline before patching.
- Record the exact implementation idea.
- Measure correctness and performance.
- Keep or revert based on data, but do not silently erase negative evidence.
