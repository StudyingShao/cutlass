# CMX Prebuilt TMA Descriptor Experiment

Date: 2026-06-02
GPU: NVIDIA H20, `CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3`

## Goal

Evaluate whether a Triton `ragged_tma.py`-style idea can improve CMX
`mxfp4 x mxfp8`: avoid per-group TMA descriptor field rewrites in the GEMM
mainloop by preparing per-expert descriptors in the precomputed scheduler
builder kernel.

The target repeated benchmark shapes are:

```text
E256_M24_N512_K4096:  Coop C1x1x1 Tile<128,32,512> Stages=4
E32_M128_N4096_K4096: Coop C1x1x1 Tile<128,64,256> Stages=5
```

Both commands include the scheduler/builder kernel in the timed region.

## Implemented Path

The retained implementation is the full prebuilt-descriptor path under
`CUTLASS_MIXED_GEMM_PRECOMPUTED_GROUP_OFFSETS=1`. The older experimental
`CUTLASS_MIXED_GEMM_PREBUILT_TMA_*` switches were removed after the codebase
settled on full prebuild as the only active precomputed-scheduler path.

Data flow:

```text
precomputed builder kernel:
  one CTA per expert
  build work-tile list
  compute group token offsets
  write one grouped-L A/weight TMA descriptor
  write one B TMA descriptor per expert
  write one activation-scale TMA descriptor per expert

main GEMM:
  work-map entry gives expert/local tile
  group switch updates current descriptor pointers
  skips B and activation-scale descriptor addr/dim/stride rewrites
  still fences/acquires descriptors before use
```

This is not a host-only descriptor bake. The descriptors are generated on the
device from runtime device problem shapes and current pointers.

## Commands

Build:

```bash
cmake --build cutlass_mixed_gemm/build -j --target \
  69_hopper_mxfp4_mxfp8_grouped_gemm_precomp_prebuilt_desc_coop_1x1x1_m128n32k512 \
  69_hopper_mxfp4_mxfp8_grouped_gemm_precomp_prebuilt_desc_coop_1x1x1_m128n64k256
```

M24:

```bash
CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
./cutlass_mixed_gemm/build/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_mxfp4_mxfp8_grouped_gemm_precomp_prebuilt_desc_coop_1x1x1_m128n32k512 \
  --m=24 --n=512 --k=4096 --groups=256 \
  --iterations=300 --warmup=20 --total_routed_tokens=6144
```

E32:

```bash
CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
./cutlass_mixed_gemm/build/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_mxfp4_mxfp8_grouped_gemm_precomp_prebuilt_desc_coop_1x1x1_m128n64k256 \
  --m=128 --n=4096 --k=4096 --groups=32 \
  --iterations=300 --warmup=20 --total_routed_tokens=4096
```

## Results

Baseline rows are the current precomputed scheduler path without prebuilt TMA
descriptors. Prebuilt rows include the descriptor builder work.

```text
Shape                         Baseline us  Prebuilt us  Delta
E256_M24_N512_K4096             262.168      262.920    +0.752 us / +0.29%
E32_M128_N4096_K4096            791.640      751.921    -39.719 us / -5.02%
```

Correctness passed with the existing sparse low-precision comparison policy:

```text
E256_M24_N512_K4096:   P99 846 0.03%, P98 428 0.01%, P95 175 0.01%
E32_M128_N4096_K4096: P99 4344 0.03%, P98 2144 0.01%, P95 893 0.01%
```

## Follow-up: Skip Unused Template Descriptor Copies

The full prebuilt path no longer consumes the per-CTA shared-memory B or
activation-scale template descriptors. The mainloop uses current prebuilt
descriptor pointers for both operands. A small follow-up therefore skipped the
`tensormaps_init()` copies for these unused descriptors when the corresponding
prebuilt mode is enabled.

Command shape was unchanged: fixed targets, GPU 3, `--iterations=300
--warmup=20`, and scheduler/builder included.

```text
Shape                         Prebuilt P1 us  Skip unused init us
E256_M24_N512_K4096              262.920          259.711, 259.975
E32_M128_N4096_K4096             751.921          752.055
```

This is the first positive result on the M24 target for the prebuilt descriptor
line. It is still a small gain, but it also does not regress the E32 target.

## Follow-up: Prebuild Weight A Descriptor

The next experiment also prebuilt the grouped weight/A TMA descriptor in the
same device builder. This keeps the production device-only model: the builder
reads the device problem shapes and mainloop params, then writes one grouped-L
A descriptor plus the existing per-expert B and activation-scale descriptors.

The first implementation was incorrect because it converted A descriptor
strides with `sizeof(ElementA)`. For the MXFP4 weight operand this treats a
4-bit element as one byte and doubles the descriptor stride. The fix matches the
mainloop path and converts strides with `cutlass::sizeof_bits<ElementA>`.

Fixed targets, GPU 3, `--iterations=300 --warmup=20`, and scheduler/builder
included:

```text
Shape                         Skip unused init us  + prebuilt A desc us
E256_M24_N512_K4096              259.711, 259.975      259.434
E32_M128_N4096_K4096             752.055               748.793
```

This is a valid but small improvement. It removes A descriptor initialization
and first-use A shape/stride update from the GEMM mainloop, but the M24 target
is still dominated by other mainloop and scheduler costs. The E32 shape shows a
slightly clearer benefit.

### Follow-up: Move prebuilt A descriptor pointer setup out of group update

After prebuilding the A descriptor, `current_tma_desc_a_` still got assigned on
every expert switch even though the descriptor pointer is global and constant.
Moving that assignment into `tensormaps_init()` removes the per-group pointer
write without adding a runtime state branch.

```text
Shape                         Prebuilt A desc us  Init-time A desc ptr us
E256_M24_N512_K4096              259.434              259.276
E32_M128_N4096_K4096             748.793              748.422
```

The gain is small but positive on both retained target shapes, so this change is
kept.

## Negative Experiments

### Contiguous weight-scale base/stride

The normal grouped path keeps weight-scale pointers and strides in per-expert
arrays. For the current packed MXFP4/MXFP8 benchmark setup, weight scales are
laid out contiguously by expert, so the group-switch path can derive the current
scale pointer from a base pointer and a constant group stride instead of loading
`ptr_S[next_batch]` and `dS[next_batch]`.

```text
Shape                         Init-time A desc ptr us  Contiguous scale us
E256_M24_N512_K4096              259.276                  258.692, 258.911
E32_M128_N4096_K4096             748.422                  749.301, 752.317
```

The first M24 runs looked mildly positive, but a same-session A/B rebuild did
not reproduce the benefit:

```text
Shape                         Contiguous scale us  No-contiguous rebuild us
E256_M24_N512_K4096              260.069, 260.713      259.894, 260.612
```

This is at best noise-level and may slightly disturb codegen. The code path was
removed instead of keeping another compile-time mode with no stable performance
benefit.

### Activation-scale-only prebuilt descriptor

Only `CUTLASS_MIXED_GEMM_PREBUILT_TMA_ACTSCALE_DESC=1` was tested.

```text
Shape                         Avg us
E256_M24_N512_K4096           265.836
E32_M128_N4096_K4096          790.203
```

This is not useful. If either B or activation scale still uses the normal
runtime descriptor update path, the mainloop still pays the group-switch update
and acquire machinery.

### Prebuilt-only acquire split

Changing the cooperative mainloop to acquire only the current prebuilt
descriptors looked acceptable on M24 but regressed E32:

```text
Shape                         Avg us
E256_M24_N512_K4096           262.130
E32_M128_N4096_K4096          788.608, 789.423
```

The branch/codegen and acquire behavior changed the hot loop enough that it was
not stable. The code was reverted to the original P1 acquire behavior.

### Split initialization branch

A second attempt moved the prebuilt/non-prebuilt selection out of the hot loop.

```text
Shape                         Avg us
E256_M24_N512_K4096           264.635
E32_M128_N4096_K4096          772.449
```

Also negative, so it was reverted.

### Combined descriptor release in builder

The builder was changed to issue one `tma_descriptor_fence_release()` after both
B and activation-scale descriptors instead of releasing after each descriptor.

```text
Shape                         Original P1 us  Combined-release us
E256_M24_N512_K4096              262.920          263.437
E32_M128_N4096_K4096             751.921          752.219
```

This is noise-level negative. The code was reverted to the original P1
per-descriptor release behavior.

### Empty shared-memory descriptors

A follow-up tried to make the unused prebuilt-mode B and activation-scale smem
descriptor fields empty in `TensorMapStorage`, after the template descriptor
copies had already been skipped.

```text
Shape                         Skip unused init us  Empty smem desc us
E256_M24_N512_K4096              259.975              259.980
E32_M128_N4096_K4096             752.055              751.999
```

This had no measurable performance effect and was manually reverted for the
performance-focused pass. Keep it as a cleanup TODO: unused path-specific smem
descriptor fields should eventually be conditionally empty so the prebuilt mode
does not carry dead descriptor state.

Cleanup TODO: re-apply this kind of unused-path removal near the end of the
optimization pass. Even when it is performance-neutral, the prebuilt descriptor
mode should not keep dead shared-memory descriptor fields once the performance
critical changes have settled. The motivation is code cleanliness and removing
unused-path residue, not latency.

### Skip empty descriptor commit/wait

When A, B, and activation-scale descriptors are all prebuilt, the mainloop no
longer performs shared-memory descriptor replacement before
`tensormaps_cp_fence_release()`. A follow-up tried to skip the unconditional
`tma_desc_commit_group()` / `tma_desc_wait_group()` in that all-prebuilt mode.

```text
Shape                         Prebuilt A desc us  Skip empty commit/wait us
E256_M24_N512_K4096              259.434              259.389
E32_M128_N4096_K4096             748.793              748.865
```

The M24 movement is too small to trust and E32 regressed slightly. The code was
reverted.

### Skip initial empty tensormap update/release

With A, B, and activation-scale descriptors all prebuilt, the initial
`tensormaps_perform_update()` and `tensormaps_cp_fence_release()` in the
cooperative wrapper no longer rewrite useful mainloop descriptor fields. A
follow-up added a compile-time `RequiresInitialTensormapUpdate` trait and
skipped that initial empty update/release in the full-prebuilt path while still
keeping the first prebuilt descriptor acquire.

```text
Shape                         Init-time A desc ptr us  Skip initial empty update us
E256_M24_N512_K4096              259.031                  259.445
E32_M128_N4096_K4096             748.467                  749.639
```

This did not help M24 and regressed E32 by about 1 us in the same-source local
rerun. The extra wrapper-level control-flow specialization appears more
expensive than the empty update/release it removes, or at least disturbs codegen
enough to erase the intended benefit. The code was reverted.

### Prefetch prebuilt B/activation-scale descriptors

CuTe exposes `prefetch_tma_descriptor()`, so another follow-up tried to prefetch
the prebuilt B and activation-scale descriptors for the current group and for
the next group switch before descriptor acquire/use. The prefetch did not
modify current descriptor state; it only issued `prefetch.tensormap` for the
descriptor addresses.

```text
Shape                         Init-time A desc ptr us  Descriptor prefetch us
E256_M24_N512_K4096              259.276                  259.899
E32_M128_N4096_K4096             749.338                  749.159
```

The E32 movement was noise-level, while M24 regressed by about 0.6 us. The
prefetch instructions do not hide enough latency to pay for their own producer
warp overhead on the target shapes. The code was reverted.

### Acquire prebuilt A descriptor once

The prebuilt A/weight descriptor is global and constant across experts, unlike
the per-expert B and activation-scale descriptors. A follow-up tried to acquire
the A descriptor only once per CTA by adding a per-thread boolean state inside
the collective.

```text
Shape                         Prebuilt A desc us  A acquire once us
E256_M24_N512_K4096              259.434              260.459
```

This regressed the M24 target by about 1 us, likely because the extra state and
branch disturbed the producer hot path more than the repeated acquire cost. E32
was not run after the M24 regression. The code was reverted.

### Use cached A descriptor pointer in the K loop

After moving `current_tma_desc_a_` setup into `tensormaps_init()`, another
follow-up tried to use that cached pointer in the TMA load hot loop instead of
reading `mainloop_params.ptr_A_prebuilt_tma_desc`.

```text
Shape                         Init-time A desc ptr us  Cached A desc ptr in load us
E256_M24_N512_K4096              259.276                  260.278
```

This regressed M24 by about 1 us, likely due to worse codegen/register behavior
in the K-loop. E32 was not run after the M24 regression. The code was reverted.

### Batch-change-only acquire for B/activation-scale

The initial descriptor acquire still needs to cover A, B, and activation-scale,
but after the first grouped batch the prebuilt A descriptor pointer is constant.
A follow-up added a separate batch-change acquire path that only acquired the
per-expert B and activation-scale descriptors, avoiding repeated A acquires
without adding per-CTA runtime state.

```text
Shape                         Baseline us  Batch-change B/actscale acquire us
E256_M24_N512_K4096              259.222               259.182
E32_M128_N4096_K4096             748.226               749.615
```

The M24 movement is only 0.04 us, which is within noise. E32 regressed by about
1.4 us. The likely cause is not the A acquire itself, but the extra
wrapper/control-flow specialization disturbing producer codegen. The code was
reverted.

### Parallel descriptor construction in builder

The builder originally has `tid==0` construct the prebuilt B and activation-scale
descriptors for each expert. A follow-up tried to use separate CTA lanes for
descriptor construction: lane 0 for the single grouped A descriptor, lane 1 for
B, and lane 2 for activation scale.

```text
Shape                         Init-time A desc ptr us  Parallel descriptor builder us
E256_M24_N512_K4096              259.276                  258.980
E32_M128_N4096_K4096             748.422                  749.451, 748.645
```

This helped M24 slightly but regressed E32. A second attempt used a generic
group-count heuristic, parallelizing descriptor construction only when
`groups >= blockDim.x` and preserving the serial builder for smaller group
counts.

```text
Shape                         Init-time A desc ptr us  Group-count heuristic builder us
E256_M24_N512_K4096              259.276                  259.876
```

The heuristic version also regressed M24, likely due to extra builder-side code
and branches. Both builder parallelization variants were reverted.

## Diagnostic: Builder/GEMM Split Timing

A temporary `--split_timing=true` diagnostic records one extra event between the
precomputed builder and main GEMM. This perturbs the total timing, so these
numbers are only for attribution, not final performance comparison.

```text
Shape                         Total us  Builder us  GEMM us
E256_M24_N512_K4096            263.965    11.792    252.173
E32_M128_N4096_K4096           751.989     9.983    742.005
```

The builder is now roughly 10-12 us in the full prebuilt-descriptor path. That
is material for the M24 target, while the E32 target is still dominated by the
main GEMM body. Future work should separate descriptor construction from
work-map generation before making more builder changes.

The same diagnostic on the matching normal precomputed-scheduler targets
without prebuilt descriptors:

```text
Shape                         Normal total us  Normal builder us  Normal GEMM us
E256_M24_N512_K4096              265.658            9.754            255.904
E32_M128_N4096_K4096             793.126            8.215            784.911
```

Compared with the normal path, full prebuilt descriptors add about 1.8-2.0 us
to the builder, but save about 3.7 us in the M24 main GEMM and about 42.9 us in
the E32 main GEMM under this diagnostic. The M24 net gain is therefore small
because the descriptor build cost consumes a large fraction of the mainloop
saving; the E32 gain remains large because the descriptor-update removal has a
much larger main GEMM effect.

Later fixed-target refresh on both production shapes, using the actual
`cutlass_mixed_gemm/build/examples/...` binaries and `--iterations=300
--warmup=20 --split_timing=true`:

```text
Shape                         Variant          Total us  Builder us  GEMM us
E256_M24_N512_K4096           normal precomp    266.138     9.639    256.499
E256_M24_N512_K4096           prebuilt desc     262.032    10.747    251.285
E32_M128_N4096_K4096          normal precomp    793.390     7.999    785.391
E32_M128_N4096_K4096          prebuilt desc     753.413    10.181    743.232
```

This refresh shows the same direction: prebuilt descriptors add about 1.1 us to
the builder, save about 5.2 us in the main GEMM, and net about 4.1 us on M24.
On E32 they add about 2.2 us to the builder but save about 42.2 us in the main
GEMM. The builder is still large enough that further M24 work should not ignore
it.

### Deferred: split large per-expert work-map fill

For large-tile experts, the current builder still maps one expert to one CTA and
that CTA writes all `group_tiles` entries for the expert. This is simple and
works well enough for the current M24 target, but it can become a fill-bandwidth
limit on larger shapes such as E32 with much larger channel/token tile counts.

Possible future direction: after `group_start` is known, split a large expert's
work-map fill across multiple CTAs while preserving the same final work-map
ordering. This needs more design discussion because it changes the builder grid
topology and either requires a prior offset phase or a different way to assign
global output ranges. Do not mix this into the current descriptor-prebuild
optimization pass.

### Builder initial barrier removal

The builder originally had a CTA-wide `__syncthreads()` immediately after
`tid==0` computed the current group's `GroupInfo` and built any prebuilt TMA
descriptors. That barrier is unnecessary: `GroupInfo` is only read after the
later prefix-reduction barrier, and descriptor construction does not depend on
other threads. Removing it lets non-zero threads start the prefix scan while
`tid==0` constructs descriptors, and also removes one CTA barrier.

Fixed prebuilt-desc targets after rebuild:

```text
Shape                         Split total us  Builder us  GEMM us  Standard us
E256_M24_N512_K4096              261.988        10.562    251.426     260.566
E32_M128_N4096_K4096             753.568        10.052    743.516     750.743
```

Compared with the immediately prior split refresh, builder time moved from
10.747 to 10.562 us on M24 and from 10.181 to 10.052 us on E32. Total timing is
noise-level, but the change is semantically cleaner and removes a real
unnecessary barrier, so keep it unless later measurements show a regression.

### Warp-parallel descriptor construction in builder

The earlier lane-parallel builder experiment was not retained because it was
not stable on E32. The useful version must respect the CuTe/TMA requirement that
`tensormap.cp_fenceproxy...aligned` is called by a whole warp. This follow-up
therefore gives the builder three shared-memory descriptor slots and uses
separate full warps:

```text
warp0: grouped A descriptor, only group 0
warp1: per-expert B descriptor
warp2: per-expert activation-scale descriptor
```

This keeps the runtime/device-only builder model and still rebuilds descriptor
dims/strides from the per-expert problem shape. An address-only descriptor
update was not applied here because it would need an explicit uniform shape
contract; the current generic builder relies on the per-expert problem shape to
define each TMA window.

Fixed prebuilt-desc targets after rebuild, GPU 3, `--iterations=300
--warmup=20 --split_timing=true`:

```text
Shape                         Before builder us  Warp-parallel builder us  Total us  GEMM us
E256_M24_N512_K4096                12.0847              9.908              261.926   252.017
E32_M128_N4096_K4096               10.052               8.376              750.291   741.915
```

Correctness stayed on the same sparse low-precision policy:

```text
E256_M24_N512_K4096:   P99 846 0.03%, P98 428 0.01%, P95 175 0.01%
E32_M128_N4096_K4096: P99 4344 0.03%, P98 2144 0.01%, P95 893 0.01%
```

The M24 baseline comes from the immediately prior single-scratch shared-memory
publish implementation. The E32 baseline comes from the preceding fixed-target
split record after builder barrier removal. The result is positive enough to
keep for review.

### Rank-aware descriptor property replace

The builder currently calls `fill_tma_gmem_shape_stride()` and then replaces 5
global dims and 4 global strides in the descriptor. A follow-up tried to infer
the actual TMA rank from CuTe's `AuxParams::TmaGmemBasis` and only emit
`rank` dim replacements plus `rank - 1` stride replacements.

This kept correctness on M24 but regressed builder time:

```text
Shape                         Baseline builder us  Rank-aware builder us
E256_M24_N512_K4096                  10.562                11.482
```

Decision: revert the experiment. Although it removes several inline
`tensormap.replace` instructions on paper, the extra templated helper/codegen
appears to increase builder-side instruction footprint or scheduling pressure
enough to lose in practice.

### One-shot prebuilt weight descriptor acquire

The prebuilt A/weight descriptor is global and does not change across groups,
while prebuilt B/activation and activation-scale descriptors are selected per
group. A follow-up tried to acquire the prebuilt A descriptor only once per CTA
and keep acquiring B/activation plus activation-scale on every batch change.

```text
Shape                         Baseline split us  One-shot A-acquire split us
E256_M24_N512_K4096                261.988                 262.304

Shape                         Baseline GEMM us   One-shot A-acquire GEMM us
E256_M24_N512_K4096                251.426                 251.669
```

Decision: revert the experiment. The added hot-path state/branch did not reduce
GEMM time; either the A descriptor acquire is not a meaningful bottleneck, or
the branch/codegen cost cancels the saved acquire.

A stricter ablation also skipped the prebuilt A descriptor acquire entirely,
without adding a state branch. M24 correctness stayed within the existing
sparse-outlier behavior, but performance did not improve:

```text
Shape                         Skip-A-acquire split us  Builder us  GEMM us
E256_M24_N512_K4096                    262.957          11.386    251.571
```

Decision: revert this too. It has no speed benefit and weakens the intended
TMA proxy acquire/release semantics for a descriptor written by the builder
kernel.

### Conditional descriptor commit/wait in full-prebuilt mode

Full prebuilt A/B/activation-scale mode does not release shared-memory
tensormap descriptors in `tensormaps_cp_fence_release()`, but the function still
unconditionally performs `tma_desc_commit_group()` and
`tma_desc_wait_group()`. A follow-up guarded those two operations behind a
compile-time check that at least one shared descriptor may be released.

```text
Shape                         Baseline split us  Conditional commit/wait us
E256_M24_N512_K4096                262.338                 262.598

Shape                         Baseline GEMM us   Conditional GEMM us
E256_M24_N512_K4096                251.615                 251.875
```

Decision: revert the experiment. It is logically attractive, but it did not
improve the fixed M24 target and touches descriptor proxy ordering machinery.

### Ragged desc-once comparison

`CUTLASS_MIXED_GEMM_RAGGED_TMA_DESC_ONCE=1` uses a single large ragged TMA
descriptor for both B/activation and activation scale, avoiding per-expert
descriptor arrays. It was correct but slower than the current full-prebuilt
path:

```text
Shape                         Full prebuilt us  Ragged desc-once us
E256_M24_N512_K4096              259.247             262.919
E32_M128_N4096_K4096             749.666             784.069
```

A narrower temporary combination kept A and activation-scale descriptors
prebuilt, but changed only B/activation to ragged desc-once:

```text
Shape                         Full prebuilt us  Ragged-B + prebuilt A/actscale us
E256_M24_N512_K4096              259.247                   261.280
```

This also lost to full prebuilt. The B ragged coordinate-domain path itself is
not cheap enough to replace the per-expert B descriptor build on the target
shape. The temporary CMake target was removed after recording the result.

### Builder prefix warp-reduction

The precomputed work-map builder computes each expert's global tile prefix inside
one CTA. The original implementation reduces the per-thread partial sums through
shared memory. A follow-up replaced this with warp shuffle reductions plus one
final warp-level reduction.

```text
Shape                         Full prebuilt us  Warp-reduce builder us
E256_M24_N512_K4096              259.247             259.449, 260.434
E32_M128_N4096_K4096             749.666             749.040
```

The E32 movement was slightly positive, but M24 was neutral-to-negative and the
second M24 run regressed. This is not stable enough for the target path, so the
code was reverted after recording the result.

Post-revert sanity:

```text
Shape                         Avg us
E256_M24_N512_K4096           259.038
E32_M128_N4096_K4096          749.242
```

### Worker-sticky work-map order

The precomputed scheduler normally emits a group-major work-map. Each persistent
CTA consumes entries at `worker + sequence * total_grid_size`, so a worker can
switch experts frequently. A worker-sticky experiment instead tried to make each
worker consume multiple tiles from the same expert consecutively, hoping to
reduce prebuilt descriptor pointer/acquire churn and improve B/activation-scale
descriptor locality in the GEMM mainloop.

Three builder variants were tested:

```text
Shape                         Variant                         Total us  Builder us  GEMM us
E256_M24_N512_K4096           full prebuilt split baseline     263.965    11.792    252.173
E256_M24_N512_K4096           sticky, per-expert prefix scan   411.562   163.094    248.468
E256_M24_N512_K4096           sticky, worker scans groups      437.399   189.202    248.197
E256_M24_N512_K4096           sticky, assignment enumeration   266.712    18.746    247.967
E32_M128_N4096_K4096          full prebuilt split baseline     751.989     9.983    742.005
E32_M128_N4096_K4096          sticky, assignment enumeration   877.862    10.614    867.248
```

The optimized assignment-enumeration builder computes each worker queue by
walking `assignment = worker + q * total_grid_size` and deriving
`group = assignment / workers_per_group`, `lane = assignment % workers_per_group`.
That fixed the builder explosion, but the path is still not useful:

```text
Shape                         Full prebuilt us  Sticky assignment us
E256_M24_N512_K4096              259.038             262.766
```

M24 shows the intended main-GEMM effect: worker-sticky ordering saved about
4.2 us in GEMM split timing. The builder still costs about 7 us more, so total
latency regresses. E32 is a stronger rejection: the builder is fine, but the GEMM
body regresses by about 125 us. The likely cause is that forcing each worker to
stay on an expert disrupts the existing swizzle/raster order and worsens
memory-system locality across CTAs, especially for the large output surface.

Decision: do not keep worker-sticky as an active code path. The useful lesson is
that reducing group-switch work can help M24 locally, but work-map order must not
fight the scheduler swizzle/raster locality.

Post-cleanup sanity after removing the worker-sticky source/CMake experiment
path:

```text
Shape                         Avg us    Builder us  GEMM us
E256_M24_N512_K4096           259.507    11.627     251.901
E32_M128_N4096_K4096          749.560    10.227     742.767
```

The split rows use `--split_timing=true`, so their totals are diagnostic rather
than final comparison points. The standard Avg rows confirm the active
full-prebuilt path returned to the previous baseline after the sticky experiment
was removed.

### Static zero L coordinate for prebuilt B/activation-scale descriptors

The full-prebuilt path builds one B descriptor and one activation-scale
descriptor per expert. Those descriptors already bind the expert base pointer,
and the builder gives them a logical L dimension of 1 with zero L stride. The
mainloop producer already passes a runtime `mock_l_coord = 0` into `load()` for
B and activation-scale; a follow-up changed only the prebuilt B/activation-scale
paths to make that zero a compile-time `Int<0>`. A remains unchanged because the
current A descriptor is global across experts and still uses the grouped L
coordinate.

Fixed targets, GPU 3, `--iterations=300 --warmup=20`:

```text
Shape                         Clean full-prebuilt us  Static L=0 us
E256_M24_N512_K4096                  259.507             259.088
E32_M128_N4096_K4096                 749.560             749.306
```

M24 split timing after the change:

```text
Shape                         Total us  Builder us  GEMM us
E256_M24_N512_K4096            262.519    10.691    251.829
```

The measured gain is small and close to run-to-run noise, as expected because
the value was already zero at runtime. The change is kept because it makes the
prebuilt descriptor coordinate semantics explicit and does not regress either
retained target shape.

### Uniform-K tile-count specialization

Grouped MoE production shapes are var-M, but K is normally uniform across
experts. An experiment added `CUTLASS_MIXED_GEMM_ASSUME_UNIFORM_K_TILE_COUNT`
to replace per-tile `TileScheduler::get_work_k_tile_count(...)` calls with the
global `k_tile_count`, and to skip the producer-side grouped problem-shape reload
that was only needed to recompute K tile count. This does not assume uniform M;
epilogue/store bounds still use the grouped problem shape.

Fixed targets, GPU 3, `--iterations=300 --warmup=20`:

```text
Shape                         Baseline us  Uniform-K us
E256_M24_N512_K4096              259.031      259.067
E32_M128_N4096_K4096             748.467      748.949
```

M24 split timing showed a diagnostic GEMM-only improvement, but it did not
survive the final standard timing:

```text
Shape                         Variant     Total us  Builder us  GEMM us
E256_M24_N512_K4096           baseline     262.499    10.795    251.704
E256_M24_N512_K4096           uniform-K    260.577    10.878    249.699
```

Decision: do not keep uniform-K as an active code path. The final standard
timings are flat on M24 and slightly worse on E32, so removing these scheduler
queries is not a reliable hot-loop win. The result suggests the compiler already
handles this path well enough, or that the extra compile-time specialization
changes scheduling/register allocation in a way that cancels the saved integer
work.

### Activation-scale TMA window reuse for TileK=256

For the E32 best config (`TileK=256`, scale group 32), one compute K tile needs
8 e8m0 activation-scale chunks, but the TMA copy is rounded to a 16-chunk
window for 16B alignment. Adjacent K tiles therefore use two halves of the same
aligned scale window. A follow-up tried to load the activation-scale TMA window
only on even K tiles and have odd K tiles read the second half from the previous
pipeline stage.

This is not a uniform-M fast path; it depends only on `TileK / scale_group` and
the 16B TMA alignment window. The implementation compiled, but the E32 standard
run hung before producing correctness output:

```text
Shape                         Result
E32_M128_N4096_K4096          hung > 60s, process killed at ~1m19s
```

Decision: revert the experiment. The current mainloop uses one pipeline barrier
for A, B, weight scale, and activation-scale transfers. Simply omitting the
activation-scale TMA on odd K tiles breaks that producer/consumer transaction
protocol. A correct version would need a separate activation-scale lifetime or
barrier/count scheme, not just a conditional TMA skip.

### Activation-scale fragment hoist

Another follow-up tried to reduce repeated activation-scale conversion in
`apply_groupwise_scale_mn()` by moving `tCrScaleN` conversion and
`scale_m * scale_n` outside the `e` dimension loop. The hypothesis was that
activation scale would be independent of the `e` coordinate for a fixed
`m/n/mma_m` coordinate.

This failed correctness immediately on the M24 fixed target:

```text
Shape                         Result
E256_M24_N512_K4096           large output/reference mismatches from first checked rows
```

Decision: revert the experiment. In the current CuTe fragment/layout mapping,
`tCrScaleN(coord)` must be indexed with the full accumulator coordinate. Even
if the logical activation-scale tile looks independent of `e`, the fragment
mapping uses `e` to select the lane/register element that matches the
corresponding accumulator value. Reusing the `e=0` scale value across `e`
corrupts scaling.

### Activation-scale TMA invariant hoist

A local mainloop experiment moved activation-scale TMA loop-invariant CuTe
objects out of the K loop: the activation-scale smem tensor, TMA slice, logical
L coordinate, and destination partition. The dataflow and number of TMA copies
were unchanged; this only tested whether repeated object/coordinate setup left
real hot-loop instructions.

```text
Shape                         Clean prebuilt us  Invariant-hoist us
E256_M24_N512_K4096              260.341             260.518
```

Decision: revert the experiment. Correctness stayed within the existing
sparse-outlier pass/fail behavior, but the timing did not improve. The compiler
already appears to eliminate most of this setup, or the hoisted objects
increase register pressure/codegen cost enough to cancel the intended saving.

## Analysis

Prebuilding only helps when it removes the full group-switch descriptor update
from the GEMM mainloop. The E32 shape benefits because it has a large output
surface and enough per-group work for descriptor-update removal to show up. The
M24 target is dominated by other costs: the descriptor builder work and remaining
mainloop overhead roughly cancel the saved descriptor rewrites.

The current path is still useful because it confirms that device-side
per-expert descriptor construction is correct and can help larger shapes. It is
not yet enough to move the M24 target below the old CMX baseline.

## Decision

Keep the full prebuilt A, B, and activation-scale descriptor path. The retained
builder constructs descriptors with separate full warps for A, B, and
activation scale, and the retained mainloop consumes prebuilt descriptors
directly without shared-memory descriptor replacement.

Do not keep the actscale-only, acquire-split, split-init, combined-release,
descriptor-prefetch, contiguous weight-scale, worker-sticky work-map, uniform-K
tile-count specialization, warp-reduce prefix, activation-scale TMA-window
reuse, activation-scale fragment hoist, or activation-scale invariant-hoist
experiments as active code paths. Earlier lane-parallel and group-count
heuristic descriptor-builder variants were also removed; only the later
whole-warp descriptor builder is retained.

Final cleanup sanity after removing the contiguous weight-scale experiment:

```text
Shape                         Avg us
E256_M24_N512_K4096           259.247
E32_M128_N4096_K4096          749.666
```

Next optimization should focus on reducing remaining mainloop overhead or using
prebuilt descriptors to simplify the hot loop without changing codegen
negatively on E32.

## Follow-up: Builder token-offset cleanup and static tile math

The full-prebuilt descriptor path does not use
`group_token_offsets`; those offsets are only needed by the
`RAGGED_TMA_DESC_ONCE_TOKEN_OFFSETS` variant. A builder cleanup therefore
conditions the offset allocation, shared-memory reduction lane, and device
stores on that macro. The same pass also added a templated builder entry point
so the work-map builder can compile tile/cluster constants into
`get_group_info()` and `make_work_tile()` instead of using runtime tile-shape
arguments.

Fixed prebuilt-desc targets, GPU 3, `--iterations=300 --warmup=20
--split_timing=true`:

```text
Shape                         Before total  Before builder  Before GEMM  New total  New builder  New GEMM
E256_M24_N512_K4096              261.926        9.908        252.017     261.566      9.312      252.254
E256_M24_N512_K4096 repeat       261.926        9.908        252.017     261.699      9.471      252.228
E32_M128_N4096_K4096             750.291        8.376        741.915     751.253      7.409      743.843
```

Correctness stayed on the same sparse low-precision policy:

```text
E256_M24_N512_K4096:   P99 846 0.03%, P98 428 0.01%, P95 175 0.01%
E32_M128_N4096_K4096: P99 4344 0.03%, P98 2144 0.01%, P95 893 0.01%
```

An ablation kept the token-offset cleanup but temporarily switched the call
site back to the runtime tile/cluster builder wrapper:

```text
Shape                         Runtime-wrapper total  Builder us  GEMM us
E256_M24_N512_K4096                    262.844        10.652    252.192
E32_M128_N4096_K4096                   751.389         7.727    743.662
```

This says the static builder path is useful for the M24 target, while the
token-offset cleanup alone is not enough to improve M24. E32 also shows a clear
builder-time improvement. Because this patch only changes the builder and the
wrapper calls into the builder, the GEMM split movement is measurement noise for
this experiment and should not be used to reject the builder-side change.

One earlier run of the static-builder version produced `259.595 us` total with
`7.750 us` builder on M24, but that did not reproduce after the same-source
rebuild and rerun. Do not use that single row as a stable conclusion.

Decision: keep this as a review candidate in the worktree because it removes a
real unused token-offset path and gives a stable builder reduction on both
retained shapes, roughly 0.4-0.6 us on M24 and about 1.0 us on E32 versus the
previous warp-parallel builder record. Do not stage it yet; it still needs code
review.

## Follow-up: Remove Ragged Experimental Code Paths

The ragged TMA experiments above did not beat the full prebuilt descriptor
path, and keeping their benchmark targets/macros in code made the active CMX
path harder to review. The cleanup keeps the experiment record in this document
only and removes the ragged benchmark targets plus the inactive ragged
activation/token-offset mainloop paths from the buildable code.

Retained fixed prebuilt-desc targets were rebuilt and rerun on GPU 3 with
`--iterations=300 --warmup=20 --split_timing=true`:

```text
Shape                         Avg us   Builder us  GEMM us   Correctness
E256_M24_N512_K4096           260.785     8.663    252.122   P99 846 0.03%, P98 428 0.01%, P95 175 0.01%
E32_M128_N4096_K4096          752.584     7.460    745.124   P99 4344 0.03%, P98 2144 0.01%, P95 893 0.01%
```

The E32 GEMM split moved versus the previous record, but this cleanup does not
change the retained non-ragged dataflow. Treat the total/GEMM delta as normal
measurement noise; the important outcome is that both retained fixed targets
still build and pass the same sparse low-precision correctness policy after the
experimental ragged paths are removed.

## Follow-up: Collapse to Full-Prebuilt Descriptor Path

After code review, the old prebuilt-descriptor feature switch was removed. The
active precomputed-scheduler targets now always use full prebuilt descriptors
when `CUTLASS_MIXED_GEMM_PRECOMPUTED_GROUP_OFFSETS=1`, and the legacy
mainloop-side shared-memory descriptor rewrite path was removed from the CMX
mainloop used by these targets.

Cleanup details:

- Removed `CUTLASS_MIXED_GEMM_PREBUILT_TMA_DESC` from the fixed target
  definitions.
- Removed the mainloop workspace descriptor allocation; `get_workspace_size()`
  is now zero for this mainloop path.
- Removed shared-memory `TensorMapStorage` descriptor fields and descriptor
  copy/update/release work from the full-prebuilt mainloop.
- Renamed `maybe_build_prebuilt_tma_descriptors()` to
  `build_prebuilt_tma_descriptors()` and removed the old null-pointer skip
  behavior from the builder.

Retained fixed targets rebuilt successfully after this cleanup:

```text
69_hopper_mxfp4_mxfp8_grouped_gemm_precomp_prebuilt_desc_coop_1x1x1_m128n32k512
69_hopper_mxfp4_mxfp8_grouped_gemm_precomp_prebuilt_desc_coop_1x1x1_m128n64k256
```

Correctness stayed on the same sparse low-precision policy:

```text
E256_M24_N512_K4096:   P99 846 0.03%, P98 428 0.01%, P95 175 0.01%
E32_M128_N4096_K4096: P99 4344 0.03%, P98 2144 0.01%, P95 893 0.01%
```

Event timing with `--iterations=300 --warmup=10 --split_timing=true` after the
cleanup was:

```text
Shape                         Total us  Builder us  GEMM us
E256_M24_N512_K4096            262.048     9.472    252.576
E32_M128_N4096_K4096           751.459     7.524    743.935
```

These event rows are sanity checks, not final NCU evidence. The useful outcome
for review is that the source now has one full-prebuilt implementation instead
of an old descriptor-update path plus optional prebuilt variants.
