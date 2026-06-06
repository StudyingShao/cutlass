# CMX Fused E8M0 Correctness Revalidation

Date: 2026-06-05
GPU: NVIDIA H20, `CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3`

## Goal

Revalidate the new CMX Humming-style fused e8m0 pre-MMA scaling path after
adding a real reference implementation. Earlier optimization runs reported
sub-140 us results on `E256_M24_N512_K4096`, but those logs did not include
reference checks. They must be treated as exploration signals, not valid
performance conclusions.

The active semantic target is:

```text
MXFP4 weight payload + Humming-style e8m0 exponent offset
FP8 activation payload
pre-MMA FP4->FP8 fused e8m0 conversion
per-token epilogue scale, including Humming residual/global scale
direct accumulator epilogue, no post-MMA intermediate scaling
```

## Humming Scale Semantics

Humming preprocessing clamps the raw e8m0 exponent range and emits a small
offset used by the fused FP4->FP8 conversion:

```text
scale_range   = min(scale_max - scale_min, 11)
scale_min_new = scale_max - scale_range
weight_scale  = max(raw_scale, scale_min_new) - scale_min_new + 1
scale_factor  = 2^(scale_min_new - 127) / 2
```

The CMX synthetic benchmark now mirrors this: raw e8m0 scales cycle through
`114..130`, producing `scale_min_new=119`, offsets in `[1,12]`, and residual
scale `1/512`. The residual is folded into the epilogue token scale so the
reference and kernel compare the same full semantics.

## Current Valid Measurements

Validation is still executed before timing even when `--compare=false` is used.

Formal 300-iteration performance runs:

```text
Shape                    Target / config                                           Avg us    Correctness
E256 M24 N512 K4096      fixed pp 1x1x1 Tile<64,32,256>                            156.9     P99/P98/P95/Humming=0
E256 M24 N512 K4096      fixed pp 1x1x1 Tile<64,32,512>                            156.0     P99/P98/P95/Humming=0
E256 M24 N512 K4096      profiler k256 best: pp 1x1x1 Tile<64,32,256>              156.750   P99/P98/P95/Humming=0
E256 M24 N512 K4096      profiler k512 best: pp 2x1x1 Tile<64,32,512>              154.607   P99/P98/P95/Humming=0
E32 M128 N4096 K4096     profiler k512 best: pp 1x1x1 Tile<64,64,512>              508.792   P99/P98/P95/Humming=0
```

After tightening pass/fail, the following 300-iteration runs remain valid:

```text
Shape                    Target / config                                           Avg us    Correctness
E256 M24 N512 K4096      profiler k512 best: pp 2x1x1 Tile<64,32,512>              154.801   all run candidates P99/P98/P95/Humming=0
E32 M128 N4096 K4096     profiler k512 best: pp 1x1x1 Tile<64,64,512>              510.099   all run candidates P99/P98/P95/Humming=0
```

Short correctness sweeps during the correctness investigation used
`--iterations=20 --warmup=5 --compare=false --explore=true`; the timing is
useful for sanity only, not final performance ranking:

```text
Shape                    Target / config                                           Avg us    Correctness
E256 M24 N512 K4096      profiler k128 best: pp 1x1x1 Tile<64,32,128>              161.800   all run candidates P99/P98/P95/Humming=0
E32 M128 N4096 K4096     profiler k256 best: pp 1x1x1 Tile<64,64,256>              509.197   all run candidates P99/P98/P95/Humming=0
E32 M128 N4096 K4096     profiler k128 best: pp 1x1x1 Tile<64,32,128>              598.630   all run candidates P99/P98/P95/Humming=0
```

## Invalid Historical Results

The old `wait1` logs reported:

```text
E256 M24 N512 K4096:
  k128 140.543 us
  k256 139.523 us
  k512 139.951 us

E32 M128 N4096 K4096:
  k128 503.178 us
  k256 503.490 us
  k512 504.649 us
```

Those logs have no `P99_error_count`, `P98_error_count`, `P95_error_count`, or
Humming tolerance output. They are no longer accepted as valid benchmark
results. Any optimization that only showed benefit in that phase must be
re-tested under the current reference.

## Root Cause

The failing configurations were not theoretical invalid shapes. They were
valid topologies that exposed an A-operand register lifetime hazard in the
prescale mainloop.

The prescale path converts FP4 to FP8 into `tCrA_mma(_,_,k_block)` and then
feeds that register fragment to WGMMA. The overlap path preloads and converts
the first A fragment of the next K tile after `warpgroup_commit_batch()` using
`warpgroup_wait<1>`. For some topologies, this overwrote A operand registers
before the prior WGMMA batch had safely consumed them.

Diagnostic evidence:

- `TileN=64,K128` failed with normal code but passed when a temporary `printf`
  was inserted inside the fused conversion. The `printf` changed scheduling and
  effectively hid the lifetime race, while making runtime unusably large.
- Constant-scale runs (`--debug_input_scale=true`) still reproduced the failure
  before the wait fix, so the problem was not e8m0 value selection.
- Explicitly validating the scale fragment showed the debug scale offsets were
  read as `1`; scale staging itself was not the root cause.
- A failed experiment that tried to retile the scale fragment to the LDSM load
  layout broke previously passing `TileN=16/32` configs, so the fix should not
  rewrite scale layout for the fast path.

Fix:

```text
Before reloading/converting A for the next K tile, use warpgroup_wait<0> for:
  - direct-accum no-scale
  - fused e8m0 cooperative schedules
  - fused e8m0 TileN >= 64 with clustered CTA
  - fused e8m0 TileN >= 64 and TileK == 128

Keep warpgroup_wait<1> for ordinary pingpong TileN=32 and for the validated
pp 1x1x1 TileN=64,TileK=512 fast path.
```

This removes the empirical topology skip from `can_implement()`. The profiler
now executes the formerly skipped legal topologies and they must pass the strict
reference gate.

## Correctness Gate Update

The benchmark now returns four device-side error counters to host:

```text
P99_error_count
P98_error_count
P95_error_count
Humming_tol_error_count
```

For direct/fused benchmark paths, all four counters must be zero for the run to
pass. Generic mixed low-precision paths keep the previous sparse-outlier policy.

The benchmark no longer accepts profiler candidates that only pass relaxed
statistics. The root-cause fix was verified with legal topologies executed, not
skipped.

Final short sweeps after the selective wait fix:

```text
Shape / target                   FAILED  SKIPPED  Best config                                      Avg us
E256 M24 N512 K4096, k128        0       0        pp 1x1x1 Tile<64,32,128> S=23                  161.577
E256 M24 N512 K4096, k512        0       0        pp 2x1x1 Tile<64,32,512> S=6                   155.233
E32 M128 N4096 K4096, k512       0       0        pp 1x1x1 Tile<64,64,512> S=4                   508.738
```

The intermediate all-`TileN>=64` wait fix also produced correct results, but it
regressed the E32 k512 best path from about `510 us` to `515 us`. The final
cluster-aware predicate restores the fast `pp 1x1x1 TileN=64,TileK=512` overlap
while still fixing the clustered large-N failures.

## Initial Ablation Recheck

After adding the strict direct/fused pass/fail gate, the previous direct
ablation targets were recompiled and re-run on:

```text
E256 M24 N512 K4096, k512, --iterations=20 --warmup=5 --explore=true --compare=false
```

Results:

```text
Target                                 Result under current gate
direct accum no-scale                  invalid: P99/P98/P95=0, but Humming_tol=6 for every config
direct accum token-scale               invalid: P99=21, Humming_tol=20 for every config
post-MMA weight-scale + token-scale    invalid target/reference state: ~95% elements fail
fused e8m0 token-scale                 valid: all run candidates P99/P98/P95/Humming=0
```

Implication: the earlier direct no-scale and direct token-scale timings are
useful only as exploratory lower bounds. They are not valid production or
apple-to-apple performance data until their reference/rounding semantics are
made exact enough to pass the current gate.

## Pair-Only 0/2 1/3 Converter

The fused e8m0 converter now uses only the explicit paired path. It requires
`ScalePackCount == 2` and an even number of `fp4x8` operands at compile time.
This matches the WGMMA A operand row-scale pattern:

```text
fp8x4 chunk:  0      1      2      3
scale:        s0     s1     s0     s1
```

So the code converts two adjacent `fp4x8` operands as a pair, using `s0` for
chunks `0/2` and `s1` for chunks `1/3`. A temporary compile-time assertion
confirmed that the E256 best topology actually instantiates this path:

```text
E256 M24 N512 K4096
pp 2x1x1 Tile<64,32,512> Stages=6
ScalePackCount == 2
DstVecCount % 2 == 0
```

The previous single-scale and generic high/low fallback source paths were
removed after verification. Unsupported future layouts should fail at compile
time instead of silently taking another conversion path.

Correctness remained exact-zero on the k512 profiler sweeps. The first pair of
numbers came from the earlier 300-iteration validation before deleting the
fallback source paths; the second pair rechecked the final pair-only source with
20 profiler iterations after the cleanup.

```text
Shape                         Best config                                Avg us    Correctness
E256 M24 N512 K4096            pp 2x1x1 Tile<64,32,512> S=6              155.382   P99/P98/P95/Humming=0
E32 M128 N4096 K4096           pp 1x1x1 Tile<64,64,512> S=4              508.582   all run candidates P99/P98/P95/Humming=0
E256 M24 N512 K4096 final      pp 2x1x1 Tile<64,32,512> S=6              154.923   all run candidates P99/P98/P95/Humming=0
E32 M128 N4096 K4096 final     pp 1x1x1 Tile<64,64,512> S=4              509.581   all run candidates P99/P98/P95/Humming=0
```

NCU and SASS did not show a material instruction-count reduction versus the
unpaired source loop. The NCU commands used the required flags
`--clock-control none --cache-control all` and filtered the main GEMM kernel
with `device_kernel.*GemmUniversal.*MixedInputPreScale`.

```text
Metric / opcode                         unpaired source     paired source
REG                                      168                 168
PRMT                                     308                 308
LOP3.LUT                                 1471                1471
IMAD                                     383                 383
IADD3                                    282                 282
integer thread inst                      628210718           628210686
bit thread inst                          138685260           138685324
GMMA warp inst                           1048576             1048576
LDSM warp inst                           524288              524288
```

Decision: keep the pair-only implementation for semantic clarity and to avoid
depending on ptxas to rediscover the row-scale pairing. Do not count it as a
performance win in benchmark summaries.

## Two-Slot Converted-A Experiment

Goal: reduce the tile-boundary A operand lifetime dependency by adding a second
converted FP8 A fragment buffer. LDSM staging stayed single-buffered; only the
converted `tCrA_mma` fragment alternated slots between K tiles. The experiment
also tried to use `warpgroup_wait<1>` before reloading A for the fused path.

Command shape:

```text
E256 M24 N512 K4096, k512
--iterations=20 --warmup=5 --compare=false --explore=true
```

Result:

```text
Implementation             Best config                              Avg us    Correctness
baseline pair-only         pp 2x1x1 Tile<64,32,512> S=6             154.923   all candidates zero
two-slot converted-A       pp 1x1x1 Tile<64,32,512> S=6             290.104   cooperative TileN 32/64 failed
```

The pingpong candidates stayed correct but slowed down heavily. Cooperative
TileN 32/64 candidates reintroduced nonzero P99/P98/P95/Humming errors. ptxas
also still injected `warpgroup.wait` around GMMA register lifetime regions.

Decision: do not keep this implementation. The extra converted-A fragment
register footprint is too expensive, and it does not safely remove the
cooperative lifetime hazard. Future work should not reattempt this exact
two-slot design unless it also changes how the compiler sees GMMA operand
liveness.

## LDSM Lookahead Experiment

NCU on the E256 fixed best config showed the main GEMM at about `153.54 us`
with `79.54%` compute throughput, `51.43%` memory throughput, and scheduler
issue limited by L1TEX scoreboard dependency:

```text
Config: E256 M24 N512 K4096, pp 2x1x1 Tile<64,32,512> S=6
NCU flags: --clock-control none --cache-control all
Duration: 153.54 us
No Eligible: 58.51%
Active Warps / Scheduler: 2.23
Eligible Warps / Scheduler: 0.46
Top stated stall: L1TEX scoreboard, about 43.3% of issue spacing
```

Experiment: increase A smem->register LDSM lookahead from two k-blocks to three
k-blocks. This did not add another converted-A fragment; it only issued LDSM
for `k+3` instead of `k+2` and preloaded k-block 2 in the tile prologue.

```text
Implementation          Avg us    Builder us    GEMM residual us    Correctness
baseline lookahead=2    155.629   4.127         151.502             P99/P98/P95/Humming=0
lookahead=3             157.652   4.143         153.509             P99/P98/P95/Humming=0
```

Decision: do not keep lookahead=3. The extra early LDSM scheduling increased
GEMM residual time by about `2 us`; the existing lookahead=2 is the better
balance for this topology.

## Direct-Smem Scale Experiment

Hypothesis: the packed `Array<e8m0,16>` scale fragment might inflate register
pressure. Try not copying the whole packed scale tile into registers; instead,
read the packed row scale from smem at FP4->FP8 convert time and index the
current `scale_idx`.

Result on the same E256 fixed config:

```text
Implementation             REG    Avg us    Builder us    GEMM residual us    Correctness
baseline scale RF copy     168    155.497   4.130         151.367             P99/P98/P95/Humming=0
direct smem scale read     168    168.093   4.135         163.958             P99/P98/P95/Humming=0
```

Decision: do not keep direct-smem scale. It did not reduce register usage and
it made the L1TEX dependency much worse by moving scale loads into the convert
critical path.

## Topology And Swizzle Recheck

After the pair-only converter cleanup and failed mainloop experiments, the
valid fixed/profiler results remain:

```text
Shape                    Ktile    Best config                                Avg us    Correctness
E256 M24 N512 K4096       128     pp 1x1x1 Tile<64,32,128> S=23              162.024   all candidates zero
E256 M24 N512 K4096       256     pp 1x1x1 Tile<64,32,256> S=12              156.987   all candidates zero
E256 M24 N512 K4096       512     pp 2x1x1 Tile<64,32,512> S=6               155.261   all candidates zero
E32 M128 N4096 K4096      256     pp 1x1x1 Tile<64,64,256> S=8               508.992   all candidates zero
E32 M128 N4096 K4096      512     pp 1x1x1 Tile<64,64,512> S=4               508.816   all candidates zero
```

Scheduler swizzle was rechecked on the E256 fixed best config:

```text
Swizzle    Avg us    Builder us    GEMM residual us    Correctness
1          155.234   3.988         151.246             P99/P98/P95/Humming=0
2          155.497   4.130         151.367             P99/P98/P95/Humming=0
4          155.488   4.136         151.351             P99/P98/P95/Humming=0
8          155.249   3.997         151.252             P99/P98/P95/Humming=0
```

The E32 k512 profiler with `--swizzle=1` regressed the best candidate to
`509.515 us` versus the default swizzle-2 result of `508.816 us`.

Decision: keep the existing default swizzle. The E256 swizzle benefit is below
`0.3 us` and not stable across the E32 shape.

## Scale Copy Ordering Experiment

Hypothesis: the boundary sequence

```text
A0 LDSM -> A1 LDSM -> scale smem-to-reg -> WGMMA wait -> convert A0
```

might leave the scale load on the convert critical path. Two alternative
orders were tested on the E256 fixed best target
`pp 2x1x1 Tile<64,32,512> S=6` with 300 timing iterations:

```text
Implementation         Avg us samples                 Builder/GEMM split            Correctness
baseline order         155.501                        4.118 / 151.383              P99/P98/P95/Humming=0
scale -> A0 -> A1      155.203, 155.230, 156.117      ~4.00-4.14 / ~151.2-152.0   P99/P98/P95/Humming=0
A0 -> scale -> A1      155.209, 155.496, 155.223      ~4.01-4.13 / ~151.2-151.4   P99/P98/P95/Humming=0
```

The same scale-first code was sanity-checked on E32 k512 profiler sweep:

```text
Shape                    Best config                                Avg us    Correctness
E32 M128 N4096 K4096      pp 1x1x1 Tile<64,64,512> S=4               509.410   all candidates zero
```

Decision: do not keep either ordering change as a performance patch. The
correctness gate passed, but the observed differences are noise-level and not
stable across repeated runs or the E32 shape.

## Exp-Offset LUT Arithmetic Experiment

Hypothesis: the fused FP4->FP8 converter computes both
`exp_offset * 0x08080800` and `exp_offset * 0x08080808` for each row scale.
Replacing this with one `exp_offset * 0x08080808` plus a mask for the low LUT
could reduce IMAD pressure.

Result on the E256 fixed best target:

```text
Implementation              Avg us, 1000 iters    GEMM residual    SASS coarse count
original arithmetic          156.289               152.106          IMAD 1851, LOP3 1471
shared offset + mask          156.308               152.136          IMAD 1805, LOP3 1567
```

The opcode count was collected with `cuobjdump --dump-sass` on the fixed target
binary, so it includes non-main helper kernels and should be treated as coarse
evidence only. The direction did reduce IMAD count, but it increased LOP3 count
and did not improve runtime.

Decision: do not keep this rewrite. The original arithmetic is clearer and
performs the same within measurement noise.

## Pre-GMMA A(k+2) LDSM Experiment

Hypothesis: keep the existing lookahead depth but issue the already-planned
`A(k+2)` LDSM before the current GMMA instead of after it:

```text
before: GMMA(k) -> LDSM A(k+2) -> convert A(k+1)
after:  LDSM A(k+2) -> GMMA(k) -> convert A(k+1)
```

Result on E256 fixed best target:

```text
Implementation              Avg us    Builder us    GEMM residual us    Correctness
baseline                     ~155.5    ~4.1          ~151.4              P99/P98/P95/Humming=0
pre-GMMA A(k+2) LDSM         156.976   3.985         152.991             P99/P98/P95/Humming=0
```

Decision: do not keep this implementation. It is correct, but moving LDSM
ahead of GMMA delays issue and increases GEMM residual by about `1.6 us`.

## Current NCU Reconfirmation

The current baseline was rebuilt after reverting the negative experiments and
rechecked on E256 fixed best:

```text
Command shape: E256 M24 N512 K4096, pp 2x1x1 Tile<64,32,512> S=6
Timing:        155.229 us total, 4.007 us builder, 151.221 us GEMM residual
Correctness:   P99/P98/P95/Humming=0
```

NCU was collected with explicit `--clock-control none --cache-control all`:

```text
Main GEMM duration                 152.90 us
Compute throughput                 79.64%
Memory throughput                  51.64%
L1/TEX hit rate                    67.77%
L2 hit rate                        20.48%
Active warps / scheduler           2.23
Eligible warps / scheduler         0.46
No eligible                        58.50%
Top stall                          L1TEX scoreboard, ~43.3% of issue spacing
```

`SourceCounters` did not provide useful source-line stall attribution for this
build; it only reported branch counters. The next real optimization likely
needs a deeper mainloop redesign that changes GMMA operand lifetime or the
load/convert dependency chain, not another local instruction reorder.

## Explicit Wait Removal Experiment

Hypothesis: the fast validated path uses explicit `warpgroup_wait<1>` before
converting the first A fragment of the next K tile, while ptxas also injects
`warpgroup.wait` around GMMA register lifetimes. Removing the explicit
`wait<1>` might leave correctness protected by compiler-inserted waits and
save a small boundary cost.

Result on E256 fixed best:

```text
Implementation              Correctness
baseline wait<1>             P99/P98/P95/Humming=0
remove explicit wait<1>      P99=259427, P98=206057, P95=135501, Humming=260761
```

Decision: keep the explicit `warpgroup_wait<1>`. The compiler-inserted wait is
not sufficient to protect the next-tile A register overwrite.

## Builder Prefix Reduction Experiment

Hypothesis: the builder CTA computes `group_start` with a shared-memory tree
reduction over 128 threads. Replacing this with a two-level warp-shuffle
reduction might reduce `__syncthreads()` cost in the small builder kernel.

The first implementation used a full warp mask while only four lanes entered
the second-level shuffle and hung. After fixing the mask semantics by having
all lanes in warp 0 participate, correctness passed:

```text
Implementation                 Avg us    Builder us    GEMM residual us    Correctness
warp-shuffle prefix, 50 iters   155.658   5.126         150.532             P99/P98/P95/Humming=0
warp-shuffle prefix, 300 iters  155.659   4.002         151.657             P99/P98/P95/Humming=0
```

Decision: do not keep this rewrite. It is more complex, had a real mask hazard
in the first version, and did not improve the stable builder-only timing over
the original shared-memory reduction.

## Compile-Time K-Block Convert and A-Reload Wait Fix

Hypothesis: the prescale mainloop still passed runtime `k_block` and
`scale_idx` values into the fused e8m0 FP4-to-FP8 conversion even though
`K_BLOCK_MAX` is compile-time known. Rewriting the K loop with
`cute::for_each` and passing `cute::Int<KBlock>` / `cute::Int<ScaleIdx>` should
let the compiler fold scale-pack indexing and reduce a small amount of
mainloop overhead without changing the pipeline.

Implementation kept:

- Static `KBlock` / `ScaleIdx` overload for
  `convert_A_kblock_fused_e8m0_pre_mma_to_slot`.
- Mainloop K-block iteration uses `cute::for_each` and compile-time GMMA A/B
  slot indexing.
- Pair FP4-to-FP8 conversion remains the only fused e8m0 conversion path.
- Correctness fix: fused pre-MMA scale now uses full `warpgroup_wait<0>`
  before reloading A when `TileN >= 128`. This protects the next-tile A
  register overwrite for wide pingpong WGMMA shapes.

The wait fix was required by this reproduced failure:

```text
Shape: E256 M24 N512 K4096, k256 sweep
Before fix:
  Pingpong C1x1x1 Tile<64,128,256> S=5
  P99=287083, P98=172304, P95=65655, Humming=275384

Dynamic-convert ablation:
  Same config still failed, so the failure was not caused by the static
  KBlock/ScaleIdx helper.

After full-wait condition:
  Same config passed with P99/P98/P95/Humming=0.
```

Performance and correctness after the kept implementation:

```text
Shape / target                         Result
E256 fixed k512 pp2x1 m64n32k512       154.152 us total, 3.992 us builder,
                                       150.160 us GEMM residual, all errors 0
E256 k512 sweep                        best 153.394 us, PP C1x1x1 Tile<64,32,512>,
                                       all printed candidates passed
E256 k256 sweep                        best 155.798 us, PP C1x1x1 Tile<64,32,256>,
                                       all printed candidates passed
E32 k512 sweep                         best 508.763 us, PP C1x1x1 Tile<64,64,512>,
                                       all printed candidates passed
```

NCU reconfirmation for the static K-block implementation on E256 fixed k512,
collected with explicit `--clock-control none --cache-control all`:

```text
Main GEMM duration                 151.49 us
Compute throughput                 80.20%
Memory throughput                  52.14%
Active warps / scheduler           2.22
Eligible warps / scheduler         0.47
No eligible                        58.20%
Top stall                          L1TEX scoreboard, ~43.5% of issue spacing
```

Decision: keep this change. It gives a small but measurable main-GEMM
improvement on the fixed E256 path, preserves E32 performance, and fixes an
actual profiler correctness hole for wide pingpong A operand reload.

## Full-Wait Safety Recheck and A-Register Ring Experiment

After review, the `TileN >= 128` full-wait condition was not a strong enough
code-level guarantee. The unsafe operation is K-tile boundary A operand reuse:
after `warpgroup_commit_batch`, the next tile converts into
`tCrA_mma(_,_,0)`. If the code only executes `warpgroup_wait<1>`, one committed
WGMMA group is still allowed to be outstanding, so overwriting slot 0 cannot be
proven safe from the source code alone.

Conservative safety baseline:

```text
Change:       Always use warpgroup_wait<0> before converting next-tile A slot 0.
Shape:        E256 M24 N512 K4096, fixed pp2x1 Tile<64,32,512> S=6
Correctness:  P99/P98/P95/Humming=0
Timing run 1: 162.624 us total, 4.807 us builder, 157.817 us GEMM residual
Timing run 2: 162.565 us total, 4.745 us builder, 157.820 us GEMM residual
```

This version is code-level safe but costs about `7.7 us` in main GEMM residual
relative to the previous 154 us result.

A-register ring experiment:

```text
Idea:         Allocate two complete tCrA_mma register fragments and alternate
              them across K tiles. Keep wait<1>, but write the other A buffer,
              so one outstanding WGMMA group cannot observe overwritten A regs.
Correctness:  P99/P98/P95/Humming=0
Timing:       267.067 us total, 4.961 us builder, 262.106 us GEMM residual
```

Decision: do not keep the full A-register ring. It is correct, but the doubled
A operand fragment creates too much register/scheduling pressure and is far
slower than even the conservative full-wait baseline.

## Per-GMMA Commit Experiment

Hypothesis: instead of committing all GMMAs in one K tile as one group and then
using `warpgroup_wait<0>` before reusing `tCrA_mma(_,_,0)`, commit each GMMA as
its own group. After each commit, use a rolling
`warpgroup_wait<min(K_BLOCK_MAX - 1, 7)>`. This should retire the oldest group
that reads A operand slot 0 while allowing later K-block GMMAs to remain
outstanding.

Result on E256 fixed k512 pp2x1 Tile<64,32,512> S=6:

```text
Correctness:  P99/P98/P95/Humming=0
Timing:       197.439 us total, 3.987 us builder, 193.452 us GEMM residual
ptxas:        inserted warpgroup.wait and warned that wgmma.mma_async
              instructions were serialized because non-WGMMA instructions
              define input registers inside the pipeline stage.
```

Decision: do not keep the per-GMMA commit form as implemented. The source-level
safety idea is valid, but the compiler serialized enough of the WGMMA pipeline
that it is much slower than the conservative full-wait baseline. A coarser
commit group, such as 2 or 4 GMMAs per group, may still be worth testing because
it can drain the reused A slots with fewer committed groups and fewer waits.

## Grouped GMMA Commit Experiment

Hypothesis: commit a small fixed number of GMMAs together instead of either the
whole K tile or each individual GMMA. A group size of 4 drains the first four A
operand slots at the K-tile boundary, which is enough for the next tile's early
converts, while preserving later K-block GMMAs in flight and avoiding the
per-GMMA commit/wait overhead.

Implementation:

```text
K_COMMIT_GROUP_SIZE = 4
K_COMMIT_GROUPS     = ceil(K_BLOCK_MAX / 4)
K_WAIT_MAX          = min(K_COMMIT_GROUPS - 1, 7)

After every fourth GMMA, or at the last K block:
  warpgroup_commit_batch()
  warpgroup_wait<K_WAIT_MAX>()
```

E256 fixed k512 pp2x1 Tile<64,32,512> S=6:

```text
Run 1 correctness: P99/P98/P95/Humming=0
Run 1 timing:      143.495 us total, 4.125 us builder, 139.370 us GEMM residual

Run 2 correctness: P99/P98/P95/Humming=0
Run 2 timing:      143.671 us total, 4.047 us builder, 139.624 us GEMM residual

Long run correctness: P99/P98/P95/Humming=0
Long run timing:      148.815 us total, 4.235 us builder, 144.580 us GEMM residual
Command context:      --warmup=50 --iterations=1000
```

E256 k256 short sweep, `--explore=true --warmup=5 --iterations=20`:

```text
All printed candidates had P99/P98/P95/Humming=0, including the previously
failing Pingpong Shape<_1,_1,_1> Tile<64,128,256> S=5 candidate.
Short-sweep best: 168.285 us, Pingpong Shape<_1,_1,_1> Tile<64,32,256> S=12.
```

E256 k256 repeated sweep, `--explore=true --warmup=10 --iterations=50`:

```text
All printed candidates again had P99/P98/P95/Humming=0.
Repeated-sweep best: 168.080 us, Pingpong Shape<_1,_1,_1> Tile<64,32,256> S=12.
```

E32 k512 short sweep, `--explore=true --warmup=5 --iterations=20`:

```text
Shape:            E32 M128 N4096 K4096
Correctness:      all printed candidates had P99/P98/P95/Humming=0
Short-sweep best: 509.056 us, Pingpong Shape<_2,_1,_1> Tile<64,64,512> S=4
Historical note:  previous short-sweep best was 508.763 us, so this is
                  effectively unchanged for the large-N E32 shape.
```

Decision: keep the grouped GMMA commit mainloop. It is both source-level safe
and substantially faster than the full-wait safety baseline on the main E256
fixed config.

## Cleanup Decision

After the correctness revalidation, the direct-accum no-scale and direct-accum
token-scale targets remain experiment history only. They were useful for
isolating scale overhead, but they are not the production Humming-style path and
should not stay in the code as separate benchmark targets.

Code cleanup:

```text
Removed code paths:
  CUTLASS_MIXED_GEMM_DIRECT_ACCUM_NO_SCALE
  direct_accum_noscale benchmark targets
  direct_accum_token_scale benchmark targets
  post-MMA weight-scale + token-scale comparison targets
  direct no-scale reference kernel

Renamed token-scale selector:
  old: CUTLASS_MIXED_GEMM_DIRECT_ACCUM_TOKEN_SCALE
  new: CUTLASS_MIXED_GEMM_EPILOGUE_TOKEN_SCALE

Reason:
  the token scale is an epilogue policy, not a direct-accum mainloop mode.
```

The retained source path is the Humming-style semantic path:

```text
CUTLASS_MIXED_GEMM_FUSED_E8M0_PRE_MMA_SCALE
CUTLASS_MIXED_GEMM_EPILOGUE_TOKEN_SCALE
```

The epilogue callback file only exposes the per-token scale operation needed by
this path. The unused pointer-array scalar scale callback was removed during
review cleanup so the new header does not advertise an unverified side path.

## Post-Cleanup Review Recheck

After removing experiment targets and tightening the epilogue callback
arguments, the code was rebuilt and rechecked on the two review shapes.

```text
Shape                    Target / config                                Avg us    Correctness
E256 M24 N512 K4096      fixed pp 2x1x1 Tile<64,32,512> S=6              143.487   P99/P98/P95/Humming=0
E32 M128 N4096 K4096     profiler k512 best pp 2x1x1 Tile<64,64,512> S=4 508.622   all printed candidates P99/P98/P95/Humming=0
```

Build/test commands:

```bash
cmake --build build -j --target 69_hopper_mxfp4_fp8_grouped_gemm_fused_e8m0_token_scale_pp_2x1x1_m64n32k512
CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
  ./build/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_mxfp4_fp8_grouped_gemm_fused_e8m0_token_scale_pp_2x1x1_m64n32k512 \
  --groups=256 --m=24 --n=512 --k=4096 --warmup=20 --iterations=300 --split_timing=true

cmake --build build -j --target 69_hopper_mxfp4_fp8_grouped_gemm_fused_e8m0_token_scale_k512
CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
  ./build/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_mxfp4_fp8_grouped_gemm_fused_e8m0_token_scale_k512 \
  --groups=32 --m=128 --n=4096 --k=4096 --warmup=5 --iterations=20 --explore=true --split_timing=true
```

## Required Next Steps

1. Keep fused benchmark paths on exact-zero pass/fail for all reported
   performance numbers. Performance results must clearly separate exact-zero,
   Humming-tolerance-zero, and relaxed sparse-outlier cases.

2. Re-run ablation under the current reference.
   Earlier optimizations must be reclassified as:

```text
valid positive: correctness unchanged and speed improves
neutral/noise: correctness unchanged and speed is noise-level
invalid: performance came from incorrect or unsupported configurations
```

3. Do not cite the sub-140 us numbers as achieved performance until they are
   reproduced with the current reference and all selected configs pass.
