# 2026-06-11 CMX Ktile-Independent Weight Scale Final Status

## Goal

Record the final status of the CMX Ktile-independent weight-scale work before
commit.

The production requirement is:

```text
One physical weight-scale layout must support profiler-selected Ktile 128/256/512.
Runtime must not require one repacked scale tensor per Ktile.
```

This document focuses on the tradeoff against the original implementation,
because future optimization work needs to know which regressions are real and
which are expected consequences of removing the TileK-specific packed scale
layout.

## Machine And Measurement Context

Current measurements were taken on H20:

```text
GPU: CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3
Postscale command:
  --warmup=20 --iterations=300 --explore=true --compare=false --split_timing=false
Prescale command:
  --warmup=20 --iterations=300 --explore=true --compare=true --split_timing=true
```

Raw sweep artifacts are local under:

```text
cutlass_mixed_gemm/.claude/optimization-logs/2026-06-11-cmx-final-sweep/
```

The final summary files from that run are:

```text
final_sweep_summary.md
final_sweep_comparison.md
final_config_comparison.csv
```

The current final sweep re-ran only the current implementation.  Original and
previous values were loaded from existing historical CSV files.

## Original Implementation

The original scale layout was TileK-packed.  For a fixed kernel candidate, the
scale tensor was effectively packed for that candidate's `TileShapeK`:

```text
packed scale tile:
  TileM x (TileK / group_size)
```

This is efficient for the selected kernel because the producer can load exactly
the scale region consumed by one CTA with a compact contiguous transfer, and
the consumer-side scale fragment shape matches the converter's expectation.

The cost is that preprocessing depends on `TileShapeK`.  That is not acceptable
for production profiler-map dispatch:

- the best config map can contain Ktile 128, 256, and 512 for different runtime
  shapes;
- model loading cannot know which future runtime shape will select which Ktile;
- storing one packed scale copy per Ktile increases model memory;
- a super-K packed format increases bytes read for smaller Ktiles.

So the original implementation is a good fixed-topology performance reference,
but it is not a production-friendly scale contract when profiler freedom is
required.

## Current Implementation

The committed design uses a Ktile-independent folded scale layout.

Preprocessing folds every fixed `64x128` weight-coordinate scale block:

```text
raw e8m0 scale bytes:
  64 x (128 / 32) = 64 x 4

folded e8m0 scale bytes:
  16 x 16
```

The fold splits the M64 block into four M16 warp-local slices:

```text
fold_col = warp_slice * 4 + kg4
row      = m16
```

At runtime, the same folded physical scale tensor supports k128/k256/k512:

```text
copy bytes per CTA scale stage = TileM * (TileK / 32)
copy footprint                 = folded M16 x folded K16 x (TileM / 64) x (TileK / 128)
```

The mainloop consumes the copied scale through a logical expanded view whose
scale K mode advances every 32 original K elements.

This implementation is now applied to both relevant CMX families:

- prescale / Humming-style pre-MMA e8m0 scaling path;
- postscale mixed-input path for int4 x fp8, mxfp4 x bf16, mxfp4 x fp8, and
  mxfp4 x mxfp8.

Current implementation constraints are intentional and match the profiler
candidate set:

```text
TileM % 64 == 0
TileK % 128 == 0
K     % 128 == 0
```

No extra user-visible requirement such as `M % 256 == 0` should be introduced.
The required M/N/K tile divisibility remains the normal kernel-candidate
contract.

## Final H20 Best-Config Result

Final sweep result:

```text
runs                       30
nonzero status             0
current skipped/failed     0
matched configs            1084
unmatched historical       8
unmatched current          12
```

Best-config comparison:

```text
kind       path                               shape                     orig_us    prev_us    curr_us   vs_orig   vs_prev current best
--------------------------------------------------------------------------------------------------------------------------------------
postscale  int4_fp8                           e256_m24_n512_k4096       190.025    182.769    180.334    -5.10%    -1.33% k512 Coop Shape<_2,_1,_1> Tile<_128,_32,_512> S=4
postscale  int4_fp8                           e32_m128_n4096_k4096      567.175    566.000    565.219    -0.34%    -0.14% k128 Coop Shape<_2,_1,_1> Tile<_128,_128,_128> S=7
postscale  mxfp4_bf16                         e256_m24_n512_k4096       344.971    354.406    352.720     2.25%    -0.48% k256 Coop Shape<_1,_1,_1> Tile<_128,_32,_256> S=6
postscale  mxfp4_bf16                         e32_m128_n4096_k4096     1170.430   1148.970   1148.080    -1.91%    -0.08% k128 Coop Shape<_1,_1,_1> Tile<_128,_64,_128> S=7
postscale  mxfp4_fp8                          e256_m24_n512_k4096       222.942    227.606    223.767     0.37%    -1.69% k128 Coop Shape<_2,_1,_1> Tile<_128,_32,_128> S=15
postscale  mxfp4_fp8                          e32_m128_n4096_k4096      678.972    696.034    696.331     2.56%     0.04% k256 Coop Shape<_2,_1,_1> Tile<_128,_64,_256> S=6
postscale  mxfp4_mxfp8                        e256_m24_n512_k4096       250.251    245.432    243.566    -2.67%    -0.76% k512 Coop Shape<_2,_1,_1> Tile<_128,_32,_512> S=4
postscale  mxfp4_mxfp8                        e32_m128_n4096_k4096      743.908    741.744    741.030    -0.39%    -0.10% k256 Coop Shape<_1,_2,_1> Tile<_128,_64,_256> S=5
prescale   mxfp4_fp8_fused_e8m0_token_scale   e256_m24_n512_k4096       143.119    145.750    146.254     2.19%     0.35% k512 PP   Shape<_2,_1,_1> Tile<_64,_32,_512>  S=6
prescale   mxfp4_fp8_fused_e8m0_token_scale   e32_m128_n4096_k4096      508.288    507.362    507.599    -0.14%     0.05% k512 PP   Shape<_1,_2,_1> Tile<_64,_64,_512>  S=4
```

Best-config conclusion:

- the final implementation is within noise of the previous optimized version;
- no best config regressed by more than about 2.6% against original on H20;
- the primary remaining concern is not best-config performance, but config-level
  outliers that are still slower than the original TileK-packed layout.

## Config-Level Status

Config-level summary:

```text
matched configs                  1084
slower than previous by >5%      0
slower than previous by >1%      17
faster than previous by >5%      40
faster than previous by >1%      96
worst slower vs previous         +2.95%
best faster vs previous          -60.25%

slower than original by >5%      50
slower than original by >1%      291
faster than original by >5%      97
faster than original by >1%      285
worst slower vs original         +17.72%
best faster vs original          -96.38%
```

## Current vs Original Data

The full config-level comparison contains 1084 matched rows.  The local CSV is:

```text
cutlass_mixed_gemm/.claude/optimization-logs/2026-06-11-cmx-final-sweep/final_config_comparison.csv
```

The aggregate current-vs-original distribution by path and shape is:

```text
kind       path                               shape                   rows  slow>10  slow>5  fast>5  worst_slow  best_fast
--------------------------------------------------------------------------------------------------------------------------
postscale  int4_fp8                           e256_m24_n512_k4096      110        3       4       8     +14.49%   -14.33%
postscale  int4_fp8                           e32_m128_n4096_k4096     110        3       8       9     +14.84%   -14.00%
postscale  mxfp4_bf16                         e256_m24_n512_k4096      100        0       4       5      +8.74%   -16.16%
postscale  mxfp4_bf16                         e32_m128_n4096_k4096     100        0       3       5      +6.43%   -17.45%
postscale  mxfp4_fp8                          e256_m24_n512_k4096      112        5       8       8     +16.16%   -21.02%
postscale  mxfp4_fp8                          e32_m128_n4096_k4096     112        5      12       8     +17.72%   -18.17%
postscale  mxfp4_mxfp8                        e256_m24_n512_k4096      108        0       3      16      +8.17%   -96.22%
postscale  mxfp4_mxfp8                        e32_m128_n4096_k4096     108        0       1      21      +5.89%   -96.38%
prescale   mxfp4_fp8_fused_e8m0_token_scale   e256_m24_n512_k4096      112        1       3       8     +14.50%   -10.48%
prescale   mxfp4_fp8_fused_e8m0_token_scale   e32_m128_n4096_k4096     112        1       4       9     +15.00%   -10.28%
```

All configs slower than original by more than 10% are listed below.  None of
these is the current best config for the two primary shapes, and none is slower
than the previous optimized folded-scale sweep by more than 1%.

```text
kind       path                               shape                     k    orig_us    curr_us   vs_orig   vs_prev config
-----------------------------------------------------------------------------------------------------------------------------------
postscale  mxfp4_fp8                          e32_m128_n4096_k4096    256   1888.210   2222.890   +17.72%    -1.59% PP   Shape<_1,_1,_1> Tile<_64,_128,_256> S=5
postscale  mxfp4_fp8                          e256_m24_n512_k4096     512   1481.110   1720.460   +16.16%    -1.03% Coop Shape<_2,_2,_1> Tile<_128,_64,_512> S=3
postscale  mxfp4_fp8                          e32_m128_n4096_k4096    512   1480.390   1708.810   +15.43%    -2.34% Coop Shape<_2,_2,_1> Tile<_128,_64,_512> S=3
prescale   mxfp4_fp8_fused_e8m0_token_scale   e32_m128_n4096_k4096    256   1205.330   1386.160   +15.00%    -1.55% Coop Shape<_1,_2,_1> Tile<_256,_128,_256> S=2
postscale  mxfp4_fp8                          e256_m24_n512_k4096     512    366.629    421.580   +14.99%    -1.99% Coop Shape<_2,_1,_1> Tile<_128,_64,_512> S=3
postscale  int4_fp8                           e32_m128_n4096_k4096    256   6332.570   7272.290   +14.84%    +0.03% Coop Shape<_2,_2,_1> Tile<_256,_128,_256> S=2
prescale   mxfp4_fp8_fused_e8m0_token_scale   e256_m24_n512_k4096     256   1205.840   1380.720   +14.50%    -1.73% Coop Shape<_1,_2,_1> Tile<_256,_128,_256> S=2
postscale  int4_fp8                           e256_m24_n512_k4096     256   6330.410   7247.860   +14.49%    -0.00% Coop Shape<_2,_2,_1> Tile<_256,_128,_256> S=2
postscale  mxfp4_fp8                          e256_m24_n512_k4096     512    679.795    777.633   +14.39%    -1.91% Coop Shape<_1,_2,_1> Tile<_128,_64,_512> S=3
postscale  mxfp4_fp8                          e32_m128_n4096_k4096    512    684.902    774.856   +13.13%    -2.16% Coop Shape<_2,_1,_1> Tile<_128,_64,_512> S=3
postscale  mxfp4_fp8                          e256_m24_n512_k4096     256   2121.380   2396.800   +12.98%    -2.44% PP   Shape<_1,_1,_1> Tile<_64,_128,_256> S=5
postscale  int4_fp8                           e256_m24_n512_k4096     256   3013.160   3401.040   +12.87%    -0.01% Coop Shape<_1,_2,_1> Tile<_256,_128,_256> S=2
postscale  mxfp4_fp8                          e32_m128_n4096_k4096    512    678.972    765.621   +12.76%    -3.93% Coop Shape<_1,_2,_1> Tile<_128,_64,_512> S=3
postscale  int4_fp8                           e32_m128_n4096_k4096    256   3020.710   3394.960   +12.39%    -0.01% Coop Shape<_1,_2,_1> Tile<_256,_128,_256> S=2
postscale  mxfp4_fp8                          e32_m128_n4096_k4096    512    726.675    810.280   +11.51%   -10.20% PP   Shape<_1,_1,_1> Tile<_64,_64,_512> S=4
postscale  int4_fp8                           e32_m128_n4096_k4096    256   1555.040   1728.930   +11.18%    +0.01% Coop Shape<_2,_1,_1> Tile<_256,_128,_256> S=2
postscale  int4_fp8                           e256_m24_n512_k4096     256   1560.630   1731.120   +10.92%    +0.06% Coop Shape<_2,_1,_1> Tile<_256,_128,_256> S=2
postscale  mxfp4_fp8                          e256_m24_n512_k4096     512    383.566    424.984   +10.80%    -9.72% PP   Shape<_1,_1,_1> Tile<_64,_64,_512> S=4
```

The current-vs-original data is mixed rather than one-sided.  There are also
many configs faster than original, especially in mxfp4 x mxfp8 and some
mxfp4 x fp8 / mxfp4 x bf16 families.  This is why the remaining decision should
be based on selected best configs plus known outlier families, not on a single
global average.

Important interpretation:

```text
The remaining large gaps are against original, not against the previous
optimized folded-scale sweep.
```

For configs slower than original by more than 10%:

```text
count >10% vs original           18
>5% slower vs previous            0
>1% slower vs previous            0
current-best configs              0
```

The >10% vs original rows cluster in a small number of topology families:

```text
path                              pattern                                      count
------------------------------------------------------------------------------------
int4_fp8                          k256 Coop Tile<256,128,256> S=2                6
mxfp4_fp8                         k512 Coop Tile<128,64,512>  S=3                6
mxfp4_fp8                         k256 PP   Tile<64,128,256> S=5                2
mxfp4_fp8                         k512 PP   Tile<64,64,512>  S=4                2
prescale fused token scale        k256 Coop Tile<256,128,256> S=2                2
```

These configs are not selected as current best on the two primary shapes.  They
remain documented because other shapes may choose a different topology.

## Why The Outliers Remain

The folded layout removes the TileK dependency, but it also changes the scale
producer and consumer contract:

- original: scale layout is already packed exactly for the candidate TileK;
- current: scale layout is fixed and must be interpreted through the folded
  M16/K16 physical layout and a consumer-expanded logical view;
- original can be locally optimal for a fixed topology;
- current is globally usable across Ktile choices.

The worst remaining families share two traits:

1. they are sensitive to scale load / scale extract latency because the
   mainloop has little slack, especially `Stages=2` or narrow-N cases;
2. they benefit more from the original candidate-specific packed scale layout
   than from the folded layout.

We already optimized the folded layout enough to remove the severe previous
outliers.  The final H20 sweep has no >5% slowdown against the immediately
previous optimized folded implementation.  Further work should therefore be a
targeted scale-loader policy for the listed topology families, not another
global folded-layout rewrite.

## H200 Retest Guidance

H200 may shift the tradeoff.  The current H20 result should not be treated as
the final hardware-independent conclusion.

Retest on H200 with the same command shape:

```bash
CUDA_VISIBLE_DEVICES=<h200_gpu> CLOCK_GPU_INDEX=<h200_gpu> \
./build/examples/69_hopper_mixed_dtype_grouped_gemm/<target> \
  --groups=<E> --m=<per_expert_m> --n=<N> --k=<K> \
  --warmup=20 --iterations=300 --explore=true ...
```

Use the same path split:

```text
postscale:
  --compare=false --split_timing=false

prescale:
  --compare=true --split_timing=true
```

Recommended H200 checks:

1. Re-run the two primary shapes:

```text
E256_M24_N512_K4096
E32_M128_N4096_K4096
```

2. Compare best-config performance first.  If best configs remain within a few
   percent of original, the production tradeoff is likely acceptable.

3. Then inspect only the known outlier families:

```text
int4_fp8                   k256 Coop Tile<256,128,256> S=2
mxfp4_fp8                  k512 Coop Tile<128,64,512>  S=3
mxfp4_fp8                  k256 PP   Tile<64,128,256> S=5
mxfp4_fp8                  k512 PP   Tile<64,64,512>  S=4
prescale fused token scale k256 Coop Tile<256,128,256> S=2
```

4. If H200 reduces the outlier gap, the issue is likely dominated by H20 memory
   pipeline / scoreboard sensitivity.  If H200 keeps the same topology pattern,
   the issue is likely in the scale SMEM-to-RF layout and converter input path.

5. For final H200 optimization evidence, NCU runs must explicitly include:

```bash
--clock-control none --cache-control all
```

Use NCU only on the selected outlier family and its original counterpart, not on
the full sweep.

## Future Optimization Entry Points

Do not reintroduce Ktile-specific scale preprocessing unless the production
contract changes.

Possible targeted fixes:

1. Add topology-specific folded-scale consumer policies only for the outlier
   families.  The first candidates are:

```text
TileM=256, TileN=128, TileK=256, Stages=2
TileM=128, TileN=64,  TileK=512, Stages=3
TileM=64,  TileN=128, TileK=256, Stages=5
TileM=64,  TileN=64,  TileK=512, Stages=4
```

2. Compare scale SMEM-to-RF copy shape against the original
   `Array<ue8m0, TileK/32>` fragment shape.  The goal is to reduce extract/index
   dependency chains without increasing register pressure enough to hurt best
   configs.

3. If H200 still shows the same gap, use NCU to compare:

```text
long_scoreboard
smem/shared-load issue rate
register move / integer instruction count around scale extract
WGMMA issue spacing
```

4. Keep profiler behavior unchanged.  Bad performance on a shape is not a
   reason to skip a config.  Only skip configs that are illegal or fail
   correctness.

## Decision

Keep the current Ktile-independent folded scale implementation.

Rationale:

- it satisfies the production requirement of one scale layout for k128/k256/k512;
- all final H20 sweep runs completed with zero current skipped/failed configs;
- best-config performance is stable against the previous optimized sweep;
- remaining >10% gaps against original are config-level outliers, not selected
  best configs on the primary shapes;
- the outliers are documented with enough topology detail to guide a later H200
  retest and targeted optimization pass.
