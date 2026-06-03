# CMX Best Config Cache

**Last Updated**: 2026-06-02
**Status**: Active CMX optimization reference
**Source sweep**:
`.claude/optimization-logs/2026-05-21-cmx-four-impl-best-configs.md`
**Current m=8 retest**: 2026-05-26 device precomputed tile-map scheduler path

## Purpose

This document is the stable lookup table for known CMX profiler best configs.
Use it before running any CMX single-shape performance measurement.

The detailed 2026-05-21 log preserves the sweep summary and points to the local
raw-log artifact directory when present. This cache preserves the decision:
which K-tile and CUTLASS config should be used for known production comparison
shapes.

## Measurement Rule

Do not run `--explore=true` just to measure a known shape.

Use `--explore=true` only when one of these is true:

- The `(implementation, FC, M, N, K, groups)` tuple is not in this cache.
- The profiler candidate space changed, such as tile shapes, cluster shapes,
  schedules, or stage policy.
- A fixed-config retest disagrees with the cache by enough to suggest a real
  code or environment shift.
- The task is explicitly a final best-config discovery or a comparison where
  the best config itself is unknown.

For routine regression checks and single-case current-performance questions,
first look up the cached config. If the repository does not yet expose a
single-config target for that cached config, add or use such a target instead
of sweeping the full profiler set again.

## Scope

The current cache covers uniform grouped runs on GPU 3:

```text
groups = 128
FC1    = M x N=1024 x K=4096
FC2    = M x N=4096 x K=512
M      = 1, 2, 4, 8, 12, 18, 22
```

It also includes the MoE topk-derived shape measured on 2026-05-27:

```text
global M = 1024, E = groups = 256, topk = 6
per-expert m = 1024 * 6 / 256 = 24
N = 512, K = 4096

global M = 4096, E = groups = 32
per-expert m = 4096 / 32 = 128
N = 4096, K = 4096
```

Source sweep command pattern:

```bash
CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
  ./build/examples/69_hopper_mixed_dtype_grouped_gemm/${target}_k${ktile} \
  --explore=true --m=${m} --n=${n} --k=${k} --groups=128 \
  --iterations=100 --warmup=10 --compare=false
```

## MXFP4 x MXFP8

Implementation target family:
`69_hopper_mxfp4_mxfp8_grouped_gemm_k{128,256,512}`.

The 2026-05-21 numbers include runtime activation-scale pack in the timed
region. The 2026-05-25 direct raw activation-scale TMA retest at commit
`05b8c595` is preserved below as an intermediate current-status check. The
same-day sign-preprocessed FP4->FP8 retest is now the active current status.

### FC1: N=1024 K=4096

```text
M   Ktile  Best config                                      Sweep us  Current us
1   512    Coop C1x1x1 Tile 128x16x512 Stages=5             232.660   -
2   512    Coop C1x1x1 Tile 128x16x512 Stages=5             232.959   -
4   512    Coop C1x1x1 Tile 128x16x512 Stages=5             231.854   -
8   512    Coop C1x1x1 Tile 128x16x512 Stages=5             232.565   188.076
12  512    Coop C1x1x1 Tile 128x16x512 Stages=5             232.212   -
18  512    Coop C2x1x1 Tile 128x32x512 Stages=4             310.526   -
22  512    Coop C1x1x1 Tile 128x32x512 Stages=4             311.540   -
```

### FC2: N=4096 K=512

```text
M   Ktile  Best config                                      Sweep us  Current us
1   128    Coop C1x1x1 Tile 128x16x128 Stages=19            222.778   -
2   256    Coop C1x1x1 Tile 128x16x256 Stages=9             223.204   -
4   128    Coop C1x1x1 Tile 128x16x128 Stages=19            223.200   -
8   128    Coop C1x1x1 Tile 128x16x128 Stages=19            223.667   188.426
12  256    Coop C1x1x1 Tile 128x16x256 Stages=9             222.522   -
18  128    Coop C1x1x1 Tile 128x32x128 Stages=15            262.028   -
22  128    Coop C1x1x1 Tile 128x32x128 Stages=15            262.848   -
```

## Reuse Rules

For `mxfp4 x mxfp8`:

```text
FC1, M <= 12: use k512, Coop C1x1x1, Tile 128x16x512.
FC1, M = 18: use k512, Coop C2x1x1, Tile 128x32x512.
FC1, M = 22: use k512, Coop C1x1x1, Tile 128x32x512.

FC2, M in {1,4,8}:  use k128, Coop C1x1x1, Tile 128x16x128.
FC2, M in {2,12}:   use k256, Coop C1x1x1, Tile 128x16x256.
FC2, M in {18,22}:  use k128, Coop C1x1x1, Tile 128x32x128.
```

The immediate m=8 focus cases are therefore:

```text
FC1 m=8:  k512, Coop C1x1x1, Tile 128x16x512, Stages=5.
FC2 m=8:  k128, Coop C1x1x1, Tile 128x16x128, Stages=19.
```

Dedicated single-config targets exist for these two rows:

```bash
cmake --build build -j --target \
  69_hopper_mxfp4_mxfp8_grouped_gemm_fc1_m8_best \
  69_hopper_mxfp4_mxfp8_grouped_gemm_fc2_m8_best

CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
  ./build/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_mxfp4_mxfp8_grouped_gemm_fc1_m8_best \
  --m=8 --n=1024 --k=4096 --groups=128 \
  --iterations=300 --warmup=50 --compare=false

CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
  ./build/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_mxfp4_mxfp8_grouped_gemm_fc2_m8_best \
  --m=8 --n=4096 --k=512 --groups=128 \
  --iterations=300 --warmup=50 --compare=false
```

These commands intentionally do not pass `--explore=true`.

Direct raw activation-scale, pre-sign-preprocessing no-sweep check on 2026-05-25
(`--iterations=300 --warmup=50`):

```text
FC1 m=8 single-config target: 225.690 us, 38060.8 GFLOPS
FC2 m=8 single-config target: 219.341 us, 19581.3 GFLOPS
```

Sign-preprocessed FP4->FP8 follow-up on 2026-05-25, same fixed targets and
same command shape, no sweep:

```text
FC1 m=8 runs: 207.509 us / 41395.4 GFLOPS, 205.697 us / 41760.1 GFLOPS
FC2 m=8 runs: 210.723 us / 20382.1 GFLOPS, 212.503 us / 20211.3 GFLOPS
```

Compile-time fixed scheduler upper-bound experiment on 2026-05-26, same fixed
targets and same command shape, no sweep. This path is not active because it
assumes uniform per-expert token counts:

```text
FC1 m=8 runs: 181.267 us / 47388.3 GFLOPS, 181.215 us / 47401.9 GFLOPS,
              181.304 us / 47378.7 GFLOPS
FC2 m=8 runs: 168.339 us / 25513.7 GFLOPS, 166.834 us / 25743.9 GFLOPS,
              170.062 us / 25255.4 GFLOPS
```

Device precomputed tile-map scheduler check on 2026-05-26, same fixed targets,
no sweep (`--iterations=200 --warmup=50`). This is the active CMX scheduler
path for the m=8 fixed targets:

```text
FC1 m=8: 185.760 us, 46242.2 GFLOPS
FC2 m=8: 185.291 us, 23179.6 GFLOPS
```

Variable-M descriptor correctness fix retest on 2026-05-26, same fixed targets,
no sweep (`--iterations=100 --warmup=20`). This is the current correctness-fixed
status:

```text
FC1 m=8: 188.076 us, 45672.6 GFLOPS, P99 309 0.03%
FC2 m=8: 188.426 us, 22793.9 GFLOPS, P99 626 0.01%
```

Implementation note:
`cmx-fp4-fp8-sign-preprocessing.md` records the offline weight sign layout and
the converter SASS evidence. The cached best config did not change; this was a
fixed-config retest, not a profiler sweep.

Scheduler note:
`.claude/optimization-logs/2026-03-19-scheduler-work-switching.md` records the
2026-05-26 scheduler addenda. The active path uses a device precompute kernel
to build `tile_map[linear_idx] = (group, local_tile)` and avoids the generic
online group-prefix walk. Do not re-run `--explore=true` for this question.

### MoE topk shape: M=1024 E=256 N=512 K=4096 topk=6

Measured on 2026-05-27 with current CMX profiler targets:

```bash
CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
  ./build/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_mxfp4_mxfp8_grouped_gemm_k${ktile} \
  --m=24 --n=512 --k=4096 --groups=256 \
  --iterations=300 --warmup=50 --compare=false --explore=true
```

Profiler sweep result:

```text
Ktile  Avg us   GFLOPS   Best config
128    340.801  75615.5  Coop C2x1x1 Tile 128x32x128 Stages=15
256    332.813  77430.2  Coop C1x1x1 Tile 128x32x256 Stages=7
512    326.535  78918.9  Coop C1x1x1 Tile 128x32x512 Stages=4
```

Current best config:

```text
mxfp4 x mxfp8, groups=256, m=24, n=512, k=4096:
  use k512, KernelPtrArrayTmaWarpSpecializedCooperative,
  Shape<_1,_1,_1>, Shape<_128, _32, _512>, Stages=4.
```

Fixed-target prebuilt TMA descriptor check on 2026-06-02, no sweep
(`--iterations=300 --warmup=20`, scheduler/builder included):

```text
baseline precomputed path:             262.168 us
prebuilt B + activation-scale desc:    262.920 us
skip unused template desc init:        259.711 us, 259.975 us
add prebuilt weight/A desc:            259.434 us
init-time weight/A desc pointer:       259.276 us
```

For this M24 target, the prebuilt descriptor path only became a speedup after
skipping unused B and activation-scale template descriptor copies in
`tensormaps_init()`. Prebuilding the grouped weight/A descriptor adds only a
small additional gain, and moving the constant A descriptor pointer setup out of
the per-group update path is another small positive cleanup.

Profiler notes:
- The k128 rebuild required skipping a small set of ptxas register-allocation
  failing cooperative large-tile configs in the profiler source.
- Runtime profiler logs also marked some invalid or high-error candidates as
  skipped/internal-error entries; the row above is the profiler's valid
  `Best CUTLASS Config` result.

### MoE shape: M=4096 E=32 N=4096 K=4096

Measured on 2026-05-27 with current CMX profiler targets:

```bash
CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
  ./build/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_mxfp4_mxfp8_grouped_gemm_k${ktile} \
  --m=128 --n=4096 --k=4096 --groups=32 \
  --iterations=300 --warmup=50 --compare=false --explore=true
```

Profiler sweep result:

```text
Ktile  Avg us   GFLOPS    Best config
128    840.620  163497.0  Coop C1x1x1 Tile 128x64x128 Stages=11
256    786.927  174653.0  Coop C1x1x1 Tile 128x64x256 Stages=5
512    972.007  141397.0  Coop C1x1x1 Tile 128x32x512 Stages=4
```

Current best config:

```text
mxfp4 x mxfp8, groups=32, m=128, n=4096, k=4096:
  use k256, KernelPtrArrayTmaWarpSpecializedCooperative,
  Shape<_1,_1,_1>, Shape<_128, _64, _256>, Stages=5.
```

Fixed-target prebuilt TMA descriptor check on 2026-06-02, no sweep
(`--iterations=300 --warmup=20`, scheduler/builder included):

```text
baseline precomputed path:             791.640 us
prebuilt B + activation-scale desc:    751.921 us
skip unused template desc init:        752.055 us
add prebuilt weight/A desc:            748.793 us
init-time weight/A desc pointer:       748.422 us
```

For this E32 target, the prebuilt descriptor path gives about 5.02% speedup
over the baseline precomputed scheduler path. Skipping unused template
descriptor initialization is performance-neutral on this shape, while prebuilt
weight/A descriptor removal and init-time A pointer setup give small additional
gains.

Profiler notes:
- Runtime profiler logs marked several invalid/internal-error candidates as
  skipped; the row above is the profiler's valid `Best CUTLASS Config` result.

## Other CMX Families

## MXFP4 x FP8

Implementation target family:
`69_hopper_mxfp4_fp8_grouped_gemm_k{128,256,512}`.

The active m=8 fixed-config rows are:

```text
FC1 m=8: k512, Coop C1x1x1, Tile 128x16x512, Stages=5.
FC2 m=8: k128, Coop C1x1x1, Tile 128x16x128, Stages=19.
```

Source sweep rows from 2026-05-21:

```text
FC1 m=8 sweep best: 194.890 us, 44075.7 GFLOPS
FC2 m=8 sweep best: 203.388 us, 21117.2 GFLOPS
```

Dedicated single-config targets:

```bash
cmake --build build -j --target \
  69_hopper_mxfp4_fp8_grouped_gemm_fc1_m8_best \
  69_hopper_mxfp4_fp8_grouped_gemm_fc2_m8_best

CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
  ./build/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_mxfp4_fp8_grouped_gemm_fc1_m8_best \
  --m=8 --n=1024 --k=4096 --groups=128 \
  --iterations=300 --warmup=50 --compare=false

CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
  ./build/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_mxfp4_fp8_grouped_gemm_fc2_m8_best \
  --m=8 --n=4096 --k=512 --groups=128 \
  --iterations=300 --warmup=50 --compare=false
```

Sign-preprocessed FP4->FP8 fixed-config check on 2026-05-26, no sweep:

```text
FC1 m=8: 182.091 us, 47173.8 GFLOPS
FC2 m=8: 195.765 us, 21939.4 GFLOPS
```

Compile-time fixed scheduler upper-bound check on 2026-05-26, same fixed
targets and no sweep. This path is not active because it assumes uniform
per-expert token counts:

```text
FC1 m=8: 151.473 us, 56709.3 GFLOPS
FC2 m=8: 150.207 us, 28593.7 GFLOPS
```

Device precomputed tile-map scheduler check on 2026-05-26, same fixed targets,
no sweep (`--iterations=200 --warmup=50`). This is the active CMX scheduler
path for the m=8 fixed targets:

```text
FC1 m=8: 161.073 us, 53329.4 GFLOPS
FC2 m=8: 171.173 us, 25091.3 GFLOPS
```

Variable-M descriptor correctness fix retest on 2026-05-26, same fixed targets,
no sweep (`--iterations=100 --warmup=20`). This is the current correctness-fixed
status:

```text
FC1 m=8: 161.417 us, 53215.7 GFLOPS, P99 767 0.07%
FC2 m=8: 175.461 us, 24478.2 GFLOPS, P99 1801 0.04%
```

## Other CMX Families

The same source sweep also covers:

```text
int4 x fp8
mxfp4 x bf16
```

Before re-sweeping those families, check the detailed best-config table in:

```text
.claude/optimization-logs/2026-05-21-cmx-four-impl-best-configs.md
```

Promote the needed rows into this cache when they become active repeated
comparison points.

## Maintenance

Update this file when:

- A new CMX shape becomes a repeated benchmark point.
- A code change intentionally changes the best config.
- A retest shows a persistent performance shift for a cached config.
- A new single-config executable target is added for one of these cached rows.

Do not replace historical sweep values with newer numbers. Add a new
`Current us` or dated note so the cache preserves both the original discovery
result and the current status.
