# Baseline Comparison: CUTLASS Mixed GEMM vs MaloGEMM Low Latency

Date: 2026-05-14

Purpose: establish where the independent `cutlass_mixed_gemm` implementation
already helps relative to MaloGEMM's current low-latency grouped GEMM path.

## Scope

Compared paths:

- CUTLASS:
  `cutlass_mixed_gemm/build/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_fp8_grouped_gemm`
- MaloGEMM low latency:
  `build/low_latency_grouped_gemm/bench_low_latency_grouped_gemm`

The CUTLASS implementation was tested as an independent binary. No source,
header, CMake, or runtime integration was made between the two libraries.

GPU policy: benchmark on physical GPU 3 by default, using
`CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3`.

## Shapes

The main comparison used uniform 128-expert MoE-style cases:

```text
G = 128
FC1: original GEMM M = M_per_expert, N = 1024, K = 4096
FC2: original GEMM M = M_per_expert, N = 4096, K = 512
```

CUTLASS active dtype:

```text
INT4 weights x FP8 activations, BF16 output, BF16 groupwise scale
GROUP_SIZE = 128
Default tile = 128 x 16 x 512
```

Low-latency path:

```text
bench_low_latency_grouped_gemm ... dsched/moe128 benchmark path
```

The low-latency benchmark command and CUTLASS command were run serially to
avoid concurrent GPU contamination.

## Result Table

Runtime is in microseconds. `sum` is `FC1 + FC2`.

```text
m/expert  M_total  CUTLASS_FC1  CUTLASS_FC2  CUTLASS_sum  LL_FC1     LL_FC2     LL_sum     winner
1         128      181.923      208.267      390.190      138.361    74.193     212.554    low_latency 1.84x
8         1024     181.986      209.636      391.622      139.885    75.790     215.675    low_latency 1.82x
22        2816     313.733      367.007      680.740      401.531    219.874    621.406    low_latency 1.10x
64        8192     578.962      627.472      1206.434     1132.623   584.021    1716.643   CUTLASS 1.42x
128       16384    1127.470     1108.300     2235.770     2292.905   1176.157   3469.063   CUTLASS 1.55x
```

## Interpretation

The result matches the expected role split:

- Small per-expert M remains MaloGEMM low-latency territory.
- Around `m/expert = 22`, low-latency still wins overall but the margin is small.
- For large per-expert M, CUTLASS wins because its persistent cooperative TMA
  path amortizes launch/scheduler overhead and has stronger large-tile throughput.
- CUTLASS FC2 is already competitive even before FC1 crosses over; FC2 has
  `K = 512`, exactly the current INT4 x FP8 `TileShapeK`.

This supports using `cutlass_mixed_gemm` as a separate large-M comparison and
optimization target, not as a replacement for the low-latency path.

## Target192 Smoke Result

Low-latency target192 result:

```text
FC1 = 65.261 us
FC2 = 37.082 us
sum = 102.343 us
```

CUTLASS target192 observations:

- FC1 with `expert_counts_192.txt` hit an illegal memory access.
- The likely trigger is zero-M groups in the expert-count file.
- FC2 has `K = 192`, which is not suitable for the current INT4 x FP8 config
  because `TileShapeK = 512` and scale allocation uses `K / TileShapeK`.

Conclusion: target192 is not a valid final CUTLASS comparison until zero-M
handling and a K=192-capable config are fixed or separately specialized.

## Reproduction Template

Example CUTLASS command template:

```bash
CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
  ./build/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_fp8_grouped_gemm \
  --groups=128 --m=<M_PER_EXPERT> --n=<N> --k=<K> \
  --warmup=50 --iterations=500 --alpha=1 --beta=0 --compare=false
```

For `--expert_counts`, use:

```bash
CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
  ./build/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_fp8_grouped_gemm \
  --expert_counts=<counts.txt> --n=<N> --k=<K> \
  --warmup=50 --iterations=500 --alpha=1 --beta=0 --compare=false
```

Example MaloGEMM low-latency command template:

```bash
CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
  ./build/low_latency_grouped_gemm/bench_low_latency_grouped_gemm \
  <iters> <warmup> <profile> dsched <shape> <scale_group_size>
```

For final optimization evidence, use Nsight Compute only with explicit:

```bash
--clock-control none --cache-control all
```
