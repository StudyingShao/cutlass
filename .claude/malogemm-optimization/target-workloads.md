# Target Workloads

Date: 2026-05-14

This file records user-requested workload shapes that may need benchmark
coverage for the independent `cutlass_mixed_gemm` implementation. These shapes
are not automatically optimization priorities.

## MoE 256-Expert Batch-128 TopK-8

User-provided configuration:

```text
num_expert    = 256
hidden_size   = 7168
intermediate  = 3072
topk          = 8
batch_size    = 128
```

The routed expert-token count is:

```text
M_total = batch_size * topk = 128 * 8 = 1024
average M per expert = 1024 / 256 = 4
```

This is a small-M workload, despite large hidden/intermediate dimensions.

## GEMM Shapes

Assuming the existing MaloGEMM benchmark convention where FC1 fuses gate and up
projection:

```text
FC1: G = 256, per-group M = M_g, N = 2 * intermediate = 6144, K = hidden = 7168
FC2: G = 256, per-group M = M_g, N = hidden = 7168, K = intermediate = 3072
```

If FC1 is not fused, use:

```text
FC1: G = 256, per-group M = M_g, N = intermediate = 3072, K = hidden = 7168
```

For the active INT4 x FP8 CUTLASS config:

```text
TileShapeK = 512
TileShapeN = 16
GROUP_SIZE = 128
```

The target shapes satisfy the current K/N divisibility constraints:

```text
FC1 K / TileShapeK = 7168 / 512 = 14
FC2 K / TileShapeK = 3072 / 512 = 6
6144, 7168, 3072 are multiples of 64
6144 and 7168 are multiples of TileShapeN = 16
```

Scale elements per output row:

```text
FC1: 14 packed scale groups per N row
FC2:  6 packed scale groups per N row
```

## Routing Distribution Implications

Uniform synthetic distribution:

```text
M_g = 4 for all 256 experts
active experts = 256
zero-M experts = 0
```

Real routing may not be uniform. If assignments were independent and uniform,
the expected number of zero-M experts would be approximately:

```text
256 * exp(-4) ~= 4.7
```

Actual MoE routing can be more skewed depending on load-balancing and capacity
rules. Therefore zero-M handling is not optional for real `G=256, M_total=1024`
traces.

Current CUTLASS observation from target192 testing: zero-M groups can trigger an
illegal memory access in the benchmark path. Before using real routing traces
for this workload, either:

- fix zero-M support in the CUTLASS example/kernel path, or
- compact out zero-M experts before launching and preserve correct metadata for
  comparison.

## Performance Expectation

This workload is not in the large-M regime that motivated the CUTLASS path.
Average `M_g = 4` is closer to the low-latency implementation's intended use
case.

However, the large N/K values make the crossover uncertain:

- FC1 has very large K and large fused N.
- FC2 has large N and K = 3072.
- CUTLASS may amortize some fixed overhead through larger per-token work, but
  persistent grouped scheduling still sees many tiny groups.

Do not infer the winner from the previous `G=128, N/K smaller` baseline alone.
This workload needs a direct benchmark once the harness safely supports it.

## Benchmark Harness Gap

Current low-latency benchmark presets cover:

```text
target192
uniform128_mX + moe128 shape
```

They do not yet cover:

```text
uniform256_m4
G=256, FC1 N=6144 K=7168, FC2 N=7168 K=3072
```

Current CUTLASS CLI can express the uniform case with `--groups=256 --m=4`,
but the example still performs expensive reference generation and has unsafe
zero-M behavior for real traces.

Requested synthetic benchmark target:

```text
uniform256_m4
FC1: --groups=256 --m=4 --n=6144 --k=7168
FC2: --groups=256 --m=4 --n=7168 --k=3072
```

Recommended real-trace target:

```text
expert_counts_256_batch128_topk8.txt
sum(M_g) = 1024
G = 256
zero-M groups allowed only after the zero-M path is fixed or compacted
```
