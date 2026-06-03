# CMX Precomputed Work-Order Locality Experiment

Date: 2026-05-29
GPU: NVIDIA H20, `CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3`

## Goal

Explore whether changing the precomputed grouped scheduler tile-list order can
improve GEMM performance by making simultaneously running CTAs touch a more
concentrated L2 working set.

The tested experiment adds `--precomputed_work_order=<int>`:

- `0`: current precomputed order.
- `1`: cluster-aware weight-stationary order. The failed first attempt directly
  packed arbitrary `(major, minor, expert)` coordinates and broke clustered TMA
  correctness. The corrected attempt preserves each hardware cluster lane's
  major/minor offset and only reorders logical cluster coordinates.

## Correctness

Command:

```bash
timeout 90s env CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
./build/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_mxfp4_mxfp8_grouped_gemm_precomp_k256 \
  --m=128 --n=4096 --k=4096 --groups=32 --total_routed_tokens=4096 \
  --iterations=1 --warmup=0 --compare=true --explore=false \
  --precomputed_work_order=1
```

Result: exit code 0. The usual quantization error counters remain in the log.

## Performance

Commands used `--compare=false --explore=true --iterations=80 --warmup=20`.

```text
Shape                         Order  Best us   Best config
E32_M128_N4096_K4096 k256     0      787.978   Coop C1x2x1 Tile<128,64,256> S=5
E32_M128_N4096_K4096 k256     1      788.134   Coop C1x2x1 Tile<128,64,256> S=5
E256_M24_N512_K4096 k512      0      262.082   Coop C1x1x1 Tile<128,32,512> S=4
E256_M24_N512_K4096 k512      1      261.852   Coop C1x1x1 Tile<128,32,512> S=4
```

For a larger-token case, a full `E32_M1024_N4096_K4096 k256` sweep with
`--iterations=60 --warmup=20` did not finish within 900 s because later
candidates were extremely slow. The completed part already showed
`Coop C1x1x1 Tile<128,64,256> S=5` at 5.966 ms, matching a dedicated
fixed-config target added only for this experiment.

Fixed-config command used `--compare=false --explore=false --iterations=120
--warmup=30`.

```text
Shape                         Order  Avg us    Config
E32_M1024_N4096_K4096 k256    0      5959.83   Coop C1x1x1 Tile<128,64,256> S=5
E32_M1024_N4096_K4096 k256    1      5960.49   Coop C1x1x1 Tile<128,64,256> S=5
```

## Analysis

This work-order experiment does not produce a meaningful gain on the target
shapes. The reason is structural: both target shapes have only one token tile per
expert for the winning configs.

- `E32_M128_N4096_K4096`, best `Tile<128,64,256>`: per-expert `M=128`, so
  `ceil(M / 128) = 1`; only the channel/N tile dimension has multiple tiles.
- `E256_M24_N512_K4096`, best `Tile<128,32,512>`: per-expert `M=24`, so
  `ceil(M / 128) = 1`; again there is no second token tile to reorder against.

With only one token tile, an in-expert order change cannot create extra temporal
reuse between token tiles. The current expert-major list already places many
concurrent CTAs on the same expert and adjacent channel tiles, so `order=1`
mostly degenerates to the same access pattern for these shapes.

The larger `M=1024` case does have multiple token tiles, but the tested
cluster-aware weight-stationary order still showed no measurable speedup on the
fixed best completed config. This suggests the current swizzled order is already
good enough for L2 locality on this path, or the bottleneck is not improved by
simple in-expert tile-list reordering.

## Decision

Do not promote `--precomputed_work_order=1` as a production optimization for the
tested target shapes. Keep this document as the reviewable record of the
experiment, but do not keep the experimental work-order knob or code path in the
production CMX implementation unless a future experiment demonstrates a real
benefit on a representative var-M shape.
