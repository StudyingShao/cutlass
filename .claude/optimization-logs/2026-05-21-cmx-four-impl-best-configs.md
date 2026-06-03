# CMX Four Implementation Best Config Sweep

Date: 2026-05-21

## Scope

This log records a full CMX profiler sweep for these implementation families:

- `int4xfp8`: `69_hopper_int4_fp8_grouped_gemm`
- `mxfp4xbf16`: `69_hopper_mxfp4_bf16_grouped_gemm`
- `mxfp4xfp8`: `69_hopper_mxfp4_fp8_grouped_gemm`
- `mxfp4xmxfp8`: `69_hopper_mxfp4_mxfp8_grouped_gemm`

Only the current CMX focus implementations are covered. The sweep uses the
current staged CMX worktree based on commit `84cfffcb`.

## Machine And Toolchain

- GPU: GPU 3, NVIDIA H20
- CUDA: 12.6, nvcc V12.6.68
- Benchmark environment: `CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3`
- Profiler iterations: `--iterations=100 --warmup=10`
- Correctness compare during timing: `--compare=false`
- Search space: Ktile `128`, `256`, `512`; each executable runs
  `--explore=true` over the compiled profiler configs.
- `mxfp4xmxfp8` time includes the runtime activation-scale pack kernel because
  the timed profiler path wraps activation-scale repack plus GEMM in one CUDA
  event region.

Tracked CSV summaries for the accepted run:

```text
.claude/optimization-logs/2026-05-21-cmx-four-impl-best-configs-231136/
```

The raw `.log` profiler outputs are local artifacts under the same directory
when present. They are intentionally not part of the committed documentation
set; use the markdown summary and tracked CSVs as the reviewable record.

The directory `2026-05-21-cmx-four-impl-best-configs-230941/` is an earlier
partial run whose parser used an invalid awk variable name. It is not used for
the numbers below.

## Shapes

All cases use 128 groups and uniform per-expert M.

```text
FC1: N=1024 K=4096, M in 1 2 4 8 12 18 22
FC2: N=4096 K=512,  M in 1 2 4 8 12 18 22
```

Command template:

```bash
CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
  ./build/examples/69_hopper_mixed_dtype_grouped_gemm/${target}_k${ktile} \
  --explore=true --m=${m} --n=${n} --k=${k} --groups=128 \
  --iterations=100 --warmup=10 --compare=false
```

## Best Time Summary

Times are in microseconds. The suffix `k128`, `k256`, or `k512` is the best
Ktile executable for that case.

### FC1, N=1024 K=4096

```text
M   int4xfp8       mxfp4xbf16     mxfp4xfp8     mxfp4xmxfp8
1   179.089 k512   254.338 k512   194.801 k512  232.660 k512
2   179.959 k512   254.297 k512   193.807 k512  232.959 k512
4   178.583 k512   252.951 k512   195.323 k512  231.854 k512
8   179.085 k512   255.508 k512   194.890 k512  232.565 k512
12  178.078 k512   253.802 k512   195.231 k512  232.212 k512
18  210.168 k256   379.437 k256   250.619 k256  310.526 k512
22  208.731 k256   379.456 k256   250.533 k256  311.540 k512
```

### FC2, N=4096 K=512

```text
M   int4xfp8       mxfp4xbf16     mxfp4xfp8     mxfp4xmxfp8
1   195.461 k128   230.485 k128   204.311 k128  222.778 k128
2   195.324 k128   230.267 k256   203.368 k128  223.204 k256
4   196.577 k128   230.946 k128   204.614 k128  223.200 k128
8   194.269 k128   234.676 k128   203.388 k128  223.667 k128
12  195.921 k128   232.615 k256   204.218 k128  222.522 k256
18  213.935 k128   288.378 k128   234.808 k256  262.028 k128
22  214.904 k128   288.331 k128   234.563 k128  262.848 k128
```

## Best Config Detail

The rows below are the global best per `(implementation, FC, M)` after taking
the minimum over Ktile 128, 256, and 512. `Cluster`, `Tile`, and `Stages` are
copied from the profiler's `Best CUTLASS Config` line.

### FC1 Detail

```text
Impl           M   us       Ktile  Cluster          Tile                 Stg  Schedule
int4xfp8       1   179.089  512    Shape<_1,_1,_1>  Shape<_128, _16, _512>  5    KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xbf16     1   254.338  512    Shape<_1,_1,_1>  Shape<_128, _16, _512>  4    KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xfp8      1   194.801  512    Shape<_1,_1,_1>  Shape<_128, _16, _512>  5    KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xmxfp8    1   232.660  512    Shape<_1,_1,_1>  Shape<_128, _16, _512>  5    KernelPtrArrayTmaWarpSpecializedCooperative
int4xfp8       2   179.959  512    Shape<_1,_1,_1>  Shape<_128, _16, _512>  5    KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xbf16     2   254.297  512    Shape<_1,_1,_1>  Shape<_128, _16, _512>  4    KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xfp8      2   193.807  512    Shape<_1,_1,_1>  Shape<_128, _16, _512>  5    KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xmxfp8    2   232.959  512    Shape<_1,_1,_1>  Shape<_128, _16, _512>  5    KernelPtrArrayTmaWarpSpecializedCooperative
int4xfp8       4   178.583  512    Shape<_1,_1,_1>  Shape<_128, _16, _512>  5    KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xbf16     4   252.951  512    Shape<_1,_1,_1>  Shape<_128, _16, _512>  4    KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xfp8      4   195.323  512    Shape<_1,_1,_1>  Shape<_128, _16, _512>  5    KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xmxfp8    4   231.854  512    Shape<_1,_1,_1>  Shape<_128, _16, _512>  5    KernelPtrArrayTmaWarpSpecializedCooperative
int4xfp8       8   179.085  512    Shape<_1,_1,_1>  Shape<_128, _16, _512>  5    KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xbf16     8   255.508  512    Shape<_1,_1,_1>  Shape<_128, _16, _512>  4    KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xfp8      8   194.890  512    Shape<_1,_1,_1>  Shape<_128, _16, _512>  5    KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xmxfp8    8   232.565  512    Shape<_1,_1,_1>  Shape<_128, _16, _512>  5    KernelPtrArrayTmaWarpSpecializedCooperative
int4xfp8       12  178.078  512    Shape<_1,_1,_1>  Shape<_128, _16, _512>  5    KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xbf16     12  253.802  512    Shape<_1,_1,_1>  Shape<_128, _16, _512>  4    KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xfp8      12  195.231  512    Shape<_1,_1,_1>  Shape<_128, _16, _512>  5    KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xmxfp8    12  232.212  512    Shape<_1,_1,_1>  Shape<_128, _16, _512>  5    KernelPtrArrayTmaWarpSpecializedCooperative
int4xfp8       18  210.168  256    Shape<_1,_1,_1>  Shape<_128, _32, _256>  8    KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xbf16     18  379.437  256    Shape<_1,_1,_1>  Shape< _64, _32, _256>  8    KernelPtrArrayTmaWarpSpecializedPingpong
mxfp4xfp8      18  250.619  256    Shape<_1,_1,_1>  Shape< _64, _32, _256>  12   KernelPtrArrayTmaWarpSpecializedPingpong
mxfp4xmxfp8    18  310.526  512    Shape<_2,_1,_1>  Shape<_128, _32, _512>  4    KernelPtrArrayTmaWarpSpecializedCooperative
int4xfp8       22  208.731  256    Shape<_1,_1,_1>  Shape<_128, _32, _256>  8    KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xbf16     22  379.456  256    Shape<_1,_1,_1>  Shape< _64, _32, _256>  8    KernelPtrArrayTmaWarpSpecializedPingpong
mxfp4xfp8      22  250.533  256    Shape<_1,_1,_1>  Shape< _64, _32, _256>  12   KernelPtrArrayTmaWarpSpecializedPingpong
mxfp4xmxfp8    22  311.540  512    Shape<_1,_1,_1>  Shape<_128, _32, _512>  4    KernelPtrArrayTmaWarpSpecializedCooperative
```

### FC2 Detail

```text
Impl           M   us       Ktile  Cluster          Tile                 Stg  Schedule
int4xfp8       1   195.461  128    Shape<_1,_1,_1>  Shape<_128, _16, _128>  19   KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xbf16     1   230.485  128    Shape<_1,_1,_1>  Shape<_128, _16, _128>  16   KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xfp8      1   204.311  128    Shape<_1,_1,_1>  Shape<_128, _16, _128>  19   KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xmxfp8    1   222.778  128    Shape<_1,_1,_1>  Shape<_128, _16, _128>  19   KernelPtrArrayTmaWarpSpecializedCooperative
int4xfp8       2   195.324  128    Shape<_1,_1,_1>  Shape<_128, _16, _128>  19   KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xbf16     2   230.267  256    Shape<_1,_1,_1>  Shape<_128, _16, _256>  8    KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xfp8      2   203.368  128    Shape<_1,_1,_1>  Shape<_128, _16, _128>  19   KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xmxfp8    2   223.204  256    Shape<_1,_1,_1>  Shape<_128, _16, _256>  9    KernelPtrArrayTmaWarpSpecializedCooperative
int4xfp8       4   196.577  128    Shape<_1,_1,_1>  Shape<_128, _16, _128>  19   KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xbf16     4   230.946  128    Shape<_1,_1,_1>  Shape<_128, _16, _128>  16   KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xfp8      4   204.614  128    Shape<_1,_1,_1>  Shape<_128, _16, _128>  19   KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xmxfp8    4   223.200  128    Shape<_1,_1,_1>  Shape<_128, _16, _128>  19   KernelPtrArrayTmaWarpSpecializedCooperative
int4xfp8       8   194.269  128    Shape<_1,_1,_1>  Shape<_128, _16, _128>  19   KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xbf16     8   234.676  128    Shape<_1,_1,_1>  Shape<_128, _16, _128>  16   KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xfp8      8   203.388  128    Shape<_1,_1,_1>  Shape<_128, _16, _128>  19   KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xmxfp8    8   223.667  128    Shape<_1,_1,_1>  Shape<_128, _16, _128>  19   KernelPtrArrayTmaWarpSpecializedCooperative
int4xfp8       12  195.921  128    Shape<_1,_1,_1>  Shape<_128, _16, _128>  19   KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xbf16     12  232.615  256    Shape<_1,_1,_1>  Shape<_128, _16, _256>  8    KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xfp8      12  204.218  128    Shape<_1,_1,_1>  Shape<_128, _16, _128>  19   KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xmxfp8    12  222.522  256    Shape<_1,_1,_1>  Shape<_128, _16, _256>  9    KernelPtrArrayTmaWarpSpecializedCooperative
int4xfp8       18  213.935  128    Shape<_1,_1,_1>  Shape<_128, _32, _128>  15   KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xbf16     18  288.378  128    Shape<_1,_1,_1>  Shape<_128, _32, _128>  12   KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xfp8      18  234.808  256    Shape<_1,_1,_1>  Shape<_128, _32, _256>  8    KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xmxfp8    18  262.028  128    Shape<_1,_1,_1>  Shape<_128, _32, _128>  15   KernelPtrArrayTmaWarpSpecializedCooperative
int4xfp8       22  214.904  128    Shape<_1,_1,_1>  Shape<_128, _32, _128>  15   KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xbf16     22  288.331  128    Shape<_2,_1,_1>  Shape<_128, _32, _128>  12   KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xfp8      22  234.563  128    Shape<_1,_1,_1>  Shape<_128, _32, _128>  15   KernelPtrArrayTmaWarpSpecializedCooperative
mxfp4xmxfp8    22  262.848  128    Shape<_1,_1,_1>  Shape<_128, _32, _128>  15   KernelPtrArrayTmaWarpSpecializedCooperative
```

## Immediate Reuse Rules

- FC1, M <= 12: Ktile 512 wins for all four implementations in this run.
- FC1, M >= 18: `int4xfp8`, `mxfp4xbf16`, and `mxfp4xfp8` prefer Ktile 256;
  `mxfp4xmxfp8` still prefers Ktile 512.
- FC2: Ktile 128 is usually best. Exceptions in this run are:
  `mxfp4xbf16` at M=2 and M=12, `mxfp4xfp8` at M=18, and
  `mxfp4xmxfp8` at M=2 and M=12, where Ktile 256 wins.

## Verification

- Built all twelve profiler targets for the four implementation families and
  three Ktile values.
- Ran 168 profiler invocations serially on GPU 3.
- `raw_best.csv`: 169 lines, including header.
- `failures.csv`: 1 line, header only.
