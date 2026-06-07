# CMX Mainloop Transfer Experiments for W4A16 and W4A8

Date: 2026-06-05

## Goal

Check whether optimizations found in the fused pre-MMA-scale mainloop can be
transferred to the original CMX post-MMA-scale mainloop without hurting other
supported paths.

Primary test shape:

```text
groups=128, per-expert m=8, n=1024, k=4096, ktile=512
```

Paths:

- W4A16: `mxfp4_bf16`
- W4A8: `mxfp4_fp8`

Benchmark command pattern:

```bash
CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
./build/examples/69_hopper_mixed_dtype_grouped_gemm/<target> \
  --groups=128 --m=8 --n=1024 --k=4096 \
  --warmup=5 --iterations=20 --explore=true --split_timing=true --compare=false
```

The 20-iteration result was rechecked with a longer run because the W4A16
regression was small enough to be suspicious:

```bash
CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
./build/examples/69_hopper_mixed_dtype_grouped_gemm/<target> \
  --groups=128 --m=8 --n=1024 --k=4096 \
  --warmup=20 --iterations=300 --explore=true --split_timing=true --compare=false
```

Targets:

```text
69_hopper_mxfp4_bf16_grouped_gemm_k512
69_hopper_mxfp4_fp8_grouped_gemm_k512
```

## Results

```text
Variant                                      W4A16 best us   W4A8 best us   Decision
Baseline                                     221.398         152.562        Reference
Static K-block cute::Int indexing             222.506         150.864        Not common; W4A8-only candidate
Static K-block + previous-scale overlap       222.501         151.320        Reject
Static loop, BF16 runtime convert             222.910         151.635        Reject
Static loop, BF16 runtime convert/indexing    222.477         151.376        Reject as common
```

300-iteration recheck for the final static-loop experiment versus the
runtime-loop baseline:

```text
Variant                         W4A16 best us   W4A8 best us   Decision
Runtime-loop baseline             220.854         152.442        Reference
Static loop/runtime BF16 conv      222.776         150.314        W4A8 helps, W4A16 still regresses
Delta vs baseline                   +1.922          -2.128
```

Best observed configs stayed in the same family:

```text
W4A16: KernelPtrArrayTmaWarpSpecializedCooperative Shape<_2,_1,_1> Tile<_128,_16,_512> Stages=4
W4A8:  KernelPtrArrayTmaWarpSpecializedCooperative Shape<_2,_1,_1> Tile<_128,_16,_512> Stages=5
```

Correctness:

- W4A16 had zero P99/P98/P95/Humming-tolerance outliers in all runs.
- W4A8 matched the existing sparse tolerance profile for this path:
  `P99=767`, `P98=380`, `P95=146`, `Humming_tol=1`.

## Interpretation

The clean compile-time K-block version helps W4A8 by about 1.1% to 1.4% on
this shape, but it slows W4A16 by about 0.5% to 0.9%. The 300-iteration
recheck reproduced the same direction, so the W4A16 result is not explained by
too few benchmark iterations. The BF16-only runtime-convert and
runtime-indexing ablations did not recover the W4A16 baseline, so the BF16
regression is not isolated to the new `cute::Int<KBlock>` convert overload or
static tensor slice type. It appears to come from the broader static
`cute::for_each` control-flow/codegen change.

The previous-scale overlap experiment did not help either path. It worsened
W4A8 relative to the plain static K-block version, so it should not be kept.

Compile time also increased noticeably for profiler-enabled targets because
static K-block expansion increases per-topology codegen work.

## Expanded Shape Recheck

Date: 2026-06-08

The original W4A16 conclusion only covered one small-M FC1 shape. Rechecked an
intermediate shared static-loop implementation against a clean `HEAD=2e7452ed`
baseline on four FC1/FC2 shapes, using 300 iterations and profiler exploration
for both baseline and static-loop variants.

Implementation detail: this intermediate version is not a pure "make every
K-block index a `cute::Int`" change for every path. It shares the static
`cute::for_each` mainloop control flow, but keeps the BF16 MMA operand slice as
a runtime `int k_block` while FP8 uses the compile-time `cute::Int` slice. The
final full-static operand-slice decision is recorded below.

```bash
CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
./build/examples/69_hopper_mixed_dtype_grouped_gemm/<target> \
  --groups=128 --m=<m> --n=<n> --k=<k> \
  --warmup=20 --iterations=300 --explore=true --split_timing=true --compare=false
```

First pass:

```text
Shape                         Path    Baseline us   Static us    Delta us    Delta %
E128_M8_N1024_K4096            W4A16      219.432     220.237      +0.805     +0.37%
E128_M22_N1024_K4096           W4A16      396.229     350.681     -45.548    -11.50%
E128_M8_N4096_K512             W4A16      181.652     179.195      -2.457     -1.35%
E128_M22_N4096_K512            W4A16      257.081     240.909     -16.172     -6.29%

E128_M8_N1024_K4096            W4A8       153.477     152.035      -1.442     -0.94%
E128_M22_N1024_K4096           W4A8       235.438     235.200      -0.238     -0.10%
E128_M8_N4096_K512             W4A8       147.329     145.911      -1.418     -0.96%
E128_M22_N4096_K512            W4A8       221.806     218.605      -3.201     -1.44%
```

Second pass for W4A16 only, to check the surprising W4A16 direction:

```text
Shape                         Baseline us   Static us    Delta us    Delta %
E128_M8_N1024_K4096              219.717     220.618      +0.901     +0.41%
E128_M22_N1024_K4096             396.464     350.698     -45.766    -11.54%
E128_M8_N4096_K512               181.594     179.199      -2.395     -1.32%
E128_M22_N4096_K512              255.739     241.192     -14.547     -5.69%
```

The expanded test changes the conclusion. W4A8 remains a stable small win.
W4A16 is shape-dependent, but it is not a general regression: only FC1/M8
regresses slightly, while FC1/M22 and both FC2 shapes improve, with the FC1/M22
gain large enough that this shared static-loop control-flow change should not be
treated as a W4A8-only candidate.

## Full-Static BF16 Operand Indexing Check

Date: 2026-06-08

The final version changes W4A16 to use the same compile-time MMA operand slice
as W4A8:

```cpp
tCrA_mma(_,_,k_block_c)
tCrB(_,_,k_block_c,read_stage)
```

This tests whether the retained BF16 `int k_block` slice should be replaced by a
fully static slice. Same command shape as the expanded recheck, W4A16 only.

```text
Shape                         Baseline us   Runtime-slice us   Full-static us   Full vs runtime
E128_M8_N1024_K4096              219.432          220.237          220.684        +0.447 / +0.20%
E128_M22_N1024_K4096             396.229          350.681          350.739        +0.058 / +0.02%
E128_M8_N4096_K512               181.652          179.195          179.198        +0.003 / +0.00%
E128_M22_N4096_K512              257.081          240.909          241.771        +0.862 / +0.36%
```

Best configs:

```text
E128_M8_N1024_K4096     KernelPtrArrayTmaWarpSpecializedCooperative Shape<_1,_1,_1> Tile<_128,_16,_512> Stages=4
E128_M22_N1024_K4096    KernelPtrArrayTmaWarpSpecializedCooperative Shape<_2,_1,_1> Tile<_128,_32,_512> Stages=3
E128_M8_N4096_K512      KernelPtrArrayTmaWarpSpecializedPingpong    Shape<_1,_1,_1> Tile< _64,_16,_512> Stages=6
E128_M22_N4096_K512     KernelPtrArrayTmaWarpSpecializedPingpong    Shape<_1,_1,_1> Tile< _64,_32,_512> Stages=4
```

Full-static BF16 operand indexing is not a performance improvement over the
path-specialized runtime BF16 slice in the final retest, but the absolute delta
is below 1 us in all four shapes. The cleaner full-static code is acceptable.

### A Convert API Check

The first full-static patch accidentally kept the A-convert call on the old
`int k_block` API:

```c++
constexpr int k_block = decltype(k_block_c)::value;
Utils::convert_A_kblock(tCrA_load_4b_packed, tCrA_mma, k_block);
```

That meant the newly added `cute::Int<KBlock>` overload was not used by the CMX
post-scale mainloop. Rechecked the same four FC1/FC2 shapes with the caller
passing `k_block_c` directly into `convert_A_kblock`.

```text
Shape                         Path    int API us    cute::Int API us   Delta
E128_M8_N1024_K4096           W4A16      220.684            220.240     -0.444 / -0.20%
E128_M8_N1024_K4096           W4A8       152.015            152.163     +0.148 / +0.10%
E128_M22_N1024_K4096          W4A16      350.739            350.602     -0.137 / -0.04%
E128_M22_N1024_K4096          W4A8       235.127            235.552     +0.425 / +0.18%
E128_M8_N4096_K512            W4A16      179.198            178.843     -0.355 / -0.20%
E128_M8_N4096_K512            W4A8       146.128            144.998     -1.130 / -0.77%
E128_M22_N4096_K512           W4A16      241.771            241.061     -0.710 / -0.29%
E128_M22_N4096_K512           W4A8       218.107            217.915     -0.192 / -0.09%
```

The `cute::Int` API is faster in six of eight checks. The two regressions are
small FC1/W4A8 changes within 0.2%, while the FC2/W4A8 case improves by 0.77%.
Keep the `cute::Int` A-convert API.

## Decision

Keep the shared static-loop mainloop migration in the CMX post-scale mainloop
with full-static MMA operand indexing and A-convert indexing for both W4A16 and
W4A8:

- W4A16/BF16 uses compile-time `cute::Int` MMA operand slices.
- W4A8/FP8 uses compile-time `cute::Int` MMA operand slices.
- W4A16/BF16 and W4A8/FP8 both pass compile-time `cute::Int` K-block indices
  into the A-convert path.

The known tradeoff versus the path-specialized runtime BF16 slice is a small
W4A16 regression, below 1 us on the checked FC1/FC2 shapes. The full-static
version keeps the code simpler and still preserves the major gains versus the
original runtime-loop baseline on the broader FC1/FC2 coverage.
