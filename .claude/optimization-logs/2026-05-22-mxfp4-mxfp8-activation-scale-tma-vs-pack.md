# MXFP4 x MXFP8 Activation Scale Loading: Runtime Pack vs Direct TMA

**Date**: 2026-05-22
**Repository**: `cutlass_mixed_gemm`
**Active scope**: CMX only, specifically the active `int4 x fp8`,
`mxfp4 x bf16`, `mxfp4 x fp8`, and `mxfp4 x mxfp8` family that shares the
same mainloop.
**Baseline commit**: `2494d505` (`MXFP4 x MXFP8: Add activation-scale grouped GEMM path`)
**Experiment status**: Direct raw activation-scale TMA is an unstaged
experiment on top of the committed runtime-pack implementation.

## Goal

The committed `mxfp4 x mxfp8` path supports raw activation data plus a separate
activation scale tensor. The runtime requirement is that activation and its
scale remain in the original layout, with scale continuous along K. Weight data
can be preprocessed offline, but activation scale cannot.

The runtime-pack implementation handles this by launching a small helper kernel
before GEMM. That helper repacks raw activation scales into the same CTA-local,
K-tile-major layout already used by the weight scale path. The GEMM mainloop can
then load the packed activation scale with the lightweight contiguous
`SM90_BULK_COPY_G2S` path.

This experiment checks whether the helper kernel can be removed by making the
mainloop load raw activation scales directly with descriptor-based TMA.

## Hypothesis

Removing the runtime pack kernel should reduce end-to-end time because the
GEMM launch no longer needs an extra preprocessing kernel. Since the activation
matrix already uses TMA, adding activation scale TMA might appear cheap.

The counter-hypothesis is that the scale tile is too small for descriptor-based
TMA to be efficient. Runtime packing may be doing useful work by converting a
tiny strided 2D read into one contiguous bulk copy.

## Shape And Configuration

The comparison used a single development shape, not a full sweep:

```text
groups          128
M               8
N               1024
K               4096
TileShapeK      512
GROUP_SIZE      32
scale_k_tile    TileShapeK / GROUP_SIZE = 16
```

In this swapped CMX path, the CTA activation-scale tile consumed by the
mainloop is logically:

```text
TileShapeN x scale_k_tile = 16 x 16 elements
```

For `float_ue8m0_t`, that is only 256 bytes per K tile.

## Runtime-Pack Baseline

The committed baseline uses:

- Raw activation scale input stored as an activation-side tensor.
- `pack_activation_scale_ktile_mn_major_grouped_kernel` in
  `examples/69_hopper_mixed_dtype_grouped_gemm/BF16_MXFP4_Test.h`.
- Packed activation scale stored as `scale_k_tile` major, then CTA-M/N major,
  so each CTA K tile can load the required scale values contiguously.
- GEMM mainloop bulk-copy load through `SM90_BULK_COPY_G2S`.

The pack kernel exists because `SM90_BULK_COPY_G2S` is a descriptor-less 1D
copy. It cannot gather a strided 2D tile out of the original activation-scale
matrix. The helper kernel makes the future GEMM-side copy contiguous.

## Direct Raw TMA Experiment

The direct TMA experiment changed the mainloop to add a third descriptor-based
TMA path for activation scale:

- Added an activation scale TMA descriptor and tensormap.
- Added activation scale descriptor address, shape, stride, fence, and acquire
  handling.
- Added a third producer-side TMA copy in the K-tile loop, after A and B:

```cpp
copy(mainloop_params.tma_load_a.with(...), ...);
copy(mainloop_params.tma_load_b.with(...), ...);
copy(mainloop_params.tma_load_activation_scale.with(...), ...);
```

- Pointed the mainloop at raw activation scale instead of packed activation
  scale.
- Removed the runtime `prepare_activation_scale_tensor(options)` call from the
  benchmark path.

Important: this version is intentionally left unstaged while it is still under
evaluation.

## Commands

Current direct-TMA worktree:

```bash
cmake --build build -j --target 69_hopper_mxfp4_mxfp8_grouped_gemm

CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
./build/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_mxfp4_mxfp8_grouped_gemm \
  --m=8 --n=1024 --k=4096 --groups=128 \
  --iterations=300 --warmup=50 --compare=false
```

Runtime-pack baseline from the committed implementation:

```bash
git worktree add /tmp/cmx_runtime_pack_2494d505_1779448015 2494d505

cmake -S /tmp/cmx_runtime_pack_2494d505_1779448015 \
  -B /tmp/cmx_runtime_pack_2494d505_1779448015/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_NVCC_ARCHS=90a

cmake --build /tmp/cmx_runtime_pack_2494d505_1779448015/build \
  -j --target 69_hopper_mxfp4_mxfp8_grouped_gemm

CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
/tmp/cmx_runtime_pack_2494d505_1779448015/build/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_mxfp4_mxfp8_grouped_gemm \
  --m=8 --n=1024 --k=4096 --groups=128 \
  --iterations=300 --warmup=50 --compare=false
```

The temporary worktree was kept rather than deleted, following the repository
deletion-safety rule.

## Correctness

Both versions produced the same smoke-check error counts for this shape:

```text
P99_error_count 309 0.03%
P98_error_count 144 0.01%
P95_error_count 51  0.00%
```

This was treated as a smoke check only because `--compare=false` was used in
the benchmark commands.

## Performance

```text
Version                 Run1 us   Run2 us   Run3 us   Avg us   Delta vs pack
runtime pack baseline   231.385   232.839   233.010   232.411  baseline
direct raw TMA          227.885   226.618   226.570   227.024  -5.387 us / -2.32%
```

Direct raw TMA is faster end to end for this small development case, but the
gain is modest. The result does not support the simple assumption that removing
the pack kernel should expose a large speedup.

## Analysis

The direct TMA version removes a helper kernel, but it adds real work inside the
GEMM mainloop:

1. It adds a third descriptor lifecycle:
   `fill_tma_gmem_shape_stride`, descriptor field replacement, commit/wait,
   copy-fence-release, and fence-acquire.

2. It adds a third TMA copy per K tile in the producer loop. This copy shares
   the same producer pipeline and barrier accounting as the A/B TMA copies.

3. The payload is tiny. For the tested K tile, activation scale is only
   `16 x 16 x 1 byte = 256 bytes`. Descriptor-based TMA has a fixed overhead
   that is hard to amortize at this size.

4. The raw activation-scale tensor is K-contiguous per activation row, but the
   CTA wants a small 2D tile. Direct TMA loads it as a strided 2D region. The
   runtime-pack baseline converts that same region into a contiguous 256-byte
   segment and then uses descriptor-less bulk copy.

5. The activation matrix already having a TMA descriptor does not make
   activation-scale TMA free. The scale tensor has its own descriptor, own
   descriptor updates, own fences, and own TMA issue.

The current result suggests the runtime pack kernel cost is lower than expected,
or at least partially offset by the more efficient packed GEMM-side scale load.

## Lessons Learned

- Packing can be a performance optimization, not only a compatibility step.
  For tiny scale tensors it turns a strided 2D access into a single contiguous
  bulk copy.

- Descriptor-based TMA should not automatically replace `SM90_BULK_COPY_G2S`
  for small auxiliary tensors. The descriptor and issue overhead can dominate a
  128-512 byte payload.

- End-to-end comparisons must separate four numbers before making a final
  decision:

```text
1. pack kernel alone
2. GEMM with prepacked activation scale
3. direct raw-scale TMA GEMM
4. runtime pack + GEMM end to end
```

- For grouped small-M MoE shapes, per-group and per-K-tile fixed overhead is
  visible. A small amount of extra producer work can erase much of a launch-side
  saving.

- The weight scale and activation scale tensors must remain semantically
  separate in code and naming. They may share a final packed shape, but they are
  not the same input and should not be described as one generic scale.

## Next Steps

Recommended follow-up before deciding whether to keep the direct-TMA approach:

1. Measure the pack kernel alone for the same shape.
2. Measure GEMM-only with activation scale already packed.
3. Run NCU with explicit `--clock-control none --cache-control all` and inspect
   TMA issue/wait behavior for the third TMA copy.
4. Repeat for larger M cases such as M=18 and M=22, where the helper kernel and
   CTA count may change the tradeoff.
5. Check whether the activation-scale descriptor can avoid per-group shape and
   stride replacement for uniform shapes.

The direct raw TMA direction is promising enough to keep as an experiment, but
not yet strong enough to replace the runtime-pack implementation without more
profiling evidence.
