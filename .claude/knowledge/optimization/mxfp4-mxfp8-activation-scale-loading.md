# MXFP4 x MXFP8 Activation Scale Loading

**Last Updated**: 2026-05-25
**Status**: Active CMX optimization knowledge
**Related log**:
`.claude/optimization-logs/2026-05-22-mxfp4-mxfp8-activation-scale-tma-vs-pack.md`

## Background

The CMX `mxfp4 x mxfp8` path uses MXFP4 weights and FP8 activation values with
an activation-side `ue8m0` scale. Weight scale can be preprocessed offline.
Activation scale cannot: it must be accepted in the runtime input layout, with
scale groups continuous along K.

The current shared mainloop already knows how to consume packed per-K-tile scale
data efficiently. The important question is how to move raw activation scale
into that shape with the lowest end-to-end cost.

## Terminology

`scale_k_tile` means the number of scale groups covered by one CTA K tile:

```text
scale_k_tile = TileShapeK / GROUP_SIZE
```

For the active MXFP formats:

```text
GROUP_SIZE  = 32
TileShapeK  = 512 in the common fast config
scale_k_tile = 16
```

It is not a new problem dimension. It is the K-tile index measured in scale
groups rather than raw K elements.

## Two Valid Load Strategies

### Runtime Pack Plus Bulk Copy

The runtime-pack strategy launches a small helper kernel before GEMM:

```text
raw activation scale:      activation row major, K-scale contiguous
packed activation scale:   CTA K-tile major, CTA M/N tile contiguous
GEMM load path:            SM90_BULK_COPY_G2S
```

The helper kernel is:

```text
examples/69_hopper_mixed_dtype_grouped_gemm/BF16_MXFP4_Test.h
pack_activation_scale_ktile_mn_major_grouped_kernel
```

Its role is to make every CTA K tile's activation scales contiguous in memory,
so the GEMM mainloop can use descriptor-less bulk copy. Bulk copy is cheap for
small contiguous auxiliary data because it does not require a 128-byte TMA
descriptor.

### Direct Raw Activation-Scale TMA

The direct-TMA strategy skips the helper kernel and makes the GEMM mainloop load
activation scale from the original tensor with descriptor-based TMA.

That requires:

- A separate activation-scale TMA descriptor.
- A separate activation-scale tensormap.
- Descriptor address/dim/stride updates for grouped problems.
- Descriptor fences/acquires.
- A third producer-side TMA copy per K tile, in addition to A and B.

This is a real third TMA stream. It is not piggybacked on the activation matrix
TMA descriptor.

## Practical Rule

Do not assume direct descriptor-based TMA is faster for activation scale just
because it removes the pack kernel.

For the tested M=8 shape, direct raw TMA was only about 2.32% faster end to end
than runtime pack:

```text
Version                 Avg us   Delta vs pack
runtime pack baseline   232.411  baseline
direct raw TMA          227.024  -5.387 us / -2.32%
```

The modest gain matters, but it also shows that the pack kernel was not the
dominant cost.

## Why Pack Can Still Be Fast

For the common fast config:

```text
TileShapeN x scale_k_tile = 16 x 16 ue8m0 values = 256 bytes
```

This is very small for descriptor-based TMA. Direct TMA pays fixed overhead for
a tiny payload:

- Descriptor update work.
- Descriptor fence/acquire work.
- A separate TMA issue in the producer loop.
- Barrier transaction accounting.
- A strided 2D raw-scale access pattern.

The packed path pays a helper kernel, but the GEMM-side load is a simple
contiguous 256-byte bulk copy. For small-M grouped MoE shapes, this can be a
better tradeoff than a tiny descriptor-based TMA load inside every K tile.

## Measurement Checklist

Before replacing runtime pack with direct TMA, collect these numbers:

```text
1. pack kernel only
2. GEMM only with prepacked activation scale
3. GEMM only with direct raw activation-scale TMA
4. runtime pack + GEMM end to end
```

For NCU evidence, every command must include:

```bash
--clock-control none --cache-control all
```

Useful things to inspect:

- TMA issue and wait behavior.
- Producer barrier pressure.
- Whether the extra activation-scale TMA changes overlap with WGMMA.
- Descriptor update cost when all groups have uniform shapes.
- Sensitivity to M, especially M=18 and M=22.

## TODO: Upstream-Padded Activation Scale Rows

The direct raw activation-scale TMA path currently needs each activation-scale
row to expose full 16-byte TMA windows. With `GROUP_SIZE=32` and `ue8m0` scale,
one 16-byte scale window covers 16 scale groups, so it covers `16 * 32 = 512`
K elements. If the raw activation-scale row is only logically sized as
`K / 32`, then a K tile can cross into the next M row unless `K % 512 == 0`.

The preferred future contract is to ask the upstream producer to allocate
activation scale rows with padded physical stride:

```text
logical activation scale shape:  [M, K / 32]
physical stride_m in scale elems: round_up(K / 32, 16)
stride_k in scale elems:          1
padding values:                   arbitrary, never consumed
```

Then GEMM can build the activation-scale TMA descriptor from the physical
padded row stride while the mainloop still consumes only the valid logical scale
groups. This keeps the activation-scale layout K-contiguous, removes the
runtime pack kernel, and relaxes the current `K % 512 == 0` requirement without
adding a complicated tail load path inside the GEMM mainloop.

If upstream cannot provide this padded stride, the fallback direction is a
small runtime padding/packing kernel. That is still preferable to mixing a
special scalar tail path into the producer pipeline unless measurements prove
otherwise.

## Code Guidelines

- Keep weight scale and activation scale semantically separate. They are
  different runtime inputs even if they eventually share a packed shape.

- Use names that expose the difference:

```text
block_weight_scale
block_weight_scale_packed
block_activation_scale
block_activation_scale_packed
```

- Avoid generic names such as `offset_scale` when both weight and activation
  scale exist. The name should say which tensor it belongs to and whether it is
  raw or packed.

- Keep the active CMX scope narrow: `int4 x fp8`, `mxfp4 x bf16`,
  `mxfp4 x fp8`, and `mxfp4 x mxfp8` through the shared mainloop. Do not add
  compatibility shims for inactive historical paths while working on this
  optimization.

## Current Decision

The runtime-pack implementation is the committed reference. Direct raw
activation-scale TMA is a useful experiment and may be worth keeping if broader
measurements confirm the speedup, but it needs more profiling before it should
replace the packed path.
