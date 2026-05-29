This example extends Example 55 to support Grouped GEMMs in CUTLASS.

## High level overview

This example shows how to perform Grouped GEMMs on Hopper when A and B have different types. In the Grouped GEMM, multiple GEMMs with potentially different problem shapes can be excetued in a batch. The interface is similar to the standard mixed-input GEMM presented in Example 55, with a few noteworthy differences:
- inside the collective builder, replace the layout types with layout pointer types.
- in the arguments, pass the group size, array of the problem sizes, and the array of strides for matrix A and B.
- if scales and zero-points are included, also pass the array of their strides in the arguments.

Note that in Example 55, the argument `--g` is used to determine the block scale size. It is important not to confuse this with the `--groups` argument in this example, which specifies the number of GEMMs.

## Precomputed grouped scheduler limits

The profiler-enabled grouped scheduler can build its work map on device before
launching the main GEMM. In this mode each work-map entry is packed into one
`uint64_t`:

```text
channel tile idx : 20 bits
token tile idx   : 24 bits
expert idx       : 19 bits
invalid sentinel : all bits set
```

For the default `TileShape<128,16,K>` this means:

- experts `E <= 524288`;
- per-expert output channels `N <= 134217728`;
- per-expert tokens `M <= 268435456`;
- `K` is not limited by the packed scheduler entry.

The limits apply after scheduler padding to the cluster/swizzle tile multiple.
The host-side benchmark path checks these constraints before launching the
device work-map builder; the device builder also traps if a packed coordinate
overflows.

## Upcoming features

Currently, the Mixed-input Grouped GEMM only supports row-wise scaling. Please contact us if zero-points or block-wise scaling are needed.
