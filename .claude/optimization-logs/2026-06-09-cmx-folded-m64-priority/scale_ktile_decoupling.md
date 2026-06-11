# CMX fused E8M0 scale/Ktile decoupling

Date: 2026-06-09

## Goal

The fused E8M0 pre-MMA scale path originally used a weight-scale layout tied to
the profiled `TileK`. That is acceptable for a fixed benchmark config, but it
does not work as a production preprocessing format when the runtime dispatcher
may select different `TileK=128/256/512` configs from a best-config map.

The goal was to keep one offline weight-scale layout for all supported Ktiles
while preserving the main `TileM=64` performance path.

## Design constraints

- Do not generate one preprocessed scale copy per profiler Ktile.
- Keep `TileK=128/256/512` legal.
- Do not reintroduce a profiler guard for slow but correct configs.
- Keep the runtime scale traffic close to the original path.
- Prefer protecting `TileM=64` over `TileM=256` if a tradeoff is required.

The raw E8M0 scale shape creates the core issue: with group size 32, a raw
K-major scale row has only 4B for `TileK=128` and 8B for `TileK=256`. That is
below the 16B copy/TMA-friendly window we need. Any Ktile-independent layout
therefore has to fold another dimension into the physical K direction.

## Folded layout

The retained representation uses a fixed logical base tile:

```text
logical scale tile:  M64 x K128
folded payload:      16 x 16 e8m0 bytes
```

The M64 tile is split by 16 rows, matching the four warp slices used by the
tensor-core operand layout. Those four M16 slices are folded into the physical
K direction, so each folded row exposes a 16B copy row independent of the
selected compute Ktile.

The final physical order is M64-priority:

```text
folded block id = (m64_group * 4 + m64_in_group) * total_k128_blocks + k128
element offset  = folded_block * 16 * 16 + n16 * 16 + warp_slice * 4 + kg4
```

Runtime copy policy:

```text
TileM=64:   one contiguous scale bulk copy
TileM=128:  two contiguous scale bulk copies
TileM=256:  four contiguous scale bulk copies
```

This intentionally protects the most important `TileM=64` configs and accepts
extra copy issue overhead for larger M tiles.

## Implementation summary

- Host preprocessing writes scalar E8M0 bytes into the folded M64-priority
  layout.
- The fused E8M0 reference path reads a separate logical `N x (K / group_size)`
  scale buffer, so correctness validation does not depend on the folded physical
  layout used by the kernel.
- The prescale mainloop removed the weight-scale TMA descriptor path. Weight
  scale is staged with `SM90_BULK_COPY_G2S` into raw folded smem.
- The consumer views the same smem through an expanded layout that matches the
  A-operand scale pairing expected by the FP4 to FP8 converter.
- `TileN >= 128` preloads all scale K blocks into RF because scale smem-to-RF
  latency is exposed in large-N tiles.
- `TileM >= 256` uses the raw expanded scale RF path; smaller M tiles use the
  pair-indexed offset cache.

## Optimization path

Raw K-major scale was rejected because `TileK=128/256` cannot provide a 16B
inner scale window.

The first folded implementation made the preprocessing layout independent of
Ktile, but it produced severe profiler outliers, including regressions above
50% for several low-stage `TileM=256` candidates.

A 2x2 folded ordering improved some K512 cases but introduced K128 outliers, so
it was not kept as the unified layout.

The retained M64-priority ordering removed the extreme outliers and improved
the key E256 K512 result versus the earlier folded layouts while preserving the
Ktile-independent preprocessing contract.

Additional consumer-side experiments showed that some slow configs were caused
by exposed scale smem/RF dependency, not by bulk-copy traffic alone. The final
consumer policy keeps pair-indexed offset prefetch for smaller M tiles and raw
expanded scale RF for `TileM >= 256`.

## Correctness

The final sweeps passed with `Humming_tol_error_count 0` on all measured
candidate configs for:

```text
E256_M24_N512_K4096, Ktile=128/256/512
E32_M128_N4096_K4096, Ktile=128/256/512
```

## Performance summary

Best measured results:

```text
Shape                  original   latest_4x1   2x2      M64-priority
E256_M24_N512_K4096    143.119    148.414      146.837  145.750 us
E32_M128_N4096_K4096   508.288    507.544      506.890  507.362 us
```

Compared with the first Ktile-decoupled folded implementation, the retained
layout removes the >50% profiler outliers. Remaining >5% regressions are
concentrated in non-best `TileM=128/256` or skinny-`TileN` configs.

The old original `143.119 us` E256 result remains faster because that layout was
allowed to bind weight-scale preprocessing directly to `TileK=512`. The final
layout gives up that Ktile-specific preprocessing advantage to satisfy the
production constraint.

Detailed per-config performance is recorded in `m64_priority_comparison.md`.

## Decision

Keep the M64-priority folded scale layout and bulk-copy prescale mainloop as
the production direction for fused E8M0 pre-MMA scaling. Do not keep the raw
K-major TMA attempt, the 2x2 layout, the profiler guard experiment, or the old
weight-scale TMA descriptor path.
