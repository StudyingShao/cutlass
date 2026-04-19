# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## How to Use This Document

1. **READ this file** at session start for core engineering context
2. **Check `.claude/knowledge/`** before answering any technical question — deep-dive docs exist for TMA, WGMMA, SwapAB, optimization, and modules
3. **Check `.claude/skills/`** for operational workflows (benchmarking, knowledge management, optimization)
4. **When you learn something new**: document it immediately — engineering context → `CLAUDE.md`, workflows → `.claude/skills/`, technical insights → `.claude/knowledge/`
5. **When you make a mistake**: acknowledge + edit the relevant doc in the **same response**. Oral promises have zero persistence. Confirm: "已固化到 [file]:[section]"

---

## Behavioral Rules

**Check knowledge base first.** Before answering technical questions, search `.claude/knowledge/` for existing docs. 400+ line deep-dives exist; answering from CLAUDE.md summaries alone is prohibited.

```bash
# Always run this before answering technical questions
find .claude/knowledge -name "*.md"
```

**Truth over agreement.** Verify against code/docs before responding. If the user is wrong, correct them with evidence. If uncertain, say so and check first.

**Defensive operations — require explicit user instruction, even in bypass mode:**

| Operation | Rule |
|-----------|------|
| `git push` (any remote) | Only when user explicitly says "push" |
| `git push --force` / `-f` | Refuse unless user types the flag themselves |
| `git reset --hard` | Refuse; propose `git stash` instead |
| `git restore` (any scope) | Only when user names the specific file(s) |
| `rm`, `rmdir`, file deletion | Hook will intercept; queue to `/tmp/claude_pending_deletes.txt` |
| Modifying `.github/`, CI/CD configs | Confirm before editing |

These rules are enforced at two layers: this instruction (text) + `~/.claude/settings.json` PreToolUse hook. If the hook fires and blocks a command, do not attempt to rewrite the command to bypass it — surface the pending queue to the user instead.

**Temporary files:** Only write to `/tmp/`. Never create temporary files in the project directory or home directory.

---

## Repository Overview

CUTLASS is NVIDIA's CUDA C++ template abstraction library for high-performance matrix operations. This branch (`jiangs/mxfp4xfp8`) implements **Hopper (SM90) mixed-precision grouped GEMM**:

- **MXFP4 × BF16**: `float_e2m1_t` weights × `bfloat16_t` activations, group_size=32
- **INT4 × FP8**: `int4b_t` weights × `float_e4m3_t` activations, group_size=128

Both kernels use Hopper TMA and WGMMA with groupwise scaling. Test harness: `examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_fp8_grouped_gemm.cu` — switch kernels by commenting/uncommenting type definitions at the top.

**Main branch:** `main` — stable, use for PRs.

### Problem Shape Characteristics

**N and K are FIXED** across all groups; **M is VARIABLE** per group; **group count is VARIABLE**.

After SwapAB: kernel sees M'=N (fixed), N'=M (variable), K'=K (fixed).
Implication: `tiles_m = ceil(N/TileM)` is constant; `tiles_n = ceil(M/TileN)` varies per group.

---

## Build and Workflow

**Use `./shao_rebuild.sh` for all compile/test/profile tasks.** Direct `make`, `ncu`, or executable invocation is prohibited.

| Task | Purpose | Output |
|------|---------|--------|
| 0 | Rebuild `69_hopper_int4_fp8_grouped_gemm` | `shao_build_69.log` |
| 2 | Run + NCU profile **(most used)** | `shao_run_69_lut.log` |
| 4 | NSys timeline profile | console |

```bash
echo "0" | bash ./shao_rebuild.sh          # rebuild
echo "2" | bash ./shao_rebuild.sh          # test + profile
grep "Duration" shao_run_69_lut.log        # extract result (μs)
```

CMake one-time setup: `cmake .. -DCUTLASS_NVCC_ARCHS=90a` (the `a` suffix is required for TMA/WGMMA).

**For detailed benchmarking workflow:** `.claude/skills/benchmark/SKILL.md`

---

## Architecture

### Core Library Structure

```
include/cutlass/gemm/collective/sm90_*.hpp    # Hopper mainloop implementations (primary target)
include/cutlass/gemm/kernel/                   # Full kernel implementations
include/cutlass/gemm/dispatch_policy.hpp       # Tag-dispatch policies
include/cutlass/epilogue/collective/           # Epilogue implementations
include/cute/                                  # CuTe: layout and tensor library
  atom/mma_traits_sm90.hpp                    # WGMMA atoms
  atom/copy_traits_sm90_tma.hpp              # TMA copy atoms
```

### Key Concepts

**CuTe:** Hierarchical layout algebra — layouts describe both thread-to-data mappings and memory layouts with compile-time correctness checks. Replaces all CUTLASS 2.x iterators.

**TMA (Tensor Memory Accelerator):** Hopper hardware for multidimensional GMEM→SMEM movement. Two forms: descriptor-based (multidimensional tensors) and descriptor-free (1D bulk copy). See `.claude/knowledge/architecture/tma-fundamentals.md`.

**WGMMA (Warpgroup Matrix Multiply-Accumulate):** Warpgroup-wide Tensor Core instructions. Tile shape constraints by input type:

| Input type | M (fixed) | K (fixed) | N (variable) |
|-----------|-----------|-----------|-------------|
| BF16/FP16 | 64 | 16 | 16/32/64/128/256 |
| FP8/INT8 | 64 | 32 | 16/32/64/128/256 |

**SwapAB:** Maps variable problem-M to WGMMA's variable N. `C[M,N]=A[M,K]×B[K,N]` → `C^T[N,M]=B^T[N,K]×A^T[K,M]`. After swap: TileM↔problem N (fixed), TileN↔problem M (variable). K dimension plays no role in SwapAB design. See `.claude/knowledge/architecture/swapab-pattern.md`.

### Current Development Focus

**Key files:**
- `include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp` — post-scale mainloop
- `include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_prescale.hpp` — pre-scale scaffold

**Kernel configurations:**

| Kernel | Weights | Activations | Scale type | TileK |
|--------|---------|-------------|-----------|-------|
| MXFP4×BF16 | `float_e2m1_t` | `bfloat16_t` | `float_ue8m0_t` | 128 (fixed) |
| INT4×FP8 | `int4b_t` | `float_e4m3_t` | `bfloat16_t` | 128/256/512 |

Tile shapes: M∈{64,128,256}, N∈{16,32,64,128,256}. K=512 performs better for small problem M.
Scheduler impact: Pingpong supports TileM=64; Cooperative requires TileM∈{128,256}.

---

## Code Patterns and Conventions

### Naming Conventions

- `sm90_*` — Hopper (SM90) specific
- `_tma_` — uses TMA
- `_gmma_` / `_wgmma_` — uses WGMMA Tensor Core
- `_warpspecialized_` — different warps have different roles
- `_array_` — grouped/batched (pointer arrays)
- `_mixed_input_` — different precision for A and B

### CuTe Pattern Quick Reference

```cpp
auto layout = make_layout(make_shape(M, N), make_shape(_1{}, LDA));
Tensor t = make_tensor(make_gmem_ptr(ptr), layout);
auto thread_t = tiled_copy.get_slice(threadIdx.x).partition_S(t);
```

### Commit Message Format

```
<Component>: <Brief description>
```

Component prefix: `MXFP4 x BF16:`, `INT4 x FP8:`, or `MXFP4 x BF16 & INT4 x FP8:`

---

## Important Notes

- CUTLASS is header-only; no compilation needed to use as a library
- This branch focuses solely on `69_hopper_int4_fp8_grouped_gemm.cu`
- TMA descriptors must be created on CPU and copied to GPU before kernel launch
- Shared memory layouts should use swizzle patterns (`Sw<B,M,S>`) to avoid bank conflicts
- CuTe static layouts give compile-time shape checking — if it compiles, shapes are consistent
- Architecture must be `90a` (not `90`) — required for TMA/WGMMA at runtime

---

## Knowledge and Skills Index

| Resource | Location | Purpose |
|----------|---------|---------|
| Benchmarking | `.claude/skills/benchmark/SKILL.md` | NCU profiling workflow |
| Knowledge mgmt | `.claude/skills/knowledge-mgmt/SKILL.md` | Document new findings |
| Optimization workflow | `.claude/skills/optimization-workflow/SKILL.md` | Iterative optimization |
| TMA deep-dive | `.claude/knowledge/architecture/tma-fundamentals.md` | TMA usage and constraints |
| SwapAB deep-dive | `.claude/knowledge/architecture/swapab-pattern.md` | SwapAB design rationale |
| Persistent kernels | `.claude/knowledge/architecture/persistent-kernel-work-scheduling.md` | Work scheduling |
| Module overview | `.claude/knowledge/modules/hopper-mixed-precision-gemm-overview.md` | Code architecture |
| Optimization history | `.claude/knowledge/optimization/` | Past results and techniques |
