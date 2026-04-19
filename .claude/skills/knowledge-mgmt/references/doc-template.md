# Knowledge Document Template

## Standard Structure

```markdown
# [Topic Title]

**Last Updated**: YYYY-MM-DD
**Session/PR**: [Link or reference]
**Status**: Stable | Evolving | Experimental

## Background

Why this knowledge matters. What problem does it solve? When do you need it?

## Core Concepts

### [Concept 1]
Deep technical explanation with:
- Fundamental principles
- Constraints and limitations
- How it works under the hood

### [Concept 2]
...

## Practical Guidelines

### When to Use
Clear criteria for when this applies.

### How to Apply
Step-by-step with examples.

### What to Avoid
Common pitfalls and anti-patterns.

## Empirical Data

| Configuration | Metric | Notes |
|---------------|--------|-------|
| ...           | ...    | ...   |

## Lessons Learned

### What Works
Proven approaches with explanation of why they work.

### What Doesn't Work
Failed attempts and why they failed — as valuable as successes.

### Trade-offs
When to choose X over Y.

## Code References

- **File**: `path/to/file.hpp:line_range`
- **Function**: `function_name` — brief description
- **Related**: Links to other knowledge docs

## Next Steps / Open Questions

- [ ] Areas needing further investigation
- [ ] Potential optimizations to test

## Related Knowledge

- `.claude/knowledge/category/related-doc.md`
```

---

## Minimal but Complete Example

```markdown
# SwapAB Optimization Pattern

**Last Updated**: 2026-03-18
**Status**: Stable

## Background

MXFP4×BF16 and INT4×FP8 kernels use SwapAB to match WGMMA constraints.
Essential for understanding why our kernels transpose A/B matrices.

## Core Concepts

### WGMMA Constraint
Hopper WGMMA shape: M×N×K where M=64 (fixed), K=32 (fixed for FP8), N=variable.
WGMMA can only handle variability in the N dimension.

### Problem Characteristics
Our grouped GEMM: C[M,N] = A[M,K] × B[K,N]
Each group has fixed N and K, but variable M (batch dimension).
Mismatch: variable M vs WGMMA's fixed M.

### SwapAB Solution
Transpose: C^T[N,M] = B^T[N,K] × A^T[K,M]
After SwapAB: TileM ↔ problem N (fixed), TileN ↔ problem M (variable).
Result: Problem's variable M now maps to WGMMA's variable N.

## Practical Guidelines

When to use: problem has variable M dimension + FP8/INT4 WGMMA (M=64 fixed).
What to avoid: SwapAB when M is fixed and N is variable (unnecessary overhead).

## Code References

- `sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp`
- `69_hopper_int4_fp8_grouped_gemm.cu:L50-80`
- `cute/atom/mma_traits_sm90.hpp`

## Related Knowledge

- `.claude/knowledge/architecture/wgmma-constraints.md`
- `.claude/knowledge/optimization/tile-tuning-guide.md`
```

---

## Optimization/Debugging Docs: Track Evolution

For `optimization/` and `debugging/` documents, preserve experiment history:

```markdown
## Benchmark History

### Latest Findings (2026-03-25)
| Config      | Duration | Notes              |
|-------------|----------|--------------------|
| 128×16×512  | 157μs    | Best for M<64      |
| 128×32×512  | 143μs    | Better for M≥64    |

### Previous Attempts (2026-03-18)
| Config      | Duration | Why Changed        |
|-------------|----------|--------------------|
| 128×16×256  | 165μs    | K=512 proved 8% faster |

## Evolution Notes
- 2026-03-25: Discovered M threshold — K=512 not always optimal
- 2026-03-18: Initial finding — K=512 > K=256 for small M
```

When code itself evolves, preserve implementation history (not an "error" — correct for its time):

```markdown
## Implementation History

### v2: Bit Manipulation (2026-03-18) — CURRENT
**Approach**: Manual bit manipulation (ported from Triton)
**Performance**: 165μs  
**Why replaced v1**: Eliminated LUT memory lookup overhead (~7% faster)

### v1: Lookup Table (Before 2026-03-18)
**Approach**: Pre-computed LUT with 16 entries  
**Performance**: 178μs  
[Implementation preserved — was correct at the time]
```
