# Optimization Log Template

**File**: `.claude/optimization-logs/YYYY-MM-DD-[topic].md`

---

```markdown
# Optimization Session: [Topic Title]

**Date**: YYYY-MM-DD
**Target**: [Kernel/module name]
**Goal**: [Specific performance target or problem to solve]
**Status**: IN_PROGRESS / COMPLETED

---

## Baseline

**Git commit**: `xxxxxxxx`

**System**:
- GPU: [Model, e.g., NVIDIA H100 SXM5]
- CUDA: [Version]
- CUTLASS: [Branch/commit]

**Problem Configuration**:
- Problem size: M=__, N=__, K=__
- Groups: __
- Tile shape: __×__×__
- Other: swizzle=__, cluster=__

**Baseline Performance** (5-run avg):
- Duration: XXX.X μs (±Y.Y%)
- SM utilization: XX.X%
- Memory bandwidth: XX.X%
- WGMMA utilization: XX.X%

**Primary Bottleneck**: [e.g., Memory bandwidth bound, scheduler overhead]

---

## Optimization Hypotheses (Prioritized)

1. **[Hypothesis 1]** — Expected: X% — Complexity: Low/Med/High
2. **[Hypothesis 2]** — Expected: Y% — Complexity: Low/Med/High

---

## Optimization Attempts

### Attempt #1: [Short title]

**Date**: YYYY-MM-DD

**Hypothesis**: [Specific mechanism for why this should help]

**Modification**: [Files changed, key changes made]

**Expected Impact**: [e.g., "5-10% speedup by reducing LDC latency"]

**Potential Risks**: [e.g., "May increase register pressure"]

---

**Correctness**: PASS / FAIL

**Result**:
- Duration: XXX.X μs → YYY.Y μs (Δ **±Z.Z%**)
- Variance: ±W.W%
- SM utilization: XX.X% → YY.Y%

**Analysis**:
- [Did result match hypothesis?]
- [What actually happened?]
- [Unexpected effects?]

**Decision**: KEEP / REVERT / REFINE

**Rationale**: [Why this decision]

**Lessons Learned**: [Insights for future attempts]

---

### Attempt #2: [Title]

[Same structure]

---

## Cumulative Progress

| Attempt | Title | Impact | Cumulative | Status |
|---------|-------|--------|------------|--------|
| Baseline | -   | -      | 0.0%       | -      |
| #1      | ...  | +X.X%  | +X.X%      | KEPT   |
| #2      | ...  | -Y.Y%  | +X.X%      | REVERTED |

**Current**: XXX.X μs (Baseline: YYY.Y μs, Improvement: ±ZZ.Z%)

---

## Session Summary

**Duration**: [Start] → [End]
**Total attempts**: N (kept: K, reverted: R)

**Final Result**: [Baseline] μs → [Final] μs (**±XX.X%**)

**Successful Optimizations** (ranked by impact):
1. Attempt #X: [Title] (+Y.Y%)
2. Attempt #Z: [Title] (+W.W%)

**Failed Optimizations** (lessons):
1. Attempt #A: [Title] — [Why it failed]

**Key Insights**:
- [Major insight 1]
- [Major insight 2]

**Recommended Future Work**:
- [Idea 1]

**Knowledge Documents Created**:
- [Path to .claude/knowledge/ doc, if any]

**Session Status**: COMPLETED
**Final Git Commit**: `yyyyyyyy`
```
