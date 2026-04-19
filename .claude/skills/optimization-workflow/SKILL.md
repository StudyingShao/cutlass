---
name: optimization-workflow
description: Structured workflow for iterative kernel performance optimization with tracking, measurement, and decision-making. Use when the user starts a focused optimization effort, wants to systematically explore multiple optimization strategies, needs to track attempts with benchmark data, or wants a record of what worked and what didn't. Also use when the user says "let's optimize X", "I want to improve performance", or "track my optimization attempts".
---

# Optimization Workflow Skill

Systematically manage kernel performance optimization with structured tracking, measurement, and decision-making.

**Key principle**: Measure, don't guess. Every optimization hypothesis must be validated with benchmark data.

**When not to use**: One-off quick experiments (just use benchmark skill directly), or non-performance changes (correctness fixes, refactoring).

---

## The Five-Phase Loop

```
Phase 1: Initialize → establish baseline
Phase 2: Attempt   → implement one optimization
Phase 3: Measure   → benchmark with benchmark skill
Phase 4: Decide    → KEEP / REVERT / REFINE
Phase 5: Iterate   → repeat until goal reached or done
```

---

### Phase 1: Initialize Optimization Session

Create a log file and record the baseline:

```bash
# File: .claude/optimization-logs/YYYY-MM-DD-[topic].md
mkdir -p .claude/optimization-logs
```

Record:
- Git commit hash (`git rev-parse --short HEAD`)
- Kernel/module being optimized
- Optimization goal (target speedup or problem to solve)
- Baseline Duration (use benchmark skill, 5 runs, report avg ± variation)
- Key NSight metrics: SM utilization, memory bandwidth, WGMMA utilization
- Primary bottleneck from NCU analysis

Read `references/log-template.md` for the full log file structure.

---

### Phase 2: Execute One Optimization Attempt

Assign sequential ID (#1, #2, ...). Document BEFORE making changes:

```markdown
### Attempt #N: [Short title]
**Hypothesis**: Why you believe this improves performance (be specific about mechanism)
**Modification**: What code changes you will make (file names, key changes)
**Expected Impact**: Quantitative prediction (e.g., "5-10% speedup")
**Potential Risks**: Register pressure, correctness concerns, etc.
```

Then implement the change. Keep changes focused — one optimization per attempt so effects are isolable.

---

### Phase 3: Measure and Record Results

1. Verify correctness first: `./build/.../69_hopper_int4_fp8_grouped_gemm --compare=true`
   - If correctness fails, skip benchmarking — note failure and go to Phase 4 (REVERT)
2. Use benchmark skill to measure Duration (5 runs)
3. Record:

```markdown
**Result**: [baseline] μs → [new] μs (Δ ±X.X%)
**Variance**: ±Y.Y% (stable / unstable)
**Analysis**: Did it match hypothesis? What actually happened? Any surprises?
```

---

### Phase 4: Decision

| Decision | Criteria | Action |
|----------|----------|--------|
| **KEEP** | Performance improved, correctness preserved | Commit, move to next |
| **REVERT** | Degraded or correctness broken | `git checkout -- [files]`, document lesson |
| **REFINE** | Partial improvement, needs tuning | Keep code, create new attempt |

Always document the rationale and lesson learned, even for reverted attempts — failures teach as much as successes.

After deciding, update the cumulative progress table:

```markdown
| Attempt | Title | Impact | Cumulative | Status |
|---------|-------|--------|------------|--------|
| Baseline | -   | -      | 0.0%       | -      |
| #1      | ...  | +X.X%  | +X.X%      | KEPT   |
| #2      | ...  | -Y.Y%  | +X.X%      | REVERTED |
```

---

### Phase 5: Iterate or Finalize

**Stop when**:
- Reached target performance goal
- Diminishing returns (<2% gains from remaining ideas)
- High-impact ideas exhausted

**Finalization**:
1. Write session summary (see log template)
2. Commit final state:
   ```bash
   git commit -m "perf(kernel): [topic] — X.X% improvement

   - Attempt #A: [key change] (+Y.Y%)
   - Attempt #B: [key change] (+Z.Z%)

   See .claude/optimization-logs/YYYY-MM-DD-[topic].md

   Co-Authored-By: Claude <noreply@anthropic.com>"
   ```
3. If session yielded significant insights (>10% gain or novel technique), invoke knowledge-mgmt skill to create a document in `.claude/knowledge/optimization/`

---

## Integration with Other Skills

- **benchmark skill**: Use in Phase 1 (baseline) and Phase 3 (each attempt result)
- **knowledge-mgmt skill**: Use in Phase 5 finalization for transferable insights

---

## Common Failure Modes

**"Results don't match hypothesis"** — incorrect bottleneck assumption. Deep-dive with NSight Compute, update mental model before next attempt.

**"Optimization works on small problems but not large"** — different bottlenecks at different scales. Document the scaling behavior; it's useful knowledge.

**"Multiple optimizations conflict"** — competing for shared resources (registers, SMEM). Test combinations explicitly; may need to choose one approach.

---

## Reference Files

- `references/log-template.md` — Full optimization log file template with all sections
