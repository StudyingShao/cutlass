# Optimization Logs

This directory contains detailed records of kernel optimization sessions.

## Purpose

Each optimization log documents:
- Baseline performance measurements
- All optimization attempts (successes and failures)
- Performance results and analysis
- Decision rationale (keep/revert)
- Cumulative progress tracking
- Final session summary

## File Naming Convention

```
YYYY-MM-DD-[topic].md

Examples:
- 2026-03-19-scheduler-optimization.md
- 2026-03-20-mainloop-pipeline.md
- 2026-03-21-epilogue-fusion.md
```

## How to Use

**Creating new log**:
1. Invoke the `optimization-workflow` skill
2. Follow Phase 1 (Initialize Optimization Session)
3. Use the template from `.claude/skills/optimization-workflow/SKILL.md`

**During optimization**:
- Document each attempt BEFORE making changes
- Measure performance AFTER changes (use `benchmark` skill)
- Record decision and rationale
- Update cumulative progress

**After optimization session**:
- Create session summary
- Commit final code with reference to log
- If significant insights, create knowledge document in `.claude/knowledge/optimization/`

## Relationship with Knowledge Base

| Directory | Purpose | Lifecycle |
|-----------|---------|-----------|
| `.claude/optimization-logs/` | Trial-and-error history | Reference during and after session |
| `.claude/knowledge/optimization/` | Proven insights and guidelines | Permanent reference for future work |

**Flow**:
```
Optimization Log (detailed history)
        ↓
   [Extract insights]
        ↓
Knowledge Document (distilled wisdom)
```

## Artifact Policy

- Commit markdown summaries and small CSV summaries that are useful for review.
- Treat raw profiler `.log` dumps as local artifacts unless the user explicitly
  asks to preserve them in git.
- Historical notebooks that do not fit the log lifecycle, such as
  `.claude/malogemm-optimization/`, should carry an explicit status note so old
  "active" wording is not mistaken for current implementation scope.

## Log Status

Active logs (optimization in progress):
- [To be filled when sessions start]

Completed logs:
- [2026-03-19-scheduler-work-switching.md](2026-03-19-scheduler-work-switching.md)
- [2026-05-21-cmx-four-impl-best-configs.md](2026-05-21-cmx-four-impl-best-configs.md)
- [2026-05-22-mxfp4-mxfp8-activation-scale-tma-vs-pack.md](2026-05-22-mxfp4-mxfp8-activation-scale-tma-vs-pack.md)
- [2026-05-29-cmx-precomputed-work-order.md](2026-05-29-cmx-precomputed-work-order.md)
- [2026-06-02-cmx-prebuilt-tma-descriptors.md](2026-06-02-cmx-prebuilt-tma-descriptors.md)

---

**Last Updated**: 2026-06-03
