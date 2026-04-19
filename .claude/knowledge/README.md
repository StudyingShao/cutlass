# Knowledge Repository

This directory contains **deep, structured knowledge** accumulated through engineering work on the CUTLASS Hopper mixed-precision kernels.

## Purpose

Unlike CLAUDE.md (quick reference) or skills (workflows), this repository captures:
- 🧠 Deep architectural understanding
- 📊 Empirical optimization data
- 🔧 Proven techniques and patterns
- 🐛 Debugging methodologies and insights

## Structure

```
knowledge/
├── architecture/       # How hardware and software architecture works
├── optimization/       # Optimization techniques backed by data
├── modules/           # Deep code module understanding
└── debugging/         # Debugging methodologies and error patterns
```

## How This Works

### During Work Sessions

When you and Claude:
1. Deeply analyze an architecture feature
2. Optimize performance with benchmark data
3. Debug complex issues and find patterns
4. Explore and understand code modules

→ Claude will **offer to document** this knowledge

### For Future Work

When you (or Claude) need this knowledge again:
1. Claude searches this directory first
2. Applies documented principles
3. References benchmark data
4. Builds on existing understanding

**This accelerates future work by avoiding re-analysis.**

## Document Standards

Each knowledge document includes:
- **Background**: Why it matters
- **Core Concepts**: Deep technical explanation
- **Practical Guidelines**: How to apply
- **Empirical Data**: Benchmarks, tests, analysis
- **Lessons Learned**: What works, what doesn't
- **Code References**: Where to find related code
- **Last Updated**: Maintenance tracking

## Categories

### `architecture/`
**Deep understanding of how things work**

Examples:
- `tma-deep-dive.md` - TMA principles, constraints, best practices
- `wgmma-constraints.md` - WGMMA tile shapes, data flow
- `swapab-optimization.md` - Why/when/how to use SwapAB
- `cute-layout-system.md` - CuTe layout algebra

### `optimization/`
**Techniques with empirical evidence**

Examples:
- `tile-tuning-guide.md` - Tile shape selection with benchmark data
- `scaling-overlap.md` - Groupwise scaling optimization patterns
- `memory-layout-patterns.md` - Shared memory strategies
- `scheduler-selection.md` - Cooperative vs Pingpong trade-offs

### `modules/`
**Code module deep-dives**

Examples:
- `mainloop-collective.md` - Mainloop flow, customization points
- `epilogue-collective.md` - Epilogue scaling, accumulation
- `tma-descriptor-creation.md` - TMA descriptor setup

### `debugging/`
**Methods and patterns**

Examples:
- `sass-analysis-methodology.md` - How to analyze SASS
- `ncu-profiling-insights.md` - NCU metrics interpretation
- `common-errors-solutions.md` - Error patterns and fixes
- `correctness-validation.md` - How to validate kernels

## Getting Started

No documents yet? That's okay! Knowledge accumulates as you work:

1. **Do deep work** with Claude (analyze, optimize, debug)
2. **Claude offers** to document when insights are gained
3. **Review and approve** the documentation
4. **Knowledge persists** for future sessions

## Maintenance

- Add **new findings** to existing docs (don't remove old insights)
- Mark **obsolete sections** clearly if understanding evolves
- Update **"Last Updated"** date when modifying
- Cross-reference **related documents** as knowledge base grows

## Integration

- **CLAUDE.md** may reference knowledge docs: `@.claude/knowledge/architecture/swapab-optimization.md`
- **Skills** complement knowledge: skills = how to do, knowledge = why/when
- **Future Claude instances** automatically search here before deep analysis

---

**Remember**: Quality over quantity. One deep, data-backed document is worth more than ten shallow ones.

Start with what matters most to your current work, and build from there.
