---
name: knowledge-mgmt
description: Deep knowledge curation and documentation for the CUTLASS/GEMM kernel codebase. Use this skill when documenting a new technical finding, recording benchmark data, creating or updating a knowledge document, searching for existing knowledge before starting analysis, or when the user says "document this", "save what we learned", "create a knowledge doc", or "do we have notes on X?". Also use proactively after any deep technical analysis, optimization session, or debugging success to offer documentation.
---

# Knowledge Management Skill

Transform deep engineering insights into structured, reusable knowledge that persists across sessions.

**Core principle**: Write it or lose it. Oral acknowledgment has zero persistence.

---

## Project-First Mindset

All knowledge serves THIS codebase as the primary purpose.

- "We need to understand THIS kernel's TMA usage to maintain/optimize it" — correct
- "I learned about TMA, let me create general educational material" — wrong

Ask before documenting: "Does this help me work on THIS codebase?" If not, it doesn't belong here.

---

## Knowledge Repository Structure

```
.claude/knowledge/
├── architecture/     # Deep understanding of hardware/concepts used in this code
│   ├── tma-deep-dive.md
│   ├── wgmma-constraints.md
│   ├── swapab-optimization.md
│   └── cute-layout-system.md
├── optimization/     # Optimization techniques with benchmark evidence
│   ├── tile-tuning-guide.md
│   ├── scaling-overlap.md
│   └── scheduler-selection.md
├── modules/          # Per-file/module understanding of this codebase
│   ├── mainloop-collective.md
│   └── tma-descriptor-creation.md
└── debugging/        # Error patterns and debugging methodologies
    ├── sass-analysis-methodology.md
    ├── ncu-profiling-insights.md
    └── common-errors-solutions.md
```

**Category routing**:
- How hardware/architecture works → `architecture/`
- Optimization technique with benchmark data → `optimization/`
- How a specific code module works → `modules/`
- Debugging method or error pattern → `debugging/`

---

## When to Offer Documentation

Proactively offer after:
1. Deep analysis of a module or architecture component
2. Optimization with measured performance data
3. Debugging a complex issue — root cause found
4. 3+ related insights accumulated on one topic
5. User says "now I understand..." or "that makes sense..."
6. The analysis took significant effort — future instances shouldn't repeat it

**How to offer**:
> "We've built deep understanding of [topic]. Should I create
> `.claude/knowledge/[category]/[topic].md` capturing:
> - Core principles and constraints
> - Performance data and measurements
> - Lessons learned and what to avoid
> - Relevant code locations"

---

## Retrieval: Check Before Analyzing

**Before answering any technical question about this codebase, search existing knowledge:**

```bash
find .claude/knowledge -type f -name "*.md"
```

If relevant docs exist: read them, build your answer on the existing foundation, cite the source.  
If missing: answer, then offer to document new insights.

---

## Creation Workflow

### Step 1: Check criteria
- Took significant effort to understand
- Will be needed again
- Contains non-obvious insights
- Backed by data or deep analysis

### Step 2: Create document

```bash
mkdir -p .claude/knowledge/<category>
# Create .claude/knowledge/<category>/<topic>.md
```

Follow the template in `references/doc-template.md`. Key rules:
- Include clear background and motivation
- Deep content — not superficial (3 bullets anyone could guess is not knowledge)
- Practical guidelines with code examples
- Benchmark data when relevant
- Code references: `file.hpp:line_range`

### Step 3: One topic, one document — continuously refined

Never create `topic-v2.md`, `topic-discussion-2.md`. Update the existing file.  
When understanding deepens: update. When error found: correct. When better structure: refactor.

---

## Update Principles

Two workflows depending on document type:

**Workflow A (architecture/, modules/)**: Iterative refinement — correct errors, deepen understanding.  
**Workflow B (optimization/, debugging/)**: Evolutionary tracking — preserve experiment history, show progression.

When code itself changes (refactoring, new implementation): use Workflow B style inside a Workflow A doc. The old implementation was correct for its time — preserve it, add new version.

For detailed update step-by-step, read `references/update-workflow.md`.

---

## Quality Standards

A good knowledge document:
- Specific — exact numbers, file paths, line references
- Actionable — tells a future agent what to do, not just what exists
- Evidence-based — includes benchmark data, not just assertions
- Scoped — one topic well, not everything vaguely
- Dated — includes when written (`**Last Updated**: YYYY-MM-DD`)

A bad knowledge document:
- Vague principles without concrete examples
- No code references
- "I think" or "probably" without verification

---

## Skills vs Knowledge

| Skills (`.claude/skills/`) | Knowledge (`.claude/knowledge/`) |
|----------------------------|----------------------------------|
| HOW to execute workflows | WHAT we learned from doing them |
| Procedures and processes | Insights and understanding |
| Example: benchmark procedure | Example: what benchmark data reveals |

Both evolve continuously. Both are your responsibility.

---

## Reference Files

- `references/doc-template.md` — Full document template with examples
- `references/update-workflow.md` — Step-by-step guide for updating existing docs
