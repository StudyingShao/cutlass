# Knowledge Update Workflow

Detailed guidance for updating existing knowledge documents.

---

## Two Update Workflows

### Workflow A: Iterative Refinement
**Use for**: `architecture/`, `modules/` — understanding how things work.  
**Goal**: Converge to accurate, deep, well-organized knowledge.  
**Approach**: Correct errors, deepen understanding, restructure for clarity.

### Workflow B: Evolutionary Tracking
**Use for**: `optimization/`, `debugging/` — recording empirical findings.  
**Goal**: Document the optimization journey with data and learning.  
**Approach**: Add new findings without removing old ones; show progression.

**When code itself evolves** (refactoring, new algorithms): use Workflow B style *within* Workflow A documents — old implementation was correct for its time, preserve it.

---

## When to Update vs Create New

**Update existing document when:**
- Same topic, deeper understanding or more data
- Found error in previous explanation
- Better organization emerges
- Related sub-topic fits naturally

**Create new document when:**
- Completely different topic
- Different category (e.g., architecture vs optimization)
- Truly independent subject

Rule of thumb: if you're thinking "this relates to X.md", update X.md.

---

## Workflow A: Step-by-Step

### 1. Identify Update Type

**Type A — Error Correction**: Old understanding was wrong.  
**Type B — Depth Enhancement**: Correct but shallow; add deeper analysis.  
**Type C — Structural Refactor**: Content correct but poorly organized.  
**Type D — Data Addition**: New benchmark results or test cases.  
**Type E — Code Evolution**: Code changed; old version was right, new version exists.

**Critical distinction**:
- Types A/B/C/D: Old content might be wrong → correct/replace it
- Type E: Old content was right for its time → preserve, add new version side-by-side

### 2. Read Current Document

Always read the current state before editing. Understand:
- What's already documented
- What needs correction or expansion
- What structure could improve

### 3. Make the Update

**Error Correction (Type A)**:
```markdown
### [Corrected]
~~Previously thought: X~~
Corrected: Y
Why the error: [brief explanation]
```

**Depth Enhancement (Type B)**:
```markdown
[Existing shallow explanation remains unchanged]

### Deep Dive: Why This Works
[New deeper analysis section appended]

### Performance Implications
[New section with supporting data]
```

**Code Evolution (Type E)** — preserve history, don't erase:
```markdown
### v2: New Implementation (2026-05-20) — CURRENT
[New implementation]
Why this version: [reason for change]

### v1: Previous Implementation (2026-03-18 – 2026-05-20)
[Old implementation preserved]
Why superseded: [what changed and why]
```

### 4. Update Metadata

Always update the document header:
```markdown
**Last Updated**: YYYY-MM-DD
**Status**: Stable | Evolving | Experimental
```

Optionally add an Update History section at the bottom:
```markdown
## Update History

### YYYY-MM-DD: [Brief description]
- What changed and why
- What was corrected or added
```

---

## Document Lifecycle

```
Initial draft (rough but complete)
    ↓ [correct errors discovered in use]
v2 (accurate but shallow)
    ↓ [add deep analysis and data]
v3 (deep but poorly organized)
    ↓ [restructure for clarity]
v4 (clear, accurate, deep)  ← target state
    ↓ [continue as code/understanding evolves]
```

---

## Continuous Refinement in Practice

**During every work session:**
1. When you explain something, ask: "Is this already documented?"
2. If yes: "Is our doc accurate and complete for today's context?"
3. If no: "Should we update the doc with this new understanding?"

**Red flags that trigger updates:**
- "Wait, that's not quite right..."
- "We understand this better now..."
- "This new data changes things..."
- "This doc is hard to follow..."

**Green signals (doc is healthy):**
- Document correctly predicted behavior
- Document's guidance led to right solution
- Document had the data we needed
