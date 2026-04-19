# Skill: Claude Files Commit

## Purpose

Commit Claude-related project files (`.claude/`, `CLAUDE.md`) to git for cloud backup and progress tracking.

## Trigger

Use this skill when the user asks to:
- Commit Claude files, knowledge docs, skills, or optimization logs
- "Save Claude progress", "commit .claude", "push knowledge base"
- Any commit that touches only `.claude/` and/or `CLAUDE.md`

## Commit Format

```
Claude: <concise description of what changed>
```

**Rules:**
- Prefix is always `Claude:` (capital C, no brackets)
- Description: present tense, imperative, ≤72 chars total
- No `Co-Authored-By` line (never add Claude as author)
- No body paragraph needed unless changes are complex

**Examples:**
```
Claude: Add TMA fundamentals knowledge doc
Claude: Update benchmark skill with NCU flags
Claude: Add scheduler optimization log for 2026-03-19
Claude: Update CLAUDE.md with SwapAB architecture notes
```

## Files to Include

Always include in a single commit:
- `CLAUDE.md` — if modified
- `.claude/knowledge/**` — knowledge docs
- `.claude/skills/**` — skill definitions
- `.claude/optimization-logs/**` — optimization records
- `.claude/settings.local.json` — project permissions (no sensitive data)

## Workflow

1. Check which Claude files are modified/untracked:
   ```bash
   sudo git -C /TRT/CUTLASS/cutlass_github -c safe.directory=/TRT/CUTLASS/cutlass_github status --short | grep -E '(\.claude|CLAUDE\.md)'
   ```

2. Stage all Claude-related files:
   ```bash
   sudo git -C /TRT/CUTLASS/cutlass_github -c safe.directory=/TRT/CUTLASS/cutlass_github add CLAUDE.md .claude/
   ```

3. Commit with the standard format:
   ```bash
   sudo git -C /TRT/CUTLASS/cutlass_github -c safe.directory=/TRT/CUTLASS/cutlass_github commit -m "Claude: <description>"
   ```

4. Verify:
   ```bash
   sudo git -C /TRT/CUTLASS/cutlass_github -c safe.directory=/TRT/CUTLASS/cutlass_github log -1 --format='%s'
   ```

## Notes

- Always use `sudo git` + `-c safe.directory=...` in this repo (ownership mismatch)
- Never mix Claude files with code changes in the same commit
- If both `.claude/` and code changed, make two separate commits
