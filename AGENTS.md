# cutlass_mixed_gemm Agent Rules

## Strict Scope

For this directory, only inspect, modify, build, benchmark, or reason about the
following active implementation family unless the user explicitly expands the
scope:

- `mxfp4 x bf16`
- `int4 x fp8`
- `mxfp4 x mxfp8`

The current implementation work is `mxfp4 x fp8`. This is an intermediate step
toward applying scale to the activation side and turning the activation format
into `mxfp8`.

Do not proactively fix, modernize, preserve compatibility for, benchmark, or
discuss other implementations in this repository. In particular, do not patch
historical targets, scalar-scale paths, direct-convert paths, or unrelated
examples unless the user explicitly asks for them.

When reviewing changes, treat compile or API compatibility issues in unrelated
implementation families as out of scope. Do not add fallback traits, default
members, compatibility shims, or historical-target fixes just to keep inactive
paths building.

## Shared Mainloop Requirement

The three active implementations above must all work, and they currently share
the same mainloop. Keep them on the same mainloop unless the user explicitly
decides to split them.

When changing the active path, make sure the change does not break any of the
three active implementations.

## Primary File

The main file of interest is:

```text
examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_fp8_grouped_gemm.cu
```

Treat related CMake, profiler helper, and CUTLASS mainloop utility edits as
supporting changes only when they are required by this file and the three active
implementations.

## No Direct Deletion

Do not delete, remove, clean, prune, or `rm` any file or directory in this
repository, including untracked files, generated files, temporary files,
documents, build artifacts, stale experiments, or seemingly redundant content.

If a file or directory looks removable, write it into a pending-deletion review
document instead of deleting it. Include the exact path, the proposed deletion
command, and the reason. The user will review and perform the deletion manually.

This rule is absolute unless the user gives an explicit deletion command for the
specific path in the current turn.

Deletion is also guarded technically:

- Bash startup sources `~/.codex/no-delete-guard.bash`.
- PATH wrappers in `~/.local/bin/` block common deletion commands such as
  `rm`, `rmdir`, `unlink`, `git rm`, `git clean`, dangerous `git restore`, and
  `find -delete`.
- A local Codex plugin at `~/plugins/codex-no-delete-guard` registers
  `hooks.json` with a `PreToolUse` hook to reject deletion-like tool payloads.
- This repository has a local `.git/hooks/pre-commit` hook that rejects staged
  deletions.

Do not bypass these guards. Do not use absolute delete binaries such as
`/bin/rm` or `/usr/bin/rm`, do not set `REVIEWED_DELETE=1`, and do not use tools
or scripts to delete files unless the user explicitly orders deletion of the
specific path in the current turn.
