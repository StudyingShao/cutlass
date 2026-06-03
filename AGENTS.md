# cutlass_mixed_gemm Agent Rules

## Strict Scope

For this directory, only inspect, modify, build, benchmark, or reason about the
following active implementation family unless the user explicitly expands the
scope:

- `int4 x fp8`
- `mxfp4 x bf16`
- `mxfp4 x fp8`
- `mxfp4 x mxfp8`

These active dtype paths share the CMX grouped mixed GEMM harness and mainloop
unless the user explicitly decides to split a path out.

Do not proactively fix, modernize, preserve compatibility for, benchmark, or
discuss other implementations in this repository. In particular, do not patch
historical targets, scalar-scale paths, direct-convert paths, or unrelated
examples unless the user explicitly asks for them.

When reviewing changes, treat compile or API compatibility issues in unrelated
implementation families as out of scope. Do not add fallback traits, default
members, compatibility shims, or historical-target fixes just to keep inactive
paths building.

## Shared Mainloop Requirement

The four active implementations above must all work, and they currently share
the same mainloop. Keep them on the same mainloop unless the user explicitly
decides to split them.

When changing the active path, make sure the change does not break any of the
four active implementations.

## Primary File

The main file of interest is:

```text
examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_fp8_grouped_gemm.cu
```

Treat related CMake, profiler helper, and CUTLASS mainloop utility edits as
supporting changes only when they are required by this file and the four active
implementations.

## Production Shape Discipline

- Do not add or justify CMX fast paths whose primary benefit is only uniform
  expert shape, uniform per-expert M, or uniform scheduler topology.
- Treat uniform expert cases as sanity/perf-smoke tests only. They are not
  representative production cases and must not drive default kernel design.
- Production scheduler and kernel changes must target var-M / non-uniform
  expert token distributions first.
- If a uniform-topology shortcut is ever considered, it must be explicitly
  requested by the user, isolated behind an experimental guard, and proven not
  to add register pressure, branches, parameters, build complexity, or latency
  to the var-M path.

## Benchmark Requirements

- Report scheduler-builder kernels as part of the operator latency. Do not
  present GEMM-only timing as end-to-end scheduler performance.
- Use uniform-M results only as a quick diagnostic. Any scheduler claim needs a
  non-uniform var-M benchmark before it can be treated as useful for CMX.
- Keep comparisons apple-to-apple for the requested dtype path and scale group.
  Do not introduce alternate scale granularities or synthetic fixed-M tuning
  unless the user explicitly asks for that experiment.

## Design Bias

- Prefer changes that make the general var-M path faster or simpler.
- Avoid special-casing benchmark shapes just because they are easy to measure.
- When a measured win only exists for a synthetic uniform case, document it as
  such and do not promote it to the default implementation.

## Commit Message Scope

- Scope CMX commit messages by the actual affected implementation surface, not
  only by the path used for the main benchmark.
- If a change affects the shared grouped mixed GEMM scheduler, profiler,
  mainloop, TMA descriptor flow, or other infrastructure used by multiple dtype
  paths, use an implementation-wide prefix such as `Mixed grouped GEMM:`.
- If a change only affects one dtype path, start the message with that exact
  path name, for example `MXFP4 x MXFP8:`, `MXFP4 x FP8:`,
  `MXFP4 x BF16:`, or `INT4 x FP8:`.
- If a change affects a specific subset of dtype paths, name that subset
  explicitly instead of collapsing it to a single benchmarked path.

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
