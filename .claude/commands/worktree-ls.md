---
description: List all active git worktrees with status
argument-hint: ""
---

# List Git Worktrees

## Instructions

### 1. Get Worktree List

Use porcelain format for reliable, machine-parseable output:

```bash
git worktree list --porcelain
```

### 2. Parse and Check Status

Parse the porcelain output into records. Each worktree is a block of lines
separated by a blank line, with fields:
- `worktree <path>` — the absolute path
- `HEAD <sha>` — the full commit hash
- `branch refs/heads/<name>` — the branch (or `detached` if HEAD is detached)

For each worktree path, check for uncommitted changes:

```bash
git -C "$WORKTREE_PATH" status --porcelain | wc -l
```

Quote `$WORKTREE_PATH` in all commands to handle paths with spaces.

### 3. Display Results

Show a table with:
- **Path**
- **Branch**
- **Commit** (short hash — first 7 characters of HEAD)
- **Status**: "clean" or "N uncommitted changes"

If there's only the main worktree, add:
```
No additional worktrees. Use /worktree-new <name> to create one.
```
