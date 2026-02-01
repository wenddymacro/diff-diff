---
description: Remove a git worktree and optionally delete its branch
argument-hint: "<name>"
---

# Remove Git Worktree

Remove an existing worktree. Arguments: $ARGUMENTS

## Instructions

### 1. Parse Arguments

Extract **name** from `$ARGUMENTS`. If empty, abort:
```
Error: Name required. Usage: /worktree-rm <name>
Tip: Run /worktree-ls to see active worktrees.
```

Validate that **name** starts with a letter or digit, followed by `[a-zA-Z0-9._-]`.
If it starts with `-` or contains invalid characters, abort:
```
Error: Name must start with a letter or digit and contain only letters, digits, dots, hyphens, and underscores.
Got: <name>
```

### 2. Resolve Paths

```bash
MAIN_ROOT="$(git worktree list --porcelain | head -1 | sed 's/^worktree //')"
REPO_NAME="$(basename "$MAIN_ROOT")"
PARENT_DIR="$(dirname "$MAIN_ROOT")"
WORKTREE_PATH="${PARENT_DIR}/${REPO_NAME}-<name>"
```

### 3. Validate

```bash
git worktree list
```

If no worktree exists at `$WORKTREE_PATH`, abort:
```
Error: No worktree found at $WORKTREE_PATH
Active worktrees: <list them>
```

### 4. Check for Uncommitted Work

Capture the branch name first (needed for step 6, before the worktree is removed):

```bash
BRANCH=$(git -C "$WORKTREE_PATH" rev-parse --abbrev-ref HEAD)
```

Then check for uncommitted changes:

```bash
git -C "$WORKTREE_PATH" status --porcelain
```

If there are uncommitted changes, warn the user with AskUserQuestion:
- Option 1: "Abort — I have unsaved work"
- Option 2: "Remove anyway — discard changes"

**If the user chooses "Abort", stop immediately. Do NOT continue to step 5.**

### 5. Remove the Worktree

If the worktree had uncommitted changes and the user chose "Remove anyway":
```bash
git worktree remove "$WORKTREE_PATH" --force
```

If the worktree was clean (step 4 found no changes):
```bash
git worktree remove "$WORKTREE_PATH"
```

### 6. Try to Delete the Branch

Only attempt branch deletion if `$BRANCH` equals `<name>` (meaning we created it
via `/worktree-new <name>` without a base-ref). If the branch is something else
(e.g., `feature/existing-branch`), skip deletion — the user didn't create it.

```bash
git branch -d -- "$BRANCH"
```

Let the output print naturally:
- If the branch was merged, it will be deleted and git prints a confirmation.
- If not fully merged, git prints a warning — relay that to the user
  (suggest `git branch -D -- "$BRANCH"` if they want to force-delete).

### 7. Report

```
Removed worktree: $WORKTREE_PATH
[Branch status from step 6]
```
