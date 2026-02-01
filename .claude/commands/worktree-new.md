---
description: Create a new git worktree with full dev environment for parallel work
argument-hint: "<name> [base-branch]"
---

# Create Git Worktree

Create an isolated worktree for parallel development. Arguments: $ARGUMENTS

## Instructions

### 1. Parse Arguments

Parse `$ARGUMENTS` to extract:
- **name** (required): First argument — used as both directory suffix and branch name
- **base-ref** (optional): Second argument — existing branch, tag, or ref to branch
  from (creates branch `<name>` starting at that ref)

If no name is provided, abort with:
```
Error: Name required. Usage: /worktree-new <name> [base-branch]
Example: /worktree-new feature-bacon-fix
```

Validate that **name** starts with a letter or digit, followed by `[a-zA-Z0-9._-]`.
If it starts with `-` or contains spaces, slashes, or other shell metacharacters, abort:
```
Error: Name must start with a letter or digit and contain only letters, digits, dots, hyphens, and underscores.
Got: <name>
```

If **base-ref** is provided, apply the same character validation (must match
`^[a-zA-Z0-9][a-zA-Z0-9._/-]*$` — slashes are allowed for refs like `origin/main`).
Then verify the ref exists:

```bash
git rev-parse --verify --quiet "$BASE_REF"
```

If verification fails, abort:
```
Error: Ref not found: <base-ref>
Available branches:
<output of: git branch -a --format='%(refname:short)'>
```

### 2. Resolve Paths

Derive paths dynamically (do NOT hardcode the repo name):

```bash
MAIN_ROOT="$(git worktree list --porcelain | head -1 | sed 's/^worktree //')"
REPO_NAME="$(basename "$MAIN_ROOT")"
PARENT_DIR="$(dirname "$MAIN_ROOT")"
WORKTREE_PATH="${PARENT_DIR}/${REPO_NAME}-<name>"
```

Use `$WORKTREE_PATH` (the absolute path) for all subsequent commands.

### 3. Validate

```bash
git worktree list
```

- If a worktree already exists at `$WORKTREE_PATH`, abort with an error.
- If a branch named `<name>` already exists and no base-ref was given:
  - First check if the branch is already checked out in a worktree
    (parse `git worktree list --porcelain` for a `branch refs/heads/<name>` line).
  - If checked out elsewhere, abort:
    ```
    Error: Branch '<name>' is already checked out in worktree at <path>.
    Use a different name or remove that worktree first.
    ```
  - Otherwise, ask the user whether to check out that existing branch
    or pick a different name. If the user chooses to use it:
    ```bash
    git worktree add -- "$WORKTREE_PATH" "<name>"
    ```
    Then skip step 4 and continue to step 5.

### 4. Create the Worktree

```bash
# If base-ref provided (create new branch <name> starting at base-ref):
git worktree add -b "<name>" -- "$WORKTREE_PATH" "$BASE_REF"

# If no base-ref (create new branch from current HEAD):
git worktree add -b "<name>" -- "$WORKTREE_PATH"
```

### 5. Set Up Python Environment

Note to user: dependency installation may take a moment on a fresh venv.

```bash
python3 -m venv "$WORKTREE_PATH/.venv"
"$WORKTREE_PATH/.venv/bin/pip" install --upgrade pip
"$WORKTREE_PATH/.venv/bin/pip" install -e "$WORKTREE_PATH[dev]"
```

Do NOT use `-q` — let pip output stream so the user sees progress.

### 6. Build Rust Backend (best-effort)

Use `--manifest-path` to avoid changing directories:

```bash
"$WORKTREE_PATH/.venv/bin/maturin" develop --manifest-path "$WORKTREE_PATH/Cargo.toml"
```

If maturin is not installed in the venv or the build fails, note that pure-Python
mode will be used and continue. This is not an error.

### 7. Report

Print:

```
Worktree ready: $WORKTREE_PATH
Branch: <branch>

To start working:
  cd $WORKTREE_PATH && source .venv/bin/activate && claude
```
